//! What Gemma-4 writes, split as llama-server splits it: the thinking channel
//! (`<|channel>thought\n … <channel|>`) into `reasoning_content`, a tool call
//! (`<|tool_call>call:NAME{key:value,…}<tool_call|>`, strings between `<|"|>`)
//! into an OpenAI `tool_calls` entry with JSON arguments, the rest into
//! `content`. The markers are single special tokens: the split is made on the
//! ids, the text of each part decoded as it streams.

use crate::chat::detok::DetokenizationFlow;
use serde_json::{Map, Number, Value};
use tokenizers::Tokenizer;

pub const CHANNEL_OPEN: u32 = 100;
pub const CHANNEL_CLOSE: u32 = 101;
pub const TOOL_CALL_OPEN: u32 = 48;
pub const TOOL_CALL_CLOSE: u32 = 49;
/// After a tool call the model opens the response the harness must write: the end of its turn.
pub const TOOL_RESPONSE_OPEN: u32 = 50;

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Content(String),
    Reasoning(String),
    ToolCall { name: String, arguments: String },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Content,
    /// After `<|channel>`: the channel's name, up to its newline.
    Header,
    Reasoning,
    ToolCall,
}

pub struct OutputParser {
    state: State,
    content: DetokenizationFlow,
    reasoning: DetokenizationFlow,
    header: Vec<u32>,
    call: Vec<u32>,
}

impl Default for OutputParser {
    fn default() -> Self {
        OutputParser { state: State::Content, content: DetokenizationFlow::fresh(), reasoning: DetokenizationFlow::fresh(), header: Vec::new(), call: Vec::new() }
    }
}

impl OutputParser {
    /// A parser for the answer that follows `prompt`: in the thinking channel when the
    /// prompt ends inside one. Gemma's canonical template (2026-07-09) opens
    /// `<|channel>thought\n` itself after a tool response when thinking is on, so the
    /// model's first tokens are thinking, with no `<|channel>` of their own.
    pub fn after_prompt(tok: &Tokenizer, prompt: &[u32]) -> Self {
        let mut parser = OutputParser::default();
        let tail = &prompt[prompt.len().saturating_sub(16)..];
        if let Some(open) = tail.iter().rposition(|&id| id == CHANNEL_OPEN) {
            let header = &tail[open + 1..];
            if !header.contains(&CHANNEL_CLOSE) {
                parser.header = header.to_vec();
                parser.state = if tok.decode(header, false).unwrap_or_default().contains('\n') { State::Reasoning } else { State::Header };
            }
        }
        parser
    }

    fn flush(&mut self, tok: &Tokenizer, out: &mut Vec<Piece>) {
        let c = self.content.balance(tok);
        if !c.is_empty() {
            out.push(Piece::Content(c));
        }
        let r = self.reasoning.balance(tok);
        if !r.is_empty() {
            out.push(Piece::Reasoning(r));
        }
    }

    /// One generated token; what it adds to the answer.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> Vec<Piece> {
        let mut out = Vec::new();
        match (self.state, id) {
            (State::Content | State::Reasoning, CHANNEL_OPEN) => {
                self.flush(tok, &mut out);
                self.header.clear();
                self.state = State::Header;
            }
            (State::Header, _) => {
                self.header.push(id);
                if tok.decode(&self.header, false).unwrap_or_default().contains('\n') {
                    self.state = State::Reasoning;
                }
            }
            (State::Reasoning, CHANNEL_CLOSE) => {
                self.flush(tok, &mut out);
                self.state = State::Content;
            }
            (State::Content | State::Reasoning, TOOL_CALL_OPEN) => {
                self.flush(tok, &mut out);
                self.call.clear();
                self.state = State::ToolCall;
            }
            (State::ToolCall, TOOL_CALL_CLOSE) => {
                let text = tok.decode(&self.call, false).unwrap_or_default();
                out.push(match parse_call(&text) {
                    Some((name, args)) => Piece::ToolCall { name, arguments: args.to_string() },
                    // Not a call we can read: handed back as text, as written.
                    None => Piece::Content(format!("<|tool_call>{text}<tool_call|>")),
                });
                self.state = State::Content;
            }
            (State::ToolCall, _) => self.call.push(id),
            (State::Content, _) => {
                let s = self.content.push(tok, id);
                if !s.is_empty() {
                    out.push(Piece::Content(s));
                }
            }
            (State::Reasoning, _) => {
                let s = self.reasoning.push(tok, id);
                if !s.is_empty() {
                    out.push(Piece::Reasoning(s));
                }
            }
        }
        out
    }

    /// The end of the answer: what is still held, an unfinished call as text.
    pub fn finish(&mut self, tok: &Tokenizer) -> Vec<Piece> {
        let mut out = Vec::new();
        self.flush(tok, &mut out);
        if self.state == State::ToolCall && !self.call.is_empty() {
            out.push(Piece::Content(format!("<|tool_call>{}", tok.decode(&self.call, false).unwrap_or_default())));
        }
        out
    }
}

const QUOTE: &str = "<|\"|>";

/// `call:NAME{key:value,…}` → the name and the arguments as a JSON object.
pub fn parse_call(text: &str) -> Option<(String, Value)> {
    let rest = text.trim().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim().to_string();
    let mut p = Parser { s: &rest[brace..], at: 0 };
    let args = p.value()?;
    p.ws();
    (args.is_object() && p.at == p.s.len()).then_some((name, args))
}

struct Parser<'a> {
    s: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn rest(&self) -> &str {
        &self.s[self.at..]
    }

    fn ws(&mut self) {
        while self.rest().starts_with(|c: char| c.is_whitespace()) {
            self.at += self.rest().chars().next().map_or(1, char::len_utf8);
        }
    }

    fn eat(&mut self, t: &str) -> bool {
        self.ws();
        if self.rest().starts_with(t) {
            self.at += t.len();
            true
        } else {
            false
        }
    }

    fn string(&mut self) -> Option<String> {
        if !self.eat(QUOTE) {
            return None;
        }
        let end = self.rest().find(QUOTE)?;
        let s = self.rest()[..end].to_string();
        self.at += end + QUOTE.len();
        Some(s)
    }

    /// A key: between `<|"|>`, or bare up to the `:`.
    fn key(&mut self) -> Option<String> {
        self.ws();
        if self.rest().starts_with(QUOTE) {
            return self.string();
        }
        let end = self.rest().find(':')?;
        let k = self.rest()[..end].trim().to_string();
        self.at += end;
        Some(k)
    }

    fn value(&mut self) -> Option<Value> {
        self.ws();
        if self.rest().starts_with(QUOTE) {
            return self.string().map(Value::String);
        }
        if self.eat("{") {
            let mut m = Map::new();
            if self.eat("}") {
                return Some(Value::Object(m));
            }
            loop {
                let k = self.key()?;
                if !self.eat(":") {
                    return None;
                }
                let v = self.value()?;
                m.insert(k, v);
                if self.eat(",") {
                    continue;
                }
                return self.eat("}").then_some(Value::Object(m));
            }
        }
        if self.eat("[") {
            let mut a = Vec::new();
            if self.eat("]") {
                return Some(Value::Array(a));
            }
            loop {
                a.push(self.value()?);
                if self.eat(",") {
                    continue;
                }
                return self.eat("]").then_some(Value::Array(a));
            }
        }
        // A bare word: a number, true, false, null, else a string as written.
        let r = self.rest();
        let end = r.find([',', '}', ']']).unwrap_or(r.len());
        let word = r[..end].trim().to_string();
        self.at += end;
        Some(match word.as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" | "None" => Value::Null,
            _ => {
                if let Ok(i) = word.parse::<i64>() {
                    Value::Number(i.into())
                } else if let Some(n) = word.parse::<f64>().ok().and_then(Number::from_f64) {
                    Value::Number(n)
                } else {
                    Value::String(word)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn calls_in_gemma_syntax() {
        assert_eq!(parse_call("call:get_datetime{}"), Some(("get_datetime".into(), json!({}))));
        assert_eq!(
            parse_call("call:read_file{limit:200,path:<|\"|>src/main.rs<|\"|>}"),
            Some(("read_file".into(), json!({"path": "src/main.rs", "limit": 200})))
        );
        assert_eq!(
            parse_call("call:edit{edits:[{new:<|\"|>a, b}<|\"|>,old:<|\"|>x<|\"|>}],dry:false,ratio:0.5}"),
            Some(("edit".into(), json!({"edits": [{"new": "a, b}", "old": "x"}], "dry": false, "ratio": 0.5})))
        );
        assert_eq!(parse_call("call:f{x:<|\"|>unterminated}"), None);
    }

    /// Needs the E2B tokenizer (`$HOME/models/gemma-4-E2B-it-qat-q4_0-unquantized`); skipped without it.
    #[test]
    fn a_prompt_that_opens_the_thinking_channel_starts_the_answer_in_it() {
        let path = std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join("models/gemma-4-E2B-it-qat-q4_0-unquantized/tokenizer.json");
        let Ok(tok) = Tokenizer::from_file(&path) else { return };
        let ids = |text: &str| tok.encode(text, false).unwrap().get_ids().to_vec();
        let opened = ids("<|tool_response>response:f{value:<|\"|>x<|\"|>}<tool_response|><|channel>thought\n");
        let mut parser = OutputParser::after_prompt(&tok, &opened);
        let mut pieces = Vec::new();
        for id in ids("Je réfléchis.<channel|>Paris.") {
            pieces.extend(parser.push(&tok, id));
        }
        pieces.extend(parser.finish(&tok));
        let reasoning: String = pieces.iter().filter_map(|p| if let Piece::Reasoning(s) = p { Some(s.as_str()) } else { None }).collect();
        let content: String = pieces.iter().filter_map(|p| if let Piece::Content(s) = p { Some(s.as_str()) } else { None }).collect();
        assert_eq!((reasoning.as_str(), content.as_str()), ("Je réfléchis.", "Paris."));
        let closed = ids("<|channel>thought\nvu<channel|>");
        assert_eq!(OutputParser::after_prompt(&tok, &closed).state, State::Content);
        assert_eq!(OutputParser::after_prompt(&tok, &ids("<|turn>model\n")).state, State::Content);
    }
}
