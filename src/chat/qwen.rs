//! A Qwen3.5 conversation: the chat template rendered turn by turn (the
//! strings of herbert-jit-bf16's `chat/template.rs` byte for byte, without the
//! tools), the tokenizer and the special tokens the session looks at, the same
//! stop rule and the same counting of generated tokens as herbert-vk-bf16's
//! session, so that the `[ids]` of every engine compare one to one. The
//! special tokens are recognised by id, never by decoding text, so a user who
//! types "<think>" cannot open a block.

use crate::chat::detok::DetokenizationFlow;
use crate::chat::{show_top, top_from_env, TurnStats};
use crate::core::error::{HerbertError, Result};
use crate::core::sampler::{Sampler, Sampling};
use crate::metal::qwen::QwenEngine;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

/// `<|im_start|>system\n…<|im_end|>\n`, nothing without a system text.
pub fn system_message(system: Option<&str>) -> String {
    match system.map(str::trim).filter(|s| !s.is_empty()) {
        None => String::new(),
        Some(text) => format!("<|im_start|>system\n{text}<|im_end|>\n"),
    }
}

pub fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n", text.trim())
}

/// The assistant header: thinking open, or an empty thinking block.
pub fn assistant_header(think: bool) -> String {
    if think {
        "<|im_start|>assistant\n<think>\n".to_string()
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string()
    }
}

pub struct Tokens {
    pub tokenizer: Tokenizer,
    pub im_start: u32,
    pub im_end: u32,
    pub endoftext: u32,
    pub think_open: Option<u32>,
    pub think_close: Option<u32>,
    pub tool_call_open: Option<u32>,
    pub tool_call_close: Option<u32>,
}

impl Tokens {
    pub fn load(model_dir: &Path) -> Result<Tokens> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| HerbertError::ModelLoad(format!("{}: {e}", path.display())))?;
        let required = |name: &str| -> Result<u32> {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| HerbertError::ModelLoad(format!("the tokenizer has no `{name}` token")))
        };
        Ok(Tokens {
            im_start: required("<|im_start|>")?,
            im_end: required("<|im_end|>")?,
            endoftext: required("<|endoftext|>")?,
            think_open: tokenizer.token_to_id("<think>"),
            think_close: tokenizer.token_to_id("</think>"),
            tool_call_open: tokenizer.token_to_id("<tool_call>"),
            tool_call_close: tokenizer.token_to_id("</tool_call>"),
            tokenizer,
        })
    }

    /// Text to ids. Special tokens written in the text (`<|im_start|>`, …) are
    /// recognised: this is how the template's markers become ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| HerbertError::Backend(format!("tokenizer: {e}")))?
            .get_ids()
            .to_vec())
    }

    /// Ids to text, special tokens left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    pub fn is_stop(&self, id: u32) -> bool {
        id == self.im_end || id == self.endoftext
    }

    /// One line for the log: the ids the session will watch.
    pub fn summary(&self) -> String {
        format!(
            "im_start {} im_end {} endoftext {} think {:?}/{:?} tool_call {:?}/{:?} · vocab {}",
            self.im_start,
            self.im_end,
            self.endoftext,
            self.think_open,
            self.think_close,
            self.tool_call_open,
            self.tool_call_close,
            self.tokenizer.get_vocab_size(true)
        )
    }
}

pub struct Chat {
    engine: QwenEngine,
    tokens: Tokens,
    sampler: Sampler,
    system: Option<String>,
    pub think: bool,
    pub max_tokens: usize,
    /// Go on past the end of turn, up to `max_tokens`.
    pub ignore_eos: bool,
    ledger: Vec<u32>,
    assistant_open: bool,
}

impl Chat {
    pub fn new(engine: QwenEngine, tokens: Tokens, sampling: Sampling, system: Option<String>) -> Chat {
        Chat { engine, tokens, sampler: Sampler::new(sampling), system, think: true, max_tokens: 1024, ignore_eos: false, ledger: Vec::new(), assistant_open: false }
    }

    pub fn engine(&self) -> &QwenEngine {
        &self.engine
    }

    /// Back to an empty conversation (the sampler keeps its state).
    pub fn reset(&mut self) -> Result<()> {
        self.engine.reset()?;
        self.ledger.clear();
        self.assistant_open = false;
        Ok(())
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    pub fn ledger(&self) -> &[u32] {
        &self.ledger
    }

    /// One turn: the user's text in, the answer streamed to `on_text`.
    pub fn chat(&mut self, user: &str, on_text: &mut dyn FnMut(&str)) -> Result<TurnStats> {
        let mut stats = TurnStats::default();
        let mut text = String::new();
        if self.ledger.is_empty() {
            text.push_str(&system_message(self.system.as_deref()));
        } else if self.assistant_open {
            text.push_str("<|im_end|>\n");
        }
        text.push_str(&user_turn(user));
        text.push_str(&assistant_header(self.think));
        let ids = self.tokens.encode(&text)?;
        if std::env::var_os("HERBERT_METAL_PROMPT_IDS").is_some() {
            eprintln!("[prompt] {}", ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
        }
        let started = Instant::now();
        let argmax = self.engine.prefill(&ids)?;
        stats.prefill_seconds = started.elapsed().as_secs_f64();
        stats.prompt_tokens = ids.len();
        self.ledger.extend_from_slice(&ids);
        self.assistant_open = true;
        let top = top_from_env();
        let mut step = 0usize;
        show_top(top, step, self.engine.logits());
        let first = self.sampler.choose(argmax, self.engine.logits());

        let mut produced: Vec<u32> = Vec::with_capacity(self.max_tokens);
        let mut detok = DetokenizationFlow::fresh();
        let started = Instant::now();
        let ignore_eos = self.ignore_eos;
        if ignore_eos || !self.tokens.is_stop(first) {
            produced.push(first);
            on_text(&detok.push(&self.tokens.tokenizer, first));
            let room = self.max_tokens.saturating_sub(1);
            let (sampler, tokens) = (&mut self.sampler, &self.tokens);
            let fed = self.engine.generate(first, room, &mut |argmax, logits| {
                step += 1;
                show_top(top, step, logits);
                let token = sampler.choose(argmax, logits);
                if !ignore_eos && tokens.is_stop(token) {
                    return Ok(None);
                }
                produced.push(token);
                on_text(&detok.push(&tokens.tokenizer, token));
                Ok(Some(token))
            })?;
            // `first`, then one per callback except the last at the ceiling.
            produced.truncate(fed.max(1));
        }
        on_text(&detok.balance(&self.tokens.tokenizer));
        stats.decode_seconds = started.elapsed().as_secs_f64();
        stats.generated_tokens = produced.len();
        self.ledger.extend_from_slice(&produced);
        stats.position = self.engine.pos();
        Ok(stats)
    }
}
