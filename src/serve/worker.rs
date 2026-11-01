//! The thread that owns the engine and answers the chat requests one at a time
//! (one slot): the template over the request, the prefix it shares with the
//! previous request kept in the K/V cache, the rest fed token by token, then
//! sampling up to an end of turn, a tool call or the budget, the answer split
//! as it streams (`output.rs`) and sent as OpenAI chunks or one message.
//!
//! Logs, on stderr per request: the prompt, the prefix reused, the speeds, the
//! end, where the time of the steps went and the state of the process; with
//! `--log-dir`, the request, the prompt, the raw output (special tokens
//! included) and the response of each request in files of their own.

use super::output::{OutputParser, Piece, TOOL_RESPONSE_OPEN};
use super::prefix_cache::{PrefixCache, MIN_TOKENS};
use super::template::Template;
use crate::core::sampler::{Sampler, Sampling};
use crate::metal::diag::{Clocks, Proc};
use crate::metal::gemma::GemmaEngine;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;

/// What the worker sends back for one request.
pub enum Out {
    /// One SSE chunk (`data: …`).
    Chunk(Value),
    /// The whole answer of a request that does not stream.
    Message(Value),
    /// The end of a stream (`data: [DONE]`).
    Done,
    /// An HTTP status and the error body (llama-server's shape).
    Error(u16, Value),
}

pub struct Job {
    pub id: u64,
    pub body: Value,
    pub tx: Sender<Out>,
}

pub struct Worker {
    pub engine: GemmaEngine,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<Template>,
    pub model: String,
    pub log_dir: Option<PathBuf>,
    /// The tokens whose K/V the engine holds, in order.
    history: Vec<u32>,
    stops: Vec<u32>,
    /// `--keep-warm SECS`: after this long without a request, one token
    /// through the model (the conversation's cache kept), so that a machine
    /// short of memory does not compress the weights while the server idles
    /// (the next request then waited seconds in the driver).
    pub keep_warm: Option<std::time::Duration>,
    /// The tokens of the state the engine keeps aside (`--side-slot`), when it
    /// keeps one: a request that shares a longer prefix with it than with the
    /// live state swaps the two; a request that would discard most of the live
    /// state saves it first. A client's periodic agent and its conversation thus
    /// each keep their prefix.
    aside: Vec<u32>,
    /// `--prefix-cache DIR`: the K/V of the stable prefix of the requests
    /// (system message and tools) on disk, read back instead of computed.
    pub prefix_cache: Option<PrefixCache>,
}

/// Fewer shared tokens than this are not worth a copy of the state.
const SIDE_MARGIN: usize = 64;

fn unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn error_body(code: u16, kind: &str, message: &str) -> Value {
    json!({ "error": { "code": code, "message": message, "type": kind } })
}

impl Worker {
    pub fn new(engine: GemmaEngine, tokenizer: Arc<Tokenizer>, template: Arc<Template>, model: String, log_dir: Option<PathBuf>) -> Worker {
        let mut stops = engine.config().eos.clone();
        for t in ["<turn|>", "<eos>"] {
            if let Some(id) = tokenizer.token_to_id(t) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        stops.push(TOOL_RESPONSE_OPEN);
        Worker { engine, tokenizer, template, model, log_dir, history: Vec::new(), stops, keep_warm: None, aside: Vec::new(), prefix_cache: None }
    }

    pub fn run(mut self, rx: std::sync::mpsc::Receiver<Job>) {
        eprintln!("[serve] worker ready · stops {:?} · capacity {}", self.stops, self.engine.capacity());
        let mut warm_steps = 0u64;
        loop {
            let job = match self.keep_warm {
                Some(d) => match rx.recv_timeout(d) {
                    Ok(job) => job,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        let t = std::time::Instant::now();
                        warm_steps += 1;
                        match self.engine.keep_warm() {
                            // The first idle step, and the slow ones (the weights had been paged out).
                            Ok(()) if warm_steps == 1 || t.elapsed().as_secs_f64() > 1.0 => eprintln!("[warm] idle step {warm_steps} in {:.2} s", t.elapsed().as_secs_f64()),
                            Ok(()) => {}
                            Err(e) => eprintln!("[warm] idle step failed: {e}"),
                        }
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                },
                None => match rx.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                },
            };
            let (id, tx) = (job.id, job.tx.clone());
            if let Err((code, body)) = self.answer(job) {
                eprintln!("[req {id}] error {code}: {body}");
                let _ = tx.blocking_send(Out::Error(code, body));
            }
        }
    }

    /// The tokens of the request's stable prefix: the template over its system
    /// message (the first, when it is `system` or `developer`) and its tools
    /// alone, when they begin `ids` and leave at least one token after them.
    fn stable_prefix(&self, messages: &Value, tools: Option<&Value>, thinking: bool, ids: &[u32]) -> Option<usize> {
        let first = messages.as_array()?.first().filter(|m| matches!(m["role"].as_str(), Some("system" | "developer")));
        if first.is_none() && tools.and_then(Value::as_array).is_none_or(Vec::is_empty) {
            return None;
        }
        let text = self.template.render(&Value::Array(first.cloned().into_iter().collect()), tools, thinking, false).ok()?;
        let p = self.tokenizer.encode(text.as_str(), false).ok()?.get_ids().to_vec();
        (p.len() >= MIN_TOKENS && p.len() < ids.len() && ids.starts_with(&p)).then_some(p.len())
    }

    fn log_file(&self, id: u64, name: &str, data: &[u8]) {
        if let Some(d) = &self.log_dir {
            let _ = std::fs::write(d.join(format!("req-{id:05}.{name}")), data);
        }
    }

    fn answer(&mut self, job: Job) -> std::result::Result<(), (u16, Value)> {
        let (id, body, tx) = (job.id, job.body, job.tx);
        let bad = |m: String| (400u16, error_body(400, "invalid_request_error", &m));
        let internal = |m: String| (500u16, error_body(500, "server_error", &m));
        self.log_file(id, "request.json", &serde_json::to_vec_pretty(&body).unwrap_or_default());
        let messages = body.get("messages").filter(|m| m.is_array()).ok_or_else(|| bad("messages is required".into()))?;
        let tools = body.get("tools");
        let stream = body["stream"].as_bool().unwrap_or(false);
        let thinking = body.pointer("/chat_template_kwargs/enable_thinking").and_then(Value::as_bool).unwrap_or(false);
        let per_token = body["timings_per_token"].as_bool().unwrap_or(false);
        let include_usage = body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
        let sampling = Sampling {
            temperature: body["temperature"].as_f64().unwrap_or(0.8) as f32,
            top_k: body["top_k"].as_u64().unwrap_or(40) as usize,
            top_p: body["top_p"].as_f64().unwrap_or(0.95) as f32,
            min_p: body["min_p"].as_f64().unwrap_or(0.05) as f32,
            seed: body["seed"].as_u64().unwrap_or_else(|| unix()),
        };
        let max_tokens = body["max_tokens"].as_u64().or_else(|| body["max_completion_tokens"].as_u64()).or_else(|| body["n_predict"].as_u64()).unwrap_or(0) as usize;

        let prompt = self.template.render(messages, tools, thinking, true).map_err(|e| internal(e.to_string()))?;
        self.log_file(id, "prompt.txt", prompt.as_bytes());
        let ids: Vec<u32> = self.tokenizer.encode(prompt.as_str(), false).map_err(|e| internal(format!("tokenizer: {e}")))?.get_ids().to_vec();
        let cap = self.engine.capacity();
        if ids.len() >= cap {
            return Err((
                400,
                json!({ "error": { "code": 400, "type": "exceed_context_size_error", "message": format!("the request exceeds the available context size ({} tokens, n_ctx {cap})", ids.len()), "n_prompt_tokens": ids.len(), "n_ctx": cap } }),
            ));
        }
        // The prefix shared with what the K/V cache holds; at least the last
        // prompt token is fed again, for its logits.
        let shared = |h: &[u32]| h.iter().zip(&ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
        let mut common = shared(&self.history);
        let slots = self.engine.side_slots();
        if slots > 0 {
            let valid = self.engine.side_valid();
            let aside = if valid { shared(&self.aside) } else { 0 };
            let t = Instant::now();
            if valid && aside > common + SIDE_MARGIN {
                // The state kept aside goes on: the two exchanged.
                self.engine.side_swap().map_err(|e| internal(e.to_string()))?;
                std::mem::swap(&mut self.aside, &mut self.history);
                common = aside;
                eprintln!("[req {id}] side: swap, {aside} tokens of the state kept aside reused, the live one of {} tokens kept aside, {:.3} s", self.aside.len(), t.elapsed().as_secs_f64());
            } else if self.history.len() > common + SIDE_MARGIN
                && 2 * common < self.history.len()
                && (!valid || self.history.len() >= self.aside.len())
                && (self.history.len() <= slots || ids.len() < slots)
            {
                // Another conversation: the live state would lose most of what
                // it holds, saved first (unless what is aside is longer, or both
                // the live state and the request run past the head, which
                // would drop it at once).
                self.engine.side_save().map_err(|e| internal(e.to_string()))?;
                self.aside.clone_from(&self.history);
                eprintln!("[req {id}] side: save, the live state of {} tokens kept aside, {:.3} s", self.aside.len(), t.elapsed().as_secs_f64());
            }
        }
        // A sliding ring that lost the window before `common` sends the prompt back to 0.
        common = self.engine.truncate(common).map_err(|e| internal(e.to_string()))?;
        self.history.truncate(common);
        // The longest prefix on disk when the cache holds less of it; the
        // stable prefix on disk after its prefill when no file has it; the
        // prefix a one-shot request shares with the cache past its stable
        // prefix, on disk now (the cache holds it).
        let stable = if self.prefix_cache.is_some() { self.stable_prefix(messages, tools, thinking, &ids) } else { None };
        let one_shot = messages.as_array().is_some_and(|m| m.iter().all(|m| !matches!(m["role"].as_str(), Some("assistant" | "tool"))));
        let mut save_at = None;
        if let (Some(pc), Some(n)) = (self.prefix_cache.as_mut(), stable) {
            if one_shot && common >= n + MIN_TOKENS && !pc.has(&ids[..common]) {
                let t = Instant::now();
                match pc.save(&self.engine, &ids[..common], true) {
                    Ok(bytes) => eprintln!("[req {id}] prefix cache: {common} tokens shared with the cache written to disk, {:.1} MiB in {:.3} s", bytes as f64 / (1u64 << 20) as f64, t.elapsed().as_secs_f64()),
                    Err(e) => eprintln!("[req {id}] prefix cache: {common} shared tokens not written: {e}"),
                }
            }
            if let Some(m) = pc.best(&ids, common) {
                let t = Instant::now();
                match pc.load(&mut self.engine, &ids[..m]) {
                    Ok(bytes) => {
                        self.history.clear();
                        self.history.extend_from_slice(&ids[..m]);
                        eprintln!("[req {id}] prefix cache: {m} reused from disk instead of {common} from the cache, {:.1} MiB in {:.3} s", bytes as f64 / (1u64 << 20) as f64, t.elapsed().as_secs_f64());
                        common = m;
                    }
                    Err(e) => {
                        eprintln!("[req {id}] prefix cache: {e} (file removed), the prompt fed from 0");
                        self.engine.reset().map_err(|e| internal(e.to_string()))?;
                        self.history.clear();
                        common = 0;
                    }
                }
            }
            if !pc.has(&ids[..n]) {
                save_at = Some(n);
            }
        }
        let (st0, pr0, ck0) = (self.engine.stats, Proc::now(), Clocks::now());
        eprintln!(
            "[req {id}] {} messages, {} tools, thinking {thinking}, stream {stream}, temperature {} top_k {} top_p {} min_p {} seed {}, max_tokens {max_tokens} · prompt {} tokens, {common} reused from the cache, {} to feed",
            messages.as_array().map_or(0, Vec::len),
            tools.and_then(Value::as_array).map_or(0, Vec::len),
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.min_p,
            sampling.seed,
            ids.len(),
            ids.len() - common
        );

        let t_prompt = Instant::now();
        let mut from = common;
        // The client gone (a client that closes the connection), the
        // prompt stops at the next block; what was fed stays in the cache.
        let gone = |tx: &Sender<Out>| tx.is_closed();
        let abandon = |w: &mut Worker, at: usize| {
            let fed = w.engine.pos() - at;
            w.history.extend_from_slice(&ids[at..at + fed]);
            eprintln!("[req {id}] cancelled: the client left during the prompt, {} of its {} tokens fed in {:.2} s · cache now {} tokens", at + fed - common, ids.len() - common, t_prompt.elapsed().as_secs_f64(), w.history.len());
        };
        if let (Some(n), true) = (save_at, self.prefix_cache.is_some()) {
            // The prefill split at the end of the prefix (the same bits as in
            // one go), its K/V written before the rest overwrites a ring.
            if from < n {
                if self.engine.prefill_while(&ids[from..n], &mut || !gone(&tx)).map_err(|e| internal(e.to_string()))?.is_none() {
                    abandon(self, from);
                    return Ok(());
                }
                self.history.extend_from_slice(&ids[from..n]);
                from = n;
            }
            let pc = self.prefix_cache.as_mut().expect("checked");
            let t = Instant::now();
            match pc.save(&self.engine, &ids[..n], false) {
                Ok(bytes) => eprintln!("[req {id}] prefix cache: {n} tokens written to disk, {:.1} MiB in {:.3} s", bytes as f64 / (1u64 << 20) as f64, t.elapsed().as_secs_f64()),
                Err(e) => eprintln!("[req {id}] prefix cache: {n} tokens not written: {e}"),
            }
        }
        let Some(argmax) = self.engine.prefill_while(&ids[from..], &mut || !gone(&tx)).map_err(|e| internal(e.to_string()))? else {
            abandon(self, from);
            return Ok(());
        };
        self.history.extend_from_slice(&ids[from..]);
        let prompt_s = t_prompt.elapsed().as_secs_f64();
        let prompt_n = ids.len() - common;

        let created = unix();
        let cid = format!("chatcmpl-herbert-{id}");
        let model = self.model.clone();
        let chunk = |delta: Value, finish: Option<&str>, timings: Option<Value>| -> Value {
            let mut c = json!({ "id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": "herbert-metal-q4", "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
            if let Some(t) = timings {
                c["timings"] = t;
            }
            c
        };
        let timings = |predicted_n: usize, predicted_s: f64| {
            json!({
                "cache_n": common, "prompt_n": prompt_n, "prompt_ms": prompt_s * 1e3, "prompt_per_token_ms": prompt_s * 1e3 / prompt_n.max(1) as f64, "prompt_per_second": prompt_n as f64 / prompt_s.max(1e-9),
                "predicted_n": predicted_n, "predicted_ms": predicted_s * 1e3, "predicted_per_token_ms": predicted_s * 1e3 / predicted_n.max(1) as f64, "predicted_per_second": predicted_n as f64 / predicted_s.max(1e-9),
            })
        };
        if stream && tx.blocking_send(Out::Chunk(chunk(json!({ "role": "assistant", "content": null }), None, None))).is_err() {
            return Ok(());
        }

        let mut sampler = Sampler::new(sampling);
        let mut parser = OutputParser::after_prompt(&self.tokenizer, &ids);
        let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::<Value>::new());
        let mut produced: Vec<u32> = Vec::new();
        let mut finish = "stop";
        let mut cancelled = false;
        let room = if max_tokens == 0 { cap - ids.len() } else { max_tokens.min(cap - ids.len()) };
        let t_gen = Instant::now();
        let tok = self.tokenizer.clone();
        let stops = self.stops.clone();
        // One piece of the answer: kept for the final message, streamed when asked.
        let emit = |pieces: Vec<Piece>, n: usize, t: f64, content: &mut String, reasoning: &mut String, calls: &mut Vec<Value>| -> bool {
            for p in pieces {
                let delta = match p {
                    Piece::Content(s) => {
                        content.push_str(&s);
                        json!({ "content": s })
                    }
                    Piece::Reasoning(s) => {
                        reasoning.push_str(&s);
                        json!({ "reasoning_content": s })
                    }
                    Piece::ToolCall { name, arguments } => {
                        let call = json!({ "id": format!("call_{id}_{}", calls.len()), "type": "function", "function": { "name": name, "arguments": arguments } });
                        let mut d = call.clone();
                        d["index"] = json!(calls.len());
                        calls.push(call);
                        json!({ "tool_calls": [d] })
                    }
                };
                if stream && tx.blocking_send(Out::Chunk(chunk(delta, None, per_token.then(|| timings(n, t))))).is_err() {
                    return false;
                }
            }
            true
        };
        let first = sampler.choose(argmax, self.engine.logits());
        if stops.contains(&first) {
            produced.push(first);
        } else {
            produced.push(first);
            let pieces = parser.push(&tok, first);
            if !emit(pieces, 1, t_gen.elapsed().as_secs_f64(), &mut content, &mut reasoning, &mut calls) {
                cancelled = true;
            }
            if !cancelled && room > 1 {
                let fed = self
                    .engine
                    .generate(first, room - 1, &mut |argmax, logits| {
                        if tx.is_closed() {
                            // The client left (a request that does not stream sees it only here).
                            cancelled = true;
                            return Ok(None);
                        }
                        let t = sampler.choose(argmax, logits);
                        produced.push(t);
                        if stops.contains(&t) {
                            return Ok(None);
                        }
                        let pieces = parser.push(&tok, t);
                        if !emit(pieces, produced.len(), t_gen.elapsed().as_secs_f64(), &mut content, &mut reasoning, &mut calls) {
                            cancelled = true;
                            return Ok(None);
                        }
                        Ok(Some(t))
                    })
                    .map_err(|e| internal(e.to_string()))?;
                self.history.extend_from_slice(&produced[..fed]);
                if !cancelled && !stops.contains(produced.last().unwrap_or(&0)) {
                    finish = "length";
                }
            } else if room <= 1 {
                finish = "length";
            }
        }
        let tail = parser.finish(&tok);
        emit(tail, produced.len(), t_gen.elapsed().as_secs_f64(), &mut content, &mut reasoning, &mut calls);
        if !calls.is_empty() && finish == "stop" {
            finish = "tool_calls";
        }
        if cancelled {
            finish = "cancelled";
        }
        let gen_s = t_gen.elapsed().as_secs_f64();
        // Tokens counted as llama-server counts them: the end-of-turn token is one of them.
        let predicted_n = produced.len();
        let usage = json!({ "prompt_tokens": ids.len(), "completion_tokens": predicted_n, "total_tokens": ids.len() + predicted_n, "prompt_tokens_details": { "cached_tokens": common } });
        let t_all = timings(predicted_n, gen_s);
        self.log_file(id, "raw.txt", self.tokenizer.decode(&produced, false).unwrap_or_default().as_bytes());
        let answer = if stream {
            let _ = tx.blocking_send(Out::Chunk(chunk(json!({}), Some(finish), Some(t_all.clone()))));
            if include_usage {
                let _ = tx.blocking_send(Out::Chunk(json!({ "id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage, "timings": t_all })));
            }
            let _ = tx.blocking_send(Out::Done);
            json!({ "streamed": true, "content": content, "reasoning_content": reasoning, "tool_calls": calls, "finish_reason": finish, "usage": usage, "timings": t_all })
        } else {
            let mut msg = json!({ "role": "assistant", "content": content });
            if !reasoning.is_empty() {
                msg["reasoning_content"] = json!(reasoning);
            }
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            let full = json!({ "id": cid, "object": "chat.completion", "created": created, "model": model, "system_fingerprint": "herbert-metal-q4", "choices": [{ "index": 0, "message": msg, "finish_reason": finish }], "usage": usage, "timings": t_all });
            let _ = tx.blocking_send(Out::Message(full.clone()));
            full
        };
        self.log_file(id, "response.json", &serde_json::to_vec_pretty(&answer).unwrap_or_default());
        eprintln!(
            "[req {id}] {finish}: prompt {prompt_n} tokens in {prompt_s:.2} s ({:.1} tok/s), {predicted_n} generated in {gen_s:.2} s ({:.2} tok/s), {} tool calls, content {} chars, reasoning {} chars · cache now {} tokens",
            prompt_n as f64 / prompt_s.max(1e-9),
            predicted_n as f64 / gen_s.max(1e-9),
            calls.len(),
            content.len(),
            reasoning.len(),
            self.history.len()
        );
        eprintln!("[req {id}] {} · {}", Clocks::now().since(&ck0), self.engine.stats.since(&st0));
        eprintln!("[req {id}] {}", Proc::now().since(&pr0));
        Ok(())
    }
}
