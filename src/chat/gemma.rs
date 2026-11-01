//! A Gemma-4 conversation: the `<|turn>` template rendered turn by turn, the
//! stop rule of the model's `eos_token_id` (`<eos>` and `<turn|>`), the same
//! counting of generated tokens as herbert-vk-bf16's session, so that the
//! `[ids]` of both engines and of the CPU oracle compare one to one. Text only.

use crate::chat::detok::DetokenizationFlow;
use crate::chat::{show_top, top_from_env, TurnStats};
use crate::core::error::{HerbertError, Result};
use crate::core::sampler::{Sampler, Sampling};
use crate::metal::gemma::GemmaEngine;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

pub struct GemmaChat {
    engine: GemmaEngine,
    tokenizer: Tokenizer,
    stops: Vec<u32>,
    sampler: Sampler,
    system: Option<String>,
    pub max_tokens: usize,
    /// Go on past the end of turn (the logits comparison of the quality protocol).
    pub ignore_eos: bool,
    ledger: Vec<u32>,
    assistant_open: bool,
}

impl GemmaChat {
    pub fn new(engine: GemmaEngine, model_dir: &Path, sampling: Sampling, system: Option<String>) -> Result<GemmaChat> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path).map_err(|e| HerbertError::ModelLoad(format!("{}: {e}", path.display())))?;
        let mut stops = engine.config().eos.clone();
        for name in ["<turn|>", "<eos>"] {
            if let Some(id) = tokenizer.token_to_id(name) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        Ok(GemmaChat { engine, tokenizer, stops, sampler: Sampler::new(sampling), system, max_tokens: 1024, ignore_eos: false, ledger: Vec::new(), assistant_open: false })
    }

    pub fn engine(&self) -> &GemmaEngine {
        &self.engine
    }

    pub fn summary(&self) -> String {
        format!("stops {:?} · vocab {}", self.stops, self.tokenizer.get_vocab_size(true))
    }

    pub fn reset(&mut self) -> Result<()> {
        self.engine.reset()?;
        self.ledger.clear();
        self.assistant_open = false;
        Ok(())
    }

    pub fn ledger(&self) -> &[u32] {
        &self.ledger
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| HerbertError::Backend(format!("tokenizer: {e}")))?.get_ids().to_vec())
    }

    /// One turn: the user's text in, the answer streamed to `on_text`.
    pub fn chat(&mut self, user: &str, on_text: &mut dyn FnMut(&str)) -> Result<TurnStats> {
        let mut stats = TurnStats::default();
        let mut text = String::new();
        if self.ledger.is_empty() {
            text.push_str("<bos>");
            if let Some(s) = self.system.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                text.push_str(&format!("<|turn>system\n{s}<turn|>\n"));
            }
        } else if self.assistant_open {
            text.push_str("<turn|>\n");
        }
        text.push_str(&format!("<|turn>user\n{}<turn|>\n<|turn>model\n", user.trim()));
        let ids = self.encode(&text)?;
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
        let no_stops: Vec<u32> = Vec::new();
        let stop_ids = if self.ignore_eos { &no_stops } else { &self.stops };
        if !stop_ids.contains(&first) {
            produced.push(first);
            on_text(&detok.push(&self.tokenizer, first));
            let room = self.max_tokens.saturating_sub(1);
            let (sampler, stops, tokenizer) = (&mut self.sampler, stop_ids, &self.tokenizer);
            let fed = self.engine.generate(first, room, &mut |argmax, logits| {
                step += 1;
                show_top(top, step, logits);
                let token = sampler.choose(argmax, logits);
                if stops.contains(&token) {
                    return Ok(None);
                }
                produced.push(token);
                on_text(&detok.push(tokenizer, token));
                Ok(Some(token))
            })?;
            produced.truncate(fed.max(1));
        }
        on_text(&detok.balance(&self.tokenizer));
        stats.decode_seconds = started.elapsed().as_secs_f64();
        stats.generated_tokens = produced.len();
        self.ledger.extend_from_slice(&produced);
        stats.position = self.engine.pos();
        Ok(stats)
    }
}
