//! herbert-metal-chat MODEL_DIR [options]
//!
//! Loads Gemma-4 (E2B, E4B, 12B, 26B-A4B) or Qwen3.5 (`model_type` of
//! `config.json`) on the Apple GPU, its weights in 16 bits (f16 with the `f16`
//! feature, else bf16: the witness) or its projections and head in Q4_0
//! quantized at load (`--q4`), and runs the `--ask`
//! turns, or a one-question demonstration.
//!
//!   --q4                 the projections and the head in Q4_0, quantized at load
//!   --bf16               the 16-bit weights (the default)
//!   --ctx N              positions reserved for the conversation (default 4096)
//!   --max-tokens N       ceiling per answer (default 1024)
//!   --system TEXT        the system prompt (default: English, answers in French)
//!   --no-system          no system message at all
//!   --ask TEXT|@FILE     a user turn (repeatable; `@FILE` reads it from a file)
//!   --temperature T      sampling instead of greedy (Gemma's top-k 64 / top-p 0.95)
//!   --prefill-block N    tokens of a prompt block (Gemma; the projections as one Q4 product with --q4; default 16, 1 = token by token)
//!   --sliding-ring N     the sliding layers' K/V caches as rings of N positions (default 0: the whole capacity)
//!   --fresh              each --ask in a conversation of its own (the context is reset)
//!   --ignore-eos         go on past the end of turn, up to --max-tokens
//!   --no-think           Qwen3.5: the thinking block closed empty
//!   --rows N             rows per simdgroup of the projections (default 2)
//!   --tune K=V[,K=V]     Qwen: the prefill's settings (herbert-vk-bf16's names:
//!                        prefill.block, rec.chunk, rec.pcols, rec.pslices,
//!                        conv.cols, sdpa.prefill, sdpa.bq, sdpa.bk, gemm.simt…)
//!   --dump-msl DIR       write every emitted kernel to DIR/<kernel>.metal
//!
//! On stderr, per turn: `[ids]` (the generated ids, for a comparison with the
//! CPU oracle's), `[decoded]`, `[turn]` (counts and speeds).

use herbert_metal::chat::gemma::GemmaChat;
use herbert_metal::chat::qwen;
use herbert_metal::chat::TurnStats;
use herbert_metal::core::error::Result as HResult;
use herbert_metal::core::sampler::Sampling;
use herbert_metal::metal::diag::StepStats;
use herbert_metal::metal::gemma::{GemmaEngine, Options};
use herbert_metal::metal::qwen::{Options as QwenOptions, QwenEngine, Tune};
use std::io::Write;
use std::path::PathBuf;

struct Args {
    model_dir: PathBuf,
    engine: Options,
    max_tokens: usize,
    system: Option<String>,
    asks: Vec<String>,
    sampling: Sampling,
    ignore_eos: bool,
    fresh: bool,
    think: bool,
    tune: Tune,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut it = args.iter();
    let model_dir = PathBuf::from(it.next().ok_or("MODEL_DIR is required")?);
    let mut a = Args {
        model_dir,
        engine: Options { capacity: 4096, matvec_rows: 2, dump_msl: None, q4: false, logits_f32: false, prefill_block: 16, sliding_ring: 0 },
        max_tokens: 1024,
        system: Some("You are a helpful assistant. Always answer in French.".to_string()),
        asks: Vec::new(),
        sampling: Sampling::greedy(),
        ignore_eos: false,
        fresh: false,
        think: true,
        tune: Tune::default(),
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--ignore-eos" => a.ignore_eos = true,
            "--prefill-block" => a.engine.prefill_block = value()?.parse().map_err(|e| format!("--prefill-block: {e}"))?,
            "--sliding-ring" => a.engine.sliding_ring = value()?.parse().map_err(|e| format!("--sliding-ring: {e}"))?,
            "--fresh" => a.fresh = true,
            "--bf16" => a.engine.q4 = false,
            "--q4" => a.engine.q4 = true,
            "--ctx" => a.engine.capacity = value()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            "--rows" => {
                a.engine.matvec_rows = value()?.parse().map_err(|e| format!("--rows: {e}"))?;
                a.tune.apply("mv.auto=0")?
            }
            "--dump-msl" => a.engine.dump_msl = Some(PathBuf::from(value()?)),
            "--tune" => {
                let v = value()?;
                a.tune.apply(&v)?
            }
            "--max-tokens" => a.max_tokens = value()?.parse().map_err(|e| format!("--max-tokens: {e}"))?,
            "--system" => a.system = Some(value()?),
            "--no-system" => a.system = None,
            "--no-think" => a.think = false,
            "--ask" => {
                let v = value()?;
                match v.strip_prefix('@') {
                    Some(path) => a.asks.push(std::fs::read_to_string(path).map_err(|e| format!("--ask @{path}: {e}"))?),
                    None => a.asks.push(v),
                }
            }
            "--temperature" => {
                let t: f32 = value()?.parse().map_err(|e| format!("--temperature: {e}"))?;
                a.sampling = Sampling { temperature: t, top_k: 64, top_p: 0.95, min_p: 0.0, seed: 42 };
            }
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(a)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\nusage: herbert-metal-chat MODEL_DIR [--q4] [--ctx N] [--max-tokens N] [--system TEXT | --no-system] [--ask TEXT|@FILE]... [--temperature T] [--no-think] [--rows N] [--prefill-block N] [--sliding-ring N] [--tune K=V,...] [--ignore-eos] [--fresh] [--dump-msl DIR]");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(a) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// What the turn loop needs from a session, whatever the model.
trait Session {
    fn turn(&mut self, ask: &str, on_text: &mut dyn FnMut(&str)) -> HResult<TurnStats>;
    fn ledger(&self) -> &[u32];
    fn decode(&self, ids: &[u32]) -> String;
    /// GPU seconds and steps so far.
    fn gpu(&self) -> (f64, usize);
    /// The step timings so far.
    fn stats(&self) -> StepStats;
    /// A new conversation: the context emptied.
    fn reset(&mut self) -> HResult<()>;
}

impl Session for GemmaChat {
    fn stats(&self) -> StepStats {
        self.engine().stats
    }
    fn reset(&mut self) -> HResult<()> {
        GemmaChat::reset(self)
    }
    fn turn(&mut self, ask: &str, on_text: &mut dyn FnMut(&str)) -> HResult<TurnStats> {
        self.chat(ask, on_text)
    }
    fn ledger(&self) -> &[u32] {
        GemmaChat::ledger(self)
    }
    fn decode(&self, ids: &[u32]) -> String {
        GemmaChat::decode(self, ids)
    }
    fn gpu(&self) -> (f64, usize) {
        (self.engine().gpu_seconds, self.engine().steps)
    }
}

impl Session for qwen::Chat {
    fn stats(&self) -> StepStats {
        // The Qwen engine replays its recorded step without these timings.
        StepStats::default()
    }
    fn reset(&mut self) -> HResult<()> {
        qwen::Chat::reset(self)
    }
    fn turn(&mut self, ask: &str, on_text: &mut dyn FnMut(&str)) -> HResult<TurnStats> {
        self.chat(ask, on_text)
    }
    fn ledger(&self) -> &[u32] {
        qwen::Chat::ledger(self)
    }
    fn decode(&self, ids: &[u32]) -> String {
        self.tokens().decode(ids)
    }
    fn gpu(&self) -> (f64, usize) {
        (self.engine().gpu_seconds, self.engine().steps)
    }
}

/// `model_type` of `config.json`, at the top or under `text_config`.
fn model_type(dir: &std::path::Path) -> HResult<String> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let t = v.get("text_config").and_then(|t| t.get("model_type")).or_else(|| v.get("model_type"));
    Ok(t.and_then(|t| t.as_str()).unwrap_or_default().to_string())
}

fn run(a: Args) -> HResult<()> {
    let dir = herbert_metal::model::resolve_model_dir(&a.model_dir)?;
    let mut chat: Box<dyn Session> = if model_type(&dir)?.starts_with("qwen3_5") {
        let o = QwenOptions { capacity: a.engine.capacity, matvec_rows: a.engine.matvec_rows, dump_msl: a.engine.dump_msl.clone(), tune: a.tune.clone(), q4: a.engine.q4, logits_f32: false };
        let engine = QwenEngine::load(&dir, &o)?;
        eprintln!("[engine] loaded in {:.1} s", engine.load_seconds);
        let tokens = qwen::Tokens::load(&dir)?;
        eprintln!("[tokens] {}", tokens.summary());
        let mut chat = qwen::Chat::new(engine, tokens, a.sampling, a.system.clone());
        chat.max_tokens = a.max_tokens;
        chat.think = a.think;
        chat.ignore_eos = a.ignore_eos;
        Box::new(chat)
    } else {
        let engine = GemmaEngine::load(&dir, &a.engine)?;
        eprintln!("[engine] loaded in {:.1} s", engine.load_seconds);
        let mut chat = GemmaChat::new(engine, &dir, a.sampling, a.system.clone())?;
        chat.max_tokens = a.max_tokens;
        chat.ignore_eos = a.ignore_eos;
        eprintln!("[tokens] {}", chat.summary());
        Box::new(chat)
    };
    let asks = if a.asks.is_empty() { vec!["Quelle est la capitale de la France ?".to_string()] } else { a.asks.clone() };
    for (n, ask) in asks.iter().enumerate() {
        if a.fresh && n > 0 {
            chat.reset()?;
        }
        eprintln!("[ask {}]", n + 1);
        println!("\n>>> {ask}");
        let mut out = std::io::stdout();
        let (gpu0, steps0) = chat.gpu();
        let (st0, pr0, ck0) = (chat.stats(), herbert_metal::metal::diag::Proc::now(), herbert_metal::metal::diag::Clocks::now());
        let stats = chat.turn(ask, &mut |text| {
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        })?;
        println!();
        let ledger = chat.ledger();
        let generated = &ledger[ledger.len().saturating_sub(stats.generated_tokens)..];
        eprintln!("[ids] {}", generated.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
        eprintln!("[decoded] {:?}", chat.decode(generated));
        let (gpu1, steps1) = chat.gpu();
        let (gpu, steps) = (gpu1 - gpu0, steps1 - steps0);
        eprintln!(
            "[turn {}] {} prompt tokens in {:.3} s ({:.1} tok/s) · {} generated in {:.2} s ({:.2} tok/s) · position {} · GPU {:.2} ms per step",
            n + 1,
            stats.prompt_tokens,
            stats.prefill_seconds,
            stats.prefill_tokens_per_second(),
            stats.generated_tokens,
            stats.decode_seconds,
            stats.decode_tokens_per_second(),
            stats.position,
            1e3 * gpu / steps.max(1) as f64
        );
        eprintln!("[timing {}] {} · {}", n + 1, herbert_metal::metal::diag::Clocks::now().since(&ck0), chat.stats().since(&st0));
        eprintln!("[proc {}] {}", n + 1, herbert_metal::metal::diag::Proc::now().since(&pr0));
    }
    Ok(())
}
