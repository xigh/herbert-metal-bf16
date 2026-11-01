//! herbert-metal-ppl MODEL_DIR TEXT_FILE [options]
//!
//! The perplexity of Gemma-4 or Qwen3.5 over a raw text, scored as `llama-perplexity` (and herbert-vk-q4's `herbert-vk-ppl`)
//! score it: the text tokenised whole with `<bos>` in front, cut into windows of `--n-ctx` tokens whose first token is
//! replaced by `<bos>` (Qwen, whose tokenizer adds none: the text as it is, each window from its own first token), and in each window the negative log-likelihood of the tokens of its second half (position `j` in
//! `n_ctx/2 .. n_ctx − 1` predicting token `j + 1`: `n_ctx/2 − 1` targets per window). Each window runs from position 0, token
//! by token through the decode step, the head in f32 (Gemma's soft cap in f32, no 16-bit rounding of the logits; Qwen: the
//! head once more after the step's, `logits_f32`); the log-sum-exp is taken on the host in f64.
//!
//!   --n-ctx N            tokens per window (default 512)
//!   --chunks N           the first N windows only (default: all)
//!   --q4                 the projections and the head in Q4_0, quantized at load (default: 16 bits)
//!   --nll FILE           every target's NLL, f64 little-endian, in order (for a paired comparison of two arms)
//!   --tune K=V[,K=V]     Qwen: the engine's settings (herbert-metal-chat's)
//!
//! On stdout: the running perplexity per window (`[i]ppl`), then `Final estimate: PPL = … +/- …`, the mean NLL with its
//! standard error, and the share of targets that were the argmax.

use herbert_metal::metal::diag::{now, Clocks, Proc};
use herbert_metal::metal::gemma::{GemmaEngine, Options};
use herbert_metal::metal::qwen::{Options as QwenOptions, QwenEngine, Tune};
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("error: {e}\nusage: herbert-metal-ppl MODEL_DIR TEXT_FILE [--n-ctx N] [--chunks N] [--q4] [--nll FILE] [--tune K=V,...]");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut it = args.iter();
    let model_dir = PathBuf::from(it.next().ok_or("MODEL_DIR is required")?);
    let text_path = PathBuf::from(it.next().ok_or("TEXT_FILE is required")?);
    let (mut n_ctx, mut chunks, mut nll_path) = (512usize, usize::MAX, None);
    let mut tune = Tune::default();
    let mut o = Options { capacity: 0, matvec_rows: 2, dump_msl: None, q4: false, logits_f32: true, prefill_block: 0, sliding_ring: 0 };
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--n-ctx" => n_ctx = value()?.parse().map_err(|e| format!("--n-ctx: {e}"))?,
            "--chunks" => chunks = value()?.parse().map_err(|e| format!("--chunks: {e}"))?,
            "--bf16" => o.q4 = false,
            "--q4" => o.q4 = true,
            "--nll" => nll_path = Some(PathBuf::from(value()?)),
            "--tune" => tune.apply(&value()?)?,
            other => return Err(format!("unknown option {other}")),
        }
    }
    o.capacity = n_ctx;
    let dir = herbert_metal::model::resolve_model_dir(&model_dir).map_err(|e| e.to_string())?;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
    let qwen = model_type(&dir)?.starts_with("qwen3_5");
    let bos = if qwen { None } else { Some(tok.token_to_id("<bos>").ok_or("no <bos> in the tokenizer")?) };
    let text = std::fs::read_to_string(&text_path).map_err(|e| format!("{}: {e}", text_path.display()))?;
    let enc = tok.encode(text.as_str(), false).map_err(|e| format!("tokenize: {e}"))?;
    let mut ids: Vec<u32> = bos.into_iter().collect();
    ids.extend_from_slice(enc.get_ids());
    let n_chunk = (ids.len() / n_ctx).min(chunks);
    eprintln!("[ppl] {} tokens ({} bytes); {} windows of {n_ctx}, {} scored; the first ids {:?}", ids.len(), text.len(), ids.len() / n_ctx, n_chunk, &ids[..16.min(ids.len())]);
    let mut engine = if qwen {
        let qo = QwenOptions { capacity: n_ctx, matvec_rows: 2, dump_msl: None, tune, q4: o.q4, logits_f32: true };
        Engine::Qwen(Box::new(QwenEngine::load(&dir, &qo).map_err(|e| e.to_string())?))
    } else {
        Engine::Gemma(Box::new(GemmaEngine::load(&dir, &o).map_err(|e| e.to_string())?))
    };
    eprintln!("[engine] loaded in {:.1} s", engine.load_seconds());
    let t0 = std::time::Instant::now();
    let (mut sum, mut sum2, mut count, mut hits) = (0f64, 0f64, 0usize, 0usize);
    let mut nlls: Vec<f64> = Vec::new();
    let first = n_ctx / 2;
    let (mut st_prev, mut proc_prev, mut clk_prev) = (engine.stats(), Proc::now(), Clocks::now());
    eprintln!("[proc] at the start: {}", proc_prev.since(&Proc::default()));
    for c in 0..n_chunk {
        let tw = std::time::Instant::now();
        let mut host_lse = 0f64;
        engine.reset().map_err(|e| e.to_string())?;
        let start = c * n_ctx;
        for j in 0..n_ctx - 1 {
            let t = match bos {
                Some(b) if j == 0 => b,
                _ => ids[start + j],
            };
            let logits = engine.score(t).map_err(|e| e.to_string())?;
            if j < first {
                continue;
            }
            let tl = now();
            let target = ids[start + j + 1] as usize;
            let (mut best, mut arg) = (f32::NEG_INFINITY, 0usize);
            for (i, &l) in logits.iter().enumerate() {
                if l > best {
                    (best, arg) = (l, i);
                }
            }
            let lse = best as f64 + logits.iter().map(|&l| ((l - best) as f64).exp()).sum::<f64>().ln();
            let nll = lse - logits[target] as f64;
            (sum, sum2, count) = (sum + nll, sum2 + nll * nll, count + 1);
            hits += (arg == target) as usize;
            nlls.push(nll);
            host_lse += now() - tl;
        }
        println!("[{}]{:.4}", c + 1, (sum / count as f64).exp());
        let (st, pr) = (engine.stats(), Proc::now());
        eprintln!(
            "[window {}] {:.2} s awake, {}, log-sum-exp on the host {:.2} ms per target · {} · {}",
            c + 1,
            tw.elapsed().as_secs_f64(),
            Clocks::now().since(&clk_prev),
            1e3 * host_lse / (n_ctx - 1 - first) as f64,
            st.map(|s| s.since(&st_prev.unwrap_or_default()).to_string()).unwrap_or_else(|| "Qwen: no step timings".into()),
            pr.since(&proc_prev)
        );
        (st_prev, proc_prev, clk_prev) = (st, pr, Clocks::now());
        let _ = std::io::stdout().flush();
    }
    let mean = sum / count as f64;
    let var = (sum2 / count as f64 - mean * mean).max(0.0);
    let se = (var / (count as f64 - 1.0)).sqrt();
    let ppl = mean.exp();
    println!("Final estimate: PPL = {ppl:.4} +/- {:.5}", ppl * se);
    println!("mean NLL {mean:.5} +/- {se:.5} over {count} targets; argmax hits {:.2} %; {:.1} s ({:.1} tokens/s)", 100.0 * hits as f64 / count as f64, t0.elapsed().as_secs_f64(), (n_chunk * (n_ctx - 1)) as f64 / t0.elapsed().as_secs_f64());
    if let Some(p) = nll_path {
        let bytes: Vec<u8> = nlls.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

enum Engine {
    Gemma(Box<GemmaEngine>),
    Qwen(Box<QwenEngine>),
}

impl Engine {
    fn load_seconds(&self) -> f64 {
        match self {
            Engine::Gemma(e) => e.load_seconds,
            Engine::Qwen(e) => e.load_seconds,
        }
    }
    fn stats(&self) -> Option<herbert_metal::metal::diag::StepStats> {
        match self {
            Engine::Gemma(e) => Some(e.stats),
            Engine::Qwen(_) => None,
        }
    }
    fn reset(&mut self) -> herbert_metal::core::error::Result<()> {
        match self {
            Engine::Gemma(e) => e.reset(),
            Engine::Qwen(e) => e.reset(),
        }
    }
    fn score(&mut self, t: u32) -> herbert_metal::core::error::Result<&[f32]> {
        match self {
            Engine::Gemma(e) => e.score(t),
            Engine::Qwen(e) => e.score(t),
        }
    }
}

fn model_type(dir: &std::path::Path) -> Result<String, String> {
    let bytes = std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("config.json: {e}"))?;
    let t = v.get("text_config").and_then(|t| t.get("model_type")).or_else(|| v.get("model_type"));
    Ok(t.and_then(|t| t.as_str()).unwrap_or_default().to_string())
}
