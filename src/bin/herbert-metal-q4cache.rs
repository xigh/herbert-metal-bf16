//! Write the cache of quantized weights of a model, once, on a machine that
//! has its bf16 checkpoint: a new directory with `herbert-q4.safetensors`
//! (every projection in Q4_0 as the engine uploads it, and every other bf16
//! tensor it reads) and the model's small files (`config.json`, the
//! tokenizer, the chat template). The engine loads that directory without the
//! bf16 shards and without quantizing. Gemma-4, or Qwen3.5 (`model_type` of
//! `config.json`), whose cache also keeps its f32 tensors.
//!
//! usage: herbert-metal-q4cache MODEL_DIR OUT_DIR   (OUT_DIR must not exist)

use herbert_metal::core::error::Result as HResult;
use herbert_metal::metal::gemma::{GemmaEngine, Options};
use herbert_metal::metal::qwen::{Options as QwenOptions, QwenEngine, Tune};
use herbert_metal::model::gemma::Q4_CACHE;
use std::path::PathBuf;

const SMALL: [&str; 6] = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: herbert-metal-q4cache MODEL_DIR OUT_DIR");
        std::process::exit(2);
    }
    let (model, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    if out.exists() {
        eprintln!("{}: exists, nothing is overwritten", out.display());
        std::process::exit(1);
    }
    if model.join(Q4_CACHE).exists() {
        eprintln!("{}: already a cache directory", model.display());
        std::process::exit(1);
    }
    // The engine of the model, loaded only to quantize; its cache writer.
    let loaded: HResult<Box<dyn Fn(&std::path::Path) -> HResult<u64>>> = if model_type(&model).starts_with("qwen3_5") {
        let mut tune = Tune::default();
        tune.apply("prefill.block=1,mv.auto=0").expect("prefill.block=1");
        let o = QwenOptions { capacity: 64, matvec_rows: 2, dump_msl: None, tune, q4: true, logits_f32: false };
        QwenEngine::load(&model, &o).map(|e| Box::new(move |p: &std::path::Path| e.write_q4_cache(p)) as Box<dyn Fn(&std::path::Path) -> HResult<u64>>)
    } else {
        let o = Options { capacity: 64, matvec_rows: 2, dump_msl: None, q4: true, logits_f32: false, prefill_block: 0, sliding_ring: 0 };
        GemmaEngine::load(&model, &o).map(|e| Box::new(move |p: &std::path::Path| e.write_q4_cache(p)) as Box<dyn Fn(&std::path::Path) -> HResult<u64>>)
    };
    let write = match loaded {
        Ok(w) => w,
        Err(e) => {
            eprintln!("load: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = std::fs::create_dir(&out) {
        eprintln!("{}: {e}", out.display());
        std::process::exit(1);
    }
    let t = std::time::Instant::now();
    let tmp = out.join(format!("{Q4_CACHE}.partial"));
    match write(&tmp) {
        Ok(bytes) => eprintln!("[q4cache] {} bytes in {:.1} s", bytes, t.elapsed().as_secs_f64()),
        Err(e) => {
            eprintln!("write: {e}");
            std::process::exit(1);
        }
    }
    for f in SMALL {
        let src = model.join(f);
        if src.exists() {
            if let Err(e) = std::fs::copy(&src, out.join(f)) {
                eprintln!("{f}: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Err(e) = std::fs::rename(&tmp, out.join(Q4_CACHE)) {
        eprintln!("rename: {e}");
        std::process::exit(1);
    }
    eprintln!("[q4cache] {} ready", out.display());
}

/// `model_type` of `config.json`, under `text_config` or at the top; empty if unreadable.
fn model_type(dir: &std::path::Path) -> String {
    let v: serde_json::Value = std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let t = v.get("text_config").and_then(|t| t.get("model_type")).or_else(|| v.get("model_type"));
    t.and_then(|t| t.as_str()).unwrap_or_default().to_string()
}
