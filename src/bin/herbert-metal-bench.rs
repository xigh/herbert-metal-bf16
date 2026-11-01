//! herbert-metal-bench — prompt and generation speed of a Qwen3.5 or Gemma-4
//! checkpoint (`model_type` of `config.json`), measured as llama-bench
//! measures them.
//!
//! usage: herbert-metal-bench MODEL_DIR [-p N[,N…]] [-n N] [-d N] [-r R] [-w W] [--sustain S] [--cool] [--ctx N] [--rows N] [--q4] [--tune K=V,...] [--dump-msl DIR]
//!
//!   -p N[,N…]   prompt lengths (`ppN`, default 512): N pseudo-random token ids
//!               fed from an empty conversation, the time of the whole prefill
//!               (blocks, the head on the last row, the argmax read back)
//!   -n N        generated tokens (`tgN`, default 128): from an empty
//!               conversation, each token the argmax of the one before; `0`
//!               skips the test
//!   -d N        the tgN after an untimed prefill of N pseudo-random ids
//!               (llama-bench's `-d`, the test named `tgN @ dN`; default 0)
//!   -r R        timed repetitions of each test (default 3)
//!   -w W        untimed warm-up runs of each test (default 1)
//!   --sustain S after the warm-up, repeat each test for S seconds and keep
//!               the runs of the second half (the steady state of a machine
//!               that throttles: the M5), the first run reported apart
//!   --q4        the projections and the head in Q4_0, quantized at load
//!               (default: the 16-bit weights)
//!   --dump-msl DIR  write every emitted kernel to DIR/<kernel>.metal
//!
//! The ids are those of a fixed linear congruential generator below the first
//! special token (Qwen; Gemma: below the vocabulary size), the same for every
//! engine of a campaign that takes ids. Wall-clock time, the host's encoding
//! included, and the GPU time of the command buffers (`GPUEndTime −
//! GPUStartTime`, summed). One line per run on stderr, the llama-bench table
//! (mean ± standard deviation) on stdout, wall-clock then GPU. Gemma's
//! `--tune` is ignored. The engine is named `herbert-metal-f16` when built
//! with the `f16` feature, `herbert-metal-bf16+native` under
//! `HERBERT_METAL_BF16_NATIVE=1`.

use herbert_metal::core::error::Result as HResult;
use herbert_metal::metal::gemma::{GemmaEngine, Options as GemmaOptions};
use herbert_metal::metal::qwen::{Options, QwenEngine, Tune};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model_dir: PathBuf,
    prompts: Vec<usize>,
    gen: usize,
    depth: usize,
    reps: usize,
    warmup: usize,
    sustain: f64,
    /// `--cool`: after the load (its tuning heats the GPU for minutes on the M5), wait for the thermal pressure level
    /// to come back to 0 (at most 300 s) before the warm-up: the measure then does not depend on how long the
    /// tuning took.
    cool: bool,
    o: Options,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut it = args.iter();
    let model_dir = PathBuf::from(it.next().ok_or("MODEL_DIR is required")?);
    let mut a = Args {
        model_dir,
        prompts: vec![512],
        gen: 128,
        depth: 0,
        reps: 3,
        warmup: 1,
        sustain: 0.0,
        cool: false,
        o: Options { capacity: 0, matvec_rows: 2, dump_msl: None, tune: Tune::default(), q4: false, logits_f32: false },
    };
    let mut ctx = None;
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "-p" => a.prompts = value()?.split(',').map(|s| s.parse().map_err(|e| format!("-p: {e}"))).collect::<Result<_, _>>()?,
            "-n" => a.gen = value()?.parse().map_err(|e| format!("-n: {e}"))?,
            "-d" => a.depth = value()?.parse().map_err(|e| format!("-d: {e}"))?,
            "-r" => a.reps = value()?.parse::<usize>().map_err(|e| format!("-r: {e}"))?.max(1),
            "-w" => a.warmup = value()?.parse().map_err(|e| format!("-w: {e}"))?,
            "--sustain" => a.sustain = value()?.parse().map_err(|e| format!("--sustain: {e}"))?,
            "--cool" => a.cool = true,
            "--ctx" => ctx = Some(value()?.parse::<usize>().map_err(|e| format!("--ctx: {e}"))?),
            "--bf16" => a.o.q4 = false,
            "--q4" => a.o.q4 = true,
            "--rows" => {
                a.o.matvec_rows = value()?.parse().map_err(|e| format!("--rows: {e}"))?;
                a.o.tune.apply("mv.auto=0")?
            }
            "--dump-msl" => a.o.dump_msl = Some(PathBuf::from(value()?)),
            "--tune" => {
                let v = value()?;
                a.o.tune.apply(&v)?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    // Room for the longest test (and a block of rows).
    let need = a.prompts.iter().copied().max().unwrap_or(0).max(a.depth + a.gen + 1).max(a.o.tune.block as usize);
    a.o.capacity = ctx.unwrap_or(need.next_multiple_of(256));
    Ok(a)
}

/// `n` ids of a linear congruential generator, below `limit`.
fn prompt_ids(n: usize, limit: u32) -> Vec<u32> {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((x >> 33) % limit as u64) as u32
        })
        .collect()
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = if v.len() > 1 { v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64 } else { 0.0 };
    (m, var.sqrt())
}

enum Engine {
    Qwen(QwenEngine),
    Gemma(GemmaEngine),
}

impl Engine {
    fn reset(&mut self) -> HResult<()> {
        match self {
            Engine::Qwen(e) => e.reset(),
            Engine::Gemma(e) => e.reset(),
        }
    }
    fn prefill(&mut self, ids: &[u32]) -> HResult<u32> {
        match self {
            Engine::Qwen(e) => e.prefill(ids),
            Engine::Gemma(e) => e.prefill(ids),
        }
    }
    fn generate(&mut self, first: u32, steps: usize) -> HResult<usize> {
        let mut f = |argmax, _: &[u16]| Ok(Some(argmax));
        match self {
            Engine::Qwen(e) => e.generate(first, steps, &mut f),
            Engine::Gemma(e) => e.generate(first, steps, &mut f),
        }
    }
    /// GPU seconds of every command buffer submitted so far.
    fn gpu_seconds(&self) -> f64 {
        match self {
            Engine::Qwen(e) => e.gpu_seconds + e.prefill_gpu_seconds,
            Engine::Gemma(e) => e.gpu_seconds,
        }
    }
}

/// `--cool`: wait for macOS's thermal pressure level (`notifyutil -g com.apple.system.thermalpressurelevel`) to come
/// back to 0, 5 s at a time, 300 s at most; the wait printed.
fn cool_down() {
    let t = Instant::now();
    loop {
        let out = std::process::Command::new("notifyutil").args(["-g", "com.apple.system.thermalpressurelevel"]).output();
        let level = out.ok().and_then(|o| String::from_utf8(o.stdout).ok()).and_then(|s| s.split_whitespace().last().and_then(|v| v.parse::<u32>().ok()));
        match level {
            Some(0) | None => break,
            Some(l) if t.elapsed().as_secs_f64() > 300.0 => {
                eprintln!("[cool] thermal pressure still {l} after 300 s");
                break;
            }
            Some(_) => std::thread::sleep(std::time::Duration::from_secs(5)),
        }
    }
    eprintln!("[cool] waited {:.0} s after the load", t.elapsed().as_secs_f64());
}

/// `model_type` of `config.json`, at the top or under `text_config`.
fn model_type(dir: &std::path::Path) -> HResult<String> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let t = v.get("text_config").and_then(|t| t.get("model_type")).or_else(|| v.get("model_type"));
    Ok(t.and_then(|t| t.as_str()).unwrap_or_default().to_string())
}

fn run(a: Args) -> HResult<()> {
    let dir = herbert_metal::model::resolve_model_dir(&a.model_dir)?;
    let (mut e, limit) = if model_type(&dir)?.starts_with("qwen3_5") {
        let e = QwenEngine::load(&dir, &a.o)?;
        eprintln!("[engine] loaded in {:.1} s", e.load_seconds);
        // Below the first special token of Qwen3.5's vocabulary (`<|endoftext|>`, 248044).
        let limit = (e.config().vocab_size as u32).min(248_044);
        (Engine::Qwen(e), limit)
    } else {
        // The prompt by blocks of 16 rows, herbert-metal-chat's default.
        let o = GemmaOptions { capacity: a.o.capacity, matvec_rows: a.o.matvec_rows, dump_msl: a.o.dump_msl.clone(), q4: a.o.q4, logits_f32: false, prefill_block: 16, sliding_ring: 0 };
        let e = GemmaEngine::load(&dir, &o)?;
        eprintln!("[engine] loaded in {:.1} s", e.load_seconds);
        let limit = e.config().vocab_size as u32;
        (Engine::Gemma(e), limit)
    };
    if a.cool {
        cool_down();
    }
    let engine = if herbert_metal::core::bf16::F16 {
        "herbert-metal-f16"
    } else if herbert_metal::msl::bf16_native_from_env() {
        "herbert-metal-bf16+native"
    } else {
        "herbert-metal-bf16"
    };
    let engine = if a.o.q4 { format!("{engine} Q4_0") } else { engine.to_string() };
    let name = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut rows = Vec::new();
    for &p in &a.prompts {
        let ids = prompt_ids(p, limit);
        let mut runs = Vec::new(); // (seconds since the warm-up ended, tok/s wall, tok/s gpu)
        let mut i = 0;
        let mut since = None::<Instant>;
        while if a.sustain > 0.0 { since.is_none_or(|t| t.elapsed().as_secs_f64() < a.sustain) } else { i < a.warmup + a.reps } {
            e.reset()?;
            let g0 = e.gpu_seconds();
            let t = Instant::now();
            let argmax = e.prefill(&ids)?;
            let s = t.elapsed().as_secs_f64();
            let g = e.gpu_seconds() - g0;
            let warm = i < a.warmup;
            eprintln!(
                "[pp{p}] {} {:.3} s, {:.1} tok/s, gpu {:.3} s, {:.1} tok/s, argmax {argmax}",
                if warm { "warm-up" } else { "run" },
                s,
                p as f64 / s,
                g,
                p as f64 / g
            );
            if !warm {
                let at = since.get_or_insert_with(Instant::now).elapsed().as_secs_f64() - s;
                runs.push((at, p as f64 / s, p as f64 / g));
            }
            i += 1;
        }
        push_rows(&mut rows, format!("pp{p}"), &runs, a.sustain);
    }
    if a.gen > 0 {
        let first = prompt_ids(1, limit)[0];
        let ctx_ids = prompt_ids(a.depth, limit);
        let mut runs = Vec::new();
        let mut i = 0;
        let mut since = None::<Instant>;
        while if a.sustain > 0.0 { since.is_none_or(|t| t.elapsed().as_secs_f64() < a.sustain) } else { i < a.warmup + a.reps } {
            e.reset()?;
            if a.depth > 0 {
                e.prefill(&ctx_ids)?;
            }
            let g0 = e.gpu_seconds();
            let t = Instant::now();
            let fed = e.generate(first, a.gen)?;
            let s = t.elapsed().as_secs_f64();
            let g = e.gpu_seconds() - g0;
            let warm = i < a.warmup;
            eprintln!(
                "[tg{}] {} {fed} tokens {:.3} s, {:.1} tok/s, gpu {:.3} s, {:.1} tok/s",
                a.gen,
                if warm { "warm-up" } else { "run" },
                s,
                fed as f64 / s,
                g,
                fed as f64 / g
            );
            if !warm {
                let at = since.get_or_insert_with(Instant::now).elapsed().as_secs_f64() - s;
                runs.push((at, fed as f64 / s, fed as f64 / g));
            }
            i += 1;
        }
        push_rows(&mut rows, if a.depth > 0 { format!("tg{} @ d{}", a.gen, a.depth) } else { format!("tg{}", a.gen) }, &runs, a.sustain);
    }
    if let Engine::Qwen(e) = &mut e {
        if std::env::var_os("HERBERT_METAL_PFPROBE").is_some() {
            e.probe_prefetch()?;
        }
        if std::env::var("HERBERT_METAL_PROFILE").as_deref() == Ok("2") {
            for (name, s, n, gbs) in e.profile_removal()? {
                eprintln!("[profile2] {:8.3} ms ×{n:<3} {name}{}", s * 1e3, if gbs > 0.0 { format!("  {gbs:.0} GB/s") } else { String::new() });
            }
        }
        if std::env::var("HERBERT_METAL_PROFILE").as_deref() == Ok("3") {
            // A prefill block of the first prompt's ids from an empty context, dispatch by dispatch.
            e.reset()?;
            let ids = prompt_ids(a.prompts.first().copied().unwrap_or(512), limit);
            let (per, whole) = e.profile_block(&ids)?;
            let sum: f64 = per.iter().map(|p| p.1).sum();
            eprintln!("[profile3] prefill block of {} rows: {:.3} ms in one command buffer, {:.3} ms dispatch by dispatch", ids.len(), whole * 1e3, sum * 1e3);
            for (name, s, n) in &per {
                eprintln!("[profile3] {:8.3} ms {:5.1} % ×{n:<3} {name}", s * 1e3, 100.0 * s / sum);
            }
        }
        if std::env::var_os("HERBERT_METAL_PROFILE").is_some() {
            // The decode step at the end of a tgN, dispatch by dispatch.
            let (per, whole) = e.profile_step()?;
            let sum: f64 = per.iter().map(|p| p.1).sum();
            eprintln!("[profile] decode step at position {}: {:.3} ms in one command buffer, {:.3} ms dispatch by dispatch", e.pos(), whole * 1e3, sum * 1e3);
            for (name, s, n) in &per {
                eprintln!("[profile] {:8.3} ms {:5.1} % ×{n:<3} {name}", s * 1e3, 100.0 * s / sum);
            }
        }
    }
    println!("| model | engine | test | t/s |");
    println!("| --- | --- | ---: | ---: |");
    for (test, (m, sd), _) in &rows {
        println!("| {name} | {engine} | {test} | {m:.2} ± {sd:.2} |");
    }
    println!();
    println!("| model | engine | test | gpu t/s |");
    println!("| --- | --- | ---: | ---: |");
    for (test, _, (m, sd)) in &rows {
        println!("| {name} | {engine} | {test} | {m:.2} ± {sd:.2} |");
    }
    Ok(())
}

/// The table rows of one test: with `--sustain`, the first run apart (`first`)
/// and the runs that started in the second half of the span (`steady`);
/// otherwise the mean of the runs.
fn push_rows(rows: &mut Vec<(String, (f64, f64), (f64, f64))>, test: String, runs: &[(f64, f64, f64)], sustain: f64) {
    if sustain > 0.0 && !runs.is_empty() {
        let first: Vec<_> = runs.iter().take(1).collect();
        let steady: Vec<_> = runs.iter().filter(|r| r.0 >= sustain / 2.0).collect();
        let steady = if steady.is_empty() { runs.iter().rev().take(1).collect() } else { steady };
        let wall = |v: &[&(f64, f64, f64)]| v.iter().map(|r| r.1).collect::<Vec<_>>();
        let gpu = |v: &[&(f64, f64, f64)]| v.iter().map(|r| r.2).collect::<Vec<_>>();
        rows.push((format!("{test} first"), mean_sd(&wall(&first)), mean_sd(&gpu(&first))));
        rows.push((format!("{test} steady {}/{}", steady.len(), runs.len()), mean_sd(&wall(&steady)), mean_sd(&gpu(&steady))));
    } else {
        let wall: Vec<f64> = runs.iter().map(|r| r.1).collect();
        let gpu: Vec<f64> = runs.iter().map(|r| r.2).collect();
        rows.push((test, mean_sd(&wall), mean_sd(&gpu)));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\nusage: herbert-metal-bench MODEL_DIR [-p N[,N…]] [-n N] [-d N] [-r R] [-w W] [--sustain S] [--cool] [--ctx N] [--rows N] [--q4] [--tune K=V,...] [--dump-msl DIR]");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(a) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
