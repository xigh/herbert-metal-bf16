//! herbert-metal-probe — probes of the Apple GPU, every kernel emitted by
//! `src/msl` (the f16 study's probes S2 and S3).
//!
//! usage: herbert-metal-probe PROBE [--reps N] [--warmup N] [--iters N]
//!        [--models 0.8B,2B,…] [--chunks N] [--dump-msl DIR]
//!
//!   round    every f32 bit pattern, in chunks of 2^24 (256 chunks), rounded
//!            to bf16 by the integer form and by `bfloat(x)`, both emitted by
//!            `Kernel::f_to_bf16` and `Kernel::round_bf16`; each result
//!            compared with the CPU's `f32_to_bf16`, the differences counted
//!            by class of the input (zero, subnormal, normal, Inf, NaN) with
//!            the first examples of each (`--chunks N`: the first N only)
//!   round16  the same for f16: every f32 bit pattern rounded by `half(x)`
//!            (`Kernel::f_to_bf16` and `Kernel::round_bf16` in f16 mode, the
//!            engine's rounding with the `f16` feature) against the CPU's
//!            `f32_to_f16` and its widening; NaN counted by class (the CPU
//!            keeps the payload's top bits, the GPU may not)
//!   fma      `a = a·x + y` chains in f32, f16, `half2`, `bfloat` (the
//!            expression `a * b + c`, which MSL has no `fma` for) and
//!            `bfloat` through f32 (`bfloat(fma(float…))`); 1 chain
//!            (dependent) and 8 (independent) per thread, 16 steps unrolled
//!            per loop turn; G FMA lanes per second (a `half2` FMA counts two)
//!   cvt      a loop of f32 → 16-bit roundings (integer bf16, `bfloat(x)`,
//!            `half(x)`, and the same loop without rounding) and of 16-bit →
//!            f32 widenings (bf16 by a shift, f16 by `float(half)`), in
//!            registers; G conversions per second
//!   stream   `y = round16(widen16(x)·s + t)` over 2^27 values read and written
//!            as `uint4` (256 MiB each way): integer bf16, `bfloat(x)`, f16; GB/s
//!   gemm     the engine's `gemm_mma_rows` (dense, m first) emitted in bf16
//!            and in f16 (`msl::set_f16_default`), `simdgroup_bfloat8x8`
//!            against `simdgroup_half8x8` operands, f32 accumulators, on the
//!            projections of the Qwen3.5/3.8 models of `--models` at M = 512;
//!            TFLOPS, the first outputs checked against the host in f64
//!   matvec   the engine's `matvec` (2 rows per simdgroup) in bf16 and in f16
//!            on the same projections and the head; GB/s of weights + vectors
//!   hacc     the matrix units with f16 accumulators: 4 simdgroups, each a
//!            32 × 32 tile over a staged slice of 64 K (f16 operands from
//!            threadgroup memory, 8 blocks loaded per 16 `multiply_accumulate`,
//!            the GEMM's ratio), `--iters` turns; `simdgroup_float8x8` against
//!            `simdgroup_half8x8` accumulators; TFLOPS, the tile's first value
//!   h2mv     the engine's `matvec` in f16 (f32 FMA on widened operands) against
//!            the same walk with `half2` FMA into four `half2` accumulators per
//!            row (K / 256 terms each), summed in f32 across the simdgroup; the
//!            projections and the head of `--models`; GB/s, the first outputs
//!            against the host in f64
//!   mm2d     Metal 4's `matmul2d` (MSL 4.0, tiles of 64 × 32 over 4
//!            simdgroups) half×half and bfloat×bfloat → f32, with and without
//!            `relaxed_precision`, against the engine's `gemm_mma_rows` in
//!            `half8x8` and `bfloat8x8` with an f32 output, same projections,
//!            M = 512; TFLOPS; a variant the device refuses is reported and
//!            skipped
//!   naxg     the GEMM by `matmul2d` 16 × 32 × 16 on register fragments in f16
//!            (M5): the engine's `gemm_nax` (128 × 128, m first) and
//!            `gemm_mm2d` (64 × 64) against mlx's `steel_gemm_fused_nax`
//!            emitted piece by piece (`K` and strides at run time, its tile
//!            order, its two unrolled-never loops, all three), on the
//!            `[gate | up]`, `down` and DeltaNet `in` projections of
//!            `--models` (`--all 1`: the six) at M = 512 and 2048 (`--rows
//!            M,M…`); TFLOPS, outputs against the host
//!
//! Every run is one dispatch in a command buffer of its own, timed by
//! `GPUEndTime − GPUStartTime`; the variants of a probe are interleaved
//! (`--warmup` rounds, then `--reps` rounds, the order reversed every other
//! round). One `run` line per run on stdout, then the median [min–max] per
//! variant and its ratio to the first variant of the same shape.

use herbert_metal::core::bf16::{f16_to_f32, f32_to_bf16, f32_to_f16};
use herbert_metal::core::error::Result as HResult;
use herbert_metal::metal::gpu::{Buffer, Gpu, Pipeline, Range};
use herbert_metal::metal::kernels::{self as kn, Out};
use herbert_metal::metal::prefill as pf;
use herbert_metal::msl::{self, done, Elem, Emitted, Kernel, Scalar, Var};
use std::path::PathBuf;

const CHUNK: u32 = 1 << 24;
const CLASSES: [&str; 5] = ["zero", "subnormal", "normal", "inf", "nan"];
const EXAMPLES: usize = 8;

/// `h[i] = f_to_bf16(x)`, `r[i] = bits(round_bf16(x))` for `x = bits(base + i)`.
fn round_kernel(native: bool) -> Emitted {
    let mut k = Kernel::new(256, 32);
    k.set_bf16_native(native);
    let base = k.buffer(0, Elem::U32);
    let h_out = k.buffer(1, Elem::U16);
    let r_out = k.buffer(2, Elem::U32);
    let i = k.global_id();
    let zero = k.u(0);
    let b = k.ld(base, zero);
    let bits = k.iadd(b, i);
    let x = k.bits_to_f(bits);
    let h = k.f_to_bf16(x);
    k.st(h_out, i, h);
    let r = k.round_bf16(x);
    let rb = k.f_to_bits(r);
    k.st(r_out, i, rb);
    done(if native { "probe_round_native" } else { "probe_round_int" }, k, 3)
}

fn class(x: u32) -> usize {
    let (e, m) = ((x >> 23) & 0xff, x & 0x7f_ffff);
    match (e, m) {
        (0, 0) => 0,
        (0, _) => 1,
        (255, 0) => 3,
        (255, _) => 4,
        _ => 2,
    }
}

/// Per class: inputs seen, differences, the first differences `(x, got, want)`.
#[derive(Default, Clone)]
struct Tally {
    inputs: [u64; 5],
    diffs: [u64; 5],
    examples: [Vec<(u32, u32, u32)>; 5],
}

impl Tally {
    fn merge(&mut self, o: &Tally) {
        for c in 0..5 {
            self.inputs[c] += o.inputs[c];
            self.diffs[c] += o.diffs[c];
            for &e in &o.examples[c] {
                if self.examples[c].len() < EXAMPLES {
                    self.examples[c].push(e);
                }
            }
        }
    }
}

/// The four comparisons of one slice of a chunk: integer h, integer r, native h, native r.
fn compare(base: u32, ih: &[u16], ir: &[u32], nh: &[u16], nr: &[u32]) -> [Tally; 4] {
    let mut t: [Tally; 4] = Default::default();
    for j in 0..ih.len() {
        let x = base.wrapping_add(j as u32);
        let c = class(x);
        let want_h = f32_to_bf16(f32::from_bits(x)) as u32;
        let want_r = want_h << 16;
        let got = [ih[j] as u32, ir[j], nh[j] as u32, nr[j]];
        let want = [want_h, want_r, want_h, want_r];
        for k in 0..4 {
            t[k].inputs[c] += 1;
            if got[k] != want[k] {
                t[k].diffs[c] += 1;
                if t[k].examples[c].len() < EXAMPLES {
                    t[k].examples[c].push((x, got[k], want[k]));
                }
            }
        }
    }
    t
}

fn round(chunks: u32, dump: Option<PathBuf>) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let kernels = [round_kernel(false), round_kernel(true)];
    for e in &kernels {
        msl::dump(dump.as_deref(), &e.name, &e.source);
    }
    let pipes = [gpu.pipeline(&kernels[0])?, gpu.pipeline(&kernels[1])?];
    let base = gpu.alloc(16)?;
    let h = [gpu.alloc(2 * CHUNK as u64)?, gpu.alloc(2 * CHUNK as u64)?];
    let r = [gpu.alloc(4 * CHUNK as u64)?, gpu.alloc(4 * CHUNK as u64)?];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut total: [Tally; 4] = Default::default();
    let mut times = [Vec::new(), Vec::new()];
    for c in 0..chunks {
        let b = c * CHUNK;
        // SAFETY: a 16-byte shared buffer; no command buffer is in flight.
        unsafe { *(base.ptr as *mut u32) = b };
        for f in 0..2 {
            let mut cmd = gpu.begin()?;
            gpu.dispatch(&mut cmd, &pipes[f], &[Range::whole(base), Range::whole(h[f]), Range::whole(r[f])], CHUNK / 256);
            times[f].push(gpu.submit_wait(cmd)?.gpu);
        }
        // SAFETY: the buffers hold CHUNK elements each, written by the finished command buffers.
        let (ih, nh, ir, nr) = unsafe {
            (
                std::slice::from_raw_parts(h[0].ptr as *const u16, CHUNK as usize),
                std::slice::from_raw_parts(h[1].ptr as *const u16, CHUNK as usize),
                std::slice::from_raw_parts(r[0].ptr as *const u32, CHUNK as usize),
                std::slice::from_raw_parts(r[1].ptr as *const u32, CHUNK as usize),
            )
        };
        let part = (CHUNK as usize).div_ceil(threads);
        let parts: Vec<[Tally; 4]> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = ((t * part).min(CHUNK as usize), ((t + 1) * part).min(CHUNK as usize));
                    s.spawn(move || compare(b.wrapping_add(lo as u32), &ih[lo..hi], &ir[lo..hi], &nh[lo..hi], &nr[lo..hi]))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for p in &parts {
            for k in 0..4 {
                total[k].merge(&p[k]);
            }
        }
        if c % 16 == 15 || c + 1 == chunks {
            let d: Vec<u64> = total.iter().map(|t| t.diffs.iter().sum()).collect();
            eprintln!("[chunk {:3}/{chunks}] differences so far: int h {} r {}, native h {} r {}", c + 1, d[0], d[1], d[2], d[3]);
        }
    }
    let names = ["integer f_to_bf16", "integer round_bf16", "native f_to_bf16", "native round_bf16"];
    println!("| form | class | inputs | differences |");
    println!("| --- | --- | ---: | ---: |");
    for k in 0..4 {
        for c in 0..5 {
            println!("| {} | {} | {} | {} |", names[k], CLASSES[c], total[k].inputs[c], total[k].diffs[c]);
        }
    }
    for k in 0..4 {
        for c in 0..5 {
            for &(x, got, want) in &total[k].examples[c] {
                println!("example {} {}: x {x:#010x} got {got:#010x} want {want:#010x}", names[k], CLASSES[c]);
            }
        }
    }
    for f in 0..2 {
        let mut t = times[f].clone();
        t.sort_by(f64::total_cmp);
        let med = t[t.len() / 2];
        println!(
            "gpu time per chunk, {}: median {:.1} µs, min {:.1}, max {:.1} ({} chunks)",
            kernels[f].name,
            med * 1e6,
            t[0] * 1e6,
            t[t.len() - 1] * 1e6,
            t.len()
        );
    }
    Ok(())
}

/// `h[i] = half(x)` bits, `r[i] = bits(float(half(x)))` for `x = bits(base + i)`.
fn round16_kernel() -> Emitted {
    let mut k = Kernel::new(256, 32);
    k.set_f16(true);
    let base = k.buffer(0, Elem::U32);
    let h_out = k.buffer(1, Elem::U16);
    let r_out = k.buffer(2, Elem::U32);
    let i = k.global_id();
    let zero = k.u(0);
    let b = k.ld(base, zero);
    let bits = k.iadd(b, i);
    let x = k.bits_to_f(bits);
    let h = k.f_to_bf16(x);
    k.st(h_out, i, h);
    let r = k.round_bf16(x);
    let rb = k.f_to_bits(r);
    k.st(r_out, i, rb);
    done("probe_round_f16", k, 3)
}

/// The two comparisons of one slice of a chunk for f16: h, r.
fn compare16(base: u32, h: &[u16], r: &[u32]) -> [Tally; 2] {
    let mut t: [Tally; 2] = Default::default();
    for j in 0..h.len() {
        let x = base.wrapping_add(j as u32);
        let c = class(x);
        let want_h = f32_to_f16(f32::from_bits(x));
        let want_r = f16_to_f32(want_h).to_bits();
        let got = [h[j] as u32, r[j]];
        let want = [want_h as u32, want_r];
        for k in 0..2 {
            t[k].inputs[c] += 1;
            if got[k] != want[k] {
                t[k].diffs[c] += 1;
                if t[k].examples[c].len() < EXAMPLES {
                    t[k].examples[c].push((x, got[k], want[k]));
                }
            }
        }
    }
    t
}

fn round16(chunks: u32, dump: Option<PathBuf>) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let kernel = round16_kernel();
    msl::dump(dump.as_deref(), &kernel.name, &kernel.source);
    let pipe = gpu.pipeline(&kernel)?;
    let base = gpu.alloc(16)?;
    let h = gpu.alloc(2 * CHUNK as u64)?;
    let r = gpu.alloc(4 * CHUNK as u64)?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut total: [Tally; 2] = Default::default();
    let mut times = Vec::new();
    for c in 0..chunks {
        let b = c * CHUNK;
        // SAFETY: a 16-byte shared buffer; no command buffer is in flight.
        unsafe { *(base.ptr as *mut u32) = b };
        let mut cmd = gpu.begin()?;
        gpu.dispatch(&mut cmd, &pipe, &[Range::whole(base), Range::whole(h), Range::whole(r)], CHUNK / 256);
        times.push(gpu.submit_wait(cmd)?.gpu);
        // SAFETY: the buffers hold CHUNK elements each, written by the finished command buffer.
        let (hh, rr) = unsafe { (std::slice::from_raw_parts(h.ptr as *const u16, CHUNK as usize), std::slice::from_raw_parts(r.ptr as *const u32, CHUNK as usize)) };
        let part = (CHUNK as usize).div_ceil(threads);
        let parts: Vec<[Tally; 2]> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = ((t * part).min(CHUNK as usize), ((t + 1) * part).min(CHUNK as usize));
                    s.spawn(move || compare16(b.wrapping_add(lo as u32), &hh[lo..hi], &rr[lo..hi]))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for p in &parts {
            for k in 0..2 {
                total[k].merge(&p[k]);
            }
        }
        if c % 16 == 15 || c + 1 == chunks {
            let d: Vec<u64> = total.iter().map(|t| t.diffs.iter().sum()).collect();
            eprintln!("[chunk {:3}/{chunks}] differences so far: half(x) {} float(half(x)) {}", c + 1, d[0], d[1]);
        }
    }
    let names = ["half(x)", "float(half(x))"];
    println!("| form | class | inputs | differences |");
    println!("| --- | --- | ---: | ---: |");
    for k in 0..2 {
        for c in 0..5 {
            println!("| {} | {} | {} | {} |", names[k], CLASSES[c], total[k].inputs[c], total[k].diffs[c]);
        }
    }
    for k in 0..2 {
        for c in 0..5 {
            for &(x, got, want) in &total[k].examples[c] {
                println!("example {} {}: x {x:#010x} got {got:#010x} want {want:#010x}", names[k], CLASSES[c]);
            }
        }
    }
    times.sort_by(f64::total_cmp);
    println!("gpu time per chunk, {}: median {:.1} µs, min {:.1}, max {:.1} ({} chunks)", kernel.name, times[times.len() / 2] * 1e6, times[0] * 1e6, times[times.len() - 1] * 1e6, times.len());
    Ok(())
}

// ── the measured probes ───────────────────────────────────────────────────

struct Opts {
    reps: usize,
    warmup: usize,
    iters: Option<u32>,
    models: Vec<String>,
    chunks: u32,
    dump: Option<PathBuf>,
    /// `naxg`: the row counts M (default 512 and 2048).
    rows: Vec<u32>,
    /// `naxg`: all six projections of the model (default: the first three).
    all: bool,
}

/// One variant of a probe: its pipeline and bindings, the work of one run.
struct Case {
    shape: String,
    variant: String,
    pipe: Pipeline,
    ranges: Vec<Range>,
    groups: u32,
    /// Units of work per run (FMA lanes, conversions, bytes, flops).
    work: f64,
    /// The rate printed: work per second divided by `scale`, in `unit`.
    scale: f64,
    unit: &'static str,
}

/// `e` renamed `name`, its entry point with it.
fn renamed(mut e: Emitted, name: String) -> Emitted {
    let (old, new) = (msl::entry_name(&e.name), msl::entry_name(&name));
    e.source = e.source.replace(&format!("kernel void {old}("), &format!("kernel void {new}("));
    e.name = name;
    e
}

fn compile(gpu: &Gpu, e: &Emitted, o: &Opts) -> HResult<Pipeline> {
    msl::dump(o.dump.as_deref(), &e.name, &e.source);
    gpu.pipeline(e)
}

fn run_once(gpu: &Gpu, c: &Case) -> HResult<f64> {
    let mut cmd = gpu.begin()?;
    gpu.dispatch(&mut cmd, &c.pipe, &c.ranges, c.groups);
    Ok(gpu.submit_wait(cmd)?.gpu)
}

fn median(v: &[f64]) -> f64 {
    let mut t = v.to_vec();
    t.sort_by(f64::total_cmp);
    if t.len() % 2 == 1 { t[t.len() / 2] } else { (t[t.len() / 2 - 1] + t[t.len() / 2]) / 2.0 }
}

/// The cases interleaved, `warmup` rounds untimed then `reps` rounds, the
/// order reversed every other round; one line per run, then the summary.
fn measure(gpu: &Gpu, probe: &str, cases: &[Case], o: &Opts) -> HResult<()> {
    for _ in 0..o.warmup {
        for c in cases {
            run_once(gpu, c)?;
        }
    }
    let mut rates = vec![Vec::new(); cases.len()];
    for r in 0..o.reps {
        let order: Vec<usize> = if r % 2 == 0 { (0..cases.len()).collect() } else { (0..cases.len()).rev().collect() };
        for i in order {
            let c = &cases[i];
            let t = run_once(gpu, c)?;
            let rate = c.work / t / c.scale;
            println!("run\t{probe}\t{}\t{}\t{r}\t{t:.9}\t{rate:.3}\t{}", c.shape, c.variant, c.unit);
            rates[i].push(rate);
        }
    }
    println!("| probe | shape | variant | median | min | max | unit | / first of the shape |");
    println!("| --- | --- | --- | ---: | ---: | ---: | --- | ---: |");
    let mut base: Option<(String, f64)> = None;
    for (c, v) in cases.iter().zip(&rates) {
        let m = median(v);
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(0.0, f64::max);
        let b = match &base {
            Some((s, b)) if *s == c.shape => *b,
            _ => {
                base = Some((c.shape.clone(), m));
                m
            }
        };
        println!("| {probe} | {} | {} | {m:.3} | {lo:.3} | {hi:.3} | {} | {:.4} |", c.shape, c.variant, c.unit, m / b);
    }
    Ok(())
}

/// A deterministic stream of 32-bit values.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}

/// Random normal 16-bit values of magnitude 2^-6 to 2^-1 in `f16` or bf16,
/// their bits and their exact values.
fn random16(n: usize, f16: bool, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut g = Lcg(seed);
    let (mut bits, mut vals) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let r = g.next();
        let sign = (r >> 31) as u16;
        let e = 9 + (r >> 24 & 7) % 6; // 2^-6 … 2^-1
        let h = if f16 {
            (sign << 15) | ((e as u16) << 10) | (r & 0x3ff) as u16
        } else {
            (sign << 15) | (((e + 112) as u16) << 7) | (r & 0x7f) as u16
        };
        bits.push(h);
        vals.push(widen16(h, f16));
    }
    (bits, vals)
}

/// A normal (or zero) 16-bit value widened on the host.
fn widen16(h: u16, f16: bool) -> f32 {
    if !f16 {
        return f32::from_bits((h as u32) << 16);
    }
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let e = (h >> 10 & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        return sign * m * 2f32.powi(-24);
    }
    sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

fn bytes16(v: &[u16]) -> &[u8] {
    herbert_metal::core::bf16::as_bytes(v)
}

// ── fma ──

const UNROLL: u32 = 16;
const FMA_GROUPS: u32 = 4096;

/// `chains` accumulators of `t` per thread, `iters` loop turns of `UNROLL`
/// steps `a = a·x + y` each; `out[gid]` their sum in f32.
fn fma_kernel(t: Scalar, fused: bool, chains: u32, iters: u32) -> Emitted {
    let mut k = Kernel::new(256, 32);
    let inp = k.buffer(0, Elem::F32);
    let out = k.buffer(1, Elem::F32);
    let (z, o) = (k.u(0), k.u(1));
    let x0 = k.ld(inp, z);
    let y0 = k.ld(inp, o);
    let x = k.to_scalar(x0, t);
    let y = k.to_scalar(y0, t);
    let gid = k.global_id();
    let gf = k.u2f(gid);
    let tiny = k.f(1e-9);
    let s = k.fmul(gf, tiny);
    let accs: Vec<Var> = (0..chains)
        .map(|j| {
            let c = k.f(j as f32 * 0.01);
            let a = k.fadd(s, c);
            let v = k.to_scalar(a, t);
            k.var_scalar(t, v)
        })
        .collect();
    let (zero, n, one) = (k.u(0), k.u(iters), k.u(1));
    k.for_(zero, n, one, |k, _| {
        for _ in 0..UNROLL {
            for &acc in &accs {
                let a = k.get(acc);
                let r = k.fma_scalar(a, x, y, fused);
                k.set(acc, r);
            }
        }
    });
    let mut sum = k.f(0.0);
    for &acc in &accs {
        let a = k.get(acc);
        let f = k.scalar_to_f(a);
        sum = k.fadd(sum, f);
    }
    k.st(out, gid, sum);
    let tn = match (t, fused) {
        (Scalar::F32, _) => "f32",
        (Scalar::F16, _) => "f16",
        (Scalar::H2, _) => "half2",
        (Scalar::Bf16, false) => "bfloat",
        (Scalar::Bf16, true) => "bfloat_via_f32",
    };
    done(format!("probe_fma_{tn}_c{chains}_i{iters}"), k, 2)
}

fn fma(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let (x, y) = (0.999f32, 1e-4f32);
    let inp = gpu.alloc_with(herbert_metal::core::bf16::as_bytes(&[x, y]))?;
    let threads = FMA_GROUPS * 256;
    let base_iters = o.iters.unwrap_or(1024);
    let mut cases = Vec::new();
    let mut outs = Vec::new();
    for chains in [8u32, 1] {
        let iters = base_iters * 8 / chains;
        for (t, fused, name) in [(Scalar::F32, false, "f32"), (Scalar::F16, false, "f16"), (Scalar::H2, false, "half2"), (Scalar::Bf16, false, "bfloat"), (Scalar::Bf16, true, "bfloat_via_f32")] {
            let e = fma_kernel(t, fused, chains, iters);
            let pipe = compile(&gpu, &e, o)?;
            let out = gpu.alloc(4 * threads as u64)?;
            let lanes = if t == Scalar::H2 { 2.0 } else { 1.0 };
            let work = threads as f64 * iters as f64 * UNROLL as f64 * chains as f64 * lanes;
            outs.push((format!("chains {chains}"), name, out, chains, iters));
            cases.push(Case { shape: format!("chains {chains}"), variant: name.into(), pipe, ranges: vec![Range::whole(inp), Range::whole(out)], groups: FMA_GROUPS, work, scale: 1e9, unit: "G FMA lanes/s" });
        }
    }
    measure(&gpu, "fma", &cases, o)?;
    // Thread 0's sum against the recurrence in f64 (f32 inputs: the 16-bit kinds round each step).
    for (shape, name, out, chains, iters) in outs {
        // SAFETY: the buffer was written by the finished command buffers.
        let got = unsafe { *(out.ptr as *const f32) };
        let mut want = 0.0f64;
        for j in 0..chains {
            let mut a = j as f64 * 0.01;
            for _ in 0..iters * UNROLL {
                a = a * x as f64 + y as f64;
            }
            want += a;
        }
        println!("check\tfma\t{shape}\t{name}\tthread 0 sum {got:.6}\tf64 {want:.6}");
    }
    Ok(())
}

// ── cvt ──

#[derive(Clone, Copy, PartialEq)]
enum Cvt {
    None,
    Bf16Int,
    Bf16Native,
    F16,
    WidenBf16,
    WidenF16,
}

fn cvt_kernel(v: Cvt, iters: u32) -> Emitted {
    let mut k = Kernel::new(256, 32);
    k.set_f16(matches!(v, Cvt::F16 | Cvt::WidenF16));
    k.set_bf16_native(v == Cvt::Bf16Native);
    let out = k.buffer(0, Elem::U32);
    let gid = k.global_id();
    let (zero, n, one) = (k.u(0), k.u(iters), k.u(1));
    if matches!(v, Cvt::WidenBf16 | Cvt::WidenF16) {
        // Normal 16-bit patterns in [1, 2): `w`'s low bits under the exponent of 1.
        let (mask, onebits) = if v == Cvt::WidenF16 { (k.u(0x3ff), k.u(0x3c00)) } else { (k.u(0x7f), k.u(0x3f80)) };
        let w = k.var_u32(gid);
        let zf = k.f(0.0);
        let acc = k.var_f32(zf);
        let (mul, add) = (k.u(0x9e37_79b9), k.u(0x7f4a_7c15));
        k.for_(zero, n, one, |k, _| {
            for _ in 0..UNROLL {
                let wv = k.get(w);
                let w2 = k.imul(wv, mul);
                let w3 = k.iadd(w2, add);
                k.set(w, w3);
                let m = k.and(w3, mask);
                let h = k.or(m, onebits);
                let f = k.bf16_to_f(h);
                let a = k.get(acc);
                let s = k.fadd(a, f);
                k.set(acc, s);
            }
        });
        let a = k.get(acc);
        let bits = k.f_to_bits(a);
        k.st(out, gid, bits);
    } else {
        let gf = k.u2f(gid);
        let tiny = k.f(1e-6);
        let half = k.f(0.5);
        let x0 = k.fma(gf, tiny, half);
        let x = k.var_f32(x0);
        let acc = k.var_u32(zero);
        let (a_, b_) = (k.f(-0.9999), k.f(0.5));
        k.for_(zero, n, one, |k, _| {
            for _ in 0..UNROLL {
                let xv = k.get(x);
                let x2 = k.fma(xv, a_, b_);
                k.set(x, x2);
                let r = if v == Cvt::None { k.f_to_bits(x2) } else { k.f_to_bf16(x2) };
                let a = k.get(acc);
                let s = k.iadd(a, r);
                k.set(acc, s);
            }
        });
        let a = k.get(acc);
        k.st(out, gid, a);
    }
    let name = match v {
        Cvt::None => "none",
        Cvt::Bf16Int => "bf16_int",
        Cvt::Bf16Native => "bf16_bfloat",
        Cvt::F16 => "f16_half",
        Cvt::WidenBf16 => "widen_bf16",
        Cvt::WidenF16 => "widen_f16",
    };
    done(format!("probe_cvt_{name}_i{iters}"), k, 1)
}

fn cvt(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let iters = o.iters.unwrap_or(2048);
    let threads = FMA_GROUPS * 256;
    let mut cases = Vec::new();
    for (shape, vs) in [("narrow f32 -> 16", [Cvt::None, Cvt::Bf16Int, Cvt::Bf16Native, Cvt::F16].as_slice()), ("widen 16 -> f32", [Cvt::WidenBf16, Cvt::WidenF16].as_slice())] {
        for &v in vs {
            let e = cvt_kernel(v, iters);
            let pipe = compile(&gpu, &e, o)?;
            let out = gpu.alloc(4 * threads as u64)?;
            let variant = e.name.trim_start_matches("probe_cvt_").to_string();
            cases.push(Case { shape: shape.into(), variant, pipe, ranges: vec![Range::whole(out)], groups: FMA_GROUPS, work: threads as f64 * iters as f64 * UNROLL as f64, scale: 1e9, unit: "G conversions/s" });
        }
    }
    measure(&gpu, "cvt", &cases, o)
}

// ── stream ──

const STREAM_VEC4: u32 = 1 << 24;

fn stream_kernel(f16: bool, native: bool) -> Emitted {
    let mut k = Kernel::new(256, 32);
    k.set_f16(f16);
    k.set_bf16_native(native);
    let x = k.buffer(0, Elem::UVec4);
    let y = k.buffer(1, Elem::UVec4);
    let i = k.global_id();
    let (s, t) = (k.f(0.75), k.f(0.125));
    let sixteen = k.u(16);
    let v = k.ld(x, i);
    let mut words = [0; 4];
    for (c, w) in words.iter_mut().enumerate() {
        let xw = k.comp(v, c as u32);
        let lo = k.lo_f(xw);
        let hi = k.hi_f(xw);
        let lo2 = k.fma(lo, s, t);
        let hi2 = k.fma(hi, s, t);
        let hl = k.f_to_bf16(lo2);
        let hh = k.f_to_bf16(hi2);
        let hs = k.shl(hh, sixteen);
        *w = k.or(hl, hs);
    }
    let r = k.uvec4(words);
    k.st(y, i, r);
    let name = if f16 { "f16_half" } else if native { "bf16_bfloat" } else { "bf16_int" };
    done(format!("probe_stream_{name}"), k, 2)
}

fn stream(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let n = STREAM_VEC4 as usize * 8;
    let mut cases = Vec::new();
    for (f16, native) in [(false, false), (false, true), (true, false)] {
        let (bits, _) = random16(n, f16, 7);
        let x = gpu.alloc_with(bytes16(&bits))?;
        let y = gpu.alloc(2 * n as u64)?;
        let e = stream_kernel(f16, native);
        let pipe = compile(&gpu, &e, o)?;
        let variant = e.name.trim_start_matches("probe_stream_").to_string();
        cases.push(Case { shape: "2^27 values".into(), variant, pipe, ranges: vec![Range::whole(x), Range::whole(y)], groups: STREAM_VEC4 / 256, work: 4.0 * n as f64, scale: 1e9, unit: "GB/s" });
    }
    measure(&gpu, "stream", &cases, o)
}

// ── gemm and matvec: the projections of the models ──

/// `(name, K, N, out)` of the projections of a Qwen3.5 / 3.8 dense model.
fn projections(model: &str) -> Vec<(String, u32, u32, Out)> {
    // hidden, intermediate, attention heads, kv heads, linear v heads (k: 16 × 128, head_dim 256, v dim 128).
    let (h, inter, heads, kv, lv) = match model {
        "0.8B" => (1024, 3584, 8, 2, 16),
        "2B" => (2048, 6144, 8, 2, 16),
        "4B" => (2560, 9216, 16, 4, 32),
        "9B" => (4096, 12288, 16, 4, 32),
        "27B" => (5120, 17408, 24, 4, 48),
        _ => panic!("unknown model {model} (0.8B, 2B, 4B, 9B, 27B)"),
    };
    let (lk, hd, dv) = (16 * 128, 256, 128);
    vec![
        (format!("{model} gate|up"), h, 2 * inter, Out::Bf16),
        (format!("{model} down"), inter, h, Out::Residual),
        (format!("{model} dn in"), h, 2 * lk + 2 * lv * dv + 2 * lv, Out::Bf16),
        (format!("{model} dn out"), lv * dv, h, Out::Residual),
        (format!("{model} attn qkv"), h, 2 * heads * hd + 2 * kv * hd, Out::Bf16),
        (format!("{model} attn o"), heads * hd, h, Out::Residual),
    ]
}

const GEMM_M: u32 = 512;

fn gemm(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let mut state_words = [0u32; 16];
    state_words[kn::STATE_ROWS as usize] = GEMM_M;
    let state = gpu.alloc_with(herbert_metal::core::bf16::as_bytes(&state_words))?;
    let mut cases = Vec::new();
    for model in &o.models {
        for (name, kd, n, out) in projections(model) {
            for f16 in [false, true] {
                msl::set_f16_default(f16);
                let e = pf::gemm_mma_rows(kd, n, out, 32, true, pf::MmaRows::Dense);
                msl::set_f16_default(false);
                let kname = format!("{}_{}", e.name, if f16 { "half8x8" } else { "bfloat8x8" });
                let e = renamed(e, kname);
                let pipe = compile(&gpu, &e, o)?;
                let (wb, wv) = random16((n * kd) as usize, f16, 11 + n as u64);
                let (xb, xv) = random16((GEMM_M * kd) as usize, f16, 13 + kd as u64);
                let w = gpu.alloc_with(bytes16(&wb))?;
                let x = gpu.alloc_with(bytes16(&xb))?;
                let y = gpu.alloc(2 * (GEMM_M * n) as u64)?;
                let c = Case {
                    shape: format!("{name} K={kd} N={n}"),
                    variant: if f16 { "half8x8".into() } else { "bfloat8x8".into() },
                    pipe,
                    ranges: vec![Range::whole(state), Range::whole(w), Range::whole(x), Range::whole(y)],
                    groups: pf::gemm_simt_groups(GEMM_M, n),
                    work: 2.0 * GEMM_M as f64 * n as f64 * kd as f64,
                    scale: 1e12,
                    unit: "TFLOPS",
                };
                // One run on a zeroed Y (the residual adds to it), the first outputs against the host.
                run_once(&gpu, &c)?;
                check_rows(&y, &wv, &xv, kd, n, f16, &c.shape, &c.variant);
                cases.push(c);
            }
        }
    }
    measure(&gpu, "gemm", &cases, o)
}

/// Rows 0 and M − 1, columns 0..4 and N − 4..N of a GEMM's 16-bit output
/// against the f64 products of its 16-bit inputs.
fn check_rows(y: &Buffer, w: &[f32], x: &[f32], kd: u32, n: u32, f16: bool, shape: &str, variant: &str) {
    let (kd, n) = (kd as usize, n as usize);
    // SAFETY: Y holds M × N 16-bit values, written by the finished command buffer.
    let ys = unsafe { std::slice::from_raw_parts(y.ptr as *const u16, GEMM_M as usize * n) };
    let mut worst = 0.0f64;
    for r in [0, GEMM_M as usize - 1] {
        for c in (0..4).chain(n - 4..n) {
            let want: f64 = (0..kd).map(|i| x[r * kd + i] as f64 * w[c * kd + i] as f64).sum();
            let got = widen16(ys[r * n + c], f16) as f64;
            worst = worst.max((got - want).abs() / want.abs().max(1e-3));
        }
    }
    println!("check\tgemm\t{shape}\t{variant}\tworst relative error of 16 outputs {worst:.2e}");
}

fn matvec(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let mut cases = Vec::new();
    for model in &o.models {
        let h = projections(model)[0].1;
        let mut shapes: Vec<_> = projections(model).into_iter().map(|(name, kd, n, _)| (name, kd, n)).collect();
        shapes.push((format!("{model} head"), h, 248_320));
        for (name, kd, n) in shapes {
            for f16 in [false, true] {
                msl::set_f16_default(f16);
                let e = kn::matvec(kd, n, Out::Bf16, 32, 2);
                msl::set_f16_default(false);
                let kname = format!("{}_{}", e.name, if f16 { "f16" } else { "bf16" });
                let e = renamed(e, kname);
                let pipe = compile(&gpu, &e, o)?;
                let (wb, _) = random16((n * kd) as usize, f16, 17 + n as u64);
                let (xb, _) = random16(kd as usize, f16, 19);
                let w = gpu.alloc_with(bytes16(&wb))?;
                let x = gpu.alloc_with(bytes16(&xb))?;
                let y = gpu.alloc(2 * n as u64)?;
                cases.push(Case {
                    shape: format!("{name} K={kd} N={n}"),
                    variant: if f16 { "f16".into() } else { "bf16".into() },
                    pipe,
                    ranges: vec![Range::whole(w), Range::whole(x), Range::whole(y)],
                    groups: kn::matvec_groups(n, 32, 2),
                    work: 2.0 * (n as f64 * kd as f64 + kd as f64 + n as f64),
                    scale: 1e9,
                    unit: "GB/s",
                });
            }
        }
    }
    measure(&gpu, "matvec", &cases, o)
}

// ── hacc: f16 accumulators on the matrix units ──

const HACC_K: u32 = 64;

fn hacc_kernel(half_acc: bool, iters: u32) -> Emitted {
    msl::set_f16_default(true);
    let mut k = Kernel::new(128, 32);
    let a_in = k.buffer(0, Elem::UVec4);
    let b_in = k.buffer(1, Elem::UVec4);
    let out = k.buffer(2, Elem::F32);
    let rs = HACC_K / 8 + 1; // uint4 per staged row (32 rows of A, 32 columns of B as rows of K), one spare
    let a_sh = k.shared_vec4(32 * rs);
    let b_sh = k.shared_vec4(32 * rs);
    let c_sh = k.shared_vec4(4 * 32 * 32 / 8); // the tiles stored as f16, one per simdgroup
    let t = k.local_id();
    let g = k.flat_group();
    let sgid = k.subgroup_id();
    let rsc = k.u(rs);
    let kc8 = k.u(HACC_K / 8);
    let n_cells = k.u(32 * HACC_K / 8);
    let wgc = k.u(128);
    // Stage A (32 × K) and B (32 × K, B's columns as rows: W's layout) from the buffers.
    k.for_(t, n_cells, wgc, |k, e| {
        let r = k.udiv(e, kc8);
        let c = k.umod(e, kc8);
        let sa = k.imul(r, rsc);
        let sa = k.iadd(sa, c);
        let va = k.ld(a_in, e);
        k.sst(a_sh, sa, va);
        let vb = k.ld(b_in, e);
        k.sst(b_sh, sa, vb);
    });
    k.barrier();
    let rs8 = rs * 8;
    let accs_f: Vec<Var> = if half_acc { Vec::new() } else { (0..16).map(|_| k.mat_acc()).collect() };
    let accs_h: Vec<Var> = if half_acc { (0..16).map(|_| k.mat_acc_h()).collect() } else { Vec::new() };
    let (zero, n, one) = (k.u(0), k.u(iters), k.u(1));
    k.for_(zero, n, one, |k, _| {
        for ks in 0..HACC_K / 8 {
            let a: Vec<_> = (0..4)
                .map(|i| {
                    let o = k.u(i * 8 * rs8 + ks * 8);
                    k.mat_load_bf16(a_sh, o, rs8, false)
                })
                .collect();
            let b: Vec<_> = (0..4)
                .map(|j| {
                    let o = k.u(j * 8 * rs8 + ks * 8);
                    k.mat_load_bf16(b_sh, o, rs8, true)
                })
                .collect();
            for i in 0..4 {
                for j in 0..4 {
                    if half_acc {
                        k.mat_mad_h(accs_h[i * 4 + j], a[i], b[j]);
                    } else {
                        k.mat_mad(accs_f[i * 4 + j], a[i], b[j]);
                    }
                }
            }
        }
    });
    // The tile to threadgroup memory as f16, then element (0, 0) of each simdgroup's tile to `out`.
    let tile = k.u(32 * 32);
    let cbase = k.imul(sgid, tile);
    for i in 0..4 {
        for j in 0..4 {
            let o = k.u(i * 8 * 32 + j * 8);
            let at = k.iadd(cbase, o);
            if half_acc {
                k.mat_store_h(accs_h[(i * 4 + j) as usize], c_sh, at, 32);
            } else {
                let h = k.mat_to_bf16(accs_f[(i * 4 + j) as usize]);
                k.mat_store_bf16(h, c_sh, at, 32);
            }
        }
    }
    k.sg_barrier();
    let lane = k.lane();
    let is0 = k.ieq(lane, zero);
    k.if_(is0, |k| {
        let eight = k.u(8);
        let cell = k.udiv(cbase, eight);
        let v = k.sld(c_sh, cell);
        let w = k.comp(v, 0);
        let f = k.f16_to_f(w);
        let four = k.u(4);
        let at = k.imul(g, four);
        let at = k.iadd(at, sgid);
        k.st(out, at, f);
    });
    msl::set_f16_default(false);
    done(format!("probe_hacc_{}_i{iters}", if half_acc { "half8x8" } else { "float8x8" }), k, 3)
}

fn hacc(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let iters = o.iters.unwrap_or(256);
    let groups = 4096u32;
    let (ab, _) = random16((32 * HACC_K) as usize, true, 23);
    let (bb, _) = random16((32 * HACC_K) as usize, true, 29);
    let a = gpu.alloc_with(bytes16(&ab))?;
    let b = gpu.alloc_with(bytes16(&bb))?;
    let out = gpu.alloc(4 * 4 * groups as u64)?;
    let mut cases = Vec::new();
    for half_acc in [false, true] {
        let e = hacc_kernel(half_acc, iters);
        let pipe = compile(&gpu, &e, o)?;
        let c = Case {
            shape: format!("32x32 tile, K slice {HACC_K}, {iters} turns"),
            variant: if half_acc { "half8x8 acc".into() } else { "float8x8 acc".into() },
            pipe,
            ranges: vec![Range::whole(a), Range::whole(b), Range::whole(out)],
            groups,
            work: 2.0 * 32.0 * 32.0 * HACC_K as f64 * iters as f64 * 4.0 * groups as f64,
            scale: 1e12,
            unit: "TFLOPS",
        };
        run_once(&gpu, &c)?;
        // SAFETY: `out` holds 4 × groups f32 written by the finished command buffer.
        let v = unsafe { std::slice::from_raw_parts(out.ptr as *const f32, 4) };
        println!("check\thacc\t{}\t{}\ttile (0, 0) of the four simdgroups {:?}", c.shape, c.variant, v);
        cases.push(c);
    }
    measure(&gpu, "hacc", &cases, o)
}

// ── h2mv: the matvec with half2 FMA ──

/// The engine's matvec walk (2 rows per simdgroup, lanes over `uint4` chunks
/// of 8 f16) with `half2` FMA into four `half2` accumulators per row (one per
/// `uint` of the chunk), summed in f32 across the simdgroup at the end.
fn h2mv_kernel(kdim: u32, n: u32) -> Emitted {
    msl::set_f16_default(true);
    let (sg, rows) = (32u32, 2u32);
    let g = 8u32; // simdgroups per group, as the engine's rows_per_group(32)
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, Elem::U16);
    let group = k.flat_group();
    let sgid = k.subgroup_id();
    let gc = k.u(g);
    let base = k.imul(group, gc);
    let slot = k.iadd(base, sgid);
    let rc = k.u(rows);
    let row0 = k.imul(slot, rc);
    let nn = k.u(n);
    let inside = k.ult(row0, nn);
    k.if_(inside, |k| {
        let chunks = k.u(kdim / 8);
        let last = k.u(n - 1);
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                k.imul(row, chunks)
            })
            .collect();
        let lane = k.lane();
        let step = k.u(sg);
        let zero_f = k.f(0.0);
        let zero_h2 = k.to_scalar(zero_f, Scalar::H2);
        let accs: Vec<Vec<Var>> = (0..rows).map(|_| (0..4).map(|_| k.var_scalar(Scalar::H2, zero_h2)).collect()).collect();
        k.for_(lane, chunks, step, |k, c| {
            let xv = k.ld(x, c);
            let xs: Vec<_> = (0..4)
                .map(|i| {
                    let xx = k.comp(xv, i);
                    k.u_to_h2(xx)
                })
                .collect();
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], c);
                let wv = k.ld(w, at);
                for (i, &xh) in xs.iter().enumerate() {
                    let ww = k.comp(wv, i as u32);
                    let wh = k.u_to_h2(ww);
                    let a = k.get(accs[r][i]);
                    let a = k.fma_scalar(wh, xh, a, true);
                    k.set(accs[r][i], a);
                }
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        for r in 0..rows {
            let mut sum = zero_f;
            for i in 0..4 {
                let a = k.get(accs[r as usize][i]);
                let f = k.scalar_to_f(a);
                sum = k.fadd(sum, f);
            }
            let sum = k.sg_fadd(sum);
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(first, in_n);
            k.if_(write, |k| {
                let b = k.f_to_bf16(sum);
                k.st(y, row, b);
            });
        }
    });
    msl::set_f16_default(false);
    done(format!("probe_h2mv_{kdim}x{n}"), k, 3)
}

/// `y[0..4]` and `y[N − 4..N]` of a matvec's 16-bit output against the f64 products.
fn check_vec(y: &Buffer, w: &[f32], x: &[f32], kd: u32, n: u32, shape: &str, variant: &str) {
    let (kd, n) = (kd as usize, n as usize);
    // SAFETY: Y holds N 16-bit values, written by the finished command buffer.
    let ys = unsafe { std::slice::from_raw_parts(y.ptr as *const u16, n) };
    let mut worst = 0.0f64;
    for c in (0..4).chain(n - 4..n) {
        let want: f64 = (0..kd).map(|i| x[i] as f64 * w[c * kd + i] as f64).sum();
        let got = widen16(ys[c], true) as f64;
        worst = worst.max((got - want).abs() / want.abs().max(1e-3));
    }
    println!("check\th2mv\t{shape}\t{variant}\tworst relative error of 8 outputs {worst:.2e}");
}

fn h2mv(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let mut cases = Vec::new();
    for model in &o.models {
        let h = projections(model)[0].1;
        let mut shapes: Vec<_> = projections(model).into_iter().map(|(name, kd, n, _)| (name, kd, n)).collect();
        shapes.push((format!("{model} head"), h, 248_320));
        for (name, kd, n) in shapes {
            let (wb, wv) = random16((n * kd) as usize, true, 17 + n as u64);
            let (xb, xv) = random16(kd as usize, true, 19);
            let w = gpu.alloc_with(bytes16(&wb))?;
            let x = gpu.alloc_with(bytes16(&xb))?;
            for h2 in [false, true] {
                let e = if h2 {
                    h2mv_kernel(kd, n)
                } else {
                    msl::set_f16_default(true);
                    let e = kn::matvec(kd, n, Out::Bf16, 32, 2);
                    msl::set_f16_default(false);
                    let kname = format!("{}_f16", e.name);
                    renamed(e, kname)
                };
                let pipe = compile(&gpu, &e, o)?;
                let y = gpu.alloc(2 * n as u64)?;
                let c = Case {
                    shape: format!("{name} K={kd} N={n}"),
                    variant: if h2 { "half2 fma".into() } else { "f32 fma".into() },
                    pipe,
                    ranges: vec![Range::whole(w), Range::whole(x), Range::whole(y)],
                    groups: kn::matvec_groups(n, 32, 2),
                    work: 2.0 * (n as f64 * kd as f64 + kd as f64 + n as f64),
                    scale: 1e9,
                    unit: "GB/s",
                };
                run_once(&gpu, &c)?;
                check_vec(&y, &wv, &xv, kd, n, &c.shape, &c.variant);
                cases.push(c);
            }
        }
    }
    measure(&gpu, "h2mv", &cases, o)
}

// ── mm2d ──

const MM2D_TILE: (u32, u32) = (64, 32);

fn mm2d_kernel(bf16: bool, kd: u32, m: u32, n: u32, relaxed: bool) -> Emitted {
    let (tm, tn) = MM2D_TILE;
    let mut k = Kernel::new(128, 32);
    let x = k.buffer(0, Elem::U16);
    let w = k.buffer(1, Elem::U16);
    let y = k.buffer(2, Elem::F32);
    k.mm2d(bf16, x, w, y, kd, m, n, tm, tn, relaxed);
    done(format!("probe_mm2d_{}_{kd}x{n}_m{m}_t{tm}x{tn}{}", if bf16 { "bfloat" } else { "half" }, if relaxed { "_relaxed" } else { "" }), k, 3)
}

/// Rows 0 and M − 1, columns 0..4 and N − 4..N of an f32 output against the
/// f64 products of the 16-bit inputs.
fn check_f32(y: &Buffer, w: &[f32], x: &[f32], kd: u32, n: u32, shape: &str, variant: &str) {
    let (kd, n) = (kd as usize, n as usize);
    // SAFETY: Y holds M × N f32, written by the finished command buffer.
    let ys = unsafe { std::slice::from_raw_parts(y.ptr as *const f32, GEMM_M as usize * n) };
    let mut worst = 0.0f64;
    for r in [0, GEMM_M as usize - 1] {
        for c in (0..4).chain(n - 4..n) {
            let want: f64 = (0..kd).map(|i| x[r * kd + i] as f64 * w[c * kd + i] as f64).sum();
            worst = worst.max((ys[r * n + c] as f64 - want).abs() / want.abs().max(1e-3));
        }
    }
    println!("check\tmm2d\t{shape}\t{variant}\tworst relative error of 16 outputs {worst:.2e}");
}

fn mm2d(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let mut state_words = [0u32; 16];
    state_words[kn::STATE_ROWS as usize] = GEMM_M;
    let state = gpu.alloc_with(herbert_metal::core::bf16::as_bytes(&state_words))?;
    let mut cases = Vec::new();
    for model in &o.models {
        for (name, kd, n, _) in projections(model) {
            let shape = format!("{name} K={kd} N={n}");
            for f16 in [true, false] {
                let (wb, wv) = random16((n * kd) as usize, f16, 11 + n as u64);
                let (xb, xv) = random16((GEMM_M * kd) as usize, f16, 13 + kd as u64);
                let w = gpu.alloc_with(bytes16(&wb))?;
                let x = gpu.alloc_with(bytes16(&xb))?;
                let t = if f16 { "half" } else { "bfloat" };
                let mut variants: Vec<(String, Emitted, bool)> = Vec::new();
                for relaxed in [false, true] {
                    let v = format!("mm2d {t}{}", if relaxed { " relaxed" } else { "" });
                    variants.push((v, mm2d_kernel(!f16, kd, GEMM_M, n, relaxed), true));
                }
                msl::set_f16_default(f16);
                let e = pf::gemm_mma_rows(kd, n, Out::F32, 32, true, pf::MmaRows::Dense);
                msl::set_f16_default(false);
                let kname = format!("{}_{}", e.name, if f16 { "half8x8" } else { "bfloat8x8" });
                variants.push((format!("simdgroup {t}8x8"), renamed(e, kname), false));
                for (variant, e, is_mm2d) in variants {
                    let pipe = match compile(&gpu, &e, o) {
                        Ok(p) => p,
                        Err(err) => {
                            println!("unsupported\tmm2d\t{shape}\t{variant}\t{err}");
                            continue;
                        }
                    };
                    let y = gpu.alloc(4 * (GEMM_M * n) as u64)?;
                    let (ranges, groups) = if is_mm2d {
                        (vec![Range::whole(x), Range::whole(w), Range::whole(y)], (GEMM_M / MM2D_TILE.0) * (n / MM2D_TILE.1))
                    } else {
                        (vec![Range::whole(state), Range::whole(w), Range::whole(x), Range::whole(y)], pf::gemm_simt_groups(GEMM_M, n))
                    };
                    let c = Case { shape: shape.clone(), variant, pipe, ranges, groups, work: 2.0 * GEMM_M as f64 * n as f64 * kd as f64, scale: 1e12, unit: "TFLOPS" };
                    match run_once(&gpu, &c) {
                        Ok(_) => {
                            check_f32(&y, &wv, &xv, kd, n, &c.shape, &c.variant);
                            cases.push(c);
                        }
                        Err(err) => println!("unsupported\tmm2d\t{shape}\t{}\t{err}", c.variant),
                    }
                }
            }
        }
    }
    measure(&gpu, "mm2d", &cases, o)
}

// ── naxg ──

/// mlx's `steel_gemm_fused_nax` on a base M5 (128 × 128 tiles, 4 × 4
/// simdgroups of 32 × 32, `K` by 32), emitted with the pieces of
/// `pf::gemm_nax` and its differences as flags: `rt` the strides and `K`
/// read from the state (`K` at word 8) instead of written as constants;
/// `swz` mlx's tile order (`swizzle_log` 1: pairs of row tiles walking the
/// column tiles, two groups along x per pair); `two` its two loops (512 of
/// `K` with `threadgroup_barrier(mem_none)`, then 16 steps of 32), neither
/// unrolled, a volatile local in the body. Y in f32. Groups: `swz`
/// `2·ceil(N/128) · ceil(ceil(M/128)/2)`, otherwise `ceil(M/128)·ceil(N/128)`
/// (m first).
fn naxg_kernel(kd: u32, n: u32, rt: bool, swz: bool, two: bool) -> Emitted {
    let mut k = Kernel::new(512, 32);
    let _state = k.buffer(0, Elem::U32);
    let _w = k.buffer(1, Elem::UVec4);
    let _x = k.buffer(2, Elem::UVec4);
    let y = k.buffer(3, Elem::F32);
    k.written(y);
    let mut src = String::new();
    let mut w = |l: &str| {
        src.push_str(l);
        src.push('\n');
    };
    w(&format!("const uint m = b0[{}];", kn::STATE_ROWS));
    w(&if rt { "const uint K = b0[8];".to_string() } else { format!("const uint K = {kd}u;") });
    let tn = n.div_ceil(128);
    w("const uint mb = (m + 127u) / 128u;");
    if swz {
        w(&format!("const uint gx = tgid.x % {}u, gy = tgid.x / {}u;", 2 * tn, 2 * tn));
        w("const uint gm = gy * 2u + (gx & 1u), gn = gx >> 1;");
        w(&format!("if (gm >= mb || gn >= {tn}u) return;"));
    } else {
        w("const uint gm = tgid.x % mb, gn = tgid.x / mb;");
    }
    w("const ushort sgm = ushort(sgid) / 4, sgn = ushort(sgid) % 4;");
    w("const ushort qid = ushort(lane) >> 2;");
    w("const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);");
    w("const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;");
    w("const uint m0 = gm * 128u + sgm * 32u, n0 = gn * 128u + sgn * 32u;");
    w("typedef vec<half, 8> v8; typedef vec<half, 4> v4; typedef vec<float, 8> f8;");
    for a in 0..2 {
        for i in 0..2 {
            w(&format!("const device half *xa{a}{i} = (const device half *)b2 + min(m0 + {}u + fm, m - 1u) * K + fn;", 16 * a + 8 * i));
            w(&format!("const device half *wb{a}{i} = (const device half *)b1 + min(n0 + {}u + fm, {}u) * K + fn;", 16 * a + 8 * i, n - 1));
        }
    }
    w("f8 d00 = f8(0.0f), d01 = f8(0.0f), d10 = f8(0.0f), d11 = f8(0.0f);");
    let mut body = String::new();
    for (p, nm) in [("xa", "a"), ("wb", "b")] {
        for r in 0..2 {
            for kk in 0..2 {
                body.push_str(&format!(
                    "v8 {nm}{r}{kk}; {{ v4 x0 = *(const device v4 *)({p}{r}0 + k0 + {o}u); v4 x1 = *(const device v4 *)({p}{r}1 + k0 + {o}u); {nm}{r}{kk} = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }}\n",
                    o = 16 * kk
                ));
            }
        }
    }
    for mm in 0..2 {
        for kk in 0..2 {
            body.push_str(&k.nax_mma_text(&format!("d{mm}0"), &format!("d{mm}1"), &format!("a{mm}{kk}"), &format!("b0{kk}"), &format!("b1{kk}"), true, false));
            body.push('\n');
        }
    }
    if two {
        w("#pragma clang loop unroll(disable)");
        w("for (uint kb = 0; kb < K; kb += 512u) {");
        w("    threadgroup_barrier(mem_flags::mem_none);");
        w("    #pragma clang loop unroll(disable)");
        w("    for (uint k1 = 0; k1 < 512u; k1 += 32u) {");
        w("        volatile int cb; const uint k0 = kb + k1;");
        for l in body.lines() {
            w(&format!("        {l}"));
        }
        w("        (void)cb;");
        w("    }");
        w("}");
    } else {
        w("for (uint k0 = 0; k0 < K; k0 += 32u) {");
        for l in body.lines() {
            w(&format!("    {l}"));
        }
        w("}");
    }
    for mm in 0..2 {
        for nn in 0..2 {
            w(&format!("for (ushort i = 0; i < 2; ++i) {{ const uint r = m0 + {}u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {{ const uint c = n0 + {}u + fn + j; if (c < {n}u) b3[r * {n}u + c] = d{mm}{nn}[i * 4 + j]; }} }}", 16 * mm, 16 * nn));
        }
    }
    k.raw(&src);
    done(format!("probe_naxg_{kd}x{n}{}{}{}", if rt { "_rt" } else { "" }, if swz { "_swz" } else { "" }, if two { "_two" } else { "" }), k, 4)
}

/// Rows 0 and m − 1, columns 0..4 and N − 4..N of an f32 output of `m` rows against the host.
fn check_naxg(y: &Buffer, w: &[f32], x: &[f32], m: u32, kd: u32, n: u32, shape: &str, variant: &str) {
    let (m, kd, n) = (m as usize, kd as usize, n as usize);
    // SAFETY: Y holds m × N f32, written by the finished command buffer.
    let ys = unsafe { std::slice::from_raw_parts(y.ptr as *const f32, m * n) };
    let mut worst = 0.0f64;
    for r in [0, m - 1] {
        for c in (0..4).chain(n - 4..n) {
            let want: f64 = (0..kd).map(|i| x[r * kd + i] as f64 * w[c * kd + i] as f64).sum();
            worst = worst.max((ys[r * n + c] as f64 - want).abs() / want.abs().max(1e-3));
        }
    }
    println!("check\tnaxg\t{shape}\t{variant}\tworst relative error of 16 outputs {worst:.2e}");
}

fn naxg(o: &Opts) -> HResult<()> {
    let mut gpu = Gpu::open()?;
    eprintln!("[gpu] {}", gpu.summary());
    let mut cases = Vec::new();
    msl::set_f16_default(true);
    for model in &o.models {
        for (name, kd, n, _) in projections(model).into_iter().take(if o.all { 6 } else { 3 }) {
            let (wb, wv) = random16((n * kd) as usize, true, 11 + n as u64);
            let w = gpu.alloc_with(bytes16(&wb))?;
            for &m in &o.rows {
                let shape = format!("{name} K={kd} N={n} M={m}");
                let (xb, xv) = random16((m * kd) as usize, true, 13 + kd as u64);
                let x = gpu.alloc_with(bytes16(&xb))?;
                let mut words = [0u32; 16];
                words[kn::STATE_ROWS as usize] = m;
                words[8] = kd;
                let state = gpu.alloc_with(herbert_metal::core::bf16::as_bytes(&words))?;
                let (mb, tn) = (m.div_ceil(128), n.div_ceil(128));
                let plain = mb * tn;
                let swz = 2 * tn * mb.div_ceil(2);
                let variants: Vec<(String, Emitted, u32)> = vec![
                    ("engine mm2d 64x64".into(), pf::gemm_mm2d(kd, n, Out::F32, 32, true, 64, 64), pf::gemm_mm2d_groups(m, n, 64, 64)),
                    ("engine nax 128x128 m-first".into(), pf::gemm_nax(kd, n, Out::F32, 32, 0, 4, 4, false, false), plain),
                    ("engine nax 128x128 two, mlx order".into(), pf::gemm_nax(kd, n, Out::F32, 32, 2, 4, 4, false, true), plain),
                    ("nax K at run time".into(), naxg_kernel(kd, n, true, false, false), plain),
                    ("nax mlx order".into(), naxg_kernel(kd, n, false, true, false), swz),
                    ("nax mlx loops".into(), naxg_kernel(kd, n, false, false, true), plain),
                    ("nax mlx all".into(), naxg_kernel(kd, n, true, true, true), swz),
                ];
                for (variant, e, groups) in variants {
                    let pipe = match compile(&gpu, &e, o) {
                        Ok(p) => p,
                        Err(err) => {
                            println!("unsupported\tnaxg\t{shape}\t{variant}\t{err}");
                            continue;
                        }
                    };
                    let y = gpu.alloc(4 * (m * n) as u64)?;
                    let c = Case { shape: shape.clone(), variant, pipe, ranges: vec![Range::whole(state), Range::whole(w), Range::whole(x), Range::whole(y)], groups, work: 2.0 * m as f64 * n as f64 * kd as f64, scale: 1e12, unit: "TFLOPS" };
                    match run_once(&gpu, &c) {
                        Ok(_) => {
                            check_naxg(&y, &wv, &xv, m, kd, n, &c.shape, &c.variant);
                            cases.push(c);
                        }
                        Err(err) => println!("unsupported\tnaxg\t{shape}\t{}\t{err}", c.variant),
                    }
                }
            }
        }
    }
    msl::set_f16_default(false);
    measure(&gpu, "naxg", &cases, o)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: herbert-metal-probe round|round16|fma|cvt|stream|gemm|matvec|mm2d|hacc|h2mv|naxg [--reps N] [--warmup N] [--iters N] [--models 0.8B,2B,4B,9B,27B] [--chunks N] [--rows M,M…] [--all 1] [--dump-msl DIR]";
    let mut it = args.iter();
    let Some(cmd) = it.next() else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let mut o = Opts { reps: 5, warmup: 1, iters: None, models: vec!["0.8B".into(), "4B".into()], chunks: 256, dump: None, rows: vec![512, 2048], all: false };
    while let Some(flag) = it.next() {
        let ok = match (flag.as_str(), it.next()) {
            ("--reps", Some(v)) => v.parse().map(|n| o.reps = n).is_ok(),
            ("--warmup", Some(v)) => v.parse().map(|n| o.warmup = n).is_ok(),
            ("--iters", Some(v)) => v.parse().map(|n| o.iters = Some(n)).is_ok(),
            ("--models", Some(v)) => {
                o.models = v.split(',').map(String::from).collect();
                true
            }
            ("--chunks", Some(v)) => v.parse::<u32>().map(|n| o.chunks = n.clamp(1, 256)).is_ok(),
            ("--rows", Some(v)) => v.split(',').map(|x| x.parse::<u32>()).collect::<Result<Vec<_>, _>>().map(|r| o.rows = r).is_ok(),
            ("--all", Some(v)) => v.parse::<u32>().map(|n| o.all = n != 0).is_ok(),
            ("--dump-msl", Some(v)) => {
                o.dump = Some(PathBuf::from(v));
                true
            }
            _ => false,
        };
        if !ok {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
    let r = match cmd.as_str() {
        "round" => round(o.chunks, o.dump.clone()),
        "round16" => round16(o.chunks, o.dump.clone()),
        "fma" => fma(&o),
        "cvt" => cvt(&o),
        "stream" => stream(&o),
        "gemm" => gemm(&o),
        "matvec" => matvec(&o),
        "mm2d" => mm2d(&o),
        "naxg" => naxg(&o),
        "hacc" => hacc(&o),
        "h2mv" => h2mv(&o),
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
