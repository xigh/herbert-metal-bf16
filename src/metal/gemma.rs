//! Gemma-4 (E2B, E4B, 12B) on the GPU: the checkpoint uploaded, one command
//! buffer encoded per token.
//!
//! The step is the plain, unfused sequence of the Hugging Face module — each
//! operation one dispatch — so that it can be read against `modeling_gemma4`
//! line by line and compared with the CPU oracle token by token; the prompt
//! goes through it token by token too. It is herbert-vk-bf16's first Gemma
//! step (`--tune prefill.block=1 sdpa.split=0 fuse.norm=0 fuse.swiglu=0`),
//! every dimension read from `config.json`. What it has that Qwen3.5 has not:
//! - the embedding scaled by `bf16(√hidden)`;
//! - four norms per layer, applied to the sub-layer outputs, plain `w·x̂`;
//! - per-head q/k norms with a weight, a v norm without one, RoPE per kind of
//!   layer (θ 1e4 on the whole head; θ 1e6 "proportional" on a quarter of the
//!   512-wide global heads), attention scale 1, a 512-position window on the
//!   sliding layers;
//! - the last `num_kv_shared_layers` layers read the K/V of the last earlier
//!   layer of their kind (E2B: also a double-wide MLP there);
//! - the per-layer input branch: a second embedding table (looked up on the
//!   host: its row is copied from the mapping, the table stays on disk), a
//!   projection of the token's embedding, and per layer a gate/projection pair
//!   after the MLP;
//! - GeLU(tanh), a `layer_scalar` per layer, the logits soft-capped at 30.
//!
//! 12B (`gemma4_unified_text`) is the dense subset: no per-layer input branch
//! (the layer scalar then follows the MLP's residual add), no shared K/V,
//! 1 K/V head of 512 on the full-attention layers against 8 of 256 on the
//! sliding ones, a 1024-position window, and `attention_k_eq_v`: the
//! full-attention layers have no `v_proj`, their value is the raw output of
//! `k_proj` through the weightless v norm (`[q | k]` is projected, the v norm
//! reads the k part before `k_norm` and RoPE).
//!
//! 26B-A4B (`enable_moe_block`) adds, in every layer, an expert block next to
//! the dense MLP: the router reads the residual (weightless norm, `scale`,
//! `hidden^-½`, a bf16 projection to 128 scores, softmax, top 8), the 8
//! experts' GeGLU run on `pre_feedforward_layernorm_2` of the residual, their
//! outputs are added up in bf16 by increasing expert; the MLP's output and the
//! experts' each go through their own post norm, are added, and then through
//! `post_feedforward_layernorm` into the residual (`moe_kernels`).

use super::gemma_kernels as gk;
use super::gpu::{Buffer, Cmd, Gpu, Pipeline, Range};
use super::kernels as kn;
use super::moe_kernels as mk;
use super::stats::Stats;
use crate::core::bf16::{bf16_bytes_to_h16, f32_to_h16, h16_to_f32, Bf16};
use crate::core::error::{HerbertError, Result};
use crate::model::gemma::{GemmaCheckpoint, GemmaConfig, RopeKind};
use crate::msl::Emitted;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// What the engine is asked at load.
#[derive(Clone, Debug)]
pub struct Options {
    /// Positions reserved for the conversation.
    pub capacity: usize,
    /// Rows per simdgroup of the projections.
    pub matvec_rows: u32,
    /// `--dump-msl DIR`: every emitted kernel written there.
    pub dump_msl: Option<PathBuf>,
    /// The projections and the head in Q4_0, quantized at load (else bf16).
    pub q4: bool,
    /// The head writes f32 logits, soft-capped in f32, and no argmax: what
    /// the perplexity reads ([`GemmaEngine::score`]).
    pub logits_f32: bool,
    /// Tokens of a prefill block (`--prefill-block`): the projections of the
    /// block's rows run as one Q4 product (`matmul_q4`), each block of weights
    /// read once for all of them; 0 or 1 feeds the prompt token by token.
    pub prefill_block: usize,
    /// `--sliding-ring N`: the K/V caches of the sliding layers hold the last
    /// `N` positions only (a ring, `N ≥ window + prefill block`), the full
    /// layers keep the whole capacity; 0 (or `N ≥ capacity`) allocates every
    /// cache over the capacity. Going back to a position whose window has been
    /// overwritten restarts from 0 ([`GemmaEngine::truncate`]).
    pub sliding_ring: usize,
}

/// A projection's weights: bf16 `[n][k]`, or Q4_0 with its scales at `scales`.
#[derive(Clone, Copy, Debug)]
struct Wt {
    buf: Buffer,
    scales: Option<u64>,
}

/// A layer's expert block (26B-A4B).
#[derive(Clone, Copy)]
struct MoeLayer {
    /// `router.scale` `[hidden]`, `router.per_expert_scale` `[experts]` (bf16).
    router_scale: Buffer,
    per_expert_scale: Buffer,
    /// `router.proj` `[experts][hidden]`, kept in bf16.
    router: Wt,
    pre_ff_ln_2: Buffer,
    post_ff_ln_1: Buffer,
    post_ff_ln_2: Buffer,
    /// `experts.gate_up_proj` `[experts][2·inter][hidden]` and
    /// `experts.down_proj` `[experts][hidden][inter]`, each one Q4_0 stack.
    gate_up: Wt,
    down: Wt,
}

/// A layer's per-layer input branch (E2B, E4B): gate, projection, norm.
#[derive(Clone, Copy)]
struct PleLayer {
    gate: Wt,
    proj: Wt,
    post_ln: Buffer,
}

struct GLayer {
    input_ln: Buffer,
    post_attn_ln: Buffer,
    pre_ff_ln: Buffer,
    post_ff_ln: Buffer,
    layer_scalar: Buffer,
    /// `[q | k | v]` of a layer that produces K/V (`[q | k]` when the value is
    /// the key), `q` alone of one that shares.
    qkv: Wt,
    q_norm: Buffer,
    k_norm: Option<Buffer>,
    o: Wt,
    gate_up: Wt,
    down: Wt,
    /// None without the per-layer input branch (12B).
    ple: Option<PleLayer>,
    /// None for a dense model.
    moe: Option<MoeLayer>,
}

/// The buffers one token works in.
#[derive(Clone)]
struct Rows {
    state: Range,
    /// The token's embedding row, copied from the mapping (the Q4 engine).
    emb: Range,
    x: Range,
    xn: Range,
    qkv: Range,
    q: Range,
    attn: Range,
    o_out: Range,
    gu: Range,
    hb: Range,
    pg: Range,
    ph: Range,
    ple_raw: Range,
    ple_proj: Range,
    ple_in: Range,
    /// The expert block: the MLP's output after its post norm, the router's
    /// input, its scores, the route (`2·top_k` u32), the experts' `[g | u]`,
    /// GeGLU and outputs (`top_k` rows each).
    h1: Range,
    xr: Range,
    rscores: Range,
    route: Range,
    egu: Range,
    ehb: Range,
    ey: Range,
}

/// How many rows an operation runs over, and how far apart (in elements)
/// the rows of each buffer are: one row and no strides is the decode step.
#[derive(Clone, Copy, Debug)]
struct Lay {
    m: u32,
    state: u32,
    x: u32,
    qkv: u32,
    q: u32,
    scores: u32,
    gu: u32,
    hb: u32,
    ple: u32,
    ple_in: u32,
}

impl Lay {
    const ONE: Lay = Lay { m: 1, state: 0, x: 0, qkv: 0, q: 0, scores: 0, gu: 0, hb: 0, ple: 0, ple_in: 0 };
}

fn whole(b: Buffer) -> Range {
    Range::whole(b)
}

fn sub(r: Range, elems: usize, at: usize) -> Range {
    assert!(((at + elems) * 2) as u64 <= r.size, "sub-range past the end");
    Range { buf: r.buf, offset: r.offset + (at * 2) as u64, size: (elems * 2) as u64 }
}

/// A Q4_0 projection as uploaded, by its cache key (its parts' names joined by `+`).
#[derive(Clone)]
struct Quantized {
    key: String,
    w: Wt,
    n: usize,
    k: usize,
}

/// The network: weights, caches, the pipelines built so far, and the methods
/// that encode its operations.
struct Net {
    gpu: Gpu,
    list: Vec<Pipeline>,
    by_key: HashMap<String, usize>,
    dump: Option<PathBuf>,
    c: GemmaConfig,
    cap: usize,
    sg: u32,
    eps: f32,
    rows: u32,
    layers: Vec<GLayer>,
    caches: Vec<Option<(Buffer, Buffer)>>,
    /// The bf16 embedding table (the bf16 engine), or none: the Q4 engine
    /// reads the token's row from the mapping into `emb_row`.
    embed: Option<Buffer>,
    emb_row: Buffer,
    /// The tied head: the embedding table, bf16 or Q4_0.
    lm_head: Wt,
    final_norm: Buffer,
    /// The projection of the embedding to the per-layer inputs and its norm;
    /// none without the per-layer input branch.
    ple_model: Option<(Wt, Buffer)>,
    scores: Buffer,
    arg_partial: Buffer,
    arg_out: Buffer,
    image: Buffer,
    sliding_cos: Buffer,
    sliding_sin: Buffer,
    full_cos: Buffer,
    full_sin: Buffer,
    /// Seconds spent compiling kernels.
    compile_seconds: f64,
    /// `HERBERT_METAL_SDPA_FUSED=1`: the attention in one dispatch per head
    /// (the first kernel, kept as the witness of the three-dispatch one).
    sdpa_fused: bool,
    /// `1/Σ` of every head's softmax, per row of a block (f32).
    sinv: Buffer,
    /// The experts' lists of a prefill block (`moe_kernels::gather`), u32.
    glist: Buffer,
    /// `HERBERT_METAL_MOE_ROWS=1`: a block's experts row by row (the witness of the grouped product).
    moe_rows: bool,
    /// The highest position + 1 a row of the step being encoded attends from.
    attn_max: usize,
    /// Positions of the sliding layers' K/V rings (0: no ring).
    ring: u32,
    logits_f32: bool,
    /// `HERBERT_METAL_PROFILE=1`: every dispatch in a command buffer of its
    /// own, its GPU time summed by kernel; the table is printed when the engine
    /// is dropped. Slow (one submission per dispatch): for measuring only.
    prof: Option<HashMap<String, (f64, usize)>>,
    /// `HERBERT_METAL_STATS=FILE`: the range of every activation (outside the compiling pass at load).
    range_stats: Option<Stats>,
}

impl Net {
    fn get(&mut self, key: String, make: impl FnOnce() -> Emitted) -> Result<usize> {
        if let Some(&i) = self.by_key.get(&key) {
            return Ok(i);
        }
        let t = Instant::now();
        let e = make();
        crate::msl::dump(self.dump.as_deref(), &e.name, &e.source);
        self.list.push(self.gpu.pipeline(&e)?);
        self.by_key.insert(key, self.list.len() - 1);
        self.compile_seconds += t.elapsed().as_secs_f64();
        Ok(self.list.len() - 1)
    }

    fn run(&mut self, cmd: &mut Cmd, key: String, make: impl FnOnce() -> Emitted, ranges: &[Range], groups: u32) -> Result<()> {
        self.run_2d(cmd, key, make, ranges, groups, 1)
    }

    /// [`Self::run`] over `groups × height` threadgroups (`tgid.y` over `height`).
    fn run_2d(&mut self, cmd: &mut Cmd, key: String, make: impl FnOnce() -> Emitted, ranges: &[Range], groups: u32, height: u32) -> Result<()> {
        let i = self.get(key, make)?;
        if let Some(st) = self.range_stats.as_mut().filter(|_| !cmd.discarded) {
            st.one_2d(&self.gpu, "step", &self.list[i], ranges, groups, height)?;
            return Ok(());
        }
        if self.prof.is_some() && !cmd.discarded {
            let mut one = self.gpu.begin()?;
            self.gpu.dispatch_2d(&mut one, &self.list[i], ranges, groups, height);
            let d = self.gpu.submit_wait(one)?;
            let e = self.prof.as_mut().expect("profile").entry(self.list[i].name.clone()).or_default();
            e.0 += d.gpu;
            e.1 += 1;
            return Ok(());
        }
        self.gpu.dispatch_2d(cmd, &self.list[i], ranges, groups, height);
        Ok(())
    }

    /// `y = W·x`, bf16 out; the Q4 kernel when the weights are Q4_0.
    fn mv(&mut self, cmd: &mut Cmd, w: Wt, x: Range, y: Range, k: usize, n: usize) -> Result<()> {
        self.mv_out(cmd, w, x, y, k, n, false)
    }

    /// [`Self::mv`], with `y` in f32 when `f32_out`.
    #[allow(clippy::too_many_arguments)]
    fn mv_out(&mut self, cmd: &mut Cmd, w: Wt, x: Range, y: Range, k: usize, n: usize, f32_out: bool) -> Result<()> {
        let (sg, rows) = (self.sg, self.rows);
        let groups = kn::matvec_groups(n as u32, sg, rows);
        match w.scales {
            None => self.run(cmd, format!("mv_{k}x{n}_{f32_out}"), || kn::matvec(k as u32, n as u32, if f32_out { kn::Out::F32 } else { kn::Out::Bf16 }, sg, rows), &[whole(w.buf), x, y], groups),
            Some(so) => {
                let (data, scales) = (Range::part(w.buf, 0, so), Range::part(w.buf, so, w.buf.len - so));
                self.run(cmd, format!("mvq4_{k}x{n}_{f32_out}"), || kn::matvec_q4(k as u32, n as u32, if f32_out { kn::Out::F32 } else { kn::Out::Bf16 }, sg, rows), &[data, x, y, scales], groups)
            }
        }
    }

    fn norm(&mut self, cmd: &mut Cmd, x: Range, w: Buffer, y: Range) -> Result<()> {
        self.norm_rows(cmd, x, w, y, Lay::ONE)
    }

    /// RMSNorm of the `lay.m` rows of `x` (rows `lay.x` apart) into `y` (same strides).
    fn norm_rows(&mut self, cmd: &mut Cmd, x: Range, w: Buffer, y: Range, lay: Lay) -> Result<()> {
        let (h, eps, sg, s) = (self.c.hidden_size, self.eps, self.sg, lay.x);
        self.run(cmd, format!("norm_{h}_{s}"), || kn::rmsnorm(h as u32, eps, false, sg, s, s), &[x, whole(w), y], lay.m)
    }

    fn norm_add(&mut self, cmd: &mut Cmd, a: Range, w: Buffer, x: Range, scalar: Option<Buffer>) -> Result<()> {
        self.norm_add_rows(cmd, a, w, x, scalar, Lay::ONE)
    }

    /// `x += norm(a)` (then `· s`) over the `lay.m` rows, `a` and `x` rows `lay.x` apart.
    fn norm_add_rows(&mut self, cmd: &mut Cmd, a: Range, w: Buffer, x: Range, scalar: Option<Buffer>, lay: Lay) -> Result<()> {
        let (h, eps, sg, s) = (self.c.hidden_size, self.eps, self.sg, lay.x);
        let mut ranges = vec![a, whole(w), x];
        if let Some(sc) = scalar {
            ranges.push(whole(sc));
        }
        self.run(cmd, format!("rn_add_{h}_{}_{s}", scalar.is_some()), || gk::rmsnorm_add(h as u32, eps, scalar.is_some(), sg, s, s), &ranges, lay.m)
    }

    /// The embedding of the token in `state`, and its per-layer inputs.
    fn pre(&mut self, cmd: &mut Cmd, w: &Rows) -> Result<()> {
        let (h, nl, ple, sg) = (self.c.hidden_size, self.c.num_layers, self.c.ple_dim, self.sg);
        let scale = h16_to_f32(f32_to_h16((h as f32).sqrt()));
        match self.embed {
            Some(embed) => {
                let image = self.image;
                self.run(cmd, "embed".into(), || gk::embed_scaled(h as u32, scale, sg), &[w.state, whole(embed), w.x, whole(image)], (h as u32).div_ceil(256))?
            }
            None => self.run(cmd, "embed_row".into(), || gk::embed_row(h as u32, scale, sg), &[w.emb, w.x], (h as u32).div_ceil(256))?,
        }
        let Some((proj, _)) = self.ple_model else { return Ok(()) };
        self.mv(cmd, proj, w.x, w.ple_proj, h, nl * ple)?;
        self.ple_mix(cmd, w)
    }

    /// The per-layer inputs of a token from the projection of its embedding.
    fn ple_mix(&mut self, cmd: &mut Cmd, w: &Rows) -> Result<()> {
        let (h, nl, ple, sg, eps) = (self.c.hidden_size, self.c.num_layers, self.c.ple_dim, self.sg, self.eps);
        let pn = self.ple_model.expect("the per-layer input branch").1;
        self.run(cmd, "ple_mix".into(), || gk::ple_mix(nl as u32, ple as u32, h as u32, eps, sg), &[w.ple_proj, w.ple_raw, whole(pn), w.ple_in], nl as u32)
    }

    /// `Y[r] = W·X[r]` for the `m` rows of a prefill block (row `r` of `X` at
    /// `r·ldx`, of `Y` at `r·ldy`, in bf16 elements): one Q4 product, or the
    /// rows one by one through [`Self::mv`] for bf16 weights (the witness).
    #[allow(clippy::too_many_arguments)]
    fn mm(&mut self, cmd: &mut Cmd, w: Wt, x: Range, y: Range, k: usize, n: usize, m: usize, ldx: usize, ldy: usize) -> Result<()> {
        match w.scales {
            Some(so) => {
                let sg = self.sg;
                let (data, scales) = (Range::part(w.buf, 0, so), Range::part(w.buf, so, w.buf.len - so));
                self.run(
                    cmd,
                    format!("mmq4_{k}x{n}_{m}_{ldx}_{ldy}"),
                    || kn::matmul_q4(k as u32, n as u32, m as u32, ldx as u32, ldy as u32, sg),
                    &[data, x, y, scales],
                    (n as u32).div_ceil(128 / sg),
                )
            }
            None => {
                for r in 0..m {
                    self.mv(cmd, w, sub(x, k, r * ldx), sub(y, n, r * ldy), k, n)?;
                }
                Ok(())
            }
        }
    }

    /// Layer `l` up to the K/V write: norm, projections, head norms, RoPE.
    fn layer_a(&mut self, cmd: &mut Cmd, l: usize, w: &Rows) -> Result<()> {
        let (width, h) = (self.c.qkv_width_of(l), self.c.hidden_size);
        let (input_ln, qkv_w) = (self.layers[l].input_ln, self.layers[l].qkv);
        self.norm(cmd, w.x, input_ln, w.xn)?;
        self.mv(cmd, qkv_w, w.xn, sub(w.qkv, width, 0), h, width)?;
        self.heads(cmd, l, w, Lay::ONE)
    }

    /// Layer `l`'s head norms, RoPE and the K/V write, from `[q | k | v]` in
    /// `w.qkv`, for the `lay.m` rows starting at `w`.
    fn heads(&mut self, cmd: &mut Cmd, l: usize, w: &Rows, lay: Lay) -> Result<()> {
        let (c, cap, sg, eps) = (self.c.clone(), self.cap as u32, self.sg, self.eps);
        let hd = c.head_dim_of(l);
        let (qw, kw) = (c.num_heads * hd, c.kv_heads_of(l) * hd);
        let src = c.kv_source(l);
        let own_kv = src == l;
        // The value's raw projection: its own part after the key's, or the key's (`attention_k_eq_v`).
        let v_at = if c.k_eq_v_of(l) { qw } else { qw + kw };
        let (kc, vc) = self.caches[src].expect("the source layer has a cache");
        let (cos, sin) = if c.sliding[l] { (self.sliding_cos, self.sliding_sin) } else { (self.full_cos, self.full_sin) };
        let (q_norm, k_norm) = (self.layers[l].q_norm, self.layers[l].k_norm);
        let heads = c.num_heads as u32;
        let (ss, sq, sd) = (lay.state, lay.qkv, lay.q);
        let ring = if c.sliding[l] { self.ring } else { 0 };
        self.run(
            cmd,
            format!("hn_{hd}_q_{heads}_{ss}_{sq}_{sd}"),
            || gk::head_norm(heads, hd as u32, eps, true, gk::HeadOut::Rope, cap, sg, ss, sq, sd, 0),
            &[w.state, sub(w.qkv, qw, 0), whole(q_norm), whole(cos), whole(sin), w.q],
            heads * lay.m,
        )?;
        if own_kv {
            let kvh = c.kv_heads_of(l) as u32;
            self.run(
                cmd,
                format!("hn_{hd}_k_{kvh}_{ss}_{sq}_{ring}"),
                || gk::head_norm(kvh, hd as u32, eps, true, gk::HeadOut::RopeCache, cap, sg, ss, sq, 0, ring),
                &[w.state, sub(w.qkv, kw, qw), whole(k_norm.expect("k_norm")), whole(cos), whole(sin), whole(kc)],
                kvh * lay.m,
            )?;
            self.run(
                cmd,
                format!("hn_{hd}_v_{kvh}_{ss}_{sq}_{ring}"),
                || gk::head_norm(kvh, hd as u32, eps, false, gk::HeadOut::Cache, cap, sg, ss, sq, 0, ring),
                &[w.state, sub(w.qkv, kw, v_at), whole(vc)],
                kvh * lay.m,
            )?;
        }
        Ok(())
    }

    /// Layer `l` from the attention on: attention, `o_proj`, the MLP, the
    /// per-layer input branch (when the model has one) and the layer scalar.
    fn layer_b(&mut self, cmd: &mut Cmd, l: usize, w: &Rows) -> Result<()> {
        let c = self.c.clone();
        let hd = c.head_dim_of(l);
        let qw = c.num_heads * hd;
        let (h, ple, inter) = (c.hidden_size, c.ple_dim, c.inter_of(l));
        self.attend(cmd, l, w, Lay::ONE)?;
        let lw = &self.layers[l];
        let (o, post_attn, pre_ff, gate_up, down, post_ff) = (lw.o, lw.post_attn_ln, lw.pre_ff_ln, lw.gate_up, lw.down, lw.post_ff_ln);
        let (ple_w, scalar) = (lw.ple, lw.layer_scalar);
        self.mv(cmd, o, sub(w.attn, qw, 0), w.o_out, qw, h)?;
        self.norm_add(cmd, w.o_out, post_attn, w.x, None)?;
        self.norm(cmd, w.x, pre_ff, w.xn)?;
        self.mv(cmd, gate_up, w.xn, sub(w.gu, 2 * inter, 0), h, 2 * inter)?;
        self.gelu_rows(cmd, inter, true, sub(w.gu, 2 * inter, 0), None, sub(w.hb, inter, 0), Lay::ONE)?;
        self.mv(cmd, down, sub(w.hb, inter, 0), w.o_out, inter, h)?;
        let ff = match self.layers[l].moe {
            Some(m) => {
                self.experts(cmd, &m, w, 1)?;
                w.h1
            }
            None => w.o_out,
        };
        let Some(p) = ple_w else {
            // No per-layer branch: the layer scalar follows the MLP's residual add.
            return self.norm_add(cmd, ff, post_ff, w.x, Some(scalar));
        };
        self.norm_add(cmd, ff, post_ff, w.x, None)?;
        self.mv(cmd, p.gate, w.x, w.pg, h, ple)?;
        self.gelu_rows(cmd, ple, false, w.pg, Some(sub(w.ple_in, ple, l * ple)), w.ph, Lay::ONE)?;
        self.mv(cmd, p.proj, w.ph, w.o_out, ple, h)?;
        self.norm_add(cmd, w.o_out, p.post_ln, w.x, Some(scalar))
    }

    /// The expert block of the `m` rows starting at `w` (the decode step: 1),
    /// after the dense MLP left its output in `o_out`: `h1` = the MLP's output
    /// through `post_feedforward_layernorm_1`, then the route from the
    /// residual `x`, the experts on `pre_feedforward_layernorm_2(x)`, their
    /// sum through `post_feedforward_layernorm_2` added to `h1` (in `h1`).
    /// Every buffer of `w` holds the `m` rows back to back.
    fn experts(&mut self, cmd: &mut Cmd, m: &MoeLayer, w: &Rows, rows: usize) -> Result<()> {
        let c = self.c.clone();
        let moe = c.moe.expect("an expert block");
        let (h, e, kk, inter, sg, eps) = (c.hidden_size, moe.num_experts, moe.top_k, moe.inter, self.sg, self.eps);
        let one = rows == 1;
        let lay = Lay { m: rows as u32, x: if one { 0 } else { h as u32 }, ..Lay::ONE };
        let (rs, rr) = if one { (0, 0) } else { (e as u32, 2 * kk as u32) };
        // h1 = post_ff_ln_1(mlp)
        self.norm_rows(cmd, w.o_out, m.post_ff_ln_1, w.h1, lay)?;
        // The route, from the residual.
        let sx = lay.x;
        self.run(cmd, format!("rnorm_{h}_{sx}"), || mk::router_norm(h as u32, eps, sg, sx, sx), &[w.x, whole(m.router_scale), w.xr], rows as u32)?;
        for r in 0..rows {
            self.mv(cmd, m.router, sub(w.xr, h, r * h), sub(w.rscores, e, r * e), h, e)?;
        }
        self.run(cmd, format!("route_{e}_{kk}_{rs}_{rr}"), || mk::route(e as u32, kk as u32, sg, rs, rr), &[w.rscores, whole(m.per_expert_scale), w.route], rows as u32)?;
        // The experts, on pre_ff_ln_2 of the residual.
        self.norm_rows(cmd, w.x, m.pre_ff_ln_2, w.xn, lay)?;
        let (gu_n, sxr, syr) = (2 * inter, if one { 0 } else { h }, if one { 0 } else { kk * 2 * inter });
        self.expert_mv(cmd, m.gate_up, w.xn, w.egu, w.route, h, gu_n, 0, sxr, syr, rows)?;
        let glay = Lay { m: (rows * kk) as u32, gu: gu_n as u32, hb: inter as u32, ..Lay::ONE };
        self.gelu_rows(cmd, inter, true, w.egu, None, w.ehb, glay)?;
        let (sxr, syr) = (if one { 0 } else { kk * inter }, if one { 0 } else { kk * h });
        self.expert_mv(cmd, m.down, w.ehb, w.ey, w.route, inter, h, inter, sxr, syr, rows)?;
        let (sy, so) = if one { (0, 0) } else { ((kk * h) as u32, h as u32) };
        let groups = (h as u32).div_ceil(256) * rows as u32;
        self.run(cmd, format!("combine_{h}_{kk}_{sy}_{rr}_{so}"), || mk::combine(h as u32, kk as u32, sg, sy, rr, so), &[w.ey, w.route, w.o_out], groups)?;
        // h1 = bf16(h1 + post_ff_ln_2(sum))
        self.norm_add_rows(cmd, w.o_out, m.post_ff_ln_2, w.h1, None, lay)
    }

    /// The `top_k` experts of each of the `rows` rows: `Y[r][s] = W[route[r][s]]·X[r][s]`.
    /// Over a block, grouped by expert (each block of weights read once for
    /// the rows that chose it, the same bits), unless `HERBERT_METAL_MOE_ROWS`.
    #[allow(clippy::too_many_arguments)]
    fn expert_mv(&mut self, cmd: &mut Cmd, wt: Wt, x: Range, y: Range, route: Range, k: usize, n: usize, sxs: usize, sxr: usize, syr: usize, rows: usize) -> Result<()> {
        let moe = self.c.moe.expect("an expert block");
        let (kk, sg, mr) = (moe.top_k as u32, self.sg, self.rows);
        let so = wt.scales.expect("the experts are Q4_0");
        let (data, scales) = (Range::part(wt.buf, 0, so), Range::part(wt.buf, so, wt.buf.len - so));
        if rows > 1 && !self.moe_rows {
            let (e, m) = (moe.num_experts as u32, rows as u32);
            let lists = whole(self.glist);
            self.run(cmd, format!("gather_{e}_{kk}_{m}"), || mk::gather(e, kk, m, sg), &[route, lists], 1)?;
            let (sxs, sxr) = (sxs as u32, sxr as u32);
            return self.run(
                cmd,
                format!("emm_{k}x{n}_{m}_{sxs}_{sxr}"),
                || mk::matmul_q4_experts(k as u32, n as u32, e, kk, m, sg, mr, sxs, sxr),
                &[data, x, y, scales, lists],
                mk::grouped_groups(e, n as u32, sg, mr),
            );
        }
        let sr = if rows == 1 { 0 } else { 2 * kk };
        let groups = rows as u32 * kk * mk::expert_groups(n as u32, sg, mr);
        let (sxs, sxr, syr) = (sxs as u32, sxr as u32, syr as u32);
        self.run(
            cmd,
            format!("emv_{k}x{n}_{sxs}_{sxr}_{syr}_{sr}"),
            || mk::matvec_q4_experts(k as u32, n as u32, kk, sg, mr, sxs, sxr, syr, sr),
            &[data, x, y, scales, route],
            groups,
        )
    }

    /// `h = gelu(g)·u` over `n`, the fused `[g | u]` (`u` none) or apart, for
    /// the `lay.m` rows: `[g | u]` rows `lay.gu` apart and `h` rows `lay.hb`
    /// (fused, the MLP), else `g`, `h` rows `lay.ple` and `u` rows `lay.ple_in`.
    #[allow(clippy::too_many_arguments)]
    fn gelu_rows(&mut self, cmd: &mut Cmd, n: usize, fused: bool, g: Range, u: Option<Range>, h: Range, lay: Lay) -> Result<()> {
        let sg = self.sg;
        let (sgi, su, sh) = if fused { (lay.gu, 0, lay.hb) } else { (lay.ple, lay.ple_in, lay.ple) };
        let mut ranges = vec![g];
        ranges.extend(u);
        ranges.push(h);
        let groups = (n as u32).div_ceil(256) * lay.m;
        self.run(cmd, format!("gelu_{n}_{fused}_{sgi}_{su}_{sh}"), || gk::gelu_mul(n as u32, fused, sg, sgi, su, sh), &ranges, groups)
    }

    /// Layer `l`'s attention of the token in `w`: its query against the
    /// positions up to its own (the last `window` on a sliding layer).
    fn attend(&mut self, cmd: &mut Cmd, l: usize, w: &Rows, lay: Lay) -> Result<()> {
        let (c, cap, sg) = (self.c.clone(), self.cap as u32, self.sg);
        let hd = c.head_dim_of(l);
        let (kc, vc) = self.caches[c.kv_source(l)].expect("the source layer has a cache");
        let window = if c.sliding[l] { c.sliding_window as u32 } else { 0 };
        let scores = self.scores;
        let (heads, kvh) = (c.num_heads as u32, c.kv_heads_of(l) as u32);
        let (ss, sq, sc) = (lay.state, lay.q, lay.scores);
        let ring = if c.sliding[l] { self.ring } else { 0 };
        if self.sdpa_fused {
            return self.run(
                cmd,
                format!("sdpa_{hd}_{kvh}_{window}_{ss}_{sq}_{sc}"),
                || gk::sdpa(heads, kvh, hd as u32, cap, window, sg, ss, sq, sc, sq),
                &[w.state, w.q, whole(kc), whole(vc), whole(scores), w.attn],
                heads * lay.m,
            );
        }
        // The same bits in three dispatches: the scores over every (head,
        // position), the softmax, the values over every (head, dimension).
        // The widest span a row of this step attends, for the grid's height.
        let span = if window > 0 { self.attn_max.min(window as usize) } else { self.attn_max };
        let chunks = (span as u32).div_ceil(gk::SDPA_CHUNK).max(1);
        // The kernels reach row r of a block themselves (its inverses at r·heads).
        let wi = whole(self.sinv);
        self.run_2d(
            cmd,
            format!("sdpa_qk_{hd}_{kvh}_{window}_{ss}_{sq}_{sc}_{ring}"),
            || gk::sdpa_qk(heads, kvh, hd as u32, cap, window, sg, ss, sq, sc, ring),
            &[w.state, w.q, whole(kc), whole(scores)],
            heads * lay.m,
            chunks,
        )?;
        self.run(cmd, format!("sdpa_sm_{window}_{ss}_{sc}"), || gk::sdpa_softmax(heads, cap, window, sg, ss, sc), &[w.state, whole(scores), wi], heads * lay.m)?;
        let pv_h = (hd as u32).div_ceil(256);
        self.run_2d(
            cmd,
            format!("sdpa_pv_{hd}_{kvh}_{window}_{ss}_{sc}_{sq}_{ring}"),
            || gk::sdpa_pv(heads, kvh, hd as u32, cap, window, sg, ss, sc, sq, ring),
            &[w.state, whole(scores), wi, whole(vc), w.attn],
            heads * lay.m,
            pv_h,
        )
    }

    /// The `m` rows of a prefill block through every layer: the projections
    /// as Q4 products over the block, every other operation as one dispatch
    /// over the rows (each row's own position, K/V and attention range).
    fn block(&mut self, cmd: &mut Cmd, b: &Block, m: usize) -> Result<()> {
        let c = self.c.clone();
        let (h, nl, ple, sg) = (c.hidden_size, c.num_layers, c.ple_dim, self.sg);
        let rows = &b.rows[..m];
        let scale = h16_to_f32(f32_to_h16((h as f32).sqrt()));
        for w in rows {
            match self.embed {
                Some(embed) => {
                    let image = self.image;
                    self.run(cmd, "embed".into(), || gk::embed_scaled(h as u32, scale, sg), &[w.state, whole(embed), w.x, whole(image)], (h as u32).div_ceil(256))?
                }
                None => self.run(cmd, "embed_row".into(), || gk::embed_row(h as u32, scale, sg), &[w.emb, w.x], (h as u32).div_ceil(256))?,
            }
        }
        if let Some((proj, _)) = self.ple_model {
            self.mm(cmd, proj, b.x, b.ple_proj, h, nl * ple, m, h, nl * ple)?;
            for w in rows {
                self.ple_mix(cmd, w)?;
            }
        }
        let lay = Lay {
            m: m as u32,
            state: 4,
            x: h as u32,
            qkv: b.qkv_stride as u32,
            q: b.attn_stride as u32,
            scores: (c.num_heads * self.cap) as u32,
            gu: b.gu_stride as u32,
            hb: b.hb_stride as u32,
            ple: ple as u32,
            ple_in: (nl * ple) as u32,
        };
        let w0 = &rows[0];
        for l in 0..nl {
            let qw = c.num_heads * c.head_dim_of(l);
            let width = c.qkv_width_of(l);
            let inter = c.inter_of(l);
            let lw = &self.layers[l];
            let (input_ln, qkv, o, post_attn, pre_ff, gate_up, down, post_ff) = (lw.input_ln, lw.qkv, lw.o, lw.post_attn_ln, lw.pre_ff_ln, lw.gate_up, lw.down, lw.post_ff_ln);
            let (ple_w, scalar) = (lw.ple, lw.layer_scalar);
            self.norm_rows(cmd, w0.x, input_ln, w0.xn, lay)?;
            self.mm(cmd, qkv, b.xn, b.qkv, h, width, m, h, b.qkv_stride)?;
            // Every row writes its K/V before any row attends; each attends up to its own position.
            self.heads(cmd, l, w0, lay)?;
            self.attend(cmd, l, w0, lay)?;
            self.mm(cmd, o, b.attn, b.o_out, qw, h, m, b.attn_stride, h)?;
            self.norm_add_rows(cmd, w0.o_out, post_attn, w0.x, None, lay)?;
            self.norm_rows(cmd, w0.x, pre_ff, w0.xn, lay)?;
            self.mm(cmd, gate_up, b.xn, b.gu, h, 2 * inter, m, h, b.gu_stride)?;
            self.gelu_rows(cmd, inter, true, sub(w0.gu, 2 * inter, 0), None, sub(w0.hb, inter, 0), lay)?;
            self.mm(cmd, down, b.hb, b.o_out, inter, h, m, b.hb_stride, h)?;
            let ff = match self.layers[l].moe {
                Some(mo) => {
                    self.experts(cmd, &mo, w0, m)?;
                    w0.h1
                }
                None => w0.o_out,
            };
            let Some(p) = ple_w else {
                self.norm_add_rows(cmd, ff, post_ff, w0.x, Some(scalar), lay)?;
                continue;
            };
            self.norm_add_rows(cmd, ff, post_ff, w0.x, None, lay)?;
            self.mm(cmd, p.gate, b.x, b.pg, h, ple, m, h, ple)?;
            self.gelu_rows(cmd, ple, false, w0.pg, Some(sub(w0.ple_in, ple, l * ple)), w0.ph, lay)?;
            self.mm(cmd, p.proj, b.ph, b.o_out, ple, h, m, ple, h)?;
            self.norm_add_rows(cmd, w0.o_out, p.post_ln, w0.x, Some(scalar), lay)?;
        }
        Ok(())
    }

    fn head(&mut self, cmd: &mut Cmd, w: &Rows, logits: Buffer) -> Result<()> {
        let (h, vocab, sg, cap_soft) = (self.c.hidden_size, self.c.vocab_size, self.sg, self.c.softcap);
        let (final_norm, lm_head) = (self.final_norm, self.lm_head);
        self.norm(cmd, w.x, final_norm, w.xn)?;
        if self.logits_f32 {
            self.mv_out(cmd, lm_head, w.xn, whole(logits), h, vocab, true)?;
            if cap_soft > 0.0 {
                self.run(cmd, "softcap_f32".into(), || gk::softcap_f32(vocab as u32, cap_soft, sg), &[whole(logits)], (vocab as u32).div_ceil(256))?;
            }
            return Ok(());
        }
        self.mv(cmd, lm_head, w.xn, whole(logits), h, vocab)?;
        if cap_soft > 0.0 {
            self.run(cmd, "softcap".into(), || gk::softcap(vocab as u32, cap_soft, sg), &[whole(logits)], (vocab as u32).div_ceil(256))?;
        }
        let (partial, out) = (self.arg_partial, self.arg_out);
        let groups = kn::argmax_groups(vocab as u32);
        self.run(cmd, "arg1".into(), || kn::argmax_partial(vocab as u32, sg), &[whole(logits), whole(partial)], groups)?;
        self.run(cmd, "arg2".into(), || kn::argmax_final(groups, sg), &[whole(partial), whole(out)], 1)
    }
}

/// The buffers of a prefill block: one [`Rows`] per row, views of block-wide
/// buffers whose rows are `*_stride` bf16 apart (the projections read and
/// write them whole), and the host-written state, embedding and per-layer rows.
struct Block {
    rows: Vec<Rows>,
    x: Range,
    xn: Range,
    qkv: Range,
    attn: Range,
    o_out: Range,
    gu: Range,
    hb: Range,
    pg: Range,
    ph: Range,
    ple_proj: Range,
    qkv_stride: usize,
    attn_stride: usize,
    gu_stride: usize,
    hb_stride: usize,
    state: Buffer,
    emb: Buffer,
    ple_raw: Buffer,
}

pub struct GemmaEngine {
    net: Net,
    rows: Rows,
    block: Option<Block>,
    logits: Buffer,
    /// The hidden state after the embedding and after each layer (`HERBERT_METAL_DUMP`).
    dbg: Option<(Buffer, PathBuf)>,
    ck: GemmaCheckpoint,
    capacity: usize,
    pos: usize,
    argmax: u32,
    logits_cpu: Vec<Bf16>,
    /// GPU seconds of the steps run, and their count.
    pub gpu_seconds: f64,
    pub steps: usize,
    /// Where the time of the steps went.
    pub stats: super::diag::StepStats,
    /// Idle system sleep disabled while the engine lives.
    _awake: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2::runtime::NSObjectProtocol>>,
    pub load_seconds: f64,
    /// The Q4_0 projections as uploaded, what [`GemmaEngine::write_q4_cache`] writes.
    quantized: Vec<Quantized>,
    /// With a sliding ring: the position last written in each slot.
    slot_pos: Vec<u32>,
    /// A second state kept aside ([`GemmaEngine::side_enable`]), or none.
    side: Option<Side>,
}

/// A second state kept aside (`--side-slot S`): the K/V of the first `S`
/// slots of every cache (the head), its position and its ring records, in
/// host memory. A state of at most `S` positions lies wholly in the head; a
/// longer one keeps the rest (slots `S..`) in the engine's caches, whole as
/// long as the live state writes no position `≥ S`: the first such write drops
/// it ([`GemmaEngine::fence`]). The server keeps there the short requests that
/// share nothing with the conversation (a client's periodic agent), or the
/// conversation while they run.
struct Side {
    slots: usize,
    /// The heads of the caches, in the order of [`GemmaEngine::side_regions`].
    kv: Vec<u8>,
    pos: usize,
    /// `slot_pos[..min(S, ring)]` of the state kept aside.
    slot_pos: Vec<u32>,
    valid: bool,
}

/// `cos`/`sin` bf16 tables `[capacity][head_dim / 2]` of one RoPE kind: the
/// `default` type rotates every pair; the `proportional` type rotates the first
/// `partial · head_dim / 2` pairs with exponents over the whole head width and
/// leaves the others still (frequency 0).
/// A buffer of the bf16 tensors `parts`, one after the other, in the engine's
/// 16-bit format (f16 with the `f16` feature: converted in place).
fn w16(gpu: &mut Gpu, parts: &[&[u8]]) -> Result<Buffer> {
    let b = gpu.alloc_parts(parts)?;
    // SAFETY: a fresh shared buffer holding the parts; the GPU does not use it yet.
    bf16_bytes_to_h16(unsafe { std::slice::from_raw_parts_mut(b.ptr, b.len as usize) });
    Ok(b)
}

fn rope_tables(kind: RopeKind, head_dim: usize, capacity: usize, proportional: bool) -> (Vec<Bf16>, Vec<Bf16>) {
    let half = head_dim / 2;
    let angles = if proportional { (kind.partial * head_dim as f64 / 2.0) as usize } else { half };
    let inv: Vec<f32> = (0..half)
        .map(|i| if i < angles { (1.0 / kind.theta.powf(2.0 * i as f64 / head_dim as f64)) as f32 } else { 0.0 })
        .collect();
    let mut cos = Vec::with_capacity(capacity * half);
    let mut sin = Vec::with_capacity(capacity * half);
    for p in 0..capacity {
        for &f in &inv {
            let a = p as f32 * f;
            cos.push(f32_to_h16(a.cos()));
            sin.push(f32_to_h16(a.sin()));
        }
    }
    (cos, sin)
}

impl GemmaEngine {
    pub fn load(model_dir: &Path, o: &Options) -> Result<GemmaEngine> {
        let started = Instant::now();
        let mut gpu = Gpu::open()?;
        eprintln!("[metal] {}", gpu.summary());
        let ck = GemmaCheckpoint::load(model_dir)?;
        let c = ck.config.clone();
        eprintln!(
            "[loader] {} — hidden {}, {} layers ({} sharing K/V), {} heads / {} kv × {} (global {} kv × {}{}), intermediate {}, per-layer input {}, window {}, vocab {}{}",
            c.model_type,
            c.hidden_size,
            c.num_layers,
            c.num_kv_shared_layers,
            c.num_heads,
            c.num_kv_heads,
            c.head_dim,
            c.num_global_kv_heads,
            c.global_head_dim,
            if c.k_eq_v { ", v = k" } else { "" },
            c.intermediate_size,
            c.ple_dim,
            c.sliding_window,
            c.vocab_size,
            match c.moe {
                Some(m) => format!(", experts {} (top {}) × {}", m.num_experts, m.top_k, m.inter),
                None => String::new(),
            }
        );
        let (h, vocab, ple, nl) = (c.hidden_size, c.vocab_size, c.ple_dim, c.num_layers);
        let cap = o.capacity.min(c.max_position_embeddings);
        let sg = gpu.simd;

        // ── weights ──
        // A projection `[n][k]` from its parts (tensors concatenated along
        // `n`, by name): their bytes as they are, or in Q4_0 — taken from the
        // cache of quantized weights when the directory has one, else
        // quantized now. Every Q4 projection is kept by its key for the cache.
        let t_q4 = Instant::now();
        let (mut ss_w, mut ss_e) = (0f64, 0f64);
        let mut quantized: Vec<Quantized> = Vec::new();
        let mut from_cache = 0usize;
        let mut weight = |gpu: &mut Gpu, names: &[String], n: usize, k: usize| -> Result<Wt> {
            let key = names.join("+");
            if o.q4 {
                if let Some((bytes, scales)) = ck.q4(&key, n, k)? {
                    let w = Wt { buf: gpu.alloc_with(bytes)?, scales: Some(scales) };
                    quantized.push(Quantized { key, w, n, k });
                    from_cache += 1;
                    return Ok(w);
                }
            }
            let parts: Vec<&[u8]> = names.iter().map(|nm| ck.bytes_any(nm)).collect::<Result<_>>()?;
            let total: usize = parts.iter().map(|p| p.len()).sum();
            if total != 2 * n * k {
                return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {n}x{k} expected", total / 2)));
            }
            if !o.q4 {
                return Ok(Wt { buf: w16(gpu, &parts)?, scales: None });
            }
            let joined: Vec<u8> = if parts.len() == 1 { Vec::new() } else { parts.concat() };
            let src: &[u8] = if parts.len() == 1 { parts[0] } else { &joined };
            let q = super::q4::quantize(src, n, k);
            (ss_w, ss_e) = (ss_w + q.ss_w, ss_e + q.ss_e);
            let w = Wt { buf: gpu.alloc_with(&q.bytes)?, scales: Some(q.scales) };
            quantized.push(Quantized { key, w, n, k });
            Ok(w)
        };
        let nm = |parts: &[&str]| -> Vec<String> { parts.iter().map(|s| s.to_string()).collect() };
        let (embed, lm_head) = if o.q4 {
            (None, weight(&mut gpu, &nm(&["embed_tokens.weight"]), vocab, h)?)
        } else {
            let b = w16(&mut gpu, &[ck.bytes("embed_tokens.weight", vocab * h)?])?;
            (Some(b), Wt { buf: b, scales: None })
        };
        let final_norm = w16(&mut gpu, &[ck.bytes("norm.weight", h)?])?;
        let ple_model = if ple > 0 {
            let proj = weight(&mut gpu, &nm(&["per_layer_model_projection.weight"]), nl * ple, h)?;
            Some((proj, w16(&mut gpu, &[ck.bytes("per_layer_projection_norm.weight", ple)?])?))
        } else {
            None
        };
        let mut layers: Vec<GLayer> = Vec::with_capacity(nl);
        let mut t_experts = 0f64;
        for l in 0..nl {
            let p = |f: &str| format!("layers.{l}.{f}");
            let hd = c.head_dim_of(l);
            let (qw, kw) = (c.num_heads * hd, c.kv_heads_of(l) * hd);
            let inter = c.inter_of(l);
            let own_kv = c.kv_source(l) == l;
            let (qn, kn_, vn) = (p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight"));
            let qkv = if !own_kv {
                weight(&mut gpu, &[qn], qw, h)?
            } else if c.k_eq_v_of(l) {
                if ck.has_full(&format!("model.language_model.{vn}")) {
                    return Err(HerbertError::ModelLoad(format!("layer {l}: attention_k_eq_v but the checkpoint has a v_proj")));
                }
                weight(&mut gpu, &[qn, kn_], qw + kw, h)?
            } else {
                weight(&mut gpu, &[qn, kn_, vn], qw + 2 * kw, h)?
            };
            let o_w = weight(&mut gpu, &[p("self_attn.o_proj.weight")], h, qw)?;
            let down = weight(&mut gpu, &[p("mlp.down_proj.weight")], h, inter)?;
            let ple_gp = if ple > 0 {
                let gate = weight(&mut gpu, &[p("per_layer_input_gate.weight")], ple, h)?;
                Some((gate, weight(&mut gpu, &[p("per_layer_projection.weight")], h, ple)?))
            } else {
                None
            };
            let gate_up = weight(&mut gpu, &[p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")], 2 * inter, h)?;
            let mut up = |name: &str, len: usize| -> Result<Buffer> { w16(&mut gpu, &[ck.bytes(&p(name), len)?]) };
            let input_ln = up("input_layernorm.weight", h)?;
            let post_attn_ln = up("post_attention_layernorm.weight", h)?;
            let pre_ff_ln = up("pre_feedforward_layernorm.weight", h)?;
            let post_ff_ln = up("post_feedforward_layernorm.weight", h)?;
            let ple_l = match ple_gp {
                Some((gate, proj)) => Some(PleLayer { gate, proj, post_ln: up("post_per_layer_input_norm.weight", h)? }),
                None => None,
            };
            let layer_scalar = up("layer_scalar", 1)?;
            let q_norm = up("self_attn.q_norm.weight", hd)?;
            let k_norm = if own_kv { Some(up("self_attn.k_norm.weight", hd)?) } else { None };
            let moe = match c.moe {
                Some(mc) => {
                    let (e, mi) = (mc.num_experts, mc.inter);
                    let router_scale = up("router.scale", h)?;
                    let per_expert_scale = up("router.per_expert_scale", e)?;
                    let pre_ff_ln_2 = up("pre_feedforward_layernorm_2.weight", h)?;
                    let post_ff_ln_1 = up("post_feedforward_layernorm_1.weight", h)?;
                    let post_ff_ln_2 = up("post_feedforward_layernorm_2.weight", h)?;
                    // The router stays 16-bit: 128 scores whose order picks the experts.
                    let router = Wt { buf: w16(&mut gpu, &[ck.bytes(&p("router.proj.weight"), e * h)?])?, scales: None };
                    let t = Instant::now();
                    let egu = weight(&mut gpu, &[p("experts.gate_up_proj")], e * 2 * mi, h)?;
                    let edown = weight(&mut gpu, &[p("experts.down_proj")], e * h, mi)?;
                    if egu.scales.is_none() {
                        return Err(HerbertError::ModelLoad("the expert block needs the Q4 engine (no --bf16)".into()));
                    }
                    t_experts += t.elapsed().as_secs_f64();
                    Some(MoeLayer { router_scale, per_expert_scale, router, pre_ff_ln_2, post_ff_ln_1, post_ff_ln_2, gate_up: egu, down: edown })
                }
                None => None,
            };
            layers.push(GLayer { input_ln, post_attn_ln, pre_ff_ln, post_ff_ln, layer_scalar, qkv, q_norm, k_norm, o: o_w, gate_up, down, ple: ple_l, moe });
        }
        let weight_bytes = gpu.allocated;
        if from_cache > 0 {
            eprintln!("[q4] {from_cache} of {} projections read from {} in {:.1} s", quantized.len(), crate::model::gemma::Q4_CACHE, t_q4.elapsed().as_secs_f64());
        }
        if o.q4 && from_cache < quantized.len() {
            eprintln!("[q4] projections and head quantized to Q4_0 in {:.1} s, relative RMS error {:.2e}", t_q4.elapsed().as_secs_f64(), (ss_e / ss_w.max(f64::MIN_POSITIVE)).sqrt());
            if c.moe.is_some() {
                eprintln!("[q4] of which the experts: {t_experts:.1} s");
            }
        }
        // RoPE tables of the two kinds, and the K/V caches of the layers that produce them.
        let (sc, ss) = rope_tables(c.sliding_rope, c.head_dim, cap, false);
        let (fc, fs) = rope_tables(c.full_rope, c.global_head_dim, cap, true);
        let bytes = crate::core::bf16::as_bytes;
        let (sliding_cos, sliding_sin) = (gpu.alloc_with(bytes(&sc))?, gpu.alloc_with(bytes(&ss))?);
        let (full_cos, full_sin) = (gpu.alloc_with(bytes(&fc))?, gpu.alloc_with(bytes(&fs))?);
        // The sliding layers' rings: every row of a block must find the window
        // before its own position once the block's rows are written.
        let ring = if o.sliding_ring > 0 && o.sliding_ring < cap {
            if o.sliding_ring < c.sliding_window + o.prefill_block.max(1) {
                return Err(HerbertError::Backend(format!("--sliding-ring {} is below the window {} + the prefill block {}", o.sliding_ring, c.sliding_window, o.prefill_block.max(1))));
            }
            if std::env::var_os("HERBERT_METAL_SDPA_FUSED").is_some() {
                return Err(HerbertError::Backend("--sliding-ring needs the three-dispatch attention (no HERBERT_METAL_SDPA_FUSED)".into()));
            }
            o.sliding_ring
        } else {
            0
        };
        let mut caches: Vec<Option<(Buffer, Buffer)>> = Vec::with_capacity(nl);
        for l in 0..nl {
            if c.kv_source(l) == l {
                let positions = if c.sliding[l] && ring > 0 { ring } else { cap };
                let b = (c.kv_heads_of(l) * positions * c.head_dim_of(l) * 2) as u64;
                caches.push(Some((gpu.alloc(b)?, gpu.alloc(b)?)));
            } else {
                caches.push(None);
            }
        }

        // ── activations ──
        let max_hd = c.head_dim.max(c.global_head_dim);
        let max_qkv = (0..nl).map(|l| c.qkv_width_of(l)).max().unwrap_or(0);
        let max_inter = (0..nl).map(|l| c.inter_of(l)).max().unwrap_or(0);
        // The expert block's widths per row (all 0 for a dense model).
        let (ne, nk, mi) = c.moe.map_or((0, 0, 0), |m| (m.num_experts, m.top_k, m.inter));
        let moe_w = [h, h, ne, 4 * nk, nk * 2 * mi, nk * mi, nk * h];
        let mut act = |n: usize| -> Result<Range> { Ok(whole(gpu.alloc((n * 2) as u64)?)) };
        let emb_row = act(h)?.buf;
        let rows = Rows {
            state: act(8)?,
            emb: whole(emb_row),
            x: act(h)?,
            xn: act(h)?,
            qkv: act(max_qkv)?,
            q: act(c.num_heads * max_hd)?,
            attn: act(c.num_heads * max_hd)?,
            o_out: act(h)?,
            gu: act(2 * max_inter)?,
            hb: act(max_inter)?,
            pg: act(ple)?,
            ph: act(ple)?,
            ple_raw: act(nl * ple)?,
            ple_proj: act(nl * ple)?,
            ple_in: act(nl * ple)?,
            h1: act(moe_w[0])?,
            xr: act(moe_w[1])?,
            rscores: act(moe_w[2])?,
            route: act(moe_w[3])?,
            egu: act(moe_w[4])?,
            ehb: act(moe_w[5])?,
            ey: act(moe_w[6])?,
        };
        let image = act(h)?.buf;
        // The prefill block: `bm` rows of every activation, rows at fixed strides.
        let bm = o.prefill_block;
        let block = if bm > 1 {
            let qkv_stride = max_qkv;
            let attn_stride = c.num_heads * max_hd;
            let (gu_stride, hb_stride) = (2 * max_inter, max_inter);
            let x = act(bm * h)?;
            let xn = act(bm * h)?;
            let qkv = act(bm * qkv_stride)?;
            let q = act(bm * attn_stride)?;
            let attn = act(bm * attn_stride)?;
            let o_out = act(bm * h)?;
            let gu = act(bm * gu_stride)?;
            let hb = act(bm * hb_stride)?;
            let pg = act(bm * ple)?;
            let ph = act(bm * ple)?;
            let ple_proj = act(bm * nl * ple)?;
            let ple_in = act(bm * nl * ple)?;
            let ple_raw = act(bm * nl * ple)?;
            let emb = act(bm * h)?;
            let state = act(bm * 8)?;
            let moe_b: Vec<Range> = moe_w.iter().map(|&w| act(bm * w)).collect::<Result<_>>()?;
            let row = |r: Range, w: usize, i: usize| sub(r, w, i * w);
            // The expert block's views run to the end of the block: its router
            // projection reaches the rows after the first one by `sub`.
            let tail = |j: usize, i: usize| sub(moe_b[j], (bm - i) * moe_w[j], i * moe_w[j]);
            let rows = (0..bm)
                .map(|i| Rows {
                    state: Range { buf: state.buf, offset: (i * 16) as u64, size: 16 },
                    emb: row(emb, h, i),
                    x: row(x, h, i),
                    xn: row(xn, h, i),
                    qkv: row(qkv, qkv_stride, i),
                    q: row(q, attn_stride, i),
                    attn: row(attn, attn_stride, i),
                    o_out: row(o_out, h, i),
                    gu: row(gu, gu_stride, i),
                    hb: row(hb, hb_stride, i),
                    pg: row(pg, ple, i),
                    ph: row(ph, ple, i),
                    ple_raw: row(ple_raw, nl * ple, i),
                    ple_proj: row(ple_proj, nl * ple, i),
                    ple_in: row(ple_in, nl * ple, i),
                    h1: tail(0, i),
                    xr: tail(1, i),
                    rscores: tail(2, i),
                    route: tail(3, i),
                    egu: tail(4, i),
                    ehb: tail(5, i),
                    ey: tail(6, i),
                })
                .collect();
            Some(Block { rows, x, xn, qkv, attn, o_out, gu, hb, pg, ph, ple_proj, qkv_stride, attn_stride, gu_stride, hb_stride, state: state.buf, emb: emb.buf, ple_raw: ple_raw.buf })
        } else {
            None
        };
        let logits = if o.logits_f32 { act(2 * vocab)?.buf } else { act(vocab)?.buf };
        // One region of scores per row of a prefill block (the block's rows attend in one dispatch).
        let scores = gpu.alloc((c.num_heads * cap * 4 * o.prefill_block.max(1)) as u64)?;
        let sinv = gpu.alloc((c.num_heads * 4 * o.prefill_block.max(1)) as u64)?;
        let glist = gpu.alloc((4 * ne * (1 + o.prefill_block.max(1))) as u64)?;
        let arg_partial = gpu.alloc((kn::argmax_groups(vocab as u32) * 8) as u64)?;
        let arg_out = gpu.alloc(16)?;
        // `HERBERT_METAL_DUMP=path`: the hidden state after the embedding and
        // after each layer of the last step, written as f32 when the engine is dropped.
        let dbg = match std::env::var_os("HERBERT_METAL_DUMP") {
            Some(p) => Some((gpu.alloc(((nl + 1) * h * 2) as u64)?, PathBuf::from(p))),
            None => None,
        };
        eprintln!(
            "[metal] weights {:.2} GiB, everything {:.2} GiB, capacity {cap}, prefill block {}{}",
            weight_bytes as f64 / (1u64 << 30) as f64,
            gpu.allocated as f64 / (1u64 << 30) as f64,
            if bm > 1 { bm.to_string() } else { "off (token by token)".into() },
            if ring > 0 { format!(", sliding K/V ring {ring}") } else { String::new() }
        );
        let net = Net {
            gpu,
            list: Vec::new(),
            by_key: HashMap::new(),
            dump: o.dump_msl.clone(),
            c: c.clone(),
            cap,
            sg,
            eps: c.rms_norm_eps,
            rows: o.matvec_rows,
            layers,
            caches,
            embed,
            emb_row,
            lm_head,
            final_norm,
            ple_model,
            scores,
            arg_partial,
            arg_out,
            image,
            sliding_cos,
            sliding_sin,
            full_cos,
            full_sin,
            compile_seconds: 0.0,
            sdpa_fused: std::env::var_os("HERBERT_METAL_SDPA_FUSED").is_some(),
            sinv,
            glist,
            moe_rows: std::env::var_os("HERBERT_METAL_MOE_ROWS").is_some(),
            attn_max: cap,
            ring: ring as u32,
            logits_f32: o.logits_f32,
            prof: std::env::var_os("HERBERT_METAL_PROFILE").map(|_| HashMap::new()),
            range_stats: None,
        };
        let mut engine = GemmaEngine {
            net,
            rows,
            block,
            logits,
            dbg,
            ck,
            capacity: cap,
            pos: 0,
            argmax: 0,
            logits_cpu: vec![0; vocab],
            gpu_seconds: 0.0,
            steps: 0,
            stats: Default::default(),
            _awake: super::diag::stay_awake("herbert-metal: inference"),
            load_seconds: 0.0,
            quantized,
            slot_pos: vec![u32::MAX; ring],
            side: None,
        };
        if let Some(mut st) = Stats::from_env(&mut engine.net.gpu)? {
            let r = &engine.rows;
            for (b, name) in [
                (r.x, "x"),
                (r.xn, "xn"),
                (r.qkv, "qkv"),
                (r.q, "q"),
                (r.attn, "attn"),
                (r.o_out, "o_out"),
                (r.gu, "gu"),
                (r.hb, "hb"),
                (r.pg, "pg"),
                (r.ph, "ph"),
                (r.ple_raw, "ple_raw"),
                (r.ple_proj, "ple_proj"),
                (r.ple_in, "ple_in"),
            ] {
                st.name(b.buf, name);
            }
            st.name(engine.logits, "logits");
            st.name(engine.net.scores, "scores");
            for (i, c) in engine.net.caches.iter().enumerate() {
                if let Some((k, v)) = c {
                    st.name(*k, format!("k_cache.{i}"));
                    st.name(*v, format!("v_cache.{i}"));
                }
            }
            engine.net.range_stats = Some(st);
        }
        // Every kernel compiled now, by encoding a step that is not submitted.
        let mut cmd = engine.net.gpu.begin()?;
        cmd.discarded = true;
        engine.encode(&mut cmd, true)?;
        if let Some(b) = engine.block.take() {
            let m = b.rows.len();
            engine.net.block(&mut cmd, &b, m)?;
            engine.block = Some(b);
        }
        engine.net.gpu.discard(cmd);
        eprintln!("[metal] {} kernels compiled in {:.2} s", engine.net.list.len(), engine.net.compile_seconds);
        engine.load_seconds = started.elapsed().as_secs_f64();
        Ok(engine)
    }

    pub fn config(&self) -> &GemmaConfig {
        &self.net.c
    }

    /// Write the cache of quantized weights (`model::gemma::Q4_CACHE`'s
    /// layout) to `path`: every Q4_0 projection as uploaded (`q4.<key>`, its
    /// `n,k,scales` in the metadata), then every bf16 tensor of the language
    /// model that is not a part of a projection (the embedding stays: the
    /// engine reads its rows). The number of bytes written.
    pub fn write_q4_cache(&self, path: &Path) -> Result<u64> {
        use crate::model::gemma::Q4_FORMAT;
        use serde_json::{json, Map, Value};
        use std::io::Write;
        if self.quantized.is_empty() {
            return Err(HerbertError::Backend("no Q4 projection to write (--bf16?)".into()));
        }
        let parts: std::collections::HashSet<&str> = self.quantized.iter().flat_map(|q| q.key.split('+')).filter(|n| *n != "embed_tokens.weight").collect();
        let bf: Vec<String> = self.ck.names().into_iter().filter(|n| !parts.contains(n.as_str())).collect();
        let mut header = Map::new();
        let mut meta = Map::new();
        meta.insert("format".into(), Value::String(Q4_FORMAT.into()));
        let mut off = 0u64;
        let mut blobs: Vec<&[u8]> = Vec::new();
        for q in &self.quantized {
            // SAFETY: a shared buffer of `len` bytes written at load; the GPU only reads it.
            let bytes = unsafe { std::slice::from_raw_parts(q.w.buf.ptr as *const u8, q.w.buf.len as usize) };
            let name = format!("q4.{}", q.key);
            header.insert(name.clone(), json!({"dtype": "U8", "shape": [bytes.len()], "data_offsets": [off, off + bytes.len() as u64]}));
            meta.insert(name, Value::String(format!("{},{},{}", q.n, q.k, q.w.scales.expect("Q4"))));
            off += bytes.len() as u64;
            blobs.push(bytes);
        }
        for n in &bf {
            let bytes = self.ck.bytes_any(n)?;
            header.insert(format!("model.language_model.{n}"), json!({"dtype": "BF16", "shape": [bytes.len() / 2], "data_offsets": [off, off + bytes.len() as u64]}));
            off += bytes.len() as u64;
            blobs.push(bytes);
        }
        header.insert("__metadata__".into(), Value::Object(meta));
        let mut h = serde_json::to_vec(&Value::Object(header))?;
        while h.len() % 8 != 0 {
            h.push(b' ');
        }
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::with_capacity(64 << 20, f);
        w.write_all(&(h.len() as u64).to_le_bytes())?;
        w.write_all(&h)?;
        for b in blobs {
            w.write_all(b)?;
        }
        w.into_inner().map_err(|e| HerbertError::Backend(format!("{}: {e}", path.display())))?.sync_all()?;
        eprintln!("[q4cache] {} projections, {} bf16 tensors, {:.2} GiB", self.quantized.len(), bf.len(), (8 + h.len() as u64 + off) as f64 / (1u64 << 30) as f64);
        Ok(8 + h.len() as u64 + off)
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }

    /// Back to position `pos` (at most the current one): the K/V of the
    /// positions before it stay, the next token fed overwrites the rest. The
    /// server keeps the prefix a new request shares with the previous one.
    /// The position actually kept: `pos`, or 0 when a sliding ring no longer
    /// holds the window before `pos` (positions after it overwrote it).
    pub fn truncate(&mut self, pos: usize) -> Result<usize> {
        if pos > self.pos {
            return Err(HerbertError::Backend(format!("truncate to {pos} past the position {}", self.pos)));
        }
        let ring = self.slot_pos.len();
        let kept = if ring > 0 && (pos.saturating_sub(self.net.c.sliding_window)..pos).any(|t| self.slot_pos[t % ring] != t as u32) { 0 } else { pos };
        self.pos = kept;
        Ok(kept)
    }

    /// One token through every layer and the head at the current position,
    /// then back to it: the K/V of the positions before it untouched (the one
    /// written at the position is overwritten by the next token fed; a ring
    /// records it, so a later [`Self::truncate`] still sees what is lost). It
    /// keeps every buffer of the engine in use while the server idles.
    pub fn keep_warm(&mut self) -> Result<()> {
        let p = self.pos;
        if p + 1 >= self.capacity {
            return Ok(());
        }
        let r = self.step(2, true);
        self.pos = p;
        r
    }

    /// Room for a second state kept aside, `slots` positions of every cache
    /// (at most the capacity and the sliding ring): [`Self::side_save`],
    /// [`Self::side_swap`]. 0 removes it. The bytes it holds.
    pub fn side_enable(&mut self, slots: usize) -> Result<usize> {
        if slots == 0 {
            self.side = None;
            return Ok(0);
        }
        let ring = self.slot_pos.len();
        if slots > self.capacity || (ring > 0 && slots > ring) {
            return Err(HerbertError::Backend(format!("--side-slot {slots} exceeds the capacity {} or the sliding ring {ring}", self.capacity)));
        }
        let bytes = self.side_regions(slots).iter().map(|r| r.1).sum();
        self.side = Some(Side { slots, kv: vec![0; bytes], pos: 0, slot_pos: vec![u32::MAX; slots.min(ring)], valid: false });
        Ok(bytes)
    }

    /// The positions of the head a state kept aside holds (0: no side state).
    pub fn side_slots(&self) -> usize {
        self.side.as_ref().map_or(0, |s| s.slots)
    }

    /// A state is kept aside, whole.
    pub fn side_valid(&self) -> bool {
        self.side.as_ref().is_some_and(|s| s.valid)
    }

    /// The first `slots` slots of every head of every cache, in order: where
    /// they are and their bytes (the caches are `[kv heads][positions][head dim]`).
    fn side_regions(&self, slots: usize) -> Vec<(*mut u8, usize)> {
        let c = &self.net.c;
        let ring = self.slot_pos.len();
        let mut r = Vec::new();
        for (l, kv) in self.net.caches.iter().enumerate() {
            let Some((k, v)) = kv else { continue };
            let positions = if c.sliding[l] && ring > 0 { ring } else { self.capacity };
            let row = c.head_dim_of(l) * 2;
            for b in [k, v] {
                for h in 0..c.kv_heads_of(l) {
                    // SAFETY: within the cache buffer of `kv heads · positions · head dim` bf16.
                    r.push((unsafe { b.ptr.add(h * positions * row) }, slots * row));
                }
            }
        }
        r
    }

    /// The live state copied aside (its head; a longer state keeps the rest
    /// in the caches, behind the fence). What was kept aside is replaced.
    pub fn side_save(&mut self) -> Result<()> {
        let Some(mut side) = self.side.take() else { return Err(HerbertError::Backend("no side state (--side-slot)".into())) };
        let mut at = 0;
        for (p, n) in self.side_regions(side.slots) {
            // SAFETY: the GPU is idle between steps; host memory of the side state, sized for the regions.
            unsafe { std::ptr::copy_nonoverlapping(p, side.kv.as_mut_ptr().add(at), n) };
            at += n;
        }
        let n = side.slot_pos.len();
        side.slot_pos.copy_from_slice(&self.slot_pos[..n]);
        side.pos = self.pos;
        side.valid = true;
        self.side = Some(side);
        Ok(())
    }

    /// The live state and the one kept aside exchanged: the K/V of the heads,
    /// the positions and the ring records. The live state then writes past the
    /// head only at the cost of the one now aside, if that one is longer.
    pub fn side_swap(&mut self) -> Result<()> {
        let Some(mut side) = self.side.take() else { return Err(HerbertError::Backend("no side state (--side-slot)".into())) };
        if !side.valid {
            self.side = Some(side);
            return Err(HerbertError::Backend("no state kept aside".into()));
        }
        let mut at = 0;
        for (p, n) in self.side_regions(side.slots) {
            // SAFETY: as in `side_save`; the two regions do not overlap (host memory against a Metal buffer).
            unsafe { std::ptr::swap_nonoverlapping(p, side.kv.as_mut_ptr().add(at), n) };
            at += n;
        }
        let n = side.slot_pos.len();
        self.slot_pos[..n].swap_with_slice(&mut side.slot_pos);
        std::mem::swap(&mut self.pos, &mut side.pos);
        self.side = Some(side);
        Ok(())
    }

    /// The bytes of each region of the K/V of positions `0..n` (every head of
    /// every cache, the order of [`Self::prefix_write`]): what a file of
    /// [`crate::serve::prefix_cache`] records to check that it fits.
    pub fn prefix_layout(&self, n: usize) -> Vec<usize> {
        self.side_regions(n).iter().map(|r| r.1).collect()
    }

    /// The K/V of positions `0..n` of the live state written to `w`, region
    /// after region, as they are (bf16). Only when the caches hold them all:
    /// `n` at most the position, and with a sliding ring, at most the ring and
    /// none of its first `n` slots overwritten by a later position.
    pub fn prefix_write(&self, n: usize, w: &mut dyn std::io::Write) -> Result<u64> {
        let ring = self.slot_pos.len();
        if n > self.pos || (ring > 0 && (n > ring || (0..n).any(|t| self.slot_pos[t] != t as u32))) {
            return Err(HerbertError::Backend(format!("the caches no longer hold positions 0..{n} (position {}, ring {ring})", self.pos)));
        }
        let mut bytes = 0u64;
        for (p, n) in self.side_regions(n) {
            // SAFETY: the GPU is idle between steps; within the cache buffer (`side_regions`).
            w.write_all(unsafe { std::slice::from_raw_parts(p, n) })?;
            bytes += n as u64;
        }
        Ok(bytes)
    }

    /// The K/V of positions `0..n` read from `r` (what [`Self::prefix_write`]
    /// wrote, for the same tokens), the live state then at position `n`: the
    /// next token fed is position `n`. A longer state kept aside whose tail the
    /// head would cover is dropped, as by a prefill of `n` tokens.
    pub fn prefix_read(&mut self, n: usize, r: &mut dyn std::io::Read) -> Result<()> {
        let ring = self.slot_pos.len();
        if n >= self.capacity || (ring > 0 && n > ring) {
            return Err(HerbertError::Backend(format!("a prefix of {n} positions does not fit (capacity {}, ring {ring})", self.capacity)));
        }
        self.fence(n);
        // Until every region is read, the state holds nothing.
        self.pos = 0;
        for (p, len) in self.side_regions(n) {
            // SAFETY: as in `prefix_write`; the GPU is idle, nothing else reads the caches meanwhile.
            r.read_exact(unsafe { std::slice::from_raw_parts_mut(p, len) })?;
        }
        self.wrote(0, n);
        self.pos = n;
        Ok(())
    }

    /// The live state is about to write positions below `to`: past the head,
    /// the rest of a longer state kept aside would be overwritten, that state
    /// is dropped.
    fn fence(&mut self, to: usize) {
        if let Some(s) = self.side.as_mut() {
            if s.valid && s.pos > s.slots && to > s.slots {
                s.valid = false;
                eprintln!("[side] the live state reaches position {to}, past the {} slots of the head: the state of {} positions kept aside is dropped", s.slots, s.pos);
            }
        }
    }

    /// Positions `[from, to)` written into the sliding rings.
    fn wrote(&mut self, from: usize, to: usize) {
        let ring = self.slot_pos.len();
        if ring > 0 {
            for t in from..to {
                self.slot_pos[t % ring] = t as u32;
            }
        }
    }

    /// Positions reserved for the conversation.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The operations of one step: the layers, and with `head` the logits
    /// and their argmax.
    fn encode(&mut self, cmd: &mut Cmd, head: bool) -> Result<()> {
        let net = &mut self.net;
        let w = &self.rows;
        let (h, sg) = (net.c.hidden_size, net.sg);
        net.pre(cmd, w)?;
        let dbg = self.dbg.as_ref().map(|d| d.0);
        let copy_x = |net: &mut Net, cmd: &mut Cmd, slot: usize| -> Result<()> {
            if let Some(d) = dbg {
                let to = Range::part(d, (slot * h * 2) as u64, (h * 2) as u64);
                net.run(cmd, "copy_x".into(), || gk_copy(h as u32, sg), &[w.x, to], (h as u32).div_ceil(256))?;
            }
            Ok(())
        };
        copy_x(net, cmd, 0)?;
        for l in 0..net.c.num_layers {
            net.layer_a(cmd, l, w)?;
            net.layer_b(cmd, l, w)?;
            copy_x(net, cmd, l + 1)?;
        }
        if head {
            net.head(cmd, w, self.logits)?;
        }
        Ok(())
    }

    /// One token through the layers at the current position; with `head`, the
    /// logits and their argmax too.
    fn step(&mut self, token: u32, head: bool) -> Result<()> {
        if self.pos >= self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        let t0 = super::diag::now();
        let c = &self.net.c;
        let width = c.num_layers * c.ple_dim;
        // SAFETY: a shared buffer of 16 bytes; the GPU is idle.
        unsafe {
            let st = self.rows.state.buf.ptr as *mut u32;
            st.write_volatile(token);
            st.add(1).write_volatile(self.pos as u32);
            st.add(2).write_volatile(0);
            st.add(3).write_volatile(self.pos as u32 + 1);
        }
        if width > 0 {
            let row = self.ck.row("embed_tokens_per_layer.weight", token as usize, width)?;
            // SAFETY: a shared buffer of `2 · width` bytes; the GPU is idle.
            unsafe {
                std::ptr::copy_nonoverlapping(row.as_ptr(), self.rows.ple_raw.buf.ptr, row.len());
                bf16_bytes_to_h16(std::slice::from_raw_parts_mut(self.rows.ple_raw.buf.ptr, row.len()));
            }
        }
        if self.net.embed.is_none() {
            let h = self.net.c.hidden_size;
            let e = self.ck.row("embed_tokens.weight", token as usize, h)?;
            // SAFETY: a shared buffer of `2 · hidden` bytes; the GPU is idle.
            unsafe {
                std::ptr::copy_nonoverlapping(e.as_ptr(), self.net.emb_row.ptr, e.len());
                bf16_bytes_to_h16(std::slice::from_raw_parts_mut(self.net.emb_row.ptr, e.len()));
            }
        }
        self.net.attn_max = self.pos + 1;
        self.fence(self.pos + 1);
        if let Some(st) = self.net.range_stats.as_mut() {
            st.pass();
        }
        let t1 = super::diag::now();
        // The command buffer, its encoder and what Metal autoreleases while
        // building them are released at each step, not when the thread ends.
        let done = objc2::rc::autoreleasepool(|_| -> Result<super::diag::Done> {
            let mut cmd = self.net.gpu.begin()?;
            self.encode(&mut cmd, head)?;
            let t2 = super::diag::now();
            let d = self.net.gpu.submit_wait(cmd)?;
            self.stats.add(t1 - t0, t2 - t1, d);
            Ok(d)
        })?;
        self.gpu_seconds += done.gpu;
        self.steps += 1;
        self.wrote(self.pos, self.pos + 1);
        if head && !self.net.logits_f32 {
            // SAFETY: the step wrote the argmax and the logits before it completed.
            unsafe {
                self.argmax = (self.net.arg_out.ptr as *const u32).add(kn::STATE_ARGMAX as usize).read_volatile();
                std::ptr::copy_nonoverlapping(self.logits.ptr as *const u16, self.logits_cpu.as_mut_ptr(), self.logits_cpu.len());
            }
        }
        self.pos += 1;
        Ok(())
    }

    /// Append the tokens; the greedy token of the last one. With a prefill
    /// block, the tokens go by blocks, the head on the last one only.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32> {
        Ok(self.prefill_while(ids, &mut || true)?.expect("never stopped"))
    }

    /// [`Self::prefill`], `go` asked before each block (each token without
    /// blocks): when it says no, the prefill stops there, `None`. The tokens
    /// fed so far stay in the caches, the position after them (the same bits
    /// as a prefill of those tokens alone).
    pub fn prefill_while(&mut self, ids: &[u32], go: &mut dyn FnMut() -> bool) -> Result<Option<u32>> {
        let bm = self.block.as_ref().map_or(1, |b| b.rows.len());
        if bm <= 1 || ids.len() < 2 {
            for (i, &t) in ids.iter().enumerate() {
                if !go() {
                    return Ok(None);
                }
                self.step(t, i + 1 == ids.len())?;
            }
            return Ok(Some(self.argmax));
        }
        let n = ids.len();
        let mut at = 0;
        while at < n {
            if !go() {
                return Ok(None);
            }
            let m = (n - at).min(bm);
            self.step_block(&ids[at..at + m], at + m == n)?;
            at += m;
        }
        Ok(Some(self.argmax))
    }

    /// The `m` tokens of `ids` as one prefill block at the current position;
    /// with `head`, the logits of the last one.
    fn step_block(&mut self, ids: &[u32], head: bool) -> Result<()> {
        let m = ids.len();
        if self.pos + m > self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        let t0 = super::diag::now();
        let (h, width) = (self.net.c.hidden_size, self.net.c.num_layers * self.net.c.ple_dim);
        if let Some(st) = self.net.range_stats.as_mut() {
            st.pass();
        }
        let b = self.block.take().expect("a prefill block");
        let prep = (|| -> Result<()> {
            for (i, &t) in ids.iter().enumerate() {
                // SAFETY: shared buffers sized for the block's rows; the GPU is idle.
                unsafe {
                    let st = (b.state.ptr as *mut u32).add(4 * i);
                    st.write_volatile(t);
                    st.add(1).write_volatile((self.pos + i) as u32);
                    st.add(2).write_volatile(0);
                    st.add(3).write_volatile((self.pos + i) as u32 + 1);
                }
                if width > 0 {
                    let row = self.ck.row("embed_tokens_per_layer.weight", t as usize, width)?;
                    // SAFETY: as above.
                    unsafe {
                        std::ptr::copy_nonoverlapping(row.as_ptr(), b.ple_raw.ptr.add(i * width * 2), row.len());
                        bf16_bytes_to_h16(std::slice::from_raw_parts_mut(b.ple_raw.ptr.add(i * width * 2), row.len()));
                    }
                }
                if self.net.embed.is_none() {
                    let e = self.ck.row("embed_tokens.weight", t as usize, h)?;
                    // SAFETY: as above.
                    unsafe {
                        std::ptr::copy_nonoverlapping(e.as_ptr(), b.emb.ptr.add(i * h * 2), e.len());
                        bf16_bytes_to_h16(std::slice::from_raw_parts_mut(b.emb.ptr.add(i * h * 2), e.len()));
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = prep {
            self.block = Some(b);
            return Err(e);
        }
        self.net.attn_max = self.pos + m;
        self.fence(self.pos + m);
        let t1 = super::diag::now();
        let done = objc2::rc::autoreleasepool(|_| -> Result<super::diag::Done> {
            let mut cmd = self.net.gpu.begin()?;
            self.net.block(&mut cmd, &b, m)?;
            if head {
                self.net.head(&mut cmd, &b.rows[m - 1], self.logits)?;
            }
            let t2 = super::diag::now();
            let d = self.net.gpu.submit_wait(cmd)?;
            self.stats.add(t1 - t0, t2 - t1, d);
            Ok(d)
        });
        self.block = Some(b);
        let done = done?;
        self.gpu_seconds += done.gpu;
        self.steps += 1;
        self.wrote(self.pos, self.pos + m);
        if head && !self.net.logits_f32 {
            // SAFETY: the block wrote the argmax and the logits before it completed.
            unsafe {
                self.argmax = (self.net.arg_out.ptr as *const u32).add(kn::STATE_ARGMAX as usize).read_volatile();
                std::ptr::copy_nonoverlapping(self.logits.ptr as *const u16, self.logits_cpu.as_mut_ptr(), self.logits_cpu.len());
            }
        }
        self.pos += m;
        Ok(())
    }

    /// Decode from `first`: feed a token, hand the argmax and the logits to
    /// `f`, which returns the next token or `None` to stop. The number of
    /// tokens fed.
    pub fn generate(&mut self, first: u32, max_steps: usize, f: &mut dyn FnMut(u32, &[u16]) -> Result<Option<u32>>) -> Result<usize> {
        let mut token = first;
        let mut fed = 0;
        while fed < max_steps {
            self.step(token, true)?;
            fed += 1;
            match f(self.argmax, &self.logits_cpu)? {
                Some(next) => token = next,
                None => break,
            }
        }
        Ok(fed)
    }

    /// One token at the current position, and the f32 logits (soft-capped)
    /// it predicts the next token with. Needs [`Options::logits_f32`].
    pub fn score(&mut self, token: u32) -> Result<&[f32]> {
        if !self.net.logits_f32 {
            return Err(HerbertError::Backend("score needs logits_f32".into()));
        }
        self.step(token, true)?;
        // SAFETY: the step wrote `vocab` f32 into the shared buffer before it completed; the GPU is idle until the next step.
        Ok(unsafe { std::slice::from_raw_parts(self.logits.ptr as *const f32, self.net.c.vocab_size) })
    }

    /// The bf16 logits of the last position with a head.
    pub fn logits(&self) -> &[u16] {
        &self.logits_cpu
    }
}

/// `dst = src`, `n` bf16 (the debug copy of the hidden state).
fn gk_copy(n: u32, sg: u32) -> Emitted {
    use crate::msl::{done, Elem, Kernel};
    let mut k = Kernel::new(256, sg);
    let src = k.buffer(0, Elem::U16);
    let dst = k.buffer(1, Elem::U16);
    let i = k.global_id();
    let nn = k.u(n);
    let inside = k.ult(i, nn);
    k.if_(inside, |k| {
        let v = k.ld(src, i);
        k.st(dst, i, v);
    });
    done(format!("copy_{n}"), k, 2)
}

impl Drop for GemmaEngine {
    fn drop(&mut self) {
        if let Some(p) = &self.net.prof {
            let total: f64 = p.values().map(|v| v.0).sum();
            let mut r: Vec<_> = p.iter().collect();
            r.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            eprintln!("[profile] {} kernels, GPU {:.1} ms in all", r.len(), total * 1e3);
            for (n, (t, c)) in r {
                eprintln!("[profile] {:>10.2} ms {:>5.1} %  ×{:<6} {:>9.1} µs each  {n}", t * 1e3, 100.0 * t / total.max(1e-12), c, t * 1e6 / *c as f64);
            }
        }
        if let Some((b, path)) = &self.dbg {
            let n = (self.net.c.num_layers + 1) * self.net.c.hidden_size;
            // SAFETY: the buffer holds (layers + 1) · hidden bf16, written by the last step.
            let words = unsafe { std::slice::from_raw_parts(b.ptr as *const u16, n) };
            let f: Vec<f32> = words.iter().map(|&w| h16_to_f32(w)).collect();
            let _ = std::fs::write(path, crate::core::bf16::as_bytes(&f));
        }
    }
}
