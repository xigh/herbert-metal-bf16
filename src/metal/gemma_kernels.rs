//! The kernels Gemma-4 adds to the Qwen3.5 set: the embedding scaled in bf16,
//! the norm followed by the residual add, the per-head norm with RoPE and the
//! K/V cache write, the attention with a sliding window and a scale of 1, the
//! GeLU(tanh) gate, the per-layer input branch, the logit soft cap.
//!
//! Every kernel rounds to bf16 where the Hugging Face module does, so that a
//! step reproduces the reference operation by operation: a norm computes in
//! f32 and rounds once at its output; a product of two bf16 tensors rounds at
//! each op. The projections are the Qwen3.5 `matvec`. Ported from
//! herbert-vk-bf16's `vk/gemma_kernels.rs`: the same calls, emitted as MSL.

use super::kernels::pos;
use crate::msl::{done, Elem, Emitted, Kernel, Word};

const WG: u32 = 256;

/// A dispatch over the rows of a prefill block: `per_row` groups for each
/// row; the row and the group within it. One row (`rows == false`): the
/// group is the dispatch's, nothing moves, the code is the decode step's.
fn row_of(k: &mut Kernel, per_row: u32, rows: bool) -> (Option<Word>, Word) {
    let g = k.flat_group();
    if !rows {
        return (None, g);
    }
    let pr = k.u(per_row);
    (Some(k.udiv(g, pr)), k.umod(g, pr))
}

/// Move `buf` on by `row · stride` elements (nothing for a stride of 0).
fn to_row(k: &mut Kernel, buf: crate::msl::Buf, row: Option<Word>, stride: u32) {
    if let (Some(r), true) = (row, stride > 0) {
        let s = k.u(stride);
        let o = k.imul(r, s);
        k.advance(buf, o);
    }
}

/// `x = bf16(table[token] · scale)`, `scale` itself a bf16 (the reference
/// multiplies a bf16 embedding by `bf16(√hidden)`); or, when `state[2]` is
/// `r + 1`, row `r` of the image features, as they are. Bindings
/// `state, table, x, image`.
pub fn embed_scaled(hidden: u32, scale: f32, sg: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let state = k.buffer(0, Elem::U32);
    let table = k.buffer(1, Elem::U16);
    let x = k.buffer(2, Elem::U16);
    let image = k.buffer(3, Elem::U16);
    let i = k.global_id();
    let n = k.u(hidden);
    let inside = k.ult(i, n);
    k.if_(inside, |k| {
        let t = k.u(super::kernels::STATE_TOKEN);
        let token = k.ld(state, t);
        let row = k.imul(token, n);
        let at = k.iadd(row, i);
        let h = k.ld(table, at);
        let f = k.bf16_to_f(h);
        let s = k.f(scale);
        let o = k.fmul(f, s);
        let b = k.f_to_bf16(o);
        let two = k.u(2);
        let flag = k.ld(state, two);
        let zero = k.u(0);
        let is_zero = k.ieq(flag, zero);
        let is_image = k.b_not(is_zero);
        let one = k.u(1);
        let r = k.isub(flag, one);
        let r = k.select_u(is_image, r, zero);
        let rr = k.imul(r, n);
        let iat = k.iadd(rr, i);
        let ih = k.ld(image, iat);
        let out = k.select_u(is_image, ih, b);
        k.st(x, i, out);
    });
    done(format!("g_embed_{hidden}"), k, 4)
}

/// `x = bf16(row[i] · scale)`: the embedding of the token from its row,
/// copied by the host from the checkpoint's mapping (the Q4 engine keeps no
/// bf16 table on the GPU). Bindings `row, x`.
pub fn embed_row(hidden: u32, scale: f32, sg: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let row = k.buffer(0, Elem::U16);
    let x = k.buffer(1, Elem::U16);
    let i = k.global_id();
    let n = k.u(hidden);
    let inside = k.ult(i, n);
    k.if_(inside, |k| {
        let h = k.ld(row, i);
        let f = k.bf16_to_f(h);
        let s = k.f(scale);
        let o = k.fmul(f, s);
        let b = k.f_to_bf16(o);
        k.st(x, i, b);
    });
    done(format!("g_embed_row_{hidden}"), k, 2)
}

/// `x = bf16(x + bf16(rmsnorm(a)·w))`, then `x = bf16(x · s)` with `scalar`.
/// Bindings `a, w, x` and, with `scalar`, `s` (one bf16); one group, or one
/// group per row of a block with `a`, `x` rows `sa`, `sx` apart.
pub fn rmsnorm_add(d: u32, eps: f32, scalar: bool, sg: u32, sa: u32, sx: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let a = k.buffer(0, Elem::U16);
    let w = k.buffer(1, Elem::U16);
    let x = k.buffer(2, Elem::U16);
    let s = if scalar { Some(k.buffer(3, Elem::U16)) } else { None };
    let (row, _) = row_of(&mut k, 1, sa + sx > 0);
    to_row(&mut k, a, row, sa);
    to_row(&mut k, x, row, sx);
    let scratch = k.shared(k.subgroups() + 1, true);
    let lid = k.local_id();
    let (dn, step) = (k.u(d), k.u(WG));
    let zero = k.f(0.0);
    let acc = k.var_f32(zero);
    k.for_(lid, dn, step, |k, i| {
        let h = k.ld(a, i);
        let v = k.bf16_to_f(h);
        let c = k.get(acc);
        let n = k.fma(v, v, c);
        k.set(acc, n);
    });
    let c = k.get(acc);
    let ss = k.wg_reduce(c, scratch, false);
    let inv_d = k.f(1.0 / d as f32);
    let e = k.f(eps);
    let m = k.fmul(ss, inv_d);
    let m = k.fadd(m, e);
    let r = k.sqrt(m);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    k.for_(lid, dn, step, |k, i| {
        let h = k.ld(a, i);
        let v = k.bf16_to_f(h);
        let wh = k.ld(w, i);
        let wf = k.bf16_to_f(wh);
        let n = k.fmul(v, inv);
        let o = k.fmul(n, wf);
        let y = k.round_bf16(o);
        let xh = k.ld(x, i);
        let xf = k.bf16_to_f(xh);
        let sum = k.fadd(xf, y);
        let mut out = k.round_bf16(sum);
        if let Some(s) = s {
            let z = k.u(0);
            let sh = k.ld(s, z);
            let sf = k.bf16_to_f(sh);
            out = k.fmul(out, sf);
        }
        let b = k.f_to_bf16(out);
        k.st(x, i, b);
    });
    done(format!("g_rmsnorm_add_{d}{}{}", if scalar { "_s" } else { "" }, if sa + sx > 0 { format!("_r{sa}_{sx}") } else { String::new() }), k, if scalar { 4 } else { 3 })
}

/// What a head norm does after the norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadOut {
    /// RoPE, written at `[head][0..hd]` of `dst` (the query).
    Rope,
    /// RoPE, written into a K/V cache `[kv_head][capacity][hd]` at the position.
    RopeCache,
    /// No RoPE, into a cache (the value: a norm without weight).
    Cache,
}

/// RMSNorm per head over `hd` (with the weight `[hd]`, or none), RoPE as
/// `rotate_half` with the bf16 `cos`/`sin` tables `[capacity][hd/2]`, and the
/// cache write. One group per head. Bindings `state, src, [w], [cos, sin], dst`.
/// Over the rows of a block: `heads` groups per row, the state, the source and
/// (the query) the destination rows `ss`, `ssrc`, `sdst` apart.
/// With `ring > 0` the cache is a ring of `ring` positions (a sliding layer):
/// `[kv_head][ring][hd]`, the position written at `pos % ring`.
#[allow(clippy::too_many_arguments)]
pub fn head_norm(heads: u32, hd: u32, eps: f32, with_w: bool, out: HeadOut, capacity: u32, sg: u32, ss: u32, ssrc: u32, sdst: u32, ring: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let state = k.buffer(0, Elem::U32);
    let src = k.buffer(1, Elem::U16);
    let mut b = 2;
    let w = if with_w {
        b += 1;
        Some(k.buffer(b - 1, Elem::U16))
    } else {
        None
    };
    let rope = out != HeadOut::Cache;
    let tables = if rope {
        let c = k.buffer(b, Elem::U16);
        let s = k.buffer(b + 1, Elem::U16);
        b += 2;
        Some((c, s))
    } else {
        None
    };
    let dst = k.buffer(b, Elem::U16);
    b += 1;
    let scratch = k.shared(k.subgroups() + 1, true);
    let normed = k.shared(hd, true);
    let (row, h) = row_of(&mut k, heads, ss + ssrc + sdst > 0);
    to_row(&mut k, state, row, ss);
    to_row(&mut k, src, row, ssrc);
    if out == HeadOut::Rope {
        to_row(&mut k, dst, row, sdst);
    }
    let lid = k.local_id();
    let p = pos(&mut k, state);
    let (hdc, step) = (k.u(hd), k.u(WG));
    let base = k.imul(h, hdc);
    let zero = k.f(0.0);
    let acc = k.var_f32(zero);
    k.for_(lid, hdc, step, |k, d| {
        let at = k.iadd(base, d);
        let v = k.ld(src, at);
        let f = k.bf16_to_f(v);
        let c = k.get(acc);
        let n = k.fma(f, f, c);
        k.set(acc, n);
    });
    let c = k.get(acc);
    let ss = k.wg_reduce(c, scratch, false);
    let inv_d = k.f(1.0 / hd as f32);
    let e = k.f(eps);
    let m = k.fmul(ss, inv_d);
    let m = k.fadd(m, e);
    let r = k.sqrt(m);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    // The destination of element `d` of this head.
    let cap_hd = k.u(if ring > 0 { ring } else { capacity } * hd);
    let slot = if ring > 0 && out != HeadOut::Rope {
        let rc = k.u(ring);
        k.umod(p, rc)
    } else {
        p
    };
    let cache_at = |k: &mut Kernel, d: Word| {
        let head_base = k.imul(h, cap_hd);
        let row = k.imul(slot, hdc);
        let t = k.iadd(head_base, row);
        k.iadd(t, d)
    };
    k.for_(lid, hdc, step, |k, d| {
        let at = k.iadd(base, d);
        let v = k.ld(src, at);
        let f = k.bf16_to_f(v);
        let mut n = k.fmul(f, inv);
        if let Some(w) = w {
            let wh = k.ld(w, d);
            let wf = k.bf16_to_f(wh);
            n = k.fmul(n, wf);
        }
        if rope {
            let n = k.round_bf16(n);
            k.sst(normed, d, n);
        } else {
            let bf = k.f_to_bf16(n);
            let to = cache_at(k, d);
            k.st(dst, to, bf);
        }
    });
    if let Some((cos, sin)) = tables {
        k.barrier();
        let half = hd / 2;
        let halfc = k.u(half);
        k.for_(lid, halfc, step, |k, i| {
            let lo = k.sld(normed, i);
            let ih = k.iadd(i, halfc);
            let hi = k.sld(normed, ih);
            let row = k.imul(p, halfc);
            let at = k.iadd(row, i);
            let ch = k.ld(cos, at);
            let sh = k.ld(sin, at);
            let c = k.bf16_to_f(ch);
            let s = k.bf16_to_f(sh);
            // rotate_half(x) = [-x2, x1]; x·cos + rotate_half(x)·sin, each product rounded.
            let lc = k.fmul(lo, c);
            let lc = k.round_bf16(lc);
            let hc = k.fmul(hi, c);
            let hc = k.round_bf16(hc);
            let hs = k.fmul(hi, s);
            let hs = k.round_bf16(hs);
            let ls = k.fmul(lo, s);
            let ls = k.round_bf16(ls);
            let first = k.fsub(lc, hs);
            let second = k.fadd(hc, ls);
            let (a, bb) = (k.f_to_bf16(first), k.f_to_bf16(second));
            let (ia, ib) = if out == HeadOut::Rope {
                let o = k.iadd(base, i);
                let o2 = k.iadd(base, ih);
                (o, o2)
            } else {
                (cache_at(k, i), cache_at(k, ih))
            };
            k.st(dst, ia, a);
            k.st(dst, ib, bb);
        });
    }
    let _ = heads;
    let rows = if ss + ssrc + sdst > 0 { format!("_r{ss}_{ssrc}_{sdst}") } else { String::new() };
    let g = if ring > 0 { format!("_g{ring}") } else { String::new() };
    done(format!("g_head_norm_{hd}_{}{}_c{capacity}{g}{rows}", if with_w { "w" } else { "n" }, match out { HeadOut::Rope => "q", HeadOut::RopeCache => "k", HeadOut::Cache => "v" }), k, b)
}

/// Decode attention of one token, scale 1 (the norms carry the scale), the
/// last `window` positions only when `window > 0`. K/V caches
/// `[kv_head][capacity][hd]`. One group per query head. Bindings
/// `state, q, k_cache, v_cache, scores (f32), out`.
/// Over the rows of a block: `heads` groups per row, the state, the query,
/// the scores and the output rows `ss`, `sq`, `sscores`, `sout` apart.
#[allow(clippy::too_many_arguments)]
pub fn sdpa(heads: u32, kv_heads: u32, hd: u32, capacity: u32, window: u32, sg: u32, ss: u32, sq: u32, sscores: u32, sout: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let state = k.buffer(0, Elem::U32);
    let q = k.buffer(1, Elem::U16);
    let kc = k.buffer(2, Elem::U16);
    let vc = k.buffer(3, Elem::U16);
    let scores = k.buffer(4, Elem::F32);
    k.coherent(scores);
    let out = k.buffer(5, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let qs = k.shared(hd, true);
    let (row, h) = row_of(&mut k, heads, ss + sq + sscores + sout > 0);
    to_row(&mut k, state, row, ss);
    to_row(&mut k, q, row, sq);
    to_row(&mut k, scores, row, sscores);
    to_row(&mut k, out, row, sout);
    let lid = k.local_id();
    let p = pos(&mut k, state);
    let one_u = k.u(1);
    let causal = k.iadd(p, one_u);
    // A sliding layer lets the tokens of an image see their whole group
    // (`state[3]`, the end of the visible range); a full layer stays causal.
    let len = if window > 0 {
        let three = k.u(3);
        let vis = k.ld(state, three);
        let lt = k.ult(causal, vis);
        k.select_u(lt, vis, causal)
    } else {
        causal
    };
    let start = if window > 0 {
        let wc = k.u(window);
        let over = k.ult(wc, causal);
        let s = k.isub(causal, wc);
        let z = k.u(0);
        k.select_u(over, s, z)
    } else {
        k.u(0)
    };
    let group = k.u(heads / kv_heads);
    let kvh = k.udiv(h, group);
    let hdc = k.u(hd);
    let q_base = k.imul(h, hdc);
    let step = k.u(WG);
    k.for_(lid, hdc, step, |k, d| {
        let at = k.iadd(q_base, d);
        let qh = k.ld(q, at);
        let qf = k.bf16_to_f(qh);
        k.sst(qs, d, qf);
    });
    k.barrier();
    let cap = k.u(capacity);
    let cap_hd = k.u(capacity * hd);
    let kv_base = k.imul(kvh, cap_hd);
    let row_base = k.imul(h, cap);
    let neg = k.f(f32::MIN);
    let local_max = k.var_f32(neg);
    let begin = k.iadd(start, lid);
    k.for_(begin, len, step, |k, t| {
        let tr = k.imul(t, hdc);
        let kr = k.iadd(kv_base, tr);
        let zero = k.f(0.0);
        let dot = k.var_f32(zero);
        let zero_u = k.u(0);
        let one = k.u(1);
        k.for_(zero_u, hdc, one, |k, d| {
            let at = k.iadd(kr, d);
            let kh = k.ld(kc, at);
            let kf = k.bf16_to_f(kh);
            let qv = k.sld(qs, d);
            let a = k.get(dot);
            let a = k.fma(qv, kf, a);
            k.set(dot, a);
        });
        let sc = k.get(dot);
        let at = k.iadd(row_base, t);
        k.st(scores, at, sc);
        let m = k.get(local_max);
        let m = k.fmax(m, sc);
        k.set(local_max, m);
    });
    let m = k.get(local_max);
    let mx = k.wg_reduce(m, scratch, true);
    let zero = k.f(0.0);
    let local_sum = k.var_f32(zero);
    k.for_(begin, len, step, |k, t| {
        let at = k.iadd(row_base, t);
        let sc = k.ld(scores, at);
        let d = k.fsub(sc, mx);
        let e = k.exp(d);
        k.st(scores, at, e);
        let a = k.get(local_sum);
        let a = k.fadd(a, e);
        k.set(local_sum, a);
    });
    let ls = k.get(local_sum);
    let total = k.wg_reduce(ls, scratch, false);
    let one = k.f(1.0);
    let inv = k.fdiv(one, total);
    k.for_(lid, hdc, step, |k, d| {
        let zero = k.f(0.0);
        let acc = k.var_f32(zero);
        k.for_(start, len, one_u, |k, t| {
            let at = k.iadd(row_base, t);
            let e = k.ld(scores, at);
            let pw = k.fmul(e, inv);
            let tr = k.imul(t, hdc);
            let vr = k.iadd(kv_base, tr);
            let vat = k.iadd(vr, d);
            let vh = k.ld(vc, vat);
            let vf = k.bf16_to_f(vh);
            let a = k.get(acc);
            let a = k.fma(pw, vf, a);
            k.set(acc, a);
        });
        let o = k.get(acc);
        let b = k.f_to_bf16(o);
        let oat = k.iadd(q_base, d);
        k.st(out, oat, b);
    });
    let rows = if ss + sq + sscores + sout > 0 { format!("_r{ss}_{sq}_{sscores}_{sout}") } else { String::new() };
    done(format!("g_sdpa_h{heads}kv{kv_heads}d{hd}c{capacity}w{window}{rows}"), k, 6)
}

/// The positions a row attends: `[start, len)` as [`sdpa`] computes them
/// (causal, the last `window` on a sliding layer, `state[3]` widening it).
fn attn_range(k: &mut Kernel, state: crate::msl::Buf, window: u32) -> (Word, Word) {
    let p = pos(k, state);
    let one_u = k.u(1);
    let causal = k.iadd(p, one_u);
    let len = if window > 0 {
        let three = k.u(3);
        let vis = k.ld(state, three);
        let lt = k.ult(causal, vis);
        k.select_u(lt, vis, causal)
    } else {
        causal
    };
    let start = if window > 0 {
        let wc = k.u(window);
        let over = k.ult(wc, causal);
        let s = k.isub(causal, wc);
        let z = k.u(0);
        k.select_u(over, s, z)
    } else {
        k.u(0)
    };
    (start, len)
}

/// Positions per group of [`sdpa_qk`].
pub const SDPA_CHUNK: u32 = WG;

/// [`sdpa`] in three dispatches, the same bits: the scores, one thread per
/// (head, position) — `tgid.y` the chunk of [`SDPA_CHUNK`] positions — each
/// the dot product over `hd` in order (K read eight bf16 at a time).
/// Bindings `state, q, k_cache (uint4), scores (f32)`; groups `heads` per row
/// (rows `ss`, `sq`, `sscores` apart), `ceil(span / SDPA_CHUNK)` high.
/// With `ring > 0` the K cache is a ring of `ring` positions (position `t` at `t % ring`).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_qk(heads: u32, kv_heads: u32, hd: u32, capacity: u32, window: u32, sg: u32, ss: u32, sq: u32, sscores: u32, ring: u32) -> Emitted {
    assert!(hd.is_multiple_of(8));
    let mut k = Kernel::new(WG, sg);
    let state = k.buffer(0, Elem::U32);
    let q = k.buffer(1, Elem::U16);
    let kc = k.buffer(2, Elem::UVec4);
    let scores = k.buffer(3, Elem::F32);
    let qs = k.shared(hd, true);
    let rows = ss + sq + sscores > 0;
    let (row, h) = row_of(&mut k, heads, rows);
    to_row(&mut k, state, row, ss);
    to_row(&mut k, q, row, sq);
    to_row(&mut k, scores, row, sscores);
    let lid = k.local_id();
    let (start, len) = attn_range(&mut k, state, window);
    let group = k.u(heads / kv_heads);
    let kvh = k.udiv(h, group);
    let hdc = k.u(hd);
    let q_base = k.imul(h, hdc);
    let step = k.u(WG);
    k.for_(lid, hdc, step, |k, d| {
        let at = k.iadd(q_base, d);
        let qh = k.ld(q, at);
        let qf = k.bf16_to_f(qh);
        k.sst(qs, d, qf);
    });
    k.barrier();
    let c = k.group_id(1);
    let chunk = k.u(SDPA_CHUNK);
    let c0 = k.imul(c, chunk);
    let t0 = k.iadd(start, c0);
    let t = k.iadd(t0, lid);
    let inside = k.ult(t, len);
    k.if_(inside, |k| {
        let cap_hd8 = k.u(if ring > 0 { ring } else { capacity } * hd / 8);
        let kv_base = k.imul(kvh, cap_hd8);
        let hd8 = k.u(hd / 8);
        let slot = if ring > 0 {
            let rc = k.u(ring);
            k.umod(t, rc)
        } else {
            t
        };
        let tr = k.imul(slot, hd8);
        let kr = k.iadd(kv_base, tr);
        let zero = k.f(0.0);
        let dot = k.var_f32(zero);
        let zero_u = k.u(0);
        let one = k.u(1);
        k.for_(zero_u, hd8, one, |k, j| {
            let at = k.iadd(kr, j);
            let kv = k.ld(kc, at);
            let eight = k.u(8);
            let d0 = k.imul(j, eight);
            let mut a = k.get(dot);
            for i in 0..4u32 {
                let w = k.comp(kv, i);
                for (half, kf) in [(0u32, k.lo_f(w)), (1u32, k.hi_f(w))] {
                    let off = k.u(2 * i + half);
                    let d = k.iadd(d0, off);
                    let qv = k.sld(qs, d);
                    a = k.fma(qv, kf, a);
                }
            }
            k.set(dot, a);
        });
        let sc = k.get(dot);
        let cap = k.u(capacity);
        let row_base = k.imul(h, cap);
        let at = k.iadd(row_base, t);
        k.st(scores, at, sc);
    });
    let r = if rows { format!("_r{ss}_{sq}_{sscores}") } else { String::new() };
    let g = if ring > 0 { format!("_g{ring}") } else { String::new() };
    done(format!("g_sdpa_qk_h{heads}kv{kv_heads}d{hd}c{capacity}w{window}{g}{r}"), k, 4)
}

/// [`sdpa`]'s softmax, the same threads over the same positions: the max,
/// `scores = exp(score − max)` in place, `inv[h] = 1/Σ`. Bindings
/// `state, scores, inv (f32)`; groups `heads` per row (rows `ss`, `sscores`,
/// `heads` apart).
pub fn sdpa_softmax(heads: u32, capacity: u32, window: u32, sg: u32, ss: u32, sscores: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let state = k.buffer(0, Elem::U32);
    let scores = k.buffer(1, Elem::F32);
    let invb = k.buffer(2, Elem::F32);
    let scratch = k.shared(k.subgroups() + 1, true);
    let rows = ss + sscores > 0;
    let (row, h) = row_of(&mut k, heads, rows);
    to_row(&mut k, state, row, ss);
    to_row(&mut k, scores, row, sscores);
    to_row(&mut k, invb, row, if rows { heads } else { 0 });
    let lid = k.local_id();
    let (start, len) = attn_range(&mut k, state, window);
    let step = k.u(WG);
    let cap = k.u(capacity);
    let row_base = k.imul(h, cap);
    let neg = k.f(f32::MIN);
    let local_max = k.var_f32(neg);
    let begin = k.iadd(start, lid);
    k.for_(begin, len, step, |k, t| {
        let at = k.iadd(row_base, t);
        let sc = k.ld(scores, at);
        let m = k.get(local_max);
        let m = k.fmax(m, sc);
        k.set(local_max, m);
    });
    let m = k.get(local_max);
    let mx = k.wg_reduce(m, scratch, true);
    let zero = k.f(0.0);
    let local_sum = k.var_f32(zero);
    k.for_(begin, len, step, |k, t| {
        let at = k.iadd(row_base, t);
        let sc = k.ld(scores, at);
        let d = k.fsub(sc, mx);
        let e = k.exp(d);
        k.st(scores, at, e);
        let a = k.get(local_sum);
        let a = k.fadd(a, e);
        k.set(local_sum, a);
    });
    let ls = k.get(local_sum);
    let total = k.wg_reduce(ls, scratch, false);
    let one = k.f(1.0);
    let inv = k.fdiv(one, total);
    let z = k.u(0);
    let first = k.ieq(lid, z);
    k.if_(first, |k| k.st(invb, h, inv));
    let r = if rows { format!("_r{ss}_{sscores}") } else { String::new() };
    done(format!("g_sdpa_sm_h{heads}c{capacity}w{window}{r}"), k, 3)
}

/// Positions whose loads [`sdpa_pv`] issues before their products.
const PV_UNROLL: u32 = 8;

/// [`sdpa`]'s weighted sum of the values, one thread per (head, dimension),
/// the sum over the positions in order: `out = bf16(Σ_t (e_t·inv)·v_t)`, the
/// loads of [`PV_UNROLL`] positions issued before their products. Bindings
/// `state, scores (f32), inv (f32), v_cache, out`; groups `heads` per row
/// (rows `ss`, `sscores`, `heads`, `sout` apart), `ceil(hd / 256)` high.
/// With `ring > 0` the V cache is a ring of `ring` positions.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_pv(heads: u32, kv_heads: u32, hd: u32, capacity: u32, window: u32, sg: u32, ss: u32, sscores: u32, sout: u32, ring: u32) -> Emitted {
    let mut k = Kernel::new(WG.min(hd), sg);
    let state = k.buffer(0, Elem::U32);
    let scores = k.buffer(1, Elem::F32);
    let invb = k.buffer(2, Elem::F32);
    let vc = k.buffer(3, Elem::U16);
    let out = k.buffer(4, Elem::U16);
    let rows = ss + sscores + sout > 0;
    let (row, h) = row_of(&mut k, heads, rows);
    to_row(&mut k, state, row, ss);
    to_row(&mut k, scores, row, sscores);
    to_row(&mut k, invb, row, if rows { heads } else { 0 });
    to_row(&mut k, out, row, sout);
    let lid = k.local_id();
    let (start, len) = attn_range(&mut k, state, window);
    let y = k.group_id(1);
    let wgc = k.u(WG.min(hd));
    let d0 = k.imul(y, wgc);
    let d = k.iadd(d0, lid);
    let hdc = k.u(hd);
    let inside = k.ult(d, hdc);
    k.if_(inside, |k| {
        let group = k.u(heads / kv_heads);
        let kvh = k.udiv(h, group);
        let q_base = k.imul(h, hdc);
        let cap = k.u(capacity);
        let row_base = k.imul(h, cap);
        let cap_hd = k.u(if ring > 0 { ring } else { capacity } * hd);
        let kv_base = k.imul(kvh, cap_hd);
        let vd = k.iadd(kv_base, d);
        let rc = k.u(ring.max(1));
        let slot = |k: &mut Kernel, t: Word| if ring > 0 { k.umod(t, rc) } else { t };
        let inv = k.ld(invb, h);
        let zero = k.f(0.0);
        let acc = k.var_f32(zero);
        let n = k.isub(len, start);
        let mask = k.u(!(PV_UNROLL - 1));
        let n8 = k.and(n, mask);
        let mid = k.iadd(start, n8);
        let un = k.u(PV_UNROLL);
        k.for_(start, mid, un, |k, t0| {
            let mut es = Vec::new();
            let mut vs = Vec::new();
            for i in 0..PV_UNROLL {
                let ii = k.u(i);
                let t = k.iadd(t0, ii);
                let at = k.iadd(row_base, t);
                es.push(k.ld(scores, at));
                let st = slot(k, t);
                let tr = k.imul(st, hdc);
                let vat = k.iadd(vd, tr);
                let vh = k.ld(vc, vat);
                vs.push(k.bf16_to_f(vh));
            }
            let mut a = k.get(acc);
            for i in 0..PV_UNROLL as usize {
                let pw = k.fmul(es[i], inv);
                a = k.fma(pw, vs[i], a);
            }
            k.set(acc, a);
        });
        let one = k.u(1);
        k.for_(mid, len, one, |k, t| {
            let at = k.iadd(row_base, t);
            let e = k.ld(scores, at);
            let pw = k.fmul(e, inv);
            let st = slot(k, t);
            let tr = k.imul(st, hdc);
            let vat = k.iadd(vd, tr);
            let vh = k.ld(vc, vat);
            let vf = k.bf16_to_f(vh);
            let a = k.get(acc);
            let a = k.fma(pw, vf, a);
            k.set(acc, a);
        });
        let o = k.get(acc);
        let b = k.f_to_bf16(o);
        let oat = k.iadd(q_base, d);
        k.st(out, oat, b);
    });
    let r = if rows { format!("_r{ss}_{sscores}_{sout}") } else { String::new() };
    let g = if ring > 0 { format!("_g{ring}") } else { String::new() };
    done(format!("g_sdpa_pv_h{heads}kv{kv_heads}d{hd}c{capacity}w{window}{g}{r}"), k, 5)
}

/// `gelu(x)` (tanh form) in f32: `x·σ(2·√(2/π)·(x + 0.044715·x³))`.
pub(crate) fn gelu(k: &mut Kernel, x: Word) -> Word {
    let x2 = k.fmul(x, x);
    let c = k.f(0.044715);
    let one = k.f(1.0);
    let t = k.fma(c, x2, one);
    let inner = k.fmul(x, t);
    let two_k = k.f(2.0 * 0.797_884_560_802_865_4);
    let u = k.fmul(inner, two_k);
    let s = k.sigmoid(u);
    k.fmul(x, s)
}

/// `h = bf16(bf16(gelu(g)) · u)`: over the fused `[gate (n) | up (n)]` bound
/// as one buffer (`fused`), or with `u` bound apart. Bindings `gate_up, h` or
/// `g, u, h`; groups `ceil(n / 256)`.
/// Over the rows of a block: `ceil(n / 256)` groups per row, `g`, `u` and `h`
/// rows `sgi`, `su`, `sh` apart.
pub fn gelu_mul(n: u32, fused: bool, sg: u32, sgi: u32, su: u32, sh: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let g = k.buffer(0, Elem::U16);
    let u = if fused { g } else { k.buffer(1, Elem::U16) };
    let h = k.buffer(if fused { 1 } else { 2 }, Elem::U16);
    let rows = sgi + su + sh > 0;
    let i = if rows {
        let (row, gi) = row_of(&mut k, n.div_ceil(WG), true);
        to_row(&mut k, g, row, sgi);
        if !fused {
            to_row(&mut k, u, row, su);
        }
        to_row(&mut k, h, row, sh);
        let wg = k.u(WG);
        let b = k.imul(gi, wg);
        let lid = k.local_id();
        k.iadd(b, lid)
    } else {
        k.global_id()
    };
    let nn = k.u(n);
    let inside = k.ult(i, nn);
    k.if_(inside, |k| {
        let gh = k.ld(g, i);
        let ui = if fused { k.iadd(i, nn) } else { i };
        let uh = k.ld(u, ui);
        let gf = k.bf16_to_f(gh);
        let uf = k.bf16_to_f(uh);
        let a = gelu(k, gf);
        let a = k.round_bf16(a);
        let o = k.fmul(a, uf);
        let b = k.f_to_bf16(o);
        k.st(h, i, b);
    });
    done(format!("g_gelu_mul_{n}{}{}", if fused { "_f" } else { "" }, if rows { format!("_r{sgi}_{su}_{sh}") } else { String::new() }), k, if fused { 2 } else { 3 })
}

/// The per-layer inputs of one token, from the projection of its embedding
/// (`proj`, `[layers][d]`) and the raw row of the per-layer table (`raw`):
/// `out = bf16((rmsnorm(bf16(proj·hidden^-½))·w + bf16(raw·√d)) · 2^-½)`, the
/// norm per layer slice of `d`. One group per layer. Bindings `proj, raw, w, out`.
pub fn ple_mix(layers: u32, d: u32, hidden: u32, eps: f32, sg: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let proj = k.buffer(0, Elem::U16);
    let raw = k.buffer(1, Elem::U16);
    let w = k.buffer(2, Elem::U16);
    let out = k.buffer(3, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let l = k.flat_group();
    let lid = k.local_id();
    let (dn, step) = (k.u(d), k.u(WG));
    let base = k.imul(l, dn);
    let scale = k.f(1.0 / (hidden as f32).sqrt());
    let zero = k.f(0.0);
    let acc = k.var_f32(zero);
    k.for_(lid, dn, step, |k, i| {
        let at = k.iadd(base, i);
        let h = k.ld(proj, at);
        let f = k.bf16_to_f(h);
        let s = k.fmul(f, scale);
        let p = k.round_bf16(s);
        let c = k.get(acc);
        let n = k.fma(p, p, c);
        k.set(acc, n);
    });
    let c = k.get(acc);
    let ss = k.wg_reduce(c, scratch, false);
    let inv_d = k.f(1.0 / d as f32);
    let e = k.f(eps);
    let m = k.fmul(ss, inv_d);
    let m = k.fadd(m, e);
    let r = k.sqrt(m);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    let raw_scale = k.f((d as f32).sqrt());
    let half_sqrt = k.f(std::f32::consts::FRAC_1_SQRT_2);
    k.for_(lid, dn, step, |k, i| {
        let at = k.iadd(base, i);
        let h = k.ld(proj, at);
        let f = k.bf16_to_f(h);
        let s = k.fmul(f, scale);
        let p = k.round_bf16(s);
        let wh = k.ld(w, i);
        let wf = k.bf16_to_f(wh);
        let n = k.fmul(p, inv);
        let n = k.fmul(n, wf);
        let n = k.round_bf16(n);
        let rh = k.ld(raw, at);
        let rf = k.bf16_to_f(rh);
        let rs = k.fmul(rf, raw_scale);
        let rs = k.round_bf16(rs);
        let sum = k.fadd(n, rs);
        let sum = k.round_bf16(sum);
        let o = k.fmul(sum, half_sqrt);
        let b = k.f_to_bf16(o);
        k.st(out, at, b);
    });
    done(format!("g_ple_mix_{layers}x{d}"), k, 4)
}

/// `tanh` from `exp`: `1 − 2/(e^{2x} + 1)`, exact at both ends.
fn tanh(k: &mut Kernel, x: Word) -> Word {
    let two = k.f(2.0);
    let x2 = k.fmul(x, two);
    let e = k.exp(x2);
    let one = k.f(1.0);
    let d = k.fadd(e, one);
    let q = k.fdiv(two, d);
    k.fsub(one, q)
}

/// In place, in f32 with no rounding: `l = cap · tanh(l / cap)` (the head of
/// the perplexity, as llama.cpp scores). Binding `logits (f32)`; groups `ceil(n / 256)`.
pub fn softcap_f32(n: u32, cap: f32, sg: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let l = k.buffer(0, Elem::F32);
    let i = k.global_id();
    let nn = k.u(n);
    let inside = k.ult(i, nn);
    k.if_(inside, |k| {
        let f = k.ld(l, i);
        let c = k.f(cap);
        let d = k.fdiv(f, c);
        let t = tanh(k, d);
        let o = k.fmul(t, c);
        k.st(l, i, o);
    });
    done(format!("g_softcap_f32_{n}"), k, 1)
}

/// In place `l = bf16(cap · bf16(tanh(bf16(l / cap))))`. Binding `logits`;
/// groups `ceil(n / 256)`.
pub fn softcap(n: u32, cap: f32, sg: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let l = k.buffer(0, Elem::U16);
    let i = k.global_id();
    let nn = k.u(n);
    let inside = k.ult(i, nn);
    k.if_(inside, |k| {
        let h = k.ld(l, i);
        let f = k.bf16_to_f(h);
        let c = k.f(cap);
        let d = k.fdiv(f, c);
        let d = k.round_bf16(d);
        let t = tanh(k, d);
        let t = k.round_bf16(t);
        let o = k.fmul(t, c);
        let b = k.f_to_bf16(o);
        k.st(l, i, b);
    });
    done(format!("g_softcap_{n}"), k, 1)
}
