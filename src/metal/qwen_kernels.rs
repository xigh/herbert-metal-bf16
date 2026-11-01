//! The kernels Qwen3.5 adds to the shared set: the embedding, the head norms
//! with the partial RoPE and the KV append, the decode attention with its
//! output gate, SwiGLU, and the four steps of the GatedDeltaNet (causal conv,
//! L2 norms and gates, the delta rule, the gated norm). Ported from
//! herbert-vk-bf16's `vk/kernels.rs` call for call, same arithmetic, same
//! rounding points; the DeltaNet keeps its activations and states as
//! [`Elem::Act`] (bf16 with the `bf16` feature, f32 without).
//!
//! The shapes of the two kinds of layer are built from the model's `Config`
//! here, as `vk/types.rs` does.

use super::kernels::{pos, STATE_TOKEN};
use crate::core::config::Config;
use crate::msl::{done, DnScaled, Elem, Emitted, Kernel, ACT_BF16};

/// Geometry of an attention layer.
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary: u32,
    /// Positions the KV cache holds.
    pub capacity: u32,
    pub eps: f32,
}

impl AttnShape {
    pub fn from_config(c: &Config, capacity: usize) -> AttnShape {
        AttnShape {
            heads: c.num_attention_heads as u32,
            kv_heads: c.num_key_value_heads as u32,
            head_dim: c.head_dim as u32,
            rotary: c.rotary_ndims as u32,
            capacity: capacity as u32,
            eps: c.rms_norm_eps,
        }
    }

    /// Width of `q_proj`'s output: Q and its gate, interleaved per head as the
    /// checkpoint stores them, `[Q_h (hd) | gate_h (hd)]`.
    pub fn q_width(&self) -> u32 {
        2 * self.heads * self.head_dim
    }
    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
    /// Width of the fused `[q | k | v]` projection.
    pub fn qkv_width(&self) -> u32 {
        self.q_width() + 2 * self.kv_width()
    }
    /// Width of the attention output, the input of `o_proj`.
    pub fn out_width(&self) -> u32 {
        self.heads * self.head_dim
    }
}

/// Geometry of a GatedDeltaNet layer. The fused projection writes
/// `[q (nk·dk) | k (nk·dk) | v (nv·dv) | z (nv·dv) | a (nv) | b (nv)]`.
#[derive(Clone, Copy, Debug)]
pub struct DeltaShape {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv: u32,
    pub eps: f32,
}

impl DeltaShape {
    pub fn from_config(c: &Config) -> DeltaShape {
        DeltaShape {
            k_heads: c.linear_num_key_heads as u32,
            v_heads: c.linear_num_value_heads as u32,
            k_dim: c.linear_key_head_dim as u32,
            v_dim: c.linear_value_head_dim as u32,
            conv: c.linear_conv_kernel_dim as u32,
            eps: c.rms_norm_eps,
        }
    }

    pub fn key(&self) -> u32 {
        self.k_heads * self.k_dim
    }
    pub fn value(&self) -> u32 {
        self.v_heads * self.v_dim
    }
    /// Channels of the causal conv: `[q | k | v]`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key() + self.value()
    }
    pub fn z_at(&self) -> u32 {
        self.conv_dim()
    }
    pub fn a_at(&self) -> u32 {
        self.z_at() + self.value()
    }
    pub fn b_at(&self) -> u32 {
        self.a_at() + self.v_heads
    }
    /// Width of the fused projection.
    pub fn proj_width(&self) -> u32 {
        self.b_at() + self.v_heads
    }
    /// Elements of the recurrent state of the layer: `[nv][dk][dv]`.
    pub fn state_len(&self) -> u32 {
        self.v_heads * self.k_dim * self.v_dim
    }
}

// ── embedding ────────────────────────────────────────────────────────────────

/// `x = embed[token]`: bindings `state, table (u32 pairs), x (u32 pairs)`;
/// groups `ceil(hidden / 2 / 256)`.
pub fn embed(hidden: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let state = k.buffer(0, Elem::U32);
    let table = k.buffer(1, Elem::U32);
    let x = k.buffer(2, Elem::U32);
    let words = hidden / 2;
    let i = k.global_id();
    let n = k.u(words);
    let inside = k.ult(i, n);
    k.if_(inside, |k| {
        let t = k.u(STATE_TOKEN);
        let token = k.ld(state, t);
        let row = k.imul(token, n);
        let at = k.iadd(row, i);
        let v = k.ld(table, at);
        k.st(x, i, v);
    });
    done(format!("embed_{hidden}"), k, 3)
}

// ── attention ────────────────────────────────────────────────────────────────

/// Head norms, partial RoPE and the KV append, one group per head: groups
/// `heads` (Q, in place), `kv_heads` (K, into the cache), `kv_heads` (V, copied
/// into the cache). Per head: `y = bf16((x·inv)·(1+w))`, then on the first
/// `rotary` dims the pairs `(i, i + rotary/2)` rotate, `a' = a·c − b·s`,
/// `b' = a·s + b·c`, with the bf16 tables of the position.
/// Bindings `state, qkv, q_norm, k_norm, cos, sin, k_cache, v_cache`.
pub fn qk_rope_append(s: AttnShape, sg: u32) -> Emitted {
    qk_rope_append_pf(s, sg, 0)
}

/// [`qk_rope_append`] with `kernels::pf_head`'s warmer (bindings 8 and 9).
pub fn qk_rope_append_pf(s: AttnShape, sg: u32, pf: u32) -> Emitted {
    let hd = s.head_dim;
    let half = s.rotary / 2;
    let mut k = Kernel::new(hd, sg);
    super::kernels::pf_head(&mut k, s.heads + 2 * s.kv_heads, 8, pf, hd);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let qn = k.buffer(2, Elem::U16);
    let kn = k.buffer(3, Elem::U16);
    let cos = k.buffer(4, Elem::U16);
    let sin = k.buffer(5, Elem::U16);
    let kc = k.buffer(6, Elem::U16);
    let vc = k.buffer(7, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let row = k.shared(hd, true);
    let g = k.flat_group();
    let lid = k.local_id();
    let p = pos(&mut k, state);
    let heads = k.u(s.heads);
    let is_q = k.ult(g, heads);
    let hk = k.u(s.heads + s.kv_heads);
    let is_qk = k.ult(g, hk);
    let hdc = k.u(hd);
    let cap_hd = k.u(s.capacity * hd);
    k.if_else(
        is_qk,
        |k| {
            // Q head g, or K head g − heads.
            let qw = k.u(s.q_width());
            let two_hd = k.u(2 * hd);
            let kvh = k.isub(g, heads);
            let q_at = k.imul(g, two_hd);
            let kv_at = k.imul(kvh, hdc);
            let k_at = k.iadd(qw, kv_at);
            let src = k.select_u(is_q, q_at, k_at);
            let at = k.iadd(src, lid);
            let h = k.ld(qkv, at);
            let x = k.bf16_to_f(h);
            let x2 = k.fmul(x, x);
            let ss = k.wg_reduce(x2, scratch, false);
            let inv_d = k.f(1.0 / hd as f32);
            let e = k.f(s.eps);
            let m = k.fmul(ss, inv_d);
            let m = k.fadd(m, e);
            let r = k.sqrt(m);
            let one = k.f(1.0);
            let inv = k.fdiv(one, r);
            let wq = k.ld(qn, lid);
            let wk = k.ld(kn, lid);
            let wh = k.select_u(is_q, wq, wk);
            let wf = k.bf16_to_f(wh);
            let wf = k.fadd(wf, one);
            let n = k.fmul(x, inv);
            let y = k.fmul(n, wf);
            let y = k.round_bf16(y);
            k.sst(row, lid, y);
            k.barrier();
            let rot = k.u(s.rotary);
            let hc = k.u(half);
            let in_rot = k.ult(lid, rot);
            let out = k.var_f32(y);
            k.if_(in_rot, |k| {
                let first = k.ult(lid, hc);
                let j_hi = k.isub(lid, hc);
                let j = k.select_u(first, lid, j_hi);
                let ph = k.imul(p, hc);
                let t = k.iadd(ph, j);
                let ch = k.ld(cos, t);
                let sh = k.ld(sin, t);
                let c = k.bf16_to_f(ch);
                let sn = k.bf16_to_f(sh);
                let a = k.sld(row, j);
                let jb = k.iadd(j, hc);
                let b = k.sld(row, jb);
                // first half: a·c − b·s ; second half: a·s + b·c
                let ac = k.fmul(a, c);
                let bs = k.fmul(b, sn);
                let lo = k.fsub(ac, bs);
                let as_ = k.fmul(a, sn);
                let bc = k.fmul(b, c);
                let hi = k.fadd(as_, bc);
                let r = k.select_f(first, lo, hi);
                k.set(out, r);
            });
            let o = k.get(out);
            let b = k.f_to_bf16(o);
            k.if_else(
                is_q,
                |k| k.st(qkv, at, b),
                |k| {
                    let base = k.imul(kvh, cap_hd);
                    let pr = k.imul(p, hdc);
                    let a = k.iadd(base, pr);
                    let a = k.iadd(a, lid);
                    k.st(kc, a, b);
                },
            );
        },
        |k| {
            // V head g − heads − kv_heads: copied as is.
            let vh = k.isub(g, hk);
            let vbase = k.u(s.q_width() + s.kv_width());
            let off = k.imul(vh, hdc);
            let src = k.iadd(vbase, off);
            let src = k.iadd(src, lid);
            let v = k.ld(qkv, src);
            let base = k.imul(vh, cap_hd);
            let pr = k.imul(p, hdc);
            let a = k.iadd(base, pr);
            let a = k.iadd(a, lid);
            k.st(vc, a, v);
        },
    );
    let (tag, nb) = super::kernels::pf_tag(pf, 8);
    done(format!("qk_rope_append_h{}kv{}d{}{tag}", s.heads, s.kv_heads, hd), k, nb)
}

/// Decode attention, one group per query head, over the `pos + 1` cached keys.
/// Scores `q·k / √hd` in f32 (a scratch row per head), softmax max-subtracted,
/// `p = e / Σe`, `o = Σ p·v`, rounded to bf16, then the output gate
/// `bf16(o·σ(gate))`. Bindings `state, qkv, k_cache, v_cache, scores, out`.
pub fn sdpa_decode(s: AttnShape, sg: u32) -> Emitted {
    let hd = s.head_dim;
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let kc = k.buffer(2, Elem::U16);
    let vc = k.buffer(3, Elem::U16);
    let scores = k.buffer(4, Elem::F32);
    k.coherent(scores);
    let out = k.buffer(5, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let qs = k.shared(hd, true);
    let h = k.flat_group();
    let lid = k.local_id();
    let p = pos(&mut k, state);
    let one_u = k.u(1);
    let len = k.iadd(p, one_u);
    let group = k.u(s.heads / s.kv_heads);
    let kvh = k.udiv(h, group);
    let hdc = k.u(hd);
    let two_hd = k.u(2 * hd);
    let q_base = k.imul(h, two_hd);
    let step = k.u(wg);
    // q of the head into threadgroup memory.
    k.for_(lid, hdc, step, |k, d| {
        let at = k.iadd(q_base, d);
        let qh = k.ld(qkv, at);
        let q = k.bf16_to_f(qh);
        k.sst(qs, d, q);
    });
    k.barrier();
    let cap = k.u(s.capacity);
    let cap_hd = k.u(s.capacity * hd);
    let k_base = k.imul(kvh, cap_hd);
    let row_base = k.imul(h, cap);
    let scale = k.f(1.0 / (hd as f32).sqrt());
    let neg = k.f(f32::MIN);
    let local_max = k.var_f32(neg);
    k.for_(lid, len, step, |k, t| {
        let tr = k.imul(t, hdc);
        let kr = k.iadd(k_base, tr);
        let zero = k.f(0.0);
        let dot = k.var_f32(zero);
        let zero_u = k.u(0);
        let one = k.u(1);
        k.for_(zero_u, hdc, one, |k, d| {
            let at = k.iadd(kr, d);
            let kh = k.ld(kc, at);
            let kf = k.bf16_to_f(kh);
            let q = k.sld(qs, d);
            let a = k.get(dot);
            let a = k.fma(q, kf, a);
            k.set(dot, a);
        });
        let dv = k.get(dot);
        let sc = k.fmul(dv, scale);
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
    k.for_(lid, len, step, |k, t| {
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
        let zero_u = k.u(0);
        let one_u = k.u(1);
        k.for_(zero_u, len, one_u, |k, t| {
            let at = k.iadd(row_base, t);
            let e = k.ld(scores, at);
            let pw = k.fmul(e, inv);
            let tr = k.imul(t, hdc);
            let vr = k.iadd(k_base, tr);
            let vat = k.iadd(vr, d);
            let vh = k.ld(vc, vat);
            let vf = k.bf16_to_f(vh);
            let a = k.get(acc);
            let a = k.fma(pw, vf, a);
            k.set(acc, a);
        });
        let o = k.get(acc);
        let o = k.round_bf16(o);
        let gate_at = k.iadd(q_base, hdc);
        let gate_at = k.iadd(gate_at, d);
        let gh = k.ld(qkv, gate_at);
        let gf = k.bf16_to_f(gh);
        let sg_ = k.sigmoid(gf);
        let r = k.fmul(o, sg_);
        let b = k.f_to_bf16(r);
        let ob = k.imul(h, hdc);
        let oat = k.iadd(ob, d);
        k.st(out, oat, b);
    });
    done(format!("sdpa_decode_h{}kv{}d{}", s.heads, s.kv_heads, hd), k, 6)
}

/// Decode attention split over the keys (flash decoding), first pass: group
/// `h · chunks + c` takes the `c`-th of `chunks` equal runs of the `pos + 1`
/// keys of head `h`, its eight simdgroups a key at a time each (a lane per
/// eight dimensions, `q·k` reduced over the simdgroup), the softmax online
/// (max, `Σe`, `Σe·v` rescaled as the max grows), then the eight merged in
/// threadgroup memory. Writes `[o (hd) | max | Σe]` in f32 to `part[h][c]`.
/// Not the sums of [`sdpa_decode`]: `o = Σ (e·v) / Σe` in another order.
/// Bindings `state, qkv, k_cache (uint4), v_cache (uint4), part`; groups
/// `heads · chunks`.
pub fn sdpa_decode_part(s: AttnShape, sg: u32, chunks: u32) -> Emitted {
    sdpa_decode_part_pf(s, sg, chunks, 0)
}

/// [`sdpa_decode_part`] with `kernels::pf_head`'s warmer (bindings 5 and 6).
pub fn sdpa_decode_part_pf(s: AttnShape, sg: u32, chunks: u32, pf: u32) -> Emitted {
    sdpa_part_impl(s, sg, chunks, pf, false)
}

/// [`sdpa_decode_part`] with [`qk_rope_append`] done first by every group
/// (`dec.frope`, a head of 256): the group norms and rotates its head's q
/// into threadgroup memory (the same reduction over 256 invocations, the
/// same bits) and reads it from there; its K/V head likewise, written to the
/// caches at the position by the groups whose run holds it (the heads of a
/// K/V head write the same bytes), read back after a barrier. `q` is not
/// written back to `qkv`. One dispatch less an attention layer.
/// Bindings `state, qkv, k_cache, v_cache, part, q_norm, k_norm, cos, sin`
/// and the warmer's two.
pub fn sdpa_decode_part_rope(s: AttnShape, sg: u32, chunks: u32, pf: u32) -> Emitted {
    sdpa_part_impl(s, sg, chunks, pf, true)
}

fn sdpa_part_impl(s: AttnShape, sg: u32, chunks: u32, pf: u32, rope: bool) -> Emitted {
    use crate::msl::{Buf, Word};
    let hd = s.head_dim;
    assert!(sg == 32 && hd.is_multiple_of(256), "sdpa_decode_part: head_dim {hd} a multiple of 256, simdgroups of 32");
    assert!(!rope || hd == 256, "sdpa_decode_part with the rope: a head of 256");
    let nv = hd / 256;
    let wg = 256;
    let nsg = wg / sg;
    let mut k = Kernel::new(wg, sg);
    let nb = if rope { 9 } else { 5 };
    super::kernels::pf_head(&mut k, s.heads * chunks, nb, pf, wg);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let kc = k.buffer(2, Elem::UVec4);
    let vc = k.buffer(3, Elem::UVec4);
    let part = k.buffer(4, Elem::F32);
    let ms = k.shared(nsg, true);
    let ls = k.shared(nsg, true);
    let os = k.shared(nsg * hd, true);
    let g = k.flat_group();
    let chc = k.u(chunks);
    let h = k.udiv(g, chc);
    let c = k.umod(g, chc);
    let sgid = k.subgroup_id();
    let lane = k.lane();
    let lid = k.local_id();
    let p = pos(&mut k, state);
    let one_u = k.u(1);
    let len = k.iadd(p, one_u);
    let cm1 = k.u(chunks - 1);
    let per = k.iadd(len, cm1);
    let per = k.udiv(per, chc);
    let t0 = k.imul(c, per);
    let t1 = k.iadd(t0, per);
    let short = k.ult(len, t1);
    let t1 = k.select_u(short, len, t1);
    let group = k.u(s.heads / s.kv_heads);
    let kvh = k.udiv(h, group);
    let two_hd = k.u(2 * hd);
    let q_base = k.imul(h, two_hd);
    // `dec.frope`: `qk_rope_append` of head `h` and of its K/V head, q into `qrow`.
    let qrow = if rope {
        let qn = k.buffer(5, Elem::U16);
        let kn = k.buffer(6, Elem::U16);
        let cos = k.buffer(7, Elem::U16);
        let sin = k.buffer(8, Elem::U16);
        let kc16 = k.buffer(2, Elem::U16);
        let vc16 = k.buffer(3, Elem::U16);
        let scratch = k.shared(nsg + 1, true);
        let row = k.shared(hd, true);
        let qrow = k.shared(hd, true);
        let half = s.rotary / 2;
        let hdc = k.u(hd);
        let one = k.f(1.0);
        let inv_d = k.f(1.0 / hd as f32);
        let e = k.f(s.eps);
        let rot = k.u(s.rotary);
        let hc = k.u(half);
        let in_rot = k.ult(lid, rot);
        // The group whose run holds the position appends K and V.
        let below = k.ult(p, t0);
        let ge = k.b_not(below);
        let lt = k.ult(p, t1);
        let appends = k.land(ge, lt);
        let cap_hd = k.u(s.capacity * hd);
        let kv_at = k.imul(kvh, hdc);
        let cbase = k.imul(kvh, cap_hd);
        let pr = k.imul(p, hdc);
        let cat = k.iadd(cbase, pr);
        let cat = k.iadd(cat, lid);
        // `qk_rope_append`'s head: `y = bf16((x·inv)·(1+w))`, then the rotation; the bf16 of the result.
        let head = |k: &mut Kernel, at: Word, w: Buf| -> Word {
            let hx = k.ld(qkv, at);
            let x = k.bf16_to_f(hx);
            let x2 = k.fmul(x, x);
            let ss = k.wg_reduce(x2, scratch, false);
            let m = k.fmul(ss, inv_d);
            let m = k.fadd(m, e);
            let r = k.sqrt(m);
            let inv = k.fdiv(one, r);
            let wh = k.ld(w, lid);
            let wf = k.bf16_to_f(wh);
            let wf = k.fadd(wf, one);
            let n = k.fmul(x, inv);
            let y = k.fmul(n, wf);
            let y = k.round_bf16(y);
            k.sst(row, lid, y);
            k.barrier();
            let out = k.var_f32(y);
            k.if_(in_rot, |k| {
                let first = k.ult(lid, hc);
                let j_hi = k.isub(lid, hc);
                let j = k.select_u(first, lid, j_hi);
                let ph = k.imul(p, hc);
                let t = k.iadd(ph, j);
                let ch = k.ld(cos, t);
                let sh = k.ld(sin, t);
                let cf = k.bf16_to_f(ch);
                let sn = k.bf16_to_f(sh);
                let a = k.sld(row, j);
                let jb = k.iadd(j, hc);
                let b = k.sld(row, jb);
                let ac = k.fmul(a, cf);
                let bs = k.fmul(b, sn);
                let lo = k.fsub(ac, bs);
                let as_ = k.fmul(a, sn);
                let bc = k.fmul(b, cf);
                let hi = k.fadd(as_, bc);
                let r = k.select_f(first, lo, hi);
                k.set(out, r);
            });
            let o = k.get(out);
            // `row` is read by every invocation before the next head overwrites it.
            k.barrier();
            k.f_to_bf16(o)
        };
        let qa = k.iadd(q_base, lid);
        let qb = head(&mut k, qa, qn);
        let qf = k.bf16_to_f(qb);
        k.sst(qrow, lid, qf);
        let qw = k.u(s.q_width());
        let ka = k.iadd(qw, kv_at);
        let ka = k.iadd(ka, lid);
        let kb = head(&mut k, ka, kn);
        let vbase = k.u(s.q_width() + s.kv_width());
        let va = k.iadd(vbase, kv_at);
        let va = k.iadd(va, lid);
        let v = k.ld(qkv, va);
        k.if_(appends, |k| {
            k.st(kc16, cat, kb);
            k.st(vc16, cat, v);
        });
        // The caches' new row and `qrow` seen by the whole group.
        k.barrier();
        Some(qrow)
    } else {
        None
    };
    // q: the lane's 8·nv dimensions, `lane·8 + 256·i + j`.
    let mut q = Vec::new();
    let eight = k.u(8);
    let l8 = k.imul(lane, eight);
    for i in 0..nv {
        for j in 0..8 {
            let o = k.u(256 * i + j);
            let at = k.iadd(l8, o);
            match qrow {
                Some(qrow) => q.push(k.sld(qrow, at)),
                None => {
                    let at = k.iadd(q_base, at);
                    let qh = k.ld(qkv, at);
                    q.push(k.bf16_to_f(qh));
                }
            }
        }
    }
    let row = k.u(hd / 8);
    let cap = k.u(s.capacity);
    let kvb = k.imul(kvh, cap);
    let scale = k.f(1.0 / (hd as f32).sqrt());
    let neg = k.f(f32::MIN);
    let zero = k.f(0.0);
    let m = k.var_f32(neg);
    let l = k.var_f32(zero);
    let o: Vec<_> = (0..8 * nv).map(|_| k.var_f32(zero)).collect();
    let start = k.iadd(t0, sgid);
    let step = k.u(nsg);
    k.for_(start, t1, step, |k, t| {
        let tr = k.iadd(kvb, t);
        let tr = k.imul(tr, row);
        let tr = k.iadd(tr, lane);
        let mut dot = k.f(0.0);
        let mut vs = Vec::new();
        for i in 0..nv {
            let off = k.u(32 * i);
            let at = k.iadd(tr, off);
            let kv = k.ld(kc, at);
            let vv = k.ld(vc, at);
            for w in 0..4 {
                let kw = k.comp(kv, w);
                let lo = k.lo_f(kw);
                dot = k.fma(q[(8 * i + 2 * w) as usize], lo, dot);
                let hi = k.hi_f(kw);
                dot = k.fma(q[(8 * i + 2 * w + 1) as usize], hi, dot);
                let vw = k.comp(vv, w);
                vs.push(k.lo_f(vw));
                vs.push(k.hi_f(vw));
            }
        }
        let dot = k.sg_fadd(dot);
        let sc = k.fmul(dot, scale);
        let mo = k.get(m);
        let mn = k.fmax(mo, sc);
        let dm = k.fsub(mo, mn);
        let corr = k.exp(dm);
        let ds = k.fsub(sc, mn);
        let e = k.exp(ds);
        let lo = k.get(l);
        let lo = k.fmul(lo, corr);
        let lo = k.fadd(lo, e);
        k.set(l, lo);
        k.set(m, mn);
        for (j, &ov) in o.iter().enumerate() {
            let a = k.get(ov);
            let a = k.fmul(a, corr);
            let a = k.fma(e, vs[j], a);
            k.set(ov, a);
        }
    });
    // The simdgroups' runs into threadgroup memory, merged a dimension per invocation.
    let zero_u = k.u(0);
    let first = k.ieq(lane, zero_u);
    let mv = k.get(m);
    let lv = k.get(l);
    k.if_(first, |k| {
        k.sst(ms, sgid, mv);
        k.sst(ls, sgid, lv);
    });
    let hdc = k.u(hd);
    let ob = k.imul(sgid, hdc);
    for i in 0..nv {
        for j in 0..8 {
            let off = k.u(256 * i + j);
            let d = k.iadd(l8, off);
            let at = k.iadd(ob, d);
            let v = k.get(o[(8 * i + j) as usize]);
            k.sst(os, at, v);
        }
    }
    k.barrier();
    let mut mx = k.sld(ms, zero_u);
    for x in 1..nsg {
        let xi = k.u(x);
        let v = k.sld(ms, xi);
        mx = k.fmax(mx, v);
    }
    let pb = k.imul(h, chc);
    let pb = k.iadd(pb, c);
    let stride = k.u(hd + 2);
    let pb = k.imul(pb, stride);
    let wgc = k.u(wg);
    k.for_(lid, hdc, wgc, |k, d| {
        let mut acc = k.f(0.0);
        for x in 0..nsg {
            let xi = k.u(x);
            let mx_ = k.sld(ms, xi);
            let dm = k.fsub(mx_, mx);
            let f = k.exp(dm);
            let oo = k.u(x * hd);
            let at = k.iadd(oo, d);
            let ov = k.sld(os, at);
            acc = k.fma(ov, f, acc);
        }
        let at = k.iadd(pb, d);
        k.st(part, at, acc);
    });
    let lz = k.ieq(lid, zero_u);
    k.if_(lz, |k| {
        let mut sum = k.f(0.0);
        for x in 0..nsg {
            let xi = k.u(x);
            let mx_ = k.sld(ms, xi);
            let dm = k.fsub(mx_, mx);
            let f = k.exp(dm);
            let lx = k.sld(ls, xi);
            sum = k.fma(lx, f, sum);
        }
        let at = k.iadd(pb, hdc);
        k.st(part, at, mx);
        let at = k.iadd(at, one_u);
        k.st(part, at, sum);
    });
    let (tag, nb) = super::kernels::pf_tag(pf, nb);
    done(format!("sdpa_decode_part_h{}kv{}d{}_c{chunks}{}{tag}", s.heads, s.kv_heads, hd, if rope { "_rope" } else { "" }), k, nb)
}

/// [`sdpa_decode_part`]'s second pass: the `chunks` runs of each head merged
/// (`o = Σ o_c·e^(m_c − m) / Σ l_c·e^(m_c − m)`), rounded to bf16, then the
/// output gate as [`sdpa_decode`]'s. Bindings `qkv, part, out`; groups `heads`.
pub fn sdpa_decode_merge(s: AttnShape, sg: u32, chunks: u32) -> Emitted {
    sdpa_decode_merge_pf(s, sg, chunks, 0)
}

/// [`sdpa_decode_merge`] with `kernels::pf_head`'s warmer (bindings 3 and 4).
pub fn sdpa_decode_merge_pf(s: AttnShape, sg: u32, chunks: u32, pf: u32) -> Emitted {
    let hd = s.head_dim;
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    super::kernels::pf_head(&mut k, s.heads, 3, pf, wg);
    let qkv = k.buffer(0, Elem::U16);
    let part = k.buffer(1, Elem::F32);
    let out = k.buffer(2, Elem::U16);
    let h = k.flat_group();
    let lid = k.local_id();
    let chc = k.u(chunks);
    let stride = k.u(hd + 2);
    let hc = k.imul(h, chc);
    let base = k.imul(hc, stride);
    let hdc = k.u(hd);
    let mut mts = Vec::new();
    let mut mx = k.f(f32::MIN);
    for c in 0..chunks {
        let o = k.u(c * (hd + 2) + hd);
        let at = k.iadd(base, o);
        let m = k.ld(part, at);
        mts.push(m);
        mx = k.fmax(mx, m);
    }
    let mut fs = Vec::new();
    let mut sum = k.f(0.0);
    for c in 0..chunks {
        let dm = k.fsub(mts[c as usize], mx);
        let f = k.exp(dm);
        let o = k.u(c * (hd + 2) + hd + 1);
        let at = k.iadd(base, o);
        let lc = k.ld(part, at);
        sum = k.fma(lc, f, sum);
        fs.push(f);
    }
    let one = k.f(1.0);
    let inv = k.fdiv(one, sum);
    let wgc = k.u(wg);
    let two_hd = k.u(2 * hd);
    let q_base = k.imul(h, two_hd);
    k.for_(lid, hdc, wgc, |k, d| {
        let mut acc = k.f(0.0);
        for c in 0..chunks {
            let o = k.u(c * (hd + 2));
            let at = k.iadd(base, o);
            let at = k.iadd(at, d);
            let v = k.ld(part, at);
            acc = k.fma(v, fs[c as usize], acc);
        }
        let o = k.fmul(acc, inv);
        let o = k.round_bf16(o);
        let gate_at = k.iadd(q_base, hdc);
        let gate_at = k.iadd(gate_at, d);
        let gh = k.ld(qkv, gate_at);
        let gf = k.bf16_to_f(gh);
        let sg_ = k.sigmoid(gf);
        let r = k.fmul(o, sg_);
        let b = k.f_to_bf16(r);
        let ob = k.imul(h, hdc);
        let oat = k.iadd(ob, d);
        k.st(out, oat, b);
    });
    let (tag, nb) = super::kernels::pf_tag(pf, 3);
    done(format!("sdpa_decode_merge_h{}d{}_c{chunks}{tag}", s.heads, hd), k, nb)
}

// ── MLP ──────────────────────────────────────────────────────────────────────

/// `h = bf16(g·σ(g)·u)` over the fused `[gate (n) | up (n)]`.
/// Bindings `gate_up, h`; groups `ceil(n / 256)`.
pub fn swiglu(n: u32, sg: u32) -> Emitted {
    swiglu_pf(n, sg, 0)
}

/// [`swiglu`] and, with `pf` > 0, `prefetch_groups(pf)` groups more that
/// read `pf` uint4 of the binding 2 (`kernels::prefetch_part`, sink at 3).
pub fn swiglu_pf(n: u32, sg: u32, pf: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let gu = k.buffer(0, Elem::U16);
    let h = k.buffer(1, Elem::U16);
    let real = n.div_ceil(256);
    if pf > 0 {
        let src = k.buffer(2, Elem::UVec4);
        let sink = k.buffer(3, Elem::U32);
        let g = k.flat_group();
        let rc = k.u(real);
        let is_pf = k.ult(g, rc);
        let is_pf = k.b_not(is_pf);
        k.if_(is_pf, |k| {
            let gr = k.isub(g, rc);
            super::kernels::prefetch_part(k, gr, src, sink, pf);
        });
    }
    let i = k.global_id();
    let nn = k.u(n);
    let inside = k.ult(i, nn);
    k.if_(inside, |k| {
        let gh = k.ld(gu, i);
        let ui = k.iadd(i, nn);
        let uh = k.ld(gu, ui);
        let g = k.bf16_to_f(gh);
        let u = k.bf16_to_f(uh);
        let s = k.sigmoid(g);
        let gs = k.fmul(g, s);
        let o = k.fmul(gs, u);
        let b = k.f_to_bf16(o);
        k.st(h, i, b);
    });
    let tag = if pf > 0 { format!("_pf{pf}") } else { String::new() };
    done(format!("swiglu_{n}{tag}"), k, if pf > 0 { 4 } else { 2 })
}

// ── GatedDeltaNet ────────────────────────────────────────────────────────────

/// Causal depthwise conv + SiLU, in place on the `[q | k | v]` channels. The
/// state of a channel holds its last `conv` inputs, the current one last;
/// `y = silu(Σ_j state[j]·w[j])`. Bindings `proj (act), conv_state (act),
/// conv_w (bf16 [ch][conv])`; groups `ceil(conv_dim / 256)`.
pub fn dn_conv(s: DeltaShape, sg: u32) -> Emitted {
    dn_conv_pf(s, sg, 0)
}

/// [`dn_conv`] with `kernels::pf_head`'s warmer (`pf` uint4, bindings 3 and 4).
pub fn dn_conv_pf(s: DeltaShape, sg: u32, pf: u32) -> Emitted {
    let kk = s.conv;
    let mut k = Kernel::new(256, sg);
    super::kernels::pf_head(&mut k, s.conv_dim().div_ceil(256), 3, pf, 256);
    let proj = k.buffer(0, Elem::Act);
    let st = k.buffer(1, Elem::Act);
    let w = k.buffer(2, Elem::U16);
    let c = k.global_id();
    let ch = k.u(s.conv_dim());
    let inside = k.ult(c, ch);
    k.if_(inside, |k| {
        let kc = k.u(kk);
        let base = k.imul(c, kc);
        let x = k.ld(proj, c);
        let zero = k.f(0.0);
        let mut acc = zero;
        let mut shifted = Vec::new();
        for j in 0..kk {
            let v = if j + 1 < kk {
                let jn = k.u(j + 1);
                let at = k.iadd(base, jn);
                k.ld(st, at)
            } else {
                x
            };
            shifted.push(v);
        }
        for (j, &v) in shifted.iter().enumerate() {
            let jc = k.u(j as u32);
            let at = k.iadd(base, jc);
            k.st(st, at, v);
            let wh = k.ld(w, at);
            let wf = k.bf16_to_f(wh);
            acc = k.fma(v, wf, acc);
        }
        let sgm = k.sigmoid(acc);
        let y = k.fmul(acc, sgm);
        k.st(proj, c, y);
    });
    let (tag, nb) = super::kernels::pf_tag(pf, 3);
    done(format!("dn_conv_{}x{kk}{tag}", s.conv_dim()), k, nb)
}

/// L2 norm of every q and k head, in place, `x · 1/√(Σx² + 1e-6)`; and the
/// gates `β = σ(b)`, `decay = exp(−exp(A_log)·softplus(a + dt_bias))`
/// (softplus(x) = x past 20). Groups `2·k_heads + 1`, the last one for the
/// gates. Bindings `proj, a_log (f32), dt_bias (f32), gates (act [β | decay])`.
pub fn dn_l2_gates(s: DeltaShape, sg: u32) -> Emitted {
    dn_l2_gates_pf(s, sg, 0)
}

/// [`dn_l2_gates`] with `kernels::pf_head`'s warmer (bindings 4 and 5).
pub fn dn_l2_gates_pf(s: DeltaShape, sg: u32, pf: u32) -> Emitted {
    let dk = s.k_dim;
    let mut k = Kernel::new(dk, sg);
    super::kernels::pf_head(&mut k, 2 * s.k_heads + 1, 4, pf, dk);
    let proj = k.buffer(0, Elem::Act);
    let a_log = k.buffer(1, Elem::F32);
    let dt_bias = k.buffer(2, Elem::F32);
    let gates = k.buffer(3, Elem::Act);
    let scratch = k.shared(k.subgroups() + 1, true);
    let g = k.flat_group();
    let lid = k.local_id();
    let heads = k.u(2 * s.k_heads);
    let is_head = k.ult(g, heads);
    k.if_else(
        is_head,
        |k| {
            let dkc = k.u(dk);
            let base = k.imul(g, dkc);
            let at = k.iadd(base, lid);
            let x = k.ld(proj, at);
            let x2 = k.fmul(x, x);
            let ss = k.wg_reduce(x2, scratch, false);
            let e = k.f(1e-6);
            let m = k.fadd(ss, e);
            let r = k.sqrt(m);
            let one = k.f(1.0);
            let inv = k.fdiv(one, r);
            let y = k.fmul(x, inv);
            k.st(proj, at, y);
        },
        |k| {
            let nv = k.u(s.v_heads);
            let step = k.u(dk);
            k.for_(lid, nv, step, |k, i| {
                let a_at = k.u(s.a_at());
                let b_at = k.u(s.b_at());
                let ai = k.iadd(a_at, i);
                let bi = k.iadd(b_at, i);
                let a = k.ld(proj, ai);
                let b = k.ld(proj, bi);
                let beta = k.sigmoid(b);
                let bias = k.ld(dt_bias, i);
                let x = k.fadd(a, bias);
                let ex = k.exp(x);
                let one = k.f(1.0);
                let l = k.fadd(one, ex);
                let sp = k.log(l);
                let twenty = k.f(20.0);
                let big = k.fgt(x, twenty);
                let dt = k.select_f(big, x, sp);
                let al = k.ld(a_log, i);
                let ea = k.exp(al);
                let prod = k.fmul(ea, dt);
                let neg = k.fneg(prod);
                // With the `bf16` feature the gates hold g = log(decay) (vLLM keeps g, not
                // the decay: near 1 a bf16 decay is off by 2⁻⁹, a bf16 g by 2⁻⁹·|g|).
                let decay = if ACT_BF16 { neg } else { k.exp(neg) };
                k.st(gates, i, beta);
                let di = k.iadd(i, nv);
                k.st(gates, di, decay);
            });
        },
    );
    let (tag, nb) = super::kernels::pf_tag(pf, 4);
    done(format!("dn_l2_gates_k{}v{}{tag}", s.k_heads, s.v_heads), k, nb)
}

/// The gated delta rule, one group per value head, one thread per value
/// column `j`; the state `S[h]` is `[dk][dv]`. In the CPU engine's order:
/// `s *= decay; kv = Σ s·k; δ = (v − kv)·β; s += k⊗δ; o = Σ s·(q·scale)`.
/// Bindings `proj, gates, state, out (act [nv·dv])`; groups `v_heads`.
pub fn dn_recurrence(s: DeltaShape, sg: u32) -> Emitted {
    let (dk, dv) = (s.k_dim, s.v_dim);
    let mut k = Kernel::new(dv, sg);
    let proj = k.buffer(0, Elem::Act);
    let gates = k.buffer(1, Elem::Act);
    let state = k.buffer(2, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::State));
    let out = k.buffer(3, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::Out));
    let qs = k.shared(dk, true);
    let ks = k.shared(dk, true);
    let h = k.flat_group();
    let j = k.local_id();
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dkc = k.u(dk);
    let dvc = k.u(dv);
    let key = k.u(s.key());
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let step = k.u(dv);
    k.for_(j, dkc, step, |k, i| {
        let qb = k.imul(kh, dkc);
        let qa = k.iadd(qb, i);
        let q = k.ld(proj, qa);
        let q = k.fmul(q, scale);
        k.sst(qs, i, q);
        let ka = k.iadd(qa, key);
        let kv = k.ld(proj, ka);
        k.sst(ks, i, kv);
    });
    k.barrier();
    let beta = k.ld(gates, h);
    let nv = k.u(s.v_heads);
    let di = k.iadd(h, nv);
    let decay = k.ld(gates, di);
    let decay = if ACT_BF16 { k.exp(decay) } else { decay };
    let v_at = k.u(2 * s.key());
    let hv = k.imul(h, dvc);
    let vi = k.iadd(v_at, hv);
    let vi = k.iadd(vi, j);
    let v = k.ld(proj, vi);
    let sbase = k.u(dk * dv);
    let sbase = k.imul(h, sbase);
    let sbase = k.iadd(sbase, j);
    let zero = k.f(0.0);
    let kv = k.var_f32(zero);
    let zero_u = k.u(0);
    let one_u = k.u(1);
    k.for_(zero_u, dkc, one_u, |k, d| {
        let off = k.imul(d, dvc);
        let at = k.iadd(sbase, off);
        let sv = k.ld(state, at);
        let sv = k.fmul(sv, decay);
        k.st(state, at, sv);
        let kd = k.sld(ks, d);
        let a = k.get(kv);
        let a = k.fma(sv, kd, a);
        k.set(kv, a);
    });
    let kvv = k.get(kv);
    let diff = k.fsub(v, kvv);
    let delta = k.fmul(diff, beta);
    let o = k.var_f32(zero);
    k.for_(zero_u, dkc, one_u, |k, d| {
        let off = k.imul(d, dvc);
        let at = k.iadd(sbase, off);
        let sv = k.ld(state, at);
        let kd = k.sld(ks, d);
        let sv = k.fma(kd, delta, sv);
        k.st(state, at, sv);
        let qd = k.sld(qs, d);
        let a = k.get(o);
        let a = k.fma(sv, qd, a);
        k.set(o, a);
    });
    let ov = k.get(o);
    let oi = k.iadd(hv, j);
    k.st(out, oi, ov);
    done(format!("dn_recurrence_k{}v{}", s.k_heads, s.v_heads), k, 4)
}

/// The DeltaNet's gated RMSNorm ([`dn_gated_norm`]) and its `out`
/// projection (Q4_0 by llama.cpp's masks, the residual added) in one
/// dispatch: each of `persist` groups (`nsg` simdgroups) first computes the
/// whole normed input into threadgroup memory, a simdgroup per value head and
/// the heads unrolled (the same sums in the same order as [`dn_gated_norm`],
/// the same bf16), then walks its rows as `kernels::matvec_q4_split` does
/// (`rows` a simdgroup, a row's blocks over `split` simdgroups), reading `x`
/// from there. One dispatch and one input pass less a layer.
/// Bindings `W (uint4), out (act), proj (act), w (bf16 [dv]), y (bf16), scales`;
/// groups `persist`.
#[allow(clippy::too_many_arguments)]
pub fn dn_gnorm_out_q4(s: DeltaShape, n: u32, sg: u32, nsg: u32, rows: u32, split: u32, persist: u32) -> Emitted {
    let dv = s.v_dim;
    let kdim = s.value();
    assert!(sg == 32 && dv.is_multiple_of(32) && nsg.is_multiple_of(split) && kdim.is_multiple_of(32));
    let mut k = Kernel::new(nsg * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let xo = k.buffer(1, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::Out));
    let proj = k.buffer(2, Elem::Act);
    let nw = k.buffer(3, Elem::U16);
    let y = k.buffer(4, Elem::U16);
    let sc = k.buffer(5, Elem::U16);
    let xs4 = k.shared_vec4(kdim / 4);
    let xs = k.shared_f32_view(xs4);
    let red = k.shared(nsg * rows, true);
    let sgid = k.subgroup_id();
    let lane = k.lane();
    // Phase 1: the normed input, heads `sgid + nsg·i` (< v_heads), their loads first.
    let per = s.v_heads.div_ceil(nsg);
    let nvc = k.u(s.v_heads);
    let parts = dv / 32;
    let dvc = k.u(dv);
    let z_at = k.u(s.z_at());
    let mut heads = Vec::new();
    for i in 0..per {
        let o = k.u(nsg * i);
        let h = k.iadd(sgid, o);
        let hin = k.ult(h, nvc);
        let lastv = k.u(s.v_heads - 1);
        let h = k.select_u(hin, h, lastv);
        let hb = k.imul(h, dvc);
        let mut xsv = Vec::new();
        let mut zs = Vec::new();
        let mut at = Vec::new();
        for c in 0..parts {
            let co = k.u(32 * c);
            let e = k.iadd(hb, co);
            let e = k.iadd(e, lane);
            xsv.push(k.ld(xo, e));
            let ze = k.iadd(z_at, e);
            zs.push(k.ld(proj, ze));
            at.push(e);
        }
        heads.push((xsv, zs, at, hin));
    }
    let mut wfs = Vec::new();
    for c in 0..parts {
        let co = k.u(32 * c);
        let j = k.iadd(co, lane);
        let wh = k.ld(nw, j);
        wfs.push(k.bf16_to_f(wh));
    }
    let inv_d = k.f(1.0 / dv as f32);
    let e = k.f(s.eps);
    let one = k.f(1.0);
    for (xsv, zs, at, hin) in &heads {
        // `dn_gated_norm`'s wg_reduce: a simdgroup sum per 32 columns, then in order.
        let mut ss = None;
        for &x in xsv {
            let x2 = k.fmul(x, x);
            let p = k.sg_fadd(x2);
            ss = Some(match ss {
                None => p,
                Some(a) => k.fadd(a, p),
            });
        }
        let m = k.fmul(ss.expect("parts"), inv_d);
        let m = k.fadd(m, e);
        let r = k.sqrt(m);
        let inv = k.fdiv(one, r);
        for c in 0..parts as usize {
            let sz = k.sigmoid(zs[c]);
            let silu = k.fmul(zs[c], sz);
            let nn = k.fmul(xsv[c], inv);
            let nn = k.fmul(nn, wfs[c]);
            let o = k.fmul(nn, silu);
            let b = k.round_bf16(o);
            let (hin, atc) = (*hin, at[c]);
            k.if_(hin, |k| k.sst(xs, atc, b));
        }
    }
    k.barrier();
    // Phase 2: the projection, as `matvec_q4_split` with `persist` groups.
    let slots = nsg / split;
    let splitc = k.u(split);
    let local = k.udiv(sgid, splitc);
    let sub = k.umod(sgid, splitc);
    let vgroups = n.div_ceil(slots * rows);
    let group = k.flat_group();
    let vg = k.u(vgroups);
    let step_g = k.u(persist);
    k.for_(group, vg, step_g, |k, g| {
        let sc_slots = k.u(slots);
        let base = k.imul(g, sc_slots);
        let slot = k.iadd(base, local);
        let rc = k.u(rows);
        let row0 = k.imul(slot, rc);
        let nn = k.u(n);
        let nblk = k.u(kdim / 32);
        let last = k.u(n - 1);
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                k.imul(row, nblk)
            })
            .collect();
        let sgc = k.u(sg);
        let off = k.imul(sub, sgc);
        let start = k.iadd(off, lane);
        let step = k.u(sg * split);
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
        k.for_(start, nblk, step, |k, b| {
            let eight = k.u(8);
            let xb = k.imul(b, eight);
            let mut xv = Vec::with_capacity(32);
            for i in 0..8 {
                let ii = k.u(i);
                let a = k.iadd(xb, ii);
                let v = k.sld(xs4, a);
                for c in 0..4 {
                    let w = k.comp(v, c);
                    xv.push(k.bits_to_f(w));
                }
            }
            let (scaled, sx) = super::kernels::q4_masks_x(k, &xv);
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], b);
                let wv = k.ld(w, at);
                let part = super::kernels::q4_masks_dot(k, wv, &scaled, sx);
                let dh = k.ld(sc, at);
                let d = k.f16_to_f(dh);
                let a = k.get(accs[r]);
                let a = k.fma(d, part, a);
                k.set(accs[r], a);
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        let lr = k.imul(local, rc);
        for r in 0..rows {
            let a = k.get(accs[r as usize]);
            let sum = k.sg_fadd(a);
            let rr = k.u(r);
            let i = k.iadd(lr, rr);
            let i = k.imul(i, splitc);
            let i = k.iadd(i, sub);
            k.if_(first, |k| k.sst(red, i, sum));
        }
        k.barrier();
        let head = k.ieq(sub, zero_u);
        let write0 = k.land(first, head);
        for r in 0..rows {
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(write0, in_n);
            let i = k.iadd(lr, rr);
            let i = k.imul(i, splitc);
            k.if_(write, |k| {
                let mut sum = k.sld(red, i);
                for x in 1..split {
                    let o = k.u(x);
                    let at = k.iadd(i, o);
                    let p = k.sld(red, at);
                    sum = k.fadd(sum, p);
                }
                super::kernels::store(k, y, row, sum, super::kernels::Out::Residual)
            });
        }
        k.barrier();
    });
    done(format!("dn_gnorm_out_q4_v{}_{n}_s{nsg}_r{rows}_k{split}_p{persist}", s.v_heads), k, 6)
}

/// Gated RMSNorm per value head: `y = bf16(((x·inv)·w)·silu(z))`,
/// `inv = 1/√(Σx²/dv + ε)`, `w` plain (not `1 + w`). Bindings `out (act),
/// proj (act, z), w (bf16 [dv]), y (bf16 [nv·dv])`; groups `v_heads`.
pub fn dn_gated_norm(s: DeltaShape, sg: u32) -> Emitted {
    dn_gated_norm_pf(s, sg, 0)
}

/// [`dn_gated_norm`] with `kernels::pf_head`'s warmer (bindings 4 and 5).
pub fn dn_gated_norm_pf(s: DeltaShape, sg: u32, pf: u32) -> Emitted {
    let dv = s.v_dim;
    let mut k = Kernel::new(dv, sg);
    super::kernels::pf_head(&mut k, s.v_heads, 4, pf, dv);
    let out = k.buffer(0, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::Out));
    let proj = k.buffer(1, Elem::Act);
    let w = k.buffer(2, Elem::U16);
    let y = k.buffer(3, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let h = k.flat_group();
    let j = k.local_id();
    let dvc = k.u(dv);
    let hv = k.imul(h, dvc);
    let at = k.iadd(hv, j);
    let x = k.ld(out, at);
    let x2 = k.fmul(x, x);
    let ss = k.wg_reduce(x2, scratch, false);
    let inv_d = k.f(1.0 / dv as f32);
    let e = k.f(s.eps);
    let m = k.fmul(ss, inv_d);
    let m = k.fadd(m, e);
    let r = k.sqrt(m);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    let wh = k.ld(w, j);
    let wf = k.bf16_to_f(wh);
    let z_at = k.u(s.z_at());
    let zi = k.iadd(z_at, at);
    let z = k.ld(proj, zi);
    let sz = k.sigmoid(z);
    let silu = k.fmul(z, sz);
    let n = k.fmul(x, inv);
    let n = k.fmul(n, wf);
    let o = k.fmul(n, silu);
    let b = k.f_to_bf16(o);
    k.st(y, at, b);
    let (tag, nb) = super::kernels::pf_tag(pf, 4);
    done(format!("dn_gated_norm_v{}{tag}", s.v_heads), k, nb)
}

/// The DeltaNet's decode step from its projection to the recurrence's output
/// in one dispatch: [`dn_conv`]'s conv + SiLU of the group's q, k and v
/// channels, [`dn_l2_gates`]'s L2 norms and gates, then
/// [`super::prefill::dn_recurrence_rows`] on the one row (`cols` columns of a
/// value head per group, `slices` invocations a column, the state in
/// registers). Every intermediate value is rounded as the buffer it went
/// through would round it (the same bits), and `q`, `k`, `v`, the gates are
/// not written back; the L2 norms' partials are a simdgroup's per 32
/// channels, folded in order, as `dn_l2_gates`'s `wg_reduce` over a group of
/// `dk`. The conv state is only read: [`super::kernels::Extra::ConvShift`]
/// shifts it in the `out` projection's dispatch, once every group here is
/// done. `gnorm` (`cols = dv`, a group a head): [`dn_gated_norm`] as well,
/// `y` written instead of `out`. Bindings `proj (act), conv_state (act),
/// conv_w (bf16 [ch][conv]), a_log (f32), dt_bias (f32), rec_state (act),
/// out (act [nv·dv])` or, with `gnorm`, `…, rec_state (act), w (bf16 [dv]),
/// y (bf16 [nv·dv])`; then [`super::kernels::pf_head`]'s two. Groups
/// `v_heads · dv / cols`.
pub fn dn_step(s: DeltaShape, sg: u32, cols: u32, slices: u32, gnorm: bool, pf: u32) -> Emitted {
    use crate::msl::Word;
    let (dk, dv, conv) = (s.k_dim, s.v_dim, s.conv);
    assert!(
        dv.is_multiple_of(cols) && dk.is_multiple_of(slices),
        "cols {cols} / slices {slices} do not divide {dv} / {dk}"
    );
    assert!(
        !gnorm || cols == dv,
        "dn_step with the gated norm: cols = dv"
    );
    let nrows = dk / slices;
    assert!(
        nrows <= 32,
        "{nrows} state rows per invocation: raise slices"
    );
    let wg = cols * slices;
    assert!(
        wg.is_multiple_of(32) && dk.is_multiple_of(32) && dv.is_multiple_of(32),
        "dn_step: {wg} invocations, dk {dk}, dv {dv}"
    );
    let groups = s.v_heads * dv / cols;
    let nsg = wg / 32;
    let (nq, nvb) = (dk / 32, dv / 32);
    let mut k = Kernel::new(wg, sg);
    let nb = if gnorm { 8 } else { 7 };
    super::kernels::pf_head(&mut k, groups, nb, pf, wg);
    let proj = k.buffer(0, Elem::Act);
    let cst = k.buffer(1, Elem::Act);
    let cw = k.buffer(2, Elem::U16);
    let a_log = k.buffer(3, Elem::F32);
    let dt_bias = k.buffer(4, Elem::F32);
    let state = k.buffer(5, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::State));
    let (out, nw, y) = if gnorm {
        (
            None,
            Some(k.buffer(6, Elem::U16)),
            Some(k.buffer(7, Elem::U16)),
        )
    } else {
        (Some(k.buffer(6, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::Out))), None, None)
    };
    // What an `Act` buffer would make of a value stored into it.
    let round_act = |k: &mut Kernel, v: Word| if ACT_BF16 { k.round_bf16(v) } else { v };
    let qs = k.shared(dk, true);
    let ks = k.shared(dk, true);
    let red = k.shared(wg, true);
    let deltas = k.shared(cols, true);
    // The L2 norms' partials (q then k) and their two folds; the gated norm's after.
    let scratch = k.shared((2 * nq + 2).max(nvb + 1), true);
    let g = k.flat_group();
    let t = k.local_id();
    let sgid = k.subgroup_id();
    let lane = k.lane();
    let blocks = k.u(dv / cols);
    let h = k.udiv(g, blocks);
    let cb = k.umod(g, blocks);
    let colsc = k.u(cols);
    let jl = k.umod(t, colsc);
    let sl = k.udiv(t, colsc);
    let cb_base = k.imul(cb, colsc);
    let j = k.iadd(cb_base, jl);
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dkc = k.u(dk);
    let dvc = k.u(dv);
    let hv = k.imul(h, dvc);
    let zero_u = k.u(0);
    let first_lane = k.ieq(lane, zero_u);
    // `dn_conv` of channel `c`: `silu(Σ_j state[j]·w[j])`, the current input
    // last, as `proj` would hold it.
    let conv_of = |k: &mut Kernel, c: Word| -> Word {
        let kc = k.u(conv);
        let base = k.imul(c, kc);
        let x = k.ld(proj, c);
        let mut acc = k.f(0.0);
        for jj in 0..conv {
            let v = if jj + 1 < conv {
                let jn = k.u(jj + 1);
                let at = k.iadd(base, jn);
                k.ld(cst, at)
            } else {
                x
            };
            let jc = k.u(jj);
            let wat = k.iadd(base, jc);
            let wh = k.ld(cw, wat);
            let wf = k.bf16_to_f(wh);
            acc = k.fma(v, wf, acc);
        }
        let sgm = k.sigmoid(acc);
        let yv = k.fmul(acc, sgm);
        round_act(k, yv)
    };
    // The 2·dk channels of q and k over the group: channel `t + i·wg`; a
    // simdgroup's 32 are the block `i·nsg + sgid` of `dn_l2_gates`'s partials.
    let per = (2 * dk).div_ceil(wg);
    let qbase = k.imul(kh, dkc);
    let key = k.u(s.key());
    let two_dk = k.u(2 * dk);
    let mut vals = Vec::with_capacity(per as usize);
    for i in 0..per {
        let o = k.u(i * wg);
        let c = k.iadd(t, o);
        let inside = k.ult(c, two_dk);
        let is_q = k.ult(c, dkc);
        let cq = k.iadd(qbase, c);
        let ck = k.isub(cq, dkc);
        let ck = k.iadd(ck, key);
        let ch = k.select_u(is_q, cq, ck);
        let ch = k.select_u(inside, ch, qbase);
        let v = conv_of(&mut k, ch);
        let v2 = k.fmul(v, v);
        let p = k.sg_fadd(v2);
        let sb = k.u(i * nsg);
        let sblk = k.iadd(sb, sgid);
        let two_nq = k.u(2 * nq);
        let in_blocks = k.ult(sblk, two_nq);
        let wr = k.land(first_lane, in_blocks);
        k.if_(wr, |k| k.sst(scratch, sblk, p));
        vals.push((c, inside, is_q, v));
    }
    k.barrier();
    let first_t = k.ieq(t, zero_u);
    k.if_(first_t, |k| {
        let e = k.f(1e-6);
        let one = k.f(1.0);
        for (which, base) in [(0u32, 0u32), (1, nq)] {
            let b0 = k.u(base);
            let mut acc = k.sld(scratch, b0);
            for jj in 1..nq {
                let idx = k.u(base + jj);
                let p = k.sld(scratch, idx);
                acc = k.fadd(acc, p);
            }
            let m = k.fadd(acc, e);
            let r = k.sqrt(m);
            let inv = k.fdiv(one, r);
            let slot = k.u(2 * nq + which);
            k.sst(scratch, slot, inv);
        }
    });
    k.barrier();
    let sq = k.u(2 * nq);
    let invq = k.sld(scratch, sq);
    let sk = k.u(2 * nq + 1);
    let invk = k.sld(scratch, sk);
    let scale = k.f(1.0 / (dk as f32).sqrt());
    for &(c, inside, is_q, v) in &vals {
        let inv = k.select_f(is_q, invq, invk);
        let nv = k.fmul(v, inv);
        let nv = round_act(&mut k, nv);
        let qsc = k.fmul(nv, scale);
        let ck = k.isub(c, dkc);
        let not_q = k.b_not(is_q);
        let is_k = k.land(inside, not_q);
        k.if_(is_q, |k| k.sst(qs, c, qsc));
        k.if_(is_k, |k| k.sst(ks, ck, nv));
    }
    // The gates of head `h` (`dn_l2_gates`'s), as the `gates` buffer would hold them.
    let a_at = k.u(s.a_at());
    let b_at = k.u(s.b_at());
    let ai = k.iadd(a_at, h);
    let bi = k.iadd(b_at, h);
    let a = k.ld(proj, ai);
    let b = k.ld(proj, bi);
    let beta = k.sigmoid(b);
    let beta = round_act(&mut k, beta);
    let bias = k.ld(dt_bias, h);
    let x = k.fadd(a, bias);
    let ex = k.exp(x);
    let one = k.f(1.0);
    let l = k.fadd(one, ex);
    let sp = k.log(l);
    let twenty = k.f(20.0);
    let big = k.fgt(x, twenty);
    let dt = k.select_f(big, x, sp);
    let al = k.ld(a_log, h);
    let ea = k.exp(al);
    let prod = k.fmul(ea, dt);
    let neg = k.fneg(prod);
    let stored = if ACT_BF16 { neg } else { k.exp(neg) };
    let stored = round_act(&mut k, stored);
    let decay = if ACT_BF16 { k.exp(stored) } else { stored };
    // The recurrence of `dn_recurrence_rows` on the one row.
    let rowsc = k.u(nrows);
    let kd0 = k.imul(sl, rowsc);
    let hs = k.u(dk * dv);
    let hbase = k.imul(h, hs);
    let row0 = k.imul(kd0, dvc);
    let base = k.iadd(hbase, row0);
    let base = k.iadd(base, j);
    let mut addrs = Vec::new();
    let mut regs = Vec::new();
    let mut kds = Vec::new();
    for r in 0..nrows {
        let off = k.u(r * dv);
        let at = k.iadd(base, off);
        let sv = k.ld(state, at);
        regs.push(k.var_f32(sv));
        addrs.push(at);
        let rr = k.u(r);
        kds.push(k.iadd(kd0, rr));
    }
    let rs = k.imul(sl, colsc);
    let ri = k.iadd(rs, jl);
    let first = k.ieq(sl, zero_u);
    let v_at = k.u(2 * s.key());
    k.barrier();
    let zero = k.f(0.0);
    let mut acc = zero;
    let mut svals = Vec::new();
    for r in 0..nrows as usize {
        let sv = k.get(regs[r]);
        let sv = k.fmul(sv, decay);
        let kk = k.sld(ks, kds[r]);
        acc = k.fma(sv, kk, acc);
        svals.push(sv);
    }
    k.sst(red, ri, acc);
    k.barrier();
    // The column's sum over the slices, the same for every invocation of the column.
    let fold = |k: &mut Kernel| -> Word {
        let mut sum = k.sld(red, jl);
        for x in 1..slices {
            let o = k.u(x * cols);
            let at = k.iadd(o, jl);
            let p = k.sld(red, at);
            sum = k.fadd(sum, p);
        }
        sum
    };
    k.if_(first, |k| {
        let kv = fold(k);
        let cv = k.iadd(v_at, hv);
        let cv = k.iadd(cv, j);
        let v = conv_of(k, cv);
        let diff = k.fsub(v, kv);
        let d = k.fmul(diff, beta);
        k.sst(deltas, jl, d);
    });
    k.barrier();
    let delta = k.sld(deltas, jl);
    let mut o = zero;
    for r in 0..nrows as usize {
        let kk = k.sld(ks, kds[r]);
        let sv = k.fma(kk, delta, svals[r]);
        k.set(regs[r], sv);
        let qq = k.sld(qs, kds[r]);
        o = k.fma(sv, qq, o);
    }
    k.sst(red, ri, o);
    k.barrier();
    match (out, nw, y) {
        (Some(out), _, _) => {
            k.if_(first, |k| {
                let sum = fold(k);
                let oi = k.iadd(hv, j);
                k.st(out, oi, sum);
            });
        }
        (None, Some(nw), Some(y)) => {
            // `dn_gated_norm` of head `h`: the first `dv` invocations hold its
            // columns, their simdgroups the partials of its `wg_reduce`, in order.
            let sum = fold(&mut k);
            // As `out`, scaled ([`crate::msl::set_dn_scale`]), would give it back.
            let ov = if ACT_BF16 { k.round_bf16_scaled(sum, crate::msl::dn_scale(DnScaled::Out)) } else { sum };
            let o2 = k.fmul(ov, ov);
            let p = k.sg_fadd(o2);
            let wr = k.land(first_lane, first);
            k.if_(wr, |k| k.sst(scratch, sgid, p));
            k.barrier();
            k.if_(first_t, |k| {
                let z = k.u(0);
                let mut acc = k.sld(scratch, z);
                for jj in 1..nvb {
                    let idx = k.u(jj);
                    let p = k.sld(scratch, idx);
                    acc = k.fadd(acc, p);
                }
                let inv_d = k.f(1.0 / dv as f32);
                let e = k.f(s.eps);
                let m = k.fmul(acc, inv_d);
                let m = k.fadd(m, e);
                let r = k.sqrt(m);
                let one = k.f(1.0);
                let inv = k.fdiv(one, r);
                let slot = k.u(nvb);
                k.sst(scratch, slot, inv);
            });
            k.barrier();
            let slot = k.u(nvb);
            let inv = k.sld(scratch, slot);
            k.if_(first, |k| {
                let wh = k.ld(nw, j);
                let wf = k.bf16_to_f(wh);
                let z_at = k.u(s.z_at());
                let zi = k.iadd(z_at, hv);
                let zi = k.iadd(zi, j);
                let z = k.ld(proj, zi);
                let sz = k.sigmoid(z);
                let silu = k.fmul(z, sz);
                let n = k.fmul(ov, inv);
                let n = k.fmul(n, wf);
                let o = k.fmul(n, silu);
                let b = k.f_to_bf16(o);
                let oi = k.iadd(hv, j);
                k.st(y, oi, b);
            });
        }
        _ => unreachable!(),
    }
    for r in 0..nrows as usize {
        let v = k.get(regs[r]);
        k.st(state, addrs[r], v);
    }
    let (tag, nb) = super::kernels::pf_tag(pf, nb);
    done(
        format!(
            "dn_step_k{}v{}_c{cols}s{slices}{}{tag}",
            s.k_heads,
            s.v_heads,
            if gnorm { "_gn" } else { "" }
        ),
        k,
        nb,
    )
}

/// [`dn_gated_norm`] and the DeltaNet's Q4 `out` (by the masks, the residual
/// added) in one dispatch, without recomputing the norm: groups `0..v_heads`
/// are [`dn_gated_norm`]'s (one a head, the same bits), write `y` and count
/// themselves done on `ctr[0]` after a device fence; the next
/// `matvec_split_groups(n, rows, split)` groups are `matvec_q4_split`'s (the
/// same sums), and wait for `ctr[0] = v_heads` before they read `y`. The
/// last of them resets the counters (`ctr[1]` counts them). The projection's
/// groups start while the norm runs, and the dispatch between the two is
/// gone. Relies on the norm's groups running first (they come first in the
/// grid): Metal does not promise it. Bindings `W (uint4), out (act), proj
/// (act), w (bf16 [dv]), y (bf16, coherent), x (bf16), ctr (u32 [2]),
/// scales`; groups `v_heads + matvec_split_groups(n, sg, rows, split, 0)`.
pub fn dn_gnorm_out_sync(s: DeltaShape, n: u32, sg: u32, rows: u32, split: u32) -> Emitted {
    let dv = s.v_dim;
    let kdim = s.value();
    let g = dv / sg;
    assert!(dv == 128 && sg == 32 && g.is_multiple_of(split), "dn_gnorm_out_sync: dv {dv}");
    let mut k = Kernel::new(dv, sg);
    let w = k.buffer(0, Elem::UVec4);
    let xo = k.buffer(1, Elem::Act).scaled(crate::msl::dn_scale(DnScaled::Out));
    let proj = k.buffer(2, Elem::Act);
    let nw = k.buffer(3, Elem::U16);
    let yn = k.buffer(4, Elem::U16);
    k.coherent_device(yn);
    let yn4 = k.buffer(4, Elem::UVec4);
    let xr = k.buffer(5, Elem::U16);
    let ctr = k.buffer(6, Elem::U32);
    let sc = k.buffer(7, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let red = k.shared(g * rows, true);
    let grp = k.flat_group();
    let nvc = k.u(s.v_heads);
    let is_norm = k.ult(grp, nvc);
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let mv_groups = super::kernels::matvec_split_groups(n, sg, rows, split, 0);
    k.if_else(
        is_norm,
        |k| {
            // `dn_gated_norm`, group `grp` = head `grp`.
            let j = k.local_id();
            let dvc = k.u(dv);
            let hv = k.imul(grp, dvc);
            let at = k.iadd(hv, j);
            let x = k.ld(xo, at);
            let x2 = k.fmul(x, x);
            let ss = k.wg_reduce(x2, scratch, false);
            let inv_d = k.f(1.0 / dv as f32);
            let e = k.f(s.eps);
            let m = k.fmul(ss, inv_d);
            let m = k.fadd(m, e);
            let r = k.sqrt(m);
            let one = k.f(1.0);
            let inv = k.fdiv(one, r);
            let wh = k.ld(nw, j);
            let wf = k.bf16_to_f(wh);
            let z_at = k.u(s.z_at());
            let zi = k.iadd(z_at, at);
            let z = k.ld(proj, zi);
            let sz = k.sigmoid(z);
            let silu = k.fmul(z, sz);
            let nn = k.fmul(x, inv);
            let nn = k.fmul(nn, wf);
            let o = k.fmul(nn, silu);
            let b = k.f_to_bf16(o);
            k.st(yn, at, b);
            k.fence_device();
            k.barrier();
            let first = k.ieq(j, zero_u);
            k.if_(first, |k| {
                let _ = k.dev_atomic_add(ctr, zero_u, one_u);
            });
        },
        |k| {
            let lid = k.local_id();
            let first_t = k.ieq(lid, zero_u);
            k.if_(first_t, |k| k.spin_until_ge(ctr, zero_u, nvc));
            k.barrier();
            k.fence_device();
            let group = k.isub(grp, nvc);
            let sgid = k.subgroup_id();
            let lane = k.lane();
            let splitc = k.u(split);
            let local = k.udiv(sgid, splitc);
            let sub = k.umod(sgid, splitc);
            let slots = k.u(g / split);
            let base = k.imul(group, slots);
            let slot = k.iadd(base, local);
            let rc = k.u(rows);
            let row0 = k.imul(slot, rc);
            let nn = k.u(n);
            let nblk = k.u(kdim / 32);
            let last = k.u(n - 1);
            let bases: Vec<_> = (0..rows)
                .map(|r| {
                    let rr = k.u(r);
                    let row = k.iadd(row0, rr);
                    let lt = k.ult(row, nn);
                    let row = k.select_u(lt, row, last);
                    k.imul(row, nblk)
                })
                .collect();
            let sgc = k.u(sg);
            let off = k.imul(sub, sgc);
            let start = k.iadd(off, lane);
            let step = k.u(sg * split);
            let zero = k.f(0.0);
            let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
            k.for_(start, nblk, step, |k, b| {
                let four = k.u(4);
                let xb = k.imul(b, four);
                let mut xs = Vec::with_capacity(32);
                for i in 0..4 {
                    let ii = k.u(i);
                    let at = k.iadd(xb, ii);
                    let xv = k.ld(yn4, at);
                    for c in 0..4 {
                        let xx = k.comp(xv, c);
                        xs.push(k.lo_f(xx));
                        xs.push(k.hi_f(xx));
                    }
                }
                let (scaled, sx) = super::kernels::q4_masks_x(k, &xs);
                for r in 0..rows as usize {
                    let at = k.iadd(bases[r], b);
                    let wv = k.ld(w, at);
                    let part = super::kernels::q4_masks_dot(k, wv, &scaled, sx);
                    let dh = k.ld(sc, at);
                    let d = k.f16_to_f(dh);
                    let a = k.get(accs[r]);
                    let a = k.fma(d, part, a);
                    k.set(accs[r], a);
                }
            });
            let first = k.ieq(lane, zero_u);
            let lr = k.imul(local, rc);
            for r in 0..rows {
                let a = k.get(accs[r as usize]);
                let sum = k.sg_fadd(a);
                let rr = k.u(r);
                let i = k.iadd(lr, rr);
                let i = k.imul(i, splitc);
                let i = k.iadd(i, sub);
                k.if_(first, |k| k.sst(red, i, sum));
            }
            k.barrier();
            let head = k.ieq(sub, zero_u);
            let write0 = k.land(first, head);
            for r in 0..rows {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let in_n = k.ult(row, nn);
                let write = k.land(write0, in_n);
                let i = k.iadd(lr, rr);
                let i = k.imul(i, splitc);
                k.if_(write, |k| {
                    let mut sum = k.sld(red, i);
                    for x in 1..split {
                        let o = k.u(x);
                        let at = k.iadd(i, o);
                        let p = k.sld(red, at);
                        sum = k.fadd(sum, p);
                    }
                    super::kernels::store(k, xr, row, sum, super::kernels::Out::Residual)
                });
            }
            // The last projection group resets the counters for the next dispatch.
            k.barrier();
            k.if_(first_t, |k| {
                let old = k.dev_atomic_add(ctr, one_u, one_u);
                let lastg = k.u(mv_groups - 1);
                let is_last = k.ieq(old, lastg);
                k.if_(is_last, |k| {
                    k.dev_atomic_store(ctr, zero_u, zero_u);
                    k.dev_atomic_store(ctr, one_u, zero_u);
                });
            });
        },
    );
    done(format!("dn_gnorm_out_sync_v{}_{n}_r{rows}_k{split}", s.v_heads), k, 8)
}

// ── the sparse MLP of qwen3_5_moe ────────────────────────────────────────────
//
// herbert-cuda-q4's `cuda/moe_kernels.rs` (`route_qwen`, `combine`,
// `qwen_moe_out`), rounding where `Qwen3_5MoeSparseMoeBlock` does; the
// experts' projections are the bf16 `matvec`, the expert's rows found through
// the route.

/// The route (`Qwen3_5MoeTopKRouter`): the bf16 scores' softmax in f32, the
/// `top_k` best (the lowest expert on a tie), each divided by their sum (from
/// the best down), then rounded to bf16. Written in increasing expert order —
/// the order in which the module adds the experts up — as `route[s]` the
/// expert and `route[top_k + s]` the f32 bits of its bf16 weight. Bindings
/// `scores (bf16 [rows][experts]), route (u32 [rows][2·top_k])`; one group of
/// `experts` threads per row (one for a decode step).
pub fn route_qwen(experts: u32, top_k: u32, sg: u32) -> Emitted {
    assert!(experts.is_multiple_of(sg) && top_k >= 1 && top_k <= experts && experts <= 1024);
    let mut k = Kernel::new(experts, sg);
    let scores = k.buffer(0, Elem::U16);
    let route = k.buffer(1, Elem::U32);
    let scratch = k.shared(k.subgroups() + 1, true);
    let probs = k.shared(experts, true);
    let lid = k.local_id();
    let row = k.flat_group();
    let ec = k.u(experts);
    let srow = k.imul(row, ec);
    let si = k.iadd(srow, lid);
    let sh = k.ld(scores, si);
    let k2 = k.u(2 * top_k);
    let rbase = k.imul(row, k2);
    let s = k.bf16_to_f(sh);
    let mx = k.wg_reduce(s, scratch, true);
    let d = k.fsub(s, mx);
    let ex = k.exp(d);
    let sum = k.wg_reduce(ex, scratch, false);
    let p = k.fdiv(ex, sum);
    k.sst(probs, lid, p);
    k.barrier();
    let zero = k.u(0);
    let first = k.ieq(lid, zero);
    k.if_(first, |k| {
        let mut chosen = Vec::new();
        let mut vals = Vec::new();
        for _ in 0..top_k {
            let m2 = k.f(-2.0);
            let best = k.var_f32(m2);
            let z = k.u(0);
            let idx = k.var_u32(z);
            let (z, end, one) = (k.u(0), k.u(experts), k.u(1));
            k.for_(z, end, one, |k, e| {
                let v = k.sld(probs, e);
                let b = k.get(best);
                let gt = k.fgt(v, b);
                let nb = k.select_f(gt, v, b);
                k.set(best, nb);
                let i = k.get(idx);
                let ni = k.select_u(gt, e, i);
                k.set(idx, ni);
            });
            let i = k.get(idx);
            let b = k.get(best);
            let gone = k.f(-1.0);
            k.sst(probs, i, gone);
            chosen.push(i);
            vals.push(b);
        }
        let mut total = k.f(0.0);
        for &v in &vals {
            total = k.fadd(total, v);
        }
        for i in 0..top_k as usize {
            let w = k.fdiv(vals[i], total);
            let w = k.round_bf16(w);
            let mut rank = k.u(0);
            for j in 0..top_k as usize {
                if j != i {
                    let lt = k.ult(chosen[j], chosen[i]);
                    let one = k.u(1);
                    let r1 = k.iadd(rank, one);
                    rank = k.select_u(lt, r1, rank);
                }
            }
            let ri = k.iadd(rbase, rank);
            k.st(route, ri, chosen[i]);
            let kc = k.u(top_k);
            let at = k.iadd(ri, kc);
            let wb = k.f_to_bits(w);
            k.st(route, at, wb);
        }
    });
    done(format!("route_qwen_{experts}_k{top_k}"), k, 2)
}

/// The projections of the `top_k` routed experts of every row: slot `s` of
/// row `r` (`g = r·top_k + s`) reads the rows of expert `route[r·2·top_k + s]`
/// in `W` (`[experts·n][k]` bf16), its input at `r·x_row + s·x_stride`
/// (`x_stride` 0: every slot of a row the same input) and writes `y[g·n..]` in
/// bf16; the arithmetic of `kernels::matvec`. Bindings `W (uint4), x (uint4),
/// y, route`; groups `rows · top_k · matvec_groups(n)`.
pub fn experts_matvec(kdim: u32, n: u32, top_k: u32, x_row: u32, x_stride: u32, sg: u32, rows: u32) -> Emitted {
    assert!(kdim.is_multiple_of(8) && x_stride.is_multiple_of(8) && x_row.is_multiple_of(8), "K = {kdim}, x row {x_row}, x stride {x_stride}: not multiples of 8");
    let g = (128 / sg).max(1);
    let per_slot = super::kernels::matvec_groups(n, sg, rows);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, Elem::U16);
    let route = k.buffer(3, Elem::U32);
    let group = k.flat_group();
    let ps = k.u(per_slot);
    let slot = k.udiv(group, ps);
    let local = k.umod(group, ps);
    let kk = k.u(top_k);
    let token = k.udiv(slot, kk);
    let s_in = k.umod(slot, kk);
    let k2 = k.u(2 * top_k);
    let rbase = k.imul(token, k2);
    let ri = k.iadd(rbase, s_in);
    let expert = k.ld(route, ri);
    let sgid = k.subgroup_id();
    let gc = k.u(g);
    let base = k.imul(local, gc);
    let lslot = k.iadd(base, sgid);
    let rc = k.u(rows);
    let row0 = k.imul(lslot, rc);
    let nn = k.u(n);
    let inside = k.ult(row0, nn);
    k.if_(inside, |k| {
        let chunks = k.u(kdim / 8);
        let last = k.u(n - 1);
        let e_rows = k.imul(expert, nn);
        let xs_c = k.u(x_stride / 8);
        let xr_c = k.u(x_row / 8);
        let x_tok = k.imul(token, xr_c);
        let x_slot = k.imul(s_in, xs_c);
        let x_at = k.iadd(x_tok, x_slot);
        // Rows past the end read the last row and are not written.
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                let row = k.iadd(e_rows, row);
                k.imul(row, chunks)
            })
            .collect();
        let lane = k.lane();
        let step = k.u(sg);
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
        k.for_(lane, chunks, step, |k, c| {
            let xc = k.iadd(x_at, c);
            let xv = k.ld(x, xc);
            let xs: Vec<_> = (0..4)
                .map(|i| {
                    let xx = k.comp(xv, i);
                    (k.lo_f(xx), k.hi_f(xx))
                })
                .collect();
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], c);
                let wv = k.ld(w, at);
                let mut a = k.get(accs[r]);
                for (i, &(xl, xh)) in xs.iter().enumerate() {
                    let ww = k.comp(wv, i as u32);
                    let wl = k.lo_f(ww);
                    a = k.fma(wl, xl, a);
                    let wh = k.hi_f(ww);
                    a = k.fma(wh, xh, a);
                }
                k.set(accs[r], a);
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        let y_at = k.imul(slot, nn);
        for r in 0..rows {
            let a = k.get(accs[r as usize]);
            let sum = k.sg_fadd(a);
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(first, in_n);
            k.if_(write, |k| {
                let b = k.f_to_bf16(sum);
                let at = k.iadd(y_at, row);
                k.st(y, at, b);
            });
        }
    });
    done(format!("experts_matvec_{kdim}x{n}_k{top_k}_xr{x_row}_xs{x_stride}_r{rows}"), k, 4)
}

/// [`experts_matvec`] with `W` in Q4_0 (one `q4::quantize` of the
/// `experts·n` rows), by llama.cpp's masks (`kernels::matvec_q4s`): the lanes
/// over the blocks of 32. Bindings `W (uint4), x (uint4), y, route, scales`;
/// the groups of [`experts_matvec`].
pub fn experts_matvec_q4(kdim: u32, n: u32, top_k: u32, x_row: u32, x_stride: u32, sg: u32, rows: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32) && x_stride.is_multiple_of(8) && x_row.is_multiple_of(8), "K = {kdim} (Q4: a multiple of 32), x row {x_row}, x stride {x_stride}");
    let g = (128 / sg).max(1);
    let per_slot = super::kernels::matvec_groups(n, sg, rows);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, Elem::U16);
    let route = k.buffer(3, Elem::U32);
    let sc = k.buffer(4, Elem::U16);
    let group = k.flat_group();
    let ps = k.u(per_slot);
    let slot = k.udiv(group, ps);
    let local = k.umod(group, ps);
    let kk = k.u(top_k);
    let token = k.udiv(slot, kk);
    let s_in = k.umod(slot, kk);
    let k2 = k.u(2 * top_k);
    let rbase = k.imul(token, k2);
    let ri = k.iadd(rbase, s_in);
    let expert = k.ld(route, ri);
    let sgid = k.subgroup_id();
    let gc = k.u(g);
    let base = k.imul(local, gc);
    let lslot = k.iadd(base, sgid);
    let rc = k.u(rows);
    let row0 = k.imul(lslot, rc);
    let nn = k.u(n);
    let inside = k.ult(row0, nn);
    k.if_(inside, |k| {
        let nblk = k.u(kdim / 32);
        let last = k.u(n - 1);
        let e_rows = k.imul(expert, nn);
        let xs_c = k.u(x_stride / 8);
        let xr_c = k.u(x_row / 8);
        let x_tok = k.imul(token, xr_c);
        let x_slot = k.imul(s_in, xs_c);
        let x_at = k.iadd(x_tok, x_slot);
        // Rows past the end read the last row and are not written.
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                let row = k.iadd(e_rows, row);
                k.imul(row, nblk)
            })
            .collect();
        let lane = k.lane();
        let step = k.u(sg);
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
        k.for_(lane, nblk, step, |k, b| {
            let four = k.u(4);
            let xb = k.imul(b, four);
            let xb = k.iadd(x_at, xb);
            let mut xs = Vec::with_capacity(32);
            for i in 0..4 {
                let ii = k.u(i);
                let at = k.iadd(xb, ii);
                let xv = k.ld(x, at);
                for c in 0..4 {
                    let xx = k.comp(xv, c);
                    xs.push(k.lo_f(xx));
                    xs.push(k.hi_f(xx));
                }
            }
            let (scaled, sum_x) = super::kernels::q4_masks_x(k, &xs);
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], b);
                let wv = k.ld(w, at);
                let part = super::kernels::q4_masks_dot(k, wv, &scaled, sum_x);
                let dh = k.ld(sc, at);
                let d = k.f16_to_f(dh);
                let a = k.get(accs[r]);
                let a = k.fma(d, part, a);
                k.set(accs[r], a);
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        let y_at = k.imul(slot, nn);
        for r in 0..rows {
            let a = k.get(accs[r as usize]);
            let sum = k.sg_fadd(a);
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(first, in_n);
            k.if_(write, |k| {
                let b = k.f_to_bf16(sum);
                let at = k.iadd(y_at, row);
                k.st(y, at, b);
            });
        }
    });
    done(format!("experts_matvec_q4_{kdim}x{n}_k{top_k}_xr{x_row}_xs{x_stride}_r{rows}"), k, 5)
}

/// [`swiglu`] over `slots` rows of `[gate (n) | up (n)]`: `h[s·n + i]`.
/// Bindings `gate_up, h`; groups `ceil(slots·n / 256)`.
pub fn swiglu_slots(n: u32, slots: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let gu = k.buffer(0, Elem::U16);
    let h = k.buffer(1, Elem::U16);
    let i = k.global_id();
    let total = k.u(slots * n);
    let inside = k.ult(i, total);
    k.if_(inside, |k| {
        let nn = k.u(n);
        let s = k.udiv(i, nn);
        let j = k.umod(i, nn);
        let two_n = k.u(2 * n);
        let row = k.imul(s, two_n);
        let gi = k.iadd(row, j);
        let ui = k.iadd(gi, nn);
        let gh = k.ld(gu, gi);
        let uh = k.ld(gu, ui);
        let g = k.bf16_to_f(gh);
        let u = k.bf16_to_f(uh);
        let sg_ = k.sigmoid(g);
        let gs = k.fmul(g, sg_);
        let o = k.fmul(gs, u);
        let b = k.f_to_bf16(o);
        k.st(h, i, b);
    });
    done(format!("swiglu_{n}_s{slots}"), k, 2)
}

/// The experts added up as `index_add_` does into a bf16 zero vector, in the
/// route's (increasing expert) order: `acc = bf16(acc + bf16(y·w))`.
/// Bindings `y (bf16 [rows][top_k][d]), route, out (bf16 [rows][d])`; groups
/// `rows · ceil(d / 256)`.
pub fn moe_combine(d: u32, top_k: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let y = k.buffer(0, Elem::U16);
    let route = k.buffer(1, Elem::U32);
    let out = k.buffer(2, Elem::U16);
    let g = k.flat_group();
    let gpr = k.u(d.div_ceil(256));
    let row = k.udiv(g, gpr);
    let gi = k.umod(g, gpr);
    let wg = k.u(256);
    let base = k.imul(gi, wg);
    let lid = k.local_id();
    let i = k.iadd(base, lid);
    let dn = k.u(d);
    let inside = k.ult(i, dn);
    k.if_(inside, |k| {
        let kd = k.u(top_k * d);
        let yrow = k.imul(row, kd);
        let k2 = k.u(2 * top_k);
        let rbase = k.imul(row, k2);
        let mut acc = k.f(0.0);
        for s in 0..top_k {
            let off = k.u(s * d);
            let at = k.iadd(i, off);
            let at = k.iadd(at, yrow);
            let yh = k.ld(y, at);
            let yf = k.bf16_to_f(yh);
            let wi = k.u(top_k + s);
            let wi = k.iadd(rbase, wi);
            let wb = k.ld(route, wi);
            let wf = k.bits_to_f(wb);
            let c = k.fmul(yf, wf);
            let c = k.round_bf16(c);
            let a = k.fadd(acc, c);
            acc = k.round_bf16(a);
        }
        let b = k.f_to_bf16(acc);
        let orow = k.imul(row, dn);
        let oi = k.iadd(orow, i);
        k.st(out, oi, b);
    });
    done(format!("moe_combine_{d}_k{top_k}"), k, 3)
}

/// The end of `Qwen3_5MoeSparseMoeBlock` and the residual add:
/// `g = bf16(σ(bf16(gate·xn)))`, `x = bf16(x + bf16(experts + bf16(g·shared)))`.
/// Bindings `xn, gate (bf16 [d]), experts, shared, x` (`[rows][d]` but the
/// gate); one group per row.
pub fn moe_out(d: u32, sg: u32) -> Emitted {
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let xn = k.buffer(0, Elem::U16);
    let gw = k.buffer(1, Elem::U16);
    let ex = k.buffer(2, Elem::U16);
    let sh = k.buffer(3, Elem::U16);
    let x = k.buffer(4, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let lid = k.local_id();
    let (dn, step) = (k.u(d), k.u(wg));
    let row = k.flat_group();
    let base = k.imul(row, dn);
    let zero = k.f(0.0);
    let acc = k.var_f32(zero);
    k.for_(lid, dn, step, |k, i| {
        let ri = k.iadd(base, i);
        let h = k.ld(xn, ri);
        let v = k.bf16_to_f(h);
        let wh = k.ld(gw, i);
        let wf = k.bf16_to_f(wh);
        let a = k.get(acc);
        let a = k.fma(v, wf, a);
        k.set(acc, a);
    });
    let a = k.get(acc);
    let dot = k.wg_reduce(a, scratch, false);
    let logit = k.round_bf16(dot);
    let g = k.sigmoid(logit);
    let g = k.round_bf16(g);
    k.for_(lid, dn, step, |k, i| {
        let ri = k.iadd(base, i);
        let shh = k.ld(sh, ri);
        let sf = k.bf16_to_f(shh);
        let s = k.fmul(g, sf);
        let s = k.round_bf16(s);
        let eh = k.ld(ex, ri);
        let ef = k.bf16_to_f(eh);
        let o = k.fadd(ef, s);
        let o = k.round_bf16(o);
        let xh = k.ld(x, ri);
        let xf = k.bf16_to_f(xh);
        let y = k.fadd(xf, o);
        let yb = k.f_to_bf16(y);
        k.st(x, ri, yb);
    });
    done(format!("moe_out_{d}"), k, 5)
}
