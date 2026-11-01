//! The kernels shared by every model: the projection of one token, the
//! RMSNorm, the two passes of the argmax. Ported from herbert-vk-bf16's
//! `vk/kernels.rs`, same arithmetic, same rounding points.

use crate::msl::{done, Buf, Elem, Emitted, Kernel, Word};

/// The decode state (`state` buffer, `u32`): the token fed, its position, a
/// flag (Gemma: `r + 1` for row `r` of the image buffer, unused here) or the
/// argmax, the end of the visible range.
pub const STATE_TOKEN: u32 = 0;
pub const STATE_POS: u32 = 1;
pub const STATE_ARGMAX: u32 = 2;
/// Qwen: the rows of a prefill block (`1` for a decode step); Gemma: the end
/// of the visible range.
pub const STATE_ROWS: u32 = 3;

/// The position of the token fed.
pub(crate) fn pos(k: &mut Kernel, state: Buf) -> Word {
    let i = k.u(STATE_POS);
    k.ld(state, i)
}

/// Simdgroups per threadgroup for the row-per-simdgroup kernels: 128 threads.
fn rows_per_group(sg: u32) -> u32 {
    (128 / sg).max(1)
}

/// Where a projection writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// `y[n] = bf16(acc)`.
    Bf16,
    /// `y[n] = acc` in f32.
    F32,
    /// `y[n] = bf16(y[n] + bf16(acc))`: the projection and the residual add.
    Residual,
    /// The MLP's `[gate | up]` projection and its SwiGLU in one: a simdgroup's
    /// rows are `rows / 2` gate rows and their up rows (`r` and `n / 2 + r`),
    /// `y[r] = bf16(g·σ(g)·u)` of the two bf16-rounded sums (`swiglu`'s bits),
    /// `y` of `n / 2`. The split kernels only ([`matvec_q4_split`]).
    Swiglu,
}

/// RMSNorm of one vector of `d`: `y = bf16((x·inv)·(1 + w))` (`plus_one`, the
/// convention of every Qwen3.5 norm) or `·w` (Gemma-4),
/// `inv = 1/√(Σx²/d + ε)`. Bindings `x, w, y` (bf16); one group, or one group
/// per row of a block with `x`, `y` rows `sx`, `sy` apart.
pub fn rmsnorm(d: u32, eps: f32, plus_one: bool, sg: u32, sx: u32, sy: u32) -> Emitted {
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let x = k.buffer(0, Elem::U16);
    let w = k.buffer(1, Elem::U16);
    let y = k.buffer(2, Elem::U16);
    if sx + sy > 0 {
        let row = k.flat_group();
        for (b, s) in [(x, sx), (y, sy)] {
            let st = k.u(s);
            let o = k.imul(row, st);
            k.advance(b, o);
        }
    }
    let scratch = k.shared(k.subgroups() + 1, true);
    let lid = k.local_id();
    if d.is_multiple_of(wg) && d / wg <= 32 {
        // The invocation's `d / wg` values loaded at once and kept: the same
        // sums in the same order, without a load's latency per value.
        let n = d / wg;
        let mut xs = Vec::with_capacity(n as usize);
        let mut at = Vec::with_capacity(n as usize);
        for i in 0..n {
            let o = k.u(i * wg);
            let a = k.iadd(lid, o);
            let h = k.ld(x, a);
            xs.push(h);
            at.push(a);
        }
        let mut acc = k.f(0.0);
        let vs: Vec<Word> = xs.iter().map(|&h| k.bf16_to_f(h)).collect();
        for &v in &vs {
            acc = k.fma(v, v, acc);
        }
        let ss = k.wg_reduce(acc, scratch, false);
        let inv_d = k.f(1.0 / d as f32);
        let e = k.f(eps);
        let m = k.fmul(ss, inv_d);
        let m = k.fadd(m, e);
        let r = k.sqrt(m);
        let one = k.f(1.0);
        let inv = k.fdiv(one, r);
        let ws: Vec<Word> = at.iter().map(|&a| k.ld(w, a)).collect();
        for i in 0..n as usize {
            let mut wf = k.bf16_to_f(ws[i]);
            if plus_one {
                let one = k.f(1.0);
                wf = k.fadd(wf, one);
            }
            let nn = k.fmul(vs[i], inv);
            let o = k.fmul(nn, wf);
            let b = k.f_to_bf16(o);
            k.st(y, at[i], b);
        }
        let rows = if sx + sy > 0 { format!("_r{sx}_{sy}") } else { String::new() };
        return done(format!("rmsnorm_{d}{}{rows}_u", if plus_one { "_p1" } else { "" }), k, 3);
    }
    let (dn, step) = (k.u(d), k.u(wg));
    let zero_f = k.f(0.0);
    let acc = k.var_f32(zero_f);
    k.for_(lid, dn, step, |k, i| {
        let h = k.ld(x, i);
        let v = k.bf16_to_f(h);
        let a = k.get(acc);
        let s = k.fma(v, v, a);
        k.set(acc, s);
    });
    let a = k.get(acc);
    let ss = k.wg_reduce(a, scratch, false);
    let inv_d = k.f(1.0 / d as f32);
    let e = k.f(eps);
    let m = k.fmul(ss, inv_d);
    let m = k.fadd(m, e);
    let r = k.sqrt(m);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    k.for_(lid, dn, step, |k, i| {
        let h = k.ld(x, i);
        let v = k.bf16_to_f(h);
        let wh = k.ld(w, i);
        let mut wf = k.bf16_to_f(wh);
        if plus_one {
            let one = k.f(1.0);
            wf = k.fadd(wf, one);
        }
        let n = k.fmul(v, inv);
        let o = k.fmul(n, wf);
        let b = k.f_to_bf16(o);
        k.st(y, i, b);
    });
    let rows = if sx + sy > 0 { format!("_r{sx}_{sy}") } else { String::new() };
    done(format!("rmsnorm_{d}{}{rows}", if plus_one { "_p1" } else { "" }), k, 3)
}

/// `y = W·x` written as `out` says, `W` `[n][k]` bf16 row-major, `x` `[k]`
/// bf16, f32 accumulation. Each simdgroup reduces `rows` consecutive rows of
/// `W`, sharing the loads of `x`; the lanes stride over `k` by 8 bf16 (16
/// bytes). Bindings `W (uint4), x (uint4), y`; groups [`matvec_groups`].
pub fn matvec(kdim: u32, n: u32, out: Out, sg: u32, rows: u32) -> Emitted {
    assert!(kdim.is_multiple_of(8), "K = {kdim} is not a multiple of 8");
    let g = rows_per_group(sg);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
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
        // Rows past the end read the last row and are not written.
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
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
        k.for_(lane, chunks, step, |k, c| {
            let xv = k.ld(x, c);
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
        for r in 0..rows {
            let a = k.get(accs[r as usize]);
            let sum = k.sg_fadd(a);
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(first, in_n);
            k.if_(write, |k| store(k, y, row, sum, out));
        }
    });
    done(format!("matvec_{kdim}x{n}{}_r{rows}", out_tag(out)), k, 3)
}

/// `y[row]` from the f32 sum of a projection, as `out` says.
pub(crate) fn store(k: &mut Kernel, y: Buf, row: Word, sum: Word, out: Out) {
    match out {
        Out::F32 => k.st(y, row, sum),
        Out::Bf16 => {
            let b = k.f_to_bf16(sum);
            k.st(y, row, b);
        }
        Out::Residual => {
            let r = k.round_bf16(sum);
            let old = k.ld(y, row);
            let of = k.bf16_to_f(old);
            let s = k.fadd(of, r);
            let b = k.f_to_bf16(s);
            k.st(y, row, b);
        }
        Out::Swiglu => unreachable!("Out::Swiglu: two sums, stored by matvec_split_impl"),
    }
}

/// What a projection's `x` binding holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XSrc {
    /// `x` itself, bf16 `[k]`.
    Plain,
    /// The MLP's `[gate (k) | up (k)]` (bf16): `x = bf16(g·σ(g)·u)`, the
    /// `swiglu` of `qwen_kernels` done where `x` is loaded (the same bits).
    Swiglu,
    /// llama.cpp's masks prepared ahead ([`q4_prep_x`], Q4 by the masks
    /// only): `x·2^-s` in f32 `[k]` at `x`, the blocks' `Σx` at binding 4.
    Prepped,
    /// `x` is the residual stream (bf16 `[k]`) and the projection's input is
    /// its RMSNorm: each group computes [`rmsnorm`] (`eps` the f32's bits,
    /// `plus_one`; the same sums in the same order, the same bf16) into
    /// threadgroup memory first, the weights `w` at the binding after the
    /// matvec's. One dispatch less a norm.
    Norm { eps: u32, plus_one: bool },
}

/// `bf16(g·σ(g)·u)` as f32, `swiglu`'s arithmetic.
fn swiglu_one(k: &mut Kernel, g: Word, u: Word) -> Word {
    let s = k.sigmoid(g);
    let gs = k.fmul(g, s);
    let o = k.fmul(gs, u);
    k.round_bf16(o)
}

/// The eight bf16 of `xv` (a `uint4`) as f32, in order; with `XSrc::Swiglu`
/// those of `bf16(g·σ(g)·u)`, `g` from `xv` and `u` from `uv`.
fn x_unit(k: &mut Kernel, xv: Word, uv: Option<Word>) -> Vec<Word> {
    let mut v = Vec::with_capacity(8);
    for c in 0..4 {
        let xx = k.comp(xv, c);
        let (lo, hi) = (k.lo_f(xx), k.hi_f(xx));
        match uv {
            None => {
                v.push(lo);
                v.push(hi);
            }
            Some(uv) => {
                let uu = k.comp(uv, c);
                let (ul, uh) = (k.lo_f(uu), k.hi_f(uu));
                v.push(swiglu_one(k, lo, ul));
                v.push(swiglu_one(k, hi, uh));
            }
        }
    }
    v
}

fn x_tag(xs: XSrc) -> &'static str {
    match xs {
        XSrc::Plain => "",
        XSrc::Swiglu => "_swi",
        XSrc::Prepped => "_pre",
        XSrc::Norm { .. } => "_nrm",
    }
}

fn out_tag(out: Out) -> &'static str {
    match out {
        Out::Bf16 => "",
        Out::F32 => "_f32",
        Out::Residual => "_res",
        Out::Swiglu => "_sw",
    }
}

/// Extra groups appended to a projection's dispatch, past
/// [`matvec_split_groups`]: a small kernel's work done in the matvec's wake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extra {
    None,
    /// The DeltaNet's causal conv state shifted for its `ch` channels of
    /// `conv` taps, `state[c] = [state[c][1..], proj[c]]`, a thread a channel:
    /// what `dn_conv` writes, done once every group of `dn_step` has read the
    /// state. Bindings `proj (act), conv_state (act)` after the matvec's.
    ConvShift {
        ch: u32,
        conv: u32,
    },
}

fn extra_tag(extra: Extra) -> &'static str {
    match extra {
        Extra::None => "",
        Extra::ConvShift { .. } => "_cs",
    }
}

/// The groups of a split matvec of `n` rows written as `out` (half the rows
/// for [`Out::Swiglu`]), plus those of `extra`.
pub fn matvec_full_groups(
    n: u32,
    out: Out,
    sg: u32,
    rows: u32,
    split: u32,
    persist: u32,
    extra: Extra,
) -> u32 {
    let (n, rows) = if out == Out::Swiglu {
        (n / 2, rows / 2)
    } else {
        (n, rows)
    };
    let wg = rows_per_group(sg) * sg;
    matvec_split_groups(n, sg, rows, split, persist)
        + match extra {
            Extra::None => 0,
            Extra::ConvShift { ch, .. } => ch.div_ceil(wg),
        }
}

/// [`rmsnorm`]'s `_u` path over `x` of `d` (a multiple of 256, `d / 256 ≤
/// 32`), by a group of `wg` threads (`256 / wg` of `rmsnorm`'s per thread):
/// the same sums in the same order (a simdgroup's partial per virtual
/// simdgroup, folded in order), the normed bf16 into `xs` (a
/// [`Kernel::shared_u16`] of `d`), a barrier after.
fn norm_into(
    k: &mut Kernel,
    x: Buf,
    w: Buf,
    xs: crate::msl::Shared,
    d: u32,
    eps: f32,
    plus_one: bool,
    wg: u32,
) {
    assert!(
        d.is_multiple_of(256)
            && d / 256 <= 32
            && 256u32.is_multiple_of(wg)
            && wg.is_multiple_of(32),
        "fused norm: d = {d}, group of {wg}"
    );
    let per = 256 / wg;
    let n = d / 256;
    let gsub = wg / 32;
    let scratch = k.shared(9, true);
    let lid = k.local_id();
    let sgid = k.subgroup_id();
    let lane = k.lane();
    let mut at = Vec::new();
    let mut vs = Vec::new();
    let mut accs = Vec::new();
    for v in 0..per {
        let mut hs = Vec::with_capacity(n as usize);
        for i in 0..n {
            let o = k.u(v * wg + i * 256);
            let a = k.iadd(lid, o);
            hs.push(k.ld(x, a));
            at.push(a);
        }
        let mut acc = k.f(0.0);
        for h in hs {
            let f = k.bf16_to_f(h);
            acc = k.fma(f, f, acc);
            vs.push(f);
        }
        accs.push(acc);
    }
    let zero = k.u(0);
    let first_lane = k.ieq(lane, zero);
    for (v, &acc) in accs.iter().enumerate() {
        let p = k.sg_fadd(acc);
        let o = k.u(v as u32 * gsub);
        let slot = k.iadd(sgid, o);
        k.if_(first_lane, |k| k.sst(scratch, slot, p));
    }
    k.barrier();
    let first = k.ieq(lid, zero);
    k.if_(first, |k| {
        let z = k.u(0);
        let mut acc = k.sld(scratch, z);
        for j in 1..8 {
            let jj = k.u(j);
            let p = k.sld(scratch, jj);
            acc = k.fadd(acc, p);
        }
        let slot = k.u(8);
        k.sst(scratch, slot, acc);
    });
    k.barrier();
    let slot = k.u(8);
    let ss = k.sld(scratch, slot);
    let inv_d = k.f(1.0 / d as f32);
    let e = k.f(eps);
    let m = k.fmul(ss, inv_d);
    let m = k.fadd(m, e);
    let r = k.sqrt(m);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    let ws: Vec<Word> = at.iter().map(|&a| k.ld(w, a)).collect();
    for i in 0..at.len() {
        let mut wf = k.bf16_to_f(ws[i]);
        if plus_one {
            let one = k.f(1.0);
            wf = k.fadd(wf, one);
        }
        let nn = k.fmul(vs[i], inv);
        let o = k.fmul(nn, wf);
        let b = k.f_to_bf16(o);
        k.sst(xs, at[i], b);
    }
    k.barrier();
}

/// `y = W·x` written as `out` says, with `W` in Q4_0 (`q4::quantize`): `[n][k/32]` blocks of
/// one `uint4` each, a two's complement nibble per weight (byte `b` of word
/// `i`: weight `8i + b` low, `8i + 4 + b` high), and the f16 scales `[n][k/32]`
/// in a binding of their own. Per block, `Σ q·x` in f32, then `acc += d·Σ`.
/// Each simdgroup reduces `rows` consecutive rows, the lanes stride over the
/// blocks. Bindings `W (uint4), x (uint4), y, scales`; groups [`matvec_groups`].
/// `out` as [`matvec`]'s.
pub fn matvec_q4(kdim: u32, n: u32, out: Out, sg: u32, rows: u32) -> Emitted {
    matvec_q4_impl(kdim, n, out, sg, rows, false, 1)
}

/// [`matvec_q4`] by llama.cpp's masks: the word's nibbles made offset binary
/// (`⊕ 0x88888888`, `q + 8`), each taken in place (`w & 15·2^s`, exact in
/// f32) against `x·2^-s` (exact: a power of two), the bias out of the sum,
/// per block `d·(Σ (q+8)·x − 8·Σx)`; three operations a weight instead of
/// six, the sums in another order.
pub fn matvec_q4s(kdim: u32, n: u32, out: Out, sg: u32, rows: u32) -> Emitted {
    matvec_q4_impl(kdim, n, out, sg, rows, true, 1)
}

/// [`matvec_q4`] (`masks` false) or [`matvec_q4s`] with the blocks of each
/// row shared by `split` simdgroups of the group (block `b` to simdgroup
/// `(b / sg) % split`), their sums added in threadgroup memory in simdgroup
/// order: `split` times the simdgroups on a matrix of few rows, the sums in
/// another order. `persist` > 0: only that many groups, each taking the
/// rows of a group after another (the last ones finishing together). Groups
/// [`matvec_split_groups`].
#[allow(clippy::too_many_arguments)]
pub fn matvec_q4_split(
    kdim: u32,
    n: u32,
    out: Out,
    sg: u32,
    rows: u32,
    masks: bool,
    split: u32,
    persist: u32,
    xsrc: XSrc,
    extra: Extra,
) -> Emitted {
    assert!(rows_per_group(sg).is_multiple_of(split), "split {split}");
    matvec_split_impl(
        kdim,
        n,
        out,
        sg,
        rows,
        Some(masks),
        split,
        persist,
        xsrc,
        extra,
    )
}

/// [`matvec`] with the forms of [`matvec_q4_split`]. Groups
/// [`matvec_split_groups`].
#[allow(clippy::too_many_arguments)]
pub fn matvec_split(
    kdim: u32,
    n: u32,
    out: Out,
    sg: u32,
    rows: u32,
    split: u32,
    persist: u32,
    xsrc: XSrc,
    extra: Extra,
) -> Emitted {
    assert!(
        kdim.is_multiple_of(8) && rows_per_group(sg).is_multiple_of(split),
        "K = {kdim}, split {split}"
    );
    matvec_split_impl(kdim, n, out, sg, rows, None, split, persist, xsrc, extra)
}

/// `ceil(n / (rows · simdgroups per group / split))`, or `persist`.
pub fn matvec_split_groups(n: u32, sg: u32, rows: u32, split: u32, persist: u32) -> u32 {
    let g = n.div_ceil(rows_per_group(sg) / split * rows);
    if persist > 0 { persist.min(g) } else { g }
}

fn matvec_q4_impl(kdim: u32, n: u32, out: Out, sg: u32, rows: u32, masks: bool, split: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32), "Q4: K = {kdim} is not a multiple of 32");
    let g = rows_per_group(sg);
    assert!(split >= 1 && g.is_multiple_of(split), "split {split} of {g} simdgroups");
    if split > 1 {
        return matvec_split_impl(kdim, n, out, sg, rows, Some(masks), split, 0, XSrc::Plain, Extra::None);
    }
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
    let sc = k.buffer(3, Elem::U16);
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
        let nblk = k.u(kdim / 32);
        let last = k.u(n - 1);
        // Rows past the end read the last row and are not written.
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                k.imul(row, nblk)
            })
            .collect();
        let lane = k.lane();
        let step = k.u(sg);
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
        k.for_(lane, nblk, step, |k, b| {
            // The 32 activations of the block, in order.
            let four = k.u(4);
            let xb = k.imul(b, four);
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
            let (eight, fifteen, eight_f) = (k.u(8), k.u(15), k.f(8.0));
            let (scaled, sum_x) = if masks {
                let (sc, sx) = q4_masks_x(k, &xs);
                (sc, Some(sx))
            } else {
                (Vec::new(), None)
            };
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], b);
                let wv = k.ld(w, at);
                let mut part = k.f(0.0);
                if let Some(sx) = sum_x {
                    part = q4_masks_dot(k, wv, &scaled, sx);
                    let dh = k.ld(sc, at);
                    let d = k.f16_to_f(dh);
                    let a = k.get(accs[r]);
                    let a = k.fma(d, part, a);
                    k.set(accs[r], a);
                    continue;
                }
                for i in 0..4u32 {
                    let ww = k.comp(wv, i);
                    for byte in 0..4u32 {
                        for (shift, e) in [(8 * byte, 8 * i + byte), (8 * byte + 4, 8 * i + 4 + byte)] {
                            let sh = k.u(shift);
                            let v = k.shr(ww, sh);
                            let nib = k.and(v, fifteen);
                            let biased = k.xor(nib, eight);
                            let qf = k.u2f(biased);
                            let q = k.fsub(qf, eight_f);
                            part = k.fma(q, xs[e as usize], part);
                        }
                    }
                }
                let dh = k.ld(sc, at);
                let d = k.f16_to_f(dh);
                let a = k.get(accs[r]);
                let a = k.fma(d, part, a);
                k.set(accs[r], a);
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        for r in 0..rows {
            let a = k.get(accs[r as usize]);
            let sum = k.sg_fadd(a);
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(first, in_n);
            k.if_(write, |k| store(k, y, row, sum, out));
        }
    });
    done(format!("matvec_q4{}_{kdim}x{n}{}_r{rows}", if masks { "s" } else { "" }, out_tag(out)), k, 4)
}

/// The split and persistent forms of [`matvec`] (`q4` None) and of
/// [`matvec_q4`] / [`matvec_q4s`] (`q4` the masks): the units of a row (8
/// bf16, or a Q4_0 block) shared by `split` simdgroups, unit `u` to
/// simdgroup `(u / sg) % split`.
#[allow(clippy::too_many_arguments)]
fn matvec_split_impl(
    kdim: u32,
    n: u32,
    out: Out,
    sg: u32,
    rows: u32,
    q4: Option<bool>,
    split: u32,
    persist: u32,
    xsrc: XSrc,
    extra: Extra,
) -> Emitted {
    let masks = q4 == Some(true);
    let g = rows_per_group(sg);
    let slots = g / split;
    let wg = g * sg;
    let norm = matches!(xsrc, XSrc::Norm { .. });
    // `Out::Swiglu`: a simdgroup's `rows` are `rows / 2` gate rows and their up rows.
    let pair = out == Out::Swiglu;
    assert!(!pair || rows.is_multiple_of(2), "Out::Swiglu: rows {rows}");
    let half = if pair { rows / 2 } else { rows };
    let n_out = if pair { n / 2 } else { n };
    let mut k = Kernel::new(wg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, if norm { Elem::U16 } else { Elem::UVec4 });
    let y = k.buffer(
        2,
        if out == Out::F32 {
            Elem::F32
        } else {
            Elem::U16
        },
    );
    let sc = if q4.is_some() {
        Some(k.buffer(3, Elem::U16))
    } else {
        None
    };
    assert!(
        xsrc != XSrc::Prepped || masks,
        "XSrc::Prepped: Q4 by the masks"
    );
    let mut nb = if q4.is_some() { 4 } else { 3 };
    let sums = if xsrc == XSrc::Prepped {
        nb += 1;
        Some(k.buffer(nb - 1, Elem::F32))
    } else {
        None
    };
    let nw = if norm {
        nb += 1;
        Some(k.buffer(nb - 1, Elem::U16))
    } else {
        None
    };
    // `persist` > 0: that many groups, each walking the slots' groups
    // `group, group + persist, …` (the same count for every invocation of a
    // group, so the barriers stay uniform).
    let vgroups = n_out.div_ceil(slots * half);
    let real = if persist > 0 {
        persist.min(vgroups)
    } else {
        vgroups
    };
    let group = k.flat_group();
    // `Extra::ConvShift`: the groups past the matvec's shift the conv state and leave.
    if let Extra::ConvShift { ch, conv } = extra {
        let proj = k.buffer(nb, Elem::Act);
        let st = k.buffer(nb + 1, Elem::Act);
        nb += 2;
        let rc = k.u(real);
        let below = k.ult(group, rc);
        let is_x = k.b_not(below);
        k.if_(is_x, |k| {
            let gr = k.isub(group, rc);
            let wgc = k.u(wg);
            let lid = k.local_id();
            let c = k.imul(gr, wgc);
            let c = k.iadd(c, lid);
            let chc = k.u(ch);
            let inside = k.ult(c, chc);
            k.if_(inside, |k| {
                let kc = k.u(conv);
                let base = k.imul(c, kc);
                let xv = k.ld(proj, c);
                let mut vals = Vec::with_capacity(conv as usize);
                for j in 0..conv {
                    vals.push(if j + 1 < conv {
                        let jn = k.u(j + 1);
                        let at = k.iadd(base, jn);
                        k.ld(st, at)
                    } else {
                        xv
                    });
                }
                for (j, &v) in vals.iter().enumerate() {
                    let jc = k.u(j as u32);
                    let at = k.iadd(base, jc);
                    k.st(st, at, v);
                }
            });
            k.ret();
        });
    }
    let red = k.shared(g * rows, true);
    // `XSrc::Norm`: the normed input into threadgroup memory, read from there.
    let xs4 = nw.map(|nw| {
        let XSrc::Norm { eps, plus_one } = xsrc else {
            unreachable!()
        };
        let xs16 = k.shared_u16(kdim);
        let view = k.shared_vec4_view(xs16);
        norm_into(&mut k, x, nw, xs16, kdim, f32::from_bits(eps), plus_one, wg);
        view
    });
    let ldx = |k: &mut Kernel, at: Word| -> Word {
        match xs4 {
            Some(s) => k.sld(s, at),
            None => k.ld(x, at),
        }
    };
    let sgid = k.subgroup_id();
    let splitc = k.u(split);
    let local = k.udiv(sgid, splitc);
    let sub = k.umod(sgid, splitc);
    let body = |k: &mut Kernel, group: Word| {
        let sc_slots = k.u(slots);
        let base = k.imul(group, sc_slots);
        let slot = k.iadd(base, local);
        let rc = k.u(rows);
        let hc = k.u(half);
        let row0 = k.imul(slot, hc);
        let nn = k.u(n_out);
        let nblk = k.u(if q4.is_some() { kdim / 32 } else { kdim / 8 });
        let last = k.u(n_out - 1);
        // Rows past the end read the last row and are not written; every
        // simdgroup reaches the barrier. `Out::Swiglu`: row `r ≥ half` is the
        // up row of gate row `r − half`, `n / 2` further.
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r % half);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let mut row = k.select_u(lt, row, last);
                if pair && r >= half {
                    let o = k.u(n_out);
                    row = k.iadd(row, o);
                }
                k.imul(row, nblk)
            })
            .collect();
        let lane = k.lane();
        let sgc = k.u(sg);
        let off = k.imul(sub, sgc);
        let start = k.iadd(off, lane);
        let step = k.u(sg * split);
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
        k.for_(start, nblk, step, |k, b| {
            // `up` of the `[gate | up]` input, `kdim / 8` uint4 after `gate`.
            let ugap = k.u(kdim / 8);
            let Some(sc) = sc else {
                // bf16: as `matvec`, eight weights a unit.
                let xv = ldx(k, b);
                let uv = if xsrc == XSrc::Swiglu {
                    let ua = k.iadd(b, ugap);
                    Some(ldx(k, ua))
                } else {
                    None
                };
                let xu = x_unit(k, xv, uv);
                let xs: Vec<_> = (0..4).map(|i| (xu[2 * i], xu[2 * i + 1])).collect();
                for r in 0..rows as usize {
                    let at = k.iadd(bases[r], b);
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
                return;
            };
            let four = k.u(4);
            let xb = k.imul(b, four);
            let mut xs = Vec::with_capacity(32);
            let pre = sums.map(|sums| {
                let eight = k.u(8);
                let pb = k.imul(b, eight);
                let mut scaled = Vec::with_capacity(32);
                for i in 0..8 {
                    let ii = k.u(i);
                    let at = k.iadd(pb, ii);
                    let xv = ldx(k, at);
                    for c in 0..4 {
                        let w = k.comp(xv, c);
                        scaled.push(k.bits_to_f(w));
                    }
                }
                let sx = k.ld(sums, b);
                (scaled, sx)
            });
            for i in 0..if pre.is_some() { 0 } else { 4 } {
                let ii = k.u(i);
                let at = k.iadd(xb, ii);
                let xv = ldx(k, at);
                let uv = if xsrc == XSrc::Swiglu {
                    let ua = k.iadd(at, ugap);
                    Some(ldx(k, ua))
                } else {
                    None
                };
                xs.extend(x_unit(k, xv, uv));
            }
            let (eight, fifteen, eight_f) = (k.u(8), k.u(15), k.f(8.0));
            let ms = if pre.is_some() {
                pre
            } else if masks {
                Some(q4_masks_x(k, &xs))
            } else {
                None
            };
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], b);
                let wv = k.ld(w, at);
                let part = match &ms {
                    Some((scaled, sx)) => q4_masks_dot(k, wv, scaled, *sx),
                    None => {
                        let mut part = k.f(0.0);
                        for i in 0..4u32 {
                            let ww = k.comp(wv, i);
                            for byte in 0..4u32 {
                                for (shift, e) in
                                    [(8 * byte, 8 * i + byte), (8 * byte + 4, 8 * i + 4 + byte)]
                                {
                                    let sh = k.u(shift);
                                    let v = k.shr(ww, sh);
                                    let nib = k.and(v, fifteen);
                                    let biased = k.xor(nib, eight);
                                    let qf = k.u2f(biased);
                                    let q = k.fsub(qf, eight_f);
                                    part = k.fma(q, xs[e as usize], part);
                                }
                            }
                        }
                        part
                    }
                };
                let dh = k.ld(sc, at);
                let d = k.f16_to_f(dh);
                let a = k.get(accs[r]);
                let a = k.fma(d, part, a);
                k.set(accs[r], a);
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        // red[(local · rows + r) · split + sub]
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
        // The row's sum: its `split` partials in simdgroup order.
        let fold = |k: &mut Kernel, r: u32| -> Word {
            let rr = k.u(r);
            let i = k.iadd(lr, rr);
            let i = k.imul(i, splitc);
            let mut sum = k.sld(red, i);
            for x in 1..split {
                let o = k.u(x);
                let at = k.iadd(i, o);
                let p = k.sld(red, at);
                sum = k.fadd(sum, p);
            }
            sum
        };
        for r in 0..half {
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let in_n = k.ult(row, nn);
            let write = k.land(write0, in_n);
            k.if_(write, |k| {
                let sum = fold(k, r);
                if pair {
                    // `swiglu`'s bits: the gate and up sums as bf16, then `bf16(g·σ(g)·u)`.
                    let usum = fold(k, r + half);
                    let gq = k.round_bf16(sum);
                    let uq = k.round_bf16(usum);
                    let h = swiglu_one(k, gq, uq);
                    let b = k.f_to_bf16(h);
                    k.st(y, row, b);
                } else {
                    store(k, y, row, sum, out)
                }
            });
        }
        if persist > 0 {
            // `red` is read by the head before the next slot writes it.
            k.barrier();
        }
    };
    if persist > 0 {
        let vg = k.u(vgroups);
        let step = k.u(persist);
        k.for_(group, vg, step, |k, g| body(k, g));
    } else {
        body(&mut k, group);
    }
    let tag = if persist > 0 {
        format!("_p{persist}")
    } else {
        String::new()
    };
    let kind = match q4 {
        None => "matvec",
        Some(false) => "matvec_q4",
        Some(true) => "matvec_q4s",
    };
    done(
        format!(
            "{kind}_{kdim}x{n}{}{}{}_r{rows}_k{split}{tag}",
            x_tag(xsrc),
            out_tag(out),
            extra_tag(extra)
        ),
        k,
        nb,
    )
}

/// `Y[r] = bf16(W·X[r])` for the `m` rows of a prefill block, `W` in Q4_0 as
/// [`matvec_q4`]: one simdgroup per output row, the lanes over the blocks of
/// 32; each block of weights is loaded and decoded once and applied to the `m`
/// rows (row `r` of `X` at `r·ldx`, of `Y` at `r·ldy`, in bf16 elements). The
/// sum of each row runs in the order of [`matvec_q4`]'s: a block gives the
/// same bits as the decode step. Bindings `W (uint4), X (uint4), Y, scales`;
/// groups `ceil(n / 4)`.
pub fn matmul_q4(kdim: u32, n: u32, m: u32, ldx: u32, ldy: u32, sg: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32) && ldx.is_multiple_of(8), "Q4 block: K = {kdim}, ldx = {ldx}");
    let g = rows_per_group(sg);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, Elem::U16);
    let sc = k.buffer(3, Elem::U16);
    let group = k.flat_group();
    let sgid = k.subgroup_id();
    let gc = k.u(g);
    let base = k.imul(group, gc);
    let row = k.iadd(base, sgid);
    let nn = k.u(n);
    let inside = k.ult(row, nn);
    k.if_(inside, |k| {
        let nblk = k.u(kdim / 32);
        let wrow = k.imul(row, nblk);
        let lane = k.lane();
        let step = k.u(sg);
        let zero = k.f(0.0);
        let accs: Vec<_> = (0..m).map(|_| k.var_f32(zero)).collect();
        k.for_(lane, nblk, step, |k, b| {
            let at = k.iadd(wrow, b);
            let wv = k.ld(w, at);
            let dh = k.ld(sc, at);
            let d = k.f16_to_f(dh);
            // The 32 weights of the block, in the order matvec_q4 sums them.
            let (eight, fifteen, eight_f) = (k.u(8), k.u(15), k.f(8.0));
            let mut q = Vec::with_capacity(32);
            for i in 0..4u32 {
                let ww = k.comp(wv, i);
                for byte in 0..4u32 {
                    for (shift, e) in [(8 * byte, 8 * i + byte), (8 * byte + 4, 8 * i + 4 + byte)] {
                        let sh = k.u(shift);
                        let v = k.shr(ww, sh);
                        let nib = k.and(v, fifteen);
                        let biased = k.xor(nib, eight);
                        let qf = k.u2f(biased);
                        q.push((e, k.fsub(qf, eight_f)));
                    }
                }
            }
            let four = k.u(4);
            let xb = k.imul(b, four);
            for (r, acc) in accs.iter().enumerate() {
                let off = k.u(r as u32 * (ldx / 8));
                let xr = k.iadd(off, xb);
                let mut xs = Vec::with_capacity(32);
                for i in 0..4 {
                    let ii = k.u(i);
                    let a = k.iadd(xr, ii);
                    let xv = k.ld(x, a);
                    for c in 0..4 {
                        let xx = k.comp(xv, c);
                        xs.push(k.lo_f(xx));
                        xs.push(k.hi_f(xx));
                    }
                }
                let mut part = k.f(0.0);
                for &(e, qe) in &q {
                    part = k.fma(qe, xs[e as usize], part);
                }
                let a = k.get(*acc);
                let a = k.fma(d, part, a);
                k.set(*acc, a);
            }
        });
        let zero_u = k.u(0);
        let first = k.ieq(lane, zero_u);
        for (r, acc) in accs.iter().enumerate() {
            let a = k.get(*acc);
            let sum = k.sg_fadd(a);
            k.if_(first, |k| {
                let off = k.u(r as u32 * ldy);
                let at = k.iadd(off, row);
                let bf = k.f_to_bf16(sum);
                k.st(y, at, bf);
            });
        }
    });
    done(format!("matmul_q4_{kdim}x{n}_m{m}_x{ldx}_y{ldy}"), k, 4)
}

/// `W` `[n][k]` in Q4_0 as [`matvec_q4`] written out as bf16 `[n][k]`, each
/// weight `bf16(d·q)` (`d·q` exact in f32, one rounding to nearest even): the
/// matrix a bf16 prefill GEMM reads. One thread per block of 32, its four
/// words giving the four `uint4` of 8 bf16. Bindings `W (uint4), scales,
/// out (uint4)`; groups [`dequant_q4_groups`].
pub fn dequant_q4(kdim: u32, n: u32, sg: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32), "Q4: K = {kdim} is not a multiple of 32");
    let mut k = Kernel::new(256, sg);
    let w = k.buffer(0, Elem::UVec4);
    let sc = k.buffer(1, Elem::U16);
    let out = k.buffer(2, Elem::UVec4);
    let g = k.global_id();
    let total = k.u(n * (kdim / 32));
    let inside = k.ult(g, total);
    k.if_(inside, |k| {
        let wv = k.ld(w, g);
        let dh = k.ld(sc, g);
        let d = k.f16_to_f(dh);
        let four = k.u(4);
        let base = k.imul(g, four);
        for i in 0..4u32 {
            let ww = k.comp(wv, i);
            let v = q4_word_bf16(k, ww, d);
            let ii = k.u(i);
            let at = k.iadd(base, ii);
            k.st(out, at, v);
        }
    });
    done(format!("dequant_q4_{kdim}x{n}"), k, 3)
}

/// The cache warmer of a small kernel's extra groups ([`prefetch_uv4`]'s
/// reads): group `g` of them reads uint4 `2048·g ..` of the `n` of `src`.
pub(crate) fn prefetch_part(k: &mut Kernel, g: Word, src: Buf, sink: Buf, n: u32) {
    prefetch_part_wg(k, g, src, sink, n, 256)
}

/// [`prefetch_part`] for groups of `wg` invocations: group `g` reads uint4
/// `8·wg·g ..` of the `n` of `src`.
pub(crate) fn prefetch_part_wg(k: &mut Kernel, g: Word, src: Buf, sink: Buf, n: u32, wg: u32) {
    let lid = k.local_id();
    let per = k.u(8 * wg);
    let base = k.imul(g, per);
    let base = k.iadd(base, lid);
    let nn = k.u(n);
    let last = k.u(n - 1);
    let mut acc = k.u(0);
    for i in 0..8 {
        let o = k.u(wg * i);
        let at = k.iadd(base, o);
        let inside = k.ult(at, nn);
        let at = k.select_u(inside, at, last);
        let v = k.ld(src, at);
        for c in 0..4 {
            let w = k.comp(v, c);
            acc = k.xor(acc, w);
        }
    }
    let magic = k.u(0x9e37_79b9);
    let hit = k.ieq(acc, magic);
    k.if_(hit, |k| {
        let z = k.u(0);
        k.st(sink, z, acc);
    });
}

/// A small kernel's cache warmer, at the head of its code: the groups from
/// `real` on read `n` uint4 of binding `nbuf` ([`prefetch_part_wg`], the sink
/// at `nbuf + 1`) and return; the others go on. With `n = 0`, nothing.
/// Groups to add: [`prefetch_groups_wg`].
pub(crate) fn pf_head(k: &mut Kernel, real: u32, nbuf: u32, n: u32, wg: u32) {
    if n == 0 {
        return;
    }
    let src = k.buffer(nbuf, Elem::UVec4);
    let sink = k.buffer(nbuf + 1, Elem::U32);
    let g = k.flat_group();
    let rc = k.u(real);
    let below = k.ult(g, rc);
    let is_pf = k.b_not(below);
    k.if_(is_pf, |k| {
        let gr = k.isub(g, rc);
        prefetch_part_wg(k, gr, src, sink, n, wg);
        k.ret();
    });
}

/// Groups of `wg` invocations that read `n` uint4 ([`pf_head`]).
pub fn prefetch_groups_wg(n: u32, wg: u32) -> u32 {
    n.div_ceil(8 * wg)
}

/// The name suffix and buffer count of a kernel with [`pf_head`].
pub(crate) fn pf_tag(n: u32, nbuf: u32) -> (String, u32) {
    if n > 0 { (format!("_pf{n}"), nbuf + 2) } else { (String::new(), nbuf) }
}

/// Groups of 256 invocations that read `n` uint4 ([`prefetch_part`]).
pub fn prefetch_groups(n: u32) -> u32 {
    n.div_ceil(2048)
}

/// For measuring a weight layout (`HERBERT_METAL_PFPROBE`): [`matvec_q4s`]
/// with split `split`, each row's blocks by chunks of 32, a chunk the 32
/// blocks (512 bytes) then their 32 f16 scales (64 bytes), 576 bytes in all,
/// a row's chunks after one another. Bindings `W (uint4), x (uint4), y,
/// W as u16`; groups [`matvec_split_groups`].
pub fn matvec_q4s_chunked(kdim: u32, n: u32, out: Out, sg: u32, rows: u32, split: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32 * 32) || kdim.is_multiple_of(32), "K = {kdim}");
    let g = rows_per_group(sg);
    let slots = g / split;
    let nchunk = (kdim / 32).div_ceil(32);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
    let w16 = k.buffer(3, Elem::U16);
    let red = k.shared(g * rows, true);
    let group = k.flat_group();
    let sgid = k.subgroup_id();
    let lane = k.lane();
    let splitc = k.u(split);
    let local = k.udiv(sgid, splitc);
    let sub = k.umod(sgid, splitc);
    let sc_slots = k.u(slots);
    let base = k.imul(group, sc_slots);
    let slot = k.iadd(base, local);
    let rc = k.u(rows);
    let row0 = k.imul(slot, rc);
    let nn = k.u(n);
    let last = k.u(n - 1);
    let nblk = k.u(kdim / 32);
    let rbases: Vec<_> = (0..rows)
        .map(|r| {
            let rr = k.u(r);
            let row = k.iadd(row0, rr);
            let lt = k.ult(row, nn);
            let row = k.select_u(lt, row, last);
            let nc = k.u(nchunk * 36);
            k.imul(row, nc)
        })
        .collect();
    let zero = k.f(0.0);
    let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
    let cn = k.u(nchunk);
    k.for_(sub, cn, splitc, |k, c| {
        let thirty2 = k.u(32);
        let b0 = k.imul(c, thirty2);
        let b = k.iadd(b0, lane);
        let inside = k.ult(b, nblk);
        k.if_(inside, |k| {
            let four = k.u(4);
            let xb = k.imul(b, four);
            let mut xs = Vec::with_capacity(32);
            for i in 0..4 {
                let ii = k.u(i);
                let at = k.iadd(xb, ii);
                let xv = k.ld(x, at);
                for cc in 0..4 {
                    let xx = k.comp(xv, cc);
                    xs.push(k.lo_f(xx));
                    xs.push(k.hi_f(xx));
                }
            }
            let (scaled, sx) = q4_masks_x(k, &xs);
            for r in 0..rows as usize {
                let c36 = k.u(36);
                let cb = k.imul(c, c36);
                let cb = k.iadd(rbases[r], cb);
                let at = k.iadd(cb, lane);
                let wv = k.ld(w, at);
                let part = q4_masks_dot(k, wv, &scaled, sx);
                let s32 = k.iadd(cb, thirty2);
                let eight = k.u(8);
                let s16 = k.imul(s32, eight);
                let s16 = k.iadd(s16, lane);
                let dh = k.ld(w16, s16);
                let d = k.f16_to_f(dh);
                let a = k.get(accs[r]);
                let a = k.fma(d, part, a);
                k.set(accs[r], a);
            }
        });
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
            for xx in 1..split {
                let o = k.u(xx);
                let at = k.iadd(i, o);
                let p = k.sld(red, at);
                sum = k.fadd(sum, p);
            }
            store(k, y, row, sum, out)
        });
    }
    done(format!("matvec_q4s_chunked_{kdim}x{n}{}_r{rows}_k{split}", out_tag(out)), k, 4)
}

/// Reads `n` uint4 of `src` and keeps nothing (a cache warmer, for
/// measuring): each invocation 8 uint4, `2048` per group, coalesced; their
/// xor written to `sink` only if it is a value it never is in practice.
/// Bindings `src (uint4), sink (u32)`; groups `n / 2048`.
pub fn prefetch_uv4(n: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let src = k.buffer(0, Elem::UVec4);
    let sink = k.buffer(1, Elem::U32);
    let g = k.flat_group();
    let lid = k.local_id();
    let per = k.u(2048);
    let base = k.imul(g, per);
    let base = k.iadd(base, lid);
    let nn = k.u(n);
    let mut acc = k.u(0);
    for i in 0..8 {
        let o = k.u(256 * i);
        let at = k.iadd(base, o);
        let inside = k.ult(at, nn);
        let last = k.u(n - 1);
        let at = k.select_u(inside, at, last);
        let v = k.ld(src, at);
        for c in 0..4 {
            let w = k.comp(v, c);
            acc = k.xor(acc, w);
        }
    }
    let magic = k.u(0x9e37_79b9);
    let hit = k.ieq(acc, magic);
    k.if_(hit, |k| {
        let z = k.u(0);
        k.st(sink, z, acc);
    });
    done(format!("prefetch_uv4_{n}"), k, 2)
}

/// The input of [`matvec_q4s`] prepared once for every row: element `e` of
/// `x` (bf16) as `x·2^-s` in f32 ([`q4_masks_x`]'s), then each block's `Σx`
/// (in order, as [`q4_masks_x`] sums). Bindings `x, xp (f32 [k + k/32])`;
/// groups `k / 256`.
pub fn q4_prep_x(kdim: u32, sg: u32) -> Emitted {
    assert!(kdim.is_multiple_of(256), "q4_prep_x: K = {kdim}");
    let mut k = Kernel::new(256, sg);
    let x = k.buffer(0, Elem::U16);
    let xp = k.buffer(1, Elem::F32);
    let sh = k.shared(256, true);
    let i = k.global_id();
    let lid = k.local_id();
    let h = k.ld(x, i);
    let v = k.bf16_to_f(h);
    k.sst(sh, lid, v);
    // 2^-s of the element's place: j = e % 8, s = 8j (j < 4), 8(j − 4) + 4.
    let seven = k.u(7);
    let j = k.and(i, seven);
    let four = k.u(4);
    let lo = k.ult(j, four);
    let eight = k.u(8);
    let a = k.imul(j, eight);
    let jm = k.isub(j, four);
    let b = k.imul(jm, eight);
    let b = k.iadd(b, four);
    let s = k.select_u(lo, a, b);
    // 2^-s exactly: the exponent field 127 − s.
    let e127 = k.u(127);
    let ex = k.isub(e127, s);
    let twenty3 = k.u(23);
    let bits = k.shl(ex, twenty3);
    let f = k.bits_to_f(bits);
    let sc = k.fmul(v, f);
    k.st(xp, i, sc);
    k.barrier();
    let blocks = k.u(8);
    let first = k.ult(lid, blocks);
    k.if_(first, |k| {
        let thirty2 = k.u(32);
        let base = k.imul(lid, thirty2);
        let mut sum = k.sld(sh, base);
        for t in 1..32 {
            let o = k.u(t);
            let at = k.iadd(base, o);
            let v = k.sld(sh, at);
            sum = k.fadd(sum, v);
        }
        let g = k.flat_group();
        let gb = k.imul(g, blocks);
        let bi = k.iadd(gb, lid);
        let off = k.u(kdim);
        let at = k.iadd(off, bi);
        k.st(xp, at, sum);
    });
    done(format!("q4_prep_x_{kdim}"), k, 2)
}

/// llama.cpp's masks ([`matvec_q4s`]): the 32 activations of a block each
/// scaled by `2^-s` of its nibble's place in its word, and their sum.
pub(crate) fn q4_masks_x(k: &mut Kernel, xs: &[Word]) -> (Vec<Word>, Word) {
    let mut sx = xs[0];
    for &v in &xs[1..] {
        sx = k.fadd(sx, v);
    }
    let sc = (0..32u32)
        .map(|e| {
            let j = e % 8;
            let shift = if j < 4 { 8 * j } else { 8 * (j - 4) + 4 };
            let f = k.f((-(shift as f32)).exp2());
            k.fmul(xs[e as usize], f)
        })
        .collect();
    (sc, sx)
}

/// `Σ q·x` of one Q4_0 block `wv` (a `uint4`) by the masks, against the
/// activations of [`q4_masks_x`]: the nibbles made offset binary (`q + 8`, a
/// word at a time), each taken in place, `Σ (q+8)·x − 8·Σx`.
pub(crate) fn q4_masks_dot(k: &mut Kernel, wv: Word, scaled: &[Word], sum_x: Word) -> Word {
    let bias = k.u(0x8888_8888);
    let mut part = k.f(0.0);
    for i in 0..4u32 {
        let ww = k.comp(wv, i);
        let ww = k.xor(ww, bias);
        for j in 0..8u32 {
            let shift = if j < 4 { 8 * j } else { 8 * (j - 4) + 4 };
            let m = k.u(15 << shift);
            let v = k.and(ww, m);
            let f = k.u2f(v);
            part = k.fma(f, scaled[(8 * i + j) as usize], part);
        }
    }
    let minus8 = k.f(-8.0);
    k.fma(minus8, sum_x, part)
}

/// Word `ww` of a Q4_0 block of scale `d` (weights `8i..8i+8` of word `i`:
/// `8i+b` in the low nibble of byte `b`, `8i+4+b` in its high one) as the
/// `uint4` of its eight weights `bf16(d·q)`, in order.
pub(crate) fn q4_word_bf16(k: &mut Kernel, ww: Word, d: Word) -> Word {
    let (eight, fifteen, eight_f, sixteen) = (k.u(8), k.u(15), k.f(8.0), k.u(16));
    let mut bf = Vec::with_capacity(8);
    for j in 0..8u32 {
        let shift = if j < 4 { 8 * j } else { 8 * (j - 4) + 4 };
        let sh = k.u(shift);
        let v = k.shr(ww, sh);
        let nib = k.and(v, fifteen);
        let biased = k.xor(nib, eight);
        let qf = k.u2f(biased);
        let q = k.fsub(qf, eight_f);
        let p = k.fmul(d, q);
        bf.push(k.f_to_bf16(p));
    }
    let words: Vec<Word> = (0..4)
        .map(|c| {
            let hi = k.shl(bf[2 * c + 1], sixteen);
            k.or(bf[2 * c], hi)
        })
        .collect();
    k.uvec4([words[0], words[1], words[2], words[3]])
}

pub fn dequant_q4_groups(kdim: u32, n: u32) -> u32 {
    (n * (kdim / 32)).div_ceil(256)
}

pub fn matvec_groups(n: u32, sg: u32, rows: u32) -> u32 {
    n.div_ceil(rows_per_group(sg) * rows)
}

/// Logits per group of the first argmax pass.
pub const ARGMAX_CHUNK: u32 = 4096;

/// `ceil(n / ARGMAX_CHUNK)`.
pub fn argmax_groups(n: u32) -> u32 {
    n.div_ceil(ARGMAX_CHUNK)
}

/// First pass: per chunk of [`ARGMAX_CHUNK`] logits, `(max bits, index)` with
/// the first index on a tie. Bindings `logits (bf16), partial (u32 pairs)`.
pub fn argmax_partial(n: u32, sg: u32) -> Emitted {
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let logits = k.buffer(0, Elem::U16);
    let partial = k.buffer(1, Elem::U32);
    let fscratch = k.shared(k.subgroups() + 1, true);
    let uscratch = k.shared(k.subgroups() + 1, false);
    let g = k.flat_group();
    let lid = k.local_id();
    let chunk = k.u(ARGMAX_CHUNK);
    let first_i = k.imul(g, chunk);
    let start = k.iadd(first_i, lid);
    let end = k.iadd(first_i, chunk);
    let nn = k.u(n);
    let lt = k.ult(end, nn);
    let end = k.select_u(lt, end, nn);
    let step = k.u(wg);
    let neg = k.f(f32::NEG_INFINITY);
    let best = k.var_f32(neg);
    let none = k.u(u32::MAX);
    let best_i = k.var_u32(none);
    k.for_(start, end, step, |k, i| {
        let h = k.ld(logits, i);
        let v = k.bf16_to_f(h);
        let b = k.get(best);
        let better = k.fgt(v, b);
        k.if_(better, |k| {
            k.set(best, v);
            k.set(best_i, i);
        });
    });
    let b = k.get(best);
    let m = k.wg_reduce(b, fscratch, true);
    let bi = k.get(best_i);
    let is_max = k.feq(b, m);
    let cand = k.select_u(is_max, bi, none);
    let idx = k.wg_umin(cand, uscratch);
    let zero = k.u(0);
    let first = k.ieq(lid, zero);
    k.if_(first, |k| {
        let two = k.u(2);
        let at = k.imul(g, two);
        let mb = k.f_to_bits(m);
        k.st(partial, at, mb);
        let one = k.u(1);
        let at1 = k.iadd(at, one);
        k.st(partial, at1, idx);
    });
    done(format!("argmax_partial_{n}"), k, 2)
}

/// Second pass, one group: the best of the partials (first index on a tie),
/// into `state[ARGMAX]`. Bindings `partial, state`.
pub fn argmax_final(groups: u32, sg: u32) -> Emitted {
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let partial = k.buffer(0, Elem::U32);
    let state = k.buffer(1, Elem::U32);
    let fscratch = k.shared(k.subgroups() + 1, true);
    let uscratch = k.shared(k.subgroups() + 1, false);
    let lid = k.local_id();
    let gc = k.u(groups);
    let step = k.u(wg);
    let neg = k.f(f32::NEG_INFINITY);
    let best = k.var_f32(neg);
    let none = k.u(u32::MAX);
    let best_i = k.var_u32(none);
    k.for_(lid, gc, step, |k, g| {
        let two = k.u(2);
        let at = k.imul(g, two);
        let vb = k.ld(partial, at);
        let v = k.bits_to_f(vb);
        let one = k.u(1);
        let at1 = k.iadd(at, one);
        let i = k.ld(partial, at1);
        let b = k.get(best);
        let better = k.fgt(v, b);
        k.if_(better, |k| {
            k.set(best, v);
            k.set(best_i, i);
        });
    });
    let b = k.get(best);
    let m = k.wg_reduce(b, fscratch, true);
    let bi = k.get(best_i);
    let is_max = k.feq(b, m);
    let cand = k.select_u(is_max, bi, none);
    let idx = k.wg_umin(cand, uscratch);
    let zero = k.u(0);
    let first = k.ieq(lid, zero);
    k.if_(first, |k| {
        let at = k.u(STATE_ARGMAX);
        k.st(state, at, idx);
    });
    done(format!("argmax_final_{groups}"), k, 2)
}

/// Threadgroups of [`range_stats`]; each writes [`STATS_FIELDS`] f32.
pub const STATS_GROUPS: u32 = 256;
/// `max |x|`, `min |x| ≠ 0`, then the counts: non-zero, `0 < |x| < 2^-14`,
/// `0 < |x| < 2^-24`, NaN or Inf.
pub const STATS_FIELDS: u32 = 6;

/// The range statistics of `HERBERT_METAL_STATS=1` (the f16 study's probe S4):
/// the `n` values of `x` (bf16, or f32 when `f32_in`) folded per threadgroup
/// into [`STATS_FIELDS`] partials. Bindings `x, n (u32), out (f32
/// [STATS_GROUPS][STATS_FIELDS])`; [`STATS_GROUPS`] groups.
pub fn range_stats(f32_in: bool, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let x = k.buffer(0, if f32_in { Elem::F32 } else { Elem::U16 });
    let nb = k.buffer(1, Elem::U32);
    let out = k.buffer(2, Elem::F32);
    let scratch = k.shared(k.subgroups() + 1, true);
    let zero_u = k.u(0);
    let n = k.ld(nb, zero_u);
    let gid = k.global_id();
    let total = k.u(STATS_GROUPS * 256);
    let (zero, one, inf) = (k.f(0.0), k.f(1.0), k.f(f32::INFINITY));
    let (t14, t24) = (k.f(2f32.powi(-14)), k.f(2f32.powi(-24)));
    let (abs_mask, max_finite) = (k.u(0x7fff_ffff), k.u(0x7f7f_ffff));
    let mx = k.var_f32(zero);
    let mn = k.var_f32(inf);
    let counts: Vec<_> = (0..4).map(|_| k.var_f32(zero)).collect();
    k.for_(gid, n, total, |k, i| {
        let v = k.ld(x, i);
        let bits = if f32_in {
            k.f_to_bits(v)
        } else {
            let f = k.bf16_to_f(v);
            k.f_to_bits(f)
        };
        let ab = k.and(bits, abs_mask);
        let a = k.bits_to_f(ab);
        let nonfinite = k.ult(max_finite, ab);
        let finite = k.b_not(nonfinite);
        let pos = k.fgt(a, zero);
        let pos = k.land(pos, finite);
        let m = k.get(mx);
        let m2 = k.fmax(m, a);
        let m3 = k.select_f(finite, m2, m);
        k.set(mx, m3);
        let l = k.get(mn);
        let l2 = k.fmin(l, a);
        let l3 = k.select_f(pos, l2, l);
        k.set(mn, l3);
        let below14 = k.fgt(t14, a);
        let below14 = k.land(pos, below14);
        let below24 = k.fgt(t24, a);
        let below24 = k.land(pos, below24);
        for (c, cond) in counts.iter().zip([pos, below14, below24, nonfinite]) {
            let add = k.select_f(cond, one, zero);
            let old = k.get(*c);
            let s = k.fadd(old, add);
            k.set(*c, s);
        }
    });
    let m = k.get(mx);
    let m = k.wg_reduce(m, scratch, true);
    let l = k.get(mn);
    let nl = k.fneg(l);
    let nl = k.wg_reduce(nl, scratch, true);
    let l = k.fneg(nl);
    let mut fields = vec![m, l];
    for c in &counts {
        let v = k.get(*c);
        fields.push(k.wg_reduce(v, scratch, false));
    }
    let lid = k.local_id();
    let first = k.ieq(lid, zero_u);
    let g = k.flat_group();
    let nf = k.u(STATS_FIELDS);
    let base = k.imul(g, nf);
    k.if_(first, |k| {
        for (j, &f) in fields.iter().enumerate() {
            let jj = k.u(j as u32);
            let at = k.iadd(base, jj);
            k.st(out, at, f);
        }
    });
    done(if f32_in { "range_stats_f32" } else { "range_stats_bf16" }, k, 3)
}
