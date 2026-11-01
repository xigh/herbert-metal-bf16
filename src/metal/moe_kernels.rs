//! The kernels of 26B-A4B's expert block (`Gemma4TextRouter`,
//! `Gemma4TextExperts` of `modeling_gemma4`), rounding where the module does:
//! - the router's input: the weightless RMSNorm of the residual (one rounding),
//!   times `router.scale` (rounded), times `hidden^-½` (rounded);
//! - the route: the softmax of the 128 bf16 scores in f32, the 8 best, their
//!   probabilities divided by their sum and multiplied by `per_expert_scale`,
//!   all in f32; the route is written in increasing expert order, the order in
//!   which the module adds the experts up;
//! - the experts' projections: `matvec_q4`'s arithmetic, the expert's rows
//!   found through the route;
//! - the combination: `index_add_` into a bf16 zero vector, expert by expert,
//!   each `bf16(bf16(y)·w)` added and rounded.
//!
//! Every kernel also runs over the rows of a prefill block (one route per row).

use crate::msl::{done, Elem, Emitted, Kernel, Word};

const WG: u32 = 256;

/// The router's input of one token: `y = bf16(bf16(bf16(x·inv)·scale)·root)`,
/// `inv = 1/√(Σx²/d + ε)`, `root = d^-½` (f32). Bindings `x, scale, y`; one
/// group, or one group per row of a block with `x`, `y` rows `sx`, `sy` apart.
pub fn router_norm(d: u32, eps: f32, sg: u32, sx: u32, sy: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
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
    let (dn, step) = (k.u(d), k.u(WG));
    let zero = k.f(0.0);
    let acc = k.var_f32(zero);
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
    let root = k.f((d as f64).powf(-0.5) as f32);
    k.for_(lid, dn, step, |k, i| {
        let h = k.ld(x, i);
        let v = k.bf16_to_f(h);
        let n = k.fmul(v, inv);
        let n = k.round_bf16(n);
        let wh = k.ld(w, i);
        let wf = k.bf16_to_f(wh);
        let t = k.fmul(n, wf);
        let t = k.round_bf16(t);
        let o = k.fmul(t, root);
        let b = k.f_to_bf16(o);
        k.st(y, i, b);
    });
    done(format!("moe_router_norm_{d}{}", if sx + sy > 0 { format!("_r{sx}_{sy}") } else { String::new() }), k, 3)
}

/// Threads of [`route`]: the experts, in whole simdgroups.
fn route_threads(experts: u32, sg: u32) -> u32 {
    experts.div_ceil(sg) * sg
}

/// The route of one token from the router's bf16 scores `[experts]`: the
/// softmax in f32, the `top_k` best probabilities (the lowest expert on a
/// tie), each divided by their sum (taken from the best down, as
/// `torch.topk` returns them) and multiplied by `per_expert_scale`. Written
/// as `route[0..top_k]` the experts in increasing order, `route[top_k..]`
/// their weights (f32 bits) in the same order. Bindings `scores, pes, route`;
/// one group, or one group per row of a block with the scores and the routes
/// rows `ss`, `sr` apart.
pub fn route(experts: u32, top_k: u32, sg: u32, ss: u32, sr: u32) -> Emitted {
    assert!(top_k >= 1 && top_k <= experts, "top {top_k} of {experts}");
    let mut k = Kernel::new(route_threads(experts, sg), sg);
    let scores = k.buffer(0, Elem::U16);
    let pes = k.buffer(1, Elem::U16);
    let out = k.buffer(2, Elem::U32);
    if ss + sr > 0 {
        let row = k.flat_group();
        for (b, s) in [(scores, ss), (out, sr)] {
            let st = k.u(s);
            let o = k.imul(row, st);
            k.advance(b, o);
        }
    }
    let fscratch = k.shared(k.subgroups() + 1, true);
    let uscratch = k.shared(k.subgroups() + 1, false);
    let chosen = k.shared(top_k, false);
    let probs = k.shared(top_k + 1, true);
    let lid = k.local_id();
    let en = k.u(experts);
    let mine = k.ult(lid, en);
    let neg = k.f(f32::NEG_INFINITY);
    let s_var = k.var_f32(neg);
    k.if_(mine, |k| {
        let h = k.ld(scores, lid);
        let v = k.bf16_to_f(h);
        k.set(s_var, v);
    });
    let s = k.get(s_var);
    let mx = k.wg_reduce(s, fscratch, true);
    let zero = k.f(0.0);
    let ex_var = k.var_f32(zero);
    k.if_(mine, |k| {
        let d = k.fsub(s, mx);
        let e = k.exp(d);
        k.set(ex_var, e);
    });
    let ex = k.get(ex_var);
    let sum = k.wg_reduce(ex, fscratch, false);
    let p = k.fdiv(ex, sum);
    // Probabilities still in the running; −1 below any of them once chosen or past the experts.
    let minus = k.f(-1.0);
    let p0 = k.select_f(mine, p, minus);
    let avail = k.var_f32(p0);
    let none = k.u(u32::MAX);
    let z = k.u(0);
    let first = k.ieq(lid, z);
    for r in 0..top_k {
        let a = k.get(avail);
        let best = k.wg_reduce(a, fscratch, true);
        let is_best = k.feq(a, best);
        let cand = k.select_u(is_best, lid, none);
        let idx = k.wg_umin(cand, uscratch);
        let won = k.ieq(lid, idx);
        k.if_(won, |k| k.set(avail, minus));
        k.if_(first, |k| {
            let rr = k.u(r);
            k.sst(chosen, rr, idx);
            k.sst(probs, rr, best);
        });
    }
    k.barrier();
    k.if_(first, |k| {
        let z = k.u(0);
        let s0 = k.sld(probs, z);
        let acc = k.var_f32(s0);
        for r in 1..top_k {
            let rr = k.u(r);
            let pr = k.sld(probs, rr);
            let a = k.get(acc);
            let a = k.fadd(a, pr);
            k.set(acc, a);
        }
        let a = k.get(acc);
        let slot = k.u(top_k);
        k.sst(probs, slot, a);
    });
    k.barrier();
    let kn = k.u(top_k);
    let writer = k.ult(lid, kn);
    k.if_(writer, |k| {
        let e = k.sld(chosen, lid);
        let pr = k.sld(probs, lid);
        let slot = k.u(top_k);
        let total = k.sld(probs, slot);
        let w = k.fdiv(pr, total);
        let sh = k.ld(pes, e);
        let sf = k.bf16_to_f(sh);
        let w = k.fmul(w, sf);
        // Its rank among the chosen: how many chosen experts are lower.
        let zero = k.u(0);
        let rank = k.var_u32(zero);
        for j in 0..top_k {
            let jj = k.u(j);
            let o = k.sld(chosen, jj);
            let lt = k.ult(o, e);
            let one = k.u(1);
            let r = k.get(rank);
            let r1 = k.iadd(r, one);
            let r = k.select_u(lt, r1, r);
            k.set(rank, r);
        }
        let r = k.get(rank);
        k.st(out, r, e);
        let at = k.iadd(r, kn);
        let wb = k.f_to_bits(w);
        k.st(out, at, wb);
    });
    done(format!("moe_route_{experts}_k{top_k}{}", if ss + sr > 0 { format!("_r{ss}_{sr}") } else { String::new() }), k, 3)
}

/// Simdgroups per threadgroup of the expert projection: 128 threads.
fn rows_per_group(sg: u32) -> u32 {
    (128 / sg).max(1)
}

/// Groups of [`matvec_q4_experts`] for one slot of one row.
pub fn expert_groups(n: u32, sg: u32, rows: u32) -> u32 {
    n.div_ceil(rows_per_group(sg) * rows)
}

/// `Y[r][s] = bf16(W[e]·X[r][s])` for the `top_k` experts `e = route[r][s]`
/// of each row `r` of a block: `W` the `[experts][n][k]` stack in Q4_0 (one
/// `q4::quantize` of its `experts·n` rows), the sum of each output as
/// `matvec_q4`'s. `X[r][s]` at `r·sxr + s·sxs` bf16 (`sxs` 0: every slot
/// reads the same input), `Y[r][s]` at `r·syr + s·n`, the route at `r·sr`.
/// Bindings `W (uint4), X (uint4), Y, scales, route`; groups
/// `rows_of_block · top_k · expert_groups(n)`.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q4_experts(kdim: u32, n: u32, top_k: u32, sg: u32, rows: u32, sxs: u32, sxr: u32, syr: u32, sr: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32) && sxs.is_multiple_of(8) && sxr.is_multiple_of(8), "Q4 experts: K = {kdim}, strides {sxs} {sxr}");
    let g = rows_per_group(sg);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, Elem::U16);
    let sc = k.buffer(3, Elem::U16);
    let route = k.buffer(4, Elem::U32);
    let gps = expert_groups(n, sg, rows);
    let group = k.flat_group();
    let per_row = k.u(top_k * gps);
    let brow = k.udiv(group, per_row);
    let rem = k.umod(group, per_row);
    let gpsw = k.u(gps);
    let slot = k.udiv(rem, gpsw);
    let gi = k.umod(rem, gpsw);
    for (b, s) in [(route, sr), (y, syr)] {
        if s > 0 {
            let st = k.u(s);
            let o = k.imul(brow, st);
            k.advance(b, o);
        }
    }
    let nn_y = k.u(n);
    let yo = k.imul(slot, nn_y);
    k.advance(y, yo);
    if sxr > 0 {
        let st = k.u(sxr / 8);
        let o = k.imul(brow, st);
        k.advance(x, o);
    }
    if sxs > 0 {
        let st = k.u(sxs / 8);
        let o = k.imul(slot, st);
        k.advance(x, o);
    }
    let expert = k.ld(route, slot);
    let sgid = k.subgroup_id();
    let gc = k.u(g);
    let base = k.imul(gi, gc);
    let sl = k.iadd(base, sgid);
    let rc = k.u(rows);
    let row0 = k.imul(sl, rc);
    let nn = k.u(n);
    let inside = k.ult(row0, nn);
    k.if_(inside, |k| {
        let nblk = k.u(kdim / 32);
        let last = k.u(n - 1);
        let first_row = k.imul(expert, nn);
        // Rows past the end read the expert's last row and are not written.
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                let row = k.iadd(first_row, row);
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
            for r in 0..rows as usize {
                let at = k.iadd(bases[r], b);
                let wv = k.ld(w, at);
                let mut part = k.f(0.0);
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
            k.if_(write, |k| {
                let bf = k.f_to_bf16(sum);
                k.st(y, row, bf);
            });
        }
    });
    done(format!("moe_mvq4_{kdim}x{n}_k{top_k}_r{rows}_x{sxs}_{sxr}_y{syr}_s{sr}"), k, 5)
}

/// The experts' outputs `Y[s]` (`[top_k][d]` bf16) of one token added up as
/// `index_add_` does into a bf16 zero vector, in the route's (increasing
/// expert) order: `acc = bf16(acc + bf16(Y[s]·w[s]))`. Bindings
/// `Y, route, out`; groups `ceil(d / 256)`, or as many per row of a block with
/// `Y`, the routes and `out` rows `sy`, `sr`, `so` apart.
pub fn combine(d: u32, top_k: u32, sg: u32, sy: u32, sr: u32, so: u32) -> Emitted {
    let mut k = Kernel::new(WG, sg);
    let y = k.buffer(0, Elem::U16);
    let route = k.buffer(1, Elem::U32);
    let out = k.buffer(2, Elem::U16);
    let rows = sy + sr + so > 0;
    let i: Word = if rows {
        let g = k.flat_group();
        let pr = k.u(d.div_ceil(WG));
        let row = k.udiv(g, pr);
        let gi = k.umod(g, pr);
        for (b, s) in [(y, sy), (route, sr), (out, so)] {
            let st = k.u(s);
            let o = k.imul(row, st);
            k.advance(b, o);
        }
        let wg = k.u(WG);
        let b = k.imul(gi, wg);
        let lid = k.local_id();
        k.iadd(b, lid)
    } else {
        k.global_id()
    };
    let dn = k.u(d);
    let inside = k.ult(i, dn);
    k.if_(inside, |k| {
        let zero = k.f(0.0);
        let acc = k.var_f32(zero);
        for s in 0..top_k {
            let off = k.u(s * d);
            let at = k.iadd(off, i);
            let yh = k.ld(y, at);
            let yf = k.bf16_to_f(yh);
            let wa = k.u(top_k + s);
            let wb = k.ld(route, wa);
            let wf = k.bits_to_f(wb);
            let c = k.fmul(yf, wf);
            let c = k.round_bf16(c);
            let a = k.get(acc);
            let a = k.fadd(a, c);
            let a = k.round_bf16(a);
            k.set(acc, a);
        }
        let a = k.get(acc);
        let b = k.f_to_bf16(a);
        k.st(out, i, b);
    });
    done(format!("moe_combine_{d}_k{top_k}{}", if rows { format!("_r{sy}_{sr}_{so}") } else { String::new() }), k, 3)
}

/// The experts' lists of a block of `m` rows, from their routes (`2·top_k`
/// u32 per row): `lists[e]` the number of (row, slot) pairs that chose expert
/// `e`, then at `experts + e·m` those pairs (`row·top_k + slot`, increasing).
/// Bindings `routes, lists`; one group.
pub fn gather(experts: u32, top_k: u32, m: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(route_threads(experts, sg), sg);
    let routes = k.buffer(0, Elem::U32);
    let lists = k.buffer(1, Elem::U32);
    let e = k.local_id();
    let en = k.u(experts);
    let mine = k.ult(e, en);
    k.if_(mine, |k| {
        let zero = k.u(0);
        let cnt = k.var_u32(zero);
        let mc = k.u(m);
        let base0 = k.imul(e, mc);
        let base = k.iadd(base0, en);
        let pairs = k.u(m * top_k);
        let one = k.u(1);
        k.for_(zero, pairs, one, |k, p| {
            let kk = k.u(top_k);
            let r = k.udiv(p, kk);
            let s = k.umod(p, kk);
            let k2 = k.u(2 * top_k);
            let ra = k.imul(r, k2);
            let at = k.iadd(ra, s);
            let idx = k.ld(routes, at);
            let hit = k.ieq(idx, e);
            k.if_(hit, |k| {
                let c = k.get(cnt);
                let w = k.iadd(base, c);
                k.st(lists, w, p);
                let one = k.u(1);
                let c1 = k.iadd(c, one);
                k.set(cnt, c1);
            });
        });
        let c = k.get(cnt);
        k.st(lists, e, c);
    });
    done(format!("moe_gather_{experts}_k{top_k}_m{m}"), k, 2)
}

/// Groups of [`matmul_q4_experts`].
pub fn grouped_groups(experts: u32, n: u32, sg: u32, rows: u32) -> u32 {
    experts * n.div_ceil(rows_per_group(sg) * rows)
}

/// [`matvec_q4_experts`] over a block of `m` rows grouped by expert: the
/// groups of an expert run `matvec_q4`'s body (`rows` output rows per
/// simdgroup) once per (row, slot) pair of its list ([`gather`]), one pair
/// after the other, so that its weights come from DRAM once per block and
/// from the cache for the next pairs; each output is summed in `matvec_q4`'s
/// order, the bits are the row-by-row kernel's. `X[pair]` at
/// `row·sxr + slot·sxs` bf16, `Y[pair]` at `pair·n`. Bindings
/// `W (uint4), X (uint4), Y, scales, lists`; groups [`grouped_groups`].
#[allow(clippy::too_many_arguments)]
pub fn matmul_q4_experts(kdim: u32, n: u32, experts: u32, top_k: u32, m: u32, sg: u32, rows: u32, sxs: u32, sxr: u32) -> Emitted {
    assert!(kdim.is_multiple_of(32) && sxs.is_multiple_of(8) && sxr.is_multiple_of(8), "Q4 grouped experts: K = {kdim}, strides {sxs} {sxr}");
    let g = rows_per_group(sg);
    let mut k = Kernel::new(g * sg, sg);
    let w = k.buffer(0, Elem::UVec4);
    let x = k.buffer(1, Elem::UVec4);
    let y = k.buffer(2, Elem::U16);
    let sc = k.buffer(3, Elem::U16);
    let lists = k.buffer(4, Elem::U32);
    let group = k.flat_group();
    let gpn = k.u(n.div_ceil(g * rows));
    let e = k.udiv(group, gpn);
    let gi = k.umod(group, gpn);
    let cnt = k.ld(lists, e);
    let sgid = k.subgroup_id();
    let gc = k.u(g);
    let b0 = k.imul(gi, gc);
    let sl = k.iadd(b0, sgid);
    let rc = k.u(rows);
    let row0 = k.imul(sl, rc);
    let nn = k.u(n);
    let inside = k.ult(row0, nn);
    k.if_(inside, |k| {
        let nblk = k.u(kdim / 32);
        let last = k.u(n - 1);
        let first_row = k.imul(e, nn);
        // Rows past the end read the expert's last row and are not written.
        let bases: Vec<_> = (0..rows)
            .map(|r| {
                let rr = k.u(r);
                let row = k.iadd(row0, rr);
                let lt = k.ult(row, nn);
                let row = k.select_u(lt, row, last);
                let row = k.iadd(first_row, row);
                k.imul(row, nblk)
            })
            .collect();
        let en = k.u(experts);
        let mc = k.u(m);
        let lb0 = k.imul(e, mc);
        let lbase = k.iadd(lb0, en);
        let zero_u = k.u(0);
        let one_u = k.u(1);
        k.for_(zero_u, cnt, one_u, |k, j| {
            let la = k.iadd(lbase, j);
            let p = k.ld(lists, la);
            let kk = k.u(top_k);
            let r = k.udiv(p, kk);
            let s = k.umod(p, kk);
            let o1 = k.u(sxr / 8);
            let a1 = k.imul(r, o1);
            let o2 = k.u(sxs / 8);
            let a2 = k.imul(s, o2);
            let xr = k.iadd(a1, a2);
            let lane = k.lane();
            let step = k.u(sg);
            let zero = k.f(0.0);
            let accs: Vec<_> = (0..rows).map(|_| k.var_f32(zero)).collect();
            k.for_(lane, nblk, step, |k, b| {
                let four = k.u(4);
                let xb0 = k.imul(b, four);
                let xb = k.iadd(xr, xb0);
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
                for rr in 0..rows as usize {
                    let at = k.iadd(bases[rr], b);
                    let wv = k.ld(w, at);
                    let mut part = k.f(0.0);
                    for i in 0..4u32 {
                        let ww = k.comp(wv, i);
                        for byte in 0..4u32 {
                            for (shift, el) in [(8 * byte, 8 * i + byte), (8 * byte + 4, 8 * i + 4 + byte)] {
                                let sh = k.u(shift);
                                let v = k.shr(ww, sh);
                                let nib = k.and(v, fifteen);
                                let biased = k.xor(nib, eight);
                                let qf = k.u2f(biased);
                                let q = k.fsub(qf, eight_f);
                                part = k.fma(q, xs[el as usize], part);
                            }
                        }
                    }
                    let dh = k.ld(sc, at);
                    let d = k.f16_to_f(dh);
                    let a = k.get(accs[rr]);
                    let a = k.fma(d, part, a);
                    k.set(accs[rr], a);
                }
            });
            let z = k.u(0);
            let first = k.ieq(lane, z);
            let pn = k.imul(p, nn);
            for rr in 0..rows {
                let a = k.get(accs[rr as usize]);
                let sum = k.sg_fadd(a);
                let ro = k.u(rr);
                let row = k.iadd(row0, ro);
                let in_n = k.ult(row, nn);
                let wr = k.land(first, in_n);
                k.if_(wr, |k| {
                    let at = k.iadd(pn, row);
                    let bf = k.f_to_bf16(sum);
                    k.st(y, at, bf);
                });
            }
        });
    });
    done(format!("moe_mmq4_{kdim}x{n}_e{experts}_k{top_k}_m{m}_r{rows}_x{sxs}_{sxr}"), k, 5)
}
