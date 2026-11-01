//! The gated delta rule of a prefill block on the matrix units
//! (`rec.chunk=4`, the `bf16` feature): herbert-vk-bf16's `vk/dntc.rs`
//! (the chunked form of FLA, chunks of `C` rows, bf16 operands and f32
//! accumulators) with each of its 16 × 16 tiles as 2 × 2 simdgroup matrices
//! of 8 × 8; the same operators, the same rounding points.
//!
//! With `G_t = Σ_{i≤t} g_i` (g = log decay, cumulated from the chunk's first
//! row), `Γ_t = exp(G_t)` and the state `S0` (`[dk][dv]`) at the chunk's start:
//! `A_ts = β_t·exp(G_t−G_s)·(k_t·k_s)` for `s < t`, `T = (I + A)⁻¹`;
//! `Δ = T·diag(β)·V − T·diag(β·Γ)·K·S0`;
//! `O = scale·diag(Γ)·Q·S0 + (scale·exp(G_t−G_s)·(q_t·k_s))_{s≤t}·Δ`;
//! `S = Γ_C·S0 + (diag(Γ_C/Γ)·K)ᵀ·Δ` — the row-by-row recurrence
//! (`s *= decay; δ = β(v − sᵀk); s += kδᵀ; o = sᵀq·scale`) summed by chunks.
//!
//! [`dn_mma_prep`] computes, per (chunk, value head), what depends on q, k, β
//! and g only, into a scratch: `T·diag(β)`, `P` (the masked, decayed, scaled
//! `Q·Kᵀ`), `W = T·diag(β·Γ)·K`, `scale·diag(Γ)·Q`, `diag(Γ_C/Γ)·K` (bf16) and
//! `Γ_C` (f32). [`dn_mma_state`] walks the chunks of the block in order per
//! (value head, slice of `DVS` columns), the slice of the state held in f32
//! accumulators from the first chunk to the last.

use super::kernels::STATE_ROWS;
use super::qwen_kernels::DeltaShape;
use crate::msl::{dn_scale, done, Buf, DnScaled, Elem, Emitted, Kernel, Var, Word};

/// Rows of a chunk.
pub const C: u32 = 32;
/// Columns of the state per group of [`dn_mma_state`].
pub const DVS: u32 = 32;
/// Invocations per group: four simdgroups of 32.
const WG: u32 = 128;
/// The matrix of the hardware, 8 × 8.
const T8: u32 = 8;

pub fn mat_len(s: DeltaShape) -> u32 {
    3 * C * C + 3 * C * s.k_dim
}
const OFF_TB: u32 = 0;
const OFF_TBG: u32 = C * C;
const OFF_P: u32 = 2 * C * C;
fn off_w(_s: DeltaShape) -> u32 {
    3 * C * C
}
fn off_qg(s: DeltaShape) -> u32 {
    3 * C * C + C * s.k_dim
}
fn off_kd(s: DeltaShape) -> u32 {
    3 * C * C + 2 * C * s.k_dim
}

fn rows(k: &mut Kernel, state: Buf) -> Word {
    let i = k.u(STATE_ROWS);
    k.ld(state, i)
}

/// The eight bf16 of a `uvec4` as f32.
fn unpack8(k: &mut Kernel, v: Word) -> Vec<Word> {
    let mut f = Vec::with_capacity(8);
    for e in 0..4 {
        let u = k.comp(v, e);
        f.push(k.lo_f(u));
        f.push(k.hi_f(u));
    }
    f
}

/// A `uvec4`, or zero when `keep` is false.
pub(crate) fn zero_unless(k: &mut Kernel, keep: Word, v: Word) -> Word {
    let z = k.u(0);
    let c: Vec<Word> = (0..4)
        .map(|e| {
            let x = k.comp(v, e);
            k.select_u(keep, x, z)
        })
        .collect();
    k.uvec4([c[0], c[1], c[2], c[3]])
}

/// Per (chunk, value head): the chunk's operators (module doc). Bindings
/// `state, proj (bf16, as uint4), gates (bf16 [β | g]), mat (bf16 scratch),
/// vec (f32 scratch, Γ_C)`; groups `ceil(m / C) · v_heads`.
pub fn dn_mma_prep(s: DeltaShape, sg: u32) -> Emitted {
    let (dk, nv) = (s.k_dim, s.v_heads);
    assert!(crate::msl::ACT_BF16, "the matrix-unit recurrence needs the bf16 feature");
    assert!(sg == 32 && dk.is_multiple_of(16) && s.proj_width().is_multiple_of(8) && s.key().is_multiple_of(8), "wave {sg}, dk {dk}");
    let mut k = Kernel::new(WG, sg);
    let st = k.buffer(0, Elem::U32);
    // Option c (`msl::set_dn_scale`): every operator of the scratch stored
    // times 2^tscale; W = TβΓ·K comes out of the scaled TβΓ scaled already.
    let te = dn_scale(DnScaled::Tc);
    let tup = |k: &mut Kernel, x: Word| if te > 0 { let s = k.pow2(te as i32); k.fmul(x, s) } else { x };
    let proj4 = k.buffer(1, Elem::UVec4);
    let gates = k.buffer(2, Elem::Act);
    let mat = k.buffer(3, Elem::U16);
    let vecb = k.buffer(4, Elem::F32);
    // The scratch is written and read back by the group across barriers.
    k.coherent(mat);
    let rs = dk / 8 + 1; // uvec4 per staged row of K or Q, one spare against bank conflicts
    let kq = k.shared_vec4(2 * C * rs);
    let kk_s = k.shared(C * C, true);
    let qk_s = k.shared(C * C, true);
    let t_s = k.shared(C * C, true);
    let beta_s = k.shared(C, true);
    let g_s = k.shared(C, true);
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let nvc = k.u(nv);
    let cidx = k.udiv(g, nvc);
    let h = k.umod(g, nvc);
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let cc = k.u(C);
    let r0 = k.imul(cidx, cc);
    let t = k.local_id();
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let zero = k.f(0.0);
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let mlen = k.u(mat_len(s));
    let gbase = k.imul(g, mlen);
    // 1. K and Q rows of the chunk into workgroup memory (zero past m).
    let cpr = dk / 8;
    let cprc = k.u(cpr);
    let rsc = k.u(rs);
    let pw8 = k.u(s.proj_width() / 8);
    let key8 = k.u(s.key() / 8);
    let dk8 = k.u(dk / 8);
    let qoff = k.imul(kh, dk8);
    let qbase_sh = k.u(C * rs);
    for i in 0..(C * cpr).div_ceil(WG) {
        let ic = k.u(i * WG);
        let idx = k.iadd(t, ic);
        let lim = k.u(C * cpr);
        let in_tile = k.ult(idx, lim);
        k.if_(in_tile, |k| {
            let row = k.udiv(idx, cprc);
            let cell = k.umod(idx, cprc);
            let grow = k.iadd(r0, row);
            let inside = k.ult(grow, m);
            let last = k.isub(m, one_u);
            let rr = k.select_u(inside, grow, last);
            let ra = k.imul(rr, pw8);
            let qa = k.iadd(ra, qoff);
            let qa = k.iadd(qa, cell);
            let ka = k.iadd(qa, key8);
            let qv = k.ld(proj4, qa);
            let kv = k.ld(proj4, ka);
            let qv = zero_unless(k, inside, qv);
            let kv = zero_unless(k, inside, kv);
            let sa = k.imul(row, rsc);
            let sa = k.iadd(sa, cell);
            k.sst(kq, sa, kv);
            let sq = k.iadd(sa, qbase_sh);
            k.sst(kq, sq, qv);
        });
    }
    // 2. β and g of the chunk's rows (zero past m).
    let two_nv = k.u(2 * nv);
    let in_c = k.ult(t, cc);
    k.if_(in_c, |k| {
        let grow = k.iadd(r0, t);
        let inside = k.ult(grow, m);
        let last = k.isub(m, one_u);
        let rr = k.select_u(inside, grow, last);
        let gi = k.imul(rr, two_nv);
        let gi = k.iadd(gi, h);
        let beta = k.ld(gates, gi);
        let di = k.iadd(gi, nvc);
        let gl = k.ld(gates, di);
        let beta = k.select_f(inside, beta, zero);
        let gl = k.select_f(inside, gl, zero);
        k.sst(beta_s, t, beta);
        k.sst(g_s, t, gl);
    });
    k.barrier();
    // 3. G, cumulated over the chunk (one invocation).
    let first = k.ieq(t, zero_u);
    k.if_(first, |k| {
        let mut acc = k.f(0.0);
        for i in 0..C {
            let ic = k.u(i);
            let x = k.sld(g_s, ic);
            acc = k.fadd(acc, x);
            k.sst(g_s, ic, acc);
        }
    });
    k.barrier();
    // 4. K·Kᵀ and Q·Kᵀ: a 16 × 16 block of each per simdgroup, 2 × 2 matrices.
    let sgid = k.subgroup_id();
    let two = k.u(2);
    let ti = k.udiv(sgid, two);
    let tj = k.umod(sgid, two);
    let t16 = k.u(16);
    let ti16 = k.imul(ti, t16);
    let tj16 = k.imul(tj, t16);
    // bf16 per staged row of K or Q.
    let rs8 = rs * 8;
    let rs8c = k.u(rs8);
    let a_row = k.imul(ti16, rs8c);
    let b_row = k.imul(tj16, rs8c);
    let qbase_el = k.u(C * rs8);
    let acc_kk: Vec<Var> = (0..4).map(|_| k.mat_acc()).collect();
    let acc_qk: Vec<Var> = (0..4).map(|_| k.mat_acc()).collect();
    for kk in 0..dk / T8 {
        let off = k.u(kk * T8);
        let mut a_k = Vec::new();
        let mut a_q = Vec::new();
        let mut b = Vec::new();
        for i in 0..2 {
            let io = k.u(i * T8 * rs8);
            let at = k.iadd(a_row, io);
            let at = k.iadd(at, off);
            a_k.push(k.mat_load_bf16(kq, at, rs8, false));
            let aq = k.iadd(at, qbase_el);
            a_q.push(k.mat_load_bf16(kq, aq, rs8, false));
            let bt = k.iadd(b_row, io);
            let bt = k.iadd(bt, off);
            b.push(k.mat_load_bf16(kq, bt, rs8, true));
        }
        for i in 0..2 {
            for j in 0..2 {
                k.mat_mad(acc_kk[i * 2 + j], a_k[i], b[j]);
                k.mat_mad(acc_qk[i * 2 + j], a_q[i], b[j]);
            }
        }
    }
    let out_at = k.imul(ti16, cc);
    let out_at = k.iadd(out_at, tj16);
    for i in 0..2u32 {
        for j in 0..2u32 {
            let o = k.u(i * T8 * C + j * T8);
            let at = k.iadd(out_at, o);
            k.mat_store(acc_kk[(i * 2 + j) as usize], kk_s, at, C);
            k.mat_store(acc_qk[(i * 2 + j) as usize], qk_s, at, C);
        }
    }
    k.barrier();
    // 5. A (into kk_s), P (scratch), T = I to start.
    let tb_at = k.u(OFF_TB);
    let tbg_at = k.u(OFF_TBG);
    let p_at = k.u(OFF_P);
    for i in 0..(C * C) / WG {
        let ic = k.u(i * WG);
        let e = k.iadd(t, ic);
        let r = k.udiv(e, cc);
        let c = k.umod(e, cc);
        let gr = k.sld(g_s, r);
        let gc = k.sld(g_s, c);
        let dg = k.fsub(gr, gc);
        let f = k.exp(dg);
        let below = k.ult(c, r);
        let diag = k.ieq(c, r);
        let upto = k.lor(below, diag);
        let br = k.sld(beta_s, r);
        let kkv = k.sld(kk_s, e);
        let a = k.fmul(br, f);
        let a = k.fmul(a, kkv);
        let a = k.select_f(below, a, zero);
        k.sst(kk_s, e, a);
        let qkv = k.sld(qk_s, e);
        let p = k.fmul(scale, f);
        let p = k.fmul(p, qkv);
        let p = k.select_f(upto, p, zero);
        let p = tup(&mut k, p);
        let pb = k.f_to_bf16(p);
        let pi = k.iadd(gbase, p_at);
        let pi = k.iadd(pi, e);
        k.st(mat, pi, pb);
        let one = k.f(1.0);
        let tv = k.select_f(diag, one, zero);
        k.sst(t_s, e, tv);
    }
    k.barrier();
    // 6. T = (I + A)⁻¹ by forward substitution, a row at a time.
    for r in 1..C {
        let rc = k.u(r);
        let col = k.ult(t, rc);
        k.if_(col, |k| {
            let rb = k.u(r * C);
            let mut acc = k.f(0.0);
            for s2 in 0..r {
                let ai = k.u(r * C + s2);
                let a = k.sld(kk_s, ai);
                let ti_ = k.u(s2 * C);
                let ti_ = k.iadd(ti_, t);
                let tv = k.sld(t_s, ti_);
                acc = k.fma(a, tv, acc);
            }
            let neg = k.fneg(acc);
            let at = k.iadd(rb, t);
            k.sst(t_s, at, neg);
        });
        k.barrier();
    }
    // 7. Tβ and TβΓ into the scratch.
    for i in 0..(C * C) / WG {
        let ic = k.u(i * WG);
        let e = k.iadd(t, ic);
        let c = k.umod(e, cc);
        let tv = k.sld(t_s, e);
        let bc = k.sld(beta_s, c);
        let tbv = k.fmul(tv, bc);
        let gc = k.sld(g_s, c);
        let gam = k.exp(gc);
        let tbgv = k.fmul(tbv, gam);
        let tbv = tup(&mut k, tbv);
        let tbgv = tup(&mut k, tbgv);
        let x = k.f_to_bf16(tbv);
        let at = k.iadd(gbase, tb_at);
        let at = k.iadd(at, e);
        k.st(mat, at, x);
        let x = k.f_to_bf16(tbgv);
        let at = k.iadd(gbase, tbg_at);
        let at = k.iadd(at, e);
        k.st(mat, at, x);
    }
    k.barrier();
    // 8. W = TβΓ·K, C × dk: per simdgroup 16 rows × dk/2 columns, 2 × 8 matrices.
    let w_at = k.u(off_w(s));
    let dkc = k.u(dk);
    let ncols = dk / 2 / T8;
    let accs: Vec<Var> = (0..2 * ncols).map(|_| k.mat_acc()).collect();
    let half = k.u(dk / 2);
    let col_base = k.imul(tj, half);
    for kk in 0..C / T8 {
        let a: Vec<Word> = (0..2)
            .map(|i| {
                let ai = k.u(i * T8);
                let r = k.iadd(ti16, ai);
                let at = k.imul(r, cc);
                let ko = k.u(kk * T8);
                let at = k.iadd(at, ko);
                let at = k.iadd(at, tbg_at);
                let at = k.iadd(at, gbase);
                k.mat_load_bf16_dev(mat, at, C, false)
            })
            .collect();
        for q in 0..ncols {
            let br = k.u(kk * T8 * rs8 + q * T8);
            let bi = k.iadd(br, col_base);
            let b = k.mat_load_bf16(kq, bi, rs8, false);
            for i in 0..2 {
                k.mat_mad(accs[(i * ncols + q) as usize], a[i as usize], b);
            }
        }
    }
    for i in 0..2u32 {
        for q in 0..ncols {
            let wb = k.mat_to_bf16(accs[(i * ncols + q) as usize]);
            let ai = k.u(i * T8);
            let r = k.iadd(ti16, ai);
            let wi = k.imul(r, dkc);
            let qo = k.u(q * T8);
            let wi = k.iadd(wi, qo);
            let wi = k.iadd(wi, col_base);
            let wi = k.iadd(wi, w_at);
            let wi = k.iadd(wi, gbase);
            k.mat_store_bf16_dev(wb, mat, wi, dk);
        }
    }
    // 9. QΓ·scale and KD = diag(Γ_C/Γ)·K, eight elements per staged cell.
    let qg_at = k.u(off_qg(s));
    let kd_at = k.u(off_kd(s));
    let lastc = k.u(C - 1);
    let glast = k.sld(g_s, lastc);
    for i in 0..(C * cpr).div_ceil(WG) {
        let ic = k.u(i * WG);
        let idx = k.iadd(t, ic);
        let lim = k.u(C * cpr);
        let in_tile = k.ult(idx, lim);
        k.if_(in_tile, |k| {
            let row = k.udiv(idx, cprc);
            let cell = k.umod(idx, cprc);
            let sa = k.imul(row, rsc);
            let sa = k.iadd(sa, cell);
            let kv = k.sld(kq, sa);
            let sq = k.iadd(sa, qbase_sh);
            let qv = k.sld(kq, sq);
            let gr = k.sld(g_s, row);
            let eg = k.exp(gr);
            let qs = k.fmul(eg, scale);
            let dg = k.fsub(glast, gr);
            let ks = k.exp(dg);
            let eight = k.u(8);
            let col0 = k.imul(cell, eight);
            let rb = k.imul(row, dkc);
            let rb = k.iadd(rb, col0);
            let rb = k.iadd(rb, gbase);
            let qf = unpack8(k, qv);
            let kf = unpack8(k, kv);
            for e in 0..8 {
                let ec = k.u(e);
                let at = k.iadd(rb, ec);
                let x = k.fmul(qf[e as usize], qs);
                let x = tup(k, x);
                let x = k.f_to_bf16(x);
                let a1 = k.iadd(at, qg_at);
                k.st(mat, a1, x);
                let y = k.fmul(kf[e as usize], ks);
                let y = tup(k, y);
                let y = k.f_to_bf16(y);
                let a2 = k.iadd(at, kd_at);
                k.st(mat, a2, y);
            }
        });
    }
    // 10. Γ_C.
    k.if_(first, |k| {
        let gamma = k.exp(glast);
        k.st(vecb, g, gamma);
    });
    done(format!("dn_mma_prep_k{}v{}_c{C}", s.k_heads, s.v_heads), k, 5)
}

/// Per (value head, slice of [`DVS`] columns): the chunks of the block in
/// order, the slice of the state in f32 accumulators (module doc). Bindings
/// `state, proj (bf16, as uint4), mat, vec, rec_state (bf16 [nv][dk][dv]),
/// out (bf16 [m][nv·dv])`; groups `v_heads · dv / DVS`.
pub fn dn_mma_state(s: DeltaShape, sg: u32) -> Emitted {
    let (dk, dv, nv) = (s.k_dim, s.v_dim, s.v_heads);
    assert!(crate::msl::ACT_BF16, "the matrix-unit recurrence needs the bf16 feature");
    assert!(sg == 32 && dk == 4 * 32 && dv.is_multiple_of(DVS), "wave {sg}, dk {dk}, dv {dv}");
    let mut k = Kernel::new(WG, sg);
    let st = k.buffer(0, Elem::U32);
    let proj4 = k.buffer(1, Elem::UVec4);
    let mat = k.buffer(2, Elem::U16);
    let vecb = k.buffer(3, Elem::F32);
    let state = k.buffer(4, Elem::U16);
    let out = k.buffer(5, Elem::U16);
    // Option c (`msl::set_dn_scale`): the state (in memory and staged in
    // sbuf) times 2^se, the scratch's operators times 2^te, `out` times 2^oe;
    // the f32 accumulators hold the unscaled values between the products.
    let (se, oe, te) = (dn_scale(DnScaled::State), dn_scale(DnScaled::Out), dn_scale(DnScaled::Tc));
    let scale_by = |k: &mut Kernel, v: Var, e: i32| {
        if e != 0 {
            let f = k.pow2(e);
            k.mat_scale_in_place(v, f);
        }
    };
    // uint4 per staged row of the slice, one spare against bank conflicts; bf16 per row.
    let rv = DVS / 8 + 1;
    let rv8 = rv * 8;
    let sbuf = k.shared_vec4(dk * rv);
    let dbuf = k.shared_vec4(C * rv);
    let vbuf = k.shared_vec4(C * rv);
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let slices = k.u(dv / DVS);
    let h = k.udiv(g, slices);
    let cs = k.umod(g, slices);
    let dvsc = k.u(DVS);
    let c0 = k.imul(cs, dvsc);
    let t = k.local_id();
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let sgid = k.subgroup_id();
    let two = k.u(2);
    let t16 = k.u(16);
    let rvc = k.u(rv);
    let rv8c = k.u(rv8);
    let cc = k.u(C);
    let dvc = k.u(dv);
    let dkc = k.u(dk);
    let nvc = k.u(nv);
    // This simdgroup's matrices of the state: rows 32·sgid + 8·a (a < 4), columns 8·b (b < 4).
    let hs = k.u(dk * dv);
    let hbase = k.imul(h, hs);
    let r32 = k.u(32);
    let srow0 = k.imul(sgid, r32);
    let mut s_vars = Vec::new();
    let mut s_at = Vec::new();
    for a in 0..4u32 {
        for b in 0..DVS / T8 {
            let ac = k.u(a * T8);
            let row = k.iadd(srow0, ac);
            let idx = k.imul(row, dvc);
            let idx = k.iadd(idx, hbase);
            let idx = k.iadd(idx, c0);
            let co = k.u(b * T8);
            let idx = k.iadd(idx, co);
            let sb = k.mat_load_bf16_dev(state, idx, dv, false);
            let sv = k.mat_from_bf16(sb);
            scale_by(&mut k, sv, -(se as i32));
            s_vars.push(sv);
            s_at.push((idx, row, b));
        }
    }
    let ti = k.udiv(sgid, two);
    let tj = k.umod(sgid, two);
    let ti16 = k.imul(ti, t16);
    let tj16 = k.imul(tj, t16);
    let mlen = k.u(mat_len(s));
    let tb_at = k.u(OFF_TB);
    let p_at = k.u(OFF_P);
    let w_at = k.u(off_w(s));
    let qg_at = k.u(off_qg(s));
    let kd_at = k.u(off_kd(s));
    let pw8 = k.u(s.proj_width() / 8);
    let v8 = k.u((2 * s.key()) / 8);
    let dv8 = k.u(dv / 8);
    let hv8 = k.imul(h, dv8);
    let eight = k.u(8);
    let c08 = k.udiv(c0, eight);
    let value = k.u(s.value());
    let hv = k.imul(h, dvc);
    let chunks = {
        let cm = k.u(C - 1);
        let mm = k.iadd(m, cm);
        k.udiv(mm, cc)
    };
    // The A operand at `off` of the scratch: rows ti16 + 8·i, columns `col`, stride `ld`.
    let a_dev = |k: &mut Kernel, base: Word, off: Word, ld: u32, i: u32, col: u32| {
        let io = k.u(i * T8);
        let r = k.iadd(ti16, io);
        let ldc = k.u(ld);
        let at = k.imul(r, ldc);
        let cl = k.u(col);
        let at = k.iadd(at, cl);
        let at = k.iadd(at, off);
        let at = k.iadd(at, base);
        k.mat_load_bf16_dev(mat, at, ld, false)
    };
    // The B operand of a staged buffer: rows `row`, columns tj16 + 8·j.
    let b_sh = |k: &mut Kernel, sh, row: u32, j: u32| {
        let o = k.u(row * rv8 + j * T8);
        let at = k.iadd(tj16, o);
        k.mat_load_bf16(sh, at, rv8, false)
    };
    k.for_(zero_u, chunks, one_u, |k, ci| {
        let cn = k.imul(ci, nvc);
        let cn = k.iadd(cn, h);
        let base = k.imul(cn, mlen);
        let r0 = k.imul(ci, cc);
        // a. The state's slice (bf16) and V's into threadgroup memory.
        for (n, &(_, row, b)) in s_at.iter().enumerate() {
            scale_by(k, s_vars[n], se as i32);
            let sb = k.mat_to_bf16(s_vars[n]);
            scale_by(k, s_vars[n], -(se as i32));
            let at = k.imul(row, rv8c);
            let co = k.u(b * T8);
            let at = k.iadd(at, co);
            k.mat_store_bf16(sb, sbuf, at, rv8);
        }
        {
            let cpr = DVS / 8;
            let cprc = k.u(cpr);
            let lim = k.u(C * cpr);
            let in_tile = k.ult(t, lim);
            k.if_(in_tile, |k| {
                let row = k.udiv(t, cprc);
                let cell = k.umod(t, cprc);
                let grow = k.iadd(r0, row);
                let inside = k.ult(grow, m);
                let last = k.isub(m, one_u);
                let rr = k.select_u(inside, grow, last);
                let ra = k.imul(rr, pw8);
                let va = k.iadd(ra, v8);
                let va = k.iadd(va, hv8);
                let va = k.iadd(va, c08);
                let va = k.iadd(va, cell);
                let v = k.ld(proj4, va);
                let v = zero_unless(k, inside, v);
                let sa = k.imul(row, rvc);
                let sa = k.iadd(sa, cell);
                k.sst(vbuf, sa, v);
            });
        }
        k.barrier();
        // b. Δ = Tβ·V − W·S0, a 16 × 16 block per simdgroup, into dbuf.
        let acc_u: Vec<Var> = (0..4).map(|_| k.mat_acc()).collect();
        for kk in 0..C / T8 {
            let a: Vec<Word> = (0..2).map(|i| a_dev(k, base, tb_at, C, i, kk * T8)).collect();
            let b: Vec<Word> = (0..2).map(|j| b_sh(k, vbuf, kk * T8, j)).collect();
            for i in 0..2 {
                for j in 0..2 {
                    k.mat_mad(acc_u[i * 2 + j], a[i], b[j]);
                }
            }
        }
        let acc_w: Vec<Var> = (0..4).map(|_| k.mat_acc()).collect();
        for kk in 0..dk / T8 {
            let a: Vec<Word> = (0..2).map(|i| a_dev(k, base, w_at, dk, i, kk * T8)).collect();
            let b: Vec<Word> = (0..2).map(|j| b_sh(k, sbuf, kk * T8, j)).collect();
            for i in 0..2 {
                for j in 0..2 {
                    k.mat_mad(acc_w[i * 2 + j], a[i], b[j]);
                }
            }
        }
        for i in 0..2u32 {
            for j in 0..2u32 {
                let n = (i * 2 + j) as usize;
                // Tβ·V carries 2^te, W·S0 2^(te + se).
                scale_by(k, acc_u[n], -(te as i32));
                scale_by(k, acc_w[n], -((te + se) as i32));
                k.mat_sub_in_place(acc_u[n], acc_w[n]);
                let d = k.mat_to_bf16(acc_u[n]);
                let ro = k.u(i * T8);
                let r = k.iadd(ti16, ro);
                let at = k.imul(r, rv8c);
                let at = k.iadd(at, tj16);
                let jo = k.u(j * T8);
                let at = k.iadd(at, jo);
                k.mat_store_bf16(d, dbuf, at, rv8);
            }
        }
        k.barrier();
        // c. O = QΓ·S0 + P·Δ.
        let acc_o: Vec<Var> = (0..4).map(|_| k.mat_acc()).collect();
        for kk in 0..dk / T8 {
            let a: Vec<Word> = (0..2).map(|i| a_dev(k, base, qg_at, dk, i, kk * T8)).collect();
            let b: Vec<Word> = (0..2).map(|j| b_sh(k, sbuf, kk * T8, j)).collect();
            for i in 0..2 {
                for j in 0..2 {
                    k.mat_mad(acc_o[i * 2 + j], a[i], b[j]);
                }
            }
        }
        // QΓ·S0 carries 2^(te + se), P·Δ 2^te.
        for n in 0..4 {
            scale_by(k, acc_o[n], -(se as i32));
        }
        for kk in 0..C / T8 {
            let a: Vec<Word> = (0..2).map(|i| a_dev(k, base, p_at, C, i, kk * T8)).collect();
            let b: Vec<Word> = (0..2).map(|j| b_sh(k, dbuf, kk * T8, j)).collect();
            for i in 0..2 {
                for j in 0..2 {
                    k.mat_mad(acc_o[i * 2 + j], a[i], b[j]);
                }
            }
        }
        for i in 0..2u32 {
            for j in 0..2u32 {
                scale_by(k, acc_o[(i * 2 + j) as usize], oe as i32 - te as i32);
                let o = k.mat_to_bf16(acc_o[(i * 2 + j) as usize]);
                let ro = k.u(i * T8);
                let orow = k.iadd(r0, ti16);
                let orow = k.iadd(orow, ro);
                let oi = k.imul(orow, value);
                let oi = k.iadd(oi, hv);
                let oi = k.iadd(oi, c0);
                let oi = k.iadd(oi, tj16);
                let jo = k.u(j * T8);
                let oi = k.iadd(oi, jo);
                k.mat_store_bf16_dev(o, out, oi, s.value());
            }
        }
        // d. S = Γ_C·S0 + KDᵀ·Δ.
        let gamma = k.ld(vecb, cn);
        // KDᵀ·Δ carries 2^te: Γ_C·S0 taken to it, the sum brought back.
        let gamma = if te > 0 { let f = k.pow2(te as i32); k.fmul(gamma, f) } else { gamma };
        for (n, &(_, row, b)) in s_at.iter().enumerate() {
            k.mat_scale_in_place(s_vars[n], gamma);
            for kk in 0..C / T8 {
                let ai = k.u(kk * T8 * dk);
                let ai = k.iadd(ai, row);
                let ai = k.iadd(ai, kd_at);
                let ai = k.iadd(ai, base);
                let a = k.mat_load_bf16_dev(mat, ai, dk, true);
                let bo = k.u(kk * T8 * rv8 + b * T8);
                let bm = k.mat_load_bf16(dbuf, bo, rv8, false);
                k.mat_mad(s_vars[n], a, bm);
            }
            scale_by(k, s_vars[n], -(te as i32));
        }
        // The next chunk overwrites sbuf, dbuf and vbuf.
        k.barrier();
    });
    for (n, &(idx, _, _)) in s_at.iter().enumerate() {
        scale_by(&mut k, s_vars[n], se as i32);
        let sb = k.mat_to_bf16(s_vars[n]);
        k.mat_store_bf16_dev(sb, state, idx, dv);
    }
    let _ = dkc;
    done(format!("dn_mma_state_k{}v{}_c{C}d{DVS}", s.k_heads, s.v_heads), k, 6)
}
