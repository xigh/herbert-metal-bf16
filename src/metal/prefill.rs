//! The emitted kernels of a prefill block: `m` rows of the prompt at once
//! (`m ≤ prefill.block`, read from `state[ROWS]`, the first row's position from
//! `state[POS]`). Same numerical contract as the decode kernels, which are left
//! untouched: one rounding to nearest even at the output of each kernel.
//!
//! The projections are a GEMM `Y[m][N] = X[m][K]·Wᵀ` (`W` `[N][K]`, the
//! checkpoint's layout); the DeltaNet conv and recurrence walk the rows in
//! order inside the kernel, their state held in registers for the whole block.
//!
//! herbert-vk-bf16's `vk/prefill.rs` (main, 8440202) call for call, without
//! its cooperative-matrix GEMMs (Vulkan's) and the SIMT GEMM of `gemm.simt=3`;
//! the WY forms of the recurrence (`dn_recurrence_chunk`, `dn_chunk_prep`,
//! `dn_chunk_apply`) read and write the DeltaNet's activations as
//! [`Elem::Act`] here, the decay from its logarithm in the bf16 build (vk-bf16
//! refuses them with the `bf16` feature).

use super::kernels::{pos, Out, STATE_ROWS};
use super::qwen_kernels::{AttnShape, DeltaShape};
use crate::msl::{done, Buf, Elem, Emitted, Kernel, Word};

fn rows(k: &mut Kernel, state: Buf) -> Word {
    let i = k.u(STATE_ROWS);
    k.ld(state, i)
}

/// The output tile `(gm, gn)` of group `g` in a GEMM of `m` rows (tiles of
/// `bm`) and `n_tiles` column tiles. `m_first`: consecutive groups walk the
/// row tiles of one column tile, so a tile of `W` serves every row tile while
/// it is in the cache; otherwise they walk the column tiles of one row tile,
/// and each row tile reads all of `W` again.
fn tile_of(k: &mut Kernel, g: Word, m: Word, bm: u32, n_tiles: u32, m_first: bool) -> (Word, Word) {
    if m_first {
        let round = k.u(bm - 1);
        let bmc = k.u(bm);
        let mm = k.iadd(m, round);
        let mb = k.udiv(mm, bmc);
        (k.umod(g, mb), k.udiv(g, mb))
    } else {
        let nb = k.u(n_tiles);
        (k.udiv(g, nb), k.umod(g, nb))
    }
}

/// `x[r] = embed[tokens[r]]` for the `m` rows. Bindings `state, tokens (u32),
/// table (u32 pairs), x (u32 pairs)`; groups `ceil(m · hidden / 2 / 256)`.
pub fn embed_rows(hidden: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let state = k.buffer(0, Elem::U32);
    let tokens = k.buffer(1, Elem::U32);
    let table = k.buffer(2, Elem::U32);
    let x = k.buffer(3, Elem::U32);
    let words = k.u(hidden / 2);
    let m = rows(&mut k, state);
    let total = k.imul(m, words);
    let i = k.global_id();
    let inside = k.ult(i, total);
    k.if_(inside, |k| {
        let r = k.udiv(i, words);
        let c = k.umod(i, words);
        let token = k.ld(tokens, r);
        let at = k.imul(token, words);
        let at = k.iadd(at, c);
        let v = k.ld(table, at);
        k.st(x, i, v);
    });
    done(format!("embed_rows_{hidden}"), k, 4)
}

/// RMSNorm of each row, one group per row. Bindings `x, w, y` (bf16); groups `m`.
pub fn rmsnorm_rows(d: u32, eps: f32, plus_one: bool, sg: u32) -> Emitted {
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let x = k.buffer(0, Elem::U16);
    let w = k.buffer(1, Elem::U16);
    let y = k.buffer(2, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let row = k.flat_group();
    let dc = k.u(d);
    let base = k.imul(row, dc);
    let lid = k.local_id();
    let step = k.u(wg);
    let zero = k.f(0.0);
    let acc = k.var_f32(zero);
    k.for_(lid, dc, step, |k, i| {
        let at = k.iadd(base, i);
        let h = k.ld(x, at);
        let v = k.bf16_to_f(h);
        let a = k.get(acc);
        let s = k.fma(v, v, a);
        k.set(acc, s);
    });
    let a = k.get(acc);
    let ss = k.wg_reduce(a, scratch, false);
    let inv_d = k.f(1.0 / d as f32);
    let e = k.f(eps);
    let mm = k.fmul(ss, inv_d);
    let mm = k.fadd(mm, e);
    let r = k.sqrt(mm);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    k.for_(lid, dc, step, |k, i| {
        let at = k.iadd(base, i);
        let h = k.ld(x, at);
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
        k.st(y, at, b);
    });
    done(format!("rmsnorm_rows_{d}{}", if plus_one { "_p1" } else { "" }), k, 3)
}

/// Tile of the SIMT GEMM: 64 × 64 outputs per group of 256 invocations, each
/// 4 × 4, `K` walked by 16 through workgroup memory.
const GT: u32 = 64;
const GK: u32 = 16;

/// `Y[m][N] = X[m][K]·Wᵀ` on the vector units: the tile of `X` and of `W` staged
/// in workgroup memory as f32, 4 × 4 accumulators per invocation, the output
/// rounded once (or added to the residual, or kept f32) as [`Out`] says.
///
/// `v2`: the staged tiles stored `K`-major and each invocation on rows
/// `ty + 16·i`, columns `tx + 16·j` — neighbouring lanes read neighbouring
/// words; the first layout (rows `4·ty + i`, columns `4·tx + j`, tiles
/// row-major) has 16 lanes on one bank for every read of `W`.
/// Bindings `state, W (bf16), X (bf16), Y`; groups [`gemm_simt_groups`].
pub fn gemm_simt(kdim: u32, n: u32, out: Out, sg: u32, v2: bool, m_first: bool) -> Emitted {
    assert!(kdim.is_multiple_of(GK), "K = {kdim} is not a multiple of {GK}");
    let wg = 256;
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let w = k.buffer(1, Elem::U16);
    let x = k.buffer(2, Elem::U16);
    let y = k.buffer(3, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
    let a_s = k.shared(GT * GK, true);
    let b_s = k.shared(GT * GK, true);
    let m = rows(&mut k, state);
    let g = k.flat_group();
    let (gm, gn) = tile_of(&mut k, g, m, GT, n.div_ceil(GT), m_first);
    let tile = k.u(GT);
    let m0 = k.imul(gm, tile);
    let n0 = k.imul(gn, tile);
    let lid = k.local_id();
    let sixteen = k.u(16);
    let tx = k.umod(lid, sixteen);
    let ty = k.udiv(lid, sixteen);
    let kc = k.u(kdim);
    let nc = k.u(n);
    let gk = k.u(GK);
    let four = k.u(4);
    let zero = k.f(0.0);
    let accs: Vec<_> = (0..16).map(|_| k.var_f32(zero)).collect();
    let zero_u = k.u(0);
    k.for_(zero_u, kc, gk, |k, k0| {
        // Stage: 1024 elements of each tile, four per invocation.
        for q in 0..4 {
            let qc = k.u(q);
            let e = k.imul(lid, four);
            let e = k.iadd(e, qc);
            let r = k.udiv(e, gk);
            let kk = k.umod(e, gk);
            let col = k.iadd(k0, kk);
            // X tile
            let row = k.iadd(m0, r);
            let in_m = k.ult(row, m);
            let xr = k.imul(row, kc);
            let xa = k.iadd(xr, col);
            let zero = k.f(0.0);
            let xv = k.var_f32(zero);
            k.if_(in_m, |k| {
                let h = k.ld(x, xa);
                let f = k.bf16_to_f(h);
                k.set(xv, f);
            });
            let xf = k.get(xv);
            let sa = if v2 {
                let gtc = k.u(GT);
                let t = k.imul(kk, gtc);
                k.iadd(t, r)
            } else {
                e
            };
            k.sst(a_s, sa, xf);
            // W tile
            let nn = k.iadd(n0, r);
            let in_n = k.ult(nn, nc);
            let wr = k.imul(nn, kc);
            let wa = k.iadd(wr, col);
            let wv = k.var_f32(zero);
            k.if_(in_n, |k| {
                let h = k.ld(w, wa);
                let f = k.bf16_to_f(h);
                k.set(wv, f);
            });
            let wf = k.get(wv);
            let sb = if v2 {
                let gtc = k.u(GT);
                let t = k.imul(kk, gtc);
                k.iadd(t, r)
            } else {
                e
            };
            k.sst(b_s, sb, wf);
        }
        k.barrier();
        for kk in 0..GK {
            let kkc = k.u(kk);
            let a: Vec<_> = (0..4)
                .map(|i| {
                    let at = if v2 {
                        let base = k.u(kk * GT + 16 * i);
                        k.iadd(base, ty)
                    } else {
                        let r = k.imul(ty, four);
                        let ic = k.u(i);
                        let r = k.iadd(r, ic);
                        let at = k.imul(r, gk);
                        k.iadd(at, kkc)
                    };
                    k.sld(a_s, at)
                })
                .collect();
            let b: Vec<_> = (0..4)
                .map(|j| {
                    let at = if v2 {
                        let base = k.u(kk * GT + 16 * j);
                        k.iadd(base, tx)
                    } else {
                        let c = k.imul(tx, four);
                        let jc = k.u(j);
                        let c = k.iadd(c, jc);
                        let at = k.imul(c, gk);
                        k.iadd(at, kkc)
                    };
                    k.sld(b_s, at)
                })
                .collect();
            for i in 0..4 {
                for j in 0..4 {
                    let v = accs[i * 4 + j];
                    let acc = k.get(v);
                    let acc = k.fma(a[i], b[j], acc);
                    k.set(v, acc);
                }
            }
        }
        k.barrier();
    });
    for i in 0..4u32 {
        for j in 0..4u32 {
            let (r, c) = if v2 {
                let ic = k.u(16 * i);
                let jc = k.u(16 * j);
                (k.iadd(ty, ic), k.iadd(tx, jc))
            } else {
                let r = k.imul(ty, four);
                let ic = k.u(i);
                let c = k.imul(tx, four);
                let jc = k.u(j);
                (k.iadd(r, ic), k.iadd(c, jc))
            };
            let row = k.iadd(m0, r);
            let col = k.iadd(n0, c);
            let in_m = k.ult(row, m);
            let in_n = k.ult(col, nc);
            let inside = k.land(in_m, in_n);
            let acc = k.get(accs[(i * 4 + j) as usize]);
            k.if_(inside, |k| {
                let yr = k.imul(row, nc);
                let at = k.iadd(yr, col);
                match out {
                    Out::F32 => k.st(y, at, acc),
                    Out::Bf16 => {
                        let b = k.f_to_bf16(acc);
                        k.st(y, at, b);
                    }
                    Out::Residual => {
                        let r = k.round_bf16(acc);
                        let old = k.ld(y, at);
                        let of = k.bf16_to_f(old);
                        let s = k.fadd(of, r);
                        let b = k.f_to_bf16(s);
                        k.st(y, at, b);
                    }
                    Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
                }
            });
        }
    }
    let tag = match out {
        Out::Bf16 => "bf16",
        Out::F32 => "f32",
        Out::Residual => "res",
        Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
    };
    done(format!("gemm_simt_{kdim}x{n}_{tag}{}{}", if v2 { "_v2" } else { "" }, if m_first { "_mf" } else { "" }), k, 4)
}

pub fn gemm_simt_groups(m: u32, n: u32) -> u32 {
    m.div_ceil(GT) * n.div_ceil(GT)
}

/// Tile of the matrix-unit GEMM: 64 × 64 outputs per group of four
/// simdgroups (2 × 2, each 32 × 32 as 4 × 4 matrices of 8 × 8), `K` staged
/// by 32.
const MT: u32 = 64;
const MK: u32 = 32;

/// The rows of a matrix-unit GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaRows {
    /// `Y[r] = X[r]·Wᵀ` for the `m` rows of the block.
    Dense,
    /// The experts' `[gate | up]` of the block: tile `t` of the table
    /// ([`moe_sort`]) holds rows `start..start + count` of the sorted list,
    /// all routed to expert `e`; sorted row `p` reads the token row
    /// `list[p] / top_k` of `X` and `W`'s rows `e·N..`, and writes `Y[p]`.
    Gather { top_k: u32, tiles: u32, bm: u32 },
    /// The experts' `down`: sorted row `p` reads `X[p]` and `W`'s rows
    /// `e·N..`, and writes `Y[list[p]]` (the row's slot, `row·top_k + slot`).
    Scatter { tiles: u32, bm: u32 },
}

impl MmaRows {
    /// Rows of a tile: 64 dense, `bm` (16, 32 or 64) over experts.
    fn bm(&self) -> u32 {
        match *self {
            MmaRows::Dense => MT,
            MmaRows::Gather { bm, .. } | MmaRows::Scatter { bm, .. } => bm,
        }
    }
}

/// The organisation of the matrix-unit GEMM over dense rows, each field a
/// measurement arm (on the M1 the classic kernel runs at 0.98
/// TFLOPS where llama.cpp's, 64 × 32 tiles on 8 KiB of threadgroup memory
/// staged by 8 × 8 blocks, reaches at least 1.3; `gemm.auto` times them at
/// load, `gemm.mform` names one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaForm {
    /// Rows of the tile: 64 (2 × 2 simdgroups of 32 × 32 outputs) or 32 (1 × 4 of 32 × 16).
    pub bm: u32,
    /// X and W staged by 8 × 8 blocks, each contiguous (`simdgroup_load` at
    /// stride 8, the eight rows of a block on distinct banks), instead of rows
    /// of 32 K (stride 32, four rows of a block on the same banks).
    pub blocks: bool,
    /// The epilogue through W's staging area, a strip of 8 rows per simdgroup
    /// per round, instead of a tile-sized f32 array: 8 KiB (6 on 32 rows) of
    /// threadgroup memory per group instead of 24 (14).
    pub small: bool,
    /// The operands staged as f16 (`bf16 → half` as they are staged,
    /// `simdgroup_half8x8`): exact for values in `[2^-17, 2^16)` or zero, a
    /// rounding otherwise, so not the contract; timed by `gemm.auto`, picked
    /// only with `gemm.half=1` (what llama.cpp's `mul_mm` does on the M1).
    /// Never with the `f16` feature: the operands are f16 already, every
    /// form multiplies `simdgroup_half8x8` ([`MmaForm::of`] drops it).
    pub half: bool,
    /// The K step: 32 (one Q4 block) or 64 (half the barriers).
    pub kstep: u32,
    /// Two staging areas: the next step staged while this one is multiplied,
    /// one barrier a step instead of two.
    pub dbuf: bool,
}

impl MmaForm {
    /// The kernel as first written: 64 rows, rows of 32 K, the f32 tile.
    pub const CLASSIC: MmaForm = MmaForm { bm: 64, blocks: false, small: false, half: false, kstep: MK, dbuf: false };

    /// The form `gemm.mform` names: 0 the classic, 1 by blocks, 2 by blocks
    /// with the small epilogue, 3 the same on 32 rows; from 3: 4 the operands
    /// as f16, 5 a K step of 64, 6 double-buffered, 7 f16 and double-buffered,
    /// 8 f16 on 64 rows, 9 a K step of 64 double-buffered. With the `f16`
    /// feature, 4, 7 and 8 are 3, 6 and 2 (no staging conversion to make).
    pub fn of(n: u32) -> MmaForm {
        let m32 = MmaForm { bm: 32, blocks: true, small: true, ..MmaForm::CLASSIC };
        let half = !crate::core::bf16::F16;
        match n {
            1 => MmaForm { bm: 64, blocks: true, ..MmaForm::CLASSIC },
            2 => MmaForm { bm: 64, blocks: true, small: true, ..MmaForm::CLASSIC },
            3 => m32,
            4 => MmaForm { half, ..m32 },
            5 => MmaForm { kstep: 64, ..m32 },
            6 => MmaForm { dbuf: true, ..m32 },
            7 => MmaForm { half, dbuf: true, ..m32 },
            8 => MmaForm { half, bm: 64, ..m32 },
            9 => MmaForm { kstep: 64, dbuf: true, ..m32 },
            _ => MmaForm::CLASSIC,
        }
    }

    /// What the kernel's name carries.
    fn tag(&self) -> String {
        let rows = if self.bm == MT { String::new() } else { format!("_m{}", self.bm) };
        let kstep = if self.kstep == MK { String::new() } else { format!("_k{}", self.kstep) };
        format!("{rows}{}{}{}{kstep}{}", if self.blocks { "_blk" } else { "" }, if self.small { "_sm" } else { "" }, if self.half { "_h" } else { "" }, if self.dbuf { "_d" } else { "" })
    }
}

/// Groups of a dense matrix-unit GEMM of tiles of `bm` rows × 64 columns.
pub fn gemm_mma_groups(m: u32, n: u32, bm: u32) -> u32 {
    m.div_ceil(bm) * n.div_ceil(MT)
}

/// `Y[m][N] = X[m][K]·Wᵀ` on the matrix units (`gemm.mma=1`): the tiles of `X`
/// and `W` staged in threadgroup memory as bf16 (`uint4` reads, eight at a
/// time), `simdgroup_multiply_accumulate` of bf16 operands into f32
/// accumulators (the product of two bf16 is exact in f32), the accumulators
/// through threadgroup memory for the epilogue of [`Out`], the same as
/// [`gemm_simt`]'s. Rows of the tiles past `m` or `n` read the last valid
/// row and are not stored. Bindings `state, W (bf16), X (bf16), Y`; groups
/// [`gemm_simt_groups`].
pub fn gemm_mma(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool) -> Emitted {
    gemm_mma_rows(kdim, n, out, sg, m_first, MmaRows::Dense)
}

/// [`gemm_mma`] in the form [`MmaForm`] says; groups [`gemm_mma_groups`].
pub fn gemm_mma_form(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, form: MmaForm) -> Emitted {
    gemm_mma_impl(kdim, n, out, sg, m_first, MmaRows::Dense, form, false)
}

/// [`gemm_mma_form`] on `W` in Q4_0 (`kernels::matvec_q4`'s layout): each
/// staged cell of `W` is one word of a block of 32 (the tile's `K` step),
/// decoded there to `bf16(d·q)` as `kernels::dequant_q4` decodes it, so the
/// sums are those of `dequant_q4` then [`gemm_mma_form`], bit for bit.
/// Bindings `state, W (Q4 data), X (bf16), Y, scales (f16)`; groups [`gemm_mma_groups`].
pub fn gemm_mma_q4(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, form: MmaForm) -> Emitted {
    gemm_mma_impl(kdim, n, out, sg, m_first, MmaRows::Dense, form, true)
}

/// `Y[m][N] = X[m][K]·Wᵀ` by Metal 4's `matmul2d` (`gemm.mma=2`): one tile of
/// `tm` × `tn` outputs (16, 32 or 64 each) per group of four simdgroups, all of `K` in one
/// operation read from device memory ([`Kernel::mm2d_tile`]: bf16 products,
/// f32 sums), the tile through threadgroup memory for the epilogue of
/// [`Out`], the same as [`gemm_mma_rows`]'s. Bindings `state, W (bf16), X
/// (bf16), Y`; groups [`gemm_mm2d_groups`].
pub fn gemm_mm2d(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, tm: u32, tn: u32) -> Emitted {
    gemm_mm2d_rows(kdim, n, out, sg, m_first, tm, tn, MmaRows::Dense)
}

/// [`gemm_mm2d`] over the rows [`MmaRows`] says, tiles of `tm` × `tn` (`tm`
/// the tile's `bm` over experts).
/// `matmul2d` reads rows that follow one another: with `Gather`, `X` holds
/// the sorted rows already ([`moe_gather`]: `X[p]` is the token row of pair
/// `list[p]`), so both forms read sorted row `p` at `X[p]`; `Scatter` writes
/// `Y[list[p]]`. Bindings those of [`gemm_mma_rows`]; groups
/// [`gemm_mm2d_expert_groups`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_mm2d_rows(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, tm: u32, tn: u32, rows_of: MmaRows) -> Emitted {
    let sizes = [16, 32, 64];
    assert!(sg == 32 && sizes.contains(&tm) && sizes.contains(&tn) && kdim.is_multiple_of(8), "K = {kdim} (a multiple of 8), tiles of {tm} × {tn} (16, 32 or 64), simdgroup {sg}");
    assert!(rows_of == MmaRows::Dense || tm == rows_of.bm(), "expert tiles of {} rows", rows_of.bm());
    let wg = 128;
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let w = k.buffer(1, Elem::UVec4);
    let x = k.buffer(2, Elem::UVec4);
    let y = k.buffer(3, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
    let experts = rows_of != MmaRows::Dense;
    let (list, table) = if experts { (Some(k.buffer(4, Elem::U32)), Some(k.buffer(5, Elem::U32))) } else { (None, None) };
    let c_s = k.shared(tm * tn, true);
    let m = rows(&mut k, state);
    let g = k.flat_group();
    let n_tiles = n.div_ceil(tn);
    let (gm, gn) = match rows_of {
        MmaRows::Dense => tile_of(&mut k, g, m, tm, n_tiles, m_first),
        MmaRows::Gather { tiles, .. } | MmaRows::Scatter { tiles, .. } => {
            let tc = k.u(tiles);
            let ntc = k.u(n_tiles);
            if m_first { (k.umod(g, tc), k.udiv(g, tc)) } else { (k.udiv(g, ntc), k.umod(g, ntc)) }
        }
    };
    let (tmc, tnc, nc) = (k.u(tm), k.u(tn), k.u(n));
    // The tile from its first sorted row `m0`, its `row_count` rows and its
    // first row of `W` (`w0`); its first column `n0`.
    let tail = |k: &mut Kernel, m0: Word, row_count: Word, w0: Word, n0: Word| {
        let nleft = k.isub(nc, n0);
        let nfull = k.ult(tnc, nleft);
        let col_count = k.select_u(nfull, tnc, nleft);
        k.mm2d_tile(x, m0, row_count, w, w0, col_count, kdim, tm, tn, c_s);
        k.barrier();
        let all = k.u(tm * tn);
        let wgc = k.u(wg);
        let lid = k.local_id();
        k.for_(lid, all, wgc, |k, e| {
            let r = k.udiv(e, tnc);
            let c = k.umod(e, tnc);
            let in_m = k.ult(r, row_count);
            let in_n = k.ult(c, col_count);
            let inside = k.land(in_m, in_n);
            k.if_(inside, |k| {
                let acc = k.sld(c_s, e);
                let row = k.iadd(m0, r);
                let row = match (rows_of, list) {
                    (MmaRows::Scatter { .. }, Some(l)) => k.ld(l, row),
                    _ => row,
                };
                let col = k.iadd(n0, c);
                let yr = k.imul(row, nc);
                let at = k.iadd(yr, col);
                match out {
                    Out::F32 => k.st(y, at, acc),
                    Out::Bf16 => {
                        let b = k.f_to_bf16(acc);
                        k.st(y, at, b);
                    }
                    Out::Residual => {
                        let r = k.round_bf16(acc);
                        let old = k.ld(y, at);
                        let of = k.bf16_to_f(old);
                        let s = k.fadd(of, r);
                        let b = k.f_to_bf16(s);
                        k.st(y, at, b);
                    }
                    Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
                }
            });
        });
    };
    match table {
        None => {
            let m0 = k.imul(gm, tmc);
            let n0 = k.imul(gn, tnc);
            let left = k.isub(m, m0);
            let full = k.ult(tmc, left);
            let row_count = k.select_u(full, tmc, left);
            tail(&mut k, m0, row_count, n0, n0);
        }
        Some(tb) => {
            let n0 = k.imul(gn, tnc);
            let zero = k.u(0);
            let count = k.ld(tb, zero);
            let valid = k.ult(gm, count);
            k.if_(valid, |k| {
                let (one, three) = (k.u(1), k.u(3));
                let at = k.imul(gm, three);
                let at = k.iadd(at, one);
                let e = k.ld(tb, at);
                let a1 = k.iadd(at, one);
                let start = k.ld(tb, a1);
                let a2 = k.iadd(a1, one);
                let cnt = k.ld(tb, a2);
                let w_base = k.imul(e, nc);
                let w0 = k.iadd(w_base, n0);
                tail(k, start, cnt, w0, n0);
            });
        }
    }
    let tag = match out {
        Out::Bf16 => "bf16",
        Out::F32 => "f32",
        Out::Residual => "res",
        Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
    };
    let rows_tag = match rows_of {
        MmaRows::Dense => String::new(),
        MmaRows::Gather { top_k, tiles, bm } => format!("_gather_k{top_k}_t{tiles}_m{bm}"),
        MmaRows::Scatter { tiles, bm } => format!("_scatter_t{tiles}_m{bm}"),
    };
    let tm_tag = if tm == MT || experts { String::new() } else { format!("_r{tm}") };
    done(format!("gemm_mm2d_{kdim}x{n}_{tag}_t{tn}{tm_tag}{}{rows_tag}", if m_first { "_mf" } else { "" }), k, if experts { 6 } else { 4 })
}

/// Groups of a [`gemm_mm2d_rows`] over expert tiles: every tile the table
/// may hold times the column tiles of `tn`; the groups past the table's count return.
pub fn gemm_mm2d_expert_groups(n: u32, tiles: u32, tn: u32) -> u32 {
    tiles * n.div_ceil(tn)
}

/// The rows of `X` the experts' `[gate | up]` reads, in sorted order:
/// `XS[p] = X[list[p] / top_k]` for the `m·top_k` pairs of the block, a
/// `uint4` per invocation. Bindings `state, list, X, XS` (bf16 `[·][hidden]`);
/// groups [`moe_gather_groups`].
pub fn moe_gather(hidden: u32, top_k: u32, sg: u32) -> Emitted {
    assert!(hidden.is_multiple_of(8));
    let mut k = Kernel::new(256, sg);
    let state = k.buffer(0, Elem::U32);
    let list = k.buffer(1, Elem::U32);
    let x = k.buffer(2, Elem::UVec4);
    let xs = k.buffer(3, Elem::UVec4);
    let cells = k.u(hidden / 8);
    let kk = k.u(top_k);
    let m = rows(&mut k, state);
    let pairs = k.imul(m, kk);
    let total = k.imul(pairs, cells);
    let i = k.global_id();
    let inside = k.ult(i, total);
    k.if_(inside, |k| {
        let p = k.udiv(i, cells);
        let c = k.umod(i, cells);
        let pair = k.ld(list, p);
        let row = k.udiv(pair, kk);
        let at = k.imul(row, cells);
        let at = k.iadd(at, c);
        let v = k.ld(x, at);
        k.st(xs, i, v);
    });
    done(format!("moe_gather_{hidden}_k{top_k}"), k, 4)
}

/// Groups of a [`moe_gather`] of up to `rows` rows.
pub fn moe_gather_groups(rows: u32, top_k: u32, hidden: u32) -> u32 {
    (rows * top_k * hidden / 8).div_ceil(256)
}

/// Groups of a [`gemm_mm2d`] of `m` rows: tiles of `tm` × `tn`.
pub fn gemm_mm2d_groups(m: u32, n: u32, tm: u32, tn: u32) -> u32 {
    m.div_ceil(tm) * n.div_ceil(tn)
}

/// The tile `(gm, gn)` of group `tgid.x` for [`gemm_nax`] (MSL text): `band = 0`
/// consecutive groups walk the row tiles of one column tile ([`tile_of`]'s
/// `m_first`); `band > 0` they walk every column tile of `band` row tiles,
/// the row tiles of the band interleaved (mlx's swizzle, 2 on the M5), the
/// last band holding what is left.
fn nax_tile_order(w: &mut impl FnMut(&str), bm: u32, n_tiles: u32, band: u32) {
    w(&format!("const uint mb = (m + {}u) / {bm}u;", bm - 1));
    if band == 0 {
        w("const uint gm = tgid.x % mb, gn = tgid.x / mb;");
    } else {
        w(&format!("const uint full = mb / {band}u, per = {}u;", band * n_tiles));
        w(&format!("uint gm, gn; if (tgid.x < full * per) {{ const uint idx = tgid.x % per; gm = (tgid.x / per) * {band}u + idx % {band}u; gn = idx / {band}u; }} else {{ const uint rem = mb - full * {band}u, idx = tgid.x - full * per; gm = full * {band}u + idx % rem; gn = idx / rem; }}"));
    }
}

/// `Y[m][N] = X[m][K]·Wᵀ` by `matmul2d` of 16 × 32 × 16 run by one simdgroup
/// each on register fragments (mlx's `steel_gemm_fused_nax` on the M5): one
/// tile of `32·wm × 32·wn` outputs per group of `wm × wn` simdgroups, each
/// simdgroup 32 × 32 (four f32 fragments), `K` walked by 32, the fragments of
/// `X` and `W` read straight from device memory (no stage: the groups of a
/// row share it through the caches), the rows past `m` and `N` clamped to the
/// last (their outputs are not written). The epilogue of [`gemm_mm2d`], value
/// by value from the fragments. `relaxed`: the descriptor's reduced precision;
/// `two`: mlx's loops over `K` (512 at a time with `threadgroup_barrier(mem_none)`,
/// then 16 steps of 32, neither unrolled, a volatile local in the body, `K`
/// hidden from the compiler): measured ×1,5–2 on `down` (K = 6144) on the M5
/// (`herbert-metal-probe naxg`), even at K = 2048. `K` a multiple of 512 then.
/// Bindings `state, W (16-bit), X (16-bit), Y`; groups [`gemm_mm2d_groups`]
/// with `tm = 32·wm`, `tn = 32·wn`. `K` a multiple of 32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nax(kdim: u32, n: u32, out: Out, sg: u32, band: u32, wm: u32, wn: u32, relaxed: bool, two: bool) -> Emitted {
    assert!(sg == 32 && kdim.is_multiple_of(if two { 512 } else { 32 }) && (1..=4).contains(&wm) && (1..=4).contains(&wn), "gemm_nax: K = {kdim} (a multiple of 32, of 512 with two), {wm} × {wn} simdgroups, simdgroup {sg}");
    let (bm, bn) = (32 * wm, 32 * wn);
    let mut k = Kernel::new(wm * wn * sg, sg);
    let _state = k.buffer(0, Elem::U32);
    let _w = k.buffer(1, Elem::UVec4);
    let _x = k.buffer(2, Elem::UVec4);
    let y = k.buffer(3, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
    k.written(y);
    let t = k.t16_name();
    let n_tiles = n.div_ceil(bn);
    let mut src = String::new();
    let mut w = |l: &str| {
        src.push_str(l);
        src.push('\n');
    };
    w(&format!("const uint m = b0[{STATE_ROWS}];"));
    nax_tile_order(&mut w, bm, n_tiles, band);
    w(&format!("const ushort sgm = ushort(sgid) / {wn}, sgn = ushort(sgid) % {wn};"));
    w("const ushort qid = ushort(lane) >> 2;");
    w("const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);");
    w("const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;");
    w(&format!("const uint m0 = gm * {bm}u + sgm * 32u, n0 = gn * {bn}u + sgn * 32u;"));
    w(&format!("typedef vec<{t}, 8> v8; typedef vec<{t}, 4> v4; typedef vec<float, 8> f8;"));
    // The row pointers of this lane's fragments: X rows m0 + 16·a + fm + 8·i, W rows n0 + 16·b + fm + 8·i.
    for a in 0..2 {
        for i in 0..2 {
            w(&format!("const device {t} *xa{a}{i} = (const device {t} *)b2 + min(m0 + {}u + fm, m - 1u) * {kdim}u + fn;", 16 * a + 8 * i));
            w(&format!("const device {t} *wb{a}{i} = (const device {t} *)b1 + min(n0 + {}u + fm, {}u) * {kdim}u + fn;", 16 * a + 8 * i, n - 1));
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
            body.push_str(&k.nax_mma_text(&format!("d{mm}0"), &format!("d{mm}1"), &format!("a{mm}{kk}"), &format!("b0{kk}"), &format!("b1{kk}"), true, relaxed));
            body.push('\n');
        }
    }
    if two {
        // `K` from an expression the compiler cannot fold (m is never ~0): the loops stay loops.
        w(&format!("const uint kk_end = m == 0xffffffffu ? 0u : {kdim}u;"));
        w("#pragma clang loop unroll(disable)");
        w("for (uint kb = 0; kb < kk_end; kb += 512u) {");
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
        w(&format!("for (uint k0 = 0; k0 < {kdim}u; k0 += 32u) {{"));
        for l in body.lines() {
            w(&format!("    {l}"));
        }
        w("}");
    }
    let store = |y: &str, at: &str, acc: &str| -> String {
        match out {
            Out::F32 => format!("{y}[{at}] = {acc};"),
            Out::Bf16 => format!("{y}[{at}] = ushort({});", k.f_to_16_text(acc)),
            Out::Residual => {
                let r = k.h16_to_f_text(&k.f_to_16_text(acc));
                let old = k.h16_to_f_text(&format!("{y}[{at}]"));
                format!("{{ const float rr = {r}; const float of = {old}; {y}[{at}] = ushort({}); }}", k.f_to_16_text("of + rr"))
            }
            Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
        }
    };
    for mm in 0..2 {
        for nn in 0..2 {
            w(&format!("for (ushort i = 0; i < 2; ++i) {{ const uint r = m0 + {}u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {{", 16 * mm));
            w(&format!("    const uint c = n0 + {}u + fn + j;", 16 * nn));
            w(&format!("    if (c < {n}u) {{ const float acc = d{mm}{nn}[i * 4 + j]; const uint at = r * {n}u + c; {} }}", store("b3", "at", "acc")));
            w("} }");
        }
    }
    k.raw(&src);
    let tag = match out {
        Out::Bf16 => "bf16",
        Out::F32 => "f32",
        Out::Residual => "res",
        Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
    };
    done(format!("gemm_nax_{kdim}x{n}_{tag}_{bm}x{bn}_b{band}{}{}", if relaxed { "_relaxed" } else { "" }, if two { "_two" } else { "" }), k, 4)
}

/// The `[gate | up]` projection and the SwiGLU in one: [`gemm_nax`] where
/// each simdgroup computes 32 rows × 16 features, its two fragments of
/// columns the gate rows `f..f + 16` of `W` and their up rows `nf + f..`, so
/// a lane holds the gate and the up of the same outputs; it writes
/// `h = round16(round16(g)·σ(round16(g))·round16(u))`, the values
/// [`gemm_mm2d`] (`Out::Bf16`) then [`swiglu_rows`] give, bit for bit (the
/// sums of the GEMM in their own order). Bindings `state, W ([2·nf][K]
/// 16-bit), X, H ([m][nf] 16-bit)`; groups `ceil(m / (32·wm)) · ceil(nf / (16·wn))`.
pub fn gemm_nax_swiglu(kdim: u32, nf: u32, sg: u32, band: u32, wm: u32, wn: u32) -> Emitted {
    assert!(sg == 32 && kdim.is_multiple_of(32) && (1..=4).contains(&wm) && (1..=4).contains(&wn), "gemm_nax_swiglu: K = {kdim} (a multiple of 32), {wm} × {wn} simdgroups, simdgroup {sg}");
    let (bm, bf) = (32 * wm, 16 * wn);
    let mut k = Kernel::new(wm * wn * sg, sg);
    let _state = k.buffer(0, Elem::U32);
    let _w = k.buffer(1, Elem::UVec4);
    let _x = k.buffer(2, Elem::UVec4);
    let hb = k.buffer(3, Elem::U16);
    k.written(hb);
    let t = k.t16_name();
    let f_tiles = nf.div_ceil(bf);
    let mut src = String::new();
    let mut w = |l: &str| {
        src.push_str(l);
        src.push('\n');
    };
    w(&format!("const uint m = b0[{STATE_ROWS}];"));
    nax_tile_order(&mut w, bm, f_tiles, band);
    w(&format!("const ushort sgm = ushort(sgid) / {wn}, sgn = ushort(sgid) % {wn};"));
    w("const ushort qid = ushort(lane) >> 2;");
    w("const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);");
    w("const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;");
    w(&format!("const uint m0 = gm * {bm}u + sgm * 32u, f0 = gn * {bf}u + sgn * 16u;"));
    w(&format!("typedef vec<{t}, 8> v8; typedef vec<{t}, 4> v4; typedef vec<float, 8> f8;"));
    for a in 0..2 {
        for i in 0..2 {
            w(&format!("const device {t} *xa{a}{i} = (const device {t} *)b2 + min(m0 + {}u + fm, m - 1u) * {kdim}u + fn;", 16 * a + 8 * i));
        }
    }
    // Fragment 0 of W: the gate rows; fragment 1: their up rows.
    for (b, off) in [(0, 0), (1, nf)] {
        for i in 0..2 {
            w(&format!("const device {t} *wb{b}{i} = (const device {t} *)b1 + ({off}u + min(f0 + {}u + fm, {}u)) * {kdim}u + fn;", 8 * i, nf - 1));
        }
    }
    w("f8 d00 = f8(0.0f), d01 = f8(0.0f), d10 = f8(0.0f), d11 = f8(0.0f);");
    w(&format!("for (uint k0 = 0; k0 < {kdim}u; k0 += 32u) {{"));
    for (p, nm, rs) in [("xa", "a", 2), ("wb", "b", 2)] {
        for r in 0..rs {
            for kk in 0..2 {
                w(&format!(
                    "    v8 {nm}{r}{kk}; {{ v4 x0 = *(const device v4 *)({p}{r}0 + k0 + {o}u); v4 x1 = *(const device v4 *)({p}{r}1 + k0 + {o}u); {nm}{r}{kk} = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }}",
                    o = 16 * kk
                ));
            }
        }
    }
    for mm in 0..2 {
        for kk in 0..2 {
            let text = k.nax_mma_text(&format!("d{mm}0"), &format!("d{mm}1"), &format!("a{mm}{kk}"), &format!("b0{kk}"), &format!("b1{kk}"), true, false);
            for l in text.lines() {
                w(&format!("    {l}"));
            }
        }
    }
    w("}");
    let g = k.h16_to_f_text(&k.f_to_16_text("gacc"));
    let u = k.h16_to_f_text(&k.f_to_16_text("uacc"));
    let o = k.f_to_16_text("(g * (1.0f / (1.0f + exp(-g)))) * u");
    for mm in 0..2 {
        w(&format!("for (ushort i = 0; i < 2; ++i) {{ const uint r = m0 + {}u + fm + 8u * i; if (r < m) for (ushort j = 0; j < 4; ++j) {{", 16 * mm));
        w("    const uint f = f0 + fn + j;");
        w(&format!("    if (f < {nf}u) {{ const float gacc = d{mm}0[i * 4 + j], uacc = d{mm}1[i * 4 + j]; const float g = {g}; const float u = {u}; b3[r * {nf}u + f] = ushort({o}); }}"));
        w("} }");
    }
    k.raw(&src);
    done(format!("gemm_nax_swiglu_{kdim}x{nf}_{bm}x{bf}_b{band}"), k, 4)
}

/// Groups of a [`gemm_mma_rows`] over expert tiles: every tile the table may
/// hold times the column tiles; the groups past the table's count return.
pub fn gemm_mma_expert_groups(n: u32, tiles: u32) -> u32 {
    tiles * n.div_ceil(MT)
}

/// Expert tiles of a block of `rows` rows: every (row, slot) in tiles of
/// `bm` rows, one partial tile at most per expert.
pub fn moe_tiles(rows: u32, top_k: u32, experts: u32, bm: u32) -> u32 {
    (rows * top_k).div_ceil(bm) + experts
}

/// [`gemm_mma`] over the rows [`MmaRows`] says. With experts, bindings
/// `state, W, X, Y, list (u32), tiles (u32)`; groups [`gemm_mma_expert_groups`].
pub fn gemm_mma_rows(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, rows_of: MmaRows) -> Emitted {
    gemm_mma_impl(kdim, n, out, sg, m_first, rows_of, MmaForm { bm: rows_of.bm(), ..MmaForm::CLASSIC }, false)
}

/// [`gemm_mma_rows`] with `W` in Q4_0 as [`gemm_mma_q4`] reads it (the
/// experts' stack one `q4::quantize` of its `experts·n` rows). Bindings those
/// of [`gemm_mma_rows`], then the scales (f16).
pub fn gemm_mma_rows_q4(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, rows_of: MmaRows) -> Emitted {
    gemm_mma_impl(kdim, n, out, sg, m_first, rows_of, MmaForm { bm: rows_of.bm(), ..MmaForm::CLASSIC }, true)
}

/// [`gemm_mma_rows`] in the form `form` (its `bm` those of `rows_of` over
/// experts); `q4`: `W` in Q4_0 ([`gemm_mma_q4`], [`gemm_mma_rows_q4`]).
#[allow(clippy::too_many_arguments)]
fn gemm_mma_impl(kdim: u32, n: u32, out: Out, sg: u32, m_first: bool, rows_of: MmaRows, form: MmaForm, q4: bool) -> Emitted {
    assert!(sg == 32 && kdim.is_multiple_of(MK), "K = {kdim} (a multiple of {MK}), simdgroup {sg}");
    let bm = form.bm;
    assert!(rows_of == MmaRows::Dense || bm == rows_of.bm(), "expert tiles of {} rows", rows_of.bm());
    assert!(bm == 16 || bm == 32 || bm == 64, "tile of {bm} rows");
    // Four simdgroups: 2 × 2 of 32 × 32 for 64 rows, 1 × 4 of bm × 16 below.
    let sg_rows = if bm == 64 { 2 } else { 1 };
    let (tm, tn) = (bm / sg_rows, MT / (4 / sg_rows));
    let (mi_n, mj_n) = (tm / 8, tn / 8);
    let wg = 128;
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let w = k.buffer(1, if q4 { Elem::U32 } else { Elem::UVec4 });
    let x = k.buffer(2, Elem::UVec4);
    let y = k.buffer(3, if out == Out::F32 { Elem::F32 } else { Elem::U16 });
    let experts = rows_of != MmaRows::Dense;
    let (list, table) = if experts { (Some(k.buffer(4, Elem::U32)), Some(k.buffer(5, Elem::U32))) } else { (None, None) };
    let scales = if q4 { Some(k.buffer(if experts { 6 } else { 4 }, Elem::U16)) } else { None };
    let mk = form.kstep;
    assert!((mk == 32 || mk == 64) && kdim.is_multiple_of(mk), "K = {kdim}, a K step of {mk}");
    let (a_cells, b_cells) = (bm * mk / 8, MT * mk / 8);
    let nbuf = if form.dbuf { 2 } else { 1 };
    let a_s = k.shared_vec4(a_cells * nbuf);
    let b_s = k.shared_vec4(b_cells * nbuf);
    // The f32 tile of the epilogue, or (small) W's staging area seen as floats, a strip of 8 rows per simdgroup per round.
    let c_s = if form.small { k.shared_f32_view(b_s) } else { k.shared(bm * MT, true) };
    let m = rows(&mut k, state);
    let g = k.flat_group();
    let n_tiles = n.div_ceil(MT);
    let lid = k.local_id();
    let nc = k.u(n);
    let tile = k.u(MT);
    let rtile = k.u(bm);
    let one = k.u(1);
    let zero_u = k.u(0);
    // The tile: its first row of outputs (`m0`, dense) or its expert, first
    // sorted row and row count (experts); its first column.
    let (gm, gn) = match rows_of {
        MmaRows::Dense => tile_of(&mut k, g, m, bm, n_tiles, m_first),
        MmaRows::Gather { tiles, .. } | MmaRows::Scatter { tiles, .. } => {
            let tc = k.u(tiles);
            let ntc = k.u(n_tiles);
            if m_first { (k.umod(g, tc), k.udiv(g, tc)) } else { (k.udiv(g, ntc), k.umod(g, ntc)) }
        }
    };
    let n0 = k.imul(gn, tile);
    let valid = match table {
        Some(tb) => {
            let count = k.ld(tb, zero_u);
            k.ult(gm, count)
        }
        None => k.ieq(zero_u, zero_u),
    };
    k.if_(valid, |k| {
        // (row base, rows in the tile, expert's first row of W)
        let (m0, row_count, w_base) = match table {
            Some(tb) => {
                let three = k.u(3);
                let at = k.imul(gm, three);
                let at = k.iadd(at, one);
                let e = k.ld(tb, at);
                let a1 = k.iadd(at, one);
                let start = k.ld(tb, a1);
                let a2 = k.iadd(a1, one);
                let cnt = k.ld(tb, a2);
                (start, cnt, k.imul(e, nc))
            }
            None => {
                let m0 = k.imul(gm, rtile);
                let left = k.isub(m, m0);
                let full = k.ult(rtile, left);
                let cnt = k.select_u(full, rtile, left);
                (m0, cnt, zero_u)
            }
        };
        let last_r = k.isub(row_count, one);
        // The row of X that local row `r` (clamped to the tile's rows) reads.
        let x_row = |k: &mut Kernel, r: Word| -> Word {
            let over = k.ult(last_r, r);
            let r = k.select_u(over, last_r, r);
            let p = k.iadd(m0, r);
            match (rows_of, list) {
                (MmaRows::Gather { top_k, .. }, Some(l)) => {
                    let i = k.ld(l, p);
                    let kk = k.u(top_k);
                    k.udiv(i, kk)
                }
                _ => p,
            }
        };
        let sgid = k.subgroup_id();
        let sgc = k.u(4 / sg_rows);
        let sm = k.udiv(sgid, sgc);
        let sn = k.umod(sgid, sgc);
        let accs: Vec<_> = (0..mi_n * mj_n).map(|_| k.mat_acc()).collect();
        let row_cells = k.u(kdim / 8);
        let last_n = k.u(n - 1);
        let cells = k.u(mk / 8);
        let eight = k.u(8);
        let kc = k.u(kdim);
        let step = k.u(mk);
        // Where cell (`r`, `c`) of a staged tile of `rows` rows goes: row-major
        // (`r·4 + c`), or by 8 × 8 blocks (`(c·rows/8 + r/8)·8 + r mod 8`), a
        // cell being one row of a block.
        let staged = |k: &mut Kernel, e: Word, r: Word, c: Word, rows: u32| -> Word {
            if !form.blocks {
                return e;
            }
            let r8 = k.udiv(r, eight);
            let rm = k.umod(r, eight);
            let per_c = k.u(rows / 8);
            let t = k.imul(c, per_c);
            let t = k.iadd(t, r8);
            let t = k.imul(t, eight);
            k.iadd(t, rm)
        };
        // `MmaForm::half`: the eight bf16 of a staged cell as f16, and whether
        // all eight are exact in f16 (each read back equal to itself; NaN and
        // an overflow are not).
        let to_half = |k: &mut Kernel, v: Word| -> (Word, Word) {
            let sixteen = k.u(16);
            let mut exact = k.ieq(zero_u, zero_u);
            let words: Vec<Word> = (0..4)
                .map(|c| {
                    let w = k.comp(v, c);
                    let lo = k.lo_f(w);
                    let hi = k.hi_f(w);
                    let lh = k.f_to_f16(lo);
                    let hh = k.f_to_f16(hi);
                    let lb = k.f16_to_f(lh);
                    let hb = k.f16_to_f(hh);
                    let e1 = k.feq(lb, lo);
                    let e2 = k.feq(hb, hi);
                    let e = k.land(e1, e2);
                    exact = k.land(exact, e);
                    let hh = k.shl(hh, sixteen);
                    k.or(lh, hh)
                })
                .collect();
            (k.uvec4([words[0], words[1], words[2], words[3]]), exact)
        };
        // `MmaForm::half`: step `i` flags its area `i mod 2` with `i + 1` when
        // a cell of X could not be staged as f16 (W's are, its scales checked
        // at load); a stale flag carries another step's tag.
        let hflag = if form.half { Some(k.shared(2, false)) } else { None };
        let step_tag = |k: &mut Kernel, k0: Word| -> (Word, Word) {
            let i = k.udiv(k0, step);
            let slot = k.and(i, one);
            let tag = k.iadd(i, one);
            (slot, tag)
        };
        // Stage X's tile of the K step at `k0` into area `sel` (0 or 1): bm·mk/8
        // cells; `raw`: as bf16 whatever the form (the fallback of a flagged step).
        let stage_x = |k: &mut Kernel, k0: Word, sel: Word, raw: bool| {
            let c0 = k.udiv(k0, eight);
            let ac = k.u(a_cells);
            let a_base = k.imul(sel, ac);
            let (slot, tag) = step_tag(k, k0);
            for q in 0..a_cells.div_ceil(wg) {
                let qc = k.u(q * wg);
                let e = k.iadd(lid, qc);
                let lim = k.u(a_cells);
                let in_a = k.ult(e, lim);
                k.if_(in_a, |k| {
                    let r = k.udiv(e, cells);
                    let c = k.umod(e, cells);
                    let cc = k.iadd(c0, c);
                    let row = x_row(k, r);
                    let xa = k.imul(row, row_cells);
                    let xa = k.iadd(xa, cc);
                    let xv = k.ld(x, xa);
                    let at = staged(k, e, r, c, bm);
                    let at = k.iadd(at, a_base);
                    match hflag {
                        Some(flag) if !raw => {
                            let (hv, exact) = to_half(k, xv);
                            k.if_else(
                                exact,
                                |k| k.sst(a_s, at, hv),
                                |k| {
                                    k.sst(a_s, at, xv);
                                    k.sst(flag, slot, tag);
                                },
                            );
                        }
                        _ => k.sst(a_s, at, xv),
                    }
                });
            }
        };
        // Stage W's tile of the K step at `k0` into area `sel`: MT·mk/8 cells, as
        // f16 with `MmaForm::half` unless `raw` (the fallback of a flagged step).
        let stage_w = |k: &mut Kernel, k0: Word, sel: Word, raw: bool| {
            let c0 = k.udiv(k0, eight);
            let bc = k.u(b_cells);
            let b_base = k.imul(sel, bc);
            for q in 0..b_cells / wg {
                let qc = k.u(q * wg);
                let e = k.iadd(lid, qc);
                let r = k.udiv(e, cells);
                let c = k.umod(e, cells);
                let cc = k.iadd(c0, c);
                let col = k.iadd(n0, r);
                let over = k.ult(last_n, col);
                let col = k.select_u(over, last_n, col);
                let col = k.iadd(col, w_base);
                let wv = match scales {
                    // Block `k0/32 + c/4` of the row (a block is 32 K, four cells), its word `c mod 4`.
                    Some(sc) => {
                        let nblk = k.u(kdim / 32);
                        let t32 = k.u(32);
                        let kb = k.udiv(k0, t32);
                        let four = k.u(4);
                        let cb = k.udiv(c, four);
                        let cw = k.umod(c, four);
                        let blk = k.imul(col, nblk);
                        let blk = k.iadd(blk, kb);
                        let blk = k.iadd(blk, cb);
                        let wi = k.imul(blk, four);
                        let wi = k.iadd(wi, cw);
                        let ww = k.ld(w, wi);
                        let dh = k.ld(sc, blk);
                        let d = k.f16_to_f(dh);
                        super::kernels::q4_word_bf16(k, ww, d)
                    }
                    None => {
                        let wa = k.imul(col, row_cells);
                        let wa = k.iadd(wa, cc);
                        k.ld(w, wa)
                    }
                };
                let wv = if form.half && !raw { to_half(k, wv).0 } else { wv };
                let at = staged(k, e, r, c, MT);
                let at = k.iadd(at, b_base);
                k.sst(b_s, at, wv);
            }
        };
        let stage = |k: &mut Kernel, k0: Word, sel: Word| {
            stage_x(k, k0, sel, false);
            stage_w(k, k0, sel, false);
        };
        // The MMAs of the staged step in area `sel`, the operands f16 (`half`) or bf16.
        let compute = |k: &mut Kernel, sel: Word, half: bool| {
            let ae = k.u(a_cells * 8);
            let a_el = k.imul(sel, ae);
            let be = k.u(b_cells * 8);
            let b_el = k.imul(sel, be);
            let load = |k: &mut Kernel, sh: crate::msl::Shared, at: Word, stride: u32, transpose: bool| -> Word {
                if half { k.mat_load_f16(sh, at, stride, transpose) } else { k.mat_load_bf16(sh, at, stride, transpose) }
            };
            for kk in (0..mk).step_by(8) {
                let kkc = k.u(kk);
                let span_a = k.u(tm * mk);
                let span_b = k.u(tn * mk);
                // By blocks: block (`kk/8`, row block) of the simdgroup's strip, 64 elements each, stride 8.
                let blk_a = k.u(kk / 8 * (bm / 8));
                let blk_b = k.u(kk / 8 * (MT / 8));
                let sixty_four = k.u(64);
                let a: Vec<_> = (0..mi_n)
                    .map(|i| {
                        if form.blocks {
                            let rb = k.u(tm / 8);
                            let base = k.imul(sm, rb);
                            let ib = k.u(i);
                            let at = k.iadd(base, ib);
                            let at = k.iadd(at, blk_a);
                            let at = k.imul(at, sixty_four);
                            let at = k.iadd(at, a_el);
                            load(k, a_s, at, 8, false)
                        } else {
                            let base = k.imul(sm, span_a);
                            let off = k.u(i * 8 * mk);
                            let at = k.iadd(base, off);
                            let at = k.iadd(at, kkc);
                            let at = k.iadd(at, a_el);
                            load(k, a_s, at, mk, false)
                        }
                    })
                    .collect();
                let b: Vec<_> = (0..mj_n)
                    .map(|j| {
                        if form.blocks {
                            let cb = k.u(tn / 8);
                            let base = k.imul(sn, cb);
                            let jb = k.u(j);
                            let at = k.iadd(base, jb);
                            let at = k.iadd(at, blk_b);
                            let at = k.imul(at, sixty_four);
                            let at = k.iadd(at, b_el);
                            load(k, b_s, at, 8, true)
                        } else {
                            let base = k.imul(sn, span_b);
                            let off = k.u(j * 8 * mk);
                            let at = k.iadd(base, off);
                            let at = k.iadd(at, kkc);
                            let at = k.iadd(at, b_el);
                            load(k, b_s, at, mk, true)
                        }
                    })
                    .collect();
                for i in 0..mi_n as usize {
                    for j in 0..mj_n as usize {
                        k.mat_mad(accs[i * mj_n as usize + j], a[i], b[j]);
                    }
                }
            }
        };
        // The step at `k0` staged in area `sel`, after the barrier: `MmaForm::half`
        // reads its flag (the same for the whole group); a flagged step has X
        // staged again as bf16 (W as f16 is exact, but the matrices must be
        // of one type: W too, then) and multiplies as bf16, the others as f16.
        let multiply = |k: &mut Kernel, k0: Word, sel: Word| match hflag {
            Some(flag) => {
                let (slot, tag) = step_tag(k, k0);
                let f = k.sld(flag, slot);
                let inexact = k.ieq(f, tag);
                k.if_(inexact, |k| {
                    stage_x(k, k0, sel, true);
                    stage_w(k, k0, sel, true);
                    k.barrier();
                });
                k.if_else(inexact, |k| compute(k, sel, false), |k| compute(k, sel, true));
            }
            None => compute(k, sel, false),
        };
        if form.dbuf {
            // Step `i` multiplied from area `i mod 2` while step `i + 1` is staged into the other.
            stage(k, zero_u, zero_u);
            k.barrier();
            k.for_(zero_u, kc, step, |k, k0| {
                let i = k.udiv(k0, step);
                let sel = k.and(i, one);
                let nsel = k.isub(one, sel);
                let next = k.iadd(k0, step);
                let has_next = k.ult(next, kc);
                k.if_(has_next, |k| stage(k, next, nsel));
                multiply(k, k0, sel);
                k.barrier();
            });
        } else {
            k.for_(zero_u, kc, step, |k, k0| {
                stage(k, k0, zero_u);
                k.barrier();
                multiply(k, k0, zero_u);
                k.barrier();
            });
        }
        // One output (row `r` of the tile, column `c` of the tile) from `acc`.
        let emit_out = |k: &mut Kernel, r: Word, c: Word, acc: Word| {
            let col = k.iadd(n0, c);
            let in_m = k.ult(r, row_count);
            let in_n = k.ult(col, nc);
            let inside = k.land(in_m, in_n);
            k.if_(inside, |k| {
                let p = k.iadd(m0, r);
                let row = match (rows_of, list) {
                    (MmaRows::Scatter { .. }, Some(l)) => k.ld(l, p),
                    _ => p,
                };
                let yr = k.imul(row, nc);
                let at = k.iadd(yr, col);
                match out {
                    Out::F32 => k.st(y, at, acc),
                    Out::Bf16 => {
                        let b = k.f_to_bf16(acc);
                        k.st(y, at, b);
                    }
                    Out::Residual => {
                        let r = k.round_bf16(acc);
                        let old = k.ld(y, at);
                        let of = k.bf16_to_f(old);
                        let s = k.fadd(of, r);
                        let b = k.f_to_bf16(s);
                        k.st(y, at, b);
                    }
                    Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
                }
            });
        };
        let wgc = k.u(wg);
        if form.small {
            // Round `i`: every simdgroup stores its strip `i` (8 rows × tn) into W's staging area
            // (simdgroup-major, 8·tn floats each: 4·8·tn ≤ 1024, the area's size), the group writes it out.
            let strip = k.u(8 * tn);
            let sgoff = k.imul(sgid, strip);
            let all = k.u(4 * 8 * tn);
            let tnc = k.u(tn);
            let tmc = k.u(tm);
            for i in 0..mi_n {
                for j in 0..mj_n {
                    let off = k.u(j * 8);
                    let at = k.iadd(sgoff, off);
                    k.mat_store(accs[(i * mj_n + j) as usize], c_s, at, tn);
                }
                k.barrier();
                let i8 = k.u(i * 8);
                k.for_(lid, all, wgc, |k, e| {
                    let s = k.udiv(e, strip);
                    let rem = k.umod(e, strip);
                    let rr = k.udiv(rem, tnc);
                    let c = k.umod(rem, tnc);
                    let s_m = k.udiv(s, sgc);
                    let s_n = k.umod(s, sgc);
                    let r = k.imul(s_m, tmc);
                    let r = k.iadd(r, i8);
                    let r = k.iadd(r, rr);
                    let cc = k.imul(s_n, tnc);
                    let cc = k.iadd(cc, c);
                    let acc = k.sld(c_s, e);
                    emit_out(k, r, cc, acc);
                });
                k.barrier();
            }
        } else {
            let rows_span = k.u(tm * MT);
            let cols_span = k.u(tn);
            let rb = k.imul(sm, rows_span);
            let cb = k.imul(sn, cols_span);
            let base = k.iadd(rb, cb);
            for i in 0..mi_n {
                for j in 0..mj_n {
                    let off = k.u(i * 8 * MT + j * 8);
                    let at = k.iadd(base, off);
                    k.mat_store(accs[(i * mj_n + j) as usize], c_s, at, MT);
                }
            }
            k.barrier();
            let all = k.u(bm * MT);
            let mtc = k.u(MT);
            k.for_(lid, all, wgc, |k, e| {
                let r = k.udiv(e, mtc);
                let c = k.umod(e, mtc);
                let acc = k.sld(c_s, e);
                emit_out(k, r, c, acc);
            });
        }
    });
    let tag = match out {
        Out::Bf16 => "bf16",
        Out::F32 => "f32",
        Out::Residual => "res",
        Out::Swiglu => unreachable!("Out::Swiglu: the decode step's matvec only"),
    };
    let rows_tag = match rows_of {
        MmaRows::Dense => String::new(),
        MmaRows::Gather { top_k, tiles, bm } => format!("_gather_k{top_k}_t{tiles}_m{bm}"),
        MmaRows::Scatter { tiles, bm } => format!("_scatter_t{tiles}_m{bm}"),
    };
    done(format!("gemm_mma_{kdim}x{n}_{tag}{}{rows_tag}{}{}", if m_first { "_mf" } else { "" }, form.tag(), if q4 { "_q4" } else { "" }), k, 4 + 2 * experts as u32 + q4 as u32)
}

/// The routed (row, slot) pairs of a block sorted by expert, and the table of
/// the expert tiles. `list[p]` is the pair `row·top_k + slot`, the pairs of
/// expert `e` together (in increasing pair order), the experts in increasing
/// order; `tiles[0]` is the number of tiles, tile `t` is `tiles[1 + 3t..]` =
/// (expert, first sorted row, rows ≤ `bm`). One group of `experts` threads, the
/// route's experts staged in threadgroup memory when they fit in 32 KiB with
/// the offsets (blocks of up to 512 rows at `top_k` 8), by slices of
/// [`SORT_SLICE`] pairs otherwise. Bindings `state, route ([rows][2·top_k]),
/// list, tiles`; one group.
/// Pairs [`moe_sort`] stages at a time when a block's do not fit.
pub const SORT_SLICE: u32 = 4096;

pub fn moe_sort(experts: u32, top_k: u32, rows_max: u32, bm: u32, sg: u32) -> Emitted {
    assert!(experts.is_multiple_of(sg) && experts <= 1024);
    let pairs_max = rows_max * top_k;
    let mut k = Kernel::new(experts, sg);
    let state = k.buffer(0, Elem::U32);
    let route = k.buffer(1, Elem::U32);
    let list = k.buffer(2, Elem::U32);
    let table = k.buffer(3, Elem::U32);
    let staged = (pairs_max + experts + 1) * 4 <= 32 * 1024;
    let ex = k.shared(if staged { pairs_max } else { SORT_SLICE }, false);
    let offs = k.shared(experts + 1, false);
    let m = rows(&mut k, state);
    let kk = k.u(top_k);
    let pairs = k.imul(m, kk);
    let e = k.local_id();
    let ec = k.u(experts);
    let zero = k.u(0);
    let one = k.u(1);
    // The expert of pair `i`.
    let from_route = |k: &mut Kernel, i: Word| -> Word {
        let r = k.udiv(i, kk);
        let s = k.umod(i, kk);
        let k2 = k.u(2 * top_k);
        let at = k.imul(r, k2);
        let at = k.iadd(at, s);
        k.ld(route, at)
    };
    // The expert of every pair, into threadgroup memory.
    if staged {
        k.for_(e, pairs, ec, |k, i| {
            let x = from_route(k, i);
            k.sst(ex, i, x);
        });
        k.barrier();
    }
    // `f(pair, expert)` for every pair in order, in every thread: from the
    // staged experts, or a slice at a time staged between two barriers.
    let scan = |k: &mut Kernel, f: &mut dyn FnMut(&mut Kernel, Word, Word)| {
        if staged {
            k.for_(zero, pairs, one, |k, i| {
                let x = k.sld(ex, i);
                f(k, i, x);
            });
            return;
        }
        let ch = k.u(SORT_SLICE);
        k.for_(zero, pairs, ch, |k, c0| {
            k.barrier();
            k.for_(e, ch, ec, |k, j| {
                let p = k.iadd(c0, j);
                let inside = k.ult(p, pairs);
                k.if_(inside, |k| {
                    let x = from_route(k, p);
                    k.sst(ex, j, x);
                });
            });
            k.barrier();
            let left = k.isub(pairs, c0);
            let full = k.ult(ch, left);
            let n = k.select_u(full, ch, left);
            k.for_(zero, n, one, |k, j| {
                let x = k.sld(ex, j);
                let p = k.iadd(c0, j);
                f(k, p, x);
            });
        });
    };
    // Pairs per expert.
    let cnt = k.var_u32(zero);
    scan(&mut k, &mut |k, _, x| {
        let mine = k.ieq(x, e);
        let c = k.get(cnt);
        let c1 = k.iadd(c, one);
        let c = k.select_u(mine, c1, c);
        k.set(cnt, c);
    });
    let c = k.get(cnt);
    k.sst(offs, e, c);
    k.barrier();
    // Thread 0: the offsets and the tile table, the experts in order.
    let first = k.ieq(e, zero);
    k.if_(first, |k| {
        let total = k.var_u32(zero);
        let tiles = k.var_u32(zero);
        let mt = k.u(bm);
        let three = k.u(3);
        k.for_(zero, ec, one, |k, x| {
            let c = k.sld(offs, x);
            let start = k.get(total);
            k.sst(offs, x, start);
            k.for_(zero, c, mt, |k, j| {
                let t = k.get(tiles);
                let at = k.imul(t, three);
                let at = k.iadd(at, one);
                k.st(table, at, x);
                let a1 = k.iadd(at, one);
                let st = k.iadd(start, j);
                k.st(table, a1, st);
                let a2 = k.iadd(a1, one);
                let left = k.isub(c, j);
                let full = k.ult(mt, left);
                let rows = k.select_u(full, mt, left);
                k.st(table, a2, rows);
                let t1 = k.iadd(t, one);
                k.set(tiles, t1);
            });
            let s2 = k.iadd(start, c);
            k.set(total, s2);
        });
        let t = k.get(tiles);
        k.st(table, zero, t);
    });
    k.barrier();
    // Each expert's pairs at its offset, in increasing pair order.
    let off = k.sld(offs, e);
    let pos = k.var_u32(off);
    scan(&mut k, &mut |k, i, x| {
        let mine = k.ieq(x, e);
        k.if_(mine, |k| {
            let p = k.get(pos);
            k.st(list, p, i);
            let p1 = k.iadd(p, one);
            k.set(pos, p1);
        });
    });
    let _ = pairs_max;
    done(format!("moe_sort_{experts}_k{top_k}_r{rows_max}_m{bm}"), k, 4)
}

/// [`moe_sort`] by counting: every thread walks a stride of the pairs and
/// counts each pair's expert with a threadgroup atomic, thread 0 makes the
/// offsets and the table, then every thread places its pairs at its experts'
/// cursors (atomics again). The pairs of an expert are together, in an order
/// that may change from run to run (each sorted row is computed alone, so
/// the outputs do not). Same bindings, one group.
pub fn moe_sort_atomic(experts: u32, top_k: u32, rows_max: u32, bm: u32, sg: u32) -> Emitted {
    assert!(experts.is_multiple_of(sg) && experts <= 1024);
    let mut k = Kernel::new(experts, sg);
    let state = k.buffer(0, Elem::U32);
    let route = k.buffer(1, Elem::U32);
    let list = k.buffer(2, Elem::U32);
    let table = k.buffer(3, Elem::U32);
    let counts = k.shared_atomic(experts);
    let cursor = k.shared_atomic(experts);
    let offs = k.shared(experts + 1, false);
    let m = rows(&mut k, state);
    let kk = k.u(top_k);
    let pairs = k.imul(m, kk);
    let e = k.local_id();
    let ec = k.u(experts);
    let zero = k.u(0);
    let one = k.u(1);
    let from_route = |k: &mut Kernel, i: Word| -> Word {
        let r = k.udiv(i, kk);
        let s = k.umod(i, kk);
        let k2 = k.u(2 * top_k);
        let at = k.imul(r, k2);
        let at = k.iadd(at, s);
        k.ld(route, at)
    };
    k.atomic_store(counts, e, zero);
    k.barrier();
    k.for_(e, pairs, ec, |k, i| {
        let x = from_route(k, i);
        let _ = k.atomic_add(counts, x, one);
    });
    k.barrier();
    let c = k.atomic_load(counts, e);
    k.sst(offs, e, c);
    k.barrier();
    // Thread 0: the offsets and the tile table, the experts in order.
    let first = k.ieq(e, zero);
    k.if_(first, |k| {
        let total = k.var_u32(zero);
        let tiles = k.var_u32(zero);
        let mt = k.u(bm);
        let three = k.u(3);
        k.for_(zero, ec, one, |k, x| {
            let c = k.sld(offs, x);
            let start = k.get(total);
            k.sst(offs, x, start);
            k.for_(zero, c, mt, |k, j| {
                let t = k.get(tiles);
                let at = k.imul(t, three);
                let at = k.iadd(at, one);
                k.st(table, at, x);
                let a1 = k.iadd(at, one);
                let st = k.iadd(start, j);
                k.st(table, a1, st);
                let a2 = k.iadd(a1, one);
                let left = k.isub(c, j);
                let full = k.ult(mt, left);
                let rows = k.select_u(full, mt, left);
                k.st(table, a2, rows);
                let t1 = k.iadd(t, one);
                k.set(tiles, t1);
            });
            let s2 = k.iadd(start, c);
            k.set(total, s2);
        });
        let t = k.get(tiles);
        k.st(table, zero, t);
    });
    k.barrier();
    let off = k.sld(offs, e);
    k.atomic_store(cursor, e, off);
    k.barrier();
    k.for_(e, pairs, ec, |k, i| {
        let x = from_route(k, i);
        let p = k.atomic_add(cursor, x, one);
        k.st(list, p, i);
    });
    done(format!("moe_sort_atomic_{experts}_k{top_k}_r{rows_max}_m{bm}"), k, 4)
}

/// Head norms, partial RoPE and the KV append of every row: groups
/// `m · (heads + 2·kv_heads)`, row `g / (heads + 2·kv_heads)` at position
/// `pos + row`. Same bindings as the decode kernel.
pub fn qk_rope_append_rows(s: AttnShape, sg: u32) -> Emitted {
    let hd = s.head_dim;
    let half = s.rotary / 2;
    let per = s.heads + 2 * s.kv_heads;
    let mut k = Kernel::new(hd, sg);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let qn = k.buffer(2, Elem::U16);
    let kn = k.buffer(3, Elem::U16);
    let cos = k.buffer(4, Elem::U16);
    let sin = k.buffer(5, Elem::U16);
    let kc = k.buffer(6, Elem::U16);
    let vc = k.buffer(7, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let rowbuf = k.shared(hd, true);
    let gg = k.flat_group();
    let perc = k.u(per);
    let row = k.udiv(gg, perc);
    let g = k.umod(gg, perc);
    let lid = k.local_id();
    let p0 = pos(&mut k, state);
    let p = k.iadd(p0, row);
    let widthc = k.u(s.qkv_width());
    let rbase = k.imul(row, widthc);
    let heads = k.u(s.heads);
    let is_q = k.ult(g, heads);
    let hk = k.u(s.heads + s.kv_heads);
    let is_qk = k.ult(g, hk);
    let hdc = k.u(hd);
    let cap_hd = k.u(s.capacity * hd);
    k.if_else(
        is_qk,
        |k| {
            let qw = k.u(s.q_width());
            let two_hd = k.u(2 * hd);
            let kvh = k.isub(g, heads);
            let q_at = k.imul(g, two_hd);
            let kv_at = k.imul(kvh, hdc);
            let k_at = k.iadd(qw, kv_at);
            let src = k.select_u(is_q, q_at, k_at);
            let src = k.iadd(rbase, src);
            let at = k.iadd(src, lid);
            let h = k.ld(qkv, at);
            let x = k.bf16_to_f(h);
            let x2 = k.fmul(x, x);
            let ss = k.wg_reduce(x2, scratch, false);
            let inv_d = k.f(1.0 / hd as f32);
            let e = k.f(s.eps);
            let mm = k.fmul(ss, inv_d);
            let mm = k.fadd(mm, e);
            let r = k.sqrt(mm);
            let one = k.f(1.0);
            let inv = k.fdiv(one, r);
            let wq = k.ld(qn, lid);
            let wk = k.ld(kn, lid);
            let wh = k.select_u(is_q, wq, wk);
            let wf = k.bf16_to_f(wh);
            let wf = k.fadd(wf, one);
            let n = k.fmul(x, inv);
            let yv = k.fmul(n, wf);
            let yv = k.round_bf16(yv);
            k.sst(rowbuf, lid, yv);
            k.barrier();
            let rot = k.u(s.rotary);
            let hc = k.u(half);
            let in_rot = k.ult(lid, rot);
            let outv = k.var_f32(yv);
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
                let a = k.sld(rowbuf, j);
                let jb = k.iadd(j, hc);
                let b = k.sld(rowbuf, jb);
                let ac = k.fmul(a, c);
                let bs = k.fmul(b, sn);
                let lo = k.fsub(ac, bs);
                let as_ = k.fmul(a, sn);
                let bc = k.fmul(b, c);
                let hi = k.fadd(as_, bc);
                let r = k.select_f(first, lo, hi);
                k.set(outv, r);
            });
            let o = k.get(outv);
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
            let vh = k.isub(g, hk);
            let vbase = k.u(s.q_width() + s.kv_width());
            let off = k.imul(vh, hdc);
            let src = k.iadd(vbase, off);
            let src = k.iadd(rbase, src);
            let src = k.iadd(src, lid);
            let v = k.ld(qkv, src);
            let base = k.imul(vh, cap_hd);
            let pr = k.imul(p, hdc);
            let a = k.iadd(base, pr);
            let a = k.iadd(a, lid);
            k.st(vc, a, v);
        },
    );
    done(format!("qk_rope_append_rows_h{}kv{}d{}", s.heads, s.kv_heads, hd), k, 8)
}

/// Causal attention of every row, one group per (row, head) over the
/// `pos + row + 1` cached keys — the decode kernel's computation, row by row,
/// with its output gate. Bindings `state, qkv, k_cache, v_cache, scores
/// ([m][heads][capacity] f32), out`; groups `m · heads`.
pub fn sdpa_rows(s: AttnShape, sg: u32) -> Emitted {
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
    let gg = k.flat_group();
    let headsc = k.u(s.heads);
    let row = k.udiv(gg, headsc);
    let h = k.umod(gg, headsc);
    let lid = k.local_id();
    let p0 = pos(&mut k, state);
    let p = k.iadd(p0, row);
    let one_u = k.u(1);
    let len = k.iadd(p, one_u);
    let group = k.u(s.heads / s.kv_heads);
    let kvh = k.udiv(h, group);
    let hdc = k.u(hd);
    let two_hd = k.u(2 * hd);
    let widthc = k.u(s.qkv_width());
    let rbase = k.imul(row, widthc);
    let q_base = k.imul(h, two_hd);
    let q_base = k.iadd(rbase, q_base);
    let step = k.u(wg);
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
    let row_base = k.imul(gg, cap);
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
    let out_w = k.u(s.out_width());
    let obase = k.imul(row, out_w);
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
        let sgm = k.sigmoid(gf);
        let r = k.fmul(o, sgm);
        let b = k.f_to_bf16(r);
        let hb = k.imul(h, hdc);
        let oat = k.iadd(obase, hb);
        let oat = k.iadd(oat, d);
        k.st(out, oat, b);
    });
    done(format!("sdpa_rows_h{}kv{}d{}", s.heads, s.kv_heads, hd), k, 6)
}

/// Causal attention of the block by tiles of `bq` queries ("flash"): one group
/// per (query tile, head), 256 invocations, one dimension of the head each
/// (`head_dim` = 256). The tile's queries are staged in workgroup memory; the
/// keys up to the tile's last position are walked by tiles of `bk`, staged
/// (bf16 pairs, one spare word per row against bank conflicts); scores
/// `q·k/√hd`, masked past each query's position, then an online softmax per
/// query (running max and sum, the accumulators rescaled), the values of the
/// tile staged the same way and accumulated in registers. Output as
/// [`sdpa_rows`]: `bf16(o/l)`, then the gate. Bindings `state, qkv, k_cache
/// (u32 pairs), v_cache (u32 pairs), out`; groups `ceil(m / bq) · heads`.
pub fn sdpa_flash(s: AttnShape, sg: u32, bq: u32, bk: u32) -> Emitted {
    let hd = s.head_dim;
    let wg = 256;
    assert_eq!(hd, wg, "one invocation per dimension of a {hd}-wide head");
    assert!((bq * bk).is_multiple_of(wg) && bq <= wg, "bq {bq} × bk {bk}");
    let per = bq * bk / wg;
    let half = hd / 2;
    let stride = half + 1;
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let kc = k.buffer(2, Elem::U32);
    let vc = k.buffer(3, Elem::U32);
    let out = k.buffer(4, Elem::U16);
    let qs = k.shared(bq * hd, true);
    let kv = k.shared(bk * stride, false);
    let sp = k.shared(bq * bk, true);
    let mrow = k.shared(bq, true);
    let lrow = k.shared(bq, true);
    let corr = k.shared(bq, true);
    let m = rows(&mut k, state);
    let p0 = pos(&mut k, state);
    let gg = k.flat_group();
    let headsc = k.u(s.heads);
    let h = k.umod(gg, headsc);
    let qt = k.udiv(gg, headsc);
    let bqc = k.u(bq);
    let r0 = k.imul(qt, bqc);
    let t = k.local_id();
    let hdc = k.u(hd);
    let widthc = k.u(s.qkv_width());
    let two_hd = k.u(2 * hd);
    let hq = k.imul(h, two_hd);
    let group = k.u(s.heads / s.kv_heads);
    let kvh = k.udiv(h, group);
    let cap = k.u(s.capacity);
    let cap_m1 = k.u(s.capacity - 1);
    let halfc = k.u(half);
    let kv_base = k.u(s.capacity * half);
    let kv_base = k.imul(kvh, kv_base);
    let stc = k.u(stride);
    let wgc = k.u(wg);
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let neg = k.f(f32::MIN);
    let zero_f = k.f(0.0);
    // The rows of this tile, and the keys they see: up to the last row's position.
    let left = k.isub(m, r0);
    let short = k.ult(left, bqc);
    let rows_here = k.select_u(short, left, bqc);
    let last = k.iadd(p0, r0);
    let kv_end = k.iadd(last, rows_here);
    let _ = cap;
    // Stage the queries (zero past the block).
    let qn = k.u(bq * hd);
    k.for_(t, qn, wgc, |k, e| {
        let i = k.udiv(e, hdc);
        let d = k.umod(e, hdc);
        let row = k.iadd(r0, i);
        let inside = k.ult(row, m);
        let qv = k.var_f32(zero_f);
        k.if_(inside, |k| {
            let at = k.imul(row, widthc);
            let at = k.iadd(at, hq);
            let at = k.iadd(at, d);
            let qh = k.ld(qkv, at);
            let q = k.bf16_to_f(qh);
            k.set(qv, q);
        });
        let q = k.get(qv);
        k.sst(qs, e, q);
    });
    let is_row = k.ult(t, bqc);
    k.if_(is_row, |k| {
        k.sst(mrow, t, neg);
        k.sst(lrow, t, zero_f);
    });
    let accs: Vec<_> = (0..bq).map(|_| k.var_f32(zero_f)).collect();
    k.barrier();
    let scale = k.f(1.0 / (hd as f32).sqrt());
    let bkc = k.u(bk);
    let tile_words = k.u(bk * half);
    // Stage one tile of K or V: `bk` rows of `hd` bf16 at `k0`.
    let stage = |k: &mut Kernel, src: Buf, k0: Word| {
        k.for_(t, tile_words, wgc, |k, e| {
            let j = k.udiv(e, halfc);
            let w = k.umod(e, halfc);
            let kp = k.iadd(k0, j);
            let over = k.ult(cap_m1, kp);
            let kp = k.select_u(over, cap_m1, kp);
            let at = k.imul(kp, halfc);
            let at = k.iadd(kv_base, at);
            let at = k.iadd(at, w);
            let v = k.ld(src, at);
            let sa = k.imul(j, stc);
            let sa = k.iadd(sa, w);
            k.sst(kv, sa, v);
        });
    };
    k.for_(zero_u, kv_end, bkc, |k, k0| {
        stage(k, kc, k0);
        k.barrier();
        // Scores: invocation t computes `per` of the bq × bk.
        for x in 0..per {
            let xc = k.u(x * wg);
            let e = k.iadd(t, xc);
            let i = k.udiv(e, bkc);
            let j = k.umod(e, bkc);
            let qrow = k.imul(i, hdc);
            let krow = k.imul(j, stc);
            let dot = k.var_f32(zero_f);
            k.for_(zero_u, halfc, one_u, |k, w| {
                let kw_at = k.iadd(krow, w);
                let kw = k.sld(kv, kw_at);
                let two = k.u(2);
                let d0 = k.imul(w, two);
                let qa = k.iadd(qrow, d0);
                let q0 = k.sld(qs, qa);
                let qa1 = k.iadd(qa, one_u);
                let q1 = k.sld(qs, qa1);
                let lo = k.lo_f(kw);
                let hi = k.hi_f(kw);
                let a = k.get(dot);
                let a = k.fma(q0, lo, a);
                let a = k.fma(q1, hi, a);
                k.set(dot, a);
            });
            let dv = k.get(dot);
            let sc = k.fmul(dv, scale);
            // Valid: the key is at or before the query's position, the row in the block.
            let kp = k.iadd(k0, j);
            let qp = k.iadd(last, i);
            let after = k.ult(qp, kp);
            let row = k.iadd(r0, i);
            let in_m = k.ult(row, m);
            let not_after = k.b_not(after);
            let valid = k.land(not_after, in_m);
            let v = k.select_f(valid, sc, neg);
            k.sst(sp, e, v);
        }
        k.barrier();
        // Online softmax, one invocation per query row.
        k.if_(is_row, |k| {
            let base = k.imul(t, bkc);
            let mo = k.sld(mrow, t);
            let mx = k.var_f32(mo);
            k.for_(zero_u, bkc, one_u, |k, j| {
                let at = k.iadd(base, j);
                let v = k.sld(sp, at);
                let a = k.get(mx);
                let a = k.fmax(a, v);
                k.set(mx, a);
            });
            let mn = k.get(mx);
            let dlt = k.fsub(mo, mn);
            let c = k.exp(dlt);
            let sum = k.var_f32(zero_f);
            k.for_(zero_u, bkc, one_u, |k, j| {
                let at = k.iadd(base, j);
                let v = k.sld(sp, at);
                let dd = k.fsub(v, mn);
                let e = k.exp(dd);
                k.sst(sp, at, e);
                let a = k.get(sum);
                let a = k.fadd(a, e);
                k.set(sum, a);
            });
            let lo = k.sld(lrow, t);
            let sm = k.get(sum);
            let l = k.fma(lo, c, sm);
            k.sst(lrow, t, l);
            k.sst(mrow, t, mn);
            k.sst(corr, t, c);
        });
        k.barrier();
        stage(k, vc, k0);
        k.barrier();
        // Accumulate P·V for this invocation's dimension.
        let w = k.shr(t, one_u);
        let odd = k.and(t, one_u);
        let is_odd = k.ieq(odd, one_u);
        for (i, acc) in accs.iter().enumerate() {
            let ic = k.u(i as u32);
            let c = k.sld(corr, ic);
            let a0 = k.get(*acc);
            let a0 = k.fmul(a0, c);
            let av = k.var_f32(a0);
            let pbase = k.u(i as u32 * bk);
            k.for_(zero_u, bkc, one_u, |k, j| {
                let pa = k.iadd(pbase, j);
                let p = k.sld(sp, pa);
                let va = k.imul(j, stc);
                let va = k.iadd(va, w);
                let vw = k.sld(kv, va);
                let lo = k.lo_f(vw);
                let hi = k.hi_f(vw);
                let vf = k.select_f(is_odd, hi, lo);
                let a = k.get(av);
                let a = k.fma(p, vf, a);
                k.set(av, a);
            });
            let a = k.get(av);
            k.set(*acc, a);
        }
        // K/V and the scores are overwritten by the next tile.
        k.barrier();
    });
    // Out: bf16(o / l), then the gate.
    let out_w = k.u(s.out_width());
    for (i, acc) in accs.iter().enumerate() {
        let ic = k.u(i as u32);
        let row = k.iadd(r0, ic);
        let inside = k.ult(row, m);
        let a = k.get(*acc);
        k.if_(inside, |k| {
            let l = k.sld(lrow, ic);
            let o = k.fdiv(a, l);
            let o = k.round_bf16(o);
            let at = k.imul(row, widthc);
            let at = k.iadd(at, hq);
            let at = k.iadd(at, hdc);
            let at = k.iadd(at, t);
            let gh = k.ld(qkv, at);
            let gf = k.bf16_to_f(gh);
            let sgm = k.sigmoid(gf);
            let r = k.fmul(o, sgm);
            let b = k.f_to_bf16(r);
            let ob = k.imul(row, out_w);
            let hb = k.imul(h, hdc);
            let oat = k.iadd(ob, hb);
            let oat = k.iadd(oat, t);
            k.st(out, oat, b);
        });
    }
    done(format!("sdpa_flash_h{}kv{}d{}_q{bq}k{bk}", s.heads, s.kv_heads, hd), k, 5)
}

/// Causal attention of the block on the matrix units ("mma"): one group per
/// (tile of `qb` queries, K/V head), its `qb · G` rows (`G = heads /
/// kv_heads`, the heads that share the K/V head, `qb` queries each) over
/// `qb · G / 8` simdgroups of 8 rows, one head per simdgroup. The keys are
/// walked by tiles of `bk`: `S = Q·Kᵀ` by `simdgroup_multiply_accumulate` of
/// 16-bit operands (the rows of `qkv`, the rows of the K cache transposed)
/// into f32; S to the simdgroup's scratch, then the online softmax by rows
/// (four lanes per row, `bk / 4` columns each, max and sum by
/// `simd_shuffle_xor`), masked past each query's position; P rounded to 16
/// bits (the reference module rounds the softmax to the activations' type
/// before `P·V`), the output accumulators (8 × `head_dim` f32, `head_dim /
/// 8` matrices) rescaled by the diagonal of `exp(m_old − m_new)` through
/// `simdgroup_multiply` (exact), then `O += P·V`. Two ways to the keys:
/// `staged` false, every simdgroup loads its K and V blocks from the caches
/// (no threadgroup barrier, each simdgroup stops at its own last query);
/// `staged` true, the group stages the tile of K, then of V, in workgroup
/// memory (`uint4` cells, one spare per row against bank conflicts), three
/// barriers per tile, every simdgroup walking up to the group's last query.
/// `qreg`: the simdgroup's blocks of Q loaded once into registers
/// (`head_dim / 8` matrices) instead of once per tile; `kt`: `Sᵀ = K·Qᵀ`
/// instead, the blocks of K loaded straight and those of Q transposed, S
/// kept transposed in the scratch (`[bk][8]`); the rescaling is skipped when
/// no row's max moved (`simd_all`, the same values). Output as
/// [`sdpa_rows`]: `bf16(o / l)`, then the gate, `bk` columns at a time
/// through the scratch. Bindings `state, qkv, k_cache, v_cache, out`
/// (16-bit; the caches as `uint4` when staged); groups `ceil(m / qb) ·
/// kv_heads`. `capacity` a multiple of `bk` (the last tile's rows are read
/// whole), `qb` a divisor of the block (the queries past `m` are read, not
/// written).
pub fn sdpa_mma(s: AttnShape, sg: u32, qb: u32, bk: u32, staged: bool, qreg: bool, kt: bool) -> Emitted {
    let hd = s.head_dim;
    let g = s.heads / s.kv_heads;
    let rows_g = qb * g;
    assert!(sg == 32 && hd.is_multiple_of(8) && qb.is_multiple_of(8) && bk.is_multiple_of(8) && (8..=64).contains(&bk) && s.capacity.is_multiple_of(bk), "sdpa_mma: simdgroup {sg}, head_dim {hd}, qb {qb}, bk {bk}, capacity {}", s.capacity);
    let nsg = rows_g / 8;
    let wg = nsg * sg;
    assert!(wg <= 1024, "sdpa_mma: {rows_g} rows per group ({qb} queries × {g} heads)");
    let cpl = bk / 4; // columns of S per lane (four lanes per row)
    let cells = hd / 8; // uint4 per row of K or V
    let srow = cells + 1; // staged: cells per row, one spare
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let kv_elem = if staged { Elem::UVec4 } else { Elem::U16 };
    let kc = k.buffer(2, kv_elem);
    let vc = k.buffer(3, kv_elem);
    let out = k.buffer(4, Elem::U16);
    let scratch = k.shared(nsg * 8 * bk, true);
    let psh = k.shared_vec4(nsg * bk);
    let dsh = k.shared(nsg * 64, true);
    let stage = staged.then(|| k.shared_vec4(bk * srow));
    let m = rows(&mut k, state);
    let p0 = pos(&mut k, state);
    let gg = k.flat_group();
    let kvhc = k.u(s.kv_heads);
    let kvh = k.umod(gg, kvhc);
    let qt = k.udiv(gg, kvhc);
    let qbc = k.u(qb);
    let r0 = k.imul(qt, qbc);
    let sgid = k.subgroup_id();
    let lane = k.lane();
    let t = k.local_id();
    let eight = k.u(8);
    let sg8 = k.imul(sgid, eight);
    let head_local = k.udiv(sg8, qbc);
    let qoff = k.umod(sg8, qbc);
    let gc = k.u(g);
    let h = k.imul(kvh, gc);
    let h = k.iadd(h, head_local);
    let hdc = k.u(hd);
    let widthc = k.u(s.qkv_width());
    let two_hd = k.u(2 * hd);
    let hq = k.imul(h, two_hd);
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let three_u = k.u(3);
    let four = k.u(4);
    let wgc = k.u(wg);
    let neg = k.f(f32::MIN);
    let zero_f = k.f(0.0);
    let scale = k.f(1.0 / (hd as f32).sqrt());
    // The simdgroup's first query row, this lane's row and columns.
    let row_sg = k.iadd(r0, qoff);
    let has_rows = k.ult(row_sg, m);
    let lrow = k.udiv(lane, four);
    let lq = k.umod(lane, four);
    let cplc = k.u(cpl);
    let lcol0 = k.imul(lq, cplc);
    let qrow = k.iadd(row_sg, lrow);
    let qpos = k.iadd(p0, qrow);
    let in_m = k.ult(qrow, m);
    let first_of_row = k.ieq(lq, zero_u);
    let scwc = k.u(8 * bk);
    let sc_base = k.imul(sgid, scwc);
    let pbase = k.imul(sgid, scwc); // 16-bit elements
    let d64 = k.u(64);
    let dbase = k.imul(sgid, d64);
    let kv_base = if staged { k.u(s.capacity * cells) } else { k.u(s.capacity * hd) };
    let kv_base = k.imul(kvh, kv_base);
    let q_at = k.imul(row_sg, widthc);
    let q_at = k.iadd(q_at, hq);
    // The diagonal of scales, zero off the diagonal, once.
    let d_at = k.iadd(dbase, lane);
    k.sst(dsh, d_at, zero_f);
    let thirty_two = k.u(32);
    let d_at = k.iadd(d_at, thirty_two);
    k.sst(dsh, d_at, zero_f);
    let mrun = k.var_f32(neg);
    let lrun = k.var_f32(zero_f);
    let accs: Vec<_> = (0..hd / 8).map(|_| k.mat_acc()).collect();
    let qregs: Vec<_> = if qreg {
        (0..hd / 8)
            .map(|kk| {
                let o = k.u(kk * 8);
                let qa = k.iadd(q_at, o);
                k.mat_load_bf16_dev(qkv, qa, s.qkv_width(), kt)
            })
            .collect()
    } else {
        Vec::new()
    };
    if staged {
        k.barrier();
    } else {
        k.sg_barrier();
    }
    let one_f = k.f(1.0);
    // The keys' end: the group's last query (staged), or the simdgroup's (none without rows).
    let span = if staged { r0 } else { row_sg };
    let last = k.iadd(span, if staged { qbc } else { eight });
    let short = k.ult(m, last);
    let last = k.select_u(short, m, last);
    let end = k.iadd(p0, last);
    let end = if staged { end } else { k.select_u(has_rows, end, zero_u) };
    let bkc = k.u(bk);
    let diag_at = k.imul(lrow, eight);
    let diag_at = k.iadd(diag_at, lrow);
    let diag_at = k.iadd(dbase, diag_at);
    let s_row = k.imul(lrow, bkc);
    let s_row = k.iadd(s_row, lcol0);
    let stage_cells = k.u(bk * cells);
    let cellsc = k.u(cells);
    let srowc = k.u(srow);
    // Stage the tile of `src` (keys `k0`..): `bk` rows of `cells` uint4.
    let do_stage = |k: &mut Kernel, src: Buf, k_at: Word| {
        let st = stage.unwrap();
        k.for_(t, stage_cells, wgc, |k, e| {
            let j = k.udiv(e, cellsc);
            let w = k.umod(e, cellsc);
            let at = k.imul(j, cellsc);
            let at = k.iadd(k_at, at);
            let at = k.iadd(at, w);
            let v = k.ld(src, at);
            let sa = k.imul(j, srowc);
            let sa = k.iadd(sa, w);
            k.sst(st, sa, v);
        });
    };
    k.for_(zero_u, end, bkc, |k, k0| {
        let k_at = k.imul(k0, if staged { cellsc } else { hdc });
        let k_at = k.iadd(kv_base, k_at);
        if staged {
            do_stage(k, kc, k_at);
            k.barrier();
        }
        // S = Q·Kᵀ for this tile: 8 rows × bk keys, bk / 8 accumulators.
        let saccs: Vec<_> = (0..bk / 8).map(|_| k.mat_acc()).collect();
        for kk in 0..hd / 8 {
            let qblk = if qreg {
                qregs[kk as usize]
            } else {
                let o = k.u(kk * 8);
                let qa = k.iadd(q_at, o);
                k.mat_load_bf16_dev(qkv, qa, s.qkv_width(), kt)
            };
            for (j, sacc) in saccs.iter().enumerate() {
                let kblk = if let Some(st) = stage {
                    let o = k.u(j as u32 * 8 * 8 * srow + kk * 8);
                    k.mat_load_bf16(st, o, 8 * srow, !kt)
                } else {
                    let o = k.u(j as u32 * 8 * hd + kk * 8);
                    let ka = k.iadd(k_at, o);
                    k.mat_load_bf16_dev(kc, ka, hd, !kt)
                };
                if kt {
                    k.mat_mad(*sacc, kblk, qblk);
                } else {
                    k.mat_mad(*sacc, qblk, kblk);
                }
            }
        }
        for (j, sacc) in saccs.iter().enumerate() {
            // S [8][bk], or Sᵀ [bk][8]: block j at column, or row, 8j.
            let o = k.u(if kt { j as u32 * 64 } else { j as u32 * 8 });
            let at = k.iadd(sc_base, o);
            k.mat_store(*sacc, scratch, at, if kt { 8 } else { bk });
        }
        k.sg_barrier();
        // Online softmax, this lane's cpl columns of its row.
        let mut vals = Vec::with_capacity(cpl as usize);
        let local = k.var_f32(neg);
        for c in 0..cpl {
            let cc = k.u(c);
            let at = if kt {
                let col = k.iadd(lcol0, cc);
                let at = k.imul(col, eight);
                k.iadd(at, lrow)
            } else {
                k.iadd(s_row, cc)
            };
            let at = k.iadd(sc_base, at);
            let v = k.sld(scratch, at);
            let sc = k.fmul(v, scale);
            let kp = k.iadd(k0, lcol0);
            let kp = k.iadd(kp, cc);
            let after = k.ult(qpos, kp);
            let not_after = k.b_not(after);
            let valid = k.land(not_after, in_m);
            let v = k.select_f(valid, sc, neg);
            let a = k.get(local);
            let a = k.fmax(a, v);
            k.set(local, a);
            vals.push(v);
        }
        let lm = k.get(local);
        let tile_max = k.sg_fmax_clustered(lm, 4);
        let mo = k.get(mrun);
        let mn = k.fmax(mo, tile_max);
        let dlt = k.fsub(mo, mn);
        let corr = k.exp(dlt);
        let sum = k.var_f32(zero_f);
        let mut ps = Vec::with_capacity(cpl as usize);
        for v in vals {
            let d = k.fsub(v, mn);
            let p = k.exp(d);
            let a = k.get(sum);
            let a = k.fadd(a, p);
            k.set(sum, a);
            ps.push(p);
        }
        let ls = k.get(sum);
        let tile_sum = k.sg_fadd_clustered(ls, 4);
        let lo = k.get(lrun);
        let l = k.fma(lo, corr, tile_sum);
        k.set(lrun, l);
        k.set(mrun, mn);
        // P in 16 bits, pairs packed; the row's scale on the diagonal.
        for c in (0..cpl).step_by(2) {
            let p0h = k.f_to_bf16(ps[c as usize]);
            let p1h = k.f_to_bf16(ps[c as usize + 1]);
            let sixteen = k.u(16);
            let p1h = k.shl(p1h, sixteen);
            let pair = k.or(p0h, p1h);
            let cc = k.u(c);
            let e = k.iadd(s_row, cc);
            let e = k.iadd(pbase, e);
            let cell = k.shr(e, three_u);
            let comp = k.shr(e, one_u);
            let comp = k.and(comp, three_u);
            k.sst_comp(psh, cell, comp, pair);
        }
        k.if_(first_of_row, |k| {
            k.sst(dsh, diag_at, corr);
        });
        k.sg_barrier();
        // O = diag(corr)·O unless no row moved, P's blocks, then O += P·V.
        let same = k.feq(corr, one_f);
        let all_same = k.sg_all(same);
        let moved = k.b_not(all_same);
        k.if_(moved, |k| {
            let dm = k.mat_load_f32(dsh, dbase, 8);
            for acc in &accs {
                k.mat_mul_left(*acc, dm);
            }
        });
        let pblks: Vec<_> = (0..bk / 8)
            .map(|j| {
                let o = k.u(j * 8);
                let at = k.iadd(pbase, o);
                k.mat_load_bf16(psh, at, bk, false)
            })
            .collect();
        if staged {
            k.barrier();
            do_stage(k, vc, k_at);
            k.barrier();
        }
        for (d, acc) in accs.iter().enumerate() {
            for (j, pblk) in pblks.iter().enumerate() {
                let vblk = if let Some(st) = stage {
                    let o = k.u(j as u32 * 8 * 8 * srow + d as u32 * 8);
                    k.mat_load_bf16(st, o, 8 * srow, false)
                } else {
                    let o = k.u(j as u32 * 8 * hd + d as u32 * 8);
                    let va = k.iadd(k_at, o);
                    k.mat_load_bf16_dev(vc, va, hd, false)
                };
                k.mat_mad(*acc, *pblk, vblk);
            }
        }
        // The scratch, P (and the staged tile) are overwritten by the next tile.
        if staged {
            k.barrier();
        } else {
            k.sg_barrier();
        }
    });
    // Out: bf16(o / l), then the gate, bk columns of the 8 rows at a time through the scratch.
    let out_w = k.u(s.out_width());
    let l = k.get(lrun);
    let o_row = k.iadd(sc_base, s_row);
    let gate_row = k.imul(qrow, widthc);
    let gate_row = k.iadd(gate_row, hq);
    let gate_row = k.iadd(gate_row, hdc);
    let gate_row = k.iadd(gate_row, lcol0);
    let ob = k.imul(qrow, out_w);
    let hb = k.imul(h, hdc);
    let ob = k.iadd(ob, hb);
    let ob = k.iadd(ob, lcol0);
    for chunk in 0..hd / bk {
        for c in 0..bk / 8 {
            let o = k.u(c * 8);
            let at = k.iadd(sc_base, o);
            k.mat_store(accs[(chunk * (bk / 8) + c) as usize], scratch, at, bk);
        }
        k.sg_barrier();
        k.if_(in_m, |k| {
            for c in 0..cpl {
                let cc = k.u(c);
                let at = k.iadd(o_row, cc);
                let a = k.sld(scratch, at);
                let o = k.fdiv(a, l);
                let o = k.round_bf16(o);
                let dc = k.u(chunk * bk + c);
                let ga = k.iadd(gate_row, dc);
                let gh = k.ld(qkv, ga);
                let gf = k.bf16_to_f(gh);
                let sgm = k.sigmoid(gf);
                let r = k.fmul(o, sgm);
                let b = k.f_to_bf16(r);
                let oa = k.iadd(ob, dc);
                k.st(out, oa, b);
            }
        });
        k.sg_barrier();
    }
    done(format!("sdpa_mma_h{}kv{}d{}_q{qb}k{bk}{}{}{}", s.heads, s.kv_heads, hd, if staged { "_st" } else { "" }, if qreg { "_qr" } else { "" }, if kt { "_kt" } else { "" }), k, 5)
}

/// Causal attention of the block by Metal 4's `matmul2d` ("mm2d"): one group
/// per (tile of `qb` queries, head), `nsg` simdgroups running every tensor
/// operation together. The keys are walked by tiles of `bk`, staged (K, then
/// V) in workgroup memory as [`sdpa_mma`] stages them; `S = Q·Kᵀ` by one
/// `matmul2d` (Q read from `qkv` as a strided tensor, K from the stage), its
/// cooperative destination stored to a f32 tile, the online softmax by rows
/// as in [`sdpa_mma`] (`wg / qb` lanes per row), P rounded to 16 bits into a
/// tile, the output kept in `head_dim / cw` cooperative tensors of `qb × cw`
/// f32 across the tiles, each row scaled by `exp(m_old − m_new)` element by
/// element (the coordinates of the cooperative tensor), then `O += P·V` by
/// `matmul2d` in `multiply_accumulate` mode, V from the stage. Output as
/// [`sdpa_rows`]: `bf16(o / l)`, then the gate, `cw` columns at a time
/// through the stage seen as f32. Bindings `state, qkv, k_cache (uint4),
/// v_cache (uint4), out`; groups `ceil(m / qb) · heads`. `capacity` a
/// multiple of `bk`, `qb` a divisor of the block, `wg / qb` lanes per row a
/// power of two.
pub fn sdpa_mm2d(s: AttnShape, sg: u32, qb: u32, bk: u32, nsg: u32) -> Emitted {
    let hd = s.head_dim;
    let g = s.heads / s.kv_heads;
    let wg = nsg * sg;
    assert!(sg == 32 && hd.is_multiple_of(64) && (8..=64).contains(&bk) && bk.is_multiple_of(8) && s.capacity.is_multiple_of(bk) && qb >= 8 && wg.is_multiple_of(qb), "sdpa_mm2d: simdgroup {sg}, head_dim {hd}, qb {qb}, bk {bk}, {nsg} simdgroups, capacity {}", s.capacity);
    let tpr = wg / qb; // lanes per row of the softmax
    assert!(tpr.is_power_of_two() && tpr <= sg && bk.is_multiple_of(tpr) && (bk / tpr).is_multiple_of(2), "sdpa_mm2d: {tpr} lanes per row, bk {bk}");
    // Measured on the M3 Ultra (2026-10-31): 32 queries over 8 simdgroups compute wrong values (the ids diverge, the
    // argmax of the bench changes), 32 over 4 and 64 over 8 are right: at least 8 rows per simdgroup.
    assert!(qb >= 8 * nsg, "sdpa_mm2d: {qb} queries over {nsg} simdgroups (at least 8 rows each)");
    let cpl = bk / tpr;
    let cells = hd / 8;
    let srow = cells + 1;
    let stage_floats = bk * srow * 4;
    let cw = if stage_floats >= qb * 64 { 64 } else { 32 }; // columns of O per cooperative tensor, stored through the stage
    assert!(stage_floats >= qb * cw && hd.is_multiple_of(cw), "sdpa_mm2d: the stage ({stage_floats} floats) holds {qb} × {cw}");
    let epc = cw / tpr; // output columns per lane per round
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let qkv = k.buffer(1, Elem::U16);
    let kc = k.buffer(2, Elem::UVec4);
    let vc = k.buffer(3, Elem::UVec4);
    let out = k.buffer(4, Elem::U16);
    let stage = k.shared_vec4(bk * srow);
    let stage_f = k.shared_f32_view(stage);
    let ssh = k.shared(qb * bk, true);
    let psh = k.shared_vec4(qb * bk / 8);
    let mrow = k.shared(qb, true);
    let lrow = k.shared(qb, true);
    let corr_sh = k.shared(qb, true);
    let m = rows(&mut k, state);
    let p0 = pos(&mut k, state);
    let gg = k.flat_group();
    let headsc = k.u(s.heads);
    let h = k.umod(gg, headsc);
    let qt = k.udiv(gg, headsc);
    let qbc = k.u(qb);
    let r0 = k.imul(qt, qbc);
    let gc = k.u(g);
    let kvh = k.udiv(h, gc);
    let t = k.local_id();
    let hdc = k.u(hd);
    let widthc = k.u(s.qkv_width());
    let two_hd = k.u(2 * hd);
    let hq = k.imul(h, two_hd);
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let three_u = k.u(3);
    let wgc = k.u(wg);
    let neg = k.f(f32::MIN);
    let zero_f = k.f(0.0);
    let scale = k.f(1.0 / (hd as f32).sqrt());
    let tprc = k.u(tpr);
    let lr = k.udiv(t, tprc);
    let lq = k.umod(t, tprc);
    let cplc = k.u(cpl);
    let lcol0 = k.imul(lq, cplc);
    let qrow = k.iadd(r0, lr);
    let qpos = k.iadd(p0, qrow);
    let in_m = k.ult(qrow, m);
    let first_of_row = k.ieq(lq, zero_u);
    let bkc = k.u(bk);
    let s_row = k.imul(lr, bkc);
    let s_row = k.iadd(s_row, lcol0);
    let cellsc = k.u(cells);
    let srowc = k.u(srow);
    let kv_base = k.u(s.capacity * cells);
    let kv_base = k.imul(kvh, kv_base);
    // m, l of the rows.
    let is_row = k.ult(t, qbc);
    k.if_(is_row, |k| {
        k.sst(mrow, t, neg);
        k.sst(lrow, t, zero_f);
    });
    // The tensors (fixed addresses) and the operations.
    let q_at = k.imul(r0, widthc);
    let q_at = k.iadd(q_at, hq);
    let t_q = k.tensor_dev16(qkv, q_at, hd, qbc, s.qkv_width());
    let t_k = k.tensor_tg16(stage, zero_u, hd, bk, hd + 8);
    let t_s = k.tensor_tg_f32(ssh, zero_u, bk, qb, bk);
    let t_p = k.tensor_tg16(psh, zero_u, bk, qb, bk);
    let t_v: Vec<_> = (0..hd / cw)
        .map(|j| {
            let o = k.u(j * cw);
            k.tensor_tg16(stage, o, cw, bk, hd + 8)
        })
        .collect();
    let op_s = k.mm2d_op(qb, bk, Some(hd), true, false);
    let op_pv = k.mm2d_op(qb, cw, Some(bk), false, true);
    let accs: Vec<_> = (0..hd / cw).map(|j| k.mm2d_coop(op_pv, t_p, t_v[j as usize])).collect();
    k.barrier();
    let end = k.iadd(r0, qbc);
    let short = k.ult(m, end);
    let end = k.select_u(short, m, end);
    let end = k.iadd(p0, end);
    let stage_cells = k.u(bk * cells);
    let do_stage = |k: &mut Kernel, src: Buf, k_at: Word| {
        k.for_(t, stage_cells, wgc, |k, e| {
            let j = k.udiv(e, cellsc);
            let w = k.umod(e, cellsc);
            let at = k.imul(j, cellsc);
            let at = k.iadd(k_at, at);
            let at = k.iadd(at, w);
            let v = k.ld(src, at);
            let sa = k.imul(j, srowc);
            let sa = k.iadd(sa, w);
            k.sst(stage, sa, v);
        });
    };
    k.for_(zero_u, end, bkc, |k, k0| {
        let k_at = k.imul(k0, cellsc);
        let k_at = k.iadd(kv_base, k_at);
        do_stage(k, kc, k_at);
        k.barrier();
        // S = Q·Kᵀ, to the f32 tile.
        let c_s = k.mm2d_coop(op_s, t_q, t_k);
        k.mm2d_run(op_s, t_q, t_k, c_s);
        k.coop_store(c_s, t_s);
        k.barrier();
        // Online softmax, this lane's cpl columns of its row.
        let mut vals = Vec::with_capacity(cpl as usize);
        let local = k.var_f32(neg);
        for c in 0..cpl {
            let cc = k.u(c);
            let at = k.iadd(s_row, cc);
            let v = k.sld(ssh, at);
            let sc = k.fmul(v, scale);
            let kp = k.iadd(k0, lcol0);
            let kp = k.iadd(kp, cc);
            let after = k.ult(qpos, kp);
            let not_after = k.b_not(after);
            let valid = k.land(not_after, in_m);
            let v = k.select_f(valid, sc, neg);
            let a = k.get(local);
            let a = k.fmax(a, v);
            k.set(local, a);
            vals.push(v);
        }
        let lm = k.get(local);
        let tile_max = k.sg_fmax_clustered(lm, tpr);
        let mo = k.sld(mrow, lr);
        let mn = k.fmax(mo, tile_max);
        let dlt = k.fsub(mo, mn);
        let corr = k.exp(dlt);
        let sum = k.var_f32(zero_f);
        let mut ps = Vec::with_capacity(cpl as usize);
        for v in vals {
            let d = k.fsub(v, mn);
            let p = k.exp(d);
            let a = k.get(sum);
            let a = k.fadd(a, p);
            k.set(sum, a);
            ps.push(p);
        }
        let ls = k.get(sum);
        let tile_sum = k.sg_fadd_clustered(ls, tpr);
        for c in (0..cpl).step_by(2) {
            let p0h = k.f_to_bf16(ps[c as usize]);
            let p1h = k.f_to_bf16(ps[c as usize + 1]);
            let sixteen = k.u(16);
            let p1h = k.shl(p1h, sixteen);
            let pair = k.or(p0h, p1h);
            let cc = k.u(c);
            let e = k.iadd(s_row, cc);
            let cell = k.shr(e, three_u);
            let comp = k.shr(e, one_u);
            let comp = k.and(comp, three_u);
            k.sst_comp(psh, cell, comp, pair);
        }
        k.if_(first_of_row, |k| {
            let lo = k.sld(lrow, lr);
            let l = k.fma(lo, corr, tile_sum);
            k.sst(lrow, lr, l);
            k.sst(mrow, lr, mn);
            k.sst(corr_sh, lr, corr);
        });
        k.barrier();
        // O = diag(corr)·O, then the tile of V in the stage, then O += P·V.
        for acc in &accs {
            k.coop_scale_rows(*acc, corr_sh, zero_u);
        }
        do_stage(k, vc, k_at);
        k.barrier();
        for (j, acc) in accs.iter().enumerate() {
            k.mm2d_run(op_pv, t_p, t_v[j], *acc);
        }
        k.barrier();
    });
    // Out: bf16(o / l), then the gate, cw columns of the qb rows at a time through the stage.
    let out_w = k.u(s.out_width());
    let l = k.sld(lrow, lr);
    let epcc = k.u(epc);
    let lc = k.imul(lq, epcc);
    let cwc = k.u(cw);
    let o_row = k.imul(lr, cwc);
    let o_row = k.iadd(o_row, lc);
    let gate_row = k.imul(qrow, widthc);
    let gate_row = k.iadd(gate_row, hq);
    let gate_row = k.iadd(gate_row, hdc);
    let gate_row = k.iadd(gate_row, lc);
    let ob = k.imul(qrow, out_w);
    let hb = k.imul(h, hdc);
    let ob = k.iadd(ob, hb);
    let ob = k.iadd(ob, lc);
    let t_o = k.tensor_tg_f32(stage_f, zero_u, cw, qb, cw);
    for (chunk, acc) in accs.iter().enumerate() {
        k.coop_store(*acc, t_o);
        k.barrier();
        k.if_(in_m, |k| {
            for c in 0..epc {
                let cc = k.u(c);
                let at = k.iadd(o_row, cc);
                let a = k.sld(stage_f, at);
                let o = k.fdiv(a, l);
                let o = k.round_bf16(o);
                let dc = k.u(chunk as u32 * cw + c);
                let ga = k.iadd(gate_row, dc);
                let gh = k.ld(qkv, ga);
                let gf = k.bf16_to_f(gh);
                let sgm = k.sigmoid(gf);
                let r = k.fmul(o, sgm);
                let b = k.f_to_bf16(r);
                let oa = k.iadd(ob, dc);
                k.st(out, oa, b);
            }
        });
        k.barrier();
    }
    done(format!("sdpa_mm2d_h{}kv{}d{}_q{qb}k{bk}_sg{nsg}", s.heads, s.kv_heads, hd), k, 5)
}

/// Causal attention of the block by `matmul2d` of 16 × 32 × 16 run by one
/// simdgroup each ("nax", the form of mlx's `attention_nax_dsplit` on the
/// M5): one group per (tile of 64 queries, head), 4 row groups of 16 queries
/// × `wn = head_dim / 128` slices of 128 columns (8 simdgroups for 256). Each
/// simdgroup keeps its 16 × 128 slice of Q in registers (fragments), reads
/// its slice of K and V by fragments straight from the caches (no stage),
/// `S = Q·Kᵀ` over its columns into two fragments of 16 × 16, the `wn`
/// partial scores summed through workgroup memory in a fixed order (every
/// slice gets the same S), the online softmax by rows (the four lanes of a
/// row, `simd_shuffle_xor` 1 and 8), P rounded to 16 bits as the module of
/// HF does, `O` (16 × 128 f32 in fragments) scaled by `exp(m_old − m_new)`
/// unless no max moved, `O += P·V`. Output as [`sdpa_mm2d`]: `round16(o / l)`
/// times the gate's sigmoid. `relaxed`: the descriptor's reduced precision
/// (mlx's choice; off: the contract's f32 sums); `exp2`: the softmax in base 2
/// (scores times `log2(e)`, `exp2`), mlx's form, a few ULP from `exp`. Bindings `state, qkv,
/// k_cache (uint4), v_cache (uint4), out`; groups `ceil(m / 64) · heads`;
/// `capacity` a multiple of 32. In the bf16 witness the roundings are
/// `bfloat(x)` (exact on normals, subnormals flushed: probe S2).
pub fn sdpa_nax(s: AttnShape, sg: u32, relaxed: bool, exp2: bool) -> Emitted {
    let hd = s.head_dim;
    let g = s.heads / s.kv_heads;
    let wn = if hd >= 256 { hd / 128 } else { 1 };
    let bdh = hd / wn;
    let tdh = bdh / 16;
    assert!(sg == 32 && hd.is_multiple_of(wn) && bdh.is_multiple_of(32) && s.capacity.is_multiple_of(32), "sdpa_nax: simdgroup {sg}, head_dim {hd}, capacity {}", s.capacity);
    let nsg = 4 * wn;
    let mut k = Kernel::new(nsg * sg, sg);
    let _state = k.buffer(0, Elem::U32);
    let _qkv = k.buffer(1, Elem::U16);
    let _kc = k.buffer(2, Elem::UVec4);
    let _vc = k.buffer(3, Elem::UVec4);
    let out = k.buffer(4, Elem::U16);
    k.written(out);
    let t = k.t16_name();
    let (heads, cap, width, outw) = (s.heads, s.capacity, s.qkv_width(), s.out_width());
    let scale = 1.0 / (hd as f32).sqrt();
    let mut src = String::new();
    let mut w = |l: &str| {
        src.push_str(l);
        src.push('\n');
    };
    w(&format!("threadgroup float xs[{}];", 4 * wn * 512));
    w(&format!("const uint m = b0[{STATE_ROWS}];"));
    w(&format!("const uint p0 = b0[{}];", super::kernels::STATE_POS));
    w(&format!("const uint h = tgid.x % {heads}u;"));
    w(&format!("const uint r0 = (tgid.x / {heads}u) * 64u;"));
    w(&format!("const uint kvh = h / {g}u;"));
    w(&format!("const ushort rg = ushort(sgid) / {wn}, dg = ushort(sgid) % {wn};"));
    w("const ushort qid = ushort(lane) >> 2;");
    w("const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);");
    w("const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;");
    w("const uint row0 = r0 + rg * 16u + fm;");
    w(&format!("const uint dcol = dg * {bdh}u + fn;"));
    w(&format!("const device {t} *qh = (const device {t} *)b1;"));
    w(&format!("const device {t} *kh = (const device {t} *)b2 + kvh * {}u;", cap * hd));
    w(&format!("const device {t} *vh = (const device {t} *)b3 + kvh * {}u;", cap * hd));
    w(&format!("typedef vec<{t}, 8> v8; typedef vec<{t}, 4> v4;"));
    w(&format!("v8 q[{tdh}];"));
    for id in 0..tdh {
        for i in 0..2 {
            w(&format!(
                "{{ const uint r = row0 + {}u; v4 x = r < m ? *(const device v4 *)(qh + r * {width}u + h * {}u + dcol + {}u) : v4(0); q[{id}][{}] = x[0]; q[{id}][{}] = x[1]; q[{id}][{}] = x[2]; q[{id}][{}] = x[3]; }}",
                8 * i,
                2 * hd,
                id * 16,
                4 * i,
                4 * i + 1,
                4 * i + 2,
                4 * i + 3
            ));
        }
    }
    w(&format!("typedef vec<float, 8> f8; f8 o[{tdh}];"));
    w(&format!("for (ushort i = 0; i < {tdh}; ++i) o[i] = f8(0.0f);"));
    w("float mx0 = -FLT_MAX, mx1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;");
    w("const uint qp0 = p0 + row0, qp1 = qp0 + 8u;");
    w("const uint end = p0 + min(r0 + 64u, m);");
    w("for (uint k0 = 0; k0 < end; k0 += 32u) {");
    w("    f8 s0 = f8(0.0f), s1 = f8(0.0f);");
    // A fragment of 16 rows × 16 columns from a cache: rows (keys) row + fm, row + fm + 8, columns col + fn..
    let frag = |name: &str, base: &str, key: String, col: String| -> String {
        format!(
            "    v8 {name}; {{ const device {t} *p = {base} + ({key} + fm) * {hd}u + {col}; v4 x0 = *(const device v4 *)p; v4 x1 = *(const device v4 *)(p + {}u); {name} = v8(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]); }}",
            8 * hd
        )
    };
    for id in 0..tdh {
        w(&frag(&format!("ka{id}"), "kh", "k0".into(), format!("dcol + {}u", id * 16)));
        w(&frag(&format!("kb{id}"), "kh", "k0 + 16u".into(), format!("dcol + {}u", id * 16)));
        let mm = k.nax_mma_text("s0", "s1", &format!("q[{id}]"), &format!("ka{id}"), &format!("kb{id}"), true, relaxed);
        for l in mm.lines() {
            w(&format!("    {l}"));
        }
    }
    if wn > 1 {
        w(&format!("    threadgroup float *slot = xs + (rg * {wn}u + dg) * 512u + ushort(lane) * 16u;"));
        w("    for (ushort e = 0; e < 8; ++e) { slot[e] = s0[e]; slot[8 + e] = s1[e]; }");
        w("    threadgroup_barrier(mem_flags::mem_threadgroup);");
        w(&format!("    threadgroup float *peer = xs + rg * {}u + ushort(lane) * 16u;", wn * 512));
        w("    for (ushort e = 0; e < 8; ++e) { s0[e] = peer[e]; s1[e] = peer[8 + e]; }");
        w(&format!("    for (ushort pg = 1; pg < {wn}; ++pg) for (ushort e = 0; e < 8; ++e) {{ s0[e] += peer[pg * 512u + e]; s1[e] += peer[pg * 512u + 8u + e]; }}"));
        w("    threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
    // Scale, mask, the row max (row i of the lane: elements 4i..4i + 4 of both fragments).
    // `exp2`: the scores in base 2 (the scale times log2(e)), exp2 for exp (mlx's form; a few ULP of the softmax).
    let (sc, ex) = if exp2 { (scale * std::f32::consts::LOG2_E, "exp2") } else { (scale, "exp") };
    w(&format!("    const float sc = {sc:?}f;"));
    w("    float lm0 = -FLT_MAX, lm1 = -FLT_MAX;");
    w("    for (ushort e = 0; e < 8; ++e) {");
    w("        const uint kp = k0 + fn + (e & 3);");
    w("        const uint qp = e < 4 ? qp0 : qp1;");
    w("        s0[e] = kp <= qp ? s0[e] * sc : -FLT_MAX;");
    w("        s1[e] = kp + 16u <= qp ? s1[e] * sc : -FLT_MAX;");
    w("        if (e < 4) lm0 = fmax(lm0, fmax(s0[e], s1[e])); else lm1 = fmax(lm1, fmax(s0[e], s1[e]));");
    w("    }");
    w("    lm0 = fmax(lm0, simd_shuffle_xor(lm0, 1)); lm0 = fmax(lm0, simd_shuffle_xor(lm0, 8));");
    w("    lm1 = fmax(lm1, simd_shuffle_xor(lm1, 1)); lm1 = fmax(lm1, simd_shuffle_xor(lm1, 8));");
    w("    const float mn0 = fmax(mx0, lm0), mn1 = fmax(mx1, lm1);");
    w(&format!("    const float c0 = {ex}(mx0 - mn0), c1 = {ex}(mx1 - mn1);"));
    w("    float ls0 = 0.0f, ls1 = 0.0f;");
    w("    v8 p0f, p1f;");
    w("    for (ushort e = 0; e < 8; ++e) {");
    w("        const float mn = e < 4 ? mn0 : mn1;");
    w(&format!("        const float a = {ex}(s0[e] - mn), b = {ex}(s1[e] - mn);"));
    w("        if (e < 4) ls0 += a + b; else ls1 += a + b;");
    w(&format!("        p0f[e] = {t}(a); p1f[e] = {t}(b);"));
    w("    }");
    w("    ls0 += simd_shuffle_xor(ls0, 1); ls0 += simd_shuffle_xor(ls0, 8);");
    w("    ls1 += simd_shuffle_xor(ls1, 1); ls1 += simd_shuffle_xor(ls1, 8);");
    w("    l0 = fma(l0, c0, ls0); l1 = fma(l1, c1, ls1);");
    w("    mx0 = mn0; mx1 = mn1;");
    w("    if (!simd_all(c0 == 1.0f && c1 == 1.0f)) {");
    w(&format!("        for (ushort i = 0; i < {tdh}; ++i) for (ushort e = 0; e < 8; ++e) o[i][e] *= e < 4 ? c0 : c1;"));
    w("    }");
    for id in (0..tdh).step_by(2) {
        for ik in 0..2 {
            w(&frag(&format!("va{id}_{ik}"), "vh", format!("k0 + {}u", ik * 16), format!("dcol + {}u", id * 16)));
            w(&frag(&format!("vb{id}_{ik}"), "vh", format!("k0 + {}u", ik * 16), format!("dcol + {}u", id * 16 + 16)));
            let pf = if ik == 0 { "p0f" } else { "p1f" };
            let mm = k.nax_mma_text(&format!("o[{id}]"), &format!("o[{}]", id + 1), pf, &format!("va{id}_{ik}"), &format!("vb{id}_{ik}"), false, relaxed);
            for l in mm.lines() {
                w(&format!("    {l}"));
            }
        }
    }
    w("}");
    // Out: round16(o / l), times sigmoid(gate), rows row0 and row0 + 8.
    w(&format!("device {t} *oh = (device {t} *)b4;"));
    for i in 0..2 {
        w(&format!("{{ const uint r = row0 + {}u; if (r < m) {{ const float l = l{i};", 8 * i));
        w(&format!("    for (ushort id = 0; id < {tdh}; ++id) for (ushort j = 0; j < 4; ++j) {{"));
        w("        const uint d = dcol + id * 16u + j;");
        w(&format!("        const float ov = float({t}(o[id][{} + j] / l));", 4 * i));
        w(&format!("        const float gv = float(qh[r * {width}u + h * {}u + {hd}u + d]);", 2 * hd));
        w("        const float sg = 1.0f / (1.0f + exp(-gv));");
        w(&format!("        oh[r * {outw}u + h * {hd}u + d] = {t}(ov * sg);"));
        w("    }");
        w("} }");
    }
    k.raw(&src);
    done(format!("sdpa_nax_h{}kv{}d{hd}{}{}", s.heads, s.kv_heads, if relaxed { "_relaxed" } else { "" }, if exp2 { "_exp2" } else { "" }), k, 5)
}

/// `h = bf16(g·σ(g)·u)` for every row of `[gate (n) | up (n)]`. Bindings
/// `state, gate_up, h`; groups `ceil(m · n / 256)`.
pub fn swiglu_rows(n: u32, sg: u32) -> Emitted {
    let mut k = Kernel::new(256, sg);
    let state = k.buffer(0, Elem::U32);
    let gu = k.buffer(1, Elem::U16);
    let h = k.buffer(2, Elem::U16);
    let m = rows(&mut k, state);
    let nn = k.u(n);
    let total = k.imul(m, nn);
    let i = k.global_id();
    let inside = k.ult(i, total);
    k.if_(inside, |k| {
        let r = k.udiv(i, nn);
        let c = k.umod(i, nn);
        let two_n = k.u(2 * n);
        let base = k.imul(r, two_n);
        let gi = k.iadd(base, c);
        let ui = k.iadd(gi, nn);
        let gh = k.ld(gu, gi);
        let uh = k.ld(gu, ui);
        let g = k.bf16_to_f(gh);
        let u = k.bf16_to_f(uh);
        let s = k.sigmoid(g);
        let gs = k.fmul(g, s);
        let o = k.fmul(gs, u);
        let b = k.f_to_bf16(o);
        k.st(h, i, b);
    });
    done(format!("swiglu_rows_{n}"), k, 3)
}

/// Causal conv + SiLU over the rows, in order: one invocation per channel, the
/// channel's state in registers for the block. Bindings `state, proj (f32
/// [m][proj_width]), conv_state, conv_w`; groups `ceil(conv_dim / 256)`.
pub fn dn_conv_rows(s: DeltaShape, sg: u32) -> Emitted {
    let kk = s.conv;
    let mut k = Kernel::new(256, sg);
    let state = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let st = k.buffer(2, Elem::Act);
    let w = k.buffer(3, Elem::U16);
    let m = rows(&mut k, state);
    let c = k.global_id();
    let ch = k.u(s.conv_dim());
    let inside = k.ult(c, ch);
    k.if_(inside, |k| {
        let kc = k.u(kk);
        let base = k.imul(c, kc);
        let mut ws = Vec::new();
        let mut regs = Vec::new();
        for j in 0..kk {
            let jc = k.u(j);
            let at = k.iadd(base, jc);
            let wh = k.ld(w, at);
            ws.push(k.bf16_to_f(wh));
            let v = k.ld(st, at);
            regs.push(k.var_f32(v));
        }
        let pw = k.u(s.proj_width());
        let zero = k.u(0);
        let one = k.u(1);
        k.for_(zero, m, one, |k, t| {
            let tr = k.imul(t, pw);
            let at = k.iadd(tr, c);
            let x = k.ld(proj, at);
            for j in 0..(kk - 1) as usize {
                let nxt = k.get(regs[j + 1]);
                k.set(regs[j], nxt);
            }
            k.set(regs[kk as usize - 1], x);
            let zero = k.f(0.0);
            let mut acc = zero;
            for j in 0..kk as usize {
                let v = k.get(regs[j]);
                acc = k.fma(v, ws[j], acc);
            }
            let sgm = k.sigmoid(acc);
            let y = k.fmul(acc, sgm);
            k.st(proj, at, y);
        });
        for j in 0..kk {
            let jc = k.u(j);
            let at = k.iadd(base, jc);
            let v = k.get(regs[j as usize]);
            k.st(st, at, v);
        }
    });
    done(format!("dn_conv_rows_{}x{kk}", s.conv_dim()), k, 4)
}

/// [`dn_conv_rows`] with the block's rows cut into `256 / cols` segments:
/// one invocation per (channel, segment), `cols` consecutive channels per
/// group. Each invocation first reads the `conv − 1` inputs before its
/// segment (rows of the block, or the conv state before the first row), the
/// group meets, then it walks its segment in order in place as
/// [`dn_conv_rows`]; the segment holding the last row writes the state back.
/// Same sums in the same order. Bindings as [`dn_conv_rows`]; groups
/// `conv_dim / cols`.
pub fn dn_conv_seg(s: DeltaShape, sg: u32, cols: u32) -> Emitted {
    let kk = s.conv;
    let wg = 256;
    assert!(wg % cols == 0 && s.conv_dim().is_multiple_of(cols), "conv.cols {cols}");
    let segs = wg / cols;
    let mut k = Kernel::new(wg, sg);
    let state = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let st = k.buffer(2, Elem::Act);
    let w = k.buffer(3, Elem::U16);
    let m = rows(&mut k, state);
    let t = k.local_id();
    let colsc = k.u(cols);
    let lc = k.umod(t, colsc);
    let seg = k.udiv(t, colsc);
    let g = k.flat_group();
    let c = k.imul(g, colsc);
    let c = k.iadd(c, lc);
    let kc = k.u(kk);
    let base = k.imul(c, kc);
    let pw = k.u(s.proj_width());
    let zero = k.u(0);
    let one = k.u(1);
    // Rows per segment, and this segment's.
    let segc = k.u(segs);
    let sm1 = k.u(segs - 1);
    let len = k.iadd(m, sm1);
    let len = k.udiv(len, segc);
    let r0 = k.imul(seg, len);
    let r1 = k.iadd(r0, len);
    let past = k.ult(m, r1);
    let r1 = k.select_u(past, m, r1);
    let active = k.ult(r0, m);
    // The window before the segment: x[r0 − kk + j], from the block or the state.
    let mut ws = Vec::new();
    let mut regs = Vec::new();
    for j in 0..kk {
        let jc = k.u(j);
        let wa = k.iadd(base, jc);
        let wh = k.ld(w, wa);
        ws.push(k.bf16_to_f(wh));
        let back = k.u(kk - j);
        let after = k.ult(r0, back);
        let from_block = k.b_not(after);
        let zf = k.f(0.0);
        let v = k.var_f32(zf);
        // A segment past the block reads nothing (it has no row).
        k.if_(active, |k| {
            k.if_else(
            from_block,
            |k| {
                let r = k.isub(r0, back);
                let at = k.imul(r, pw);
                let at = k.iadd(at, c);
                let x = k.ld(proj, at);
                k.set(v, x);
            },
            |k| {
                // x[r0 − back] with r0 − back < 0: state entry kk + r0 − back.
                let at = k.iadd(base, kc);
                let at = k.iadd(at, r0);
                let at = k.isub(at, back);
                let x = k.ld(st, at);
                k.set(v, x);
            },
            );
        });
        let x = k.get(v);
        regs.push(k.var_f32(x));
    }
    // Every window is read before any segment overwrites its rows.
    k.barrier();
    k.for_(r0, r1, one, |k, t| {
        let tr = k.imul(t, pw);
        let at = k.iadd(tr, c);
        let x = k.ld(proj, at);
        for j in 0..(kk - 1) as usize {
            let nxt = k.get(regs[j + 1]);
            k.set(regs[j], nxt);
        }
        k.set(regs[kk as usize - 1], x);
        let zf = k.f(0.0);
        let mut acc = zf;
        for j in 0..kk as usize {
            let v = k.get(regs[j]);
            acc = k.fma(v, ws[j], acc);
        }
        let sgm = k.sigmoid(acc);
        let y = k.fmul(acc, sgm);
        k.st(proj, at, y);
    });
    // The segment holding the last row: its window is the new state.
    let last = k.isub(m, one);
    let ge = k.ult(last, r0);
    let ge = k.b_not(ge);
    let lt = k.ult(last, r1);
    let mine = k.land(ge, lt);
    let _ = zero;
    k.if_(mine, |k| {
        for j in 0..kk {
            let jc = k.u(j);
            let at = k.iadd(base, jc);
            let v = k.get(regs[j as usize]);
            k.st(st, at, v);
        }
    });
    done(format!("dn_conv_seg_{}x{kk}_c{cols}", s.conv_dim()), k, 4)
}

/// L2 norms of the q and k heads and the gates, for every row: groups
/// `m · (2·k_heads + 1)`. Bindings `proj, a_log, dt_bias, gates ([m][β | decay])`.
pub fn dn_l2_gates_rows(s: DeltaShape, sg: u32) -> Emitted {
    let dk = s.k_dim;
    let per = 2 * s.k_heads + 1;
    let mut k = Kernel::new(dk, sg);
    let proj = k.buffer(0, Elem::Act);
    let a_log = k.buffer(1, Elem::F32);
    let dt_bias = k.buffer(2, Elem::F32);
    let gates = k.buffer(3, Elem::Act);
    let scratch = k.shared(k.subgroups() + 1, true);
    let gg = k.flat_group();
    let perc = k.u(per);
    let row = k.udiv(gg, perc);
    let g = k.umod(gg, perc);
    let pw = k.u(s.proj_width());
    let rbase = k.imul(row, pw);
    let lid = k.local_id();
    let heads = k.u(2 * s.k_heads);
    let is_head = k.ult(g, heads);
    k.if_else(
        is_head,
        |k| {
            let dkc = k.u(dk);
            let base = k.imul(g, dkc);
            let base = k.iadd(rbase, base);
            let at = k.iadd(base, lid);
            let x = k.ld(proj, at);
            let x2 = k.fmul(x, x);
            let ss = k.wg_reduce(x2, scratch, false);
            let e = k.f(1e-6);
            let mm = k.fadd(ss, e);
            let r = k.sqrt(mm);
            let one = k.f(1.0);
            let inv = k.fdiv(one, r);
            let y = k.fmul(x, inv);
            k.st(proj, at, y);
        },
        |k| {
            let nv = k.u(s.v_heads);
            let two_nv = k.u(2 * s.v_heads);
            let gbase = k.imul(row, two_nv);
            let step = k.u(dk);
            k.for_(lid, nv, step, |k, i| {
                let a_at = k.u(s.a_at());
                let b_at = k.u(s.b_at());
                let ai = k.iadd(a_at, i);
                let ai = k.iadd(rbase, ai);
                let bi = k.iadd(b_at, i);
                let bi = k.iadd(rbase, bi);
                let a = k.ld(proj, ai);
                let b = k.ld(proj, bi);
                let beta = k.sigmoid(b);
                let bias = k.ld(dt_bias, i);
                let xx = k.fadd(a, bias);
                let ex = k.exp(xx);
                let one = k.f(1.0);
                let l = k.fadd(one, ex);
                let sp = k.log(l);
                let twenty = k.f(20.0);
                let big = k.fgt(xx, twenty);
                let dt = k.select_f(big, xx, sp);
                let al = k.ld(a_log, i);
                let ea = k.exp(al);
                let prod = k.fmul(ea, dt);
                let neg = k.fneg(prod);
                // With the `bf16` feature the gates hold g = log(decay) (vLLM keeps g, not
                // the decay: near 1 a bf16 decay is off by 2⁻⁹, a bf16 g by 2⁻⁹·|g|).
                let decay = if crate::msl::ACT_BF16 { neg } else { k.exp(neg) };
                let gi = k.iadd(gbase, i);
                k.st(gates, gi, beta);
                let di = k.iadd(gi, nv);
                k.st(gates, di, decay);
            });
        },
    );
    done(format!("dn_l2_gates_rows_k{}v{}", s.k_heads, s.v_heads), k, 4)
}

/// The gated delta rule over the rows of the block, in order, the state of the
/// group's columns held in registers from the first row to the last: `cols`
/// columns of one head per group, `slices` invocations per column (the
/// [`super::kernels::dn_recurrence_sliced`] layout). Bindings `state, proj,
/// gates, rec_state, out (f32 [m][nv·dv])`; groups `v_heads · dv / cols`.
pub fn dn_recurrence_rows(s: DeltaShape, sg: u32, cols: u32, slices: u32) -> Emitted {
    let (dk, dv) = (s.k_dim, s.v_dim);
    assert!(dv.is_multiple_of(cols) && dk.is_multiple_of(slices), "cols {cols} / slices {slices} do not divide {dv} / {dk}");
    let nrows = dk / slices;
    assert!(nrows <= 32, "{nrows} state rows per invocation: raise slices");
    let wg = cols * slices;
    let mut k = Kernel::new(wg, sg);
    let st = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let gates = k.buffer(2, Elem::Act);
    let state = k.buffer(3, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::State));
    let out = k.buffer(4, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::Out));
    let qs = k.shared(dk, true);
    let ks = k.shared(dk, true);
    let red = k.shared(wg, true);
    let deltas = k.shared(cols, true);
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let t = k.local_id();
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
    let key = k.u(s.key());
    let pw = k.u(s.proj_width());
    let nv = k.u(s.v_heads);
    let two_nv = k.u(2 * s.v_heads);
    let value = k.u(s.value());
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let step = k.u(wg);
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
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let first = k.ieq(sl, zero_u);
    let v_at = k.u(2 * s.key());
    let hv = k.imul(h, dvc);
    k.for_(zero_u, m, one_u, |k, row| {
        let rbase = k.imul(row, pw);
        k.for_(t, dkc, step, |k, i| {
            let qb = k.imul(kh, dkc);
            let qa = k.iadd(qb, i);
            let qa = k.iadd(rbase, qa);
            let q = k.ld(proj, qa);
            let q = k.fmul(q, scale);
            k.sst(qs, i, q);
            let ka = k.iadd(qa, key);
            let kv = k.ld(proj, ka);
            k.sst(ks, i, kv);
        });
        k.barrier();
        let gb = k.imul(row, two_nv);
        let bi = k.iadd(gb, h);
        let beta = k.ld(gates, bi);
        let di = k.iadd(bi, nv);
        let decay = k.ld(gates, di);
        let decay = if crate::msl::ACT_BF16 { k.exp(decay) } else { decay };
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
        k.if_(first, |k| {
            let mut kv = k.sld(red, jl);
            for x in 1..slices {
                let o = k.u(x * cols);
                let at = k.iadd(o, jl);
                let p = k.sld(red, at);
                kv = k.fadd(kv, p);
            }
            let vi = k.iadd(v_at, hv);
            let vi = k.iadd(vi, j);
            let vi = k.iadd(rbase, vi);
            let v = k.ld(proj, vi);
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
        k.if_(first, |k| {
            let mut sum = k.sld(red, jl);
            for x in 1..slices {
                let off = k.u(x * cols);
                let at = k.iadd(off, jl);
                let p = k.sld(red, at);
                sum = k.fadd(sum, p);
            }
            let ob = k.imul(row, value);
            let oi = k.iadd(ob, hv);
            let oi = k.iadd(oi, j);
            k.st(out, oi, sum);
        });
        // q, k and the partials are overwritten by the next row.
        k.barrier();
    });
    for r in 0..nrows as usize {
        let v = k.get(regs[r]);
        k.st(state, addrs[r], v);
    }
    done(format!("dn_recurrence_rows_k{}v{}_c{cols}s{slices}", s.k_heads, s.v_heads), k, 5)
}

/// The gated delta rule over the rows of the block by chunks of 16 positions,
/// in the WY form (the CPU engine's `deltanet_chunk.rs`, in this engine's order:
/// the state decays before it is read). For a chunk from the state `S₀`, with
/// the decays `a_t`, the gates `β_t`, `γ_t = a_0⋯a_t`, `r_ts = a_{s+1}⋯a_t`:
///
/// ```text
/// A_ts = β_t·r_ts·(k_t·k_s)  (s < t)      T = (I + A)⁻¹ (forward substitution)
/// Δ    = T·diag(β)·V − T·diag(β·γ)·(K·S₀)
/// O    = diag(γ)·Q̃·S₀ + M·Δ               M_ts = r_ts·(q̃_t·k_s)  (s ≤ t), q̃ = q/√dk
/// S    = γ_15·S₀ + Σ_t r_15,t·k_t·Δ_tᵀ
/// ```
///
/// A group holds `cols` value columns of one head, `slices` invocations per
/// column on consecutive lanes (`dk / slices` state rows each, in registers for
/// the whole block); `K·S₀` and `Q̃·S₀` are reduced over the slices by a
/// clustered subgroup sum. Rows past the block get `β = 0`, `a = 1` and zero
/// inputs. Other sums than the sequential kernel: judged at the tokens.
/// Bindings as [`dn_recurrence_rows`]; groups `v_heads · dv / cols`.
pub fn dn_recurrence_chunk(s: DeltaShape, sg: u32, cols: u32, slices: u32) -> Emitted {
    const C: u32 = 16;
    let (dk, dv) = (s.k_dim, s.v_dim);
    assert!(dv.is_multiple_of(cols) && dk.is_multiple_of(slices) && sg.is_multiple_of(slices), "cols {cols} / slices {slices} / wave {sg}");
    let nrows = dk / slices;
    let wg = cols * slices;
    assert!(wg.is_multiple_of(sg) && wg >= C, "group of {wg}");
    let mut k = Kernel::new(wg, sg);
    let st = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let gates = k.buffer(2, Elem::Act);
    let state = k.buffer(3, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::State));
    let out = k.buffer(4, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::Out));
    let qs = k.shared(C * dk, true);
    let ks = k.shared(C * dk, true);
    let vs = k.shared(C * cols, true);
    let beta_s = k.shared(C, true);
    let dec_s = k.shared(C, true);
    let gam_s = k.shared(C, true);
    let kk_s = k.shared(C * C, true);
    let qk_s = k.shared(C * C, true);
    let a_s = k.shared(C * C, true);
    let m_s = k.shared(C * C, true);
    let t_s = k.shared(C * C, true);
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let t = k.local_id();
    let blocks = k.u(dv / cols);
    let h = k.udiv(g, blocks);
    let cb = k.umod(g, blocks);
    let slc = k.u(slices);
    // Slices of a column on consecutive lanes.
    let jl = k.udiv(t, slc);
    let sl = k.umod(t, slc);
    let colsc = k.u(cols);
    let cb_base = k.imul(cb, colsc);
    let j = k.iadd(cb_base, jl);
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dkc = k.u(dk);
    let dvc = k.u(dv);
    let key = k.u(s.key());
    let pw = k.u(s.proj_width());
    let nv = k.u(s.v_heads);
    let two_nv = k.u(2 * s.v_heads);
    let value = k.u(s.value());
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let wgc = k.u(wg);
    let cc = k.u(C);
    let zero_u = k.u(0);
    let zero = k.f(0.0);
    let one = k.f(1.0);
    let rowsc = k.u(nrows);
    let kd0 = k.imul(sl, rowsc);
    let hs = k.u(dk * dv);
    let hbase = k.imul(h, hs);
    let row0 = k.imul(kd0, dvc);
    let base = k.iadd(hbase, row0);
    let base = k.iadd(base, j);
    let mut addrs = Vec::new();
    let mut regs = Vec::new();
    for r in 0..nrows {
        let off = k.u(r * dv);
        let at = k.iadd(base, off);
        let sv = k.ld(state, at);
        regs.push(k.var_f32(sv));
        addrs.push(at);
    }
    let v_at = k.u(2 * s.key());
    let hv = k.imul(h, dvc);
    let first_slice = k.ieq(sl, zero_u);
    k.for_(zero_u, m, cc, |k, c0| {
        // 1. Stage the chunk: q̃, k (dk each), v (this group's columns), β, a.
        let qk_n = k.u(C * dk);
        k.for_(t, qk_n, wgc, |k, e| {
            let ti = k.udiv(e, dkc);
            let d = k.umod(e, dkc);
            let row = k.iadd(c0, ti);
            let inside = k.ult(row, m);
            let qv = k.var_f32(zero);
            let kv = k.var_f32(zero);
            k.if_(inside, |k| {
                let rb = k.imul(row, pw);
                let qb = k.imul(kh, dkc);
                let qa = k.iadd(rb, qb);
                let qa = k.iadd(qa, d);
                let q = k.ld(proj, qa);
                let q = k.fmul(q, scale);
                k.set(qv, q);
                let ka = k.iadd(qa, key);
                let kx = k.ld(proj, ka);
                k.set(kv, kx);
            });
            let q = k.get(qv);
            k.sst(qs, e, q);
            let kx = k.get(kv);
            k.sst(ks, e, kx);
        });
        let v_n = k.u(C * cols);
        k.for_(t, v_n, wgc, |k, e| {
            let ti = k.udiv(e, colsc);
            let c = k.umod(e, colsc);
            let row = k.iadd(c0, ti);
            let inside = k.ult(row, m);
            let vv = k.var_f32(zero);
            k.if_(inside, |k| {
                let rb = k.imul(row, pw);
                let va = k.iadd(rb, v_at);
                let va = k.iadd(va, hv);
                let va = k.iadd(va, cb_base);
                let va = k.iadd(va, c);
                let x = k.ld(proj, va);
                k.set(vv, x);
            });
            let x = k.get(vv);
            k.sst(vs, e, x);
        });
        let is_t = k.ult(t, cc);
        k.if_(is_t, |k| {
            let row = k.iadd(c0, t);
            let inside = k.ult(row, m);
            let bv = k.var_f32(zero);
            let av = k.var_f32(one);
            k.if_(inside, |k| {
                let gb = k.imul(row, two_nv);
                let bi = k.iadd(gb, h);
                let b = k.ld(gates, bi);
                k.set(bv, b);
                let di = k.iadd(bi, nv);
                let a = k.ld(gates, di);
                // The bf16 build keeps log(decay) (`dn_l2_gates_rows`).
                let a = if crate::msl::ACT_BF16 { k.exp(a) } else { a };
                k.set(av, a);
            });
            let b = k.get(bv);
            k.sst(beta_s, t, b);
            let a = k.get(av);
            k.sst(dec_s, t, a);
        });
        k.barrier();
        // 2. K·Kᵀ and Q̃·Kᵀ (C × C), γ by one invocation.
        let cc2 = k.u(C * C);
        k.for_(t, cc2, wgc, |k, e| {
            let ti = k.udiv(e, cc);
            let si = k.umod(e, cc);
            let tb = k.imul(ti, dkc);
            let sb = k.imul(si, dkc);
            let one_u = k.u(1);
            let a1 = k.var_f32(zero);
            let a2 = k.var_f32(zero);
            k.for_(zero_u, dkc, one_u, |k, d| {
                let ka = k.iadd(sb, d);
                let kx = k.sld(ks, ka);
                let ta = k.iadd(tb, d);
                let kt = k.sld(ks, ta);
                let qt = k.sld(qs, ta);
                let x = k.get(a1);
                let x = k.fma(kt, kx, x);
                k.set(a1, x);
                let y = k.get(a2);
                let y = k.fma(qt, kx, y);
                k.set(a2, y);
            });
            let x = k.get(a1);
            k.sst(kk_s, e, x);
            let y = k.get(a2);
            k.sst(qk_s, e, y);
        });
        let t0 = k.ieq(t, zero_u);
        k.if_(t0, |k| {
            let mut gm = one;
            for u in 0..C {
                let uc = k.u(u);
                let a = k.sld(dec_s, uc);
                gm = k.fmul(gm, a);
                k.sst(gam_s, uc, gm);
            }
        });
        k.barrier();
        // 3. A and M, with r_ts = a_{s+1}⋯a_t as a product.
        k.for_(t, cc2, wgc, |k, e| {
            let ti = k.udiv(e, cc);
            let si = k.umod(e, cc);
            let one_u = k.u(1);
            let rv = k.var_f32(one);
            let s1 = k.iadd(si, one_u);
            let t1 = k.iadd(ti, one_u);
            k.for_(s1, t1, one_u, |k, u| {
                let a = k.sld(dec_s, u);
                let x = k.get(rv);
                let x = k.fmul(x, a);
                k.set(rv, x);
            });
            let r = k.get(rv);
            let lower = k.ult(si, ti);
            let s_le_t = k.ult(si, t1);
            let b = k.sld(beta_s, ti);
            let kk = k.sld(kk_s, e);
            let av = k.fmul(b, r);
            let av = k.fmul(av, kk);
            let av = k.select_f(lower, av, zero);
            k.sst(a_s, e, av);
            let qk = k.sld(qk_s, e);
            let mv = k.fmul(r, qk);
            let mv = k.select_f(s_le_t, mv, zero);
            k.sst(m_s, e, mv);
        });
        k.barrier();
        // 4. T = (I + A)⁻¹, one invocation per column, rows in order.
        k.if_(is_t, |k| {
            for ti in 0..C {
                let mut acc = zero;
                let tic = k.u(ti);
                let diag = k.ieq(t, tic);
                let e0 = k.select_f(diag, one, zero);
                for jj in 0..ti {
                    let ai = k.u(ti * C + jj);
                    let a = k.sld(a_s, ai);
                    let tj = k.u(jj * C);
                    let tj = k.iadd(tj, t);
                    let tv = k.sld(t_s, tj);
                    acc = k.fma(a, tv, acc);
                }
                let v = k.fsub(e0, acc);
                let at = k.u(ti * C);
                let at = k.iadd(at, t);
                k.sst(t_s, at, v);
            }
        });
        k.barrier();
        // 5. K·S₀ and Q̃·S₀ for this column, reduced over the slices.
        let mut ks0 = Vec::new();
        let mut qs0 = Vec::new();
        for ti in 0..C {
            let rb = k.u(ti * dk);
            let rb = k.iadd(rb, kd0);
            let mut pk = zero;
            let mut pq = zero;
            for (r, reg) in regs.iter().enumerate() {
                let rc = k.u(r as u32);
                let at = k.iadd(rb, rc);
                let kx = k.sld(ks, at);
                let qx = k.sld(qs, at);
                let sv = k.get(*reg);
                pk = k.fma(kx, sv, pk);
                pq = k.fma(qx, sv, pq);
            }
            ks0.push(k.sg_fadd_clustered(pk, slices));
            qs0.push(k.sg_fadd_clustered(pq, slices));
        }
        // 6. Δ_t = Σ_s T_ts·β_s·(v_s − γ_s·(K·S₀)_s).
        let mut us = Vec::new();
        for si in 0..C {
            let sc = k.u(si);
            let b = k.sld(beta_s, sc);
            let gm = k.sld(gam_s, sc);
            let va = k.u(si * cols);
            let va = k.iadd(va, jl);
            let v = k.sld(vs, va);
            let gk = k.fmul(gm, ks0[si as usize]);
            let d = k.fsub(v, gk);
            us.push(k.fmul(b, d));
        }
        let mut deltas = Vec::new();
        for ti in 0..C {
            let mut acc = zero;
            for si in 0..=ti {
                let ta = k.u(ti * C + si);
                let tv = k.sld(t_s, ta);
                acc = k.fma(tv, us[si as usize], acc);
            }
            deltas.push(acc);
        }
        // 7. O_t = γ_t·(Q̃·S₀)_t + Σ_{s≤t} M_ts·Δ_s, written by slice 0.
        for ti in 0..C {
            let tc = k.u(ti);
            let gm = k.sld(gam_s, tc);
            let mut o = k.fmul(gm, qs0[ti as usize]);
            for si in 0..=ti {
                let ma = k.u(ti * C + si);
                let mv = k.sld(m_s, ma);
                o = k.fma(mv, deltas[si as usize], o);
            }
            let row = k.iadd(c0, tc);
            let inside = k.ult(row, m);
            let write = k.land(inside, first_slice);
            k.if_(write, |k| {
                let ob = k.imul(row, value);
                let oi = k.iadd(ob, hv);
                let oi = k.iadd(oi, j);
                k.st(out, oi, o);
            });
        }
        // 8. S = γ_15·S + Σ_t r_15,t·k_t·Δ_t.
        let last = k.u(C - 1);
        let g15 = k.sld(gam_s, last);
        // r_15,t = γ_15 / γ_t would divide; the product of the decays after t.
        let mut rl = Vec::with_capacity(C as usize);
        let mut acc_r = one;
        let mut rev = Vec::new();
        for ti in (0..C).rev() {
            rev.push(acc_r);
            let tc = k.u(ti);
            let a = k.sld(dec_s, tc);
            acc_r = k.fmul(acc_r, a);
        }
        for ti in 0..C {
            rl.push(rev[(C - 1 - ti) as usize]);
        }
        for (r, reg) in regs.iter().enumerate() {
            let sv = k.get(*reg);
            let mut acc = k.fmul(g15, sv);
            for ti in 0..C {
                let ka = k.u(ti * dk + r as u32);
                let ka = k.iadd(ka, kd0);
                let kx = k.sld(ks, ka);
                let w = k.fmul(rl[ti as usize], kx);
                acc = k.fma(w, deltas[ti as usize], acc);
            }
            k.set(*reg, acc);
        }
        // The shared tiles are overwritten by the next chunk.
        k.barrier();
    });
    for (r, reg) in regs.iter().enumerate() {
        let v = k.get(*reg);
        k.st(state, addrs[r], v);
    }
    done(format!("dn_recurrence_chunk_k{}v{}_c{cols}s{slices}", s.k_heads, s.v_heads), k, 5)
}

/// The gated delta rule over the rows of the block without a barrier: the
/// `slices` invocations of a column on consecutive lanes, each with
/// `dk / slices` state rows in registers, reading its own rows of `q` and `k`
/// from memory; `kv` and `o` reduced over the slices by clustered subgroup
/// sums, `δ` computed by every slice of the column, `o` written by slice 0.
/// `vec`: the slice's rows of `q` and `k` read four floats at a time (`proj`
/// bound a second time as `uvec4`), same sums in the same order.
/// `pair`: `o` of a row reduced with `kv` of the next one (both read the state
/// the row left), so a row's path holds one clustered sum instead of two;
/// same sums in the same order.
/// Bindings as [`dn_recurrence_rows`]; groups `v_heads · dv / cols`.
pub fn dn_recurrence_cl(s: DeltaShape, sg: u32, cols: u32, slices: u32, vec: bool, pair: bool) -> Emitted {
    let (dk, dv) = (s.k_dim, s.v_dim);
    assert!(dv.is_multiple_of(cols) && dk.is_multiple_of(slices) && sg.is_multiple_of(slices), "cols {cols} / slices {slices} / wave {sg}");
    let nrows = dk / slices;
    let wg = cols * slices;
    assert!(wg.is_multiple_of(sg), "group of {wg}");
    let mut k = Kernel::new(wg, sg);
    let st = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let gates = k.buffer(2, Elem::Act);
    let state = k.buffer(3, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::State));
    let out = k.buffer(4, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::Out));
    // `proj` again, four floats per element (aligned: the row width, the key
    // offset and each slice's first row are multiples of 4).
    // Floats per `uvec4`: 4 f32, or 8 bf16 with the `bf16` feature.
    let per4: u32 = if crate::msl::ACT_BF16 { 8 } else { 4 };
    let vec = vec && nrows.is_multiple_of(per4) && s.proj_width().is_multiple_of(per4) && s.key().is_multiple_of(per4) && dk.is_multiple_of(per4);
    let proj4 = if vec { Some(k.buffer(1, Elem::UVec4)) } else { None };
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let t = k.local_id();
    let blocks = k.u(dv / cols);
    let h = k.udiv(g, blocks);
    let cb = k.umod(g, blocks);
    let slc = k.u(slices);
    let jl = k.udiv(t, slc);
    let sl = k.umod(t, slc);
    let colsc = k.u(cols);
    let cb_base = k.imul(cb, colsc);
    let j = k.iadd(cb_base, jl);
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dkc = k.u(dk);
    let dvc = k.u(dv);
    let key = k.u(s.key());
    let pw = k.u(s.proj_width());
    let nv = k.u(s.v_heads);
    let two_nv = k.u(2 * s.v_heads);
    let value = k.u(s.value());
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let zero = k.f(0.0);
    let rowsc = k.u(nrows);
    let kd0 = k.imul(sl, rowsc);
    let hs = k.u(dk * dv);
    let hbase = k.imul(h, hs);
    let row0 = k.imul(kd0, dvc);
    let base = k.iadd(hbase, row0);
    let base = k.iadd(base, j);
    let mut addrs = Vec::new();
    let mut regs = Vec::new();
    for r in 0..nrows {
        let off = k.u(r * dv);
        let at = k.iadd(base, off);
        let sv = k.ld(state, at);
        regs.push(k.var_f32(sv));
        addrs.push(at);
    }
    let v_at = k.u(2 * s.key());
    let hv = k.imul(h, dvc);
    let first_slice = k.ieq(sl, zero_u);
    let qoff = k.imul(kh, dkc);
    let qoff = k.iadd(qoff, kd0);
    // The inputs of a row: its k and q rows (this invocation's slice), v of the
    // column, β and the decay. Those of the next row are loaded while the
    // current one is reduced (the loop carries them).
    let last_row = k.isub(m, one_u);
    let load = |k: &mut Kernel, row: Word| -> (Vec<Word>, Vec<Word>, Word, Word, Word) {
        let rb = k.imul(row, pw);
        let qa = k.iadd(rb, qoff);
        let ka = k.iadd(qa, key);
        let mut kx = Vec::with_capacity(nrows as usize);
        let mut qx = Vec::with_capacity(nrows as usize);
        if let Some(p4) = proj4 {
            let sh = k.u(per4.trailing_zeros());
            let ka4 = k.shr(ka, sh);
            let qa4 = k.shr(qa, sh);
            // The floats of one `uvec4`, in order.
            let unpack = |k: &mut Kernel, v: Word| -> Vec<Word> {
                let mut f = Vec::with_capacity(per4 as usize);
                for e in 0..4 {
                    let u = k.comp(v, e);
                    if crate::msl::ACT_BF16 {
                        f.push(k.lo_f(u));
                        f.push(k.hi_f(u));
                    } else {
                        f.push(k.bits_to_f(u));
                    }
                }
                f
            };
            for c in 0..nrows / per4 {
                let cc = k.u(c);
                let at = k.iadd(ka4, cc);
                let v = k.ld(p4, at);
                kx.extend(unpack(k, v));
                let at = k.iadd(qa4, cc);
                let v = k.ld(p4, at);
                for q in unpack(k, v) {
                    qx.push(k.fmul(q, scale));
                }
            }
        } else {
            for r in 0..nrows {
                let rc = k.u(r);
                let at = k.iadd(ka, rc);
                kx.push(k.ld(proj, at));
                let at = k.iadd(qa, rc);
                let q = k.ld(proj, at);
                qx.push(k.fmul(q, scale));
            }
        }
        let vi = k.iadd(rb, v_at);
        let vi = k.iadd(vi, hv);
        let vi = k.iadd(vi, j);
        let v = k.ld(proj, vi);
        let gb = k.imul(row, two_nv);
        let bi = k.iadd(gb, h);
        let beta = k.ld(gates, bi);
        let di = k.iadd(bi, nv);
        let decay = k.ld(gates, di);
        let decay = if crate::msl::ACT_BF16 { k.exp(decay) } else { decay };
        (kx, qx, v, beta, decay)
    };
    let (k0, q0, v0, b0, d0) = load(&mut k, zero_u);
    let kvar: Vec<_> = k0.iter().map(|&x| k.var_f32(x)).collect();
    let qvar: Vec<_> = q0.iter().map(|&x| k.var_f32(x)).collect();
    let vvar = k.var_f32(v0);
    let bvar = k.var_f32(b0);
    let dvar = k.var_f32(d0);
    // `pair`: the scaled q of the previous row, its `o` still to reduce.
    let qprev: Vec<_> = q0.iter().map(|&x| k.var_f32(x)).collect();
    k.for_(zero_u, m, one_u, |k, row| {
        let kx: Vec<_> = kvar.iter().map(|&v| k.get(v)).collect();
        let qx: Vec<_> = qvar.iter().map(|&v| k.get(v)).collect();
        let v = k.get(vvar);
        let beta = k.get(bvar);
        let decay = k.get(dvar);
        // Next row's inputs (the last row reloads itself).
        let nxt = k.iadd(row, one_u);
        let past = k.ult(last_row, nxt);
        let nxt = k.select_u(past, last_row, nxt);
        let (nk, nq, nvv, nb, nd) = load(k, nxt);
        let mut sv = Vec::with_capacity(nrows as usize);
        let mut acc = zero;
        let mut acc_o = zero;
        for r in 0..nrows as usize {
            let x0 = k.get(regs[r]);
            if pair {
                let qp = k.get(qprev[r]);
                acc_o = k.fma(x0, qp, acc_o);
            }
            let x = k.fmul(x0, decay);
            acc = k.fma(x, kx[r], acc);
            sv.push(x);
        }
        let kv = k.sg_fadd_clustered(acc, slices);
        if pair {
            // `o` of the previous row, from the state it left.
            let o = k.sg_fadd_clustered(acc_o, slices);
            let started = k.ult(zero_u, row);
            let write = k.land(first_slice, started);
            k.if_(write, |k| {
                let prev = k.isub(row, one_u);
                let ob = k.imul(prev, value);
                let oi = k.iadd(ob, hv);
                let oi = k.iadd(oi, j);
                k.st(out, oi, o);
            });
        }
        let diff = k.fsub(v, kv);
        let delta = k.fmul(diff, beta);
        let mut o = zero;
        for r in 0..nrows as usize {
            let nsv = k.fma(kx[r], delta, sv[r]);
            k.set(regs[r], nsv);
            if !pair {
                o = k.fma(nsv, qx[r], o);
            }
        }
        if pair {
            for (var, &q) in qprev.iter().zip(qx.iter()) {
                k.set(*var, q);
            }
        } else {
            let o = k.sg_fadd_clustered(o, slices);
            k.if_(first_slice, |k| {
                let ob = k.imul(row, value);
                let oi = k.iadd(ob, hv);
                let oi = k.iadd(oi, j);
                k.st(out, oi, o);
            });
        }
        for (var, x) in kvar.iter().zip(nk) {
            k.set(*var, x);
        }
        for (var, x) in qvar.iter().zip(nq) {
            k.set(*var, x);
        }
        k.set(vvar, nvv);
        k.set(bvar, nb);
        k.set(dvar, nd);
    });
    if pair {
        // `o` of the last row.
        let mut o = zero;
        for r in 0..nrows as usize {
            let x = k.get(regs[r]);
            let q = k.get(qprev[r]);
            o = k.fma(x, q, o);
        }
        let o = k.sg_fadd_clustered(o, slices);
        k.if_(first_slice, |k| {
            let ob = k.imul(last_row, value);
            let oi = k.iadd(ob, hv);
            let oi = k.iadd(oi, j);
            k.st(out, oi, o);
        });
    }
    for (r, reg) in regs.iter().enumerate() {
        let v = k.get(*reg);
        k.st(state, addrs[r], v);
    }
    done(format!("dn_recurrence_cl_k{}v{}_c{cols}s{slices}{}{}", s.k_heads, s.v_heads, if vec { "_v4" } else { "" }, if pair { "_pair" } else { "" }), k, 5)
}

/// [`dn_recurrence_cl`] with its inputs staged `c` rows at a time: the group
/// loads the rows of `q` and `k` of its key head, `v` of its columns, β and the
/// decay of the next `c` rows into registers (`uvec4` reads, every invocation
/// a share) while it runs the current `c` rows from workgroup memory, so the
/// latency of global memory is paid once per `c` rows instead of once per row.
/// Same sums in the same order as [`dn_recurrence_cl`].
/// Bindings as [`dn_recurrence_rows`]; groups `v_heads · dv / cols`.
pub fn dn_recurrence_st(s: DeltaShape, sg: u32, cols: u32, slices: u32, c: u32) -> Emitted {
    let (dk, dv) = (s.k_dim, s.v_dim);
    assert!(dv.is_multiple_of(cols) && dk.is_multiple_of(slices) && sg.is_multiple_of(slices), "cols {cols} / slices {slices} / wave {sg}");
    let nrows = dk / slices;
    let wg = cols * slices;
    assert!(wg.is_multiple_of(sg), "group of {wg}");
    let q4 = dk / 4;
    assert!(
        nrows.is_multiple_of(4) && s.proj_width().is_multiple_of(4) && s.key().is_multiple_of(4) && (c * q4).is_multiple_of(wg) && 2 * c <= wg,
        "staging {c} rows with a group of {wg}"
    );
    let mut k = Kernel::new(wg, sg);
    let st = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::F32);
    let proj4 = k.buffer(1, Elem::UVec4);
    let gates = k.buffer(2, Elem::F32);
    let state = k.buffer(3, Elem::F32);
    let out = k.buffer(4, Elem::F32);
    let sq = k.shared_vec4(c * q4);
    let sk = k.shared_vec4(c * q4);
    let svs = k.shared(c * cols, true);
    let sbd = k.shared(2 * c, true);
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let t = k.local_id();
    let blocks = k.u(dv / cols);
    let h = k.udiv(g, blocks);
    let cb = k.umod(g, blocks);
    let slc = k.u(slices);
    let jl = k.udiv(t, slc);
    let sl = k.umod(t, slc);
    let colsc = k.u(cols);
    let cb_base = k.imul(cb, colsc);
    let j = k.iadd(cb_base, jl);
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dvc = k.u(dv);
    let pw = k.u(s.proj_width());
    let pw4 = k.u(s.proj_width() / 4);
    let nv = k.u(s.v_heads);
    let two_nv = k.u(2 * s.v_heads);
    let value = k.u(s.value());
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let zero = k.f(0.0);
    let rowsc = k.u(nrows);
    let kd0 = k.imul(sl, rowsc);
    let hs = k.u(dk * dv);
    let hbase = k.imul(h, hs);
    let row0 = k.imul(kd0, dvc);
    let base = k.iadd(hbase, row0);
    let base = k.iadd(base, j);
    let mut addrs = Vec::new();
    let mut regs = Vec::new();
    for r in 0..nrows {
        let off = k.u(r * dv);
        let at = k.iadd(base, off);
        let sv = k.ld(state, at);
        regs.push(k.var_f32(sv));
        addrs.push(at);
    }
    let hv = k.imul(h, dvc);
    let first_slice = k.ieq(sl, zero_u);
    let last_row = k.isub(m, one_u);
    let cc = k.u(c);
    // This invocation's share of a stage: cells `t + i·wg` of the q and k
    // tiles (`c` rows × `dk/4` cells), entries `t + i·wg` of the v tile (`c`
    // rows × `cols`), and β (t < c) or the decay (c ≤ t < 2c).
    let q4c = k.u(q4);
    let qk_off = k.imul(kh, q4c);
    let key4 = k.u(s.key() / 4);
    let n_qk = c * q4 / wg;
    let n_v = (c * cols).div_ceil(wg);
    let v_at = k.u(2 * s.key());
    let v_base = k.iadd(v_at, hv);
    let v_base = k.iadd(v_base, cb_base);
    let row_of = |k: &mut Kernel, c0: Word, tt: Word| {
        let r = k.iadd(c0, tt);
        let past = k.ult(last_row, r);
        k.select_u(past, last_row, r)
    };
    let is_decay = k.ult(t, cc);
    let is_decay = k.b_not(is_decay);
    let bd_t = {
        let tm = k.isub(t, cc);
        k.select_u(is_decay, tm, t)
    };
    let bd_t = {
        let lim = k.isub(cc, one_u);
        let past = k.ult(lim, bd_t);
        k.select_u(past, lim, bd_t)
    };
    let fetch = |k: &mut Kernel, c0: Word| -> Vec<Word> {
        let mut vals = Vec::new();
        for i in 0..n_qk {
            let ic = k.u(i * wg);
            let idx = k.iadd(t, ic);
            let tt = k.udiv(idx, q4c);
            let cell = k.umod(idx, q4c);
            let r = row_of(k, c0, tt);
            let ra = k.imul(r, pw4);
            let ra = k.iadd(ra, qk_off);
            let qa = k.iadd(ra, cell);
            vals.push(k.ld(proj4, qa));
            let ka = k.iadd(qa, key4);
            vals.push(k.ld(proj4, ka));
        }
        for i in 0..n_v {
            let ic = k.u(i * wg);
            let idx = k.iadd(t, ic);
            let lim = k.u(c * cols - 1);
            let past = k.ult(lim, idx);
            let idx = k.select_u(past, lim, idx);
            let tt = k.udiv(idx, colsc);
            let jj = k.umod(idx, colsc);
            let r = row_of(k, c0, tt);
            let ra = k.imul(r, pw);
            let va = k.iadd(ra, v_base);
            let va = k.iadd(va, jj);
            vals.push(k.ld(proj, va));
        }
        let r = row_of(k, c0, bd_t);
        let gb = k.imul(r, two_nv);
        let gi = k.iadd(gb, h);
        let off = k.select_u(is_decay, nv, zero_u);
        let gi = k.iadd(gi, off);
        vals.push(k.ld(gates, gi));
        vals
    };
    let first = fetch(&mut k, zero_u);
    let mut vars = Vec::new();
    let mut is_vec = Vec::new();
    for (n, v) in first.iter().enumerate() {
        let vec = n < 2 * n_qk as usize;
        vars.push(if vec { k.var_uvec4(*v) } else { k.var_f32(*v) });
        is_vec.push(vec);
    }
    let two_c = k.u(2 * c);
    let in_bd = k.ult(t, two_c);
    let two = k.u(2);
    let kd4 = k.shr(kd0, two);
    k.for_(zero_u, m, cc, |k, c0| {
        // The previous stage is read by everyone before it is overwritten.
        k.barrier();
        let mut n = 0usize;
        for i in 0..n_qk {
            let ic = k.u(i * wg);
            let idx = k.iadd(t, ic);
            let qv = k.get(vars[n]);
            let mut sc = Vec::with_capacity(4);
            for e in 0..4 {
                let u = k.comp(qv, e);
                let f = k.bits_to_f(u);
                let f = k.fmul(f, scale);
                sc.push(k.f_to_bits(f));
            }
            let qs = k.uvec4([sc[0], sc[1], sc[2], sc[3]]);
            k.sst(sq, idx, qs);
            let kv = k.get(vars[n + 1]);
            k.sst(sk, idx, kv);
            n += 2;
        }
        for i in 0..n_v {
            let ic = k.u(i * wg);
            let idx = k.iadd(t, ic);
            let lim = k.u(c * cols);
            let inside = k.ult(idx, lim);
            let v = k.get(vars[n]);
            k.if_(inside, |k| k.sst(svs, idx, v));
            n += 1;
        }
        let bd = k.get(vars[n]);
        k.if_(in_bd, |k| k.sst(sbd, t, bd));
        k.barrier();
        // The next stage's inputs, in flight while this one runs.
        let nxt = k.iadd(c0, cc);
        let next = fetch(k, nxt);
        for (var, v) in vars.iter().zip(next) {
            k.set(*var, v);
        }
        let left = k.isub(m, c0);
        let short = k.ult(left, cc);
        let count = k.select_u(short, left, cc);
        k.for_(zero_u, count, one_u, |k, tt| {
            let rb = k.imul(tt, q4c);
            let cell0 = k.iadd(rb, kd4);
            let mut kx = Vec::with_capacity(nrows as usize);
            let mut qx = Vec::with_capacity(nrows as usize);
            for c4 in 0..nrows / 4 {
                let o = k.u(c4);
                let at = k.iadd(cell0, o);
                let kv = k.sld(sk, at);
                let qv = k.sld(sq, at);
                for e in 0..4 {
                    let u = k.comp(kv, e);
                    kx.push(k.bits_to_f(u));
                    let u = k.comp(qv, e);
                    qx.push(k.bits_to_f(u));
                }
            }
            let vi = k.imul(tt, colsc);
            let vi = k.iadd(vi, jl);
            let v = k.sld(svs, vi);
            let beta = k.sld(sbd, tt);
            let di = k.iadd(tt, cc);
            let decay = k.sld(sbd, di);
            let mut sv = Vec::with_capacity(nrows as usize);
            let mut acc = zero;
            for r in 0..nrows as usize {
                let x = k.get(regs[r]);
                let x = k.fmul(x, decay);
                acc = k.fma(x, kx[r], acc);
                sv.push(x);
            }
            let kvs = k.sg_fadd_clustered(acc, slices);
            let diff = k.fsub(v, kvs);
            let delta = k.fmul(diff, beta);
            let mut o = zero;
            for r in 0..nrows as usize {
                let nsv = k.fma(kx[r], delta, sv[r]);
                k.set(regs[r], nsv);
                o = k.fma(nsv, qx[r], o);
            }
            let o = k.sg_fadd_clustered(o, slices);
            k.if_(first_slice, |k| {
                let row = k.iadd(c0, tt);
                let ob = k.imul(row, value);
                let oi = k.iadd(ob, hv);
                let oi = k.iadd(oi, j);
                k.st(out, oi, o);
            });
        });
    });
    for (r, reg) in regs.iter().enumerate() {
        let v = k.get(*reg);
        k.st(state, addrs[r], v);
    }
    done(format!("dn_recurrence_st_k{}v{}_c{cols}s{slices}_r{c}", s.k_heads, s.v_heads), k, 5)
}

/// Floats of one (chunk, value head) in the scratch of the two-pass chunked
/// recurrence: `W1 = T·diag(β)`, `W2 = T·diag(β·γ)`, `M` (16 × 16 each), `γ`
/// and `r_15,t` (16 each).
pub const CHUNK_PREP: u32 = 3 * 256 + 2 * 16;

/// First pass of the two-pass chunked recurrence: one group per (chunk of 16
/// rows, value head), one invocation per (t, s): `k_t·k_s`, `q̃_t·k_s`, `r_ts`,
/// `A`, `M`, then `T = (I + A)⁻¹` by 16 invocations (rows in order), written as
/// `W1`, `W2`, `M`, `γ`, `r_15,·` (see [`dn_recurrence_chunk`]). Bindings
/// `state, proj, gates, prep (f32)`; groups `ceil(m / 16) · v_heads`.
pub fn dn_chunk_prep(s: DeltaShape, sg: u32) -> Emitted {
    const C: u32 = 16;
    let dk = s.k_dim;
    let wg = C * C;
    let mut k = Kernel::new(wg, sg);
    let st = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let gates = k.buffer(2, Elem::Act);
    let prep = k.buffer(3, Elem::F32);
    let qs = k.shared(C * dk, true);
    let ks = k.shared(C * dk, true);
    let beta_s = k.shared(C, true);
    let dec_s = k.shared(C, true);
    let a_s = k.shared(C * C, true);
    let t_s = k.shared(C * C, true);
    let m = rows(&mut k, st);
    let gg = k.flat_group();
    let nvc = k.u(s.v_heads);
    let h = k.umod(gg, nvc);
    let ch = k.udiv(gg, nvc);
    let cc = k.u(C);
    let c0 = k.imul(ch, cc);
    let t = k.local_id();
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dkc = k.u(dk);
    let key = k.u(s.key());
    let pw = k.u(s.proj_width());
    let two_nv = k.u(2 * s.v_heads);
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let wgc = k.u(wg);
    let zero_u = k.u(0);
    let one_u = k.u(1);
    let zero = k.f(0.0);
    let one = k.f(1.0);
    let qk_n = k.u(C * dk);
    k.for_(t, qk_n, wgc, |k, e| {
        let ti = k.udiv(e, dkc);
        let d = k.umod(e, dkc);
        let row = k.iadd(c0, ti);
        let inside = k.ult(row, m);
        let qv = k.var_f32(zero);
        let kv = k.var_f32(zero);
        k.if_(inside, |k| {
            let rb = k.imul(row, pw);
            let qb = k.imul(kh, dkc);
            let qa = k.iadd(rb, qb);
            let qa = k.iadd(qa, d);
            let q = k.ld(proj, qa);
            let q = k.fmul(q, scale);
            k.set(qv, q);
            let ka = k.iadd(qa, key);
            let kx = k.ld(proj, ka);
            k.set(kv, kx);
        });
        let q = k.get(qv);
        k.sst(qs, e, q);
        let kx = k.get(kv);
        k.sst(ks, e, kx);
    });
    let is_t = k.ult(t, cc);
    k.if_(is_t, |k| {
        let row = k.iadd(c0, t);
        let inside = k.ult(row, m);
        let bv = k.var_f32(zero);
        let av = k.var_f32(one);
        k.if_(inside, |k| {
            let gb = k.imul(row, two_nv);
            let bi = k.iadd(gb, h);
            let b = k.ld(gates, bi);
            k.set(bv, b);
            let nv = k.u(s.v_heads);
            let di = k.iadd(bi, nv);
            let a = k.ld(gates, di);
            // The bf16 build keeps log(decay) (`dn_l2_gates_rows`).
            let a = if crate::msl::ACT_BF16 { k.exp(a) } else { a };
            k.set(av, a);
        });
        let b = k.get(bv);
        k.sst(beta_s, t, b);
        let a = k.get(av);
        k.sst(dec_s, t, a);
    });
    k.barrier();
    // One (t, s) per invocation.
    let ti = k.udiv(t, cc);
    let si = k.umod(t, cc);
    let tb = k.imul(ti, dkc);
    let sb = k.imul(si, dkc);
    let a1 = k.var_f32(zero);
    let a2 = k.var_f32(zero);
    k.for_(zero_u, dkc, one_u, |k, d| {
        let ka = k.iadd(sb, d);
        let kx = k.sld(ks, ka);
        let ta = k.iadd(tb, d);
        let kt = k.sld(ks, ta);
        let qt = k.sld(qs, ta);
        let x = k.get(a1);
        let x = k.fma(kt, kx, x);
        k.set(a1, x);
        let y = k.get(a2);
        let y = k.fma(qt, kx, y);
        k.set(a2, y);
    });
    let rv = k.var_f32(one);
    let s1 = k.iadd(si, one_u);
    let t1 = k.iadd(ti, one_u);
    k.for_(s1, t1, one_u, |k, u| {
        let a = k.sld(dec_s, u);
        let x = k.get(rv);
        let x = k.fmul(x, a);
        k.set(rv, x);
    });
    let r = k.get(rv);
    let lower = k.ult(si, ti);
    let s_le_t = k.ult(si, t1);
    let b = k.sld(beta_s, ti);
    let kk = k.get(a1);
    let av = k.fmul(b, r);
    let av = k.fmul(av, kk);
    let av = k.select_f(lower, av, zero);
    k.sst(a_s, t, av);
    let qk = k.get(a2);
    let mv = k.fmul(r, qk);
    let mv = k.select_f(s_le_t, mv, zero);
    let prep_size = k.u(CHUNK_PREP);
    let slot = k.imul(ch, nvc);
    let slot = k.iadd(slot, h);
    let pbase = k.imul(slot, prep_size);
    let m_at = k.u(2 * C * C);
    let m_at = k.iadd(pbase, m_at);
    let m_at = k.iadd(m_at, t);
    k.st(prep, m_at, mv);
    k.barrier();
    k.if_(is_t, |k| {
        for ti in 0..C {
            let mut acc = zero;
            let tic = k.u(ti);
            let diag = k.ieq(t, tic);
            let e0 = k.select_f(diag, one, zero);
            for jj in 0..ti {
                let ai = k.u(ti * C + jj);
                let a = k.sld(a_s, ai);
                let tj = k.u(jj * C);
                let tj = k.iadd(tj, t);
                let tv = k.sld(t_s, tj);
                acc = k.fma(a, tv, acc);
            }
            let v = k.fsub(e0, acc);
            let at = k.u(ti * C);
            let at = k.iadd(at, t);
            k.sst(t_s, at, v);
        }
    });
    let t0 = k.ieq(t, zero_u);
    k.if_(t0, |k| {
        // γ_t and r_15,t.
        let mut gm = one;
        for u in 0..C {
            let uc = k.u(u);
            let a = k.sld(dec_s, uc);
            gm = k.fmul(gm, a);
            let at = k.u(3 * C * C + u);
            let at = k.iadd(pbase, at);
            k.st(prep, at, gm);
        }
        let mut acc = one;
        for u in (0..C).rev() {
            let at = k.u(3 * C * C + C + u);
            let at = k.iadd(pbase, at);
            k.st(prep, at, acc);
            let uc = k.u(u);
            let a = k.sld(dec_s, uc);
            acc = k.fmul(acc, a);
        }
    });
    k.barrier();
    // W1 = T·diag(β), W2 = T·diag(β·γ): γ_s recomputed as a product.
    let tv = k.sld(t_s, t);
    let bs = k.sld(beta_s, si);
    let w1 = k.fmul(tv, bs);
    let gv = k.var_f32(one);
    k.for_(zero_u, s1, one_u, |k, u| {
        let a = k.sld(dec_s, u);
        let x = k.get(gv);
        let x = k.fmul(x, a);
        k.set(gv, x);
    });
    let gs = k.get(gv);
    let w2 = k.fmul(w1, gs);
    let w1_at = k.iadd(pbase, t);
    k.st(prep, w1_at, w1);
    let off = k.u(C * C);
    let w2_at = k.iadd(w1_at, off);
    k.st(prep, w2_at, w2);
    done(format!("dn_chunk_prep_k{}v{}", s.k_heads, s.v_heads), k, 4)
}

/// Second pass of the two-pass chunked recurrence: one group per (value head,
/// block of `cols` columns), the chunks in order, the state in registers;
/// per chunk `K·S₀`, `Q̃·S₀` (clustered sums over the slices), `Δ = W1·V −
/// W2·(K·S₀)`, `O = diag(γ)·Q̃·S₀ + M·Δ`, `S = γ_15·S + Σ_t r_15,t·k_t·Δ_t`.
/// Bindings `state, proj, prep, rec_state, out`; groups `v_heads · dv / cols`.
pub fn dn_chunk_apply(s: DeltaShape, sg: u32, cols: u32, slices: u32) -> Emitted {
    const C: u32 = 16;
    let (dk, dv) = (s.k_dim, s.v_dim);
    assert!(dv.is_multiple_of(cols) && dk.is_multiple_of(slices) && sg.is_multiple_of(slices), "cols {cols} / slices {slices} / wave {sg}");
    let nrows = dk / slices;
    let wg = cols * slices;
    let mut k = Kernel::new(wg, sg);
    let st = k.buffer(0, Elem::U32);
    let proj = k.buffer(1, Elem::Act);
    let prep = k.buffer(2, Elem::F32);
    let state = k.buffer(3, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::State));
    let out = k.buffer(4, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::Out));
    let qs = k.shared(C * dk, true);
    let ks = k.shared(C * dk, true);
    let vs = k.shared(C * cols, true);
    let pp = k.shared(CHUNK_PREP, true);
    let m = rows(&mut k, st);
    let g = k.flat_group();
    let t = k.local_id();
    let blocks = k.u(dv / cols);
    let h = k.udiv(g, blocks);
    let cb = k.umod(g, blocks);
    let slc = k.u(slices);
    let jl = k.udiv(t, slc);
    let sl = k.umod(t, slc);
    let colsc = k.u(cols);
    let cb_base = k.imul(cb, colsc);
    let j = k.iadd(cb_base, jl);
    let ratio = k.u(s.v_heads / s.k_heads);
    let kh = k.udiv(h, ratio);
    let dkc = k.u(dk);
    let dvc = k.u(dv);
    let key = k.u(s.key());
    let pw = k.u(s.proj_width());
    let nvc = k.u(s.v_heads);
    let value = k.u(s.value());
    let scale = k.f(1.0 / (dk as f32).sqrt());
    let wgc = k.u(wg);
    let cc = k.u(C);
    let zero_u = k.u(0);
    let zero = k.f(0.0);
    let rowsc = k.u(nrows);
    let kd0 = k.imul(sl, rowsc);
    let hs = k.u(dk * dv);
    let hbase = k.imul(h, hs);
    let row0 = k.imul(kd0, dvc);
    let base = k.iadd(hbase, row0);
    let base = k.iadd(base, j);
    let mut addrs = Vec::new();
    let mut regs = Vec::new();
    for r in 0..nrows {
        let off = k.u(r * dv);
        let at = k.iadd(base, off);
        let sv = k.ld(state, at);
        regs.push(k.var_f32(sv));
        addrs.push(at);
    }
    let v_at = k.u(2 * s.key());
    let hv = k.imul(h, dvc);
    let first_slice = k.ieq(sl, zero_u);
    let prep_size = k.u(CHUNK_PREP);
    k.for_(zero_u, m, cc, |k, c0| {
        let qk_n = k.u(C * dk);
        k.for_(t, qk_n, wgc, |k, e| {
            let ti = k.udiv(e, dkc);
            let d = k.umod(e, dkc);
            let row = k.iadd(c0, ti);
            let inside = k.ult(row, m);
            let qv = k.var_f32(zero);
            let kv = k.var_f32(zero);
            k.if_(inside, |k| {
                let rb = k.imul(row, pw);
                let qb = k.imul(kh, dkc);
                let qa = k.iadd(rb, qb);
                let qa = k.iadd(qa, d);
                let q = k.ld(proj, qa);
                let q = k.fmul(q, scale);
                k.set(qv, q);
                let ka = k.iadd(qa, key);
                let kx = k.ld(proj, ka);
                k.set(kv, kx);
            });
            let q = k.get(qv);
            k.sst(qs, e, q);
            let kx = k.get(kv);
            k.sst(ks, e, kx);
        });
        let v_n = k.u(C * cols);
        k.for_(t, v_n, wgc, |k, e| {
            let ti = k.udiv(e, colsc);
            let c = k.umod(e, colsc);
            let row = k.iadd(c0, ti);
            let inside = k.ult(row, m);
            let vv = k.var_f32(zero);
            k.if_(inside, |k| {
                let rb = k.imul(row, pw);
                let va = k.iadd(rb, v_at);
                let va = k.iadd(va, hv);
                let va = k.iadd(va, cb_base);
                let va = k.iadd(va, c);
                let x = k.ld(proj, va);
                k.set(vv, x);
            });
            let x = k.get(vv);
            k.sst(vs, e, x);
        });
        let ch = k.udiv(c0, cc);
        let slot = k.imul(ch, nvc);
        let slot = k.iadd(slot, h);
        let pbase = k.imul(slot, prep_size);
        k.for_(t, prep_size, wgc, |k, e| {
            let at = k.iadd(pbase, e);
            let x = k.ld(prep, at);
            k.sst(pp, e, x);
        });
        k.barrier();
        let mut ks0 = Vec::new();
        let mut qs0 = Vec::new();
        for ti in 0..C {
            let rb = k.u(ti * dk);
            let rb = k.iadd(rb, kd0);
            let mut pk = zero;
            let mut pq = zero;
            for (r, reg) in regs.iter().enumerate() {
                let rc = k.u(r as u32);
                let at = k.iadd(rb, rc);
                let kx = k.sld(ks, at);
                let qx = k.sld(qs, at);
                let sv = k.get(*reg);
                pk = k.fma(kx, sv, pk);
                pq = k.fma(qx, sv, pq);
            }
            ks0.push(k.sg_fadd_clustered(pk, slices));
            qs0.push(k.sg_fadd_clustered(pq, slices));
        }
        let vals: Vec<_> = (0..C)
            .map(|si| {
                let va = k.u(si * cols);
                let va = k.iadd(va, jl);
                k.sld(vs, va)
            })
            .collect();
        let mut deltas = Vec::new();
        for ti in 0..C {
            let mut acc = zero;
            for si in 0..=ti {
                let w1 = k.u(ti * C + si);
                let w1 = k.sld(pp, w1);
                let w2 = k.u(C * C + ti * C + si);
                let w2 = k.sld(pp, w2);
                acc = k.fma(w1, vals[si as usize], acc);
                let neg = k.fmul(w2, ks0[si as usize]);
                acc = k.fsub(acc, neg);
            }
            deltas.push(acc);
        }
        for ti in 0..C {
            let ga = k.u(3 * C * C + ti);
            let gm = k.sld(pp, ga);
            let mut o = k.fmul(gm, qs0[ti as usize]);
            for si in 0..=ti {
                let ma = k.u(2 * C * C + ti * C + si);
                let mv = k.sld(pp, ma);
                o = k.fma(mv, deltas[si as usize], o);
            }
            let tc = k.u(ti);
            let row = k.iadd(c0, tc);
            let inside = k.ult(row, m);
            let write = k.land(inside, first_slice);
            k.if_(write, |k| {
                let ob = k.imul(row, value);
                let oi = k.iadd(ob, hv);
                let oi = k.iadd(oi, j);
                k.st(out, oi, o);
            });
        }
        let g15 = k.u(3 * C * C + C - 1);
        let g15 = k.sld(pp, g15);
        let rl: Vec<_> = (0..C)
            .map(|ti| {
                let at = k.u(3 * C * C + C + ti);
                k.sld(pp, at)
            })
            .collect();
        for (r, reg) in regs.iter().enumerate() {
            let sv = k.get(*reg);
            let mut acc = k.fmul(g15, sv);
            for ti in 0..C {
                let ka = k.u(ti * dk + r as u32);
                let ka = k.iadd(ka, kd0);
                let kx = k.sld(ks, ka);
                let w = k.fmul(rl[ti as usize], kx);
                acc = k.fma(w, deltas[ti as usize], acc);
            }
            k.set(*reg, acc);
        }
        k.barrier();
    });
    for (r, reg) in regs.iter().enumerate() {
        let v = k.get(*reg);
        k.st(state, addrs[r], v);
    }
    done(format!("dn_chunk_apply_k{}v{}_c{cols}s{slices}", s.k_heads, s.v_heads), k, 5)
}

/// Gated RMSNorm of every (row, value head): groups `m · v_heads`. Bindings
/// `out (f32 [m][nv·dv]), proj (f32, z), w, y (bf16 [m][nv·dv])`.
pub fn dn_gated_norm_rows(s: DeltaShape, sg: u32) -> Emitted {
    let dv = s.v_dim;
    let mut k = Kernel::new(dv, sg);
    let out = k.buffer(0, Elem::Act).scaled(crate::msl::dn_scale(crate::msl::DnScaled::Out));
    let proj = k.buffer(1, Elem::Act);
    let w = k.buffer(2, Elem::U16);
    let y = k.buffer(3, Elem::U16);
    let scratch = k.shared(k.subgroups() + 1, true);
    let gg = k.flat_group();
    let nvc = k.u(s.v_heads);
    let row = k.udiv(gg, nvc);
    let j = k.local_id();
    let dvc = k.u(dv);
    // `gg · dv + j` is the element of (row, head) in `[m][nv·dv]`.
    let hv = k.imul(gg, dvc);
    let at = k.iadd(hv, j);
    let x = k.ld(out, at);
    let x2 = k.fmul(x, x);
    let ss = k.wg_reduce(x2, scratch, false);
    let inv_d = k.f(1.0 / dv as f32);
    let e = k.f(s.eps);
    let mm = k.fmul(ss, inv_d);
    let mm = k.fadd(mm, e);
    let r = k.sqrt(mm);
    let one = k.f(1.0);
    let inv = k.fdiv(one, r);
    let wh = k.ld(w, j);
    let wf = k.bf16_to_f(wh);
    let pw = k.u(s.proj_width());
    let value = k.u(s.value());
    let rb = k.imul(row, pw);
    let z_at = k.u(s.z_at());
    let rv = k.imul(row, value);
    let in_row = k.isub(at, rv);
    let zi = k.iadd(rb, z_at);
    let zi = k.iadd(zi, in_row);
    let z = k.ld(proj, zi);
    let sz = k.sigmoid(z);
    let silu = k.fmul(z, sz);
    let n = k.fmul(x, inv);
    let n = k.fmul(n, wf);
    let o = k.fmul(n, silu);
    let b = k.f_to_bf16(o);
    k.st(y, at, b);
    done(format!("dn_gated_norm_rows_v{}", s.v_heads), k, 4)
}
