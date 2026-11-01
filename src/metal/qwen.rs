//! Qwen3.5 (0.8B to 9B, Qwen3.8-27B) and its sparse `qwen3_5_moe` variant
//! (Qwen3.6-35B-A3B) on the GPU: the checkpoint uploaded, one command buffer
//! encoded per token.
//!
//! The step is herbert-vk-bf16's decode step with its first, plain kernels
//! (`--tune sdpa.split=0 fuse.norm=0 fuse.dn=0 fuse.swiglu=0 rec.slices=0`),
//! each operation one dispatch; every dimension read from `config.json`. The
//! prompt goes by blocks of `prefill.block` rows through herbert-vk-bf16's
//! block kernels (`metal/prefill.rs`: GEMMs, attention by tiles of queries,
//! the DeltaNet's conv and recurrence walking the rows of the block), the head
//! on the last row only; `prefill.block=1` feeds it token by token through the
//! decode step. Per layer: the RMSNorm `(1 + w)`;
//! for a full-attention layer the fused `[q | k | v]` projection, the head
//! norms with the partial RoPE and the KV append in one kernel, the decode
//! attention with its output gate, `o_proj` added to the residual; for a
//! GatedDeltaNet layer the fused `[qkv | z | a | b]` projection, the causal
//! conv, the L2 norms and gates, the delta rule, the gated norm, `out_proj`
//! added to the residual; then the second norm, `[gate | up]`, SwiGLU, `down`
//! added to the residual. In `qwen3_5_moe` that MLP is the shared expert, its
//! output kept apart; the router's scores (`mlp.gate`), the route, the
//! `top_k` experts' `[gate | up]`, SwiGLU and `down`, their combination, and
//! the block's end (the shared expert gated by `σ(shared_expert_gate·x)`,
//! added to the experts and to the residual) follow.

use super::gpu::{Buffer, Gpu, Pipeline, Profile, Range};
use super::stats::Stats;
use super::kernels::{self as kn, Out, STATE_ROWS};
use super::dnmma;
use super::prefill as pf;
use super::qwen_kernels::{self as qk, AttnShape, DeltaShape};
use crate::core::bf16::{as_bytes, bf16_bytes_to_h16, f32_to_h16, Bf16};
use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::model::qwen::Checkpoint;
use crate::msl::{Emitted, ACT_BF16, ACT_BYTES};
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
    /// The prefill's settings.
    pub tune: Tune,
    /// The projections in Q4_0 quantized at load; `false` = the checkpoint's
    /// bf16 (the witness).
    pub q4: bool,
    /// The head once more after the argmax, its sums kept in f32
    /// ([`QwenEngine::score`], herbert-metal-ppl): the 16-bit logits and the
    /// argmax are those of the step without it.
    pub logits_f32: bool,
}

/// The prefill's settings (`--tune key=value`, herbert-vk-bf16's names).
#[derive(Clone, Debug)]
pub struct Tune {
    /// `prefill.block`: rows per block; `1` the prompt token by token through
    /// the decode step. Not given: the buffers hold 2048 rows (fewer if the
    /// reserved positions are fewer) and with `gemm.auto` the rows per block,
    /// 512, 1024 or 2048, are timed at load (`chunk`).
    pub block: u32,
    /// `prefill.block` was given: blocks of that many rows, none timed.
    pub block_set: bool,
    /// `rec.chunk`: the block's delta rule. `0` the rows in order, a barrier
    /// per row (`dn_recurrence_rows`); `1` by chunks of 16 rows in the WY form
    /// (`dn_recurrence_chunk`); `2` the same in two passes (`dn_chunk_prep`,
    /// `dn_chunk_apply`); `3` the rows in order without a barrier
    /// (`dn_recurrence_cl`, herbert-vk-bf16's default off NVIDIA); `4` by
    /// chunks of 32 rows on the matrix units (`metal/dnmma.rs`, herbert-vk-bf16's
    /// tensor-core recurrence, the `bf16` feature).
    pub rec: u32,
    /// `rec.pcols`, `rec.pslices`: value columns per group, invocations per column.
    pub rec_cols: u32,
    pub rec_slices: u32,
    /// `rec.vec`, `rec.pair` (`rec.chunk=3`): q and k read four at a time; o
    /// reduced with the next row's kv.
    pub rec_vec: u32,
    pub rec_pair: u32,
    /// `rec.stage` (`rec.chunk=3`): inputs staged that many rows at a time (`dn_recurrence_st`).
    pub rec_stage: u32,
    /// `conv.cols`: the conv by segments of the block, channels per group (`0`: one channel per invocation).
    pub conv_cols: u32,
    /// `sdpa.prefill`: `0` one group per (row, head), `1` tiles of `sdpa.bq` queries × `sdpa.bk` keys
    /// (`sdpa_flash`, scalar), `2` the matrix units (`sdpa_mma`): groups of `sdpa.mq` queries × the
    /// heads of one K/V head, 8 rows per simdgroup, keys by tiles of `sdpa.mk` loaded by each simdgroup
    /// from the caches; `3` the same with the tiles of K and V staged in workgroup memory by the group;
    /// `4` Metal 4's `matmul2d` (`sdpa_mm2d`): groups of `sdpa.mq` queries × one head over `sdpa.msg`
    /// simdgroups, the output in cooperative tensors; `5` `matmul2d` of 16 × 32 × 16 by each simdgroup on
    /// register fragments (`sdpa_nax`, mlx's form on the M5): groups of 64 queries × one head, the head's
    /// columns split over `head_dim / 128` simdgroups; `6` the same with the descriptor's reduced precision;
    /// `7` the same with the softmax in base 2 (`exp2`, mlx's form).
    /// `sdpa.mq=0`: 16 for `2` and `3`, 64 for `4` to `7`.
    pub sdpa: u32,
    pub sdpa_bq: u32,
    pub sdpa_bk: u32,
    pub sdpa_mq: u32,
    pub sdpa_mk: u32,
    /// `sdpa.qreg` (`sdpa.prefill` 2 or 3): `1` the blocks of Q kept in registers across the tiles of keys.
    pub sdpa_qreg: u32,
    /// `sdpa.kt` (`sdpa.prefill` 2 or 3): `1` `Sᵀ = K·Qᵀ`, the blocks of K loaded straight, those of Q transposed.
    pub sdpa_kt: u32,
    /// `sdpa.msg` (`sdpa.prefill=4`): simdgroups per group.
    pub sdpa_msg: u32,
    /// `sdpa.auto`: `1` the prefill's attention kernel timed at load on a block (the scalar flash,
    /// `sdpa_mma` staged with Q in registers, `sdpa_mm2d`, `sdpa_nax`), the fastest kept; off when `sdpa.prefill` is given.
    pub sdpa_auto: u32,
    pub sdpa_set: bool,
    /// `auto.heat`: milliseconds of continuous GPU load before a timed comparison at load whose GPU has been idle
    /// (compiling, mostly) for more than 200 ms, so that the candidates are timed in the sustained regime: the M5's
    /// performance state drops within 2.5 s of load, and timed cold, blocks of 1024 rows won over 2048 on the 4B at
    /// pp8192 (7.4 against 7.7 s) and lost 9 % in the sustained run. **Measured negative, `0` by default**: the M5 does not settle in a regime, it throttles
    /// further the longer the GPU has been loaded (the 4B's blocks of 1024 rows: 7.4 s after 190 s of tuning, 8.9 s
    /// after 280 s, 9.5 s with the heater, every block slower alike), so heating before timing only drags every
    /// candidate down and the picks into the noise (with the heater: 8192 picked, 808–866 tok/s). What the timing
    /// can do is stay short and interleaved; what a measure must do is cool down after the load
    /// (`herbert-metal-bench --cool`).
    pub auto_heat: u32,
    /// `gemm.simt`: `1` the first layout, `2` the staged tiles `K`-major.
    pub simt: u32,
    /// `gemm.mma`: `1` the GEMMs on the matrix units (`simdgroup_matrix`) instead of the SIMT GEMM;
    /// `2` the dense ones by Metal 4's `matmul2d` (`gemm_mm2d`; the experts' stay on `gemm_mma_rows`);
    /// `3` the dense ones by `matmul2d` of 16 × 32 × 16 on register fragments (`gemm_nax`, 128 rows ×
    /// `gemm.tn` columns, at least 32).
    pub mma: u32,
    /// `gemm.tn` (`gemm.mma=2`): columns of a `matmul2d` tile, 64, 32 or 16 (rows: 64).
    pub tn: u32,
    /// `gemm.mform` (`gemm.mma=1`): the form of `gemm_mma_rows` over dense rows
    /// (`pf::MmaForm::of`): 0 the classic 64 × 64 tile through a 16 KiB f32
    /// tile, 1 X and W staged by 8 × 8 blocks, 2 blocks and the epilogue
    /// through the staging area (8 KiB per group), 3 the same on 32 rows (6 KiB).
    pub mform: u32,
    /// `gemm.half`: `1` (default) the GEMM forms whose operands are staged as
    /// f16 (`MmaForm::half`, `gemm.mform` 4, 7, 8) are eligible at `gemm.auto`
    /// when exact: W is Q4 with every scale in `[2^-14, 2^13)` or zero
    /// (checked at load: every `bf16(d·q)` is then an f16), and the kernel
    /// stages a step's X as bf16 and multiplies as bf16 when one of its values
    /// is not an f16 (the same products, f32 sums). `0`: timed and logged only.
    pub gemm_half: u32,
    /// `gemm.auto`: `1` each GEMM shape of the prefill timed at load on
    /// `gemm_mma_rows` and `gemm_mm2d` (dense: tiles of 64 or 32 rows × 64, 32
    /// or 16 columns; experts: `moe.tile` rows × 64, 32 or 16), at 64, 128,
    /// 256 rows and `prefill.block`, the fastest kept for each; a block runs
    /// the pipelines timed at the first of those ≥ its rows. Skipped when
    /// `gemm.mma` or `gemm.tn` is given. On the M3 Ultra neither kernel wins everywhere by much the same margin
    /// (`matmul2d` 64 × 32 pp512 ×1,13 to ×1,34 over the six Qwen).
    pub auto: u32,
    /// `gemm.mma` or `gemm.tn` was given: the GEMMs as they say, nothing timed.
    pub gemm_set: bool,
    /// `gemm.mfirst`: consecutive groups walk the row tiles of a column tile.
    pub m_first: u32,
    /// `moe.grouped`: `1` the experts of a block by GEMMs over their routed
    /// rows sorted by expert (`moe_sort`, `gemm_mma_rows`); `0` a matvec per
    /// (row, slot).
    pub moe_grouped: u32,
    /// `moe.tile`: rows of an expert tile (16, 32 or 64). 16 by default: a
    /// block of 512 rows routes ~16 rows to each of the 35B's 256 experts
    /// (M3 Ultra, one run: pp512 16 / 32 / 64 rows 1 879 / 1 685 /
    /// 1 273 tok/s).
    pub moe_tile: u32,
    /// `moe.tile` was given: with `gemm.auto`, the experts' tiles keep it
    /// instead of 16, 32 or 64 rows timed for each bucket of rows.
    pub moe_tile_set: bool,
    /// `q4.fuse` (Q4 weights, given = nothing timed): `0` each Q4 matrix
    /// decoded to bf16 (`dequant_q4`) before the GEMM `gemm.mma` names, `1`
    /// decoded as `gemm_mma_rows` stages it (`gemm_mma_q4`, `gemm.mma=1`).
    /// With `gemm.auto` both are timed.
    pub q4_fuse: u32,
    /// `mv.auto`: `1` the rows per simdgroup of each projection of the decode
    /// step (`matvec`, `matvec_q4`) timed at load, 1, 2, 4 or 8, the fastest
    /// kept; `0` the engine's `--rows` for all.
    pub mv_auto: u32,
    /// `mv.masks` (`mv.auto=0`, Q4): `1` the decode's projections by `matvec_q4s`.
    pub mv_masks: u32,
    /// `dec.rec`: the decode step's delta rule. `0` one group per value head,
    /// the state read and written twice (`dn_recurrence`); `1` the block's
    /// `dn_recurrence_rows` on one row, `dec.cols` columns per group,
    /// `dec.slices` invocations per column, the state in registers (the
    /// default: M3 Ultra, tg128, 0.8B 343 → 473 tok/s, 27B 38.4 → 42.1).
    pub dec_rec: u32,
    /// `mv.split` (`mv.auto=0`, Q4): simdgroups sharing a row's blocks
    /// (`matvec_q4_split`), 1, 2 or 4; `mv.splitauto` (`mv.auto=1`): `1` the
    /// splits 2 and 4 timed with the other forms.
    pub mv_split: u32,
    /// `dec.sdpa`: the decode step's attention. `0` one group per head
    /// (`sdpa_decode`); `c` > 0 the keys split in `c` runs per head
    /// (`sdpa_decode_part`, `sdpa_decode_merge`), at most 64; 32 by default
    /// (M3 Ultra, 0.8B tg128 after 4096 positions: 82 → 485 tok/s).
    pub dec_sdpa: u32,
    /// `dec.swiglu`: `1` the decode step's SwiGLU done by `down` as it loads
    /// its input (`XSrc::Swiglu`, the same bits, one dispatch less a layer).
    pub dec_swiglu: u32,
    /// `dec.prep`: `1` each Q4 projection's input prepared once for
    /// llama.cpp's masks (`q4_prep_x`: `x·2^-s` in f32 and each block's `Σx`),
    /// the matvec no longer scaling and summing it per row group (the same bits).
    pub dec_prep: u32,
    /// `dec.conc`: `1` command buffers encoded concurrently with a barrier
    /// before every dispatch (the serial order); `dec.pf=KB` (implies it):
    /// the cache warmers of [`QwenEngine::with_prefetch`], 0 = none.
    pub dec_conc: u32,
    pub dec_pf: u32,
    /// `dec.pfk=KB`: the small kernels before some projections (the SwiGLU
    /// before `down`, …) read that many KiB of its weights in extra groups
    /// of their own dispatch, warming the system cache (0 = none).
    pub dec_pfk: u32,
    /// `dec.auto`: `1` the decode step's forms timed at load
    /// ([`QwenEngine::tune_step`]); `0` the defaults or what `--tune` gives.
    pub dec_auto: u32,
    /// `dec.gnorm`: `1` the DeltaNet's gated norm done by its Q4 `out`
    /// projection in the same dispatch (`dn_gnorm_out_q4`, the same bits);
    /// `2` the norm's groups and the projection's in one dispatch, the latter
    /// waiting on a counter (`dn_gnorm_out_sync`, given only).
    pub dec_gnorm: u32,
    /// `dec.pfdn=KB`: the small kernels before the DeltaNet's `out` (conv,
    /// gates, gated norm) and before the attention's `o` (rope, the two
    /// attention passes) each read that many KiB of its weights in extra
    /// groups, part after part from the start (0 = none).
    pub dec_pfdn: u32,
    /// `dec.fnorm`: `1` each layer's two RMSNorms done by the projection that
    /// follows (`XSrc::Norm`: every group computes the norm into threadgroup
    /// memory first, the same bits), two dispatches less a layer (the head's
    /// norm and a MoE layer's post norm stay).
    pub dec_fnorm: u32,
    /// `dec.fswi`: `1` the SwiGLU done in the `gate_up` projection's epilogue
    /// (`Out::Swiglu`: a simdgroup's rows are gate rows and their up rows,
    /// the same bits), one dispatch less a layer; nothing with `dec.swiglu=1`.
    pub dec_fswi: u32,
    /// `pf.fswi`: the prefill's SwiGLU in the `[gate | up]` GEMM (`gemm_nax_swiglu`): `0` apart
    /// (`swiglu_rows`), `1` fused, `2` (default) timed at load against the GEMM and `swiglu_rows` with `gemm.auto`.
    pub pf_fswi: u32,
    /// `dec.fdn`: the DeltaNet's small kernels. `1` conv, L2 norms, gates and
    /// recurrence in one dispatch (`dn_step` on `dec.cols` × `dec.slices`),
    /// the gated norm apart; `2` the gated norm in it too (a group a head);
    /// the conv state shifted by extra groups of the `out` projection
    /// (`Extra::ConvShift`). `dec.gnorm` is ignored with it.
    pub dec_fdn: u32,
    /// `dec.frope`: `1` the attention's head norms, RoPE and K/V append done
    /// by the first pass of the attention (`sdpa_decode_part_rope`, `dec.sdpa`
    /// > 0, heads of 256), one dispatch less an attention layer.
    pub dec_frope: u32,
    /// The keys `--tune` gave.
    pub given: Vec<String>,
    /// `mv.cold` (`mv.auto=1`): `1` each form timed over the projections of
    /// the shape, layer after layer (from memory, as in the step); `0` on one
    /// matrix again and again (from the cache once it fits there).
    pub mv_cold: u32,
    pub mv_split_auto: u32,
    /// `mv.persist` (`mv.auto=0`): persistent groups of the matvecs (0: one
    /// group per slot); `mv.persists` (`mv.auto=1`): the counts timed besides
    /// 0, `/`-separated (default none).
    pub mv_persist: u32,
    pub mv_persists: Vec<u32>,
    pub dec_cols: u32,
    pub dec_slices: u32,
    /// `dn.sscale`, `dn.oscale`, `dn.tscale`: the DeltaNet's recurrent state,
    /// the recurrence's output and dnmma's chunk operators stored times `2^e`
    /// (option c of the f16 numerical contract, `msl::set_dn_scale`);
    /// 0 = unscaled.
    pub dn_scale: [u32; 3],
}

impl Default for Tune {
    /// herbert-vk-bf16's defaults off NVIDIA, tiles of 16 × 16 for the
    /// attention (16 × 32 needs more than Metal's 32 KiB of threadgroup
    /// memory for a head of 256), and what the M3 Ultra measured at pp512
    /// (0.8B / 4B / 9B, tok/s): blocks of 512 rows (vs 256, SIMT GEMM:
    /// 4 341 → 5 291, 838 → 881, 449 → 465), the GEMMs on the matrix units
    /// walking the row tiles first (SIMT 5 284 / 881 / 465 → 9 523 / 1 856 /
    /// 1 027); the recurrence by chunks on the matrix units, chosen
    /// over the fastest by a hair (0.8B / 9B / 27B: `rec.chunk=3`
    /// 9 519 / 1 028 / 298, `4` 9 336 / 1 024 / 299, `1` 5 485 / 870 / 258),
    /// `rec.chunk=3` without the `bf16` feature.
    fn default() -> Tune {
        Tune { block: 2048, block_set: false, rec: if crate::msl::ACT_BF16 { 4 } else { 3 }, rec_cols: 16, rec_slices: 16, rec_vec: 0, rec_pair: 0, rec_stage: 0, conv_cols: 32, sdpa: 1, sdpa_bq: 16, sdpa_bk: 16, sdpa_mq: 0, sdpa_mk: 32, sdpa_qreg: 0, sdpa_kt: 0, sdpa_msg: 8, sdpa_auto: 1, sdpa_set: false, auto_heat: 0, simt: 1, mma: 1, tn: 64, mform: 0, gemm_half: 1, auto: 1, gemm_set: false, m_first: 1, moe_grouped: 1, moe_tile: 16, moe_tile_set: false, q4_fuse: 0, mv_auto: 1, mv_masks: 0, dec_rec: 1, dec_sdpa: 32, dec_swiglu: 0, dec_prep: 0, dec_conc: 0, dec_pf: 0, dec_pfk: 0, dec_auto: 1, dec_gnorm: 0, dec_pfdn: 0, dec_fnorm: 0, dec_fswi: 0, pf_fswi: 2, dec_fdn: 0, dec_frope: 0, given: Vec::new(), mv_cold: 1, mv_split: 1, mv_split_auto: 1, mv_persist: 0, mv_persists: Vec::new(), dec_cols: 16, dec_slices: 16, dn_scale: [0; 3] }
    }
}

impl Tune {
    /// `key=value` settings, comma separated.
    /// `sdpa.mq`, or its default for the attention kernel (16 for `sdpa_mma`, 64 for `sdpa_mm2d` and `sdpa_nax`).
    pub fn mq(&self) -> u32 {
        if self.sdpa_mq != 0 {
            self.sdpa_mq
        } else if self.sdpa >= 4 {
            64
        } else {
            16
        }
    }

    pub fn apply(&mut self, list: &str) -> std::result::Result<(), String> {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (key, value) = kv.split_once('=').ok_or_else(|| format!("--tune {kv}: key=value"))?;
            self.given.push(key.to_string());
            if key == "mv.persists" {
                self.mv_persists = value.split('/').filter(|v| !v.is_empty()).map(|v| v.parse().map_err(|e| format!("--tune {kv}: {e}"))).collect::<std::result::Result<_, _>>()?;
                continue;
            }
            let n: u32 = value.parse().map_err(|e| format!("--tune {kv}: {e}"))?;
            match key {
                "prefill.block" => (self.block, self.block_set) = (n.max(1), true),
                "rec.chunk" => self.rec = n,
                "rec.pcols" => self.rec_cols = n,
                "rec.pslices" => self.rec_slices = n,
                "rec.vec" => self.rec_vec = n,
                "rec.pair" => self.rec_pair = n,
                "rec.stage" => self.rec_stage = n,
                "conv.cols" => self.conv_cols = n,
                "sdpa.prefill" => (self.sdpa, self.sdpa_set) = (n, true),
                "sdpa.bq" => self.sdpa_bq = n,
                "sdpa.bk" => self.sdpa_bk = n,
                "sdpa.mq" => self.sdpa_mq = n,
                "sdpa.mk" => self.sdpa_mk = n,
                "sdpa.qreg" => self.sdpa_qreg = n,
                "sdpa.kt" => self.sdpa_kt = n,
                "sdpa.msg" => self.sdpa_msg = n,
                "sdpa.auto" => self.sdpa_auto = n,
                "auto.heat" => self.auto_heat = n,
                "gemm.simt" => self.simt = n,
                "gemm.mma" => (self.mma, self.gemm_set) = (n, true),
                "gemm.tn" => (self.tn, self.gemm_set) = (n, true),
                "gemm.mform" => (self.mform, self.gemm_set) = (n, true),
                "gemm.half" => self.gemm_half = n,
                "gemm.auto" => self.auto = n,
                "gemm.mfirst" => self.m_first = n,
                "moe.grouped" => self.moe_grouped = n,
                "moe.tile" => (self.moe_tile, self.moe_tile_set) = (n, true),
                "q4.fuse" => (self.q4_fuse, self.gemm_set) = (n, true),
                "mv.auto" => self.mv_auto = n,
                "mv.masks" => (self.mv_masks, self.mv_auto) = (n, 0),
                "dec.rec" => self.dec_rec = n,
                "mv.split" => (self.mv_split, self.mv_auto) = (n, 0),
                "mv.persist" => (self.mv_persist, self.mv_auto) = (n, 0),
                "mv.splitauto" => self.mv_split_auto = n,
                "dec.sdpa" => self.dec_sdpa = n.min(64),
                "mv.cold" => self.mv_cold = n,
                "dec.swiglu" => self.dec_swiglu = n,
                "dec.prep" => self.dec_prep = n,
                "dec.conc" => self.dec_conc = n,
                "dec.pf" => self.dec_pf = n,
                "dec.pfk" => self.dec_pfk = n,
                "dec.auto" => self.dec_auto = n,
                "dec.gnorm" => self.dec_gnorm = n,
                "dec.pfdn" => self.dec_pfdn = n,
                "dec.fnorm" => self.dec_fnorm = n,
                "dec.fswi" => self.dec_fswi = n,
                "pf.fswi" => self.pf_fswi = n,
                "dec.fdn" => self.dec_fdn = n,
                "dec.frope" => self.dec_frope = n,
                "dec.cols" => self.dec_cols = n,
                "dec.slices" => self.dec_slices = n,
                "dn.sscale" => self.dn_scale[0] = n,
                "dn.oscale" => self.dn_scale[1] = n,
                "dn.tscale" => self.dn_scale[2] = n,
                _ => return Err(format!("--tune {key}: unknown")),
            }
        }
        if self.q4_fuse > 1 || (self.q4_fuse == 1 && self.mma != 1) {
            return Err(format!("--tune q4.fuse={}: 0, or 1 with gemm.mma=1", self.q4_fuse));
        }
        if self.mma > 3 || !([16, 32, 64].contains(&self.tn) || (self.mma == 3 && self.tn == 128)) || self.mform > 9 {
            return Err(format!("--tune gemm.mma={} gemm.tn={} gemm.mform={}: gemm.mma 0..2, gemm.tn 16, 32 or 64, gemm.mform 0..9", self.mma, self.tn, self.mform));
        }
        if ![16, 32, 64].contains(&self.moe_tile) {
            return Err(format!("--tune moe.tile={}: 16, 32 or 64", self.moe_tile));
        }
        if self.rec > 4 || self.sdpa > 7 || !(1..=2).contains(&self.simt) {
            return Err(format!("--tune: rec.chunk 0..4, sdpa.prefill 0..7, gemm.simt 1..2 ({self:?})"));
        }
        if self.sdpa >= 2 && (!self.sdpa_mq.is_multiple_of(8) || ![8, 16, 32, 64].contains(&self.sdpa_mk) || !self.block.is_multiple_of(self.mq())) {
            return Err(format!("--tune sdpa.prefill=2|3|4: sdpa.mq a multiple of 8 dividing prefill.block, sdpa.mk 8, 16, 32 or 64 ({self:?})"));
        }
        if self.rec == 4 && (!crate::msl::ACT_BF16 || (self.block > 1 && !self.block.is_multiple_of(super::dnmma::C))) {
            return Err(format!("--tune rec.chunk=4: the bf16 feature and prefill.block a multiple of {}", super::dnmma::C));
        }
        if self.dn_scale.iter().any(|&e| e > 15) || (!crate::msl::ACT_BF16 && self.dn_scale != [0; 3]) {
            return Err(format!("--tune dn.sscale/oscale/tscale {:?}: 0..15, and the bf16 feature (16-bit states)", self.dn_scale));
        }
        // `dn_recurrence_st` still reads the DeltaNet's activations as f32.
        if crate::msl::ACT_BF16 && self.rec_stage > 0 {
            return Err("--tune rec.stage: not available with the bf16 feature".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "prefill.block={} rec.chunk={} rec.pcols={} rec.pslices={} rec.vec={} rec.pair={} rec.stage={} conv.cols={} sdpa.prefill={} sdpa.bq={} sdpa.bk={} sdpa.mq={} sdpa.mk={} sdpa.qreg={} sdpa.kt={} sdpa.msg={} sdpa.auto={} auto.heat={} gemm.simt={} gemm.auto={} gemm.mma={} gemm.tn={} gemm.mform={} gemm.half={} gemm.mfirst={} moe.grouped={} moe.tile={} q4.fuse={} mv.auto={} mv.masks={} mv.split={} mv.splitauto={} mv.persist={} mv.persists={:?} dec.sdpa={} mv.cold={} dec.swiglu={} dec.prep={} dec.conc={} dec.pf={} dec.pfk={} dec.auto={} dec.gnorm={} dec.pfdn={} dec.fnorm={} dec.fswi={} pf.fswi={} dec.fdn={} dec.frope={} dec.rec={} dec.cols={} dec.slices={} dn.sscale={} dn.oscale={} dn.tscale={}",
            self.block, self.rec, self.rec_cols, self.rec_slices, self.rec_vec, self.rec_pair, self.rec_stage, self.conv_cols, self.sdpa, self.sdpa_bq, self.sdpa_bk, self.sdpa_mq, self.sdpa_mk, self.sdpa_qreg, self.sdpa_kt, self.sdpa_msg, if self.sdpa_auto != 0 && !self.sdpa_set { 1 } else { 0 }, self.auto_heat, self.simt, if self.auto != 0 && !self.gemm_set { 1 } else { 0 }, self.mma, self.tn, self.mform, self.gemm_half, self.m_first, self.moe_grouped, self.moe_tile, self.q4_fuse, self.mv_auto, self.mv_masks, self.mv_split, self.mv_split_auto, self.mv_persist, self.mv_persists, self.dec_sdpa, self.mv_cold, self.dec_swiglu, self.dec_prep, self.dec_conc, self.dec_pf, self.dec_pfk, self.dec_auto, self.dec_gnorm, self.dec_pfdn, self.dec_fnorm, self.dec_fswi, self.pf_fswi, self.dec_fdn, self.dec_frope, self.dec_rec, self.dec_cols, self.dec_slices, self.dn_scale[0], self.dn_scale[1], self.dn_scale[2]
        )
    }
}

/// The output of the DeltaNet projection: the DeltaNet's activations.
const ACT_OUT: Out = if ACT_BF16 { Out::Bf16 } else { Out::F32 };

/// Whether every `bf16(d·q)` of a Q4 matrix is an f16: each f16 scale `d` is
/// zero or a normal in `[2^-14, 2^13)` (exponent field 1..=27), so that `d·q`
/// for `|q| ≤ 8` stays in `[2^-14, 2^16)`, where a bf16 is an f16. Read from
/// the shared buffer on the CPU (`w.scales`, `(len − scales) / 2` f16).
fn scales_half_exact(w: Wt) -> bool {
    let Some(so) = w.scales else { return false };
    // SAFETY: a shared buffer of `len` bytes, the scales from `so` on, written at load; the GPU does not write them.
    let scales = unsafe { std::slice::from_raw_parts(w.buf.ptr.add(so as usize) as *const u16, ((w.buf.len - so) / 2) as usize) };
    scales.iter().all(|&b| {
        let e = (b >> 10) & 0x1f;
        (b & 0x7fff) == 0 || (1..=27).contains(&e)
    })
}

/// The persistent group counts `dec.auto` offers `mv.auto` (one axis).
const MV_PERSISTS: [u32; 4] = [640, 1280, 2560, 5120];

/// A form of the decode step's matvec (`mv.auto`): rows per simdgroup, Q4
/// by llama.cpp's masks, simdgroups sharing a row, persistent groups (0: one
/// group per slot).
#[derive(Clone, Copy, Debug)]
struct MvForm {
    rows: u32,
    masks: bool,
    split: u32,
    persist: u32,
}

/// A projection's weights `[n][k]`: the checkpoint's bf16, or Q4_0 as
/// `q4::quantize` lays it out, its scales at `scales` in the same buffer.
#[derive(Clone, Copy)]
struct Wt {
    buf: Buffer,
    scales: Option<u64>,
    n: u32,
    k: u32,
}

impl Wt {
    fn bf16(buf: Buffer, n: u32, k: u32) -> Wt {
        Wt { buf, scales: None, n, k }
    }

    /// The bindings of a projection of one row (`matvec` or `matvec_q4`).
    fn mv_ranges(self, x: Range, y: Range) -> Vec<Range> {
        self.bind(&[], &[x, y])
    }

    /// `before`, the weights, `after`, and the scales of a Q4 matrix last.
    fn bind(self, before: &[Range], after: &[Range]) -> Vec<Range> {
        let mut v = before.to_vec();
        match self.scales {
            None => {
                v.push(r(self.buf));
                v.extend_from_slice(after);
            }
            Some(so) => {
                v.push(Range::part(self.buf, 0, so));
                v.extend_from_slice(after);
                v.push(Range::part(self.buf, so, self.buf.len - so));
            }
        }
        v
    }
}

enum Mixer {
    Attention { qkv: Wt, q_norm: Buffer, k_norm: Buffer, o: Wt, k_cache: Buffer, v_cache: Buffer },
    Delta { proj: Wt, conv_w: Buffer, a_log: Buffer, dt_bias: Buffer, norm: Buffer, out: Wt, conv_state: Buffer, rec_state: Buffer },
}

/// The sparse MLP of a `qwen3_5_moe` layer: the router, the experts'
/// `[gate | up]` (`[experts·2·inter][hidden]`) and `down`
/// (`[experts·hidden][inter]`), the shared expert's gate.
#[derive(Clone, Copy)]
struct MoeLayer {
    router: Wt,
    gate_up: Wt,
    down: Wt,
    shared_gate: Buffer,
}

struct Layer {
    input_norm: Buffer,
    post_norm: Buffer,
    /// The dense MLP, or the shared expert of a sparse one.
    gate_up: Wt,
    down: Wt,
    mixer: Mixer,
    moe: Option<MoeLayer>,
}

/// The buffers a step (row 0) or a prefill block (`prefill.block` rows) works in.
#[derive(Clone, Copy)]
struct Acts {
    state: Buffer,
    /// The parts after the first of a block split (`HERBERT_METAL_PFSPLIT`, up to 4): their rows and position.
    state2: [Buffer; 3],
    /// The tokens of a prefill block.
    tokens: Buffer,
    /// `rec.chunk=2`: the first pass's operators of every (chunk, value head).
    dn_prep: Buffer,
    /// `rec.chunk=4`: the chunk operators (bf16) and `Γ_C` (f32) of every (chunk, value head).
    tc_mat: Buffer,
    tc_vec: Buffer,
    x: Buffer,
    xn: Buffer,
    qkv: Buffer,
    scores: Buffer,
    attn_out: Buffer,
    gu: Buffer,
    h: Buffer,
    proj: Buffer,
    gates: Buffer,
    dn_out: Buffer,
    dn_y: Buffer,
    logits: Buffer,
    /// [`Options::logits_f32`]: the head's sums in f32.
    logits32: Buffer,
    arg_partial: Buffer,
    arg_out: Buffer,
    /// `dec.prep`: a projection's input prepared for the masks (`q4_prep_x`).
    xp: Buffer,
    /// `dec.gnorm=2`: the counters of `dn_gnorm_out_sync`, 16 bytes a layer, zero between dispatches.
    ctr: Buffer,
    // `qwen3_5_moe`: router scores, route, the experts' rows, their sum, the shared expert's output.
    scores_e: Buffer,
    route: Buffer,
    /// `moe.grouped=1`: the routed (row, slot) pairs sorted by expert, the table of expert tiles.
    moe_list: Buffer,
    moe_tiles: Buffer,
    egu: Buffer,
    ehb: Buffer,
    ey: Buffer,
    esum: Buffer,
    sout: Buffer,
    /// The token rows of the sorted pairs, for a `matmul2d` `[gate | up]` (`moe_gather`).
    xs: Buffer,
    /// Q4: the bf16 matrix of the next prefill GEMM, decoded from its Q4_0 (`dequant_q4`).
    wdq: Buffer,
}

/// A buffer holding the 16-bit tensors `parts` (name, elements) one after the
/// other, read from the shards without the page cache when they are bf16,
/// copied from the mapping otherwise; in the engine's format (bf16, or f16
/// with the `f16` feature: converted in place, nearest even).
fn upload(gpu: &mut Gpu, ck: &Checkpoint, parts: &[(&str, usize)]) -> Result<Buffer> {
    let total: usize = parts.iter().map(|p| p.1 * 2).sum();
    let b = gpu.alloc(total as u64)?;
    // SAFETY: a fresh shared buffer of at least `total` bytes; the GPU does not use it yet.
    let dst = unsafe { std::slice::from_raw_parts_mut(b.ptr, total) };
    if ck.read_bf16_into(parts, dst)? {
        bf16_bytes_to_h16(dst);
    } else {
        let mut off = 0;
        for &(key, len) in parts {
            let v = ck.h16(key, len)?;
            dst[off..off + len * 2].copy_from_slice(as_bytes(&v));
            off += len * 2;
        }
    }
    Ok(b)
}

/// The projections as they are loaded: bf16, or Q4_0 from the cache of
/// quantized weights, or quantized now.
struct Loader {
    q4: bool,
    /// Sums of squares of the weights quantized here and of their error.
    ss: (f64, f64),
    /// Every Q4 projection by its key (its parts' names joined by `+`), for the cache.
    quantized: Vec<(String, Wt)>,
    from_cache: usize,
}

impl Loader {
    /// The projection `[n][k]` made of the bf16 tensors `parts`.
    fn weight(&mut self, gpu: &mut Gpu, ck: &Checkpoint, parts: &[(&str, usize)], n: usize, k: usize) -> Result<Wt> {
        let total: usize = parts.iter().map(|p| p.1).sum();
        if total != n * k {
            return Err(HerbertError::ModelLoad(format!("{}: {total} elements, {n}x{k} expected", parts[0].0)));
        }
        if !self.q4 {
            return Ok(Wt::bf16(upload(gpu, ck, parts)?, n as u32, k as u32));
        }
        let key = parts.iter().map(|p| p.0).collect::<Vec<_>>().join("+");
        let w = if let Some((bytes, scales)) = ck.q4(&key, n, k)? {
            let b = gpu.alloc(bytes as u64)?;
            // SAFETY: a fresh shared buffer of `bytes` bytes; the GPU does not use it yet.
            ck.read_q4_into(&key, unsafe { std::slice::from_raw_parts_mut(b.ptr, bytes) })?;
            self.from_cache += 1;
            Wt { buf: b, scales: Some(scales), n: n as u32, k: k as u32 }
        } else {
            let mut raw = vec![0u8; 2 * total];
            if !ck.read_bf16_into(parts, &mut raw)? {
                let mut off = 0;
                for &(key, len) in parts {
                    let v = ck.bf16(key, len)?;
                    raw[off..off + len * 2].copy_from_slice(as_bytes(&v));
                    off += len * 2;
                }
            }
            let q = super::q4::quantize(&raw, n, k);
            drop(raw);
            (self.ss.0, self.ss.1) = (self.ss.0 + q.ss_w, self.ss.1 + q.ss_e);
            Wt { buf: gpu.alloc_with(&q.bytes)?, scales: Some(q.scales), n: n as u32, k: k as u32 }
        };
        self.quantized.push((key, w));
        Ok(w)
    }
}

fn r(b: Buffer) -> Range {
    Range::whole(b)
}

/// The pipelines of a prefill block.
#[derive(Clone)]
struct PrefillPipes {
    embed: usize,
    norm: usize,
    gate_up: usize,
    /// `None`: the SwiGLU done by `gate_up` (`pf.fswi`), which writes `h`.
    swiglu: Option<usize>,
    down: usize,
    qkv: usize,
    rope: usize,
    sdpa: usize,
    /// The attention kernel's `sdpa.prefill` and `sdpa.mq` (its groups): `sdpa.auto` picks per bucket.
    sdpa_kind: (u32, u32),
    o: usize,
    /// proj, conv, l2 and gates, recurrence, gated norm, out.
    delta: Option<[usize; 6]>,
    chunk_prep: Option<usize>,
    tc_prep: Option<usize>,
    /// `qwen3_5_moe`: the shared expert's down (bf16 out), the router, the
    /// route, the experts' `[gate | up]`, SwiGLU, the experts' down, the
    /// combination, the block's end.
    moe: Option<[usize; 8]>,
    /// `moe.grouped=1`: the sort, the experts' `[gate | up]` and `down` over
    /// the sorted rows; the gather of `X`'s rows ahead of a `matmul2d` `[gate | up]`.
    moe_grouped: Option<([usize; 3], Option<usize>)>,
    /// The expert tiles the kernels are emitted for (those of the bucket's rows).
    moe_tiles: u32,
}

/// One dispatch of the recorded step.
#[derive(Clone)]
struct Op {
    pipe: usize,
    ranges: Vec<Range>,
    groups: u32,
}

pub struct QwenEngine {
    gpu: Gpu,
    /// The checkpoint, kept for [`QwenEngine::write_q4_cache`].
    ck: Checkpoint,
    /// The Q4 projections by key.
    quantized: Vec<(String, Wt)>,
    list: Vec<Pipeline>,
    by_key: HashMap<String, usize>,
    dump: Option<PathBuf>,
    c: Config,
    attn: AttnShape,
    delta: DeltaShape,
    sg: u32,
    rows: u32,
    embed: Buffer,
    lm_head: Wt,
    final_norm: Buffer,
    /// Q4: the `dequant_q4` pipeline of each projection shape `(k, n)` a prefill GEMM reads.
    dq: HashMap<(u32, u32), usize>,
    /// The GEMM pipelines that read a Q4 matrix themselves (`gemm_mma_q4`).
    q4_fused: std::collections::HashSet<usize>,
    cos: Buffer,
    sin: Buffer,
    layers: Vec<Layer>,
    a: Acts,
    /// The step, recorded once at load and encoded again for every token.
    ops: Vec<Op>,
    tune: Tune,
    /// The prefill's pipelines per bucket of rows, increasing: a block of `m`
    /// rows runs those of the first bucket ≥ `m` (`gemm.auto` times the GEMMs
    /// at the bucket's rows; one bucket, `prefill.block`, otherwise).
    pf: Vec<(u32, PrefillPipes)>,
    /// The final norm, the head, the two passes of the argmax.
    head_pipes: [usize; 4],
    /// The step's projections as `record` pushes them: the op's index, its weights.
    mv_marks: Vec<(usize, Wt)>,
    /// `dec.pf`: the cache warmers' pipelines, dispatched beside the op before them.
    pf_pipes: std::collections::HashSet<usize>,
    /// `dec.gnorm=1`: the form of `dn_gnorm_out_q4` timed (simdgroups, rows, split, groups).
    gnorm_pick: Option<(u32, u32, u32, u32)>,
    /// `dec.gnorm=2`: the rows × split of `dn_gnorm_out_sync`.
    gsync_pick: Option<(u32, u32)>,
    /// `dec.conc` or `dec.pf`: the step's command buffer encoded concurrently.
    step_conc: bool,
    /// The groups of the head's projection.
    head_groups: u32,
    /// `mv.auto`: the rows per simdgroup kept for each projection shape of the step.
    mv_pick: HashMap<String, MvForm>,
    capacity: usize,
    pos: usize,
    argmax: u32,
    logits_cpu: Vec<Bf16>,
    /// [`Options::logits_f32`]: the f32 logits of the last step.
    logits32_cpu: Vec<f32>,
    compile_seconds: f64,
    /// The `matmul2d` pipelines and their tile, rows × columns (their groups: `gemm_mm2d_groups`).
    gemm_tn: HashMap<usize, (u32, u32)>,
    /// `gemm.auto`: the pipeline kept for each GEMM shape, and the seconds spent timing.
    gemm_pick: HashMap<String, usize>,
    autotune_seconds: f64,
    /// When the GPU last ran a timed comparison (the heater of `auto.heat` runs when it was more than 200 ms ago).
    last_timed: Option<Instant>,
    /// Rows per prefill block: `prefill.block`, or the fastest timed at load.
    chunk: u32,
    /// `gemm.auto` with experts: the route buffer holds a block's route from the router, to time the experts on.
    real_route: bool,
    /// GPU seconds of the steps run, and their count.
    pub gpu_seconds: f64,
    /// GPU seconds of the prefill blocks.
    pub prefill_gpu_seconds: f64,
    pub steps: usize,
    pub load_seconds: f64,
    /// `HERBERT_METAL_PROFILE=1`: the prefill blocks' kernels, the steps'.
    prof: Option<(Profile, Profile)>,
    /// `HERBERT_METAL_STATS=FILE`: the range of every activation.
    stats: Option<Stats>,
}

impl QwenEngine {
    pub fn load(model_dir: &Path, o: &Options) -> Result<QwenEngine> {
        let started = Instant::now();
        let mut gpu = Gpu::open()?;
        eprintln!("[metal] {}", gpu.summary());
        let ck = Checkpoint::load(model_dir)?;
        let c = ck.config.clone();
        eprintln!(
            "[loader] {} — hidden {}, {} layers, {} heads / {} kv × {}, intermediate {}, vocab {}, rotary {}, linear k {}×{} v {}×{}, tied head {}",
            c.model_type,
            c.hidden_size,
            c.num_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.intermediate_size,
            c.vocab_size,
            c.rotary_ndims,
            c.linear_num_key_heads,
            c.linear_key_head_dim,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.tie_word_embeddings
        );
        if !c.attn_output_gate {
            return Err(HerbertError::Config("this engine serves the gated attention of Qwen3.5 (attn_output_gate)".into()));
        }
        let (h, inter, vocab) = (c.hidden_size, c.intermediate_size, c.vocab_size);
        let cap = o.capacity.min(c.max_position_embeddings);
        // `sdpa_mma` and `sdpa_mm2d` read the last tile of keys whole (sdpa.mk at most 64).
        let cap = cap.next_multiple_of(64);
        let attn = AttnShape::from_config(&c, cap);
        let delta = DeltaShape::from_config(&c);
        let sg = gpu.simd;

        // ── weights ──
        // Every projection and the head in Q4_0 quantized now (`q4`), or the
        // checkpoint's bf16; the embedding, the norms, the conv and the
        // router stay bf16.
        let q4 = o.q4;
        let t_q4 = Instant::now();
        let mut ld = Loader { q4, ss: (0.0, 0.0), quantized: Vec::new(), from_cache: 0 };
        let embed = upload(&mut gpu, &ck, &[("embed_tokens", vocab * h)])?;
        let head_key = if ck.has("lm_head") {
            "lm_head"
        } else if c.tie_word_embeddings {
            "embed_tokens"
        } else {
            return Err(HerbertError::ModelLoad("no lm_head and tie_word_embeddings is false".into()));
        };
        let lm_head = if !q4 && head_key == "embed_tokens" { Wt::bf16(embed, vocab as u32, h as u32) } else { ld.weight(&mut gpu, &ck, &[(head_key, vocab * h)], vocab, h)? };
        let final_norm = upload(&mut gpu, &ck, &[("norm", h)])?;
        let mut layers = Vec::with_capacity(c.num_layers);
        for l in 0..c.num_layers {
            let p = |f: &str| format!("{l}.{f}");
            let input_norm = upload(&mut gpu, &ck, &[(&p("input_layernorm.weight"), h)])?;
            let post_norm = upload(&mut gpu, &ck, &[(&p("post_attention_layernorm.weight"), h)])?;
            // `qwen3_5_moe`: the dense MLP is the shared expert.
            let mlp = if c.moe.is_some() { "mlp.shared_expert" } else { "mlp" };
            let gate_up = ld.weight(&mut gpu, &ck, &[(&p(&format!("{mlp}.gate_proj.weight")), inter * h), (&p(&format!("{mlp}.up_proj.weight")), inter * h)], 2 * inter, h)?;
            let down = ld.weight(&mut gpu, &ck, &[(&p(&format!("{mlp}.down_proj.weight")), h * inter)], h, inter)?;
            let moe = match c.moe {
                Some(m) => {
                    let (e, mi) = (m.num_experts, m.inter);
                    Some(MoeLayer {
                        router: Wt::bf16(upload(&mut gpu, &ck, &[(&p("mlp.gate.weight"), e * h)])?, e as u32, h as u32),
                        gate_up: ld.weight(&mut gpu, &ck, &[(&p("mlp.experts.gate_up_proj"), e * 2 * mi * h)], e * 2 * mi, h)?,
                        down: ld.weight(&mut gpu, &ck, &[(&p("mlp.experts.down_proj"), e * h * mi)], e * h, mi)?,
                        shared_gate: upload(&mut gpu, &ck, &[(&p("mlp.shared_expert_gate.weight"), h)])?,
                    })
                }
                None => None,
            };
            let mixer = if c.is_deltanet_layer(l) {
                let (value, nv) = (delta.value() as usize, delta.v_heads as usize);
                let conv_dim = delta.conv_dim() as usize;
                let proj = ld.weight(
                    &mut gpu,
                    &ck,
                    &[
                        (&p("linear_attn.in_proj_qkv.weight"), conv_dim * h),
                        (&p("linear_attn.in_proj_z.weight"), value * h),
                        (&p("linear_attn.in_proj_a.weight"), nv * h),
                        (&p("linear_attn.in_proj_b.weight"), nv * h),
                    ],
                    delta.proj_width() as usize,
                    h,
                )?;
                let conv_w = upload(&mut gpu, &ck, &[(&p("linear_attn.conv1d.weight"), conv_dim * delta.conv as usize)])?;
                let a_log = gpu.alloc_with(as_bytes(&ck.f32(&p("linear_attn.A_log"), nv)?))?;
                let dt_bias = gpu.alloc_with(as_bytes(&ck.f32(&p("linear_attn.dt_bias"), nv)?))?;
                let norm = upload(&mut gpu, &ck, &[(&p("linear_attn.norm.weight"), delta.v_dim as usize)])?;
                let out = ld.weight(&mut gpu, &ck, &[(&p("linear_attn.out_proj.weight"), h * value)], h, value)?;
                let conv_state = gpu.alloc((conv_dim * delta.conv as usize * ACT_BYTES) as u64)?;
                let rec_state = gpu.alloc((delta.state_len() as usize * ACT_BYTES) as u64)?;
                Mixer::Delta { proj, conv_w, a_log, dt_bias, norm, out, conv_state, rec_state }
            } else {
                let (qw, kvw) = (attn.q_width() as usize, attn.kv_width() as usize);
                let qkv = ld.weight(&mut gpu, &ck, &[(&p("self_attn.q_proj.weight"), qw * h), (&p("self_attn.k_proj.weight"), kvw * h), (&p("self_attn.v_proj.weight"), kvw * h)], qw + 2 * kvw, h)?;
                let q_norm = upload(&mut gpu, &ck, &[(&p("self_attn.q_norm.weight"), c.head_dim)])?;
                let k_norm = upload(&mut gpu, &ck, &[(&p("self_attn.k_norm.weight"), c.head_dim)])?;
                let o = ld.weight(&mut gpu, &ck, &[(&p("self_attn.o_proj.weight"), h * attn.out_width() as usize)], h, attn.out_width() as usize)?;
                let cache_bytes = (kvw * cap * 2) as u64;
                let k_cache = gpu.alloc(cache_bytes)?;
                let v_cache = gpu.alloc(cache_bytes)?;
                Mixer::Attention { qkv, q_norm, k_norm, o, k_cache, v_cache }
            };
            layers.push(Layer { input_norm, post_norm, gate_up, down, mixer, moe });
        }
        let weight_bytes = gpu.allocated;
        if q4 {
            eprintln!(
                "[q4] {} projections in Q4_0 ({} from the cache), loaded in {:.1} s{}",
                ld.quantized.len(),
                ld.from_cache,
                t_q4.elapsed().as_secs_f64(),
                if ld.ss.0 > 0.0 { format!("; quantized here: relative RMS error {:.4}", (ld.ss.1 / ld.ss.0).sqrt()) } else { String::new() }
            );
        }
        // The largest matrix a prefill GEMM reads, decoded to bf16 before it.
        let wdq_bytes = if q4 && o.tune.block > 1 {
            let mut big = 0u64;
            for l in &layers {
                let mut ws = vec![l.gate_up, l.down];
                match l.mixer {
                    Mixer::Attention { qkv, o, .. } => ws.extend([qkv, o]),
                    Mixer::Delta { proj, out, .. } => ws.extend([proj, out]),
                }
                big = ws.iter().filter(|w| w.scales.is_some()).map(|w| w.n as u64 * w.k as u64 * 2).fold(big, u64::max);
            }
            big
        } else {
            0
        };

        // RoPE tables of the reserved positions, f32 then rounded to 16 bits once.
        let half = c.rotary_ndims / 2;
        let inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / c.rope_theta.powf((2 * i) as f32 / c.rotary_ndims as f32)).collect();
        let mut cos = Vec::with_capacity(cap * half);
        let mut sin = Vec::with_capacity(cap * half);
        for pos in 0..cap {
            for &f in &inv_freq {
                let angle = pos as f32 * f;
                cos.push(f32_to_h16(angle.cos()));
                sin.push(f32_to_h16(angle.sin()));
            }
        }
        let cos = gpu.alloc_with(as_bytes(&cos))?;
        let sin = gpu.alloc_with(as_bytes(&sin))?;

        // ── activations, for a prefill block ──
        let mut tune = o.tune.clone();
        crate::msl::set_dn_scale(tune.dn_scale[0], tune.dn_scale[1], tune.dn_scale[2]);
        if tune.block_set && tune.block as usize > cap {
            return Err(HerbertError::Config(format!("--tune prefill.block={} exceeds the {cap} reserved positions", tune.block)));
        }
        if !tune.block_set {
            // The largest block of 8192, 4096 or 2048 rows whose activations take at most a sixteenth of the
            // working set (the prefill's blocks are then chosen by the prompt's length, `autotune_chunk`).
            let per_row = 4 + 4 * h + 2 * (attn.qkv_width() + attn.out_width()) as usize + 6 * inter
                + (delta.proj_width() as usize + 2 * delta.v_heads as usize + delta.value() as usize) * ACT_BYTES
                + 2 * delta.value() as usize
                + if tune.rec == 4 { delta.v_heads as usize * dnmma::mat_len(delta) as usize * 2 / dnmma::C as usize + 1 } else { 0 };
            let budget = gpu.working_set() as usize / 16;
            let fits = [8192u32, 4096].into_iter().find(|&b| b as usize * per_row <= budget).unwrap_or(2048);
            // Rows of 64 so that every bucket and the recurrence's chunks divide them.
            tune.block = fits.max(tune.block).min(cap as u32 / 64 * 64).max(1);
        }
        let auto = tune.auto != 0 && !tune.gemm_set;
        let rows = tune.block as usize;
        let mut act = |bytes: usize| gpu.alloc(bytes.max(16) as u64);
        let a = Acts {
            state: act(16)?,
            state2: [act(16)?, act(16)?, act(16)?],
            tokens: act(rows * 4)?,
            dn_prep: act(if rows > 1 && tune.rec == 2 { rows.div_ceil(16) * delta.v_heads as usize * pf::CHUNK_PREP as usize * 4 } else { 16 })?,
            tc_mat: act(if rows > 1 && tune.rec == 4 { rows.div_ceil(dnmma::C as usize) * delta.v_heads as usize * dnmma::mat_len(delta) as usize * 2 } else { 16 })?,
            tc_vec: act(if rows > 1 && tune.rec == 4 { rows.div_ceil(dnmma::C as usize) * delta.v_heads as usize * 4 } else { 16 })?,
            x: act(rows * h * 2)?,
            xn: act(rows * h * 2)?,
            qkv: act(rows * attn.qkv_width() as usize * 2)?,
            // One row for the decode attention; every row of a block only for
            // the row-by-row prefill attention (`sdpa.prefill=0`).
            scores: act((if rows > 1 && tune.sdpa == 0 { rows } else { 1 } * attn.heads as usize * cap * 4).max(attn.heads as usize * tune.dec_sdpa as usize * (attn.head_dim as usize + 2) * 4))?,
            attn_out: act(rows * attn.out_width() as usize * 2)?,
            gu: act(rows * 2 * inter * 2)?,
            h: act(rows * inter * 2)?,
            proj: act(rows * delta.proj_width() as usize * ACT_BYTES)?,
            gates: act(rows * 2 * delta.v_heads as usize * ACT_BYTES)?,
            dn_out: act(rows * delta.value() as usize * ACT_BYTES)?,
            dn_y: act(rows * delta.value() as usize * 2)?,
            logits: act(vocab * 2)?,
            logits32: act(if o.logits_f32 { vocab * 4 } else { 16 })?,
            arg_partial: act(kn::argmax_groups(vocab as u32) as usize * 8)?,
            arg_out: act(16)?,
            ctr: act(c.num_layers * 16)?,
            xp: act(h.max(inter).max(delta.value() as usize).max(attn.out_width() as usize) / 32 * 33 * 4)?,
            scores_e: act(rows * c.moe.map_or(0, |m| m.num_experts) * 2)?,
            route: act(rows * c.moe.map_or(0, |m| 2 * m.top_k) * 4)?,
            moe_list: act(rows * c.moe.map_or(0, |m| m.top_k) * 4)?,
            moe_tiles: act(c.moe.map_or(0, |m| 1 + 3 * pf::moe_tiles(rows as u32, m.top_k as u32, m.num_experts as u32, if auto && !tune.moe_tile_set { 16 } else { tune.moe_tile }) as usize) * 4)?,
            egu: act(rows * c.moe.map_or(0, |m| m.top_k * 2 * m.inter) * 2)?,
            ehb: act(rows * c.moe.map_or(0, |m| m.top_k * m.inter) * 2)?,
            ey: act(rows * c.moe.map_or(0, |m| m.top_k * h) * 2)?,
            esum: act(rows * h * 2)?,
            sout: act(rows * h * 2)?,
            xs: act(if tune.moe_grouped != 0 { rows * c.moe.map_or(0, |m| m.top_k) * h * 2 } else { 16 })?,
            wdq: act(wdq_bytes as usize)?,
        };
        eprintln!(
            "[metal] weights {:.2} GiB, everything {:.2} GiB, capacity {cap}, DeltaNet activations {}",
            weight_bytes as f64 / (1u64 << 30) as f64,
            gpu.allocated as f64 / (1u64 << 30) as f64,
            if ACT_BF16 { "bf16" } else { "f32" }
        );
        let mut engine = QwenEngine {
            gpu,
            ck,
            quantized: ld.quantized,
            list: Vec::new(),
            by_key: HashMap::new(),
            dump: o.dump_msl.clone(),
            c: c.clone(),
            attn,
            delta,
            sg,
            rows: o.matvec_rows,
            embed,
            lm_head,
            final_norm,
            dq: HashMap::new(),
            q4_fused: Default::default(),
            cos,
            sin,
            layers,
            a,
            ops: Vec::new(),
            tune,
            pf: Vec::new(),
            head_pipes: [0; 4],
            mv_marks: Vec::new(),
            pf_pipes: std::collections::HashSet::new(),
            gnorm_pick: None,
            gsync_pick: None,
            step_conc: o.tune.dec_conc != 0 || o.tune.dec_pf > 0,
            head_groups: 0,
            mv_pick: HashMap::new(),
            capacity: cap,
            pos: 0,
            argmax: 0,
            logits_cpu: vec![0; vocab],
            logits32_cpu: vec![0.0; if o.logits_f32 { vocab } else { 0 }],
            compile_seconds: 0.0,
            gemm_tn: HashMap::new(),
            gemm_pick: HashMap::new(),
            autotune_seconds: 0.0,
            last_timed: None,
            chunk: 1,
            real_route: false,
            gpu_seconds: 0.0,
            prefill_gpu_seconds: 0.0,
            steps: 0,
            load_seconds: 0.0,
            prof: Profile::from_env().zip(Profile::from_env()),
            stats: None,
        };
        if let Some(mut st) = Stats::from_env(&mut engine.gpu)? {
            let a = &engine.a;
            for (b, name) in [
                (a.dn_prep, "dn_prep"),
                (a.tc_mat, "tc_mat"),
                (a.tc_vec, "tc_vec"),
                (a.x, "x"),
                (a.xn, "xn"),
                (a.qkv, "qkv"),
                (a.scores, "scores"),
                (a.attn_out, "attn_out"),
                (a.gu, "gu"),
                (a.h, "h"),
                (a.proj, "proj"),
                (a.gates, "gates"),
                (a.dn_out, "dn_out"),
                (a.dn_y, "dn_y"),
                (a.logits, "logits"),
                (a.scores_e, "scores_e"),
                (a.route, "route"),
                (a.egu, "egu"),
                (a.ehb, "ehb"),
                (a.ey, "ey"),
                (a.esum, "esum"),
                (a.sout, "sout"),
            ] {
                st.name(b, name);
            }
            for (i, l) in engine.layers.iter().enumerate() {
                match l.mixer {
                    Mixer::Attention { k_cache, v_cache, .. } => {
                        st.name(k_cache, format!("k_cache.{i}"));
                        st.name(v_cache, format!("v_cache.{i}"));
                    }
                    Mixer::Delta { conv_state, rec_state, .. } => {
                        st.name(conv_state, format!("conv_state.{i}"));
                        st.name(rec_state, format!("rec_state.{i}"));
                    }
                }
            }
            engine.stats = Some(st);
        }
        engine.reset()?;
        // Every kernel emitted and compiled now, the step recorded.
        engine.ops = engine.record()?;
        if engine.tune.dec_auto != 0 {
            engine.tune_step()?;
        }
        engine.chunk = engine.tune.block;
        if engine.tune.block > 1 {
            let block = engine.tune.block;
            let mut buckets: Vec<u32> = if auto { (6..14).map(|b| 1u32 << b).filter(|&b| b < block).collect() } else { Vec::new() };
            buckets.push(block);
            if auto && engine.c.moe.is_some() && engine.tune.moe_grouped != 0 {
                engine.route_probe()?;
            }
            for m in buckets {
                // Past 2048 rows the kernels of 2048 (they read m at run time): no GEMM timed on bigger blocks.
                let p = match engine.pf.iter().find(|(b, _)| *b == 2048) {
                    Some((_, p)) if m > 2048 => p.clone(),
                    _ => engine.prefill_pipes(m)?,
                };
                engine.pf.push((m, p));
            }
            if auto && !engine.tune.block_set {
                engine.autotune_chunk()?;
            }
        }
        eprintln!(
            "[metal] {} kernels compiled in {:.2} s, GEMMs and blocks timed in {:.2} s, {} dispatches per token, blocks of {} rows; {}",
            engine.list.len(),
            engine.compile_seconds,
            engine.autotune_seconds,
            engine.ops.len(),
            engine.chunk,
            engine.tune.summary()
        );
        engine.load_seconds = started.elapsed().as_secs_f64();
        Ok(engine)
    }

    pub fn config(&self) -> &Config {
        &self.c
    }

    /// Write the cache of quantized weights (`model::gemma::Q4_CACHE`'s
    /// format) to `path`: every Q4_0 projection as uploaded (`q4.<key>`, its
    /// `n,k,scales` in the metadata), then every tensor read that is not a
    /// part of one (the embedding stays). The number of bytes written.
    pub fn write_q4_cache(&self, path: &Path) -> Result<u64> {
        use serde_json::{json, Map, Value};
        use std::io::Write;
        if self.quantized.is_empty() {
            return Err(HerbertError::Backend("no Q4 projection to write (--bf16?)".into()));
        }
        let parts: std::collections::HashSet<String> = self.quantized.iter().flat_map(|q| q.0.split('+').map(str::to_string)).filter(|n| n != "embed_tokens").collect();
        let others = self.ck.others(&parts);
        let mut header = Map::new();
        let mut meta = Map::new();
        meta.insert("format".into(), Value::String(crate::model::gemma::Q4_FORMAT.into()));
        let mut off = 0u64;
        let mut blobs: Vec<&[u8]> = Vec::new();
        for (key, w) in &self.quantized {
            // SAFETY: a shared buffer of `len` bytes written at load; the GPU only reads it.
            let bytes = unsafe { std::slice::from_raw_parts(w.buf.ptr as *const u8, w.buf.len as usize) };
            let name = format!("q4.{key}");
            header.insert(name.clone(), json!({"dtype": "U8", "shape": [bytes.len()], "data_offsets": [off, off + bytes.len() as u64]}));
            meta.insert(name, Value::String(format!("{},{},{}", w.n, w.k, w.scales.expect("Q4"))));
            off += bytes.len() as u64;
            blobs.push(bytes);
        }
        for (name, dtype, bytes) in &others {
            let elems = bytes.len() / if *dtype == "F32" { 4 } else { 2 };
            header.insert(name.clone(), json!({"dtype": dtype, "shape": [elems], "data_offsets": [off, off + bytes.len() as u64]}));
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
        eprintln!("[q4cache] {} projections, {} other tensors, {:.2} GiB", self.quantized.len(), others.len(), (8 + h.len() as u64 + off) as f64 / (1u64 << 30) as f64);
        Ok(8 + h.len() as u64 + off)
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Back to an empty conversation: the DeltaNet states zeroed (the KV
    /// caches are overwritten position by position).
    pub fn reset(&mut self) -> Result<()> {
        for l in &self.layers {
            if let Mixer::Delta { conv_state, rec_state, .. } = l.mixer {
                for b in [conv_state, rec_state] {
                    // SAFETY: a shared buffer of `len` bytes; the GPU is idle.
                    unsafe { std::ptr::write_bytes(b.ptr, 0, b.len as usize) };
                }
            }
        }
        // SAFETY: a shared buffer of `len` bytes; the GPU is idle.
        unsafe { std::ptr::write_bytes(self.a.ctr.ptr, 0, self.a.ctr.len as usize) };
        self.pos = 0;
        Ok(())
    }

    /// The pipeline of an emitted kernel, built the first time its name is seen.
    fn get(&mut self, e: Emitted) -> Result<usize> {
        if let Some(&i) = self.by_key.get(&e.name) {
            return Ok(i);
        }
        let t = Instant::now();
        crate::msl::dump(self.dump.as_deref(), &e.name, &e.source);
        self.list.push(self.gpu.pipeline(&e)?);
        self.by_key.insert(e.name, self.list.len() - 1);
        self.compile_seconds += t.elapsed().as_secs_f64();
        Ok(self.list.len() - 1)
    }

    fn run(&mut self, ops: &mut Vec<Op>, e: Emitted, ranges: &[Range], groups: u32) -> Result<()> {
        let pipe = self.get(e)?;
        ops.push(Op { pipe, ranges: ranges.to_vec(), groups });
        Ok(())
    }

    /// The projections of the checkpoint of `w`'s shape and format, `w` first:
    /// the weights a matvec form is timed on, layer after layer as the step
    /// reads them, so that they come from memory and not from the cache.
    fn same_shape(&self, w: Wt) -> Vec<Wt> {
        let mut v = vec![w];
        let mut push = |o: Wt| {
            if o.n == w.n && o.k == w.k && o.scales.is_some() == w.scales.is_some() && o.buf.ptr != w.buf.ptr {
                v.push(o);
            }
        };
        for l in &self.layers {
            match l.mixer {
                Mixer::Attention { qkv, o, .. } => {
                    push(qkv);
                    push(o);
                }
                Mixer::Delta { proj, out, .. } => {
                    push(proj);
                    push(out);
                }
            }
            push(l.gate_up);
            push(l.down);
            if let Some(m) = l.moe {
                push(m.router);
            }
        }
        push(self.lm_head);
        v
    }

    /// `dec.gnorm=1`: the DeltaNet's gated norm and its Q4 `out` in one
    /// dispatch (`dn_gnorm_out_q4`), its form (simdgroups, rows, split,
    /// groups) timed at the first layer over every layer's `out`.
    fn gnorm_out(&mut self, ops: &mut Vec<Op>, out: Wt, norm: Buffer) -> Result<()> {
        let (delta, sg, h) = (self.delta, self.sg, self.c.hidden_size as u32);
        let a = self.a;
        let ranges = |w: Wt| w.bind(&[], &[r(a.dn_out), r(a.proj), r(norm), r(a.x)]);
        let f = match self.gnorm_pick {
            Some(f) => f,
            None => {
                let ws = self.same_shape(out);
                let mut forms = Vec::new();
                let mut cands = Vec::new();
                for nsg in [8u32, 16, 32] {
                    for (rows, split) in [(1u32, 4u32), (2, 4), (2, 2), (4, 1), (2, 8)] {
                        if !nsg.is_multiple_of(split) {
                            continue;
                        }
                        for persist in [80u32, 160, 320, 640] {
                            // A form this GPU cannot run (too many invocations a group) is left out.
                            let Ok(p) = self.get(qk::dn_gnorm_out_q4(delta, h, sg, nsg, rows, split, persist)) else {
                                continue;
                            };
                            forms.push((nsg, rows, split, persist));
                            let ops = ws.iter().map(|&w| Op { pipe: p, ranges: ranges(w), groups: persist }).collect();
                            cands.push((format!("s{nsg}r{rows}k{split}p{persist}"), p, ops));
                        }
                    }
                }
                let (best, _) = self.pick(&format!("gnorm+out {}x{h}", delta.value()), &cands)?;
                let f = forms[cands.iter().position(|c| c.1 == best).expect("candidate")];
                self.gnorm_pick = Some(f);
                f
            }
        };
        self.mv_marks.push((ops.len(), out));
        self.run(ops, qk::dn_gnorm_out_q4(delta, h, sg, f.0, f.1, f.2, f.3), &ranges(out), f.3)
    }

    /// `dec.gnorm=2`: [`qk::dn_gnorm_out_sync`], the norm's groups and the
    /// projection's in one dispatch synchronized by a counter of layer `l`'s
    /// own (`a.ctr`); its rows × split timed at the first layer.
    fn gnorm_out_sync(&mut self, ops: &mut Vec<Op>, out: Wt, norm: Buffer, l: usize) -> Result<()> {
        let (delta, sg, h) = (self.delta, self.sg, self.c.hidden_size as u32);
        let a = self.a;
        let ctr = |l: usize| Range::part(a.ctr, l as u64 * 16, 16);
        let ranges = |w: Wt, l: usize| w.bind(&[], &[r(a.dn_out), r(a.proj), r(norm), r(a.dn_y), r(a.x), ctr(l)]);
        let groups = |rows: u32, split: u32| delta.v_heads + kn::matvec_split_groups(h, sg, rows, split, 0);
        let f = match self.gsync_pick {
            Some(f) => f,
            None => {
                let ws = self.same_shape(out);
                let mut forms = Vec::new();
                let mut cands = Vec::new();
                for (rows, split) in [(1u32, 4u32), (2, 4), (4, 4), (2, 2), (4, 1), (2, 1)] {
                    let p = self.get(qk::dn_gnorm_out_sync(delta, h, sg, rows, split))?;
                    forms.push((rows, split));
                    let ops = ws.iter().enumerate().map(|(i, &w)| Op { pipe: p, ranges: ranges(w, i), groups: groups(rows, split) }).collect();
                    cands.push((format!("r{rows}k{split}"), p, ops));
                }
                let (best, _) = self.pick(&format!("gnorm+out sync {}x{h}", delta.value()), &cands)?;
                let f = forms[cands.iter().position(|c| c.1 == best).expect("candidate")];
                self.gsync_pick = Some(f);
                f
            }
        };
        self.mv_marks.push((ops.len(), out));
        self.run(ops, qk::dn_gnorm_out_sync(delta, h, sg, f.0, f.1), &ranges(out, l), groups(f.0, f.1))
    }

    /// `y = W·x` written as `out` says.
    fn mv(&mut self, ops: &mut Vec<Op>, w: Wt, x: Buffer, y: Buffer, k: u32, n: u32, out: Out) -> Result<()> {
        self.mv_x(ops, w, x, y, k, n, out, kn::XSrc::Plain)
    }

    /// [`Self::mv`] with `x` taken as `xsrc` says (the split kernels only).
    #[allow(clippy::too_many_arguments)]
    fn mv_x(
        &mut self,
        ops: &mut Vec<Op>,
        w: Wt,
        x: Buffer,
        y: Buffer,
        k: u32,
        n: u32,
        out: Out,
        xsrc: kn::XSrc,
    ) -> Result<()> {
        self.mv_full(
            ops,
            w,
            x,
            y,
            k,
            n,
            out,
            xsrc,
            None,
            kn::Extra::None,
            Vec::new(),
        )
    }

    /// [`Self::mv_x`] with, for `XSrc::Norm`, the norm's weights `norm_w`
    /// (`x` the residual stream), and `extra` groups with their bindings.
    #[allow(clippy::too_many_arguments)]
    fn mv_full(
        &mut self,
        ops: &mut Vec<Op>,
        w: Wt,
        x: Buffer,
        y: Buffer,
        k: u32,
        n: u32,
        out: Out,
        xsrc: kn::XSrc,
        norm_w: Option<Buffer>,
        extra: kn::Extra,
        extra_ranges: Vec<Range>,
    ) -> Result<()> {
        assert!(
            w.k == k && w.n == n,
            "projection {k}x{n} on weights {}x{}",
            w.k,
            w.n
        );
        assert!(
            matches!(xsrc, kn::XSrc::Norm { .. }) == norm_w.is_some(),
            "XSrc::Norm and its weights go together"
        );
        let sg = self.sg;
        // A form: rows per simdgroup, for Q4 llama.cpp's masks (`matvec_q4s`),
        // the simdgroups sharing a row's units (`matvec_q4_split`,
        // `matvec_split`) and the persistent groups.
        // `dec.prep=1` (Q4, K a multiple of 256): `x` prepared once by
        // `q4_prep_x`, then the masks' forms reading it.
        let xsrc = if xsrc == kn::XSrc::Plain
            && w.scales.is_some()
            && self.tune.dec_prep != 0
            && k.is_multiple_of(256)
        {
            kn::XSrc::Prepped
        } else {
            xsrc
        };
        let prepped = xsrc == kn::XSrc::Prepped;
        let xp = self.a.xp;
        let (x, mut after) = if prepped {
            self.run(ops, kn::q4_prep_x(k, sg), &[r(x), r(xp)], k / 256)?;
            (xp, vec![Range::part(xp, k as u64 * 4, k as u64 / 32 * 4)])
        } else {
            (x, Vec::new())
        };
        if let Some(nw) = norm_w {
            after.push(r(nw));
        }
        after.extend(extra_ranges);
        let mvr = move |w: Wt| {
            let mut v = w.mv_ranges(r(x), r(y));
            v.extend(after.iter().copied());
            v
        };
        let plain = xsrc == kn::XSrc::Plain;
        // The split kernels: a split or persistent form, or anything but a
        // plain `x`, a plain `y` and no extra groups.
        let special = !plain || out == Out::Swiglu || extra != kn::Extra::None;
        let emit = |f: MvForm| match (w.scales, f.split > 1 || f.persist > 0 || special) {
            (None, false) => kn::matvec(k, n, out, sg, f.rows),
            (None, true) => {
                kn::matvec_split(k, n, out, sg, f.rows, f.split, f.persist, xsrc, extra)
            }
            (Some(_), true) => kn::matvec_q4_split(
                k, n, out, sg, f.rows, f.masks, f.split, f.persist, xsrc, extra,
            ),
            (Some(_), false) if f.masks => kn::matvec_q4s(k, n, out, sg, f.rows),
            (Some(_), false) => kn::matvec_q4(k, n, out, sg, f.rows),
        };
        let groups = |f: MvForm| {
            if f.split > 1 || f.persist > 0 || special {
                kn::matvec_full_groups(n, out, sg, f.rows, f.split, f.persist, extra)
            } else {
                kn::matvec_groups(n, sg, f.rows)
            }
        };
        let ranges = mvr(w);
        let f = if self.tune.mv_auto == 0 {
            let rows = if out == Out::Swiglu {
                self.rows.max(2)
            } else {
                self.rows
            };
            MvForm {
                rows,
                masks: w.scales.is_some() && (self.tune.mv_masks != 0 || prepped),
                split: self.tune.mv_split,
                persist: self.tune.mv_persist,
            }
        } else {
            let key = format!(
                "{k}x{n} {out:?}{}{}{}{}",
                if w.scales.is_some() { " q4" } else { "" },
                match xsrc {
                    kn::XSrc::Plain => "",
                    kn::XSrc::Swiglu => " swiglu",
                    kn::XSrc::Prepped => " prepped",
                    kn::XSrc::Norm { .. } => " norm",
                },
                match extra {
                    kn::Extra::None => "",
                    kn::Extra::ConvShift { .. } => " convshift",
                },
                if self.tune.mv_persists.is_empty() {
                    String::new()
                } else {
                    format!(" p{:?}", self.tune.mv_persists)
                }
            );
            match self.mv_pick.get(&key) {
                Some(&f) => f,
                None => {
                    // Each candidate on the weights of every projection of the shape
                    // (`mv.cold=0`: on this one's alone, from the cache); `y` is
                    // scratch until the step runs.
                    let ws = if self.tune.mv_cold != 0 {
                        self.same_shape(w)
                    } else {
                        vec![w]
                    };
                    let mut forms = Vec::new();
                    let mut cands = Vec::new();
                    let splits: &[u32] = if self.tune.mv_split_auto != 0 {
                        &[1, 2, 4]
                    } else {
                        &[1]
                    };
                    let persists: Vec<u32> = std::iter::once(0)
                        .chain(self.tune.mv_persists.iter().copied())
                        .collect();
                    for masks in if prepped {
                        vec![true]
                    } else if w.scales.is_some() {
                        vec![false, true]
                    } else {
                        vec![false]
                    } {
                        for &split in splits {
                            for &persist in &persists {
                                for rows in [1, 2, 4, 8] {
                                    // The persistent groups on the masks only (they always win in Q4).
                                    if persist > 0 && (rows == 8 || (w.scales.is_some() && !masks))
                                    {
                                        continue;
                                    }
                                    // `Out::Swiglu`: a gate row and its up row at least.
                                    if out == Out::Swiglu && rows == 1 {
                                        continue;
                                    }
                                    let f = MvForm {
                                        rows,
                                        masks,
                                        split,
                                        persist,
                                    };
                                    let p = self.get(emit(f))?;
                                    forms.push(f);
                                    let tag = format!(
                                        "{}{}",
                                        if split > 1 {
                                            format!("k{split}")
                                        } else {
                                            String::new()
                                        },
                                        if persist > 0 {
                                            format!("p{persist}")
                                        } else {
                                            String::new()
                                        }
                                    );
                                    let ops = ws
                                        .iter()
                                        .map(|&w| Op {
                                            pipe: p,
                                            ranges: mvr(w),
                                            groups: groups(f),
                                        })
                                        .collect();
                                    cands.push((
                                        format!("{}r{rows}{tag}", if masks { "s" } else { "" }),
                                        p,
                                        ops,
                                    ));
                                }
                            }
                        }
                    }
                    let (best, _) = self.pick(&format!("matvec {key}"), &cands)?;
                    let f = forms[cands.iter().position(|c| c.1 == best).expect("candidate")];
                    self.mv_pick.insert(key, f);
                    f
                }
            }
        };
        self.mv_marks.push((ops.len(), w));
        self.run(ops, emit(f), &ranges, groups(f))
    }

    /// The dispatches of a prefill GEMM `pipe` on `w`, `X` and `Y`: a Q4
    /// matrix decoded to bf16 into `wdq` first, then read from there.
    fn gemm_ops(&self, pipe: usize, w: Wt, x: Buffer, y: Buffer, m: u32) -> Vec<Op> {
        let a = &self.a;
        let groups = match self.gemm_tn.get(&pipe) {
            Some(&(tm, tn)) => pf::gemm_mm2d_groups(m, w.n, tm, tn),
            None => pf::gemm_simt_groups(m, w.n),
        };
        match w.scales {
            None => vec![Op { pipe, ranges: vec![r(a.state), r(w.buf), r(x), r(y)], groups }],
            Some(so) if self.q4_fused.contains(&pipe) => {
                vec![Op { pipe, ranges: vec![r(a.state), Range::part(w.buf, 0, so), r(x), r(y), Range::part(w.buf, so, w.buf.len - so)], groups }]
            }
            Some(so) => vec![
                Op { pipe: self.dq[&(w.k, w.n)], ranges: vec![Range::part(w.buf, 0, so), Range::part(w.buf, so, w.buf.len - so), r(a.wdq)], groups: kn::dequant_q4_groups(w.k, w.n) },
                Op { pipe, ranges: vec![r(a.state), r(a.wdq), r(x), r(y)], groups },
            ],
        }
    }

    /// Whether the experts of the sparse layers are Q4_0.
    fn experts_q4(&self) -> bool {
        self.layers.iter().find_map(|l| l.moe).is_some_and(|m| m.gate_up.scales.is_some())
    }

    /// The `dequant_q4` pipeline of `w`'s shape, built the first time.
    fn dq_pipe(&mut self, w: Wt) -> Result<usize> {
        if let Some(&p) = self.dq.get(&(w.k, w.n)) {
            return Ok(p);
        }
        let p = self.get(kn::dequant_q4(w.k, w.n, self.sg))?;
        self.dq.insert((w.k, w.n), p);
        Ok(p)
    }

    fn norm(&mut self, ops: &mut Vec<Op>, x: Buffer, w: Buffer, y: Buffer) -> Result<()> {
        let (h, eps, sg) = (self.c.hidden_size as u32, self.c.rms_norm_eps, self.sg);
        self.run(ops, kn::rmsnorm(h, eps, true, sg, 0, 0), &[r(x), r(w), r(y)], 1)
    }

    /// The operations of one step: embedding, layers, head, argmax.
    fn record(&mut self) -> Result<Vec<Op>> {
        self.mv_marks.clear();
        let mut ops = Vec::new();
        let cmd = &mut ops;
        let a = &self.a;
        let (
            state,
            x,
            xn,
            qkv_a,
            scores,
            attn_out,
            gu,
            hb,
            proj_a,
            gates,
            dn_out,
            dn_y,
            logits,
            partial,
            arg_out,
        ) = (
            a.state,
            a.x,
            a.xn,
            a.qkv,
            a.scores,
            a.attn_out,
            a.gu,
            a.h,
            a.proj,
            a.gates,
            a.dn_out,
            a.dn_y,
            a.logits,
            a.arg_partial,
            a.arg_out,
        );
        let (attn, delta, sg) = (self.attn, self.delta, self.sg);
        let (h, inter, vocab) = (
            self.c.hidden_size as u32,
            self.c.intermediate_size as u32,
            self.c.vocab_size as u32,
        );
        let (embed, cos, sin) = (self.embed, self.cos, self.sin);
        self.run(
            cmd,
            qk::embed(h, sg),
            &[r(state), r(embed), r(x)],
            (h / 2).div_ceil(256),
        )?;
        // `dec.fnorm=1`: a norm and the projection after it in one dispatch (`XSrc::Norm`).
        let fnorm = self.tune.dec_fnorm != 0 && h.is_multiple_of(256) && h / 256 <= 32;
        let nsrc = kn::XSrc::Norm {
            eps: self.c.rms_norm_eps.to_bits(),
            plus_one: true,
        };
        for l in 0..self.layers.len() {
            let (input_norm, post_norm, gate_up, down) = (
                self.layers[l].input_norm,
                self.layers[l].post_norm,
                self.layers[l].gate_up,
                self.layers[l].down,
            );
            // The mixer's input: `xn`, or `x` with the norm fused into the projection.
            let (mx, msrc, mnw) = if fnorm {
                (x, nsrc, Some(input_norm))
            } else {
                self.norm(cmd, x, input_norm, xn)?;
                (xn, kn::XSrc::Plain, None)
            };
            match self.layers[l].mixer {
                Mixer::Attention {
                    qkv,
                    q_norm,
                    k_norm,
                    o,
                    k_cache,
                    v_cache,
                } => {
                    self.mv_full(
                        cmd,
                        qkv,
                        mx,
                        qkv_a,
                        h,
                        attn.qkv_width(),
                        Out::Bf16,
                        msrc,
                        mnw,
                        kn::Extra::None,
                        Vec::new(),
                    )?;
                    // `dec.pfdn=KB`: the small kernels before `o` read its start, a part each.
                    // `dec.frope=1`: the norms, RoPE and K/V append done by the
                    // attention's first pass (`dec.sdpa` > 0, heads of 256).
                    let frope = self.tune.dec_frope != 0 && self.tune.dec_sdpa > 0 && attn.head_dim == 256;
                    let (n0, w0) = self.pf_part(o, 0, attn.head_dim);
                    if !frope {
                        let mut rr = vec![r(state), r(qkv_a), r(q_norm), r(k_norm), r(cos), r(sin), r(k_cache), r(v_cache)];
                        rr.extend(w0.iter().copied());
                        self.run(cmd, qk::qk_rope_append_pf(attn, sg, n0), &rr, attn.heads + 2 * attn.kv_heads + kn::prefetch_groups_wg(n0, attn.head_dim))?;
                    }
                    match self.tune.dec_sdpa {
                        c if frope => {
                            let mut rr = vec![r(state), r(qkv_a), r(k_cache), r(v_cache), r(scores), r(q_norm), r(k_norm), r(cos), r(sin)];
                            rr.extend(w0);
                            self.run(cmd, qk::sdpa_decode_part_rope(attn, sg, c, n0), &rr, attn.heads * c + kn::prefetch_groups_wg(n0, 256))?;
                            let (n2, w2) = self.pf_part(o, 1, 256);
                            let mut rr = vec![r(qkv_a), r(scores), r(attn_out)];
                            rr.extend(w2);
                            self.run(cmd, qk::sdpa_decode_merge_pf(attn, sg, c, n2), &rr, attn.heads + kn::prefetch_groups_wg(n2, 256))?;
                        }
                        0 => self.run(
                            cmd,
                            qk::sdpa_decode(attn, sg),
                            &[
                                r(state),
                                r(qkv_a),
                                r(k_cache),
                                r(v_cache),
                                r(scores),
                                r(attn_out),
                            ],
                            attn.heads,
                        )?,
                        c => {
                            let (n1, w1) = self.pf_part(o, 1, 256);
                            let mut rr =
                                vec![r(state), r(qkv_a), r(k_cache), r(v_cache), r(scores)];
                            rr.extend(w1);
                            self.run(
                                cmd,
                                qk::sdpa_decode_part_pf(attn, sg, c, n1),
                                &rr,
                                attn.heads * c + kn::prefetch_groups_wg(n1, 256),
                            )?;
                            let (n2, w2) = self.pf_part(o, 2, 256);
                            let mut rr = vec![r(qkv_a), r(scores), r(attn_out)];
                            rr.extend(w2);
                            self.run(
                                cmd,
                                qk::sdpa_decode_merge_pf(attn, sg, c, n2),
                                &rr,
                                attn.heads + kn::prefetch_groups_wg(n2, 256),
                            )?;
                        }
                    }
                    self.mv(cmd, o, attn_out, x, attn.out_width(), h, Out::Residual)?;
                }
                Mixer::Delta {
                    proj,
                    conv_w,
                    a_log,
                    dt_bias,
                    norm,
                    out,
                    conv_state,
                    rec_state,
                } => {
                    self.mv_full(
                        cmd,
                        proj,
                        mx,
                        proj_a,
                        h,
                        delta.proj_width(),
                        ACT_OUT,
                        msrc,
                        mnw,
                        kn::Extra::None,
                        Vec::new(),
                    )?;
                    if self.tune.dec_fdn != 0 {
                        // `dec.fdn`: conv, L2 norms, gates and recurrence in one dispatch
                        // (`dn_step`), the gated norm in it too with `2` (a group a head,
                        // `dv · dk / 32` invocations); the conv state shifted by `out`'s
                        // extra groups once every group of `dn_step` has read it.
                        let gn = self.tune.dec_fdn == 2
                            && delta.v_dim.is_multiple_of(32)
                            && delta.k_dim.is_multiple_of(32)
                            && delta.v_dim * (delta.k_dim / 32) <= 1024;
                        let (cols, slices) = if gn {
                            (delta.v_dim, delta.k_dim / 32)
                        } else {
                            (self.tune.dec_cols, self.tune.dec_slices)
                        };
                        let wg = cols * slices;
                        let (n0, w0) = self.pf_part(out, 0, wg);
                        let mut rr = vec![
                            r(proj_a),
                            r(conv_state),
                            r(conv_w),
                            r(a_log),
                            r(dt_bias),
                            r(rec_state),
                        ];
                        if gn {
                            rr.push(r(norm));
                            rr.push(r(dn_y));
                        } else {
                            rr.push(r(dn_out));
                        }
                        rr.extend(w0);
                        self.run(
                            cmd,
                            qk::dn_step(delta, sg, cols, slices, gn, n0),
                            &rr,
                            delta.v_heads * delta.v_dim / cols + kn::prefetch_groups_wg(n0, wg),
                        )?;
                        if !gn {
                            let (n2, w2) = self.pf_part(out, 1, delta.v_dim);
                            let mut rr = vec![r(dn_out), r(proj_a), r(norm), r(dn_y)];
                            rr.extend(w2);
                            self.run(
                                cmd,
                                qk::dn_gated_norm_pf(delta, sg, n2),
                                &rr,
                                delta.v_heads + kn::prefetch_groups_wg(n2, delta.v_dim),
                            )?;
                        }
                        let shift = kn::Extra::ConvShift {
                            ch: delta.conv_dim(),
                            conv: delta.conv,
                        };
                        self.mv_full(
                            cmd,
                            out,
                            dn_y,
                            x,
                            delta.value(),
                            h,
                            Out::Residual,
                            kn::XSrc::Plain,
                            None,
                            shift,
                            vec![r(proj_a), r(conv_state)],
                        )?;
                    } else {
                        // `dec.pfdn=KB`: the small kernels before `out` read its start, a part each.
                        let (n0, w0) = self.pf_part(out, 0, 256);
                        let mut rr = vec![r(proj_a), r(conv_state), r(conv_w)];
                        rr.extend(w0);
                        self.run(
                            cmd,
                            qk::dn_conv_pf(delta, sg, n0),
                            &rr,
                            delta.conv_dim().div_ceil(256) + kn::prefetch_groups_wg(n0, 256),
                        )?;
                        let (n1, w1) = self.pf_part(out, 1, delta.k_dim);
                        let mut rr = vec![r(proj_a), r(a_log), r(dt_bias), r(gates)];
                        rr.extend(w1);
                        self.run(
                            cmd,
                            qk::dn_l2_gates_pf(delta, sg, n1),
                            &rr,
                            2 * delta.k_heads + 1 + kn::prefetch_groups_wg(n1, delta.k_dim),
                        )?;
                        if self.tune.dec_rec == 1 {
                            let (cols, slices) = (self.tune.dec_cols, self.tune.dec_slices);
                            self.run(
                                cmd,
                                pf::dn_recurrence_rows(delta, sg, cols, slices),
                                &[r(state), r(proj_a), r(gates), r(rec_state), r(dn_out)],
                                delta.v_heads * delta.v_dim / cols,
                            )?;
                        } else {
                            self.run(
                                cmd,
                                qk::dn_recurrence(delta, sg),
                                &[r(proj_a), r(gates), r(rec_state), r(dn_out)],
                                delta.v_heads,
                            )?;
                        }
                        if self.tune.dec_gnorm == 2 && out.scales.is_some() && delta.v_dim == 128 {
                            self.gnorm_out_sync(cmd, out, norm, l)?;
                        } else if self.tune.dec_gnorm != 0 && out.scales.is_some() {
                            self.gnorm_out(cmd, out, norm)?;
                        } else {
                            let (n2, w2) = self.pf_part(out, 2, delta.v_dim);
                            let mut rr = vec![r(dn_out), r(proj_a), r(norm), r(dn_y)];
                            rr.extend(w2);
                            self.run(
                                cmd,
                                qk::dn_gated_norm_pf(delta, sg, n2),
                                &rr,
                                delta.v_heads + kn::prefetch_groups_wg(n2, delta.v_dim),
                            )?;
                            self.mv(cmd, out, dn_y, x, delta.value(), h, Out::Residual)?;
                        }
                    }
                }
            }
            // The post norm: into `gate_up` with `dec.fnorm`, unless the MoE's router needs `xn`.
            let (px, psrc, pnw) = if fnorm && self.layers[l].moe.is_none() {
                (x, nsrc, Some(post_norm))
            } else {
                self.norm(cmd, x, post_norm, xn)?;
                (xn, kn::XSrc::Plain, None)
            };
            // `dec.swiglu=1`: the SwiGLU done by `down` as it loads `x`;
            // `dec.fswi=1`: by `gate_up` as it stores (`Out::Swiglu`).
            let fuse = self.tune.dec_swiglu != 0;
            let fswi = !fuse && self.tune.dec_fswi != 0;
            if fswi {
                self.mv_full(
                    cmd,
                    gate_up,
                    px,
                    hb,
                    h,
                    2 * inter,
                    Out::Swiglu,
                    psrc,
                    pnw,
                    kn::Extra::None,
                    Vec::new(),
                )?;
            } else {
                self.mv_full(
                    cmd,
                    gate_up,
                    px,
                    gu,
                    h,
                    2 * inter,
                    Out::Bf16,
                    psrc,
                    pnw,
                    kn::Extra::None,
                    Vec::new(),
                )?;
            }
            if !fuse && !fswi {
                // `dec.pfk=KB`: the SwiGLU's extra groups read the start of `down`.
                let pf = self.pf_uv4(down);
                if pf > 0 {
                    let sink = self.a.arg_out;
                    self.run(
                        cmd,
                        qk::swiglu_pf(inter, sg, pf),
                        &[
                            r(gu),
                            r(hb),
                            Range::part(down.buf, 0, pf as u64 * 16),
                            r(sink),
                        ],
                        inter.div_ceil(256) + kn::prefetch_groups(pf),
                    )?;
                } else {
                    self.run(
                        cmd,
                        qk::swiglu(inter, sg),
                        &[r(gu), r(hb)],
                        inter.div_ceil(256),
                    )?;
                }
            }
            let (dx, dsrc) = if fuse {
                (gu, kn::XSrc::Swiglu)
            } else {
                (hb, kn::XSrc::Plain)
            };
            match (self.layers[l].moe, self.c.moe) {
                (Some(m), Some(mc)) => {
                    let (e, kk, mi, rows) = (
                        mc.num_experts as u32,
                        mc.top_k as u32,
                        mc.inter as u32,
                        self.rows,
                    );
                    let a = &self.a;
                    let (scores_e, route, egu, ehb, ey, esum, sout) =
                        (a.scores_e, a.route, a.egu, a.ehb, a.ey, a.esum, a.sout);
                    self.mv_x(cmd, down, dx, sout, inter, h, Out::Bf16, dsrc)?;
                    self.mv(cmd, m.router, xn, scores_e, h, e, Out::Bf16)?;
                    self.run(cmd, qk::route_qwen(e, kk, sg), &[r(scores_e), r(route)], 1)?;
                    let groups = |n: u32| kk * kn::matvec_groups(n, sg, rows);
                    let em = |q4: bool, kdim: u32, n: u32, x_row: u32, x_stride: u32| {
                        if q4 {
                            qk::experts_matvec_q4(kdim, n, kk, x_row, x_stride, sg, rows)
                        } else {
                            qk::experts_matvec(kdim, n, kk, x_row, x_stride, sg, rows)
                        }
                    };
                    let q4 = m.gate_up.scales.is_some();
                    self.run(
                        cmd,
                        em(q4, h, 2 * mi, h, 0),
                        &m.gate_up.bind(&[], &[r(xn), r(egu), r(route)]),
                        groups(2 * mi),
                    )?;
                    self.run(
                        cmd,
                        qk::swiglu_slots(mi, kk, sg),
                        &[r(egu), r(ehb)],
                        (kk * mi).div_ceil(256),
                    )?;
                    self.run(
                        cmd,
                        em(q4, mi, h, kk * mi, mi),
                        &m.down.bind(&[], &[r(ehb), r(ey), r(route)]),
                        groups(h),
                    )?;
                    self.run(
                        cmd,
                        qk::moe_combine(h, kk, sg),
                        &[r(ey), r(route), r(esum)],
                        h.div_ceil(256),
                    )?;
                    self.run(
                        cmd,
                        qk::moe_out(h, sg),
                        &[r(xn), r(m.shared_gate), r(esum), r(sout), r(x)],
                        1,
                    )?;
                }
                _ => self.mv_x(cmd, down, dx, x, inter, h, Out::Residual, dsrc)?,
            }
        }
        let (final_norm, lm_head) = (self.final_norm, self.lm_head);
        let head = cmd.len();
        self.norm(cmd, x, final_norm, xn)?;
        // The head's dispatches are those of a prefill's last row too: four, unprepared.
        let prep = std::mem::replace(&mut self.tune.dec_prep, 0);
        self.mv(cmd, lm_head, xn, logits, h, vocab, Out::Bf16)?;
        self.tune.dec_prep = prep;
        let groups = kn::argmax_groups(vocab);
        self.run(
            cmd,
            kn::argmax_partial(vocab, sg),
            &[r(logits), r(partial)],
            groups,
        )?;
        self.run(
            cmd,
            kn::argmax_final(groups, sg),
            &[r(partial), r(arg_out)],
            1,
        )?;
        // The head's four pipelines, for the last row of a prefill.
        for i in 0..4 {
            self.head_pipes[i] = cmd[head + i].pipe;
        }
        self.head_groups = cmd[head + 1].groups;
        if !self.logits32_cpu.is_empty() {
            let logits32 = self.a.logits32;
            self.tune.dec_prep = 0;
            self.mv(cmd, lm_head, xn, logits32, h, vocab, Out::F32)?;
            self.tune.dec_prep = prep;
        }
        if self.tune.dec_pf > 0 {
            return self.with_prefetch(ops);
        }
        Ok(ops)
    }

    /// `dec.pfdn`: part `i` of `w`'s start a small kernel's extra groups
    /// (of `wg` invocations) read: `dec.pfdn` KiB from `i·dec.pfdn` KiB on,
    /// in uint4, and the two bindings (the range, the sink); none past the end.
    fn pf_part(&self, w: Wt, i: u32, wg: u32) -> (u32, Vec<Range>) {
        let per = (self.tune.dec_pfdn as u64 * 1024) / 16 / (8 * wg as u64) * (8 * wg as u64);
        let total = w.scales.unwrap_or(w.buf.len) / 16;
        let off = i as u64 * per;
        if per == 0 || off + per > total {
            return (0, Vec::new());
        }
        (per as u32, vec![Range::part(w.buf, off * 16, per * 16), r(self.a.arg_out)])
    }

    /// `dec.pfk`: the uint4 of `w` a small kernel's extra groups read, from
    /// its start (0: none).
    fn pf_uv4(&self, w: Wt) -> u32 {
        let total = w.scales.unwrap_or(w.buf.len);
        ((self.tune.dec_pfk as u64 * 1024).min(total) / 16 / 2048 * 2048) as u32
    }

    /// `dec.pf=KB`: beside each op between two projections, a read of the
    /// next projection's Q4 (or bf16) weights, at most `KB` KiB a time, from
    /// their start on: the system cache warmed while the small kernels leave
    /// the memory idle (`Gpu::dispatch_free`; `HERBERT_METAL_PFPROBE`).
    fn with_prefetch(&mut self, ops: Vec<Op>) -> Result<Vec<Op>> {
        let cap = (self.tune.dec_pf as u64 * 1024) / (2048 * 16) * (2048 * 16);
        let marks = std::mem::take(&mut self.mv_marks);
        let sink = self.a.arg_out;
        let mut out = Vec::with_capacity(ops.len() * 2);
        let mut prev = 0;
        for &(mi, w) in &marks {
            let total = w.scales.unwrap_or(w.buf.len);
            let smalls = (mi - prev).max(1) as u64;
            let per = (total / smalls).min(cap) / (2048 * 16) * (2048 * 16);
            let pf = if per > 0 { Some(self.get(kn::prefetch_uv4((per / 16) as u32, self.sg))?) } else { None };
            let mut off = 0;
            for op in &ops[prev..mi] {
                out.push(op.clone());
                if let Some(pf) = pf {
                    if off + per <= total {
                        self.pf_pipes.insert(pf);
                        out.push(Op { pipe: pf, ranges: vec![Range::part(w.buf, off, per), r(sink)], groups: (per / 16 / 2048) as u32 });
                        off += per;
                    }
                }
            }
            out.push(ops[mi].clone());
            prev = mi + 1;
        }
        out.extend(ops[prev..].iter().cloned());
        self.mv_marks = marks;
        Ok(out)
    }

    /// The pipelines of a prefill block, with the settings of `tune`; `m` the
    /// rows `gemm.auto` times the GEMMs at.
    fn prefill_pipes(&mut self, m: u32) -> Result<PrefillPipes> {
        let (attn, delta, sg, t) = (self.attn, self.delta, self.sg, self.tune.clone());
        let (h, inter) = (self.c.hidden_size as u32, self.c.intermediate_size as u32);
        let has_delta = self.layers.iter().any(|l| matches!(l.mixer, Mixer::Delta { .. }));
        let a = self.a;
        // The weights a GEMM is timed on: those of the first layer that has them.
        let attn_w = self.layers.iter().find_map(|l| match l.mixer {
            Mixer::Attention { qkv, o, .. } => Some((qkv, o)),
            _ => None,
        });
        let delta_w = self.layers.iter().find_map(|l| match l.mixer {
            Mixer::Delta { proj, out, .. } => Some((proj, out)),
            _ => None,
        });
        let (l0_gu, l0_down, moe_w) = (self.layers[0].gate_up, self.layers[0].down, self.layers.iter().find_map(|l| l.moe));
        if (t.auto != 0 && !t.gemm_set) || (t.sdpa_auto != 0 && !t.sdpa_set) {
            self.autotune_inputs(m);
        }
        let gemm = |e: &mut Self, kdim: u32, n: u32, out: Out, w: Option<Wt>, x: Buffer, y: Buffer| e.gemm(kdim, n, out, w.map(|w| (w, x, y)), m);
        let (cols, slices) = (t.rec_cols, t.rec_slices);
        let grouped = match self.c.moe {
            Some(mc) if t.moe_grouped != 0 => Some(self.expert_pipes(mc.num_experts as u32, mc.top_k as u32, mc.inter as u32, moe_w, m)?),
            _ => None,
        };
        let gate_up = self.gate_up_pipes(h, inter, l0_gu, m)?;
        let sdpa = if t.sdpa_auto != 0 && !t.sdpa_set && attn_w.is_some() { self.pick_sdpa(m)? } else { self.sdpa_pipe(&t)? };
        Ok(PrefillPipes {
            embed: self.get(pf::embed_rows(h, sg))?,
            norm: self.get(pf::rmsnorm_rows(h, self.c.rms_norm_eps, true, sg))?,
            gate_up: gate_up.0,
            swiglu: gate_up.1,
            down: gemm(self, inter, h, Out::Residual, Some(l0_down), a.h, a.x)?,
            qkv: gemm(self, h, attn.qkv_width(), Out::Bf16, attn_w.map(|w| w.0), a.xn, a.qkv)?,
            rope: self.get(pf::qk_rope_append_rows(attn, sg))?,
            sdpa,
            sdpa_kind: (self.tune.sdpa, self.tune.mq()),
            o: gemm(self, attn.out_width(), h, Out::Residual, attn_w.map(|w| w.1), a.attn_out, a.x)?,
            delta: if has_delta {
                Some([
                    gemm(self, h, delta.proj_width(), ACT_OUT, delta_w.map(|w| w.0), a.xn, a.proj)?,
                    if t.conv_cols > 0 { self.get(pf::dn_conv_seg(delta, sg, t.conv_cols))? } else { self.get(pf::dn_conv_rows(delta, sg))? },
                    self.get(pf::dn_l2_gates_rows(delta, sg))?,
                    match t.rec {
                        0 => self.get(pf::dn_recurrence_rows(delta, sg, cols, slices))?,
                        1 => self.get(pf::dn_recurrence_chunk(delta, sg, cols, slices))?,
                        2 => self.get(pf::dn_chunk_apply(delta, sg, cols, slices))?,
                        4 => self.get(dnmma::dn_mma_state(delta, sg))?,
                        _ if t.rec_stage > 0 => self.get(pf::dn_recurrence_st(delta, sg, cols, slices, t.rec_stage))?,
                        _ => self.get(pf::dn_recurrence_cl(delta, sg, cols, slices, t.rec_vec != 0, t.rec_pair != 0))?,
                    },
                    self.get(pf::dn_gated_norm_rows(delta, sg))?,
                    gemm(self, delta.value(), h, Out::Residual, delta_w.map(|w| w.1), a.dn_y, a.x)?,
                ])
            } else {
                None
            },
            chunk_prep: if has_delta && t.rec == 2 { Some(self.get(pf::dn_chunk_prep(delta, sg))?) } else { None },
            tc_prep: if has_delta && t.rec == 4 { Some(self.get(dnmma::dn_mma_prep(delta, sg))?) } else { None },
            moe: match self.c.moe {
                Some(mc) => {
                    let (e, kk, mi, rows) = (mc.num_experts as u32, mc.top_k as u32, mc.inter as u32, self.rows);
                    Some([
                        gemm(self, inter, h, Out::Bf16, Some(l0_down), a.h, a.sout)?,
                        gemm(self, h, e, Out::Bf16, moe_w.map(|w| w.router), a.xn, a.scores_e)?,
                        self.get(qk::route_qwen(e, kk, sg))?,
                        self.get(if self.experts_q4() { qk::experts_matvec_q4(h, 2 * mi, kk, h, 0, sg, rows) } else { qk::experts_matvec(h, 2 * mi, kk, h, 0, sg, rows) })?,
                        self.get(qk::swiglu_slots(mi, t.block * kk, sg))?,
                        self.get(if self.experts_q4() { qk::experts_matvec_q4(mi, h, kk, kk * mi, mi, sg, rows) } else { qk::experts_matvec(mi, h, kk, kk * mi, mi, sg, rows) })?,
                        self.get(qk::moe_combine(h, kk, sg))?,
                        self.get(qk::moe_out(h, sg))?,
                    ])
                }
                None => None,
            },
            moe_grouped: grouped.map(|g| (g.0, g.1)),
            moe_tiles: grouped.map_or(0, |g| g.3),
        })
    }

    /// The `[gate | up]` GEMM of the prefill and its SwiGLU (`None`: done by
    /// the GEMM, `gemm_nax_swiglu`, which writes `h`), as `pf.fswi` says; at
    /// `2` with `gemm.auto`, the GEMM's pick then `swiglu_rows` timed against
    /// the fused kernel on 128 or 64 rows × 64 or 32 features, the fastest kept.
    fn gate_up_pipes(&mut self, h: u32, inter: u32, w0: Wt, m: u32) -> Result<(usize, Option<usize>)> {
        let t = self.tune.clone();
        let (sg, a) = (self.sg, self.a);
        let gu = self.gemm(h, 2 * inter, Out::Bf16, Some((w0, a.xn, a.gu)), m)?;
        let sw = self.get(pf::swiglu_rows(inter, sg))?;
        let timed = t.auto != 0 && !t.gemm_set;
        if t.pf_fswi == 0 || (t.pf_fswi == 2 && !timed) || w0.scales.is_some() || !h.is_multiple_of(32) || self.c.moe.is_some() {
            return Ok((gu, Some(sw)));
        }
        let mut cands: Vec<(String, usize, Vec<Op>)> = Vec::new();
        if t.pf_fswi == 2 {
            let mut ops = self.gemm_ops(gu, w0, a.xn, a.gu, m);
            ops.push(Op { pipe: sw, ranges: vec![r(a.state), r(a.gu), r(a.h)], groups: (m * inter).div_ceil(256) });
            cands.push(("apart".to_string(), gu, ops));
        }
        for (wm, wn, band) in [(4, 4, 2), (4, 2, 2), (2, 2, 2), (4, 4, 0)] {
            match self.get(pf::gemm_nax_swiglu(h, inter, sg, band, wm, wn)) {
                Ok(p) => {
                    self.gemm_tn.insert(p, (32 * wm, 32 * wn));
                    let ops = self.gemm_ops(p, w0, a.xn, a.h, m);
                    cands.push((format!("fused{}x{}b{band}", 32 * wm, 16 * wn), p, ops));
                }
                Err(e) => eprintln!("[autotune] gate_up+swiglu: fused{}x{}b{band} not built ({e})", 32 * wm, 16 * wn),
            }
        }
        if cands.is_empty() {
            return Ok((gu, Some(sw)));
        }
        if t.pf_fswi == 1 {
            return Ok((cands[0].1, None));
        }
        let mut eligible = vec![true; cands.len()];
        let gate_up_at = |e: &Self, i: usize, rows: u32| -> Vec<Op> {
            if cands[i].0 == "apart" {
                let mut ops = e.gemm_ops(gu, w0, a.xn, a.gu, rows);
                ops.push(Op { pipe: sw, ranges: vec![r(a.state), r(a.gu), r(a.h)], groups: (rows * inter).div_ceil(256) });
                ops
            } else {
                e.gemm_ops(cands[i].1, w0, a.xn, a.h, rows)
            }
        };
        self.check_candidates(&format!("gate_up+swiglu {h}x{}, {m} rows", 2 * inter), &cands, &gate_up_at, &mut eligible, a.h, false)?;
        let (p, _) = self.pick_flagged(&format!("gate_up+swiglu {h}x{}, {m} rows", 2 * inter), &cands, &eligible)?;
        Ok(if p == gu { (gu, Some(sw)) } else { (p, None) })
    }

    /// The pipeline of a dense prefill GEMM `Y[m][n] = X[m][kdim]·Wᵀ` written
    /// as `out`: the kernel `gemm.mma` and `gemm.tn` say, or with `gemm.auto`
    /// the fastest of `gemm_mma_rows` and `gemm_mm2d` (tiles of 64 or 32 rows
    /// × 64, 32 or 16 columns) and `gemm_nax` (128 or 64 × 128 or 64), timed once per shape on `io` (W, X, Y) at `m`
    /// rows. A Q4 `W`: each of those after `dequant_q4` (the two timed
    /// together), and `gemm_mma_q4` in the four forms.
    fn gemm(&mut self, kdim: u32, n: u32, out: Out, io: Option<(Wt, Buffer, Buffer)>, m: u32) -> Result<usize> {
        let t = self.tune.clone();
        let (sg, mf) = (self.sg, t.m_first != 0);
        let q4 = io.is_some_and(|(w, ..)| w.scales.is_some());
        if let Some((w, ..)) = io.filter(|_| q4) {
            self.dq_pipe(w)?;
        }
        let mm2d = |e: &mut Self, tm: u32, tn: u32| -> Result<usize> {
            let p = e.get(pf::gemm_mm2d(kdim, n, out, sg, mf, tm, tn))?;
            e.gemm_tn.insert(p, (tm, tn));
            Ok(p)
        };
        let nax = |e: &mut Self, wm: u32, wn: u32, band: u32, two: bool| -> Result<usize> {
            let p = e.get(pf::gemm_nax(kdim, n, out, sg, band, wm, wn, false, two))?;
            e.gemm_tn.insert(p, (32 * wm, 32 * wn));
            Ok(p)
        };
        // A `gemm_mma_rows` form: its pipeline, its groups registered when its tile is not 64 rows.
        let mma_form = |e: &mut Self, form: pf::MmaForm, fused: bool| -> Result<usize> {
            let p = e.get(if fused { pf::gemm_mma_q4(kdim, n, out, sg, mf, form) } else { pf::gemm_mma_form(kdim, n, out, sg, mf, form) })?;
            if form.bm != 64 {
                e.gemm_tn.insert(p, (form.bm, 64));
            }
            if fused {
                e.q4_fused.insert(p);
            }
            Ok(p)
        };
        let (w, x, y) = match io {
            Some(io) if t.auto != 0 && !t.gemm_set => io,
            _ => {
                return match t.mma {
                    2 => mm2d(self, 64, t.tn),
                    3 => nax(self, 4, (t.tn / 32).max(1), 2, kdim.is_multiple_of(512)),
                    1 => mma_form(self, pf::MmaForm::of(t.mform), q4 && t.q4_fuse == 1),
                    _ => self.get(pf::gemm_simt(kdim, n, out, sg, t.simt == 2, mf)),
                }
            }
        };
        let key = format!("{kdim}x{n} {out:?}{}, {m} rows", if q4 { " q4" } else { "" });
        if let Some(&p) = self.gemm_pick.get(&key) {
            return Ok(p);
        }
        let mut cands: Vec<(String, usize, Vec<Op>)> = Vec::new();
        let mut add = |e: &mut Self, name: String, p: usize| {
            let ops = e.gemm_ops(p, w, x, y, m);
            cands.push((name, p, ops));
        };
        let mma = self.get(pf::gemm_mma(kdim, n, out, sg, mf))?;
        add(self, "mma".to_string(), mma);
        // The `gemm_mma_rows` forms (`MmaForm::of`); the half-staged ones (4, 7, 8) are
        // picked only when exact for every matrix of the shape (`gemm.half`, the Q4
        // scales), else timed and logged (`*`).
        let half_ok = t.gemm_half != 0 && q4 && io.is_some_and(|(w0, ..)| self.same_shape(w0).iter().all(|&w| scales_half_exact(w)));
        if q4 && m <= 64 {
            eprintln!("[autotune] gemm {kdim}x{n}{}: the half-staged forms are {}", if t.gemm_half != 0 { "" } else { " (gemm.half=0)" }, if half_ok { "exact (every scale in [2^-14, 2^13) or zero)" } else { "not eligible" });
        }
        let mut eligible = vec![true];
        let forms = [("mma_blk", 1), ("mma_sm", 2), ("mma32_sm", 3), ("mma32_sm_h", 4), ("mma32_sm_k64", 5), ("mma32_sm_d", 6), ("mma32_sm_hd", 7), ("mma_sm_h", 8), ("mma32_sm_k64d", 9)];
        // With the `f16` feature the half-staged forms are their plain ones, already there.
        for (name, f) in forms.into_iter().filter(|&(_, f)| !(crate::core::bf16::F16 && [4, 7, 8].contains(&f))) {
            match mma_form(self, pf::MmaForm::of(f), false) {
                Ok(p) => {
                    add(self, name.to_string(), p);
                    eligible.push(half_ok || !pf::MmaForm::of(f).half);
                }
                Err(e) => eprintln!("[autotune] gemm {key}: {name} not built ({e})"),
            }
        }
        for (tm, tn) in [(64, 64), (64, 32), (64, 16), (32, 64), (32, 32), (32, 16)] {
            // A device without Metal 4's tensors, or a tile `matmul2d` refuses: the other kernels.
            match mm2d(self, tm, tn) {
                Ok(p) => {
                    add(self, format!("mm2d{tm}x{tn}"), p);
                    eligible.push(true);
                }
                Err(e) => eprintln!("[autotune] gemm {key}: mm2d{tm}x{tn} not built ({e})"),
            }
        }
        if kdim.is_multiple_of(32) {
            // `two`: mlx's two loops over K (multiples of 512).
            let two_ok = kdim.is_multiple_of(512);
            for (wm, wn, band, two) in [(4, 4, 0, false), (2, 4, 0, false), (4, 2, 2, false), (2, 2, 2, false), (4, 4, 2, true), (4, 4, 0, true), (2, 4, 2, true)] {
                if two && !two_ok {
                    continue;
                }
                let name = format!("nax{}x{}b{band}{}", 32 * wm, 32 * wn, if two { "t" } else { "" });
                match nax(self, wm, wn, band, two) {
                    Ok(p) => {
                        add(self, name, p);
                        eligible.push(true);
                    }
                    Err(e) => eprintln!("[autotune] gemm {key}: {name} not built ({e})"),
                }
            }
        }
        if q4 {
            for (name, f) in [("q4mma", 0), ("q4mma_blk", 1), ("q4mma_sm", 2), ("q4mma32_sm", 3), ("q4mma32_sm_h", 4), ("q4mma32_sm_k64", 5), ("q4mma32_sm_d", 6), ("q4mma32_sm_k64d", 9)]
                .into_iter()
                .filter(|&(_, f)| !(crate::core::bf16::F16 && f == 4))
            {
                match mma_form(self, pf::MmaForm::of(f), true) {
                    Ok(p) => {
                        add(self, name.to_string(), p);
                        eligible.push(half_ok || !pf::MmaForm::of(f).half);
                    }
                    Err(e) => eprintln!("[autotune] gemm {key}: {name} not built ({e})"),
                }
            }
        }
        let f32_out = matches!(out, Out::F32);
        self.check_candidates(&format!("gemm {key}"), &cands, &|e, i, rows| e.gemm_ops(cands[i].1, w, x, y, rows), &mut eligible, y, f32_out)?;
        let (best, _) = self.pick_flagged(&format!("gemm {key}"), &cands, &eligible)?;
        self.gemm_pick.insert(key, best);
        Ok(best)
    }

    /// `moe.grouped=1`: the sort and the experts' `[gate | up]` and `down` over
    /// tiles of `moe.tile` rows, and those rows. With `gemm.auto` (and neither
    /// `gemm.mma` nor `gemm.tn` given), timed at load at `m` rows for tiles of
    /// 16, 32 and 64 rows (`moe.tile` only, if given): each of the two GEMMs on
    /// `gemm_mma_rows` and `gemm_mm2d_rows` (64, 32 and 16 columns, the
    /// `[gate | up]` with the gather its `matmul2d` needs), the fastest kept,
    /// then the tile whose two sum the least; the weights of the first sparse
    /// layer, the route the router gave the last of the first period of layers
    /// on pseudo-random ids (`real_route`: a first prefill at load).
    fn expert_pipes(&mut self, e: u32, kk: u32, mi: u32, w: Option<MoeLayer>, m: u32) -> Result<([usize; 3], Option<usize>, u32, u32)> {
        let t = self.tune.clone();
        let (sg, mf, h) = (self.sg, t.m_first != 0, self.c.hidden_size as u32);
        // Q4 experts: `gemm_mma_rows` decoding them as it stages them; `matmul2d` reads bf16 only.
        let q4 = self.experts_q4();
        let kernels = |e2: &mut Self, bm: u32| -> Result<(u32, usize, usize, usize)> {
            let tiles = pf::moe_tiles(m, kk, e, bm);
            let sort = e2.get(pf::moe_sort(e, kk, m, bm, sg))?;
            let mma = |kdim: u32, n: u32, rows_of: pf::MmaRows| if q4 { pf::gemm_mma_rows_q4(kdim, n, Out::Bf16, sg, mf, rows_of) } else { pf::gemm_mma_rows(kdim, n, Out::Bf16, sg, mf, rows_of) };
            let gu = e2.get(mma(h, 2 * mi, pf::MmaRows::Gather { top_k: kk, tiles, bm }))?;
            let dn = e2.get(mma(mi, h, pf::MmaRows::Scatter { tiles, bm }))?;
            Ok((tiles, sort, gu, dn))
        };
        let w = match w {
            Some(w) if t.auto != 0 && !t.gemm_set && self.real_route => w,
            _ => {
                let (tiles, sort, gu, dn) = kernels(self, t.moe_tile)?;
                return Ok(([sort, gu, dn], None, t.moe_tile, tiles));
            }
        };
        let a = self.a;
        let gather = self.get(pf::moe_gather(h, kk, sg))?;
        let g_op = Op { pipe: gather, ranges: vec![r(a.state), r(a.moe_list), r(a.xn), r(a.xs)], groups: pf::moe_gather_groups(m, kk, h) };
        let gu = |x: Buffer| w.gate_up.bind(&[r(a.state)], &[r(x), r(a.egu), r(a.moe_list), r(a.moe_tiles)]);
        let dn = w.down.bind(&[r(a.state)], &[r(a.ehb), r(a.ey), r(a.moe_list), r(a.moe_tiles)]);
        let mut best: Option<(f64, [usize; 3], Option<usize>, u32, u32)> = None;
        let tiles_rows: Vec<u32> = if t.moe_tile_set { vec![t.moe_tile] } else { vec![16, 32, 64] };
        for bm in tiles_rows {
            let (tiles, scan, mma_gu, mma_dn) = kernels(self, bm)?;
            let (gu_rows, dn_rows) = (pf::MmaRows::Gather { top_k: kk, tiles, bm }, pf::MmaRows::Scatter { tiles, bm });
            // The sort: each thread walking all pairs, or counting with atomics.
            let sort_op = |pipe: usize| vec![Op { pipe, ranges: vec![r(a.state), r(a.route), r(a.moe_list), r(a.moe_tiles)], groups: 1 }];
            let atomic = self.get(pf::moe_sort_atomic(e, kk, m, bm, sg))?;
            let (sort, s_sort) = self.pick(&format!("experts sort by {bm} rows, {m} rows"), &[("scan".to_string(), scan, sort_op(scan)), ("atomic".to_string(), atomic, sort_op(atomic))])?;
            // The list and the table of these tiles, by the sort kept.
            self.time_ops(&sort_op(sort), 1)?;
            let mut c_gu = vec![("mma".to_string(), mma_gu, vec![Op { pipe: mma_gu, ranges: gu(a.xn), groups: pf::gemm_mma_expert_groups(2 * mi, tiles) }])];
            let mut c_dn = vec![("mma".to_string(), mma_dn, vec![Op { pipe: mma_dn, ranges: dn.clone(), groups: pf::gemm_mma_expert_groups(h, tiles) }])];
            for tn in if q4 { vec![] } else { vec![64, 32, 16] } {
                let built = self.get(pf::gemm_mm2d_rows(h, 2 * mi, Out::Bf16, sg, mf, bm, tn, gu_rows)).and_then(|p| Ok((p, self.get(pf::gemm_mm2d_rows(mi, h, Out::Bf16, sg, mf, bm, tn, dn_rows))?)));
                match built {
                    Ok((p_gu, p_dn)) => {
                        self.gemm_tn.insert(p_gu, (bm, tn));
                        self.gemm_tn.insert(p_dn, (bm, tn));
                        c_gu.push((format!("gather+mm2d{tn}"), p_gu, vec![g_op.clone(), Op { pipe: p_gu, ranges: gu(a.xs), groups: pf::gemm_mm2d_expert_groups(2 * mi, tiles, tn) }]));
                        c_dn.push((format!("mm2d{tn}"), p_dn, vec![Op { pipe: p_dn, ranges: dn.clone(), groups: pf::gemm_mm2d_expert_groups(h, tiles, tn) }]));
                    }
                    Err(err) => eprintln!("[autotune] experts: mm2d{tn} by {bm} rows not built ({err})"),
                }
            }
            let (p_gu, s_gu) = self.pick(&format!("experts gate_up {h}x{} by {bm} rows, {m} rows", 2 * mi), &c_gu)?;
            let (p_dn, s_dn) = self.pick(&format!("experts down {mi}x{h} by {bm} rows, {m} rows"), &c_dn)?;
            let total = s_gu + s_dn + s_sort;
            eprintln!("[autotune] experts by {bm} rows, {m} rows: sort {:.3} ms, total {:.3} ms", s_sort * 1e3, total * 1e3);
            if best.as_ref().is_none_or(|b| total < b.0) {
                best = Some((total, [sort, p_gu, p_dn], if p_gu == mma_gu { None } else { Some(gather) }, bm, tiles));
            }
        }
        let (_, pipes, g, bm, tiles) = best.expect("expert tiles");
        eprintln!("[autotune] experts, {m} rows -> tiles of {bm} rows");
        Ok((pipes, g, bm, tiles))
    }

    /// The pipeline of the fastest candidate (a name, its pipeline, the
    /// dispatches timed), logged: each candidate repeated in one command
    /// The prefill's attention kernel `t` names.
    fn sdpa_pipe(&mut self, t: &Tune) -> Result<usize> {
        let (attn, sg) = (self.attn, self.sg);
        match t.sdpa {
            1 => self.get(pf::sdpa_flash(attn, sg, t.sdpa_bq, t.sdpa_bk)),
            2 | 3 => self.get(pf::sdpa_mma(attn, sg, t.mq(), t.sdpa_mk, t.sdpa == 3, t.sdpa_qreg != 0, t.sdpa_kt != 0)),
            4 => self.get(pf::sdpa_mm2d(attn, sg, t.mq(), t.sdpa_mk, t.sdpa_msg)),
            5..=7 => self.get(pf::sdpa_nax(attn, sg, t.sdpa == 6, t.sdpa == 7)),
            _ => self.get(pf::sdpa_rows(attn, sg)),
        }
    }

    /// `sdpa.auto`: the attention kernels timed on a block of `m` rows at the
    /// position `min(capacity − m, m)` (the inputs of [`Self::autotune_inputs`],
    /// the caches as they are) — the scalar flash, `sdpa_mma` staged with Q in
    /// registers, `sdpa_mm2d`, `sdpa_nax` — the fastest kept in the tune; a kernel that
    /// does not build, or whose group does not divide the block, is skipped.
    fn pick_sdpa(&mut self, m: u32) -> Result<usize> {
        let (attn, a) = (self.attn, self.a);
        let (k_cache, v_cache) = self
            .layers
            .iter()
            .find_map(|l| match l.mixer {
                Mixer::Attention { k_cache, v_cache, .. } => Some((k_cache, v_cache)),
                _ => None,
            })
            .expect("an attention layer");
        let pos = (attn.capacity.saturating_sub(m)).min(m);
        // SAFETY: a shared buffer of 16 bytes; the GPU is idle.
        unsafe {
            let st = self.a.state.ptr as *mut u32;
            st.add(1).write_volatile(pos);
            st.add(STATE_ROWS as usize).write_volatile(m);
        }
        // The caches the kernels read, filled (at load they are zeros: every kernel would then output zeros alike, and
        // the correctness check would see nothing; the first prompt overwrites them).
        Self::fill_random16(&[k_cache, v_cache], 0x2545_f491);
        let base = self.tune.clone();
        let mut variants: Vec<(&str, Tune)> = Vec::new();
        let mut t = base.clone();
        t.sdpa = 1;
        variants.push(("flash", t));
        if base.block.is_multiple_of(16) {
            let mut t = base.clone();
            (t.sdpa, t.sdpa_mq, t.sdpa_mk, t.sdpa_qreg) = (3, 16, 32, 1);
            variants.push(("mma", t));
        }
        if base.block.is_multiple_of(64) {
            let mut t = base.clone();
            (t.sdpa, t.sdpa_mq, t.sdpa_mk, t.sdpa_msg) = (4, 64, 32, 8);
            variants.push(("mm2d", t));
        }
        let mut t = base.clone();
        (t.sdpa, t.sdpa_mq) = (5, 64);
        variants.push(("nax", t));
        let mut cands = Vec::new();
        let mut tunes = Vec::new();
        for (name, t) in variants {
            match self.sdpa_pipe(&t) {
                Ok(pipe) => {
                    let groups = match t.sdpa {
                        1 => m.div_ceil(t.sdpa_bq) * attn.heads,
                        4..=7 => m.div_ceil(t.mq()) * attn.heads,
                        _ => m.div_ceil(t.mq()) * attn.kv_heads,
                    };
                    let ranges = vec![r(a.state), r(a.qkv), r(k_cache), r(v_cache), r(a.attn_out)];
                    cands.push((name.to_string(), pipe, vec![Op { pipe, ranges, groups }]));
                    tunes.push(t);
                }
                Err(e) => eprintln!("[autotune] sdpa {name}: not built ({e})"),
            }
        }
        let mut eligible = vec![true; cands.len()];
        let sdpa_at = |_: &Self, i: usize, rows: u32| -> Vec<Op> {
            let t = &tunes[i];
            let groups = match t.sdpa {
                1 => rows.div_ceil(t.sdpa_bq) * attn.heads,
                4..=7 => rows.div_ceil(t.mq()) * attn.heads,
                _ => rows.div_ceil(t.mq()) * attn.kv_heads,
            };
            vec![Op { pipe: cands[i].1, ranges: cands[i].2[0].ranges.clone(), groups }]
        };
        self.check_candidates(&format!("sdpa {m} rows at {pos}"), &cands, &sdpa_at, &mut eligible, a.attn_out, false)?;
        let (pipe, _) = self.pick_flagged(&format!("sdpa {m} rows at {pos}"), &cands, &eligible)?;
        let i = cands.iter().position(|c| c.1 == pipe).expect("the picked candidate");
        let w = &tunes[i];
        (self.tune.sdpa, self.tune.sdpa_mq, self.tune.sdpa_mk, self.tune.sdpa_qreg, self.tune.sdpa_msg) = (w.sdpa, w.sdpa_mq, w.sdpa_mk, w.sdpa_qreg, w.sdpa_msg);
        Ok(pipe)
    }

    /// buffer so that it lasts ~2 ms, five rounds with the candidates in turn,
    /// the median kept.
    fn pick(&mut self, what: &str, cands: &[(String, usize, Vec<Op>)]) -> Result<(usize, f64)> {
        let all = vec![true; cands.len()];
        self.pick_flagged(what, cands, &all)
    }

    /// The candidates of a load-time pick checked against the first one (a kernel in production) on the output `out`:
    /// each candidate runs on the full block and on a partial one (a row count that is a multiple of no tile, as a
    /// prompt's last block is), from a zeroed output (so that an accumulating output compares its product alone), with
    /// `ops_at(self, i, rows)` giving candidate `i`'s dispatches for that row count (the groups must match the rows, as
    /// in a prompt). A candidate whose output differs from the reference by more than 1/64 of the largest magnitude
    /// (two right kernels differ by their accumulation order, under a 16-bit rounding) is logged `wrong` and never
    /// picked. `matmul2d` on register fragments (`sdpa_nax`, `gemm_nax`) maps lanes to elements as the M5 does: on the
    /// M3 Ultra it builds, runs fast and returns garbage (2026-11-01: the bf16 and f16 engines at `9e80538` gave 0/48
    /// ids on the M3 Ultra). The output and the state's row count are restored afterwards.
    fn check_candidates(&mut self, what: &str, cands: &[(String, usize, Vec<Op>)], ops_at: &dyn Fn(&Self, usize, u32) -> Vec<Op>, eligible: &mut [bool], out: Buffer, f32_out: bool) -> Result<()> {
        // SAFETY: shared buffers of `len` bytes, a state of 16 bytes; the GPU is idle between the runs.
        let bytes = |b: &Buffer| unsafe { std::slice::from_raw_parts(b.ptr as *const u8, b.len as usize).to_vec() };
        let restore = |b: &Buffer, v: &[u8]| unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.ptr, v.len()) };
        let zero = |b: &Buffer| unsafe { std::ptr::write_bytes(b.ptr, 0, b.len as usize) };
        let values = |v: &[u8]| -> Vec<f32> {
            if f32_out {
                v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            } else {
                v.chunks_exact(2).map(|c| crate::core::bf16::h16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
            }
        };
        let st = self.a.state.ptr as *mut u32;
        let m = unsafe { st.add(STATE_ROWS as usize).read_volatile() };
        let before = bytes(&out);
        let row = (out.len / self.tune.block.max(1) as u64) as usize;
        let mut wrong: Vec<String> = Vec::new();
        for rows in [m, (m - m / 3 - 3).max(1)] {
            unsafe { st.add(STATE_ROWS as usize).write_volatile(rows) };
            let used = (rows as usize * row).min(before.len());
            zero(&out);
            let ops = ops_at(self, 0, rows);
            self.time_ops(&ops, 1)?;
            let reference = values(&bytes(&out)[..used]);
            let scale = reference.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
            for i in 1..cands.len() {
                if !eligible[i] {
                    continue;
                }
                zero(&out);
                let ops = ops_at(self, i, rows);
                self.time_ops(&ops, 1)?;
                let got = values(&bytes(&out)[..used]);
                let err = reference.iter().zip(&got).fold(0f32, |a, (&x, &y)| a.max((x - y).abs()));
                if !(err <= scale / 64.0) {
                    eligible[i] = false;
                    wrong.push(format!("{} ({rows} rows: max err {err:.3e} of {scale:.3e})", cands[i].0));
                }
            }
        }
        restore(&out, &before);
        unsafe { st.add(STATE_ROWS as usize).write_volatile(m) };
        if !wrong.is_empty() {
            eprintln!("[autotune] {what}: wrong against {}, never picked: {}", cands[0].0, wrong.join(", "));
        }
        Ok(())
    }

    /// [`Self::pick`] among the candidates `eligible` marks; the others are
    /// timed and logged (`*` after their name) but never picked.
    fn pick_flagged(&mut self, what: &str, cands: &[(String, usize, Vec<Op>)], eligible: &[bool]) -> Result<(usize, f64)> {
        assert!(eligible.len() == cands.len() && eligible.iter().any(|&e| e), "{what}: an eligible candidate");
        let started = Instant::now();
        self.heat(&cands[0].2)?;
        self.time_ops(&cands[0].2, 1)?;
        let one = self.time_ops(&cands[0].2, 1)?;
        let reps = ((2e-3 / one.max(1e-6)).ceil() as u32).clamp(1, 64);
        let mut times = vec![Vec::new(); cands.len()];
        for _ in 0..5 {
            for (i, c) in cands.iter().enumerate() {
                times[i].push(self.time_ops(&c.2, reps)? / reps as f64);
            }
        }
        let medians: Vec<f64> = times
            .iter_mut()
            .map(|v| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            })
            .collect();
        let best = (0..cands.len()).filter(|&i| eligible[i]).min_by(|&i, &j| medians[i].total_cmp(&medians[j])).expect("candidates");
        let line: Vec<String> = cands.iter().zip(&medians).zip(eligible).map(|((c, s), &e)| format!("{}{} {:.3} ms", c.0, if e { "" } else { "*" }, s * 1e3)).collect();
        eprintln!("[autotune] {what}: {} -> {}", line.join(", "), cands[best].0);
        self.autotune_seconds += started.elapsed().as_secs_f64();
        self.last_timed = Some(Instant::now());
        Ok((cands[best].1, medians[best]))
    }

    /// `auto.heat`: the GPU brought to its sustained regime before a timed comparison, when it has been idle for
    /// more than 200 ms: `ops` run back to back, ~50 ms a command buffer, until `auto.heat` ms of GPU time.
    fn heat(&mut self, ops: &[Op]) -> Result<()> {
        let heat = self.tune.auto_heat as f64 * 1e-3;
        if heat <= 0.0 || self.last_timed.is_some_and(|t| t.elapsed().as_secs_f64() < 0.2) {
            return Ok(());
        }
        let one = self.time_ops(ops, 1)?.max(1e-6);
        let reps = ((0.05 / one).ceil() as u32).clamp(1, 256);
        let mut t = 0.0;
        while t < heat {
            t += self.time_ops(ops, reps)?;
        }
        self.last_timed = Some(Instant::now());
        Ok(())
    }

    /// GPU seconds of `reps` runs of `ops` in one command buffer.
    fn time_ops(&self, ops: &[Op], reps: u32) -> Result<f64> {
        let mut cmd = self.gpu.begin()?;
        for _ in 0..reps {
            for op in ops {
                self.gpu.dispatch(&mut cmd, &self.list[op.pipe], &op.ranges, op.groups);
            }
        }
        Ok(self.gpu.submit_wait(cmd)?.gpu)
    }

    /// Before the GEMMs are timed: `m` rows in the state, the inputs they read
    /// filled with bf16 values in [-1, 1) (a prompt overwrites them).
    fn autotune_inputs(&mut self, m: u32) {
        let a = self.a;
        Self::fill_random16(&[a.x, a.xn, a.h, a.attn_out, a.dn_y, a.ehb, a.qkv], 0x9e37_79b9);
        // SAFETY: a shared buffer of 16 bytes; the GPU is idle.
        unsafe { (self.a.state.ptr as *mut u32).add(STATE_ROWS as usize).write_volatile(m) };
    }

    /// `bufs` filled with 16-bit values in [-1, 1) (xorshift from `seed`): the inputs of a timed or checked kernel.
    fn fill_random16(bufs: &[Buffer], mut seed: u32) {
        for b in bufs {
            let v: Vec<Bf16> = (0..b.len / 2)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    f32_to_h16((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
                })
                .collect();
            // SAFETY: a shared buffer of `len` bytes; the GPU is idle.
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, b.ptr, v.len() * 2) };
        }
    }

    /// The operations of a prefill block of `m` rows; the head runs on the
    /// last row of the `last` block only, with the decode's kernels.
    fn prefill_ops(&self, m: u32, last: bool, layers: usize) -> Vec<Op> {
        self.prefill_ops_in(m, m, last, layers)
    }

    /// [`Self::prefill_ops`] with the pipelines of the bucket of `pick` rows
    /// (at least `m`): the two halves of a split block take the same ones, so
    /// that their dispatches pair one for one.
    fn prefill_ops_in(&self, m: u32, pick: u32, last: bool, layers: usize) -> Vec<Op> {
        let kp = &self.pf.iter().find(|(b, _)| pick <= *b).or(self.pf.last()).expect("prefill pipelines").1;
        let (a, attn, delta, t) = (&self.a, self.attn, self.delta, &self.tune);
        let (h, inter, vocab) = (self.c.hidden_size as u32, self.c.intermediate_size as u32, self.c.vocab_size as u32);
        let mut ops = Vec::new();
        let mut d = |pipe: usize, ranges: Vec<Range>, groups: u32| ops.push(Op { pipe, ranges, groups });
        // A GEMM's dispatches (a Q4 matrix decoded first).
        macro_rules! gemm {
            ($pipe:expr, $w:expr, $x:expr, $y:expr) => {
                for o in self.gemm_ops($pipe, $w, $x, $y, m) {
                    d(o.pipe, o.ranges, o.groups);
                }
            };
        }
        d(kp.embed, vec![r(a.state), r(a.tokens), r(self.embed), r(a.x)], (m * h / 2).div_ceil(256));
        for l in &self.layers[..layers] {
            d(kp.norm, vec![r(a.x), r(l.input_norm), r(a.xn)], m);
            match l.mixer {
                Mixer::Attention { qkv, q_norm, k_norm, o, k_cache, v_cache } => {
                    gemm!(kp.qkv, qkv, a.xn, a.qkv);
                    d(kp.rope, vec![r(a.state), r(a.qkv), r(q_norm), r(k_norm), r(self.cos), r(self.sin), r(k_cache), r(v_cache)], m * (attn.heads + 2 * attn.kv_heads));
                    let (sk, mq) = kp.sdpa_kind;
                    if sk == 1 {
                        d(kp.sdpa, vec![r(a.state), r(a.qkv), r(k_cache), r(v_cache), r(a.attn_out)], m.div_ceil(t.sdpa_bq) * attn.heads);
                    } else if sk >= 4 {
                        d(kp.sdpa, vec![r(a.state), r(a.qkv), r(k_cache), r(v_cache), r(a.attn_out)], m.div_ceil(mq) * attn.heads);
                    } else if sk >= 2 {
                        d(kp.sdpa, vec![r(a.state), r(a.qkv), r(k_cache), r(v_cache), r(a.attn_out)], m.div_ceil(mq) * attn.kv_heads);
                    } else {
                        d(kp.sdpa, vec![r(a.state), r(a.qkv), r(k_cache), r(v_cache), r(a.scores), r(a.attn_out)], m * attn.heads);
                    }
                    gemm!(kp.o, o, a.attn_out, a.x);
                }
                Mixer::Delta { proj, conv_w, a_log, dt_bias, norm, out, conv_state, rec_state } => {
                    let [p_proj, p_conv, p_l2, p_rec, p_gn, p_out] = kp.delta.expect("DeltaNet pipelines");
                    gemm!(p_proj, proj, a.xn, a.proj);
                    d(p_conv, vec![r(a.state), r(a.proj), r(conv_state), r(conv_w)], if t.conv_cols > 0 { delta.conv_dim() / t.conv_cols } else { delta.conv_dim().div_ceil(256) });
                    d(p_l2, vec![r(a.proj), r(a_log), r(dt_bias), r(a.gates)], m * (2 * delta.k_heads + 1));
                    let groups = delta.v_heads * delta.v_dim / t.rec_cols;
                    match (kp.tc_prep, kp.chunk_prep) {
                        (Some(prep), _) => {
                            d(prep, vec![r(a.state), r(a.proj), r(a.gates), r(a.tc_mat), r(a.tc_vec)], m.div_ceil(dnmma::C) * delta.v_heads);
                            d(p_rec, vec![r(a.state), r(a.proj), r(a.tc_mat), r(a.tc_vec), r(rec_state), r(a.dn_out)], delta.v_heads * delta.v_dim / dnmma::DVS);
                        }
                        (None, Some(prep)) => {
                            d(prep, vec![r(a.state), r(a.proj), r(a.gates), r(a.dn_prep)], m.div_ceil(16) * delta.v_heads);
                            d(p_rec, vec![r(a.state), r(a.proj), r(a.dn_prep), r(rec_state), r(a.dn_out)], groups);
                        }
                        (None, None) => d(p_rec, vec![r(a.state), r(a.proj), r(a.gates), r(rec_state), r(a.dn_out)], groups),
                    }
                    d(p_gn, vec![r(a.dn_out), r(a.proj), r(norm), r(a.dn_y)], m * delta.v_heads);
                    gemm!(p_out, out, a.dn_y, a.x);
                }
            }
            d(kp.norm, vec![r(a.x), r(l.post_norm), r(a.xn)], m);
            match kp.swiglu {
                Some(sw) => {
                    gemm!(kp.gate_up, l.gate_up, a.xn, a.gu);
                    d(sw, vec![r(a.state), r(a.gu), r(a.h)], (m * inter).div_ceil(256));
                }
                None => gemm!(kp.gate_up, l.gate_up, a.xn, a.h),
            }
            match (l.moe, kp.moe, self.c.moe) {
                (Some(w), Some([p_sdown, p_router, p_route, p_gu, p_swiglu, p_down, p_comb, p_out]), Some(mc)) => {
                    let (kk, mi) = (mc.top_k as u32, mc.inter as u32);
                    let mg = |n: u32| m * kk * kn::matvec_groups(n, self.sg, self.rows);
                    gemm!(p_sdown, l.down, a.h, a.sout);
                    gemm!(p_router, w.router, a.xn, a.scores_e);
                    d(p_route, vec![r(a.scores_e), r(a.route)], m);
                    match kp.moe_grouped {
                        Some(([p_sort, p_ggu, p_gdown], gather)) => {
                            let tiles = kp.moe_tiles;
                            let eg = |pipe: usize, n: u32| match self.gemm_tn.get(&pipe) {
                                Some(&(_, tn)) => pf::gemm_mm2d_expert_groups(n, tiles, tn),
                                None => pf::gemm_mma_expert_groups(n, tiles),
                            };
                            d(p_sort, vec![r(a.state), r(a.route), r(a.moe_list), r(a.moe_tiles)], 1);
                            let x_gu = match gather {
                                Some(p) => {
                                    d(p, vec![r(a.state), r(a.moe_list), r(a.xn), r(a.xs)], pf::moe_gather_groups(m, kk, h));
                                    a.xs
                                }
                                None => a.xn,
                            };
                            d(p_ggu, w.gate_up.bind(&[r(a.state)], &[r(x_gu), r(a.egu), r(a.moe_list), r(a.moe_tiles)]), eg(p_ggu, 2 * mi));
                            d(p_swiglu, vec![r(a.egu), r(a.ehb)], (m * kk * mi).div_ceil(256));
                            d(p_gdown, w.down.bind(&[r(a.state)], &[r(a.ehb), r(a.ey), r(a.moe_list), r(a.moe_tiles)]), eg(p_gdown, h));
                        }
                        None => {
                            d(p_gu, w.gate_up.bind(&[], &[r(a.xn), r(a.egu), r(a.route)]), mg(2 * mi));
                            d(p_swiglu, vec![r(a.egu), r(a.ehb)], (m * kk * mi).div_ceil(256));
                            d(p_down, w.down.bind(&[], &[r(a.ehb), r(a.ey), r(a.route)]), mg(h));
                        }
                    }
                    d(p_comb, vec![r(a.ey), r(a.route), r(a.esum)], m * h.div_ceil(256));
                    d(p_out, vec![r(a.xn), r(w.shared_gate), r(a.esum), r(a.sout), r(a.x)], m);
                }
                _ => {
                    gemm!(kp.down, l.down, a.h, a.x);
                }
            }
        }
        // The last row of the prompt through the decode head; the blocks
        // before it only fill the caches and states.
        if last {
            let (norm, head, arg1, arg2) = (self.head_pipes[0], self.head_pipes[1], self.head_pipes[2], self.head_pipes[3]);
            let row = (m as u64 - 1) * h as u64 * 2;
            d(norm, vec![Range::part(a.x, row, h as u64 * 2), r(self.final_norm), Range::part(a.xn, 0, h as u64 * 2)], 1);
            d(head, self.lm_head.mv_ranges(r(a.xn), r(a.logits)), self.head_groups);
            let groups = kn::argmax_groups(vocab);
            d(arg1, vec![r(a.logits), r(a.arg_partial)], groups);
            d(arg2, vec![r(a.arg_partial), r(a.arg_out)], 1);
        }
        ops
    }

    /// One prefill block at the current position: its operations encoded,
    /// submitted, waited on.
    fn block(&mut self, ids: &[u32], last: bool) -> Result<()> {
        let layers = self.layers.len();
        self.prefill_gpu_seconds += self.block_in(ids, last, layers)?;
        Ok(())
    }

    /// [`Self::block`] through the first `layers` layers; its GPU seconds.
    fn block_in(&mut self, ids: &[u32], last: bool, layers: usize) -> Result<f64> {
        let m = ids.len();
        // SAFETY: shared buffers sized for a block; the GPU is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), self.a.tokens.ptr as *mut u32, m);
            let st = self.a.state.ptr as *mut u32;
            st.write_volatile(ids[m - 1]);
            st.add(1).write_volatile(self.pos as u32);
            st.add(2).write_volatile(0);
            st.add(STATE_ROWS as usize).write_volatile(m as u32);
        }
        // HERBERT_METAL_PFSPLIT=k (2 to 4; 1 means 2; with HERBERT_METAL_PFCONC=2): the block as k parts of
        // equal rows (multiples of 64) whose dispatches are interleaved one for one, each part's rows bound past
        // the ones before and its own state (rows, position), so that the barriers fall between groups of
        // independent dispatches (a measure; dense models).
        let parts = std::env::var("HERBERT_METAL_PFSPLIT").ok().map(|v| v.parse::<usize>().unwrap_or(2).clamp(2, 4)).filter(|_| self.c.moe.is_none() && m >= 128 && self.prof.is_none() && self.stats.is_none());
        let cuts: Vec<(usize, usize)> = match parts {
            Some(k) => {
                let size = m.div_ceil(k).next_multiple_of(64);
                (0..m).step_by(size).map(|at| (at, size.min(m - at))).collect()
            }
            None => vec![(0, m)],
        };
        // SAFETY: as above.
        unsafe {
            let st = self.a.state.ptr as *mut u32;
            st.add(STATE_ROWS as usize).write_volatile(cuts[0].1 as u32);
            for (p, &(at, rows)) in cuts.iter().enumerate().skip(1) {
                let s2 = self.a.state2[p - 1].ptr as *mut u32;
                s2.write_volatile(ids[m - 1]);
                s2.add(1).write_volatile((self.pos + at) as u32);
                s2.add(2).write_volatile(0);
                s2.add(STATE_ROWS as usize).write_volatile(rows as u32);
            }
        }
        let ops = if cuts.len() > 1 {
            let pick = cuts.iter().map(|c| c.1).max().unwrap_or(m) as u32;
            let n = cuts.len();
            let lists: Vec<Vec<Op>> = cuts.iter().enumerate().map(|(p, &(at, rows))| self.rows_past(self.prefill_ops_in(rows as u32, pick, last && p + 1 == n, layers), at, rows, p)).collect();
            let mut its: Vec<_> = lists.into_iter().map(|l| l.into_iter()).collect();
            let mut ops = Vec::new();
            loop {
                let mut any = false;
                for it in &mut its {
                    if let Some(o) = it.next() {
                        ops.push(o);
                        any = true;
                    }
                }
                if !any {
                    break;
                }
            }
            ops
        } else {
            self.prefill_ops(m as u32, last, layers)
        };
        if std::env::var_os("HERBERT_METAL_PFDUMP").is_some() {
            // SAFETY: 16-byte shared buffers.
            let s1 = unsafe { std::slice::from_raw_parts(self.a.state.ptr as *const u32, 4).to_vec() };
            eprintln!("[pfdump] m {m} state {s1:?} (buf{}), {} parts", self.a.state.index, cuts.len());
            for op in ops.iter().take(14) {
                let r: Vec<String> = op.ranges.iter().map(|r| format!("{}+{}", r.buf.index, r.offset)).collect();
                eprintln!("[pfdump] {} groups {} [{}]", self.list[op.pipe].name, op.groups, r.join(" "));
            }
        }
        if let Some(st) = self.stats.as_mut() {
            st.pass();
            let mut s = 0.0;
            for op in &ops {
                s += st.one(&self.gpu, "prefill", &self.list[op.pipe], &op.ranges, op.groups)?;
            }
            self.pos += m;
            return Ok(s);
        }
        if let Some((p, _)) = self.prof.as_mut() {
            for op in &ops {
                p.one(&self.gpu, &self.list[op.pipe], &op.ranges, op.groups)?;
            }
            let s = p.take();
            self.pos += m;
            return Ok(s);
        }
        // HERBERT_METAL_PFCONC=1: the block in a concurrent encoder, a buffer barrier before every dispatch;
        // =2: a barrier only before a dispatch that reads or writes what was written since the last one, or
        // writes what was read since (the bindings each kernel writes, `Pipeline::writes`; byte ranges). Measures.
        let conc = std::env::var("HERBERT_METAL_PFCONC").ok();
        let deps = conc.as_deref() == Some("2");
        let mut cmd = self.gpu.begin_with(conc.is_some())?;
        // HERBERT_METAL_PFONLY=gemm|nogemm: only the GEMMs, or all but them (a knockout measure; wrong results).
        let only = std::env::var("HERBERT_METAL_PFONLY").ok();
        let overlap = |a: &Range, b: &Range| a.buf.index == b.buf.index && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
        let (mut dirty, mut read): (Vec<Range>, Vec<Range>) = (Vec::new(), Vec::new());
        let mut barriers = 0;
        for op in &ops {
            let p = &self.list[op.pipe];
            let g = p.name.starts_with("gemm");
            if only.as_deref().is_some_and(|o| (o == "gemm") != g) {
                continue;
            }
            if !deps {
                self.gpu.dispatch(&mut cmd, p, &op.ranges, op.groups);
                continue;
            }
            let writes = |i: usize| p.writes.is_empty() || p.writes.contains(&(i as u32));
            let w: Vec<Range> = op.ranges.iter().enumerate().filter(|(i, _)| writes(*i)).map(|(_, r)| *r).collect();
            let rd: Vec<Range> = op.ranges.iter().enumerate().filter(|(i, _)| !writes(*i)).map(|(_, r)| *r).collect();
            let need = op.ranges.iter().any(|x| dirty.iter().any(|d| overlap(x, d))) || w.iter().any(|x| read.iter().any(|d| overlap(x, d)));
            if need && std::env::var_os("HERBERT_METAL_PFCONC_WHY").is_some() {
                let raw: Vec<String> = op.ranges.iter().enumerate().filter(|(_, x)| dirty.iter().any(|d| overlap(x, d))).map(|(i, x)| format!("b{i}=buf{}", x.buf.index)).collect();
                let war: Vec<String> = w.iter().filter(|x| read.iter().any(|d| overlap(x, d))).map(|x| format!("buf{}", x.buf.index)).collect();
                eprintln!("[pfconc] barrier before {} (writes {:?}): reads/writes dirty {raw:?}, writes read {war:?}", p.name, p.writes);
            }
            if need {
                self.gpu.barrier(&mut cmd);
                barriers += 1;
                dirty.clear();
                read.clear();
            }
            self.gpu.dispatch_free(&mut cmd, p, &op.ranges, op.groups);
            dirty.extend(w);
            read.extend(rd);
        }
        if deps && std::env::var_os("HERBERT_METAL_PFCONC_LOG").is_some() {
            eprintln!("[pfconc] {} dispatches, {barriers} barriers", ops.len());
        }
        let s = self.gpu.submit_wait(cmd)?.gpu;
        self.pos += m;
        Ok(s)
    }

    /// The dispatches `ops` of a half of a split block: every buffer of rows
    /// bound to the `rows` rows from `row0` (its bytes per row: its length
    /// over the block's rows; the chunk operators per 32 rows, `dn_prep` per
    /// 16), so that the two halves' bindings do not overlap; the second
    /// parts' states bound as `state2[part − 1]`.
    fn rows_past(&self, mut ops: Vec<Op>, row0: usize, rows: usize, part: usize) -> Vec<Op> {
        let a = &self.a;
        let block = self.tune.block as u64;
        let per_row = [a.tokens, a.x, a.xn, a.qkv, a.attn_out, a.gu, a.h, a.proj, a.gates, a.dn_out, a.dn_y];
        let per_chunk = [(a.tc_mat, 32u64), (a.tc_vec, 32), (a.dn_prep, 16)];
        for op in &mut ops {
            for r in &mut op.ranges {
                if part > 0 && r.buf.index == a.state.index {
                    *r = Range::whole(a.state2[part - 1]);
                } else if per_row.iter().any(|b| b.index == r.buf.index) {
                    // Added to the view's own offset (the head reads the block's last row).
                    let bytes = r.buf.len / block;
                    let whole = r.offset == 0 && r.size == r.buf.len;
                    r.offset += row0 as u64 * bytes;
                    if whole {
                        r.size = rows as u64 * bytes;
                    }
                } else if let Some(&(b, c)) = per_chunk.iter().find(|(b, _)| b.index == r.buf.index && b.len > 16) {
                    let bytes = b.len / block.div_ceil(c);
                    let whole = r.offset == 0 && r.size == r.buf.len;
                    r.offset += row0 as u64 / c * bytes;
                    if whole {
                        r.size = (rows as u64).div_ceil(c) * bytes;
                    }
                }
            }
        }
        ops
    }

    /// Before the experts are timed: `prefill.block` pseudo-random ids through
    /// the first period of layers in one block, with the experts' default
    /// kernels, so that the route buffer holds what the router chose for them.
    fn route_probe(&mut self) -> Result<()> {
        let started = Instant::now();
        let block = self.tune.block;
        let probe = self.prefill_pipes(block)?;
        self.pf.push((block, probe));
        let layers = self.period();
        let ids = Self::probe_ids(block);
        self.reset()?;
        self.block_in(&ids, false, layers)?;
        self.reset()?;
        self.pf.clear();
        self.real_route = true;
        self.autotune_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// The first layers up to and including the first full-attention one.
    fn period(&self) -> usize {
        self.layers.iter().position(|l| matches!(l.mixer, Mixer::Attention { .. })).map_or(self.layers.len(), |i| i + 1)
    }

    /// Pseudo-random ids for the timings at load.
    fn probe_ids(n: u32) -> Vec<u32> {
        (0..n).map(|i| 1000 + (i * 7919) % 50000).collect()
    }

    /// The rows per block for long prompts: `prefill.block` pseudo-random ids
    /// prefilled through every layer, cut by each of 1024, 2048, 4096, 8192
    /// rows up to `prefill.block` ([`Self::blocks_of`], blocks of equal size),
    /// GPU time after a warm-up block, three rounds in alternating order and
    /// the best of each; the fastest kept, then the conversation emptied. A prompt of `n` ids is then cut into `ceil(n / chunk)` blocks.
    fn autotune_chunk(&mut self) -> Result<()> {
        let started = Instant::now();
        let block = self.tune.block;
        let cands: Vec<u32> = [1024, 2048, 4096, 8192].into_iter().filter(|&c| c < block).chain([block]).collect();
        if cands.len() < 2 {
            return Ok(());
        }
        let layers = self.layers.len();
        let ids = Self::probe_ids(block);
        // The heater (`auto.heat`): the first candidate's blocks until the GPU is in its sustained regime. Then three
        // rounds, the order reversed every other one, the median of each candidate: on the M5 the GPU's performance
        // state drops within seconds; timed cold, 1024 rows won over 2048 on the 4B at pp8192 and lost 9 % sustained.
        let heat = self.tune.auto_heat as f64 * 1e-3;
        self.reset()?;
        let mut warmed = self.block_in(&ids[..cands[0] as usize], false, layers)?;
        while warmed < heat {
            self.reset()?;
            warmed += self.block_in(&ids[..cands[0] as usize], false, layers)?;
        }
        let mut times = vec![Vec::new(); cands.len()];
        for round in 0..3 {
            let order: Vec<usize> = if round % 2 == 0 { (0..cands.len()).collect() } else { (0..cands.len()).rev().collect() };
            for i in order {
                self.reset()?;
                self.chunk = cands[i];
                let mut t = 0.0;
                let pftime = std::env::var_os("HERBERT_METAL_PFTIME").is_some();
                for (p, part) in self.blocks_of(ids.len()).into_iter().map(|(at, len)| &ids[at..at + len]).enumerate() {
                    let b = self.block_in(part, false, layers)?;
                    if pftime {
                        eprintln!("[pftime] autotune {} rows, round {round}, block {p}: {:.3} ms", cands[i], b * 1e3);
                    }
                    t += b;
                }
                times[i].push(t);
            }
        }
        let medians: Vec<f64> = times
            .iter_mut()
            .map(|v| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            })
            .collect();
        let line: Vec<String> = cands.iter().zip(&medians).map(|(c, t)| format!("{c} {:.3} ms", t * 1e3)).collect();
        let best = (0..cands.len()).min_by(|&i, &j| medians[i].total_cmp(&medians[j])).map_or(block, |i| cands[i]);
        self.reset()?;
        self.chunk = best;
        eprintln!("[autotune] blocks over {block} rows, every layer, heated {warmed:.1} s, median of 3 rounds: {} -> {best}", line.join(", "));
        self.autotune_seconds += started.elapsed().as_secs_f64();
        self.last_timed = Some(Instant::now());
        Ok(())
    }

    /// The blocks `(start, rows)` of a prompt of `n` ids: `ceil(n / chunk)`
    /// blocks of equal size (rounded up to 64 rows, the last one shorter), so
    /// that a prompt a little longer than a block is not cut into a full
    /// block and a sliver.
    fn blocks_of(&self, n: usize) -> Vec<(usize, usize)> {
        let chunk = (self.chunk as usize).max(1);
        let k = n.div_ceil(chunk);
        let size = n.div_ceil(k).next_multiple_of(64).min(chunk);
        (0..n).step_by(size).map(|at| (at, size.min(n - at))).collect()
    }

    /// GPU seconds of `reps` runs of the step `ops` in one command buffer,
    /// encoded as the step is (`step_conc`, the cache warmers free).
    fn encode_step(&self, ops: &[Op], reps: u32) -> Result<f64> {
        let mut cmd = self.gpu.begin_with(self.step_conc)?;
        for _ in 0..reps {
            for op in ops {
                if self.pf_pipes.contains(&op.pipe) {
                    self.gpu.dispatch_free(&mut cmd, &self.list[op.pipe], &op.ranges, op.groups);
                } else {
                    self.gpu.dispatch(&mut cmd, &self.list[op.pipe], &op.ranges, op.groups);
                }
            }
        }
        Ok(self.gpu.submit_wait(cmd)?.gpu)
    }

    /// `dec.auto=1`: the forms of the decode step timed at load on the whole
    /// step, one axis after another (the best of each kept before the next):
    /// the matvecs' persistent groups among `mv.auto`'s candidates
    /// (`mv.persists`, [`MV_PERSISTS`]: picked per shape, they can lose on
    /// the step, M3 Ultra 0.8B 514 → 501 tok/s), the delta rule (`dec.rec`, `dec.cols`, `dec.slices`), the attention's
    /// runs (`dec.sdpa`), the SwiGLU in `down` (`dec.swiglu`), the inputs
    /// prepared for the masks (`dec.prep`), the cache warmers in the small
    /// kernels (`dec.pfk`), the concurrent encoder and its warmers (`dec.conc`,
    /// `dec.pf`). An axis whose key `--tune` gives keeps it. Timed at
    /// position `min(capacity − 1, 4096)` (the attention's cost grows with
    /// the context), the states reset afterwards.
    fn tune_step(&mut self) -> Result<()> {
        let started = Instant::now();
        let pos = (self.capacity - 1).min(4096) as u32;
        type Set = fn(&mut Tune, u32);
        let axes: Vec<(&str, &[&str], Vec<u32>, Set)> = vec![
            ("persist", &["mv.persists"], vec![0, 1], |t, v| {
                t.mv_persists = if v == 1 {
                    MV_PERSISTS.to_vec()
                } else {
                    Vec::new()
                }
            }),
            (
                "rec",
                &["dec.rec", "dec.cols", "dec.slices"],
                vec![0, 1, 2, 3, 4],
                |t, v| {
                    (t.dec_rec, t.dec_cols, t.dec_slices) =
                        [(0, 16, 16), (1, 16, 16), (1, 8, 16), (1, 32, 8), (1, 16, 8)][v as usize];
                },
            ),
            ("sdpa", &["dec.sdpa"], vec![0, 8, 16, 32, 64], |t, v| {
                t.dec_sdpa = v
            }),
            ("fdn", &["dec.fdn"], vec![0, 1, 2], |t, v| t.dec_fdn = v),
            ("frope", &["dec.frope"], vec![0, 1], |t, v| t.dec_frope = v),
            ("fnorm", &["dec.fnorm"], vec![0, 1], |t, v| t.dec_fnorm = v),
            ("swiglu", &["dec.swiglu"], vec![0, 1], |t, v| {
                t.dec_swiglu = v
            }),
            ("fswi", &["dec.fswi"], vec![0, 1], |t, v| t.dec_fswi = v),
            // `dec.gnorm=2` (groups waiting on others) is not offered: Metal does not
            // promise the norm's groups run first, a load could hang (and it lost: M3 Ultra
            // 27B 44.6 → 42.8 tok/s).
            ("gnorm", &["dec.gnorm"], vec![0, 1], |t, v| t.dec_gnorm = v),
            ("prep", &["dec.prep"], vec![0, 1], |t, v| t.dec_prep = v),
            ("pfk", &["dec.pfk"], vec![0, 1024, 4096], |t, v| {
                t.dec_pfk = v
            }),
            (
                "pfdn",
                &["dec.pfdn"],
                vec![0, 512, 1024, 2048, 3072, 4096],
                |t, v| t.dec_pfdn = v,
            ),
            ("conc", &["dec.conc", "dec.pf"], vec![0, 1, 2], |t, v| {
                (t.dec_conc, t.dec_pf) = [(0, 0), (1, 0), (0, 2048)][v as usize]
            }),
        ];
        let names = |axis: &str, v: u32| -> String {
            match axis {
                "rec" => [
                    "dn_recurrence",
                    "rows c16s16",
                    "rows c8s16",
                    "rows c32s8",
                    "rows c16s8",
                ][v as usize]
                    .to_string(),
                "fdn" => ["apart", "dn_step", "dn_step+gnorm"][v as usize].to_string(),
                "conc" => ["serial", "concurrent", "concurrent pf2048"][v as usize].to_string(),
                "persist" => ["none", "640/1280/2560/5120"][v as usize].to_string(),
                _ => v.to_string(),
            }
        };
        for (axis, keys, values, set) in axes {
            if keys.iter().any(|k| self.tune.given.iter().any(|g| g == k)) {
                continue;
            }
            // An axis another one's choice has made moot.
            if (axis == "gnorm" && self.tune.dec_fdn != 0)
                || (axis == "fswi" && self.tune.dec_swiglu != 0)
                || (axis == "frope" && (self.tune.dec_sdpa == 0 || self.attn.head_dim != 256))
            {
                continue;
            }
            let mut line = Vec::new();
            let mut best = (f64::MAX, values[0]);
            for &v in &values {
                set(&mut self.tune, v);
                self.step_conc = self.tune.dec_conc != 0 || self.tune.dec_pf > 0;
                // A form this GPU cannot run (M1: `dn_step` with the gated norm, 512
                // invocations a group, the pipeline takes 448) is left out.
                let ops = match self.record() {
                    Ok(ops) => ops,
                    Err(e) => {
                        line.push(format!("{} unavailable ({e})", names(axis, v)));
                        continue;
                    }
                };
                // SAFETY: a shared buffer of 16 bytes; the GPU is idle.
                unsafe {
                    let st = self.a.state.ptr as *mut u32;
                    st.write_volatile(0);
                    st.add(1).write_volatile(pos);
                    st.add(2).write_volatile(0);
                    st.add(3).write_volatile(1);
                }
                self.encode_step(&ops, 2)?;
                let mut ts: Vec<f64> = (0..7)
                    .map(|_| self.encode_step(&ops, 1))
                    .collect::<Result<_>>()?;
                ts.sort_by(f64::total_cmp);
                line.push(format!("{} {:.3} ms", names(axis, v), ts[3] * 1e3));
                if ts[3] < best.0 {
                    best = (ts[3], v);
                }
            }
            set(&mut self.tune, best.1);
            eprintln!(
                "[autotune] decode step at position {pos}, {axis}: {} -> {}",
                line.join(", "),
                names(axis, best.1)
            );
        }
        self.step_conc = self.tune.dec_conc != 0 || self.tune.dec_pf > 0;
        self.ops = self.record()?;
        self.reset()?;
        self.autotune_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// One token through the whole model at the current position.
    fn step(&mut self, token: u32) -> Result<()> {
        if self.pos >= self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        // SAFETY: a shared buffer of 16 bytes; the GPU is idle.
        unsafe {
            let st = self.a.state.ptr as *mut u32;
            st.write_volatile(token);
            st.add(1).write_volatile(self.pos as u32);
            st.add(2).write_volatile(0);
            st.add(3).write_volatile(1);
        }
        if let Some(st) = self.stats.as_mut() {
            st.pass();
            for op in &self.ops {
                self.gpu_seconds += st.one(&self.gpu, "step", &self.list[op.pipe], &op.ranges, op.groups)?;
            }
        } else if let Some((_, p)) = self.prof.as_mut() {
            for op in &self.ops {
                p.one(&self.gpu, &self.list[op.pipe], &op.ranges, op.groups)?;
            }
            self.gpu_seconds += p.take();
        } else {
            self.gpu_seconds += self.encode_step(&self.ops, 1)?;
        }
        self.steps += 1;
        // SAFETY: the step wrote the argmax and the logits before it completed.
        unsafe {
            self.argmax = (self.a.arg_out.ptr as *const u32).add(kn::STATE_ARGMAX as usize).read_volatile();
            std::ptr::copy_nonoverlapping(self.a.logits.ptr as *const u16, self.logits_cpu.as_mut_ptr(), self.logits_cpu.len());
            std::ptr::copy_nonoverlapping(self.a.logits32.ptr as *const f32, self.logits32_cpu.as_mut_ptr(), self.logits32_cpu.len());
        }
        self.pos += 1;
        Ok(())
    }

    /// Feed `token` through the decode step: the f32 logits of the position
    /// that follows. Needs [`Options::logits_f32`].
    pub fn score(&mut self, token: u32) -> Result<&[f32]> {
        if self.logits32_cpu.is_empty() {
            return Err(HerbertError::Backend("score needs logits_f32".into()));
        }
        self.step(token)?;
        Ok(&self.logits32_cpu)
    }

    /// Append the tokens; the greedy token of the last one. By the blocks
    /// [`Self::blocks_of`] says (their size chosen at load, `autotune_chunk`), or token by token through the decode step
    /// (`prefill.block=1`, or a single token).
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32> {
        if self.pos + ids.len() > self.capacity {
            return Err(HerbertError::Backend(format!("the conversation reached the {} reserved positions", self.capacity)));
        }
        if self.pf.is_empty() || ids.len() < 2 {
            for &t in ids {
                self.step(t)?;
            }
            return Ok(self.argmax);
        }
        let blocks = self.blocks_of(ids.len());
        let n = blocks.len();
        // HERBERT_METAL_PFTIME=1: each block's GPU seconds (against `autotune_chunk`'s, which timed the same blocks at load).
        let pftime = std::env::var_os("HERBERT_METAL_PFTIME").is_some();
        for (i, (at, len)) in blocks.into_iter().enumerate() {
            let before = self.prefill_gpu_seconds;
            self.block(&ids[at..at + len], i + 1 == n)?;
            if pftime {
                eprintln!("[pftime] block {i}/{n} rows {len} at {at}: {:.3} ms", (self.prefill_gpu_seconds - before) * 1e3);
            }
        }
        // SAFETY: the last block wrote the argmax and the logits before it completed.
        unsafe {
            self.argmax = (self.a.arg_out.ptr as *const u32).add(kn::STATE_ARGMAX as usize).read_volatile();
            std::ptr::copy_nonoverlapping(self.a.logits.ptr as *const u16, self.logits_cpu.as_mut_ptr(), self.logits_cpu.len());
        }
        Ok(self.argmax)
    }

    /// Decode from `first`: feed a token, hand the argmax and the logits to
    /// `f`, which returns the next token or `None` to stop. The number of
    /// tokens fed.
    pub fn generate(&mut self, first: u32, max_steps: usize, f: &mut dyn FnMut(u32, &[u16]) -> Result<Option<u32>>) -> Result<usize> {
        let mut token = first;
        let mut fed = 0;
        while fed < max_steps {
            self.step(token)?;
            fed += 1;
            match f(self.argmax, &self.logits_cpu)? {
                Some(next) => token = next,
                None => break,
            }
        }
        Ok(fed)
    }

    /// The decode step dispatch by dispatch (`HERBERT_METAL_PROFILE`), at the
    /// current position: each in a command buffer of its own, in the step's
    /// order, its GPU time; then the whole step in one. Per kernel (the
    /// emitted name): GPU seconds summed and dispatches, the largest first;
    /// and the whole step's GPU seconds. The position does not move, the
    /// activations are overwritten: for measuring only.
    /// `HERBERT_METAL_PFPROBE=1`: does reading a matrix just before its
    /// matvec make the matvec faster (the system cache keeping it)? For the
    /// shapes of the DeltaNet's `out` and of `down`, over every layer's
    /// matrix: the matvec alone, a read of the whole Q4 matrix alone, and the
    /// read then the matvec, of all or of half the matrix. Printed in µs per layer.
    pub fn probe_prefetch(&mut self) -> Result<()> {
        let sg = self.sg;
        let a = self.a;
        let shapes: Vec<(Wt, Buffer, Buffer)> = self
            .layers
            .iter()
            .find_map(|l| match l.mixer {
                Mixer::Delta { out, .. } => Some(out),
                _ => None,
            })
            .map(|o| (o, a.dn_y, a.x))
            .into_iter()
            .chain(std::iter::once((self.layers[0].down, a.h, a.x)))
            .collect();
        let sink = a.arg_out;
        for (w0, x, y) in shapes {
            if w0.scales.is_none() {
                continue;
            }
            let ws = self.same_shape(w0);
            let mv = self.get(kn::matvec_q4_split(w0.k, w0.n, Out::Residual, sg, 2, true, 4, 0, kn::XSrc::Plain, kn::Extra::None))?;
            let groups = kn::matvec_split_groups(w0.n, sg, 2, 4, 0);
            let mut line = Vec::new();
            for frac in [0u32, 25, 50, 100] {
                let mut only = Vec::new();
                let mut both = Vec::new();
                let mut alone = Vec::new();
                for &w in &ws {
                    let uv4 = (w.buf.len / 16) as u32;
                    let n = (uv4 as u64 * frac as u64 / 100) as u32 / 2048 * 2048;
                    if n > 0 {
                        let pf = self.get(kn::prefetch_uv4(n, sg))?;
                        let op = Op { pipe: pf, ranges: vec![Range::part(w.buf, 0, n as u64 * 16), r(sink)], groups: n / 2048 };
                        only.push(op.clone());
                        both.push(op);
                    }
                    let m = Op { pipe: mv, ranges: w.mv_ranges(r(x), r(y)), groups };
                    both.push(m.clone());
                    alone.push(m);
                }
                let l = ws.len() as f64;
                let t = |e: &Self, ops: &[Op]| -> Result<f64> {
                    let mut v = Vec::new();
                    for _ in 0..7 {
                        v.push(e.time_ops(ops, 1)?);
                    }
                    v.sort_by(f64::total_cmp);
                    Ok(v[3] / l * 1e6)
                };
                let (ta, to, tb) = (t(self, &alone)?, if only.is_empty() { 0.0 } else { t(self, &only)? }, t(self, &both)?);
                line.push(format!("{frac}%: mv {ta:.1} read {to:.1} read+mv {tb:.1} (mv after read {:.1})", tb - to));
            }
            eprintln!("[pfprobe] {}x{} x{}: {}", w0.k, w0.n, ws.len(), line.join(" | "));
            // The fixed cost: the same matvec on a fraction of its groups, and forms.
            let mut line = Vec::new();
            for (rows, split) in [(1, 4), (2, 4), (4, 4), (2, 2), (4, 1)] {
                let p = self.get(kn::matvec_q4_split(w0.k, w0.n, Out::Residual, sg, rows, true, split, 0, kn::XSrc::Plain, kn::Extra::None))?;
                let full = kn::matvec_split_groups(w0.n, sg, rows, split, 0);
                let mut cells = Vec::new();
                for div in [16, 8, 4, 2, 1] {
                    let ops: Vec<Op> = ws.iter().map(|w| Op { pipe: p, ranges: w.mv_ranges(r(x), r(y)), groups: (full / div).max(1) }).collect();
                    let mut v = Vec::new();
                    for _ in 0..7 {
                        v.push(self.time_ops(&ops, 1)?);
                    }
                    v.sort_by(f64::total_cmp);
                    cells.push(format!("1/{div} {:.1}", v[3] / ws.len() as f64 * 1e6));
                }
                line.push(format!("r{rows}k{split}: {}", cells.join(" ")));
            }
            let empty = self.get(kn::prefetch_uv4(2048, sg))?;
            let ops: Vec<Op> = (0..64).map(|_| Op { pipe: empty, ranges: vec![Range::part(ws[0].buf, 0, 2048 * 16), r(sink)], groups: 1 }).collect();
            let mut v = Vec::new();
            for _ in 0..7 {
                v.push(self.time_ops(&ops, 1)?);
            }
            v.sort_by(f64::total_cmp);
            eprintln!("[pfprobe] {}x{} µs per matvec on a fraction of its groups: {} | a one-group kernel {:.2}", w0.k, w0.n, line.join(" | "), v[3] / 64.0 * 1e6);
            // A chunked layout (blocks and their scales by 576-byte chunks), on
            // scratch matrices of that size, one a layer, against the layout of now.
            let nchunk = (w0.k / 32).div_ceil(32) as u64;
            let bytes = w0.n as u64 * nchunk * 576;
            let mut scratch = Vec::new();
            for _ in 0..ws.len() {
                scratch.push(self.gpu.alloc(bytes)?);
            }
            let mut line = Vec::new();
            for (rows, split) in [(1, 4), (2, 4), (2, 2), (4, 1), (1, 2)] {
                let p = self.get(kn::matvec_q4s_chunked(w0.k, w0.n, Out::Residual, sg, rows, split))?;
                let q = self.get(kn::matvec_q4_split(w0.k, w0.n, Out::Residual, sg, rows, true, split, 0, kn::XSrc::Plain, kn::Extra::None))?;
                let groups = kn::matvec_split_groups(w0.n, sg, rows, split, 0);
                let cops: Vec<Op> = scratch.iter().map(|&b| Op { pipe: p, ranges: vec![r(b), r(x), r(y), r(b)], groups }).collect();
                let qops: Vec<Op> = ws.iter().map(|w| Op { pipe: q, ranges: w.mv_ranges(r(x), r(y)), groups }).collect();
                let mut tc = Vec::new();
                let mut tq = Vec::new();
                for _ in 0..7 {
                    tc.push(self.time_ops(&cops, 1)?);
                    tq.push(self.time_ops(&qops, 1)?);
                }
                tc.sort_by(f64::total_cmp);
                tq.sort_by(f64::total_cmp);
                let l = ws.len() as f64;
                line.push(format!("r{rows}k{split}: now {:.1} chunked {:.1}", tq[3] / l * 1e6, tc[3] / l * 1e6));
            }
            eprintln!("[pfprobe] {}x{} layouts, µs per matvec: {}", w0.k, w0.n, line.join(" | "));
        }
        Ok(())
    }

    /// `HERBERT_METAL_PROFILE=2`: what each kernel costs inside the step,
    /// the step's GPU time less that of the step without its dispatches
    /// (the activations go wrong, the times hold), median of 7; and its
    /// weights' bytes over that time for the projections.
    pub fn profile_removal(&mut self) -> Result<Vec<(String, f64, usize, f64)>> {
        let t = |e: &Self, ops: &[Op]| -> Result<f64> {
            e.encode_step(ops, 1)?;
            let mut v: Vec<f64> = (0..7).map(|_| e.encode_step(ops, 1)).collect::<Result<_>>()?;
            v.sort_by(f64::total_cmp);
            Ok(v[3])
        };
        let ops = self.ops.clone();
        let whole = t(self, &ops)?;
        let mut names: Vec<usize> = ops.iter().map(|o| o.pipe).collect();
        names.sort();
        names.dedup();
        let mut out = Vec::new();
        for p in names {
            let rest: Vec<Op> = ops.iter().filter(|o| o.pipe != p).cloned().collect();
            let n = ops.len() - rest.len();
            let bytes: f64 = self.mv_marks.iter().filter(|(i, _)| ops[*i].pipe == p).map(|(_, w)| w.buf.len as f64).sum();
            let d = whole - t(self, &rest)?;
            out.push((self.list[p].name.clone(), d, n, if d > 0.0 { bytes / d / 1e9 } else { 0.0 }));
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("[profile2] step {:.3} ms", whole * 1e3);
        Ok(out)
    }

    pub fn profile_step(&mut self) -> Result<(Vec<(String, f64, usize)>, f64)> {
        let mut per: HashMap<String, (f64, usize)> = HashMap::new();
        for op in &self.ops {
            let s = self.time_ops(std::slice::from_ref(op), 1)?;
            let e = per.entry(self.list[op.pipe].name.clone()).or_default();
            e.0 += s;
            e.1 += 1;
        }
        let whole = self.time_ops(&self.ops, 1)?;
        let mut v: Vec<(String, f64, usize)> = per.into_iter().map(|(k, (s, n))| (k, s, n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok((v, whole))
    }

    /// `HERBERT_METAL_PROFILE=3`: a prefill block of `ids` (at most
    /// `prefill.block` of them) at the current position, dispatch by dispatch
    /// (each in a command buffer of its own, in the block's order), per
    /// kernel its GPU seconds and dispatches, the largest first; then the whole
    /// block in one. The position does not move, the caches and states are
    /// overwritten: for measuring only.
    pub fn profile_block(&mut self, ids: &[u32]) -> Result<(Vec<(String, f64, usize)>, f64)> {
        let m = ids.len().min(self.chunk as usize).max(1);
        if self.pf.is_empty() {
            return Err(HerbertError::Backend("the prefill by blocks is off (prefill.block=1)".into()));
        }
        // SAFETY: shared buffers sized for a block; the GPU is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), self.a.tokens.ptr as *mut u32, m);
            let st = self.a.state.ptr as *mut u32;
            st.write_volatile(ids[m - 1]);
            st.add(1).write_volatile(self.pos as u32);
            st.add(2).write_volatile(0);
            st.add(STATE_ROWS as usize).write_volatile(m as u32);
        }
        let ops = self.prefill_ops(m as u32, true, self.layers.len());
        let mut per: HashMap<String, (f64, usize)> = HashMap::new();
        for op in &ops {
            let s = self.time_ops(std::slice::from_ref(op), 1)?;
            let e = per.entry(self.list[op.pipe].name.clone()).or_default();
            e.0 += s;
            e.1 += 1;
        }
        let whole = self.time_ops(&ops, 1)?;
        let mut v: Vec<(String, f64, usize)> = per.into_iter().map(|(k, (s, n))| (k, s, n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok((v, whole))
    }

    /// The bf16 logits of the last position.
    pub fn logits(&self) -> &[u16] {
        &self.logits_cpu
    }
}

impl Drop for QwenEngine {
    fn drop(&mut self) {
        if let Some((prefill, step)) = &self.prof {
            prefill.print("prefill");
            step.print("step");
        }
    }
}
