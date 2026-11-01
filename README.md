# herbert-metal-bf16: Qwen3.5 in bf16 on Apple GPUs, every kernel emitted at load as Metal Shading Language

The Metal counterpart of [herbert-vk-bf16](https://github.com/xigh/herbert-vk-bf16) (Vulkan, SPIR-V emitted) and [herbert-cuda-bf16](https://github.com/xigh/herbert-cuda-bf16) (NVIDIA, PTX emitted): the same model family (Qwen3.5, text), the same numerical contract (bf16 weights, bf16 activations, bf16 KV cache and DeltaNet states, f32 accumulators, one rounding to nearest even where the Hugging Face module rounds). No `.metal` file, no MLX, no Metal Performance Shaders: every compute kernel is Metal Shading Language text written at load by Rust (`src/msl`, an emitter with the API of herbert-vk-bf16's SPIR-V one), every dimension known at load written as a constant, and compiled by the driver with fast math off.

Apple documents neither the instruction set of its GPUs nor the intermediate form its compiler produces, so the engine does not write assembly here: it writes source text for the shape of the model and for the chip, and hands it to the driver.

The engine runs Qwen3.5-0.8B, 2B, 4B and 9B, Qwen3.8-27B (`qwen3_5_text`) and the sparse Qwen3.6-35B-A3B (`qwen3_5_moe_text`): block prefill on the matrix units (`simdgroup_matrix`, and Metal 4's `matmul2d`), the delta rule by chunks on the matrix units, the prefill attention on the matrix units, a decode step recorded once and replayed for each token.

**bf16 by default, f16 as a feature.** Built with `--features f16`, the same tree is the f16 engine: every 16-bit value the GPU keeps (weights converted at load to nearest even, activations, K/V cache, logits, RoPE tables, DeltaNet states) is IEEE f16 instead of bf16 and every kernel is emitted in `half`; the accumulators stay f32. It is meant for chips without a fast bf16 path (the M1's GPU computes bf16 much slower than f16). On the M3 Ultra the two run at the same speed.

**Measured at load, checked against a kernel in production.** The engine does not guess which kernel form, tile, block size or expert sort is fastest on the chip it runs on: at load it times the candidates of each GEMM shape of the model, of the prefill attention and of the decode step, and keeps the fastest. Before any candidate can be picked, it is run once against a kernel already in production (the first candidate) on the full block and on a partial one, its dispatches regenerated for that row count, from a zeroed output; a candidate whose output differs by more than 1/64 of the largest magnitude is logged `wrong` and never picked. Every choice is printed on stderr as `[autotune]`.

Author: Philippe Anel. License: MIT, see `LICENSE`.

## Build and run

macOS on Apple silicon, the stable Rust toolchain. Nothing else: the Metal framework is the system's.

```sh
cargo build --release --locked                                     # the bf16 engine (default)
cargo build --release --locked --features f16 --target-dir target/f16   # the f16 engine
```

`MODEL_DIR` is a Hugging Face checkpoint directory of Qwen3.5 (`config.json`, `tokenizer.json`, the bf16 safetensors); `model_type` in `config.json` picks the engine. A chat:

```sh
target/release/herbert-metal-chat ~/models/Qwen3.5-0.8B --no-system --no-think --ask 'What is the capital of France?'
target/release/herbert-metal-chat ~/models/Qwen3.5-4B --no-system --ask @question.txt --max-tokens 200
target/f16/release/herbert-metal-chat ~/models/Qwen3.5-2B --no-system --no-think --ask 'Bonjour'
```

The first answer comes after the load-time measure, which takes most of the start-up: the driver compiles the 688 kernels of the 0.8B in 0.11 s, timing and checking the candidates takes 20 to 35 s on the 0.8B depending on the context reserved, more than two minutes on the 27B. Giving a setting with `--tune` (below) fixes that part instead of timing it.

`herbert-metal-chat` options:

| option | what it does |
|---|---|
| `--ctx N` | positions reserved (default 4096) |
| `--max-tokens N` | ceiling per answer (default 1024) |
| `--system TEXT \| --no-system` | the system turn, or none |
| `--ask TEXT\|@FILE` | a user turn (repeatable: one turn each; `@FILE` reads it from a file) |
| `--temperature T` | sampling instead of greedy |
| `--no-think` | the thinking block closed empty |
| `--tune K=V[,K=V]` | the engine's settings (below) |
| `--dump-msl DIR` | every emitted kernel written as `DIR/<name>.metal` |

On stderr, per turn: `[ids]` (the generated ids), `[decoded]` and `[turn]` (counts, speeds, GPU time per step). `HERBERT_METAL_TOP=N` adds the N best logits of every generated position, `HERBERT_METAL_PROMPT_IDS=1` the prompt ids, `HERBERT_METAL_PROFILE=1` the GPU time of every kernel (one command buffer per dispatch: for measuring only).

`herbert-metal-bench MODEL_DIR -p 512,2048 -n 128 -r 3 [-d N] [-w W] [--tune …]` measures ppN (N pseudo-random ids from an empty conversation, the whole prefill) and tgN (each token the argmax of the one before; `-d N`: after N tokens of context), as llama-bench measures them, and prints llama-bench's table, wall-clock then GPU time. `herbert-metal-probe` holds the probes of the chip (roundings over every f32 pattern, FMA, conversions, bandwidth, GEMM and matvec in bf16 and f16).

## Tuning

Every setting is a `--tune key=value`; the load prints the whole set. The defaults are the ones measured below: everything measured at load, with the correctness check.

- `gemm.auto=1` (default): each GEMM shape of the prefill timed at load on `simdgroup_matrix` (`gemm_mma_rows`, its forms) and on `matmul2d` (`gemm_mm2d`, tiles of 64 or 32 rows × 64, 32 or 16 columns), at 64 to 2 048 rows, the fastest of the first bucket at or above its rows kept for a block; the experts of the 35B on tiles of 16, 32 or 64 rows and both sorts; then blocks of 512, 1 024 and 2 048 rows timed. `gemm.auto=0` with `gemm.mma=1` (`simdgroup_matrix`) or `gemm.mma=2 gemm.tn=64|32|16` (`matmul2d`) fixes the kernel; `prefill.block=N` fixes the block (`1`: the prompt token by token).
- `sdpa.auto=1` (default): the prefill attention timed at load on a block among the scalar flash attention, `sdpa_mma` (`simdgroup_matrix`, K and V tiles staged by the group, Q kept in registers) and `sdpa_mm2d` (`matmul2d`), the fastest kept. Giving `sdpa.prefill` fixes it: `1` the scalar flash attention (the former default, the one that fell behind on long prompts), `3` `sdpa_mma` (with `sdpa.qreg=1`: Q in registers), `4` `sdpa_mm2d`.
- `dec.auto=1` (default): the decode step timed whole at position `min(capacity − 1, 4096)` along each of its axes, the best of each kept before the next: the delta rule with its state in registers, the attention split over the keys (`dec.sdpa`, 0 to 64 runs per head), the persistent and split matvecs, and the fusions (the RMSNorm done by the projection that follows, `dec.fnorm`; the SwiGLU in `gate_up`'s epilogue, `dec.fswi`; the DeltaNet's conv, norms, gates and delta rule in one dispatch, `dec.fdn`; the attention's head norms, RoPE and K/V append in its first pass, `dec.frope`). Each fusion gives the same bits as the kernels it replaces. A key given to `--tune` fixes its axis; `dec.auto=0` keeps the defaults.

For example, the kernels the defaults picked for the 0.8B on the M3 Ultra, fixed so that the GEMMs and the attention are not timed at load:

```sh
target/release/herbert-metal-bench ~/models/Qwen3.5-0.8B -p 2048 -n 0 --tune gemm.mma=2,gemm.tn=32,sdpa.prefill=3,sdpa.qreg=1
```

A load log shows the check and the timing side by side (0.8B, f16, a block of 512 rows; the star marks a candidate the check rejected):

```
[autotune] sdpa 512 rows at 512: wrong against flash, never picked: nax (512 rows: max err 7.790e-2 of 9.082e-2)
[autotune] sdpa 512 rows at 512: flash 1.444 ms, mma 0.533 ms, mm2d 0.825 ms, nax* 0.461 ms -> mma
[metal] 688 kernels compiled in 0.11 s, GEMMs and blocks timed in 24.11 s, 137 dispatches per token, blocks of 2048 rows; ...
```

`nax` is a family of `matmul2d` kernels on register fragments written for the M5's Neural Accelerators. On the M3 Ultra they build, run faster than the others and return wrong values (the mapping of matrix elements to lanes is the M5's): before the check, the engine picked them there.

## What the emitted MSL looks like

`--dump-msl DIR` writes every kernel as the driver receives it. Below, unedited passages of `sdpa_mma_h8kv2d256_q16k32_st_qr.metal`, the prefill attention of Qwen3.5-0.8B emitted by the f16 build: 8 heads, 2 K/V heads, head dimension 256, tiles of 16 queries × 32 keys, K and V staged in threadgroup memory by the group (`st`), Q kept in registers (`qr`). It is one of the 688 kernels the engine emitted at load for that model.

The signature, every buffer typed by the emitter:

```
#include <metal_stdlib>
using namespace metal;

kernel void sdpa_mma_h8kv2d256_q16k32_st_qr(
    device uint *b0 [[buffer(0)]],
    device ushort *b1 [[buffer(1)]],
    device uint4 *b2 [[buffer(2)]],
    device uint4 *b3 [[buffer(3)]],
    device ushort *b4 [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ngroups [[threadgroups_per_grid]],
    uint3 tpig [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float s0[2048];
    threadgroup uint4 s1[256];
    threadgroup float s2[512];
    threadgroup uint4 s3[1056];
```

The loop over the keys: a tile of K copied into threadgroup memory by the whole group, then four 8 × 8 f32 accumulators and the product of a block of Q (in registers, `v56`) by blocks of K read from threadgroup memory:

```
    for (uint v164 = 0u; v164 < v154; v164 += 32u) {
        uint v165 = v164 * 32u;
        uint v166 = v48 + v165;
        for (uint v167 = lid; v167 < 1024u; v167 += 256u) {
            uint v168 = v167 / 32u;
            uint v169 = v167 % 32u;
            uint v170 = v168 * 32u;
            uint v171 = v166 + v170;
            uint v172 = v171 + v169;
            uint4 v173 = b2[v172];
            uint v174 = v168 * 33u;
            uint v175 = v174 + v169;
            s3[v175] = v173;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        simdgroup_float8x8 m34 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m35 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m36 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m37 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_half8x8 v177;
        simdgroup_load(v177, (threadgroup half *)s3 + 0u, 264, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(m34, v56, v177, m34);
        simdgroup_half8x8 v179;
        simdgroup_load(v179, (threadgroup half *)s3 + 2112u, 264, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(m35, v56, v179, m35);
        ...
```

The online softmax, in f32: the row maximum reduced across the lanes that hold the row, then the exponentials and their sum:

```
        float v544 = m38;
        float v545 = simd_shuffle_xor(v544, 1u);
        float v546 = fmax(v544, v545);
        float v547 = simd_shuffle_xor(v546, 2u);
        float v548 = fmax(v546, v547);
        float v549 = m0;
        float v550 = fmax(v549, v548);
        float v551 = v549 - v550;
        float v552 = exp(v551);
        float m39 = as_type<float>(0x00000000u);
        float v553 = v450 - v550;
        float v554 = exp(v553);
        float v555 = m39;
        float v556 = v555 + v554;
        m39 = v556;
        ...
```

## Measured on

A Mac Studio M3 Ultra, 96 GB, macOS 26.5, for every table below: the only machine of the three where every model fits and the measure stays stable. The engine was also run on a MacBook Air M1 (8 GB) and a MacBook Air M5 (32 GB). Built with the stable Rust toolchain (1.98), release profile `opt-level=3`.

The comparators: mlx-lm 0.32.0 (on mlx 0.32.3) on the bf16 checkpoint (or the float16 one for the f16 references), llama.cpp `fb4b2737` (Metal) on the bf16 GGUF. One request at a time, pseudo-random ids, two runs per measure, the order of the engines reversed from one cell to the next. Scripts, logs and protocol: [herbert-metal-bench](https://github.com/xigh/herbert-metal-bench), tag `v1-2026-11-01`. The herbert numbers are this tree's (development commit `f07e234`), measured after the correctness check was added; mlx-lm's and llama.cpp's come from the first pass of the same night on the same machine, a few hours earlier. mlx-lm's and llama.cpp's rates are their own benchmarks' (wall-clock). herbert's decode rates are wall-clock too (`herbert-metal-bench`'s first table); its GPU rate (the GPU time of the command buffers, the second table) is 1 to 12 % higher there (Qwen3.5-0.8B tg128: 382 against 336) and is not used. herbert's prefill rates are its GPU rate, within 0.5 % of its wall-clock one (0.8B pp2048: 12 775 against 12 738).

### The same tokens

Two prompts per model, 48 greedy tokens, compared with mlx-lm. Generated ids identical before the first difference, out of 48, short prompt / long prompt:

| model | herbert bf16 against mlx-lm bf16 | herbert f16 against mlx-lm float16 |
|---|---|---|
| Qwen3.5-0.8B | 48 / 48 | 48 / 48 |
| Qwen3.5-2B | 48 / 25 | 48 / 20 |
| Qwen3.5-4B | 48 / 28 | 48 / 48 |
| Qwen3.5-9B | 48 / 48 | 48 / 48 |
| Qwen3.8-27B | 15 / 13 | 48 / 48 |
| Qwen3.6-35B-A3B | 48 / 24 | 48 / 48 |

Every divergence is at an exact tie of the reference itself (two tokens at the same bf16 logit, which the summation order then splits) or at one bf16 ulp (Qwen3.8-27B short, token 15: the reference has 27 against 26.875, herbert has both at 27). The tokens after a divergence stay largely common. `bench/compare.py` runs the check; the references are in `bench/reference/`.

### Prefill on long prompts (bf16, tokens/s)

herbert relative to mlx-lm:

| model | 2 048 | 8 192 | 24 576 | 61 440 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | ×1.21 | ×1.16 | ×1.05 | ×1.01 |
| Qwen3.5-2B | ×1.06 | ×1.04 | ×1.02 | ×0.99 |
| Qwen3.5-4B | ×1.12 | ×1.10 | ×1.07 | ×1.05 |
| Qwen3.5-9B | ×1.09 | ×1.07 | ×1.05 | ×1.04 |

| model, prompt | herbert | mlx-lm | llama.cpp |
|---|---:|---:|---:|
| Qwen3.5-0.8B, 2 048 | **12 775** | 10 517 | 9 377 |
| Qwen3.5-0.8B, 61 440 | **5 751** | 5 692 | 3 926 |
| Qwen3.5-0.8B, 81 920 | **4 827** | 4 784 | 3 232 |
| Qwen3.5-9B, 2 048 | **1 315** | 1 209 | 1 115 |
| Qwen3.5-9B, 61 440 | **989** | 954 | 790 |

Before the prefill attention moved to the matrix units, the 0.8B did 1 128 tokens/s at 81 920 tokens (in f16; on the M3 Ultra both modes run at the same speed), against 4 848 for mlx-lm and 3 240 for llama.cpp. With `sdpa_mma` the attention's share of the 0.8B's prefill (f16, kernels fixed as above, one prefill counted) is 6.1 % at 2 048 tokens and 59.6 % at 61 440 tokens, from 21 % and 89.5 % with the scalar flash attention. The lead shrinks with the length.

The two large models, at 512 and 8 192 tokens:

| model | 512 | 8 192 |
|---|---:|---:|
| Qwen3.8-27B, herbert | **374** | **365** |
| Qwen3.8-27B, llama.cpp | 320 | 307 |
| Qwen3.6-35B-A3B, herbert | **2 415** | **3 248** |
| Qwen3.6-35B-A3B, mlx-lm | 1 893 | 2 487 |
| Qwen3.6-35B-A3B, llama.cpp | 2 004 | 1 804 |

mlx-lm did not load Qwen3.8-27B in bf16 on this machine during the night, twice (`GPU Timeout Error` in `mx.eval(model.parameters())`); it loaded the next morning, for the decode measure below.

### Decode (bf16, tokens/s)

128 tokens from an empty context:

| | 0.8B | 2B | 4B | 9B | 27B | 35B-A3B |
|---|---:|---:|---:|---:|---:|---:|
| herbert | **336** | **170** | **82** | **46** | **14.7** | **87.0** |
| mlx-lm | 250 | 144 | 71 | 41 | 13.4 | 74.0 |
| llama.cpp | 210 | 135 | 69 | 41 | 13.5 | 71.7 |

mlx-lm's 0.8B to 9B values are `mlx_lm benchmark -p 2048 -g 128`: its 128 tokens follow a 2 048-token prompt. Measured again from an (almost) empty context the next morning, framed by herbert and llama.cpp in the same session, they hold within 2 % (256 / 144 / 71 / 41); the 27B and 35B-A3B values come from that measure (herbert-metal-bench, `raw/2026-11-01-metal-art004-mlx-tg-vide`).

128 tokens after 32 768 and after 81 920 tokens of context:

| context | | 0.8B | 2B | 4B | 9B |
|---|---|---:|---:|---:|---:|
| 32 768 | herbert | **268** | **146** | **67** | **42** |
| | mlx-lm | 201 | 125 | 62 | 38 |
| | llama.cpp | 174 | 121 | 59 | 38 |
| 81 920 | herbert | **185** | **129** | **56** | **36.5** |
| | mlx-lm | 156 | 105 | 52 | 34 |
| | llama.cpp | 154 | 109 | 47 | 33 |

On the 0.8B the decode step is 137 dispatches per token (263 before the fusions).

### f16

On the M3 Ultra the f16 engine runs within −3 % to +1 % of the bf16 one. On the MacBook Air M1 (an earlier development tree), f16 speeds the 512-token prefill up by 40 % on the 0.8B and 50 % on the 2B, ahead of llama.cpp in f16 (1 484 against 1 274 tokens/s on the 0.8B, 657 against 564 on the 2B), and `sdpa_mma` takes the 0.8B's prefill at 61 440 tokens from 115 to 460 tokens/s (llama.cpp f16: 401). On the M1, after 81 920 tokens of context, herbert decodes at 15.3 tokens/s against 21.5 for llama.cpp: not fixed yet.

## The code

| module | what it does |
|---|---|
| `core` | bf16 and f16 conversions, the error type, the sampler, the Qwen config |
| `model` | the safetensors checkpoints and their `config.json`, read in place or by `pread` |
| `msl` | the kernel emitter, simdgroup matrices and Metal 4 tensors included |
| `metal/gpu` | the device, shared buffers, pipelines, dispatch and submission |
| `metal/kernels`, `metal/qwen_kernels` | the decode kernels: projections, RMSNorm, RoPE and KV append, attention, SwiGLU, the DeltaNet steps, the MoE, the argmax |
| `metal/prefill` | the block kernels: the matrix-unit GEMMs, the prefill attention, the segmented conv |
| `metal/dnmma` | the delta rule by chunks on the matrix units |
| `metal/qwen` | the Qwen engine: load, the load-time measure and its check, the recorded step, the prefill by blocks |
| `chat` | the templates, streaming detokenization, the sessions |
| `bin` | `herbert-metal-chat`, `herbert-metal-bench`, `herbert-metal-probe` |

## Bench scripts

`bench/compare.py BINARY MODELS_DIR Qwen3.5-0.8B … [--ref f16]` runs the two prompts of `bench/reference/` and compares the greedy ids with the mlx-lm references (`reference/ref_qwen.py` for bf16, `reference/ref_f16.py` for float16), the prompt ids included. `bench/f16-campaign.sh OUT_DIR NAME ENGINE…` runs one model of a speed campaign (herbert in f16 and bf16, llama-bench, `mlx_lm benchmark`, Rapid-MLX through `bench/rapid_client.py`), every output kept, the paths set by environment variables; `bench/art004_tables.py` reads its logs back into tables. `bench/sampler.sh`, `bench/xctrace_gpu.py`, `bench/xctrace_table.py` and `bench/pftime_table.py` read the machine's state, Instruments traces and the per-block prefill times.

The campaign behind the tables above (protocol, the prediction written before the long prompts were measured, scripts as they ran, raw logs) is [herbert-metal-bench](https://github.com/xigh/herbert-metal-bench).
