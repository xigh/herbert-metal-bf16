//! herbert-metal-bf16 — Gemma-4 (E2B, E4B) in bf16 on Apple GPUs, every
//! compute kernel emitted at start-up.
//!
//! | module  | what it does |
//! |---------|--------------|
//! | `core`  | bf16 conversions, the error type, the sampler |
//! | `model` | the safetensors checkpoint and its `config.json`, read in place |
//! | `msl`   | the kernel emitter (MSL text for now, AIR later) |
//! | `metal` | the device, buffers, pipelines, the kernels and the engine |
//! | `chat`  | the Gemma-4 template, streaming detokenization, the session |
//!
//! Numerical contract, herbert-vk-bf16's: weights bf16, activations bf16,
//! accumulators f32, KV cache bf16, a rounding to nearest even where the
//! Hugging Face module rounds. Nothing is quantized.

pub mod chat;
pub mod core;
pub mod metal;
pub mod model;
pub mod msl;
pub mod serve;
