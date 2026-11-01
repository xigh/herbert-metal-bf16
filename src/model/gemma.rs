//! The Gemma-4 text checkpoint (E2B, E4B, 12B, 26B-A4B), read in place.
//!
//! What enters: the model directory (`config.json` with its `text_config`, one
//! or several `*.safetensors`). What comes out: the [`GemmaConfig`] and the
//! bytes of every tensor of the language model by its full name
//! (`model.language_model.layers.3.self_attn.q_proj.weight`), lent from the
//! mappings. Text only: the vision and audio towers are not read (their
//! configurations are parsed, as in herbert-vk-bf16). Every tensor the engine
//! uses is bf16: another dtype is refused by name.
//!
//! The cache of the quantized weights ([`Q4_CACHE`], written by
//! `herbert-metal-q4cache`): a safetensors file next to `config.json` holding
//! every projection already in Q4_0 (`q4.<key>`, U8, its geometry in the
//! metadata) and every bf16 tensor the engine reads besides. When it is there
//! the engine takes its weights from it; the bf16 shards may then be absent.

use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// RoPE of one kind of layer: base, and how much of the head rotates.
#[derive(Debug, Clone, Copy)]
pub struct RopeKind {
    pub theta: f64,
    pub partial: f64,
}

/// The vision tower and its embedder (`vision_config`), when the checkpoint has one.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    pub clipped_linears: bool,
    pub standardize: bool,
    /// `<|image|>`, `<|image>`, `<image|>`.
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
}

/// The audio tower (`audio_config`): a Conformer of chunked local attention
/// behind two strided convolutions, and the ids of its three tokens.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub output_proj_dims: usize,
    pub chunk_size: usize,
    pub context_left: usize,
    pub context_right: usize,
    pub conv_kernel_size: usize,
    pub conv_channels: [usize; 2],
    pub logit_cap: f32,
    pub residual_weight: f32,
    pub gradient_clipping: f32,
    pub rms_norm_eps: f32,
    pub clipped_linears: bool,
    /// `<|audio|>`, `<|audio>`, `<audio|>`.
    pub audio_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_id: u32,
}

/// The mixture of experts of 26B-A4B (`enable_moe_block`): in every layer,
/// next to the dense MLP, `num_experts` GeGLU experts of `inter` rows, the
/// `top_k` best of the router's softmax applied to each token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub inter: usize,
}

#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// K/V heads of the sliding layers.
    pub num_kv_heads: usize,
    /// K/V heads of the full-attention layers (`num_global_key_value_heads`,
    /// else `num_key_value_heads`): 12B has 1 against 8.
    pub num_global_kv_heads: usize,
    /// `attention_k_eq_v`: on the full-attention layers the checkpoint has no
    /// `v_proj`, the value is the raw output of `k_proj` (before `k_norm` and
    /// RoPE) through the weightless v norm (`modeling_gemma4`'s
    /// `value_states = key_states` when `v_proj is None`). 12B.
    pub k_eq_v: bool,
    /// Head width of the sliding layers.
    pub head_dim: usize,
    /// Head width of the full-attention layers.
    pub global_head_dim: usize,
    pub intermediate_size: usize,
    /// `hidden_size_per_layer_input`: width of the per-layer input branch;
    /// 0 when the model has none (12B, `gemma4_unified_text`: the decoder layer
    /// is Gemma 2's, the MLP output and residual then the layer scalar).
    pub ple_dim: usize,
    pub rms_norm_eps: f32,
    pub softcap: f32,
    pub sliding_window: usize,
    /// One flag per layer: sliding window (true) or full attention.
    pub sliding: Vec<bool>,
    /// The last `num_kv_shared_layers` layers read the K/V of an earlier layer.
    pub num_kv_shared_layers: usize,
    pub double_wide_mlp: bool,
    /// The expert block (26B-A4B), none for a dense model.
    pub moe: Option<MoeConfig>,
    pub sliding_rope: RopeKind,
    pub full_rope: RopeKind,
    pub max_position_embeddings: usize,
    pub eos: Vec<u32>,
    pub pad_token_id: u32,
    pub vision: Option<VisionConfig>,
    pub audio: Option<AudioConfig>,
}

impl GemmaConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    pub fn from_json(content: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(content).map_err(|e| HerbertError::Config(format!("config.json: {e}")))?;
        let t = root.get("text_config").unwrap_or(&root);
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let uz = |key: &str| t.get(key).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| missing(key));
        let model_type = t.get("model_type").and_then(Value::as_str).ok_or_else(|| missing("model_type"))?.to_string();
        if !model_type.starts_with("gemma4") {
            return Err(HerbertError::Config(format!("model_type {model_type:?}: not a gemma4 text model")));
        }
        let moe = if t.get("enable_moe_block").and_then(Value::as_bool).unwrap_or(false) {
            Some(MoeConfig { num_experts: uz("num_experts")?, top_k: uz("top_k_experts")?, inter: uz("moe_intermediate_size")? })
        } else {
            None
        };
        let num_layers = uz("num_hidden_layers")?;
        let sliding: Vec<bool> = t
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| missing("layer_types"))?
            .iter()
            .map(|v| match v.as_str() {
                Some("sliding_attention") => Ok(true),
                Some("full_attention") => Ok(false),
                other => Err(HerbertError::Config(format!("config.json: layer_types holds {other:?}"))),
            })
            .collect::<Result<_>>()?;
        if sliding.len() != num_layers {
            return Err(HerbertError::Config("layer_types does not have one entry per layer".into()));
        }
        let rope = |kind: &str| -> Result<RopeKind> {
            let r = t.get("rope_parameters").and_then(|r| r.get(kind)).ok_or_else(|| missing("rope_parameters"))?;
            Ok(RopeKind {
                theta: r.get("rope_theta").and_then(Value::as_f64).ok_or_else(|| missing("rope_theta"))?,
                partial: r.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0),
            })
        };
        let eos = match root.get("eos_token_id").or_else(|| t.get("eos_token_id")) {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|n| n as u32).collect(),
            Some(v) => v.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
            None => Vec::new(),
        };
        // The unified towers of 12B (`gemma4_unified_vision`, `gemma4_unified_audio`)
        // have another layout; this engine reads text only and leaves them unparsed.
        let tower = |key: &str| root.get(key).filter(|v| v.is_object() && !v.get("model_type").and_then(Value::as_str).unwrap_or("").starts_with("gemma4_unified"));
        let vision = tower("vision_config").map(|v| -> Result<VisionConfig> {
            let u = |key: &str| v.get(key).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| missing(key));
            let id = |key: &str| root.get(key).and_then(Value::as_u64).map(|n| n as u32).ok_or_else(|| missing(key));
            Ok(VisionConfig {
                hidden_size: u("hidden_size")?,
                num_layers: u("num_hidden_layers")?,
                num_heads: u("num_attention_heads")?,
                num_kv_heads: u("num_key_value_heads")?,
                head_dim: u("head_dim")?,
                intermediate_size: u("intermediate_size")?,
                patch_size: u("patch_size")?,
                pooling_kernel_size: u("pooling_kernel_size")?,
                position_embedding_size: u("position_embedding_size")?,
                rope_theta: v.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Value::as_f64).ok_or_else(|| missing("vision rope_theta"))?,
                rms_norm_eps: v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
                clipped_linears: v.get("use_clipped_linears").and_then(Value::as_bool).unwrap_or(false),
                standardize: v.get("standardize").and_then(Value::as_bool).unwrap_or(false),
                image_token_id: id("image_token_id")?,
                boi_token_id: id("boi_token_id")?,
                eoi_token_id: id("eoi_token_id")?,
            })
        });
        let vision = match vision {
            Some(v) => Some(v?),
            None => None,
        };
        let audio = tower("audio_config").map(|a| -> Result<AudioConfig> {
            let u = |key: &str| a.get(key).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| missing(key));
            let fl = |key: &str| a.get(key).and_then(Value::as_f64).map(|n| n as f32).ok_or_else(|| missing(key));
            let id = |key: &str| root.get(key).and_then(Value::as_u64).map(|n| n as u32).ok_or_else(|| missing(key));
            let ch = a.get("subsampling_conv_channels").and_then(Value::as_array).ok_or_else(|| missing("subsampling_conv_channels"))?;
            let ch = |i: usize| ch.get(i).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| missing("subsampling_conv_channels"));
            Ok(AudioConfig {
                hidden_size: u("hidden_size")?,
                num_layers: u("num_hidden_layers")?,
                num_heads: u("num_attention_heads")?,
                output_proj_dims: u("output_proj_dims")?,
                chunk_size: u("attention_chunk_size")?,
                context_left: u("attention_context_left")?,
                context_right: u("attention_context_right")?,
                conv_kernel_size: u("conv_kernel_size")?,
                conv_channels: [ch(0)?, ch(1)?],
                logit_cap: fl("attention_logit_cap")?,
                residual_weight: fl("residual_weight")?,
                gradient_clipping: fl("gradient_clipping")?,
                rms_norm_eps: a.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
                clipped_linears: a.get("use_clipped_linears").and_then(Value::as_bool).unwrap_or(false),
                audio_token_id: id("audio_token_id")?,
                boa_token_id: id("boa_token_id")?,
                eoa_token_id: id("eoa_token_id")?,
            })
        });
        let audio = match audio {
            Some(a) => Some(a?),
            None => None,
        };
        let c = GemmaConfig {
            audio,
            pad_token_id: t.get("pad_token_id").and_then(Value::as_u64).unwrap_or(0) as u32,
            vision,
            model_type,
            vocab_size: uz("vocab_size")?,
            hidden_size: uz("hidden_size")?,
            num_layers,
            num_heads: uz("num_attention_heads")?,
            num_kv_heads: uz("num_key_value_heads")?,
            num_global_kv_heads: t.get("num_global_key_value_heads").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(uz("num_key_value_heads")?),
            k_eq_v: t.get("attention_k_eq_v").and_then(Value::as_bool).unwrap_or(false),
            head_dim: uz("head_dim")?,
            global_head_dim: t.get("global_head_dim").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(uz("head_dim")?),
            intermediate_size: uz("intermediate_size")?,
            ple_dim: t.get("hidden_size_per_layer_input").and_then(Value::as_u64).unwrap_or(0) as usize,
            rms_norm_eps: t.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            softcap: t.get("final_logit_softcapping").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            sliding_window: uz("sliding_window")?,
            sliding,
            num_kv_shared_layers: t.get("num_kv_shared_layers").and_then(Value::as_u64).unwrap_or(0) as usize,
            double_wide_mlp: t.get("use_double_wide_mlp").and_then(Value::as_bool).unwrap_or(false),
            moe,
            sliding_rope: rope("sliding_attention")?,
            full_rope: rope("full_attention")?,
            max_position_embeddings: uz("max_position_embeddings")?,
            eos,
        };
        if c.num_kv_heads == 0
            || c.num_global_kv_heads == 0
            || !c.num_heads.is_multiple_of(c.num_kv_heads)
            || !c.num_heads.is_multiple_of(c.num_global_kv_heads)
            || c.num_kv_shared_layers >= c.num_layers
            || c.moe.is_some_and(|m| m.top_k == 0 || m.top_k > m.num_experts || m.num_experts > 1024)
        {
            return Err(HerbertError::Config("inconsistent head counts or shared layers".into()));
        }
        Ok(c)
    }

    /// Width of the heads of layer `l`.
    pub fn head_dim_of(&self, l: usize) -> usize {
        if self.sliding[l] {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    /// K/V heads of layer `l`.
    pub fn kv_heads_of(&self, l: usize) -> usize {
        if self.sliding[l] {
            self.num_kv_heads
        } else {
            self.num_global_kv_heads
        }
    }

    /// Layer `l`'s value is its raw key (`attention_k_eq_v`, full-attention layers only).
    pub fn k_eq_v_of(&self, l: usize) -> bool {
        self.k_eq_v && !self.sliding[l]
    }

    /// Width of layer `l`'s fused projection: `[q | k | v]`, `[q | k]` when
    /// the value is the key, `q` alone when the layer reads another's K/V.
    pub fn qkv_width_of(&self, l: usize) -> usize {
        let hd = self.head_dim_of(l);
        let (qw, kw) = (self.num_heads * hd, self.kv_heads_of(l) * hd);
        if self.kv_source(l) != l {
            qw
        } else if self.k_eq_v_of(l) {
            qw + kw
        } else {
            qw + 2 * kw
        }
    }

    /// The first layer that reads another layer's K/V.
    pub fn first_shared(&self) -> usize {
        self.num_layers - self.num_kv_shared_layers
    }

    /// The layer whose K/V cache layer `l` uses: itself, or the last
    /// non-shared layer of its own kind.
    pub fn kv_source(&self, l: usize) -> usize {
        let first = self.first_shared();
        if l < first || self.num_kv_shared_layers == 0 {
            return l;
        }
        (0..first).rev().find(|&s| self.sliding[s] == self.sliding[l]).expect("a non-shared layer of the same kind")
    }

    /// Rows of the MLP of layer `l`.
    pub fn inter_of(&self, l: usize) -> usize {
        if self.double_wide_mlp && self.num_kv_shared_layers > 0 && l >= self.first_shared() {
            2 * self.intermediate_size
        } else {
            self.intermediate_size
        }
    }
}

struct Entry {
    map: usize,
    offset: usize,
    bytes: usize,
}

/// The file name of the cache of quantized weights in a model directory.
pub const Q4_CACHE: &str = "herbert-q4.safetensors";
/// The format the cache declares (`__metadata__["format"]`): the quantizer and its layout.
pub const Q4_FORMAT: &str = "herbert-metal-q4 Q4_0 v1: blocks of 32 along k, uint4 nibbles then f16 scales";

/// A projection of the cache: its `[n][k]` geometry and where its scales start.
struct Q4Entry {
    at: Entry,
    n: usize,
    k: usize,
    scales: u64,
}

pub struct GemmaCheckpoint {
    pub dir: PathBuf,
    pub config: GemmaConfig,
    maps: Vec<Mmap>,
    entries: HashMap<String, Entry>,
    q4: HashMap<String, Q4Entry>,
}

const PREFIX: &str = "model.language_model.";
/// What is read: the language model.
const KEPT: [&str; 1] = ["model.language_model."];

impl GemmaCheckpoint {
    pub fn load(model_dir: &Path) -> Result<GemmaCheckpoint> {
        let dir = super::resolve_model_dir(model_dir)?;
        let config = GemmaConfig::from_file(&dir.join("config.json"))?;
        let index_path = dir.join("model.safetensors.index.json");
        let cache = dir.join(Q4_CACHE);
        let mut shards: Vec<String> = if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&std::fs::read(&index_path)?)?;
            let mut v: Vec<String> = index.weight_map.values().cloned().collect();
            v.sort();
            v.dedup();
            v
        } else if dir.join("model.safetensors").exists() || !cache.exists() {
            vec!["model.safetensors".to_string()]
        } else {
            // The cache alone: the bf16 shards are not on this machine.
            Vec::new()
        };
        shards.dedup();
        let mut maps = Vec::new();
        let mut entries = HashMap::new();
        for shard in &shards {
            let file = std::fs::File::open(dir.join(shard)).map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            // SAFETY: read-only mapping of a file nobody modifies meanwhile.
            let mapped = unsafe { Mmap::map(&file) }.map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            let base = mapped.as_ptr() as usize;
            {
                let st = SafeTensors::deserialize(&mapped[..])?;
                for (name, view) in st.tensors() {
                    if !KEPT.iter().any(|p| name.starts_with(p)) {
                        continue;
                    }
                    if view.dtype() != safetensors::Dtype::BF16 {
                        return Err(HerbertError::ModelLoad(format!("{name}: {:?}, bf16 expected", view.dtype())));
                    }
                    let data = view.data();
                    entries.insert(name.to_string(), Entry { map: maps.len(), offset: data.as_ptr() as usize - base, bytes: data.len() });
                }
            }
            maps.push(mapped);
        }
        let mut q4 = HashMap::new();
        if cache.exists() {
            let file = std::fs::File::open(&cache).map_err(|e| HerbertError::ModelLoad(format!("{Q4_CACHE}: {e}")))?;
            // SAFETY: read-only mapping of a file nobody modifies meanwhile.
            let mapped = unsafe { Mmap::map(&file) }.map_err(|e| HerbertError::ModelLoad(format!("{Q4_CACHE}: {e}")))?;
            let base = mapped.as_ptr() as usize;
            {
                let (_, meta) = SafeTensors::read_metadata(&mapped[..])?;
                let meta = meta.metadata().clone().unwrap_or_default();
                if meta.get("format").map(String::as_str) != Some(Q4_FORMAT) {
                    return Err(HerbertError::ModelLoad(format!("{Q4_CACHE}: format {:?}, {Q4_FORMAT:?} expected", meta.get("format"))));
                }
                let st = SafeTensors::deserialize(&mapped[..])?;
                for (name, view) in st.tensors() {
                    let data = view.data();
                    let at = Entry { map: maps.len(), offset: data.as_ptr() as usize - base, bytes: data.len() };
                    if let Some(key) = name.strip_prefix("q4.") {
                        let geo = meta.get(&name).ok_or_else(|| HerbertError::ModelLoad(format!("{Q4_CACHE}: no geometry for {name}")))?;
                        let v: Vec<u64> = geo.split(',').map(|x| x.trim().parse::<u64>()).collect::<std::result::Result<_, _>>().map_err(|_| HerbertError::ModelLoad(format!("{Q4_CACHE}: geometry {geo:?} of {name}")))?;
                        let [n, k, scales] = v[..] else { return Err(HerbertError::ModelLoad(format!("{Q4_CACHE}: geometry {geo:?} of {name}"))) };
                        q4.insert(key.to_string(), Q4Entry { at, n: n as usize, k: k as usize, scales });
                    } else if view.dtype() == safetensors::Dtype::BF16 {
                        // The shards win where both have a tensor.
                        entries.entry(name.to_string()).or_insert(at);
                    } else {
                        return Err(HerbertError::ModelLoad(format!("{Q4_CACHE}: {name}: {:?}", view.dtype())));
                    }
                }
            }
            maps.push(mapped);
        }
        Ok(GemmaCheckpoint { dir, config, maps, entries, q4 })
    }

    /// A projection of the cache by its key (its parts' names joined by `+`):
    /// the Q4_0 bytes and the offset of the scales, checked against `[n][k]`.
    pub fn q4(&self, key: &str, n: usize, k: usize) -> Result<Option<(&[u8], u64)>> {
        let Some(e) = self.q4.get(key) else { return Ok(None) };
        if (e.n, e.k) != (n, k) {
            return Err(HerbertError::ModelLoad(format!("{Q4_CACHE}: {key} is {}x{}, {n}x{k} expected", e.n, e.k)));
        }
        Ok(Some((&self.maps[e.at.map][e.at.offset..e.at.offset + e.at.bytes], e.scales)))
    }

    /// Whether the directory has a cache of quantized weights.
    pub fn has_q4_cache(&self) -> bool {
        !self.q4.is_empty()
    }

    /// The bytes of a tensor under `model.language_model.`, whatever its length.
    pub fn bytes_any(&self, name: &str) -> Result<&[u8]> {
        let full = format!("{PREFIX}{name}");
        let e = self.entries.get(&full).ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {full}")))?;
        Ok(&self.maps[e.map][e.offset..e.offset + e.bytes])
    }

    /// The names (under `model.language_model.`) of the bf16 tensors read, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.entries.keys().filter_map(|n| n.strip_prefix(PREFIX).map(str::to_string)).collect();
        v.sort();
        v
    }

    /// The bytes (bf16, little-endian) of a tensor, by its name under
    /// `model.language_model.`, checked against `len` elements.
    pub fn bytes(&self, name: &str, len: usize) -> Result<&[u8]> {
        self.bytes_full(&format!("{PREFIX}{name}"), len)
    }

    /// The same by the full name (`model.vision_tower.…`).
    pub fn bytes_full(&self, name: &str, len: usize) -> Result<&[u8]> {
        let e = self.entries.get(name).ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {name}")))?;
        if e.bytes != 2 * len {
            return Err(HerbertError::ModelLoad(format!("{name}: {} elements, {len} expected", e.bytes / 2)));
        }
        Ok(&self.maps[e.map][e.offset..e.offset + e.bytes])
    }

    pub fn has_full(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Row `row` of a `[rows][width]` table.
    pub fn row(&self, name: &str, row: usize, width: usize) -> Result<&[u8]> {
        let e = self.entries.get(&format!("{PREFIX}{name}")).ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {name}")))?;
        let (lo, hi) = (2 * row * width, 2 * (row + 1) * width);
        if hi > e.bytes {
            return Err(HerbertError::ModelLoad(format!("{name}: row {row} past the end")));
        }
        Ok(&self.maps[e.map][e.offset + lo..e.offset + hi])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text configuration of a Gemma-4 checkpoint, every layer `sliding`
    /// except each sixth, with the given extra keys.
    fn config(extra: &str, towers: &str) -> String {
        let types: Vec<&str> = (0..12).map(|l| if l % 6 == 5 { "\"full_attention\"" } else { "\"sliding_attention\"" }).collect();
        format!(
            r#"{{"eos_token_id": [1, 106], {towers}
              "text_config": {{"model_type": "gemma4_unified_text", "vocab_size": 262144, "hidden_size": 3840,
                "num_hidden_layers": 12, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
                "global_head_dim": 512, "intermediate_size": 15360, "sliding_window": 1024,
                "max_position_embeddings": 262144, "final_logit_softcapping": 30.0, "layer_types": [{}],
                "rope_parameters": {{"full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
                                    "sliding_attention": {{"rope_theta": 10000.0}}}} {extra}}}}}"#,
            types.join(", ")
        )
    }

    /// 12B (`~/models/gemma-4-12B-it-qat-q4_0-unquantized`): no per-layer
    /// branch, 1 K/V head of 512 on the full layers, the value is the key there.
    #[test]
    fn the_12b_geometry_is_read() {
        let towers = r#""vision_config": {"model_type": "gemma4_unified_vision", "mm_embed_dim": 3840},
                        "audio_config": {"model_type": "gemma4_unified_audio", "audio_embed_dim": 640},"#;
        let extra = r#", "attention_k_eq_v": true, "num_global_key_value_heads": 1, "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0"#;
        let c = GemmaConfig::from_json(&config(extra, towers)).expect("the 12B configuration");
        assert!(c.vision.is_none() && c.audio.is_none(), "the unified towers are left unparsed");
        assert_eq!((c.ple_dim, c.num_kv_shared_layers, c.sliding_window), (0, 0, 1024));
        assert_eq!((c.kv_heads_of(0), c.kv_heads_of(5)), (8, 1));
        assert_eq!((c.k_eq_v_of(0), c.k_eq_v_of(5)), (false, true));
        // Sliding: [q 16·256 | k 8·256 | v 8·256]; full: [q 16·512 | k 1·512], no v.
        assert_eq!((c.qkv_width_of(0), c.qkv_width_of(5)), (4096 + 2 * 2048, 8192 + 512));
        assert!((0..12).all(|l| c.kv_source(l) == l));
    }

    /// Without `num_global_key_value_heads` (E2B, E4B) the full layers keep
    /// `num_key_value_heads`, and without `attention_k_eq_v` every layer has its v.
    #[test]
    fn without_the_12b_keys_the_full_layers_keep_their_value() {
        let c = GemmaConfig::from_json(&config(r#", "hidden_size_per_layer_input": 256"#, "")).expect("an E4B-like configuration");
        assert_eq!((c.num_global_kv_heads, c.k_eq_v, c.ple_dim), (8, false, 256));
        assert_eq!(c.qkv_width_of(5), 8192 + 2 * 8 * 512);
    }

    /// 26B-A4B: the expert block read next to the 12B geometry (2 K/V heads
    /// of 512 on the full layers); without its keys the block is refused.
    #[test]
    fn the_26b_expert_block_is_read() {
        let extra = r#", "attention_k_eq_v": true, "num_global_key_value_heads": 2, "hidden_size_per_layer_input": 0,
                       "enable_moe_block": true, "num_experts": 128, "top_k_experts": 8, "moe_intermediate_size": 704"#;
        let c = GemmaConfig::from_json(&config(extra, "")).expect("the 26B-A4B configuration");
        assert_eq!(c.moe, Some(MoeConfig { num_experts: 128, top_k: 8, inter: 704 }));
        assert_eq!((c.kv_heads_of(0), c.kv_heads_of(5)), (8, 2));
        assert_eq!(c.qkv_width_of(5), 8192 + 2 * 512);
        assert!(GemmaConfig::from_json(&config(r#", "enable_moe_block": true"#, "")).is_err());
        assert!(GemmaConfig::from_json(&config(r#", "enable_moe_block": true, "num_experts": 4, "top_k_experts": 8, "moe_intermediate_size": 64"#, "")).is_err());
        assert_eq!(GemmaConfig::from_json(&config("", "")).expect("dense").moe, None);
    }
}
