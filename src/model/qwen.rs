//! The Qwen3.5 checkpoint, read in place — text only.
//!
//! What enters: the model directory (`config.json`, the `*.safetensors`
//! shards). What comes out: the `Config` and every tensor the engine uses,
//! under a normalized name (`embed_tokens`, `norm`, `lm_head`,
//! `<layer>.<field>` such as `3.self_attn.q_proj.weight`), in bf16 row-major —
//! the checkpoint's bytes — or in f32 for the two DeltaNet vectors kept in f32
//! (`A_log`, `dt_bias`). The multi-token-prediction head (`mtp.*`) and the
//! vision tower are skipped. herbert-vk-bf16's `model/mod.rs`.
//!
//! The shards are mapped, not copied: a bf16 tensor is the checkpoint's bytes,
//! lent as they are to the upload; an f16 or f32 one is converted to bf16 when
//! asked.
//!
//! The cache of quantized weights (`model::gemma::Q4_CACHE`, the same format
//! as Gemma's, written by `herbert-metal-q4cache`): its `q4.<key>` entries
//! (the projection of the parts `key` names, joined by `+`, as the engine
//! uploads it, `n,k,scales` in the metadata) are read instead of quantizing,
//! its other tensors under their checkpoint names; a directory may hold the
//! cache alone.

use super::resolve_model_dir;
use crate::core::bf16::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_h16, Bf16, F16};
use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// The element type of a tensor's bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Bf16,
    F16,
    F32,
}

impl Kind {
    fn size(self) -> usize {
        if self == Kind::F32 {
            4
        } else {
            2
        }
    }
}

struct Entry {
    shape: Vec<usize>,
    map: usize,
    offset: usize,
    bytes: usize,
    kind: Kind,
}

impl Entry {
    fn elements(&self) -> usize {
        self.bytes / self.kind.size()
    }
}

struct Q4Entry {
    at: Entry,
    n: usize,
    k: usize,
    scales: u64,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub config: Config,
    maps: Vec<Mmap>,
    /// The shard of each mapping, for [`Checkpoint::read_bf16_into`].
    files: Vec<PathBuf>,
    entries: HashMap<String, Entry>,
    q4: HashMap<String, Q4Entry>,
}

/// A tensor's name in the cache from the engine's key ([`normalize`] gives it back).
fn cache_name(key: &str) -> String {
    match key {
        "embed_tokens" | "norm" => format!("model.language_model.{key}.weight"),
        "lm_head" => "lm_head.weight".into(),
        _ => format!("model.language_model.layers.{key}"),
    }
}

/// One checkpoint element as bf16, whatever its type.
#[inline]
fn bf16_at(src: &[u8], kind: Kind, i: usize) -> Bf16 {
    match kind {
        Kind::Bf16 => u16::from_le_bytes([src[2 * i], src[2 * i + 1]]),
        Kind::F16 => f32_to_bf16(f16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))),
        Kind::F32 => f32_to_bf16(f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]])),
    }
}

/// The name of a tensor as the engine knows it, `None` when the engine does
/// not use it; and whether it is kept in f32.
fn normalize(name: &str) -> Option<(String, bool)> {
    let rest = name
        .strip_prefix("language_model.model.")
        .or_else(|| name.strip_prefix("model.language_model."))
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);
    match rest {
        "embed_tokens.weight" => return Some(("embed_tokens".into(), false)),
        "norm.weight" => return Some(("norm".into(), false)),
        "lm_head.weight" => return Some(("lm_head".into(), false)),
        _ => {}
    }
    if name == "lm_head.weight" {
        return Some(("lm_head".into(), false));
    }
    let rest = rest.strip_prefix("layers.")?;
    let (index, field) = rest.split_once('.')?;
    let layer: usize = index.parse().ok()?;
    let f32_kept = matches!(field, "linear_attn.A_log" | "linear_attn.dt_bias");
    let used = matches!(
        field,
        "input_layernorm.weight"
            | "post_attention_layernorm.weight"
            | "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.o_proj.weight"
            | "self_attn.q_norm.weight"
            | "self_attn.k_norm.weight"
            | "mlp.gate_proj.weight"
            | "mlp.up_proj.weight"
            | "mlp.down_proj.weight"
            | "linear_attn.in_proj_qkv.weight"
            | "linear_attn.in_proj_z.weight"
            | "linear_attn.in_proj_a.weight"
            | "linear_attn.in_proj_b.weight"
            | "linear_attn.out_proj.weight"
            | "linear_attn.conv1d.weight"
            | "linear_attn.A_log"
            | "linear_attn.dt_bias"
            | "linear_attn.norm.weight"
            | "mlp.gate.weight"
            | "mlp.experts.gate_up_proj"
            | "mlp.experts.down_proj"
            | "mlp.shared_expert.gate_proj.weight"
            | "mlp.shared_expert.up_proj.weight"
            | "mlp.shared_expert.down_proj.weight"
            | "mlp.shared_expert_gate.weight"
    );
    used.then(|| (format!("{layer}.{field}"), f32_kept))
}

impl Checkpoint {
    pub fn load(model_dir: &Path) -> Result<Checkpoint> {
        let dir = resolve_model_dir(model_dir)?;
        let config = Config::from_file(&dir.join("config.json"))?;
        let index_path = dir.join("model.safetensors.index.json");
        let cache = dir.join(super::gemma::Q4_CACHE);
        let mut shards: Vec<String> = if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&std::fs::read(&index_path)?)?;
            let set: HashSet<String> = index.weight_map.values().cloned().collect();
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            v
        } else if dir.join("model.safetensors").exists() || !cache.exists() {
            vec!["model.safetensors".to_string()]
        } else {
            // The cache alone.
            Vec::new()
        };
        let has_cache = cache.exists();
        if has_cache {
            shards.push(super::gemma::Q4_CACHE.to_string());
        }
        let mut maps = Vec::new();
        let mut files = Vec::new();
        let mut entries = HashMap::new();
        let mut q4 = HashMap::new();
        let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
        for shard in &shards {
            let is_cache = has_cache && shard == super::gemma::Q4_CACHE;
            let file = std::fs::File::open(dir.join(shard)).map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            // SAFETY: read-only mapping of a file nobody modifies meanwhile.
            let mapped = unsafe { Mmap::map(&file) }.map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            let base = mapped.as_ptr() as usize;
            {
                let meta = if is_cache {
                    let (_, meta) = SafeTensors::read_metadata(&mapped[..])?;
                    let meta = meta.metadata().clone().unwrap_or_default();
                    if meta.get("format").map(String::as_str) != Some(super::gemma::Q4_FORMAT) {
                        return Err(HerbertError::ModelLoad(format!("{shard}: format {:?}, {:?} expected", meta.get("format"), super::gemma::Q4_FORMAT)));
                    }
                    meta
                } else {
                    HashMap::new()
                };
                let st = SafeTensors::deserialize(&mapped[..])?;
                for (name, view) in st.tensors() {
                    if let Some(key) = name.strip_prefix("q4.").filter(|_| is_cache) {
                        let data = view.data();
                        let at = Entry { shape: view.shape().to_vec(), map: maps.len(), offset: data.as_ptr() as usize - base, bytes: data.len(), kind: Kind::Bf16 };
                        let geo = meta.get(&name).ok_or_else(|| HerbertError::ModelLoad(format!("{shard}: no geometry for {name}")))?;
                        let v: Vec<u64> = geo.split(',').map(|x| x.trim().parse::<u64>()).collect::<std::result::Result<_, _>>().map_err(|_| HerbertError::ModelLoad(format!("{shard}: geometry {geo:?} of {name}")))?;
                        let [n, k, scales] = v[..] else { return Err(HerbertError::ModelLoad(format!("{shard}: geometry {geo:?} of {name}"))) };
                        q4.insert(key.to_string(), Q4Entry { at, n: n as usize, k: k as usize, scales });
                        continue;
                    }
                    let Some((key, _)) = normalize(&name) else {
                        *skipped.entry(name.split('.').take(2).collect::<Vec<_>>().join(".")).or_default() += 1;
                        continue;
                    };
                    let kind = match view.dtype() {
                        safetensors::Dtype::BF16 => Kind::Bf16,
                        safetensors::Dtype::F16 => Kind::F16,
                        safetensors::Dtype::F32 => Kind::F32,
                        d => return Err(HerbertError::ModelLoad(format!("unsupported dtype {d:?} for {name}"))),
                    };
                    let data = view.data();
                    let e = Entry { shape: view.shape().to_vec(), map: maps.len(), offset: data.as_ptr() as usize - base, bytes: data.len(), kind };
                    // The shards win where both have a tensor.
                    entries.entry(key).or_insert(e);
                }
            }
            maps.push(mapped);
            files.push(dir.join(shard));
        }
        if !skipped.is_empty() {
            let list: Vec<String> = skipped.iter().map(|(p, n)| format!("{p}.* ×{n}")).collect();
            eprintln!("[loader] skipped: {}", list.join(", "));
        }
        Ok(Checkpoint { dir, config, maps, files, entries, q4 })
    }

    /// A projection of the cache by its key (its parts' names joined by `+`):
    /// its bytes and the offset of its scales, checked against `[n][k]`.
    pub fn q4(&self, key: &str, n: usize, k: usize) -> Result<Option<(usize, u64)>> {
        let Some(e) = self.q4.get(key) else { return Ok(None) };
        if (e.n, e.k) != (n, k) {
            return Err(HerbertError::ModelLoad(format!("cache: {key} is {}x{}, {n}x{k} expected", e.n, e.k)));
        }
        Ok(Some((e.at.bytes, e.scales)))
    }

    /// The bytes of the cache's projection `key` read into `dst`, page cache off.
    pub fn read_q4_into(&self, key: &str, dst: &mut [u8]) -> Result<()> {
        let e = self.q4.get(key).ok_or_else(|| HerbertError::ModelLoad(format!("cache: no {key}")))?;
        self.read_entry_into(&e.at, dst)
    }

    /// The tensors the cache writes beside the projections: every one read
    /// except `parts`, with its cache name, safetensors dtype and bytes.
    pub fn others(&self, parts: &HashSet<String>) -> Vec<(String, &'static str, &[u8])> {
        let mut keys: Vec<&String> = self.entries.keys().filter(|k| !parts.contains(*k)).collect();
        keys.sort();
        keys.into_iter()
            .map(|k| {
                let e = &self.entries[k];
                let dtype = match e.kind {
                    Kind::Bf16 => "BF16",
                    Kind::F16 => "F16",
                    Kind::F32 => "F32",
                };
                (cache_name(k), dtype, &self.maps[e.map][e.offset..e.offset + e.bytes])
            })
            .collect()
    }

    fn read_entry_into(&self, e: &Entry, dst: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        use std::os::unix::io::AsRawFd;
        let path = &self.files[e.map];
        let file = std::fs::File::open(path).map_err(|err| HerbertError::ModelLoad(format!("{}: {err}", path.display())))?;
        // SAFETY: fcntl on an open descriptor; F_NOCACHE only changes caching.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        const CHUNK: usize = 64 << 20;
        let mut done = 0;
        while done < e.bytes {
            let n = CHUNK.min(e.bytes - done);
            file.read_exact_at(&mut dst[done..done + n], (e.offset + done) as u64).map_err(|err| HerbertError::ModelLoad(format!("{}: {err}", path.display())))?;
            done += n;
        }
        Ok(())
    }

    /// Whether the checkpoint has the tensor `key`, in bf16 or as a projection of the cache.
    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key) || self.q4.contains_key(key)
    }

    fn entry(&self, key: &str) -> Result<&Entry> {
        self.entries.get(key).ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {key}")))
    }

    /// A tensor in bf16, checked against `len` elements: lent from the mapping
    /// when its bytes are bf16.
    pub fn bf16(&self, key: &str, len: usize) -> Result<Cow<'_, [Bf16]>> {
        let e = self.entry(key)?;
        if e.elements() != len {
            return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {len} expected ({:?})", e.elements(), e.shape)));
        }
        let bytes = &self.maps[e.map][e.offset..e.offset + e.bytes];
        if e.kind == Kind::Bf16 && (bytes.as_ptr() as usize).is_multiple_of(2) && cfg!(target_endian = "little") {
            // SAFETY: aligned, little-endian u16 in a mapping that lives as long as `self`.
            return Ok(Cow::Borrowed(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Bf16, len) }));
        }
        Ok(Cow::Owned((0..len).map(|i| bf16_at(bytes, e.kind, i)).collect()))
    }

    /// A tensor in the engine's 16-bit format (bf16, or f16 with the `f16`
    /// feature), each value rounded once from its stored type (nearest even).
    pub fn h16(&self, key: &str, len: usize) -> Result<Cow<'_, [u16]>> {
        if !F16 {
            return self.bf16(key, len);
        }
        let e = self.entry(key)?;
        if e.elements() != len {
            return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {len} expected ({:?})", e.elements(), e.shape)));
        }
        let b = &self.maps[e.map][e.offset..e.offset + e.bytes];
        Ok(Cow::Owned(match e.kind {
            Kind::F16 => b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect(),
            Kind::Bf16 => b.chunks_exact(2).map(|x| f32_to_h16(bf16_to_f32(u16::from_le_bytes([x[0], x[1]])))).collect(),
            Kind::F32 => b.chunks_exact(4).map(|x| f32_to_h16(f32::from_le_bytes([x[0], x[1], x[2], x[3]]))).collect(),
        }))
    }

    /// The bf16 tensors `parts` (name, elements), one after the other, read
    /// into `dst` (their concatenation) from the shard files with the page
    /// cache off (`F_NOCACHE`): a tensor goes to the GPU's buffer without also
    /// staying in memory as file cache — the two copies of a 67 GB checkpoint
    /// do not fit in 96 GB. `false` when one of them is not bf16 (the caller
    /// then copies from the mapping).
    pub fn read_bf16_into(&self, parts: &[(&str, usize)], dst: &mut [u8]) -> Result<bool> {
        let mut list = Vec::with_capacity(parts.len());
        for &(key, len) in parts {
            let e = self.entry(key)?;
            if e.elements() != len {
                return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {len} expected ({:?})", e.elements(), e.shape)));
            }
            if e.kind != Kind::Bf16 || !cfg!(target_endian = "little") {
                return Ok(false);
            }
            list.push(e);
        }
        let total: usize = list.iter().map(|e| e.bytes).sum();
        if dst.len() < total {
            return Err(HerbertError::Backend(format!("{} bytes to read into {}", total, dst.len())));
        }
        let mut at = 0;
        for e in list {
            self.read_entry_into(e, &mut dst[at..at + e.bytes])?;
            at += e.bytes;
        }
        Ok(true)
    }

    pub fn f32(&self, key: &str, len: usize) -> Result<Vec<f32>> {
        let e = self.entry(key)?;
        let b = &self.maps[e.map][e.offset..e.offset + e.bytes];
        let v: Vec<f32> = match e.kind {
            Kind::F32 => b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect(),
            Kind::F16 => b.chunks_exact(2).map(|x| f16_to_f32(u16::from_le_bytes([x[0], x[1]]))).collect(),
            Kind::Bf16 => b.chunks_exact(2).map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]]))).collect(),
        };
        if v.len() != len {
            return Err(HerbertError::ModelLoad(format!("{key}: {} elements, {len} expected ({:?})", v.len(), e.shape)));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::bf16::f16_to_f32;

    #[test]
    fn f16_widening() {
        for (h, f) in [(0x3c00u16, 1.0f32), (0x3800, 0.5), (0xc000, -2.0), (0x7bff, 65504.0), (0x0001, 2f32.powi(-24)), (0x03ff, 1023.0 / 1024.0 * 2f32.powi(-14)), (0x0400, 2f32.powi(-14)), (0x8000, -0.0)] {
            assert_eq!(f16_to_f32(h).to_bits(), f.to_bits(), "{h:#06x}");
        }
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
