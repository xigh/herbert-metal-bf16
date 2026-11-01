//! `--prefix-cache DIR`: the K/V of a request's stable prefix kept on disk, so
//! that a restarted server does not compute it again. The stable prefix is the
//! template's rendering of the system message (the first one, when it is
//! `system` or `developer`) and of the tools alone, tokenized, when those
//! tokens begin the request's: a client's instructions and tools, its periodic
//! agent's instructions. It changes only when the prompt does. And the
//! prefix a one-shot request (no assistant or tool message: a periodic
//! agent) shares with the previous one past its stable prefix, when that is at
//! least [`MIN_TOKENS`] more: the period agent's calendar, the same all day.
//! A request reads the longest file whose tokens begin its own.
//!
//! One file per prefix, `<key>-<tokens>.hkv`: the magic `HERBKV01`, the length
//! of a JSON header (u64 LE), the header (the fingerprint, the tokens, the
//! bytes of each region), then the K/V of positions `0..n` as the engine holds
//! them (bf16, [`GemmaEngine::prefix_write`]). The key hashes the fingerprint
//! (the format, the executable's bytes, the model directory and its files, the
//! weights' format, the template) and the tokens; a load checks the
//! fingerprint, the tokens and the layout before reading. Written to a
//! temporary file then renamed; the `keep` most recently used stable prefixes
//! and the [`KEEP_SHARED`] most recent shared ones stay, the others are
//! removed. Without the directory, nothing happens.

use crate::core::error::{HerbertError, Result};
use crate::metal::gemma::GemmaEngine;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAGIC: &[u8; 8] = b"HERBKV01";

/// A shorter stable prefix is not worth a file.
pub const MIN_TOKENS: usize = 64;

/// Shared prefixes kept (`.shared.hkv`): they change with the day, the stable
/// ones only with the prompt; kept apart, they never push a stable one out.
pub const KEEP_SHARED: usize = 2;

/// 64-bit FNV-1a: stable across builds and machines, enough for a file name
/// (the header holds what the key hashes and is checked).
fn fnv(h: u64, data: &[u8]) -> u64 {
    data.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}
const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

pub struct PrefixCache {
    dir: PathBuf,
    keep: usize,
    fingerprint: String,
    /// The files of this fingerprint and their tokens.
    entries: Vec<(PathBuf, Vec<u32>)>,
}

impl PrefixCache {
    /// The cache in `dir`, or none when `dir` is not a directory.
    pub fn open(dir: &Path, keep: usize, model_dir: &Path, q4: bool, template: &str) -> Result<Option<PrefixCache>> {
        if !dir.is_dir() {
            eprintln!("[prefix-cache] {} is not a directory: no prefix cache", dir.display());
            return Ok(None);
        }
        let exe = std::fs::read(std::env::current_exe()?)?;
        let model = model_dir.canonicalize()?;
        let mut files: Vec<String> = Vec::new();
        for e in std::fs::read_dir(&model)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".safetensors") || name.ends_with(".json") {
                let m = e.metadata()?;
                let mtime = m.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
                let sum = if m.len() < (64 << 20) { format!(" {:016x}", fnv(FNV0, &std::fs::read(e.path())?)) } else { String::new() };
                files.push(format!("{name} {} {mtime}{sum}", m.len()));
            }
        }
        files.sort();
        let fingerprint = format!(
            "herbert-metal-q4 prefix cache 1\nexecutable {} bytes {:016x}\nmodel {}\n{}\nweights {}\ntemplate {:016x}",
            exe.len(),
            fnv(FNV0, &exe),
            model.display(),
            files.join("\n"),
            if q4 { "q4_0" } else { "bf16" },
            fnv(FNV0, template.as_bytes())
        );
        let mut pc = PrefixCache { dir: dir.to_path_buf(), keep: keep.max(1), fingerprint, entries: Vec::new() };
        let files = pc.files();
        for (_, path) in &files {
            if let Ok(tokens) = pc.header(path) {
                pc.entries.push((path.clone(), tokens));
            }
        }
        let mut sizes: Vec<usize> = pc.entries.iter().map(|e| e.1.len()).collect();
        sizes.sort();
        eprintln!(
            "[prefix-cache] {}: {} files, {} of this executable, model and template ({sizes:?} tokens), keeps the {} most recently used stable prefixes and {KEEP_SHARED} shared ones, fingerprint {:016x}",
            dir.display(),
            files.len(),
            pc.entries.len(),
            pc.keep,
            fnv(FNV0, pc.fingerprint.as_bytes())
        );
        Ok(Some(pc))
    }

    fn path(&self, tokens: &[u32], shared: bool) -> PathBuf {
        let mut h = fnv(FNV0, self.fingerprint.as_bytes());
        for t in tokens {
            h = fnv(h, &t.to_le_bytes());
        }
        self.dir.join(format!("{h:016x}-{}{}.hkv", tokens.len(), if shared { ".shared" } else { "" }))
    }

    /// A file holds these tokens.
    pub fn has(&self, tokens: &[u32]) -> bool {
        self.entries.iter().any(|e| e.1 == tokens)
    }

    /// The tokens of the longest file that begin `ids`, past `above` tokens
    /// and short of the last one (its logits are computed): their number.
    pub fn best(&self, ids: &[u32], above: usize) -> Option<usize> {
        self.entries.iter().map(|e| e.1.len()).filter(|&n| n > above && n < ids.len()).filter(|&n| self.entries.iter().any(|e| e.1.len() == n && ids.starts_with(&e.1))).max()
    }

    /// The tokens of a file of this fingerprint (its header).
    fn header(&self, path: &Path) -> Result<Vec<u32>> {
        let mut f = std::fs::File::open(path)?;
        Ok(self.read_header(&mut f, path)?.0)
    }

    /// The header read and checked against the fingerprint: the tokens, the layout, the header's length.
    fn read_header(&self, f: &mut std::fs::File, path: &Path) -> Result<(Vec<u32>, Value, usize)> {
        let bad = |m: &str| HerbertError::Backend(format!("{}: {m}", path.display()));
        let mut head = [0u8; 16];
        f.read_exact(&mut head)?;
        if &head[..8] != MAGIC {
            return Err(bad("not a prefix cache file"));
        }
        let len = u64::from_le_bytes(head[8..].try_into().unwrap()) as usize;
        if len > (64 << 20) {
            return Err(bad("header too long"));
        }
        let mut h = vec![0u8; len];
        f.read_exact(&mut h)?;
        let mut h: Value = serde_json::from_slice(&h)?;
        if h["fingerprint"].as_str() != Some(self.fingerprint.as_str()) {
            return Err(bad("another executable, model or template"));
        }
        let tokens: Vec<u32> = serde_json::from_value(h["tokens"].take())?;
        Ok((tokens, h["layout"].take(), len))
    }

    /// The `.hkv` files and when they were last used, the most recent first.
    fn files(&self) -> Vec<(SystemTime, PathBuf)> {
        let mut v: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".hkv"))
            .map(|e| (e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH), e.path()))
            .collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v
    }

    /// The K/V of the file of `tokens` read into the engine (position
    /// `tokens.len()` after); the bytes read. The file is marked as just used.
    /// A file that does not match is removed; if the read fails midway, the
    /// engine holds nothing (position 0).
    pub fn load(&mut self, engine: &mut GemmaEngine, tokens: &[u32]) -> Result<u64> {
        let Some(i) = self.entries.iter().position(|e| e.1 == tokens) else { return Err(HerbertError::Backend("no file for these tokens".into())) };
        let path = self.entries[i].0.clone();
        let r = self.read(engine, tokens, &path);
        match &r {
            Ok(_) => {
                let _ = std::fs::File::options().write(true).open(&path).and_then(|f| f.set_modified(SystemTime::now()));
            }
            Err(_) => {
                let _ = std::fs::remove_file(&path);
                self.entries.remove(i);
            }
        }
        r
    }

    fn read(&self, engine: &mut GemmaEngine, tokens: &[u32], path: &Path) -> Result<u64> {
        let bad = |m: &str| HerbertError::Backend(format!("{}: {m}", path.display()));
        let mut f = std::fs::File::open(path)?;
        let (t, layout_h, len) = self.read_header(&mut f, path)?;
        if t != tokens {
            return Err(bad("other tokens"));
        }
        let layout = engine.prefix_layout(tokens.len());
        if layout_h != json!(layout) {
            return Err(bad("another layout of the caches"));
        }
        let bytes: u64 = layout.iter().map(|&b| b as u64).sum();
        if f.metadata()?.len() != 16 + len as u64 + bytes {
            return Err(bad("truncated"));
        }
        engine.prefix_read(tokens.len(), &mut f)?;
        Ok(bytes)
    }

    /// The K/V of `tokens` (positions `0..n` of the engine's live state)
    /// written to their file (`shared`: a prefix seen shared, not a stable
    /// one): a temporary one, renamed when complete. Then the files of its
    /// kind beyond the most recently used `keep` ([`KEEP_SHARED`]) are removed.
    /// The bytes.
    pub fn save(&mut self, engine: &GemmaEngine, tokens: &[u32], shared: bool) -> Result<u64> {
        let path = self.path(tokens, shared);
        let tmp = self.dir.join(format!(".{}.{}.tmp", path.file_name().unwrap().to_string_lossy(), std::process::id()));
        let r = (|| -> Result<u64> {
            let header = serde_json::to_vec(&json!({ "fingerprint": self.fingerprint, "tokens": tokens, "layout": engine.prefix_layout(tokens.len()) }))?;
            let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp)?);
            w.write_all(MAGIC)?;
            w.write_all(&(header.len() as u64).to_le_bytes())?;
            w.write_all(&header)?;
            let bytes = engine.prefix_write(tokens.len(), &mut w)?;
            w.into_inner().map_err(|e| HerbertError::Backend(format!("{}: {e}", tmp.display())))?.sync_all()?;
            std::fs::rename(&tmp, &path)?;
            Ok(16 + header.len() as u64 + bytes)
        })();
        match &r {
            Ok(_) => self.entries.push((path.clone(), tokens.to_vec())),
            Err(_) => {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        let keep = if shared { KEEP_SHARED } else { self.keep };
        let kind = |p: &Path| p.to_string_lossy().ends_with(".shared.hkv");
        for (_, old) in self.files().into_iter().filter(|f| kind(&f.1) == shared).skip(keep) {
            if old != path {
                eprintln!("[prefix-cache] removed {} (keeps {keep})", old.display());
                let _ = std::fs::remove_file(&old);
                self.entries.retain(|e| e.0 != old);
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_is_the_reference() {
        // FNV-1a 64 of "a" and of "foobar" (the published test vectors).
        assert_eq!(fnv(FNV0, b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv(FNV0, b"foobar"), 0x85944171f73967e8);
    }
}
