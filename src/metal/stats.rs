//! `HERBERT_METAL_STATS=FILE` (the f16 study's probe S4): the range of every
//! activation the engine keeps in 16 bits or in f32.
//!
//! Every dispatch runs alone in its command buffer, as under
//! `HERBERT_METAL_PROFILE`; after it, each binding that is an activation
//! buffer (named by the engine at load: the weights are not) and that the
//! kernel declares bf16 (`U16`, `UVec4`) or f32 is folded by
//! [`kn::range_stats`]: max |x|, min |x| ≠ 0, the non-zero values, those
//! under 2⁻¹⁴ (f16's smallest normal) and 2⁻²⁴ (its smallest subnormal), NaN
//! and Inf. The whole bound range is read, so the unwritten part of a cache
//! counts as zeros. The key is (phase, kernel, binding, buffer name, rank):
//! the rank counts the folds of the same (kernel, binding, buffer) since the
//! start of the step or block ([`Stats::pass`]), so a kernel run once per
//! layer gets its layer's rank among the layers that run it. When the engine
//! is dropped, FILE gets one tab-separated line per key.

use super::gpu::{Buffer, Gpu, Pipeline, Range};
use super::kernels as kn;
use crate::core::error::Result;
use crate::msl::Elem;
use std::collections::{BTreeMap, HashMap};

#[derive(Default, Clone, Copy)]
struct Acc {
    /// Ranges folded, values seen.
    folds: u64,
    values: u64,
    max: f32,
    min_nz: f32,
    nonzero: f64,
    below14: f64,
    below24: f64,
    nonfinite: f64,
}

pub struct Stats {
    path: String,
    pipes: [Pipeline; 2],
    n_buf: Buffer,
    part: Buffer,
    names: HashMap<u32, String>,
    acc: BTreeMap<(String, String, usize, String, usize), Acc>,
    rank: HashMap<(String, usize, String), usize>,
}

impl Stats {
    /// `HERBERT_METAL_STATS=FILE`, or nothing.
    pub fn from_env(gpu: &mut Gpu) -> Result<Option<Stats>> {
        let Some(path) = std::env::var_os("HERBERT_METAL_STATS") else { return Ok(None) };
        let pipes = [gpu.pipeline(&kn::range_stats(false, gpu.simd))?, gpu.pipeline(&kn::range_stats(true, gpu.simd))?];
        let n_buf = gpu.alloc(16)?;
        let part = gpu.alloc((kn::STATS_GROUPS * kn::STATS_FIELDS * 4) as u64)?;
        Ok(Some(Stats { path: path.to_string_lossy().into_owned(), pipes, n_buf, part, names: HashMap::new(), acc: BTreeMap::new(), rank: HashMap::new() }))
    }

    /// `b` is an activation, `name` in the table.
    pub fn name(&mut self, b: Buffer, name: impl Into<String>) {
        self.names.insert(b.index, name.into());
    }

    /// A new step or prefill block: the ranks start again.
    pub fn pass(&mut self) {
        self.rank.clear();
    }

    /// `p` dispatched alone, then its activation bindings folded; the GPU time
    /// of `p` alone.
    pub fn one(&mut self, gpu: &Gpu, phase: &str, p: &Pipeline, ranges: &[Range], groups: u32) -> Result<f64> {
        self.one_2d(gpu, phase, p, ranges, groups, 1)
    }

    /// [`Self::one`] over `groups × height` threadgroups.
    pub fn one_2d(&mut self, gpu: &Gpu, phase: &str, p: &Pipeline, ranges: &[Range], groups: u32, height: u32) -> Result<f64> {
        let mut cmd = gpu.begin()?;
        gpu.dispatch_2d(&mut cmd, p, ranges, groups, height);
        let t = gpu.submit_wait(cmd)?.gpu;
        for (i, r) in ranges.iter().enumerate() {
            let Some(name) = self.names.get(&r.buf.index) else { continue };
            let (f32_in, n) = match p.binds.get(i) {
                Some(Elem::U16 | Elem::UVec4) => (false, r.size / 2),
                Some(Elem::F32) => (true, r.size / 4),
                _ => continue,
            };
            // SAFETY: a 16-byte shared buffer; the GPU is idle.
            unsafe { *(self.n_buf.ptr as *mut u32) = n as u32 };
            let mut cmd = gpu.begin()?;
            gpu.dispatch(&mut cmd, &self.pipes[f32_in as usize], &[*r, Range::whole(self.n_buf), Range::whole(self.part)], kn::STATS_GROUPS);
            gpu.submit_wait(cmd)?;
            // SAFETY: the partials of every group, written by the finished command buffer.
            let v = unsafe { std::slice::from_raw_parts(self.part.ptr as *const f32, (kn::STATS_GROUPS * kn::STATS_FIELDS) as usize) };
            let rank = self.rank.entry((p.name.clone(), i, name.clone())).or_insert(0);
            let key = (phase.to_string(), p.name.clone(), i, name.clone(), *rank);
            *rank += 1;
            let a = self.acc.entry(key).or_insert(Acc { min_nz: f32::INFINITY, ..Acc::default() });
            a.folds += 1;
            a.values += n;
            for g in v.chunks(kn::STATS_FIELDS as usize) {
                a.max = a.max.max(g[0]);
                a.min_nz = a.min_nz.min(g[1]);
                a.nonzero += g[2] as f64;
                a.below14 += g[3] as f64;
                a.below24 += g[4] as f64;
                a.nonfinite += g[5] as f64;
            }
        }
        Ok(t)
    }
}

impl Drop for Stats {
    fn drop(&mut self) {
        let mut s = String::from("phase\tkernel\tbinding\tbuffer\trank\tfolds\tvalues\tmax_abs\tmin_abs_nonzero\tnonzero\tbelow_2^-14\tbelow_2^-24\tnan_inf\n");
        for ((phase, kernel, i, buf, rank), a) in &self.acc {
            s.push_str(&format!(
                "{phase}\t{kernel}\t{i}\t{buf}\t{rank}\t{}\t{}\t{:e}\t{:e}\t{}\t{}\t{}\t{}\n",
                a.folds, a.values, a.max, a.min_nz, a.nonzero, a.below14, a.below24, a.nonfinite
            ));
        }
        match std::fs::write(&self.path, s) {
            Ok(()) => eprintln!("[stats] {} keys written to {}", self.acc.len(), self.path),
            Err(e) => eprintln!("[stats] {}: {e}", self.path),
        }
    }
}
