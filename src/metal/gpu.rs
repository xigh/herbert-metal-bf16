//! The device, its queue, the buffers, the pipelines, and the encoding of a
//! sequence of dispatches.
//!
//! Every buffer is shared (`StorageModeShared`): on Apple silicon the host and
//! the GPU read the same memory, so the host writes the token, reads the
//! argmax and uploads the weights through `contents()`, with no staging copy.
//! A command buffer has one compute encoder of the serial kind: each dispatch
//! sees the writes of the previous ones (the buffers are hazard-tracked), the
//! order is the order of the calls.

use crate::core::error::{HerbertError, Result};
use crate::msl::Emitted;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;

/// A buffer of the arena: its index, its address on the host, its length.
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    pub index: u32,
    pub ptr: *mut u8,
    pub len: u64,
}

/// A view of a buffer bound to a kernel.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub buf: Buffer,
    pub offset: u64,
    pub size: u64,
}

impl Range {
    pub fn whole(buf: Buffer) -> Range {
        Range { buf, offset: 0, size: buf.len }
    }

    pub fn part(buf: Buffer, offset: u64, size: u64) -> Range {
        assert!(offset + size <= buf.len, "range past the end of the buffer");
        Range { buf, offset, size }
    }
}

pub struct Pipeline {
    pub name: String,
    pub threads: u32,
    /// What each binding holds ([`Emitted::binds`]).
    pub binds: Vec<crate::msl::Elem>,
    /// The bindings the kernel writes ([`Emitted::writes`]; empty: every one).
    pub writes: Vec<u32>,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Bytes allocated so far.
    pub allocated: u64,
    /// Simdgroup width the kernels are written for.
    pub simd: u32,
}

impl Gpu {
    pub fn open() -> Result<Gpu> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| HerbertError::Backend("no Metal device".into()))?;
        let queue = device.newCommandQueue().ok_or_else(|| HerbertError::Backend("newCommandQueue failed".into()))?;
        Ok(Gpu { device, queue, buffers: Vec::new(), allocated: 0, simd: 32 })
    }

    /// The device's recommended working set, bytes.
    pub fn working_set(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} · working set {:.1} GiB · max buffer {:.1} GiB · threadgroup memory {} KiB",
            self.device.name(),
            self.device.recommendedMaxWorkingSetSize() as f64 / (1u64 << 30) as f64,
            self.device.maxBufferLength() as f64 / (1u64 << 30) as f64,
            self.device.maxThreadgroupMemoryLength() / 1024
        )
    }

    /// A zeroed shared buffer of `bytes`.
    pub fn alloc(&mut self, bytes: u64) -> Result<Buffer> {
        let len = bytes.max(16);
        if len > self.device.maxBufferLength() as u64 {
            return Err(HerbertError::Backend(format!("a buffer of {} MiB exceeds maxBufferLength", len >> 20)));
        }
        let b = self
            .device
            .newBufferWithLength_options(len as usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| HerbertError::Backend(format!("newBufferWithLength({len}) failed")))?;
        let ptr = b.contents().as_ptr() as *mut u8;
        self.buffers.push(b);
        self.allocated += len;
        Ok(Buffer { index: (self.buffers.len() - 1) as u32, ptr, len })
    }

    /// A buffer holding `data`.
    pub fn alloc_with(&mut self, data: &[u8]) -> Result<Buffer> {
        self.alloc_parts(&[data])
    }

    /// A buffer holding the concatenation of `parts`.
    pub fn alloc_parts(&mut self, parts: &[&[u8]]) -> Result<Buffer> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let b = self.alloc(total as u64)?;
        let mut off = 0;
        for p in parts {
            // SAFETY: the buffer has `total` bytes and the GPU does not use it yet.
            unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), b.ptr.add(off), p.len()) };
            off += p.len();
        }
        Ok(b)
    }

    /// Compile an emitted kernel into a pipeline (fast math off; MSL 3.1, or
    /// 4.0 for a kernel of Metal 4's tensors).
    pub fn pipeline(&self, e: &Emitted) -> Result<Pipeline> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(if e.msl4 { MTLLanguageVersion::Version4_0 } else { MTLLanguageVersion::Version3_1 });
        options.setMathMode(MTLMathMode::Safe);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&e.source), Some(&options))
            .map_err(|err| HerbertError::Backend(format!("{}: MSL compile failed: {}", e.name, err.localizedDescription())))?;
        let entry = crate::msl::entry_name(&e.name);
        let function = library
            .newFunctionWithName(&NSString::from_str(&entry))
            .ok_or_else(|| HerbertError::Backend(format!("{}: no function {entry}", e.name)))?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| HerbertError::Backend(format!("{}: pipeline: {}", e.name, err.localizedDescription())))?;
        let max = state.maxTotalThreadsPerThreadgroup() as u32;
        if max < e.threads {
            return Err(HerbertError::Backend(format!("{}: {} threads per threadgroup, the pipeline takes at most {max}", e.name, e.threads)));
        }
        Ok(Pipeline { name: e.name.clone(), threads: e.threads, binds: e.binds.clone(), writes: e.writes.clone(), state })
    }

    /// A command buffer with its compute encoder open.
    pub fn begin(&self) -> Result<Cmd> {
        self.begin_with(false)
    }

    /// [`Gpu::begin`], or with `concurrent` its encoder concurrent
    /// (`MTLDispatchTypeConcurrent`), a buffer barrier before every dispatch
    /// but those of [`Gpu::dispatch_free`]: the serial order, but for those.
    pub fn begin_with(&self, concurrent: bool) -> Result<Cmd> {
        let cb = self.queue.commandBuffer().ok_or_else(|| HerbertError::Backend("commandBuffer failed".into()))?;
        let enc = if concurrent { cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent) } else { cb.computeCommandEncoder() };
        let enc = enc.ok_or_else(|| HerbertError::Backend("computeCommandEncoder failed".into()))?;
        Ok(Cmd { cb, enc, dispatches: 0, discarded: false, concurrent })
    }

    /// `kernel<<<groups, threads>>>(ranges…)`.
    pub fn dispatch(&self, cmd: &mut Cmd, p: &Pipeline, ranges: &[Range], groups: u32) {
        self.dispatch_2d(cmd, p, ranges, groups, 1)
    }

    /// `kernel<<<(groups, height), threads>>>(ranges…)`: `tgid.y` runs over `height`.
    pub fn dispatch_2d(&self, cmd: &mut Cmd, p: &Pipeline, ranges: &[Range], groups: u32, height: u32) {
        if cmd.concurrent && cmd.dispatches > 0 {
            cmd.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        self.encode(cmd, p, ranges, groups, height)
    }

    /// A buffer barrier in a concurrent command buffer: what was dispatched
    /// before it is done and visible to what comes after.
    pub fn barrier(&self, cmd: &mut Cmd) {
        cmd.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    /// A dispatch free to run beside those since the last barrier (concurrent
    /// command buffers; serial ones run it in order): a cache warmer, which
    /// writes nothing anyone reads.
    pub fn dispatch_free(&self, cmd: &mut Cmd, p: &Pipeline, ranges: &[Range], groups: u32) {
        self.encode(cmd, p, ranges, groups, 1)
    }

    fn encode(&self, cmd: &mut Cmd, p: &Pipeline, ranges: &[Range], groups: u32, height: u32) {
        cmd.enc.setComputePipelineState(&p.state);
        for (i, r) in ranges.iter().enumerate() {
            // SAFETY: the buffer lives in the arena as long as the engine; offset within it.
            unsafe { cmd.enc.setBuffer_offset_atIndex(Some(&self.buffers[r.buf.index as usize]), r.offset as usize, i) };
        }
        let g = MTLSize { width: groups as usize, height: height as usize, depth: 1 };
        let t = MTLSize { width: p.threads as usize, height: 1, depth: 1 };
        cmd.enc.dispatchThreadgroups_threadsPerThreadgroup(g, t);
        cmd.dispatches += 1;
    }

    /// Close without submitting.
    pub fn discard(&self, cmd: Cmd) {
        cmd.enc.endEncoding();
    }

    /// Close, submit, wait; the times of the command buffer.
    pub fn submit_wait(&self, cmd: Cmd) -> Result<super::diag::Done> {
        cmd.enc.endEncoding();
        let committed = super::diag::now();
        cmd.cb.commit();
        cmd.cb.waitUntilCompleted();
        let returned = super::diag::now();
        if cmd.cb.status() != MTLCommandBufferStatus::Completed {
            let why = cmd.cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_default();
            return Err(HerbertError::Backend(format!("command buffer failed: {why}")));
        }
        let (g0, g1, k0) = (cmd.cb.GPUStartTime(), cmd.cb.GPUEndTime(), cmd.cb.kernelStartTime());
        Ok(super::diag::Done { gpu: g1 - g0, wait: returned - committed, launch: g0 - committed, queue: g0 - k0, tail: returned - g1 })
    }
}

/// `HERBERT_METAL_PROFILE=1`: every dispatch in a
/// command buffer of its own, its GPU time summed by kernel, the table printed
/// when the engine is dropped. Slow (one submission per dispatch, ~30 µs of
/// launch each): for measuring only. Not under `=2` or `=3` (herbert-metal-bench's
/// costs by removal and the prefill block dispatch by dispatch time the step themselves).
pub struct Profile {
    by_kernel: HashMap<String, (f64, usize)>,
    /// GPU seconds since the last [`Profile::take`].
    pending: f64,
}

impl Profile {
    pub fn from_env() -> Option<Profile> {
        (std::env::var("HERBERT_METAL_PROFILE").as_deref() == Ok("1")).then(|| Profile { by_kernel: HashMap::new(), pending: 0.0 })
    }

    /// `p` dispatched alone, submitted, waited on, its GPU time counted.
    pub fn one(&mut self, gpu: &Gpu, p: &Pipeline, ranges: &[Range], groups: u32) -> Result<()> {
        let mut cmd = gpu.begin()?;
        gpu.dispatch(&mut cmd, p, ranges, groups);
        let d = gpu.submit_wait(cmd)?.gpu;
        let e = self.by_kernel.entry(p.name.clone()).or_default();
        e.0 += d;
        e.1 += 1;
        self.pending += d;
        Ok(())
    }

    /// The GPU seconds counted since the last call.
    pub fn take(&mut self) -> f64 {
        std::mem::take(&mut self.pending)
    }

    /// The table on stderr, the slowest kernel first.
    pub fn print(&self, what: &str) {
        let total: f64 = self.by_kernel.values().map(|v| v.0).sum();
        let mut r: Vec<_> = self.by_kernel.iter().collect();
        r.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        eprintln!("[profile {what}] {} kernels, GPU {:.1} ms in all", r.len(), total * 1e3);
        for (n, (t, c)) in r {
            eprintln!("[profile {what}] {:>10.2} ms {:>5.1} %  ×{:<6} {:>9.1} µs each  {n}", t * 1e3, 100.0 * t / total.max(1e-12), c, t * 1e6 / *c as f64);
        }
    }
}

/// An open command buffer and its encoder.
pub struct Cmd {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    pub dispatches: usize,
    /// Encoded only to compile its kernels, never submitted.
    pub discarded: bool,
    concurrent: bool,
}
