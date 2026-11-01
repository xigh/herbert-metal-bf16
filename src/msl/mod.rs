//! The kernel emitter: the API of herbert-vk-bf16's `spv::Kernel`, written out
//! as Metal Shading Language text instead of SPIR-V words.
//!
//! Every kernel of the engine is a function `xxx(shape) -> Emitted` built with
//! [`Kernel`]: the dimensions known at load time are emitted as constants, the
//! few values that change from token to token (the position) are read from a
//! buffer. The text is compiled at load by `newLibraryWithSource` with fast
//! math off. The emitter is the contract, not the MSL: the same calls are
//! meant to produce AIR later, with nothing between the emitter and the
//! driver's compiler.
//!
//! What the layer gives, as `spv` does:
//! - values in single assignment (`uint v12 = v3 * v7;`), constants inline,
//!   floats by their bits (`as_type<float>(0x3f800000u)`) so that none is
//!   rounded by a decimal round trip;
//! - device buffers (`[[buffer(N)]]`), threadgroup arrays, mutable variables
//!   (`var_*`, `get`, `set`);
//! - structured control flow (`if_`, `if_else`, `for_`);
//! - simdgroup reductions, and a threadgroup reduction that never runs a tree
//!   under a barrier: each simdgroup reduces, then thread 0 folds the partials
//!   in order;
//! - bf16 ↔ f32 by shifts, rounding to nearest even in integers, exactly like
//!   `f32_to_bf16` of the CPU engine; or, with `HERBERT_METAL_BF16_NATIVE=1`
//!   (the f16 study's probe S2), by MSL's own conversion `bfloat(x)`;
//! - or f16 in place of bf16 for every 16-bit value (`set_f16`, or
//!   [`set_f16_default`] for the kernels emitted next): `half` matrices and
//!   pointers, `half(x)` and `float(h)` conversions (the f16 study's probe S3).

use std::fmt::Write as _;

/// A value: an index into the kernel's table of names.
pub type Word = u32;

/// What one element of a device buffer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    U32,
    F32,
    /// bf16 (or f16) stored one element at a time; loads widen to `uint`.
    U16,
    /// 16 bytes: eight bf16 at once.
    UVec4,
    /// An activation or state of the DeltaNet kept between kernels: f32, or
    /// bf16 with the `bf16` feature ([`ACT_BF16`]). Loaded as f32 (widened
    /// exactly), stored from f32 (rounded to nearest even once).
    Act,
}

/// The `bf16` feature: the activations and states the DeltaNet keeps between
/// kernels ([`Elem::Act`]) are bf16 instead of f32.
pub const ACT_BF16: bool = cfg!(feature = "bf16");

/// Bytes of one [`Elem::Act`].
pub const ACT_BYTES: usize = if ACT_BF16 { 2 } else { 4 };

/// `HERBERT_METAL_BF16_NATIVE=1`: every rounding f32 → bf16 is emitted as
/// MSL's `bfloat(x)` instead of the integer form (the f16 study's probe S2).
/// Read once, at the first kernel.
pub fn bf16_native_from_env() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("HERBERT_METAL_BF16_NATIVE").is_ok_and(|v| v == "1"))
}

static F16_DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(cfg!(feature = "f16"));

/// The 16-bit format of the kernels created from now on: f16 (`true`) or
/// bf16 (`false`). The engine's, at start: f16 with the `f16` feature
/// (`core::bf16::F16`), bf16 otherwise; the probes switch it per kernel.
pub fn set_f16_default(on: bool) {
    F16_DEFAULT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The DeltaNet's 16-bit values scaled by a power of two in memory (option c
/// of the f16 numerical contract, `--tune dn.sscale/oscale/tscale`):
/// the exponents `[state, out, tc]` of the recurrent state, of the
/// recurrence's output and of dnmma's chunk operators, 0 = unscaled. A value
/// `x` is stored as `round16(x·2^e)` and read back as `float(·)·2^-e`: exact
/// in f32 while nothing overflows, the subnormal range of f16 moved `e`
/// binades down. Set once, before the engine's kernels are emitted.
static DN_SCALE: [std::sync::atomic::AtomicU32; 3] = [const { std::sync::atomic::AtomicU32::new(0) }; 3];

/// The scaled families of [`set_dn_scale`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DnScaled {
    State,
    Out,
    Tc,
}

pub fn set_dn_scale(state: u32, out: u32, tc: u32) {
    for (a, v) in DN_SCALE.iter().zip([state, out, tc]) {
        a.store(v, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The exponent of `f`'s scale, 0 = unscaled.
pub fn dn_scale(f: DnScaled) -> u32 {
    DN_SCALE[f as usize].load(std::sync::atomic::Ordering::Relaxed)
}

/// A scalar type of the probes' arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    F16,
    /// Two f16 in one register.
    H2,
    Bf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    U32,
    F32,
    Bool,
    UVec4,
    /// An 8 × 8 simdgroup matrix of bf16 (an operand), or f16 in f16 mode.
    MatB,
    /// An 8 × 8 simdgroup matrix of f32 (an accumulator).
    MatF,
    Half,
    Half2,
    Bfloat,
    /// An 8 × 8 simdgroup matrix of f16 (an operand of the half-staged GEMM forms).
    MatH,
    /// A threadgroup `atomic_uint` (a [`Kernel::shared_atomic`] array only).
    Atomic,
    /// A threadgroup `ushort` (a [`Kernel::shared_u16`] array only): loaded
    /// widened to `uint`, stored from the low 16 bits of one.
    U16,
    /// A Metal 4 tensor, operation or cooperative tensor (`auto`).
    Auto,
}

impl Ty {
    fn msl(self) -> &'static str {
        match self {
            Ty::U32 => "uint",
            Ty::F32 => "float",
            Ty::Bool => "bool",
            Ty::UVec4 => "uint4",
            Ty::MatB => "simdgroup_bfloat8x8",
            Ty::MatF => "simdgroup_float8x8",
            Ty::Half => "half",
            Ty::Half2 => "half2",
            Ty::Bfloat => "bfloat",
            Ty::MatH => "simdgroup_half8x8",
            Ty::Atomic => "atomic_uint",
            Ty::U16 => "ushort",
            Ty::Auto => "auto",
        }
    }
}

/// A device buffer (`[[buffer(index)]]`).
#[derive(Clone, Copy, Debug)]
pub struct Buf {
    index: u32,
    elem: Elem,
    /// Declared [`Elem::Act`]: loads widen a bf16 to f32, stores round to bf16.
    act: bool,
    /// A second view of a binding already declared (SPIR-V binds the same
    /// buffer twice; MSL casts the first pointer): `b{index}_a{alias}`.
    alias: u32,
    /// An [`Elem::Act`] stored scaled by `2^scale` ([`Buf::scaled`]).
    scale: u32,
}

impl Buf {
    /// This [`Elem::Act`] binding holding its values times `2^e`: [`Kernel::ld`]
    /// multiplies by `2^-e` after widening, [`Kernel::st`] by `2^e` before
    /// rounding (both exact in f32). `e = 0`: as it was.
    pub fn scaled(self, e: u32) -> Buf {
        assert!(self.act || e == 0, "only an Act buffer is scaled");
        Buf { scale: e, ..self }
    }

    fn name(&self) -> String {
        if self.alias == 0 { format!("b{}", self.index) } else { format!("b{}_a{}", self.index, self.alias) }
    }
}

fn elem_msl(e: Elem) -> &'static str {
    match e {
        Elem::U32 => "uint",
        Elem::F32 => "float",
        Elem::U16 => "ushort",
        Elem::UVec4 => "uint4",
        Elem::Act => unreachable!("an Act buffer is declared as U16 or F32"),
    }
}

/// A threadgroup array.
#[derive(Clone, Copy, Debug)]
pub struct Shared {
    id: u32,
    ty: Ty,
}

/// A mutable variable.
#[derive(Clone, Copy, Debug)]
pub struct Var {
    id: u32,
    ty: Ty,
}

pub struct Kernel {
    local_size: u32,
    subgroup: u32,
    names: Vec<(String, Ty)>,
    buffers: Vec<Buf>,
    shared: Vec<String>,
    vars: u32,
    body: String,
    depth: usize,
    /// Round f32 → bf16 by `bfloat(x)` rather than in integers.
    bf16_native: bool,
    /// f16 in place of bf16 for every 16-bit value.
    f16: bool,
    /// Uses Metal 4's tensors ([`Kernel::mm2d`], [`Kernel::mm2d_tile`]): MSL 4.0 and its headers.
    msl4: bool,
    /// Bindings declared `coherent(device)` ([`Kernel::coherent_device`]).
    coherent: Vec<u32>,
    /// Bindings the kernel writes (stores, atomics, `simdgroup_store`, [`Kernel::written`]).
    writes: Vec<u32>,
}

impl Kernel {
    /// A kernel of `local_size` threads per threadgroup, simdgroups of `subgroup`.
    pub fn new(local_size: u32, subgroup: u32) -> Kernel {
        assert!(local_size.is_multiple_of(subgroup), "{local_size} threads are not whole simdgroups of {subgroup}");
        Kernel { local_size, subgroup, names: Vec::new(), buffers: Vec::new(), shared: Vec::new(), vars: 0, body: String::new(), depth: 1, bf16_native: bf16_native_from_env(), f16: F16_DEFAULT.load(std::sync::atomic::Ordering::Relaxed), msl4: false, coherent: Vec::new(), writes: Vec::new() }
    }

    /// The rounding f32 → bf16 of this kernel: `bfloat(x)` (`true`) or the
    /// integer form (`false`), whatever `HERBERT_METAL_BF16_NATIVE` says.
    pub fn set_bf16_native(&mut self, on: bool) {
        self.bf16_native = on;
    }

    /// f16 (`true`) or bf16 (`false`) for every 16-bit value of this kernel.
    pub fn set_f16(&mut self, on: bool) {
        self.f16 = on;
    }

    /// The MSL name of the 16-bit format.
    fn t16(&self) -> &'static str {
        if self.f16 { "half" } else { "bfloat" }
    }

    pub fn local_size(&self) -> u32 {
        self.local_size
    }

    pub fn subgroup(&self) -> u32 {
        self.subgroup
    }

    /// Simdgroups per threadgroup.
    pub fn subgroups(&self) -> u32 {
        self.local_size / self.subgroup
    }

    /// The source of the kernel, its function named `entry`.
    pub fn finish(self, entry: &str) -> String {
        let mut s = String::from("#include <metal_stdlib>\n");
        if self.msl4 {
            s.push_str("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n");
        }
        s.push_str("using namespace metal;\n\n");
        let mut bufs: Vec<Buf> = self.buffers.iter().copied().filter(|b| b.alias == 0).collect();
        bufs.sort_by_key(|b| b.index);
        let _ = write!(s, "kernel void {entry}(\n");
        for b in &bufs {
            let coh = if self.coherent.contains(&b.index) { "coherent(device) " } else { "" };
            let _ = writeln!(s, "    device {coh}{ty} *b{i} [[buffer({i})]],", ty = elem_msl(b.elem), i = b.index);
        }
        s.push_str("    uint3 tgid [[threadgroup_position_in_grid]],\n");
        s.push_str("    uint3 ngroups [[threadgroups_per_grid]],\n");
        s.push_str("    uint3 tpig [[thread_position_in_grid]],\n");
        s.push_str("    uint lid [[thread_index_in_threadgroup]],\n");
        s.push_str("    uint sgid [[simdgroup_index_in_threadgroup]],\n");
        s.push_str("    uint lane [[thread_index_in_simdgroup]])\n{\n");
        for d in &self.shared {
            let _ = writeln!(s, "    {d}");
        }
        for b in self.buffers.iter().filter(|b| b.alias > 0) {
            let ty = elem_msl(b.elem);
            let coh = if self.coherent.contains(&b.index) { "coherent(device) " } else { "" };
            let _ = writeln!(s, "    device {coh}{ty} *{} = (device {coh}{ty} *)b{};", b.name(), b.index);
        }
        s.push_str(&self.body);
        s.push_str("}\n");
        s
    }

    // ── values ────────────────────────────────────────────────────────────

    fn name(&self, w: Word) -> &str {
        &self.names[w as usize].0
    }

    fn ty(&self, w: Word) -> Ty {
        self.names[w as usize].1
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.body.push_str("    ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    /// A value named by an expression that needs no statement (a constant, a builtin).
    fn inline(&mut self, expr: String, ty: Ty) -> Word {
        self.names.push((expr, ty));
        (self.names.len() - 1) as Word
    }

    /// A new value `ty vN = expr;`.
    fn def(&mut self, ty: Ty, expr: String) -> Word {
        let id = self.names.len();
        let name = format!("v{id}");
        self.line(&format!("{} {name} = {expr};", ty.msl()));
        self.names.push((name, ty));
        id as Word
    }

    pub fn u(&mut self, v: u32) -> Word {
        self.inline(format!("{v}u"), Ty::U32)
    }

    pub fn f(&mut self, v: f32) -> Word {
        self.inline(format!("as_type<float>({:#010x}u)", v.to_bits()), Ty::F32)
    }

    /// The f32 `2^e`.
    pub fn pow2(&mut self, e: i32) -> Word {
        assert!((-126..=127).contains(&e));
        self.f(f32::from_bits(((127 + e) as u32) << 23))
    }

    // ── buffers ───────────────────────────────────────────────────────────

    /// The buffer bound at `index`; declared again, a second view of the same
    /// binding with another element type.
    pub fn buffer(&mut self, index: u32, elem: Elem) -> Buf {
        let (elem, act) = match elem {
            Elem::Act if ACT_BF16 => (Elem::U16, true),
            Elem::Act => (Elem::F32, true),
            e => (e, false),
        };
        let alias = self.buffers.iter().filter(|b| b.index == index).count() as u32;
        let b = Buf { index, elem, act, alias, scale: 0 };
        self.buffers.push(b);
        b
    }

    /// Hints of the SPIR-V emitter; nothing in MSL for now.
    pub fn readonly(&mut self, _buf: Buf) {}
    pub fn coherent(&mut self, _buf: Buf) {}

    /// The binding of `buf` declared `coherent(device)`: its stores seen by
    /// the other groups of the dispatch after [`Kernel::fence_device`] (MSL 3.2:
    /// the kernel compiled as MSL 4.0).
    pub fn coherent_device(&mut self, buf: Buf) {
        self.coherent.push(buf.index);
        self.msl4 = true;
    }

    /// `atomic_thread_fence` over device memory at device scope, sequentially consistent.
    pub fn fence_device(&mut self) {
        self.msl4 = true;
        self.line("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);");
    }

    /// `atomic_fetch_add` on `buf[index]` (a `uint` binding) as a device `atomic_uint`; the old value.
    pub fn dev_atomic_add(&mut self, buf: Buf, index: Word, value: Word) -> Word {
        self.written(buf);
        let e = format!("atomic_fetch_add_explicit((device atomic_uint *)&{}[{}], {}, memory_order_relaxed)", buf.name(), self.name(index), self.name(value));
        self.def(Ty::U32, e)
    }

    /// `atomic_store` on `buf[index]` as a device `atomic_uint`.
    pub fn dev_atomic_store(&mut self, buf: Buf, index: Word, value: Word) {
        self.written(buf);
        let line = format!("atomic_store_explicit((device atomic_uint *)&{}[{}], {}, memory_order_relaxed);", buf.name(), self.name(index), self.name(value));
        self.line(&line);
    }

    /// Wait until the device `atomic_uint` `buf[index]` reaches `target`.
    pub fn spin_until_ge(&mut self, buf: Buf, index: Word, target: Word) {
        let line = format!("while (atomic_load_explicit((device atomic_uint *)&{}[{}], memory_order_relaxed) < {}) {{}}", buf.name(), self.name(index), self.name(target));
        self.line(&line);
    }
    pub fn nontemporal(&mut self, _buf: Buf) {}

    /// Move the buffer's pointer `offset` elements on, for the rest of the
    /// kernel: a kernel over the rows of a block reaches its row this way and
    /// keeps the indexing of a single row.
    pub fn advance(&mut self, buf: Buf, offset: Word) {
        let line = format!("{} += {};", buf.name(), self.name(offset));
        self.line(&line);
    }

    /// `buf[index]`; a `U16` element comes back widened to `uint`, an
    /// [`Elem::Act`] one as f32.
    pub fn ld(&mut self, buf: Buf, index: Word) -> Word {
        let at = format!("{}[{}]", buf.name(), self.name(index));
        match buf.elem {
            Elem::U32 => self.def(Ty::U32, at),
            Elem::F32 => self.def(Ty::F32, at),
            Elem::U16 if buf.act => {
                let w = self.def(Ty::U32, format!("uint({at})"));
                let v = self.bf16_to_f(w);
                if buf.scale > 0 {
                    let s = self.pow2(-(buf.scale as i32));
                    self.fmul(v, s)
                } else {
                    v
                }
            }
            Elem::U16 => self.def(Ty::U32, format!("uint({at})")),
            Elem::UVec4 => self.def(Ty::UVec4, at),
            Elem::Act => unreachable!(),
        }
    }

    /// `buf[index] = value`; a `U16` element takes the low 16 bits of a
    /// `uint`, an [`Elem::Act`] one an f32 rounded to bf16.
    pub fn st(&mut self, buf: Buf, index: Word, value: Word) {
        let value = if buf.act && buf.scale > 0 {
            let s = self.pow2(buf.scale as i32);
            self.fmul(value, s)
        } else {
            value
        };
        let value = if buf.act && buf.elem == Elem::U16 { self.f_to_bf16(value) } else { value };
        let v = match buf.elem {
            Elem::U16 => format!("ushort({})", self.name(value)),
            _ => self.name(value).to_string(),
        };
        self.written(buf);
        let line = format!("{}[{}] = {v};", buf.name(), self.name(index));
        self.line(&line);
    }

    /// Record that the kernel writes `buf` (its [`Emitted::writes`]); the
    /// stores of this emitter record themselves, [`Kernel::raw`] text says it here.
    pub fn written(&mut self, buf: Buf) {
        if !self.writes.contains(&buf.index) {
            self.writes.push(buf.index);
        }
    }

    /// A threadgroup array of `len` floats (`float_elems`) or `uint`s.
    pub fn shared(&mut self, len: u32, float_elems: bool) -> Shared {
        let id = self.shared.len() as u32;
        let ty = if float_elems { Ty::F32 } else { Ty::U32 };
        self.shared.push(format!("threadgroup {} s{id}[{len}];", ty.msl()));
        Shared { id, ty }
    }

    /// A threadgroup array of `len` `atomic_uint`, read and written by
    /// [`Kernel::atomic_store`] and [`Kernel::atomic_add`] (relaxed order;
    /// barriers order them between threads).
    pub fn shared_atomic(&mut self, len: u32) -> Shared {
        let id = self.shared.len() as u32;
        self.shared.push(format!("threadgroup atomic_uint s{id}[{len}];"));
        Shared { id, ty: Ty::Atomic }
    }

    pub fn atomic_store(&mut self, sh: Shared, index: Word, value: Word) {
        assert_eq!(sh.ty, Ty::Atomic);
        let line = format!("atomic_store_explicit(&s{}[{}], {}, memory_order_relaxed);", sh.id, self.name(index), self.name(value));
        self.line(&line);
    }

    /// `sh[index] += value`, the value before.
    pub fn atomic_add(&mut self, sh: Shared, index: Word, value: Word) -> Word {
        assert_eq!(sh.ty, Ty::Atomic);
        let e = format!("atomic_fetch_add_explicit(&s{}[{}], {}, memory_order_relaxed)", sh.id, self.name(index), self.name(value));
        self.def(Ty::U32, e)
    }

    pub fn atomic_load(&mut self, sh: Shared, index: Word) -> Word {
        assert_eq!(sh.ty, Ty::Atomic);
        let e = format!("atomic_load_explicit(&s{}[{}], memory_order_relaxed)", sh.id, self.name(index));
        self.def(Ty::U32, e)
    }

    /// A threadgroup array of `len` `uint4` cells.
    pub fn shared_vec4(&mut self, len: u32) -> Shared {
        let id = self.shared.len() as u32;
        self.shared.push(format!("threadgroup uint4 s{id}[{len}];"));
        Shared { id, ty: Ty::UVec4 }
    }

    /// The memory of a [`Kernel::shared_vec4`] array seen as floats (four per
    /// cell): an epilogue through a staging area, once its tiles are consumed.
    pub fn shared_f32_view(&mut self, of: Shared) -> Shared {
        assert_eq!(of.ty, Ty::UVec4);
        let id = self.shared.len() as u32;
        self.shared.push(format!("threadgroup float *s{id} = (threadgroup float *)s{};", of.id));
        Shared { id, ty: Ty::F32 }
    }

    /// A threadgroup array of `len` `ushort`s (bf16 written one at a time);
    /// [`Kernel::shared_vec4_view`] reads it eight at a time.
    pub fn shared_u16(&mut self, len: u32) -> Shared {
        let id = self.shared.len() as u32;
        self.shared.push(format!("threadgroup ushort s{id}[{len}];"));
        Shared { id, ty: Ty::U16 }
    }

    /// A [`Kernel::shared_u16`] array seen as `uint4`s (`len` a multiple of 8).
    pub fn shared_vec4_view(&mut self, of: Shared) -> Shared {
        assert_eq!(of.ty, Ty::U16);
        let id = self.shared.len() as u32;
        self.shared.push(format!("threadgroup uint4 *s{id} = (threadgroup uint4 *)s{};", of.id));
        Shared { id, ty: Ty::UVec4 }
    }

    pub fn sld(&mut self, sh: Shared, index: Word) -> Word {
        if sh.ty == Ty::U16 {
            let e = format!("uint(s{}[{}])", sh.id, self.name(index));
            return self.def(Ty::U32, e);
        }
        let e = format!("s{}[{}]", sh.id, self.name(index));
        self.def(sh.ty, e)
    }

    pub fn sst(&mut self, sh: Shared, index: Word, value: Word) {
        let v = if sh.ty == Ty::U16 { format!("ushort({})", self.name(value)) } else { self.name(value).to_string() };
        let line = format!("s{}[{}] = {v};", sh.id, self.name(index));
        self.line(&line);
    }

    /// `sh[index][component] = value` in a [`Kernel::shared_vec4`] array (a
    /// `uint`; the component may be computed).
    pub fn sst_comp(&mut self, sh: Shared, index: Word, component: Word, value: Word) {
        assert_eq!(sh.ty, Ty::UVec4);
        let line = format!("s{}[{}][{}] = {};", sh.id, self.name(index), self.name(component), self.name(value));
        self.line(&line);
    }

    // ── builtins ──────────────────────────────────────────────────────────

    pub fn local_id(&mut self) -> Word {
        self.inline("lid".into(), Ty::U32)
    }

    pub fn group_id(&mut self, component: u32) -> Word {
        self.inline(format!("tgid.{}", ["x", "y", "z"][component as usize]), Ty::U32)
    }

    pub fn num_groups(&mut self, component: u32) -> Word {
        self.inline(format!("ngroups.{}", ["x", "y", "z"][component as usize]), Ty::U32)
    }

    pub fn global_id(&mut self) -> Word {
        self.inline("tpig.x".into(), Ty::U32)
    }

    /// The threadgroup's index (the dispatch is one-dimensional).
    pub fn flat_group(&mut self) -> Word {
        self.inline("tgid.x".into(), Ty::U32)
    }

    pub fn subgroup_id(&mut self) -> Word {
        self.inline("sgid".into(), Ty::U32)
    }

    pub fn lane(&mut self) -> Word {
        self.inline("lane".into(), Ty::U32)
    }

    // ── variables ─────────────────────────────────────────────────────────

    fn var(&mut self, ty: Ty, init: Word) -> Var {
        let id = self.vars;
        self.vars += 1;
        let line = format!("{} m{id} = {};", ty.msl(), self.name(init));
        self.line(&line);
        Var { id, ty }
    }

    pub fn var_u32(&mut self, init: Word) -> Var {
        self.var(Ty::U32, init)
    }

    pub fn var_f32(&mut self, init: Word) -> Var {
        self.var(Ty::F32, init)
    }

    pub fn var_uvec4(&mut self, init: Word) -> Var {
        self.var(Ty::UVec4, init)
    }

    pub fn get(&mut self, v: Var) -> Word {
        self.def(v.ty, format!("m{}", v.id))
    }

    pub fn set(&mut self, v: Var, value: Word) {
        let line = format!("m{} = {};", v.id, self.name(value));
        self.line(&line);
    }

    // ── control flow ──────────────────────────────────────────────────────

    pub fn if_(&mut self, cond: Word, then: impl FnOnce(&mut Kernel)) {
        let line = format!("if ({}) {{", self.name(cond));
        self.line(&line);
        self.depth += 1;
        then(self);
        self.depth -= 1;
        self.line("}");
    }

    pub fn if_else(&mut self, cond: Word, then: impl FnOnce(&mut Kernel), other: impl FnOnce(&mut Kernel)) {
        let line = format!("if ({}) {{", self.name(cond));
        self.line(&line);
        self.depth += 1;
        then(self);
        self.depth -= 1;
        self.line("} else {");
        self.depth += 1;
        other(self);
        self.depth -= 1;
        self.line("}");
    }

    /// `for (i = start; i < end; i += step)`, unsigned.
    pub fn for_(&mut self, start: Word, end: Word, step: Word, body: impl FnOnce(&mut Kernel, Word)) {
        let id = self.names.len();
        let i = format!("v{id}");
        self.names.push((i.clone(), Ty::U32));
        let line = format!("for (uint {i} = {}; {i} < {}; {i} += {}) {{", self.name(start), self.name(end), self.name(step));
        self.line(&line);
        self.depth += 1;
        body(self, id as Word);
        self.depth -= 1;
        self.line("}");
    }

    /// Threadgroup barrier, with the threadgroup and device memory made visible.
    /// `return;`: the invocation stops here (a whole group at once, or no barrier after).
    pub fn ret(&mut self) {
        self.line("return;");
    }

    pub fn barrier(&mut self) {
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
    }

    pub fn sg_barrier(&mut self) {
        self.line("simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
    }

    // ── reductions ────────────────────────────────────────────────────────

    pub fn sg_fadd(&mut self, v: Word) -> Word {
        let e = format!("simd_sum({})", self.name(v));
        self.def(Ty::F32, e)
    }

    /// Sum over clusters of `size` consecutive lanes, every lane of a cluster
    /// getting it: a butterfly of `simd_shuffle_xor` (each step adds two equal
    /// partial sums in either order, so every lane holds the same bits).
    pub fn sg_fadd_clustered(&mut self, v: Word, size: u32) -> Word {
        assert!(size.is_power_of_two() && size <= self.subgroup, "cluster of {size}");
        let mut acc = v;
        let mut off = 1;
        while off < size {
            let e = format!("simd_shuffle_xor({}, {off}u)", self.name(acc));
            let other = self.def(Ty::F32, e);
            acc = self.fadd(acc, other);
            off *= 2;
        }
        acc
    }

    pub fn sg_fmax(&mut self, v: Word) -> Word {
        let e = format!("simd_max({})", self.name(v));
        self.def(Ty::F32, e)
    }

    /// Max over clusters of `size` consecutive lanes, every lane of a cluster
    /// getting it (a butterfly of `simd_shuffle_xor`, as [`Kernel::sg_fadd_clustered`]).
    pub fn sg_fmax_clustered(&mut self, v: Word, size: u32) -> Word {
        assert!(size.is_power_of_two() && size <= self.subgroup, "cluster of {size}");
        let mut acc = v;
        let mut off = 1;
        while off < size {
            let e = format!("simd_shuffle_xor({}, {off}u)", self.name(acc));
            let other = self.def(Ty::F32, e);
            acc = self.fmax(acc, other);
            off *= 2;
        }
        acc
    }

    /// True in every lane when `c` is true in every lane of the simdgroup.
    pub fn sg_all(&mut self, c: Word) -> Word {
        assert_eq!(self.ty(c), Ty::Bool);
        let e = format!("simd_all({})", self.name(c));
        self.def(Ty::Bool, e)
    }

    pub fn sg_umin(&mut self, v: Word) -> Word {
        let e = format!("simd_min({})", self.name(v));
        self.def(Ty::U32, e)
    }

    /// Threadgroup reduction (`add` or `max`) of one f32 per thread; every
    /// thread gets the result. `scratch` holds `subgroups() + 1` floats. The
    /// simdgroups reduce in registers; thread 0 then folds the partials in
    /// order — no tree under a barrier.
    pub fn wg_reduce(&mut self, v: Word, scratch: Shared, max: bool) -> Word {
        let r = if max { self.sg_fmax(v) } else { self.sg_fadd(v) };
        let g = self.subgroups();
        if g == 1 {
            return r;
        }
        let lane = self.lane();
        let sgid = self.subgroup_id();
        let zero = self.u(0);
        let first_lane = self.ieq(lane, zero);
        self.if_(first_lane, |k| k.sst(scratch, sgid, r));
        self.barrier();
        let lid = self.local_id();
        let first = self.ieq(lid, zero);
        self.if_(first, |k| {
            let z = k.u(0);
            let acc = k.sld(scratch, z);
            let acc_v = k.var_f32(acc);
            let one = k.u(1);
            let end = k.u(g);
            k.for_(one, end, one, |k, j| {
                let a = k.get(acc_v);
                let p = k.sld(scratch, j);
                let s = if max { k.fmax(a, p) } else { k.fadd(a, p) };
                k.set(acc_v, s);
            });
            let a = k.get(acc_v);
            let slot = k.u(g);
            k.sst(scratch, slot, a);
        });
        self.barrier();
        let slot = self.u(g);
        let res = self.sld(scratch, slot);
        // The scratch can be reused right after.
        self.barrier();
        res
    }

    /// Threadgroup minimum of one `uint` per thread, same pattern as
    /// [`Self::wg_reduce`]; `scratch` holds `subgroups() + 1` uints.
    pub fn wg_umin(&mut self, v: Word, scratch: Shared) -> Word {
        let r = self.sg_umin(v);
        let g = self.subgroups();
        if g == 1 {
            return r;
        }
        let lane = self.lane();
        let sgid = self.subgroup_id();
        let zero = self.u(0);
        let first_lane = self.ieq(lane, zero);
        self.if_(first_lane, |k| k.sst(scratch, sgid, r));
        self.barrier();
        let lid = self.local_id();
        let first = self.ieq(lid, zero);
        self.if_(first, |k| {
            let z = k.u(0);
            let acc = k.sld(scratch, z);
            let acc_v = k.var_u32(acc);
            let one = k.u(1);
            let end = k.u(g);
            k.for_(one, end, one, |k, j| {
                let a = k.get(acc_v);
                let p = k.sld(scratch, j);
                let lt = k.ult(p, a);
                let m = k.select_u(lt, p, a);
                k.set(acc_v, m);
            });
            let a = k.get(acc_v);
            let slot = k.u(g);
            k.sst(scratch, slot, a);
        });
        self.barrier();
        let slot = self.u(g);
        let res = self.sld(scratch, slot);
        self.barrier();
        res
    }

    // ── arithmetic ────────────────────────────────────────────────────────

    fn bin(&mut self, ty: Ty, a: Word, op: &str, b: Word) -> Word {
        let e = format!("{} {op} {}", self.name(a), self.name(b));
        self.def(ty, e)
    }

    fn call(&mut self, ty: Ty, f: &str, args: &[Word]) -> Word {
        let list: Vec<&str> = args.iter().map(|&a| self.name(a)).collect();
        let e = format!("{f}({})", list.join(", "));
        self.def(ty, e)
    }

    pub fn iadd(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "+", b)
    }
    pub fn isub(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "-", b)
    }
    pub fn imul(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "*", b)
    }
    pub fn udiv(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "/", b)
    }
    pub fn umod(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "%", b)
    }
    pub fn shl(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "<<", b)
    }
    pub fn shr(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, ">>", b)
    }
    pub fn and(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "&", b)
    }
    pub fn or(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "|", b)
    }
    pub fn xor(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::U32, a, "^", b)
    }
    pub fn fadd(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::F32, a, "+", b)
    }
    pub fn fsub(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::F32, a, "-", b)
    }
    pub fn fmul(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::F32, a, "*", b)
    }
    pub fn fdiv(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::F32, a, "/", b)
    }
    pub fn fneg(&mut self, a: Word) -> Word {
        let e = format!("-{}", self.name(a));
        self.def(Ty::F32, e)
    }
    pub fn fma(&mut self, a: Word, b: Word, c: Word) -> Word {
        self.call(Ty::F32, "fma", &[a, b, c])
    }
    pub fn exp(&mut self, a: Word) -> Word {
        self.call(Ty::F32, "exp", &[a])
    }
    pub fn log(&mut self, a: Word) -> Word {
        self.call(Ty::F32, "log", &[a])
    }
    pub fn sin(&mut self, a: Word) -> Word {
        self.call(Ty::F32, "sin", &[a])
    }
    pub fn cos(&mut self, a: Word) -> Word {
        self.call(Ty::F32, "cos", &[a])
    }
    pub fn sqrt(&mut self, a: Word) -> Word {
        self.call(Ty::F32, "sqrt", &[a])
    }
    pub fn rsqrt(&mut self, a: Word) -> Word {
        self.call(Ty::F32, "rsqrt", &[a])
    }
    pub fn fmax(&mut self, a: Word, b: Word) -> Word {
        self.call(Ty::F32, "fmax", &[a, b])
    }
    pub fn fmin(&mut self, a: Word, b: Word) -> Word {
        self.call(Ty::F32, "fmin", &[a, b])
    }
    /// `1 / (1 + e^(−x))`.
    pub fn sigmoid(&mut self, x: Word) -> Word {
        let n = self.fneg(x);
        let e = self.exp(n);
        let one = self.f(1.0);
        let d = self.fadd(one, e);
        self.fdiv(one, d)
    }
    pub fn u2f(&mut self, a: Word) -> Word {
        let e = format!("float({})", self.name(a));
        self.def(Ty::F32, e)
    }
    pub fn bits_to_f(&mut self, a: Word) -> Word {
        let e = format!("as_type<float>({})", self.name(a));
        self.def(Ty::F32, e)
    }
    pub fn f_to_bits(&mut self, a: Word) -> Word {
        let e = format!("as_type<uint>({})", self.name(a));
        self.def(Ty::U32, e)
    }
    /// A `uint4` of four `uint`s.
    pub fn uvec4(&mut self, c: [Word; 4]) -> Word {
        let e = format!("uint4({}, {}, {}, {})", self.name(c[0]), self.name(c[1]), self.name(c[2]), self.name(c[3]));
        self.def(Ty::UVec4, e)
    }

    /// Component `i` of a `uint4`.
    pub fn comp(&mut self, v: Word, i: u32) -> Word {
        assert_eq!(self.ty(v), Ty::UVec4);
        let e = format!("{}.{}", self.name(v), ["x", "y", "z", "w"][i as usize]);
        self.def(Ty::U32, e)
    }

    // ── comparisons and selection ─────────────────────────────────────────

    pub fn ult(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::Bool, a, "<", b)
    }
    pub fn ieq(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::Bool, a, "==", b)
    }
    pub fn fgt(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::Bool, a, ">", b)
    }
    pub fn feq(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::Bool, a, "==", b)
    }
    pub fn land(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::Bool, a, "&&", b)
    }
    pub fn lor(&mut self, a: Word, b: Word) -> Word {
        self.bin(Ty::Bool, a, "||", b)
    }
    pub fn b_not(&mut self, a: Word) -> Word {
        let e = format!("!{}", self.name(a));
        self.def(Ty::Bool, e)
    }
    pub fn select_u(&mut self, c: Word, a: Word, b: Word) -> Word {
        let e = format!("{} ? {} : {}", self.name(c), self.name(a), self.name(b));
        self.def(Ty::U32, e)
    }
    pub fn select_f(&mut self, c: Word, a: Word, b: Word) -> Word {
        let e = format!("{} ? {} : {}", self.name(c), self.name(a), self.name(b));
        self.def(Ty::F32, e)
    }

    // ── Metal 4 tensors (`matmul2d`) ───────────────────────────────────────

    /// One tile of `X·Wᵀ` by `mpp::tensor_ops::matmul2d` over the
    /// threadgroup's simdgroups, into the threadgroup array `c` of
    /// `tm × tn` f32 (row `r`, column `j` at `r·tn + j`): rows `x0..x0 + xr`
    /// of `X` and `w0..w0 + wr` of `W`, both 16-bit (bf16, or f16 in f16 mode) `[·][k]` row-major in
    /// buffers of `uint4` (eight per cell); `xr ≤ tm`, `wr ≤ tn`, the rest of
    /// the tile undefined. The products of two bf16 are exact in f32 and the
    /// sums are in f32 (`relaxed_precision` false). Every thread of the group
    /// runs it; no barrier inside (the caller orders `c`'s reads). The kernel
    /// compiles as MSL 4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn mm2d_tile(&mut self, x: Buf, x0: Word, xr: Word, w: Buf, w0: Word, wr: Word, k: u32, tm: u32, tn: u32, c: Shared) {
        assert!(c.ty == Ty::F32 && x.elem == Elem::UVec4 && w.elem == Elem::UVec4, "mm2d_tile: f32 tile, bf16 operands in uint4 buffers");
        self.msl4 = true;
        let sgs = self.subgroups();
        let t = self.t16();
        let (xn, x0n, xrn, wn, w0n, wrn) = (x.name(), self.name(x0).to_string(), self.name(xr).to_string(), w.name(), self.name(w0).to_string(), self.name(wr).to_string());
        for line in [
            "{".to_string(),
            format!("    auto tX = tensor((device {t} *){xn} + {x0n} * {k}u, dextents<int32_t, 2>({k}, int({xrn})), array<int, 2>({{1, {k}}}));"),
            format!("    auto tW = tensor((device {t} *){wn} + {w0n} * {k}u, dextents<int32_t, 2>({k}, int({wrn})), array<int, 2>({{1, {k}}}));"),
            format!("    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<{sgs}>> mm;"),
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();".to_string(),
            "    mm.run(tX, tW, cT);".to_string(),
            format!("    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(s{}, dextents<int32_t, 2>({tn}, {tm}));", c.id),
            "    cT.store(tC);".to_string(),
            "}".to_string(),
        ] {
            self.line(&line);
        }
    }

    // ── Metal 4 tensors, register fragments (`sdpa_nax`) ────────────────────

    /// Lines of MSL as given, at the current depth (a kernel whose body its
    /// emitter writes in one piece: `sdpa_nax`); the kernel compiles as MSL 4.0.
    pub fn raw(&mut self, src: &str) {
        self.msl4 = true;
        for l in src.lines() {
            self.line(l);
        }
    }

    /// The MSL name of the 16-bit format (`half`, or `bfloat` for the witness).
    pub fn t16_name(&self) -> &'static str {
        self.t16()
    }

    /// The MSL expression of [`Kernel::f_to_bf16`] on the f32 expression `f`
    /// (its 16 bits in a `uint`), for [`Kernel::raw`] text.
    pub fn f_to_16_text(&self, f: &str) -> String {
        if self.f16 {
            format!("uint(as_type<ushort>(half({f})))")
        } else if self.bf16_native {
            format!("uint(as_type<ushort>(bfloat({f})))")
        } else {
            format!("((as_type<uint>(float({f})) + 0x7fffu + ((as_type<uint>(float({f})) >> 16) & 1u)) >> 16)")
        }
    }

    /// The MSL expression of [`Kernel::bf16_to_f`] on the `uint` expression `h`.
    pub fn h16_to_f_text(&self, h: &str) -> String {
        if self.f16 { format!("float(as_type<half>(ushort({h})))") } else { format!("as_type<float>(uint({h}) << 16)") }
    }

    /// The MSL text of `C0 | C1 (+)= A · [B0 | B1]` (or `A · [B0 | B1]ᵀ`,
    /// `tr`) by one simdgroup: a `matmul2d` of 16 × 32 × 16 run by
    /// `execution_simdgroup` on cooperative tensors filled from registers.
    /// Each name is a `vec<·, 8>` fragment of a 16 × 16 tile in the layout of
    /// mlx's NAX fragments (lane `l`: rows `fm`, `fm + 8`, columns `fn..fn + 4`
    /// with `fm = ((l >> 2) & 4) | ((l >> 1) & 3)`, `fn = (((l >> 2) & 2) | (l & 1)) · 4`),
    /// `A` 16-bit, the `C` in f32 (`relaxed`: the descriptor's reduced
    /// precision). `B0` holds the rows (`tr`: the columns of the product)
    /// 0..16 of the right operand, `B1` 16..32.
    #[allow(clippy::too_many_arguments)]
    pub fn nax_mma_text(&self, c0: &str, c1: &str, a: &str, b0: &str, b1: &str, tr: bool, relaxed: bool) -> String {
        let t = self.t16();
        format!(
            "{{\n    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, {tr}, {relaxed}, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> op;\n    auto ca = op.get_left_input_cooperative_tensor<{t}, {t}, float>();\n    auto cb = op.get_right_input_cooperative_tensor<{t}, {t}, float>();\n    auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();\n    #pragma clang loop unroll(full)\n    for (ushort e = 0; e < 8; ++e) {{ ca[e] = {a}[e]; cb[e] = {b0}[e]; cb[8 + e] = {b1}[e]; cc[e] = {c0}[e]; cc[8 + e] = {c1}[e]; }}\n    op.run(ca, cb, cc);\n    #pragma clang loop unroll(full)\n    for (ushort e = 0; e < 8; ++e) {{ {c0}[e] = cc[e]; {c1}[e] = cc[8 + e]; }}\n}}"
        )
    }

    // ── Metal 4 tensors, piece by piece (`sdpa_mm2d`) ───────────────────────

    /// A 2-D tensor over a device buffer of 16-bit values: element `(i, j)`
    /// (extent 0 `i`, the inner one) at element `offset + i + j · stride`;
    /// `e0` columns, `e1` rows.
    pub fn tensor_dev16(&mut self, buf: Buf, offset: Word, e0: u32, e1: Word, stride: u32) -> Word {
        self.msl4 = true;
        let t = self.t16();
        let e = format!("tensor((device {t} *){} + {}, dextents<int32_t, 2>({e0}, int({})), array<int, 2>({{1, {stride}}}))", buf.name(), self.name(offset), self.name(e1));
        self.def(Ty::Auto, e)
    }

    /// The same over a threadgroup array of 16-bit values (a [`Kernel::shared_vec4`] array).
    pub fn tensor_tg16(&mut self, sh: Shared, offset: Word, e0: u32, e1: u32, stride: u32) -> Word {
        assert_eq!(sh.ty, Ty::UVec4);
        self.msl4 = true;
        let t = self.t16();
        let e = format!("tensor((threadgroup {t} *)s{} + {}, dextents<int32_t, 2>({e0}, {e1}), array<int, 2>({{1, {stride}}}))", sh.id, self.name(offset));
        self.def(Ty::Auto, e)
    }

    /// The same over a threadgroup array of f32.
    pub fn tensor_tg_f32(&mut self, sh: Shared, offset: Word, e0: u32, e1: u32, stride: u32) -> Word {
        assert_eq!(sh.ty, Ty::F32);
        self.msl4 = true;
        let e = format!("tensor(s{} + {}, dextents<int32_t, 2>({e0}, {e1}), array<int, 2>({{1, {stride}}}))", sh.id, self.name(offset));
        self.def(Ty::Auto, e)
    }

    /// A `matmul2d` of `m × n` outputs over `k` (`None`: read from the
    /// operands), `X·W` or `X·Wᵀ` (`tr`), into a fresh destination or
    /// accumulating into it (`acc`), run by every simdgroup of the group.
    pub fn mm2d_op(&mut self, m: u32, n: u32, k: Option<u32>, tr: bool, acc: bool) -> Word {
        self.msl4 = true;
        let sgs = self.subgroups();
        let kk = k.map_or_else(|| "static_cast<int>(dynamic_extent)".to_string(), |k| k.to_string());
        let mode = if acc { "multiply_accumulate" } else { "multiply" };
        let e = format!("mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor({m}, {n}, {kk}, false, {tr}, false, mpp::tensor_ops::matmul2d_descriptor::mode::{mode}), execution_simdgroups<{sgs}>>()");
        self.def(Ty::Auto, e)
    }

    /// The f32 destination cooperative tensor of `op` for the operands `a`,
    /// `b` (their types), its elements zeroed.
    pub fn mm2d_coop(&mut self, op: Word, a: Word, b: Word) -> Var {
        let id = self.vars;
        self.vars += 1;
        let line = format!("auto m{id} = {}.get_destination_cooperative_tensor<decltype({}), decltype({}), float>();", self.name(op), self.name(a), self.name(b));
        self.line(&line);
        self.line("#pragma clang loop unroll(full)");
        self.line(&format!("for (uint16_t e = 0; e < m{id}.get_capacity(); ++e) {{ if (m{id}.is_valid_element(e)) m{id}[e] = 0.0f; }}"));
        Var { id, ty: Ty::Auto }
    }

    /// `c = a·b` (or `c += a·b`, as `op` says).
    pub fn mm2d_run(&mut self, op: Word, a: Word, b: Word, c: Var) {
        assert_eq!(c.ty, Ty::Auto);
        let line = format!("{}.run({}, {}, m{});", self.name(op), self.name(a), self.name(b), c.id);
        self.line(&line);
    }

    /// Row `r` (extent 1) of the cooperative tensor `c` times `sh[base + r]`.
    pub fn coop_scale_rows(&mut self, c: Var, sh: Shared, base: Word) {
        assert!(c.ty == Ty::Auto && sh.ty == Ty::F32);
        self.line("#pragma clang loop unroll(full)");
        let line = format!(
            "for (uint16_t e = 0; e < m{0}.get_capacity(); ++e) {{ if (m{0}.is_valid_element(e)) {{ auto ix = m{0}.get_multidimensional_index(e); m{0}[e] *= s{1}[{2} + ix[1]]; }} }}",
            c.id, sh.id, self.name(base)
        );
        self.line(&line);
    }

    /// The cooperative tensor `c` stored into the tensor `t`.
    pub fn coop_store(&mut self, c: Var, t: Word) {
        assert_eq!(c.ty, Ty::Auto);
        let line = format!("m{}.store({});", c.id, self.name(t));
        self.line(&line);
    }

    // ── simdgroup matrices (8 × 8) ─────────────────────────────────────────
    //
    // Metal's matrix units (`simdgroup_matrix`), what the cooperative matrices
    // of the SPIR-V emitter are to Vulkan: bf16 operands, f32 accumulators.

    /// An f32 accumulator, zeroed.
    pub fn mat_acc(&mut self) -> Var {
        let id = self.vars;
        self.vars += 1;
        self.line(&format!("simdgroup_float8x8 m{id} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"));
        Var { id, ty: Ty::MatF }
    }

    /// The bf16 matrix at element `offset` of a threadgroup array of bf16
    /// (a [`Kernel::shared_vec4`] array, eight per cell), its rows `stride`
    /// elements apart; `transpose`: its transpose.
    pub fn mat_load_bf16(&mut self, sh: Shared, offset: Word, stride: u32, transpose: bool) -> Word {
        assert_eq!(sh.ty, Ty::UVec4);
        let id = self.names.len();
        let name = format!("v{id}");
        let t = self.t16();
        self.line(&format!("simdgroup_{t}8x8 {name};"));
        let line = format!("simdgroup_load({name}, (threadgroup {t} *)s{} + {}, {stride}, ulong2(0, 0), {transpose});", sh.id, self.name(offset));
        self.line(&line);
        self.names.push((name, Ty::MatB));
        id as Word
    }

    /// The f16 matrix at element `offset` of a threadgroup array of f16 (a
    /// [`Kernel::shared_vec4`] array, eight per cell), its rows `stride`
    /// elements apart; `transpose`: its transpose. For the GEMM forms whose
    /// operands are staged as half (`MmaForm::half`).
    pub fn mat_load_f16(&mut self, sh: Shared, offset: Word, stride: u32, transpose: bool) -> Word {
        assert_eq!(sh.ty, Ty::UVec4);
        let id = self.names.len();
        let name = format!("v{id}");
        self.line(&format!("simdgroup_half8x8 {name};"));
        let line = format!("simdgroup_load({name}, (threadgroup half *)s{} + {}, {stride}, ulong2(0, 0), {transpose});", sh.id, self.name(offset));
        self.line(&line);
        self.names.push((name, Ty::MatH));
        id as Word
    }

    /// An f16 accumulator (`simdgroup_half8x8`), zeroed: the probe of the
    /// matrix units with 16-bit accumulation (herbert-metal-probe `hacc`).
    pub fn mat_acc_h(&mut self) -> Var {
        let id = self.vars;
        self.vars += 1;
        self.line(&format!("simdgroup_half8x8 m{id} = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);"));
        Var { id, ty: Ty::MatH }
    }

    /// `acc = a·b + acc` into an f16 accumulator ([`Kernel::mat_acc_h`]), `a`
    /// and `b` f16 matrices.
    pub fn mat_mad_h(&mut self, acc: Var, a: Word, b: Word) {
        assert!(acc.ty == Ty::MatH && self.ty(a) == self.ty(b) && matches!(self.ty(a), Ty::MatB | Ty::MatH), "f16 accumulator, f16 operands");
        let line = format!("simdgroup_multiply_accumulate(m{0}, {1}, {2}, m{0});", acc.id, self.name(a), self.name(b));
        self.line(&line);
    }

    /// An f16 accumulator stored into a threadgroup array of f16 (a
    /// [`Kernel::shared_vec4`] array) at element `offset`, rows `stride` apart.
    pub fn mat_store_h(&mut self, acc: Var, sh: Shared, offset: Word, stride: u32) {
        assert!(acc.ty == Ty::MatH && sh.ty == Ty::UVec4);
        let line = format!("simdgroup_store(m{}, (threadgroup half *)s{} + {}, {stride});", acc.id, sh.id, self.name(offset));
        self.line(&line);
    }

    /// `acc = a·b + acc` (`a`, `b` both bf16 or both f16 matrices).
    pub fn mat_mad(&mut self, acc: Var, a: Word, b: Word) {
        assert_eq!(acc.ty, Ty::MatF);
        assert!(self.ty(a) == self.ty(b) && matches!(self.ty(a), Ty::MatB | Ty::MatH), "operands of one type");
        let line = format!("simdgroup_multiply_accumulate(m{0}, {1}, {2}, m{0});", acc.id, self.name(a), self.name(b));
        self.line(&line);
    }

    /// The bf16 matrix at element `offset` of a device buffer of bf16, its rows
    /// `stride` elements apart; `transpose`: its transpose.
    pub fn mat_load_bf16_dev(&mut self, buf: Buf, offset: Word, stride: u32, transpose: bool) -> Word {
        assert_eq!(buf.elem, Elem::U16);
        let id = self.names.len();
        let name = format!("v{id}");
        let t = self.t16();
        self.line(&format!("simdgroup_{t}8x8 {name};"));
        let line = format!("simdgroup_load({name}, (device {t} *){} + {}, {stride}, ulong2(0, 0), {transpose});", buf.name(), self.name(offset));
        self.line(&line);
        self.names.push((name, Ty::MatB));
        id as Word
    }

    /// A bf16 matrix widened exactly into a new f32 accumulator.
    pub fn mat_from_bf16(&mut self, b: Word) -> Var {
        assert_eq!(self.ty(b), Ty::MatB);
        let id = self.vars;
        self.vars += 1;
        self.line(&format!("simdgroup_float8x8 m{id};"));
        let line = if self.f16 {
            format!("for (ushort e = 0; e < 2; e++) {{ m{id}.thread_elements()[e] = float({}.thread_elements()[e]); }}", self.name(b))
        } else {
            format!(
                "for (ushort e = 0; e < 2; e++) {{ m{id}.thread_elements()[e] = as_type<float>(uint(as_type<ushort>({}.thread_elements()[e])) << 16); }}",
                self.name(b)
            )
        };
        self.line(&line);
        Var { id, ty: Ty::MatF }
    }

    /// An accumulator rounded to bf16 (nearest even, in integers, as
    /// [`Kernel::f_to_bf16`]; or by `bfloat(x)`).
    pub fn mat_to_bf16(&mut self, acc: Var) -> Word {
        assert_eq!(acc.ty, Ty::MatF);
        let id = self.names.len();
        let name = format!("v{id}");
        self.line(&format!("simdgroup_{}8x8 {name};", self.t16()));
        if self.f16 {
            self.line(&format!("for (ushort e = 0; e < 2; e++) {{ {name}.thread_elements()[e] = half(m{}.thread_elements()[e]); }}", acc.id));
        } else if self.bf16_native {
            self.line(&format!("for (ushort e = 0; e < 2; e++) {{ {name}.thread_elements()[e] = bfloat(m{}.thread_elements()[e]); }}", acc.id));
        } else {
            self.line(&format!(
                "for (ushort e = 0; e < 2; e++) {{ uint b = as_type<uint>(m{}.thread_elements()[e]); {name}.thread_elements()[e] = as_type<bfloat>(ushort((b + 0x7fffu + ((b >> 16) & 1u)) >> 16)); }}",
                acc.id
            ));
        }
        self.names.push((name, Ty::MatB));
        id as Word
    }

    /// `a -= b`, element by element.
    pub fn mat_sub_in_place(&mut self, a: Var, b: Var) {
        assert!(a.ty == Ty::MatF && b.ty == Ty::MatF);
        self.line(&format!("for (ushort e = 0; e < 2; e++) {{ m{}.thread_elements()[e] -= m{}.thread_elements()[e]; }}", a.id, b.id));
    }

    /// `a *= s`, element by element.
    pub fn mat_scale_in_place(&mut self, a: Var, s: Word) {
        assert!(a.ty == Ty::MatF && self.ty(s) == Ty::F32);
        let line = format!("for (ushort e = 0; e < 2; e++) {{ m{}.thread_elements()[e] *= {}; }}", a.id, self.name(s));
        self.line(&line);
    }

    /// A bf16 matrix stored into a threadgroup array of bf16 (a
    /// [`Kernel::shared_vec4`] array) at element `offset`, rows `stride` apart.
    pub fn mat_store_bf16(&mut self, b: Word, sh: Shared, offset: Word, stride: u32) {
        assert!(self.ty(b) == Ty::MatB && sh.ty == Ty::UVec4);
        let line = format!("simdgroup_store({}, (threadgroup {} *)s{} + {}, {stride});", self.name(b), self.t16(), sh.id, self.name(offset));
        self.line(&line);
    }

    /// A bf16 matrix stored into a device buffer of bf16 at element `offset`,
    /// rows `stride` apart.
    pub fn mat_store_bf16_dev(&mut self, b: Word, buf: Buf, offset: Word, stride: u32) {
        assert!(self.ty(b) == Ty::MatB && buf.elem == Elem::U16);
        self.written(buf);
        let line = format!("simdgroup_store({}, (device {} *){} + {}, {stride});", self.name(b), self.t16(), buf.name(), self.name(offset));
        self.line(&line);
    }

    /// An accumulator stored into an f32 threadgroup array at element
    /// `offset`, its rows `stride` elements apart.
    pub fn mat_store(&mut self, acc: Var, sh: Shared, offset: Word, stride: u32) {
        assert!(acc.ty == Ty::MatF && sh.ty == Ty::F32);
        let line = format!("simdgroup_store(m{}, s{} + {}, {stride});", acc.id, sh.id, self.name(offset));
        self.line(&line);
    }

    /// The f32 matrix at element `offset` of an f32 threadgroup array, its
    /// rows `stride` elements apart (a diagonal of scales, for [`Kernel::mat_mul_left`]).
    pub fn mat_load_f32(&mut self, sh: Shared, offset: Word, stride: u32) -> Var {
        assert_eq!(sh.ty, Ty::F32);
        let id = self.vars;
        self.vars += 1;
        self.line(&format!("simdgroup_float8x8 m{id};"));
        let line = format!("simdgroup_load(m{id}, s{} + {}, {stride});", sh.id, self.name(offset));
        self.line(&line);
        Var { id, ty: Ty::MatF }
    }

    /// `acc = d·acc`, both f32 matrices (`simdgroup_multiply`). With `d` a
    /// diagonal, row `i` of `acc` times `d[i][i]`, exactly (the other products
    /// are zeros, added exactly, unless `acc` holds an infinity or a NaN).
    pub fn mat_mul_left(&mut self, acc: Var, d: Var) {
        assert!(acc.ty == Ty::MatF && d.ty == Ty::MatF);
        let line = format!("simdgroup_multiply(m{0}, m{1}, m{0});", acc.id, d.id);
        self.line(&line);
    }

    // ── bf16 ──────────────────────────────────────────────────────────────

    /// A bf16 held in the low 16 bits of a `uint`, widened exactly.
    pub fn bf16_to_f(&mut self, h: Word) -> Word {
        let e = if self.f16 { format!("float(as_type<half>(ushort({})))", self.name(h)) } else { format!("as_type<float>({} << 16)", self.name(h)) };
        self.def(Ty::F32, e)
    }

    /// The bf16 in the low half of a word.
    pub fn lo_f(&mut self, w: Word) -> Word {
        self.bf16_to_f(w)
    }

    /// The bf16 in the high half of a word.
    pub fn hi_f(&mut self, w: Word) -> Word {
        let e = if self.f16 { format!("float(as_type<half>(ushort({} >> 16)))", self.name(w)) } else { format!("as_type<float>({} & 0xffff0000u)", self.name(w)) };
        self.def(Ty::F32, e)
    }

    /// An f16 held in the low 16 bits of a `uint`, widened exactly.
    pub fn f16_to_f(&mut self, h: Word) -> Word {
        let e = format!("float(as_type<half>(ushort({})))", self.name(h));
        self.def(Ty::F32, e)
    }

    /// An f32 converted to f16 (nearest even; exact when the value is a bf16
    /// in `[2^-17, 2^16)` or zero), its bits in the low 16 of a `uint`.
    pub fn f_to_f16(&mut self, f: Word) -> Word {
        let e = format!("uint(as_type<ushort>(half({})))", self.name(f));
        self.def(Ty::U32, e)
    }

    /// Round to nearest even, in integers, as `f32_to_bf16` of the CPU engine:
    /// `(bits + 0x7fff + lsb) >> 16`; or by `bfloat(x)`.
    pub fn f_to_bf16(&mut self, f: Word) -> Word {
        if self.f16 {
            let e = format!("uint(as_type<ushort>(half({})))", self.name(f));
            return self.def(Ty::U32, e);
        }
        if self.bf16_native {
            let e = format!("uint(as_type<ushort>(bfloat({})))", self.name(f));
            return self.def(Ty::U32, e);
        }
        let bits = self.f_to_bits(f);
        let e = format!("({0} + 0x7fffu + (({0} >> 16) & 1u)) >> 16", self.name(bits));
        self.def(Ty::U32, e)
    }

    /// What an [`Elem::Act`] buffer scaled by `2^e` ([`Buf::scaled`]) gives
    /// back of `f` stored into it: `round16(f·2^e)·2^-e`.
    pub fn round_bf16_scaled(&mut self, f: Word, e: u32) -> Word {
        if e == 0 {
            return self.round_bf16(f);
        }
        let up = self.pow2(e as i32);
        let down = self.pow2(-(e as i32));
        let x = self.fmul(f, up);
        let r = self.round_bf16(x);
        self.fmul(r, down)
    }

    /// `f` rounded to bf16, kept as f32.
    pub fn round_bf16(&mut self, f: Word) -> Word {
        if self.f16 {
            let e = format!("float(half({}))", self.name(f));
            return self.def(Ty::F32, e);
        }
        if self.bf16_native {
            let e = format!("float(bfloat({}))", self.name(f));
            return self.def(Ty::F32, e);
        }
        let h = self.f_to_bf16(f);
        self.bf16_to_f(h)
    }

    // ── Metal 4 tensors (the f16 study's probe S3, M5's Neural Accelerators) ──

    /// `Y = X·Wᵀ` by `mpp::tensor_ops::matmul2d` over 4 simdgroups: `X`
    /// `[m][k]` and `W` `[n][k]` 16-bit (`half`, or `bfloat` when `bf16`),
    /// row-major, `Y` `[m][n]` f32. One `tm × tn` tile of `Y` per group,
    /// `m / tm` groups along `m` first; the descriptor's `relaxed_precision`
    /// as `relaxed`. The kernel compiles as MSL 4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn mm2d(&mut self, bf16: bool, x: Buf, w: Buf, y: Buf, k: u32, m: u32, n: u32, tm: u32, tn: u32, relaxed: bool) {
        self.written(y);
        assert!(self.local_size == 128 && m.is_multiple_of(tm) && n.is_multiple_of(tn), "mm2d: 128 threads, whole tiles");
        self.msl4 = true;
        let t = if bf16 { "bfloat" } else { "half" };
        for line in [
            format!("const int g = int(tgid.x), rb = (g % {}) * {tm}, ra = (g / {}) * {tn};", m / tm, m / tm),
            format!("auto tX = tensor((device {t} *){} + rb * {k}, dextents<int32_t, 2>({k}, {tm}), array<int, 2>({{1, {k}}}));", x.name()),
            format!("auto tW = tensor((device {t} *){} + ra * {k}, dextents<int32_t, 2>({k}, {tn}), array<int, 2>({{1, {k}}}));", w.name()),
            format!("mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent), false, true, {relaxed}, mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<4>> mm;"),
            "auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();".to_string(),
            "mm.run(tX, tW, cT);".to_string(),
            format!("auto tY = tensor((device float *){}, dextents<int32_t, 2>({n}, {m}), array<int, 2>({{1, {n}}}));", y.name()),
            "cT.store(tY.slice(ra, rb));".to_string(),
        ] {
            self.line(&line);
        }
    }

    // ── scalars of the probes ─────────────────────────────────────────────

    fn scalar_ty(t: Scalar) -> Ty {
        match t {
            Scalar::F32 => Ty::F32,
            Scalar::F16 => Ty::Half,
            Scalar::H2 => Ty::Half2,
            Scalar::Bf16 => Ty::Bfloat,
        }
    }

    /// An f32 converted to `t` (to both lanes of a `half2`), nearest even.
    pub fn to_scalar(&mut self, f: Word, t: Scalar) -> Word {
        let e = match t {
            Scalar::F32 => self.name(f).to_string(),
            Scalar::F16 => format!("half({})", self.name(f)),
            Scalar::H2 => format!("half2(half({0}), half({0}))", self.name(f)),
            Scalar::Bf16 => format!("bfloat({})", self.name(f)),
        };
        self.def(Self::scalar_ty(t), e)
    }

    /// The two f16 of a `uint` as a `half2` (`as_type`).
    pub fn u_to_h2(&mut self, w: Word) -> Word {
        assert_eq!(self.ty(w), Ty::U32);
        let e = format!("as_type<half2>({})", self.name(w));
        self.def(Ty::Half2, e)
    }

    /// A probe scalar widened to f32 (the sum of the lanes of a `half2`).
    pub fn scalar_to_f(&mut self, v: Word) -> Word {
        let e = match self.ty(v) {
            Ty::Half2 => format!("float({0}.x) + float({0}.y)", self.name(v)),
            _ => format!("float({})", self.name(v)),
        };
        self.def(Ty::F32, e)
    }

    /// A mutable variable of type `t`, its initial value already of that type.
    pub fn var_scalar(&mut self, t: Scalar, init: Word) -> Var {
        self.var(Self::scalar_ty(t), init)
    }

    /// `a·b + c` in the type of `a`: `fma` for f32, f16 and `half2`; for
    /// `bfloat`, which has no `fma`, the expression `a * b + c` (`fused`
    /// false), or its value through f32 (`fused` true: `bfloat(fma(float…))`).
    pub fn fma_scalar(&mut self, a: Word, b: Word, c: Word, fused: bool) -> Word {
        let ty = self.ty(a);
        let (x, y, z) = (self.name(a).to_string(), self.name(b).to_string(), self.name(c).to_string());
        let e = match ty {
            Ty::Bfloat if fused => format!("bfloat(fma(float({x}), float({y}), float({z})))"),
            Ty::Bfloat => format!("{x} * {y} + {z}"),
            _ => format!("fma({x}, {y}, {z})"),
        };
        self.def(ty, e)
    }
}

/// What an emitter returns: the kernel's name (one pipeline per distinct
/// name), its MSL source, its buffer count, its threads per threadgroup.
pub struct Emitted {
    pub name: String,
    pub source: String,
    pub bindings: u32,
    pub threads: u32,
    /// What each binding holds, as the kernel declares it first (`U16` for an
    /// [`Elem::Act`] of the `bf16` feature, `F32` otherwise; `U32` when unused).
    pub binds: Vec<Elem>,
    /// Compiled as MSL 4.0 (Metal 4's tensors).
    pub msl4: bool,
    /// The bindings the kernel writes; empty: unknown, every binding taken as written.
    pub writes: Vec<u32>,
}

/// The entry point name of a kernel: its name with anything that is not
/// alphanumeric replaced.
pub fn entry_name(name: &str) -> String {
    let mut s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, 'k');
    }
    s
}

/// Close a kernel into an [`Emitted`].
pub fn done(name: impl Into<String>, k: Kernel, bindings: u32) -> Emitted {
    let name = name.into();
    let threads = k.local_size();
    let binds = (0..bindings).map(|i| k.buffers.iter().find(|b| b.index == i && b.alias == 0).map_or(Elem::U32, |b| b.elem)).collect();
    let msl4 = k.msl4;
    let writes = k.writes.clone();
    let source = k.finish(&entry_name(&name));
    Emitted { name, source, bindings, threads, binds, msl4, writes }
}

/// `--dump-msl DIR`: write a kernel's source as `DIR/<name>.metal`.
pub fn dump(dir: Option<&std::path::Path>, name: &str, source: &str) {
    if let Some(d) = dir {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(d.join(format!("{}.metal", entry_name(name))), source);
    }
}
