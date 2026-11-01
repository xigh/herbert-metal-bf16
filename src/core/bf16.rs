//! bf16 ↔ f32, as the CPU engine does them.

pub type Bf16 = u16;

/// Exact widening.
#[inline(always)]
pub fn bf16_to_f32(h: Bf16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Round to nearest even (`f32_to_bf16` of herbert-jit-bf16, bit for bit).
#[inline(always)]
pub fn f32_to_bf16(v: f32) -> Bf16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let mut result = bits >> 16;
    if round_bit == 1 && (sticky != 0 || lsb == 1) {
        result += 1;
    }
    result as u16
}

/// The bytes of a slice of plain numbers, little-endian as the GPU reads them.
pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: u16/u32/f32 have no padding and every byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ── the engine's 16-bit format ──────────────────────────────────────────────
//
// With the `f16` feature every 16-bit value the GPU keeps (weights,
// activations, K/V cache, logits, RoPE tables) is IEEE f16 instead of bf16
// (the f16 numerical contract); the kernels follow through
// `msl::set_f16_default`. The bf16 build stays the witness.

/// The `f16` feature: the engine's 16-bit format is f16.
pub const F16: bool = cfg!(feature = "f16");

/// IEEE half to f32, exactly.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize.
            let shift = m.leading_zeros() - 21;
            let m = (m << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (31, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half, round to nearest even, subnormals kept, ±Inf beyond
/// 65 504 (and from the ties above it), a NaN stays a (quiet) NaN — what MSL's
/// `half(x)` does (checked over every f32 pattern by `herbert-metal-probe round`).
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    // The value is m · 2^(e - 23 - 127) with m the 24-bit significand.
    let e = exp - 127;
    if e > 15 {
        return sign | 0x7c00;
    }
    let (m, e) = if exp == 0 { (man, -126) } else { (man | 0x80_0000, e) };
    // The f16 of exponent e (normal, e >= -14) keeps 11 bits: drop 13; below,
    // the subnormals' quantum is 2^-24: drop 13 + (-14 - e) bits.
    let drop = if e >= -14 { 13 } else { 13 + (-14 - e) } as u32;
    if drop > 25 {
        return sign; // below half the smallest subnormal: ±0
    }
    let kept = m >> drop;
    let rest = m & ((1u32 << drop) - 1);
    let half = 1u32 << (drop - 1);
    let mut q = kept;
    if rest > half || (rest == half && (kept & 1) == 1) {
        q += 1;
    }
    // Normal: q holds the hidden bit at bit 10 (or 11 after a carry, which
    // bumps the exponent); subnormal: q is the field itself (a carry into bit
    // 10 makes it the smallest normal, as the encoding wants).
    let h = if e >= -14 { (((e + 15) as u32) << 10) + (q - 0x400) } else { q };
    if h >= 0x7c00 {
        return sign | 0x7c00;
    }
    sign | h as u16
}

/// The engine's 16-bit value widened to f32.
#[inline]
pub fn h16_to_f32(h: u16) -> f32 {
    if F16 { f16_to_f32(h) } else { bf16_to_f32(h) }
}

/// f32 rounded to the engine's 16-bit format (nearest even).
#[inline]
pub fn f32_to_h16(v: f32) -> u16 {
    if F16 { f32_to_f16(v) } else { f32_to_bf16(v) }
}

/// bf16 → f16 for every bf16 pattern (one rounding: a bf16 is exact in f32).
fn bf16_to_f16_table() -> &'static [u16] {
    static T: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..=u16::MAX).map(|b| f32_to_f16(bf16_to_f32(b))).collect())
}

/// Little-endian bf16 bytes turned, in place, into the engine's format: f16
/// with the `f16` feature (nearest even), untouched otherwise. In parallel
/// over the cores for the large tensors.
pub fn bf16_bytes_to_h16(bytes: &mut [u8]) {
    if !F16 {
        return;
    }
    let t = bf16_to_f16_table();
    let one = |c: &mut [u8]| {
        for p in c.as_chunks_mut::<2>().0 {
            *p = t[u16::from_le_bytes(*p) as usize].to_le_bytes();
        }
    };
    const SLICE: usize = 8 << 20;
    if bytes.len() <= SLICE {
        one(bytes);
        return;
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let per = bytes.len().div_ceil(threads).next_multiple_of(2);
    std::thread::scope(|s| {
        for c in bytes.chunks_mut(per) {
            s.spawn(move || one(c));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan(), "{h:#06x}");
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x}");
            }
        }
    }

    /// Every midpoint between two neighbouring finite halves (exact in f32)
    /// goes to the even one, and one f32 ulp on either side to the nearer one;
    /// above the largest finite, the tie with Inf.
    #[test]
    fn f16_ties_and_neighbours() {
        for h in 0u16..0x7bff {
            let (a, b) = (f16_to_f32(h), f16_to_f32(h + 1));
            let mid = (a as f64 + b as f64) / 2.0;
            let m = mid as f32;
            assert_eq!(m as f64, mid, "{h:#06x}: midpoint not exact in f32");
            let even = if h & 1 == 0 { h } else { h + 1 };
            for s in [0x0000u16, 0x8000] {
                let neg = |f: f32| if s != 0 { -f } else { f };
                assert_eq!(f32_to_f16(neg(m)), even | s, "{h:#06x} tie");
                let below = f32::from_bits(m.to_bits() - 1);
                let above = f32::from_bits(m.to_bits() + 1);
                assert_eq!(f32_to_f16(neg(below)), h | s, "{h:#06x} below");
                assert_eq!(f32_to_f16(neg(above)), (h + 1) | s, "{h:#06x} above");
            }
        }
        // Half the largest finite's ulp above 65 504 is a tie with Inf: Inf.
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(65519.996), 0x7bff);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0);
        assert_eq!(f32_to_f16(f32::from_bits(2f32.powi(-25).to_bits() + 1)), 1);
        assert_eq!(f32_to_f16(f32::MIN_POSITIVE / 2.0), 0);
    }
}
