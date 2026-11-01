//! The checkpoint's projections as Q4_0, built at load: herbert-vk-q4's
//! `quantize_q4`, same grid, same layout, same choice of scale.

use crate::core::bf16::bf16_to_f32;
use half::f16;

/// A matrix quantized: the bytes `[data (n·k/2, padded to 256)][scales (n·k/32 f16)]`,
/// the scales' offset, and the sums of squares of the weights and of the error.
pub struct Q4 {
    pub bytes: Vec<u8>,
    pub scales: u64,
    pub ss_w: f64,
    pub ss_e: f64,
}

/// `raw` (`[n][k]` bf16 bits, as the checkpoint has them) as Q4_0: per block
/// of 32 along `k`, `w ≈ d·q` with `q` in `[-8, 7]` stored as a two's
/// complement nibble and `d` an f16. A block is one `uint4`; byte `b` of its
/// word `i` holds weight `8i + b` in its low nibble and `8i + 4 + b` in its
/// high one. `d` is the one of the 15 candidates `m / c` (`m` the signed value
/// of largest magnitude, `c` in `-8..=7`, `c ≠ 0`) that reconstructs the block
/// with the least squared error: a QAT checkpoint (`…-qat-q4_0-unquantized`)
/// sits on such a grid and comes back exactly, up to the f16 rounding of `d`.
pub fn quantize(raw: &[u8], n: usize, k: usize) -> Q4 {
    assert!(k.is_multiple_of(32) && raw.len() == 2 * n * k, "q4: {n}x{k}, {} bytes", raw.len());
    let nk2 = n * k / 2;
    let so = (nk2 as u64).next_multiple_of(256);
    let nblk = k / 32;
    let mut out = vec![0u8; so as usize + n * nblk * 2];
    let (data, scales) = out.split_at_mut(so as usize);
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(32);
    let per = n.div_ceil(threads);
    let mut sums = Vec::new();
    std::thread::scope(|s| {
        let (mut d, mut sc) = (&mut data[..nk2], &mut scales[..]);
        let mut handles = Vec::new();
        for t in 0..threads {
            let r0 = t * per;
            if r0 >= n {
                break;
            }
            let r1 = ((t + 1) * per).min(n);
            let (dh, dr) = d.split_at_mut((r1 - r0) * k / 2);
            d = dr;
            let (sh, sr) = sc.split_at_mut((r1 - r0) * nblk * 2);
            sc = sr;
            let src = &raw[2 * r0 * k..2 * r1 * k];
            handles.push(s.spawn(move || {
                let (mut ss_w, mut ss_e) = (0f64, 0f64);
                let mut w = [0f32; 32];
                let mut q = [0u8; 32];
                let mut best_q = [0u8; 32];
                for blk in 0..(r1 - r0) * nblk {
                    let off = 64 * blk;
                    let mut m = 0f32;
                    for (i, wi) in w.iter_mut().enumerate() {
                        *wi = bf16_to_f32(u16::from_le_bytes([src[off + 2 * i], src[off + 2 * i + 1]]));
                        if wi.abs() > m.abs() {
                            m = *wi;
                        }
                        ss_w += (*wi as f64) * (*wi as f64);
                    }
                    let (mut best_e, mut best_d) = (f64::INFINITY, 0u16);
                    if m != 0.0 {
                        for c in [-8i32, 7, -7, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1] {
                            let dh16 = f16::from_f32(m / c as f32);
                            let dq = dh16.to_f32();
                            if dq == 0.0 || !dq.is_finite() {
                                continue;
                            }
                            let inv = 1.0 / dq;
                            let mut e = 0f64;
                            for (i, wi) in w.iter().enumerate() {
                                let qi = (wi * inv).round().clamp(-8.0, 7.0);
                                q[i] = (qi as i32 + 8) as u8;
                                let r = (*wi - dq * qi) as f64;
                                e += r * r;
                            }
                            if e < best_e {
                                (best_e, best_d) = (e, dh16.to_bits());
                                best_q = q;
                            }
                        }
                    } else {
                        (best_e, best_q) = (0.0, [8u8; 32]);
                    }
                    if !best_e.is_finite() {
                        // Every candidate scale rounds to 0 in f16: the block reads as zeros.
                        (best_e, best_q) = (w.iter().map(|&v| (v as f64) * (v as f64)).sum(), [8u8; 32]);
                    }
                    ss_e += best_e;
                    for i in 0..4 {
                        for b in 0..4 {
                            dh[16 * blk + 4 * i + b] = ((best_q[8 * i + b] ^ 8) & 15) | (((best_q[8 * i + 4 + b] ^ 8) & 15) << 4);
                        }
                    }
                    sh[2 * blk..2 * blk + 2].copy_from_slice(&best_d.to_le_bytes());
                }
                (ss_w, ss_e)
            }));
        }
        sums = handles.into_iter().map(|h| h.join().expect("q4 quantiser thread")).collect();
    });
    let (ss_w, ss_e) = sums.iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    Q4 { bytes: out, scales: so, ss_w, ss_e }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::bf16::f32_to_bf16;

    /// A block on a Q4 grid (`d` a power of two, exact in bf16 and f16)
    /// comes back exactly: zero error, and every nibble decodes to its `q`.
    #[test]
    fn grid_block_is_exact() {
        let d = 1.0f32 / 64.0;
        let qs: Vec<i32> = (0..32).map(|i| (i * 7 % 16) - 8).collect();
        let raw: Vec<u8> = qs.iter().flat_map(|&q| f32_to_bf16(d * q as f32).to_le_bytes()).collect();
        let r = quantize(&raw, 1, 32);
        assert_eq!(r.ss_e, 0.0);
        let so = r.scales as usize;
        assert_eq!(f16::from_bits(u16::from_le_bytes([r.bytes[so], r.bytes[so + 1]])).to_f32().abs(), d);
        let sign = if f16::from_bits(u16::from_le_bytes([r.bytes[so], r.bytes[so + 1]])).to_f32() < 0.0 { -1 } else { 1 };
        for (e, &q) in qs.iter().enumerate() {
            let (i, j) = (e / 8, e % 8);
            let byte = r.bytes[4 * i + (j % 4)];
            let nib = if j < 4 { byte & 15 } else { byte >> 4 };
            assert_eq!(sign * (((nib ^ 8) as i32) - 8), q, "weight {e}");
        }
    }
}
