//! The conversation: streaming detokenization, the Gemma-4 and Qwen3.5 sessions.

pub mod detok;
pub mod gemma;
pub mod qwen;

#[derive(Default, Debug, Clone)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub generated_tokens: usize,
    pub decode_seconds: f64,
    pub position: usize,
}

impl TurnStats {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_seconds.max(1e-9)
    }
    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_seconds.max(1e-9)
    }
}

/// `HERBERT_METAL_TOP=N`: print the N best logits of every generated position
/// on stderr, to read the margin where two engines part.
pub(crate) fn top_from_env() -> usize {
    std::env::var("HERBERT_METAL_TOP").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

pub(crate) fn show_top(n: usize, step: usize, logits: &[u16]) {
    if n == 0 {
        return;
    }
    let mut v: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &b)| (crate::core::bf16::h16_to_f32(b), i)).collect();
    v.select_nth_unstable_by(n - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    v.truncate(n);
    v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let list: Vec<String> = v.iter().map(|(l, i)| format!("{i}:{l}")).collect();
    eprintln!("[top {step}] {}", list.join(" "));
}
