//! Choosing the next character.
//!
//! Nothing here draws a random number: the choice is the highest-scoring character, held off from loops by an
//! **adaptation trace** — what was just said is suppressed, and the suppression decays, the way a neuron that has just
//! fired is briefly harder to fire again. At the default decay of 0.88 the trace's half-life is 5.4 characters, so one
//! use of a character costs 2.5 logits now and 0.9 eight characters later (invisible to ordinary text), while a
//! character stuck in a loop never gets to decay and converges to `strength / (1 - decay)` = 20.8 logits, which no loop
//! survives. Variety comes from the state the model is in (which experts are loaded, how many rows it took, what it has
//! just read), not from sampling noise.

use minagi_types::DecodingConfig;

/// Only this many of the most recent characters feed the trace (0.88^64 is 3e-4: older ones are already nothing).
pub const ADAPT_WINDOW: usize = 64;

/// How to pick: the plain argmax (`raw`) or with the trace (`adapted`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decode {
    pub adapt_strength: f64,
    pub adapt_decay: f64,
    pub rep_penalty: f64,
}

impl Decode {
    /// Plain greedy: exactly what the model predicts.
    pub const RAW: Decode = Decode { adapt_strength: 0.0, adapt_decay: 0.88, rep_penalty: 1.0 };

    pub fn adapted(c: &DecodingConfig) -> Self {
        Self { adapt_strength: c.adapt_strength, adapt_decay: c.adapt_decay, rep_penalty: c.rep_penalty }
    }
}

/// The score the trace subtracts from each token, given the history (oldest first).
pub fn adaptation_trace(vocab: usize, prev: &[u32], decay: f64) -> Vec<f32> {
    let n = prev.len().min(ADAPT_WINDOW);
    let tail = &prev[prev.len() - n..];
    let mut trace = vec![0f32; vocab];
    for (i, &t) in tail.iter().enumerate() {
        if (t as usize) < vocab {
            trace[t as usize] += decay.powi((n - 1 - i) as i32) as f32;
        }
    }
    trace
}

/// The scores after the adaptation trace and the repetition penalty: what [`pick_next`] takes the argmax of.
pub fn adjust(logits: &[f32], prev: &[u32], d: &Decode) -> Vec<f32> {
    let mut l: Vec<f32> = logits.to_vec();
    if d.adapt_strength != 0.0 && !prev.is_empty() {
        let trace = adaptation_trace(l.len(), prev, d.adapt_decay);
        for (x, t) in l.iter_mut().zip(&trace) {
            *x -= d.adapt_strength as f32 * t;
        }
    }
    if d.rep_penalty != 0.0 && d.rep_penalty != 1.0 && !prev.is_empty() {
        let mut seen = vec![false; l.len()];
        for &t in prev {
            if (t as usize) < l.len() {
                seen[t as usize] = true;
            }
        }
        for (x, s) in l.iter_mut().zip(seen) {
            if s {
                *x = if *x > 0.0 { *x / d.rep_penalty as f32 } else { *x * d.rep_penalty as f32 };
            }
        }
    }
    l
}

/// The next token: argmax of `logits` after the adaptation trace and the repetition penalty (ties go to the lowest id).
pub fn pick_next(logits: &[f32], prev: &[u32], d: &Decode) -> u32 {
    let l = adjust(logits, prev, d);
    let mut best = 0usize;
    for (i, &x) in l.iter().enumerate() {
        if x > l[best] {
            best = i;
        }
    }
    best as u32
}

/// The share of this text's n-grams that were seen earlier in it (the repeat rate the UI shows; lower is better).
pub fn repeat_rate(text: &[u8], n: usize) -> f32 {
    if text.len() < n {
        return 0.0;
    }
    let mut seen = std::collections::HashSet::new();
    let mut repeats = 0usize;
    let total = text.len() - n + 1;
    for i in 0..total {
        if !seen.insert(&text[i..i + n]) {
            repeats += 1;
        }
    }
    repeats as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_is_plain_argmax_and_ties_go_to_the_lowest_id() {
        let l = [0.1, 2.0, 2.0, -1.0];
        assert_eq!(pick_next(&l, &[1, 1, 1], &Decode::RAW), 1);
    }

    #[test]
    fn the_trace_suppresses_recent_characters_and_fades() {
        let d = Decode { adapt_strength: 2.5, adapt_decay: 0.88, rep_penalty: 1.0 };
        let l = [3.0, 2.0, 0.0];
        // token 0 was just said: 3.0 - 2.5 = 0.5 < 2.0, so token 1 wins
        assert_eq!(pick_next(&l, &[0], &d), 1);
        // said twelve characters ago the trace has faded to 2.5 * 0.88^11 = 0.61: 3.0 - 0.61 = 2.39 beats 2.0 again
        let mut hist = vec![0u32];
        hist.extend([2u32; 11]);
        assert_eq!(pick_next(&l, &hist, &d), 0);
        // ...but at eight characters it is still 1.98 and loses
        let mut recent = vec![0u32];
        recent.extend([2u32; 7]);
        assert_eq!(pick_next(&l, &recent, &d), 1);
        // a character stuck in a loop converges to strength / (1 - decay)
        let looped = vec![0u32; 64];
        let t = adaptation_trace(3, &looped, 0.88);
        assert!((t[0] as f64 * 2.5 - 2.5 / (1.0 - 0.88)).abs() < 0.1);
        assert_eq!(pick_next(&l, &looped, &d), 1);
    }

    #[test]
    fn the_trace_only_looks_at_the_last_window() {
        let mut prev = vec![0u32; 200];
        prev.extend(vec![1u32; 64]);
        let t = adaptation_trace(3, &prev, 0.88);
        assert_eq!(t[0], 0.0, "everything older than the window is already nothing");
        assert!(t[1] > 0.0);
    }

    #[test]
    fn repetition_penalty_pulls_seen_tokens_toward_zero() {
        let d = Decode { adapt_strength: 0.0, adapt_decay: 0.88, rep_penalty: 2.0 };
        // 4.0 /2 = 2.0 < 3.0: the seen token loses; a seen negative logit is pushed further down
        assert_eq!(pick_next(&[4.0, 3.0], &[0], &d), 1);
        assert_eq!(pick_next(&[-1.0, -1.5], &[0], &d), 1);
    }

    #[test]
    fn repeat_rate_counts_ngrams_seen_before() {
        assert_eq!(repeat_rate(b"abcdefgh", 8), 0.0);
        assert_eq!(repeat_rate(b"abab", 8), 0.0);
        let r = repeat_rate(b"aaaaaaaaaaaa", 4); // 9 windows, 8 of them repeats
        assert!((r - 8.0 / 9.0).abs() < 1e-6);
    }
}
