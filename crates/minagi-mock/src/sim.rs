//! Deterministic learning-curve and text simulation.

/// SplitMix64: tiny, seedable, good enough for simulation noise.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo).max(1) as u64) as usize
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// What kind of run to simulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// Steady power-law improvement.
    Normal,
    /// Improves, then stalls.
    Plateau,
    /// Improves, then the loss blows up and becomes NaN.
    Diverge,
    /// Held-out gets worse while training keeps improving.
    Overfit,
    /// Fails immediately with an out-of-memory error.
    OomAtStart,
    /// Fails immediately: disk full.
    DiskFull,
    /// Reports no GPU backend.
    NoGpu,
}

impl Scenario {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "normal" => Scenario::Normal,
            "plateau" => Scenario::Plateau,
            "diverge" => Scenario::Diverge,
            "overfit" => Scenario::Overfit,
            "oom" | "oom_at_start" => Scenario::OomAtStart,
            "diskfull" | "disk_full" => Scenario::DiskFull,
            "nogpu" | "no_gpu" => Scenario::NoGpu,
            _ => return None,
        })
    }
}

/// The learning curve for a run: held-out and training loss as a function of characters read.
#[derive(Debug, Clone)]
pub struct Curve {
    pub scenario: Scenario,
    /// Loss of the untrained model.
    pub start: f64,
    /// Asymptote the curve approaches.
    pub floor: f64,
    /// Characters over which the first big improvement happens.
    pub scale: f64,
    pub alpha: f64,
}

impl Curve {
    pub fn for_scenario(scenario: Scenario) -> Self {
        Self { scenario, start: 5.58, floor: 0.9, scale: 2.0e5, alpha: 0.45 }
    }

    /// Smooth held-out loss in nats, with a per-domain offset.
    pub fn heldout(&self, chars: f64, domain_offset: f64) -> f64 {
        let base = self.floor + (self.start - self.floor) / (1.0 + chars / self.scale).powf(self.alpha);
        let mut v = base + domain_offset * (1.0 - (-chars / 4.0e6).exp()).max(0.05);
        match self.scenario {
            Scenario::Plateau => {
                let stall = 1.2e6;
                if chars > stall {
                    v = self.heldout_inner(stall, domain_offset);
                }
            }
            Scenario::Overfit => {
                let turn = 2.0e6;
                if chars > turn {
                    v = self.heldout_inner(turn, domain_offset) + 0.45 * ((chars - turn) / 2.0e6).min(2.0);
                }
            }
            Scenario::Diverge => {
                let blow = 2.5e6;
                if chars > blow {
                    v = self.heldout_inner(blow, domain_offset) + 1.5 * ((chars - blow) / 3.0e5).min(4.0);
                }
            }
            _ => {}
        }
        v
    }

    fn heldout_inner(&self, chars: f64, domain_offset: f64) -> f64 {
        let base = self.floor + (self.start - self.floor) / (1.0 + chars / self.scale).powf(self.alpha);
        base + domain_offset * (1.0 - (-chars / 4.0e6).exp()).max(0.05)
    }

    /// Training loss: a little below held-out, with the gap widening over time (much faster when overfitting).
    pub fn train(&self, chars: f64, domain_offset: f64) -> f64 {
        let held = self.heldout_inner(chars.min(self.stall_point()), domain_offset);
        let gap = match self.scenario {
            Scenario::Overfit => 0.05 + 0.5 * (chars / 4.0e6).min(1.5),
            _ => 0.04 + 0.08 * (chars / 2.0e7).min(1.0),
        };
        match self.scenario {
            Scenario::Plateau => held - gap,
            _ => (self.heldout_inner(chars, domain_offset) - gap).max(0.05),
        }
    }

    fn stall_point(&self) -> f64 {
        match self.scenario {
            Scenario::Plateau => 1.2e6,
            _ => f64::MAX,
        }
    }

    /// True once a Diverge run has blown up for good.
    pub fn is_nan_at(&self, chars: f64) -> bool {
        self.scenario == Scenario::Diverge && chars > 3.3e6
    }
}

const STORY: &str = "Once upon a time, there was a little boy named Tom. One day he found a shiny red ball in the garden. \
He picked it up and ran to show his mom. She smiled and said, \"What a lovely ball!\" Tom played with it all afternoon.";
const MATH: &str = "add 4917 + 388 = <think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305\n\
mul 37 * 12 = <think> 37*2=74 37*10=370 74+370=444 </think> 444";
const CHAT: &str = "<user>\nWhat are you?\n</user>\n<bot>\nI am a small language model that learns by reading text, one character at a time.\n</bot>";
const CODE: &str = "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1";
const GENERIC: &str =
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the hills and the wind grows quiet.";

pub fn canned_text(domain: &str) -> &'static str {
    let d = domain.to_ascii_lowercase();
    if d.contains("stor") {
        STORY
    } else if d.contains("arith") || d.contains("math") {
        MATH
    } else if d.contains("chat") || d.contains("know") {
        CHAT
    } else if d.contains("code") {
        CODE
    } else {
        GENERIC
    }
}

pub fn default_prompt(domain: &str) -> String {
    let d = domain.to_ascii_lowercase();
    if d.contains("stor") {
        "Once upon a time, there was a little boy named Tom. One day he ".into()
    } else if d.contains("arith") || d.contains("math") {
        "add 4917 + 388 = ".into()
    } else if d.contains("chat") {
        "<user>\nWhat are you?\n</user>\n<bot>\n".into()
    } else if d.contains("code") {
        "def merge_sorted(a, b):\n    ".into()
    } else {
        "The quick brown fox ".into()
    }
}

/// Text a model at loss `nats` might write: noise when untrained, words when partly trained, clean when well trained.
pub fn write_text(rng: &mut Rng, domain: &str, nats: f64, adapted: bool, n: usize) -> String {
    let canon: Vec<char> = canned_text(domain).chars().collect();
    // 0 = clean, 1 = pure noise
    let mut p = ((nats - 1.1) / 4.2).clamp(0.0, 1.0);
    if adapted {
        p *= 0.9;
    }
    let alphabet: Vec<char> = "abcdefghijklmnopqrstuvwxyz      eeetaoinsh".chars().collect();
    let mut out = String::new();
    let mut i = rng.range(0, canon.len().max(1));
    for _ in 0..n {
        let c = canon[i % canon.len()];
        i += 1;
        if rng.f64() < p {
            out.push(alphabet[rng.range(0, alphabet.len())]);
        } else {
            out.push(c);
        }
        // Untrained raw output loops on a short pattern; the repetition guard avoids it.
        if !adapted && p > 0.25 && rng.f64() < 0.02 {
            i = i.saturating_sub(rng.range(3, 12));
        }
    }
    out
}

/// Percentage of repeated 8-grams in `s`.
pub fn rep8(s: &str) -> f32 {
    let b: Vec<char> = s.chars().collect();
    if b.len() < 9 {
        return 0.0;
    }
    let mut seen = std::collections::HashSet::new();
    let mut rep = 0usize;
    let total = b.len() - 7;
    for w in b.windows(8) {
        if !seen.insert(w.to_vec()) {
            rep += 1;
        }
    }
    100.0 * rep as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_curve_falls_monotonically_toward_the_floor() {
        let c = Curve::for_scenario(Scenario::Normal);
        let mut prev = f64::MAX;
        for k in 0..30 {
            let v = c.heldout(10f64.powf(3.0 + k as f64 * 0.2), 0.0);
            assert!(v < prev + 1e-9);
            prev = v;
        }
        assert!(c.heldout(0.0, 0.0) > 5.4);
        assert!(c.heldout(3.0e6, 0.0) < 2.5, "tiny preset should reach about 2 nats within minutes");
        assert!(c.heldout(1e9, 0.0) > c.floor - 1e-6);
    }

    #[test]
    fn plateau_stalls_and_overfit_rises() {
        let p = Curve::for_scenario(Scenario::Plateau);
        assert!((p.heldout(5e6, 0.0) - p.heldout(1.2e6, 0.0)).abs() < 1e-9);
        let o = Curve::for_scenario(Scenario::Overfit);
        assert!(o.heldout(6e6, 0.0) > o.heldout(2e6, 0.0) + 0.3);
        assert!(o.train(6e6, 0.0) < o.heldout(6e6, 0.0) - 0.4, "overfit gap must be wide");
    }

    #[test]
    fn text_quality_improves_as_loss_falls() {
        let mut rng = Rng::new(7);
        let noisy = write_text(&mut rng, "stories", 5.0, false, 400);
        let clean = write_text(&mut rng, "stories", 1.2, true, 400);
        let canon = canned_text("stories");
        let hits = |s: &str| s.chars().zip(canon.chars().cycle()).filter(|(a, b)| a == b).count();
        assert!(clean.len() == 400 && noisy.len() == 400);
        assert!(clean.contains("Once upon a time") || clean.contains("little boy") || hits(&clean) > 0);
        assert!(
            clean.chars().filter(|c| c.is_uppercase()).count() >= noisy.chars().filter(|c| c.is_uppercase()).count()
        );
    }

    #[test]
    fn rng_is_deterministic_and_uniformish() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(1);
        assert_eq!(a.next_u64(), b.next_u64());
        let mean: f64 = (0..10_000).map(|_| a.f64()).sum::<f64>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02);
    }

    #[test]
    fn repeated_text_has_high_rep8() {
        assert!(rep8(&"abcdefgh".repeat(10)) > 80.0);
        assert!(rep8("the quick brown fox jumps over") < 1.0);
    }
}
