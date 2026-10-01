//! Plain-English interpretation of training curves: the "Is it learning?" verdict, a time estimate and a
//! milestone label. Pure functions over held-out results, so they are easy to test and identical in the app and mock.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::units::{UNIFORM_NATS, nats_to_bits};

/// One held-out evaluation, in nats per character.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct EvalPoint {
    pub chars: f64,
    pub nats: f64,
    pub se: f64,
    pub train_nats: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Too few evaluations to say.
    WarmingUp,
    Learning,
    Plateau,
    /// Memorising the training text: it does better on text it has seen than text it has not.
    Overfitting,
    /// The loss jumped up or became invalid.
    Diverging,
}

/// What the UI may suggest. Keys map to copy in `ui/src/content/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AdviceKey {
    KeepGoing,
    WaitForFirstCheck,
    ReadMoreText,
    AddMoreVariedText,
    StopAndKeepBest,
    ResumeFromBest,
    LowerLearningRate,
    TryTinyModel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct VerdictReport {
    pub verdict: Verdict,
    /// Relative improvement of the held-out score over the recent window (positive = better).
    pub change_pct: f64,
    pub advice: Vec<AdviceKey>,
}

/// Judge the trend of the held-out score. Points must be in training order.
pub fn verdict(points: &[EvalPoint]) -> VerdictReport {
    let report = |verdict, change_pct, advice: Vec<AdviceKey>| VerdictReport { verdict, change_pct, advice };
    let Some(last) = points.last() else {
        return report(Verdict::WarmingUp, 0.0, vec![AdviceKey::WaitForFirstCheck]);
    };
    if !last.nats.is_finite() {
        return report(Verdict::Diverging, 0.0, vec![AdviceKey::ResumeFromBest, AdviceKey::LowerLearningRate]);
    }
    let finite: Vec<&EvalPoint> = points.iter().filter(|p| p.nats.is_finite()).collect();
    if finite.len() < 3 {
        return report(Verdict::WarmingUp, 0.0, vec![AdviceKey::WaitForFirstCheck]);
    }
    let n = finite.len();
    let window = &finite[n.saturating_sub(5)..];
    let first = window[0];
    let end = window[window.len() - 1];
    let change = (first.nats - end.nats) / first.nats.max(1e-9);
    let noise = 2.0 * (first.se.max(end.se)) / first.nats.max(1e-9);

    // Overfitting: a wide train/held-out gap while held-out rises and train keeps falling. This is checked before
    // divergence because a rising held-out score with a still-falling training score is memorisation, not a blow-up.
    if let (Some(tr_first), Some(tr_end)) = (first.train_nats, end.train_nats) {
        let gap = end.nats - tr_end;
        let heldout_up = end.nats > first.nats;
        let train_down = tr_end < tr_first;
        if gap > 0.4 && heldout_up && train_down {
            return report(
                Verdict::Overfitting,
                change * 100.0,
                vec![AdviceKey::AddMoreVariedText, AdviceKey::StopAndKeepBest, AdviceKey::TryTinyModel],
            );
        }
    }

    // Diverging: the held-out score jumped well above the best it ever reached.
    let best = finite.iter().map(|p| p.nats).fold(f64::INFINITY, f64::min);
    let rise = last.nats - best;
    if rise > 0.3 && rise / best > 0.1 {
        return report(
            Verdict::Diverging,
            -(rise / best) * 100.0,
            vec![AdviceKey::ResumeFromBest, AdviceKey::LowerLearningRate],
        );
    }

    if change > noise.max(0.01) {
        report(Verdict::Learning, change * 100.0, vec![AdviceKey::KeepGoing])
    } else {
        report(Verdict::Plateau, change * 100.0, vec![AdviceKey::ReadMoreText, AdviceKey::LowerLearningRate])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Eta {
    pub seconds: f64,
    pub chars: f64,
}

/// Extrapolate when the held-out score will reach `target_nats`, fitting `nats = a - b ln(chars)` to recent points.
/// Returns `None` when the trend is not clearly improving or the answer is too far out to be meaningful.
pub fn eta_to_target(points: &[EvalPoint], target_nats: f64, chars_per_sec: f64) -> Option<Eta> {
    let pts: Vec<&EvalPoint> = points.iter().filter(|p| p.nats.is_finite() && p.chars > 1.0).collect();
    if pts.len() < 4 || chars_per_sec <= 0.0 {
        return None;
    }
    let recent = &pts[pts.len().saturating_sub(8)..];
    let n = recent.len() as f64;
    let xs: Vec<f64> = recent.iter().map(|p| p.chars.ln()).collect();
    let ys: Vec<f64> = recent.iter().map(|p| p.nats).collect();
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    if sxx < 1e-9 {
        return None;
    }
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let slope = sxy / sxx; // negative when improving
    if slope >= -1e-4 {
        return None;
    }
    let a = my - slope * mx;
    let current = recent[recent.len() - 1];
    if current.nats <= target_nats {
        return Some(Eta { seconds: 0.0, chars: 0.0 });
    }
    let chars_needed = ((target_nats - a) / slope).exp();
    if !chars_needed.is_finite() || chars_needed > current.chars * 200.0 {
        return None;
    }
    let remaining = (chars_needed - current.chars).max(0.0);
    Some(Eta { seconds: remaining / chars_per_sec, chars: remaining })
}

/// Where the model is on the road from random guessing to fluent text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Milestone {
    /// 0-based index of the current milestone.
    pub index: u32,
    pub total: u32,
    pub key: String,
}

const MILESTONES: [(&str, f64); 6] = [
    ("random_guessing", f64::INFINITY),
    ("letter_frequencies", 5.0),
    ("common_words", 3.5),
    ("short_phrases", 2.5),
    ("simple_sentences", 1.8),
    ("fluent_sentences", 1.3),
];

/// Milestone for a held-out score in bits per character (lower is better).
pub fn milestone(bits_per_char: f64) -> Milestone {
    let mut index = 0;
    for (i, (_, threshold)) in MILESTONES.iter().enumerate() {
        if bits_per_char <= *threshold {
            index = i;
        }
    }
    Milestone { index: index as u32, total: MILESTONES.len() as u32, key: MILESTONES[index].0.to_string() }
}

/// A held-out score in plain words: how many characters the model is "choosing between" on average.
pub fn effective_choices(nats: f64) -> f64 {
    nats.exp()
}

/// Bits per character of a model that guesses uniformly (the starting point), for progress rings.
pub fn start_bits() -> f64 {
    nats_to_bits(UNIFORM_NATS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(chars: f64, nats: f64) -> EvalPoint {
        EvalPoint { chars, nats, se: 0.01, train_nats: None }
    }

    fn curve(n: usize) -> Vec<EvalPoint> {
        (1..=n).map(|i| pt(i as f64 * 1e6, 5.0 - 0.9 * (i as f64).ln())).collect()
    }

    #[test]
    fn few_points_means_warming_up() {
        assert_eq!(verdict(&[]).verdict, Verdict::WarmingUp);
        assert_eq!(verdict(&curve(2)).verdict, Verdict::WarmingUp);
    }

    #[test]
    fn falling_loss_is_learning() {
        let r = verdict(&curve(8));
        assert_eq!(r.verdict, Verdict::Learning);
        assert!(r.change_pct > 0.0);
        assert!(r.advice.contains(&AdviceKey::KeepGoing));
    }

    #[test]
    fn flat_loss_is_plateau() {
        let pts: Vec<_> = (1..=8).map(|i| pt(i as f64 * 1e6, 2.0 + 0.001 * (i % 2) as f64)).collect();
        assert_eq!(verdict(&pts).verdict, Verdict::Plateau);
    }

    #[test]
    fn jump_up_is_diverging() {
        let mut pts = curve(6);
        pts.push(pt(7e6, 4.8));
        assert_eq!(verdict(&pts).verdict, Verdict::Diverging);
    }

    #[test]
    fn nan_is_diverging() {
        let mut pts = curve(6);
        pts.push(pt(7e6, f64::NAN));
        let r = verdict(&pts);
        assert_eq!(r.verdict, Verdict::Diverging);
        assert!(r.advice.contains(&AdviceKey::ResumeFromBest));
    }

    #[test]
    fn widening_gap_with_rising_heldout_is_overfitting() {
        let pts: Vec<_> = (0..6)
            .map(|i| EvalPoint {
                chars: (i + 1) as f64 * 1e6,
                nats: 1.5 + 0.08 * i as f64,
                se: 0.01,
                train_nats: Some(1.2 - 0.1 * i as f64),
            })
            .collect();
        assert_eq!(verdict(&pts).verdict, Verdict::Overfitting);
    }

    #[test]
    fn eta_extrapolates_a_log_curve() {
        let pts = curve(8); // nats = 5 - 0.9 ln(i), chars = i * 1e6
        let target = 3.0;
        let eta = eta_to_target(&pts, target, 10_000.0).expect("eta");
        // exact: i* = exp((5-3)/0.9) ~ 9.23 -> 9.23e6 chars total, 1.23e6 more than the last point (8e6)
        assert!((eta.chars - 1.23e6).abs() < 0.1e6, "chars {}", eta.chars);
        assert!((eta.seconds - eta.chars / 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn eta_none_when_not_improving_or_target_far_away() {
        let flat: Vec<_> = (1..=8).map(|i| pt(i as f64 * 1e6, 2.0)).collect();
        assert!(eta_to_target(&flat, 1.0, 1000.0).is_none());
        // An unreachable target (below zero loss) is "too far out to say", not a huge number.
        assert!(eta_to_target(&curve(8), -3.0, 1000.0).is_none());
        assert!(eta_to_target(&curve(2), 1.0, 1000.0).is_none());
    }

    #[test]
    fn milestones_advance_as_bits_fall() {
        assert_eq!(milestone(start_bits()).key, "random_guessing");
        assert_eq!(milestone(4.2).key, "letter_frequencies");
        assert_eq!(milestone(3.0).key, "common_words");
        assert_eq!(milestone(1.0).key, "fluent_sentences");
        assert_eq!(milestone(1.0).index, 5);
    }
}
