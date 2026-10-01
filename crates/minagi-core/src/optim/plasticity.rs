//! The learning rate, steered by held-out loss instead of by a schedule.
//!
//! A cosine schedule says "the run ends at step N". A model that keeps reading never ends, and when the data
//! changes (a new corpus, a different style) a step count would say the model is finished just when it is starting
//! over. So here the learning rate is a multiplier, the *scale*, that moves only when the held-out loss gives a
//! reason. It is nudged a little after **every** held-out evaluation, never in a staircase, by two rules that point
//! in opposite directions:
//!
//! * **Easing.** Held-out loss is fitted against evaluation number with an exponentially weighted least-squares
//!   line (no window, so nothing ever falls off an edge and makes the signal jump). The deciding number is an
//!   *effect size*, the improvement per evaluation measured in units of the scatter around the line, capped by
//!   the usual significance test. Clear improvement lets the scale creep up; no evidence of progress, or a
//!   decline, pulls it down, and a decline pulls harder. A second, shorter fit notices sooner when progress has
//!   stopped and can only ever lower the verdict.
//! * **Regime change.** Held-out loss that jumps by several standard errors *and is still up at the next
//!   evaluation* means the data changed, so the scale is stepped back up (doubled, up to the ceiling) and the
//!   evidence is gathered afresh. Requiring it to persist means one noisy evaluation cannot trigger it.
//!
//! Nothing here is a number the user must set: both thresholds come from the measured standard error of the
//! evaluation itself.
//!
//! This is a faithful port of `minagi/plasticity.py` (every constant and branch). [`Plasticity::state`] writes the
//! JSON the Python wrote under `plasticity` in `manifest.json`, and [`Plasticity::restore`] reads it, so a checkpoint
//! made by either implementation resumes identically in the other.
//!
//! How the training loop uses it: set the learning rate of every parameter group to `base_lr * factor()` before each
//! optimiser step, call [`Plasticity::tick`] after each step, and call [`Plasticity::observe`] after each held-out
//! evaluation. The 100-step warmup lives in [`Plasticity::factor`]; nothing else needs to apply it.
//!
//! The module also holds [`GradSnr`], a meter for how much of the gradient is signal. It is reported, not acted on.

use minagi_types::PlasticityEvent;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::VecDeque;

/// Running sums for one exponentially weighted least-squares fit of held-out loss against evaluation number.
///
/// `w` is the total weight, `w2` the total squared weight (together they give the effective sample size) and the
/// rest are the weighted sums a least-squares line needs. Field names match the Python's `S` and `F` dictionaries,
/// so these serialise to the same JSON.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Sums {
    pub w: f64,
    pub w2: f64,
    pub x: f64,
    pub y: f64,
    pub xx: f64,
    pub xy: f64,
    pub yy: f64,
}

impl Sums {
    /// Add one reading `(x, y)` after decaying everything already there by `lam`.
    fn accumulate(&mut self, lam: f64, x: f64, y: f64) {
        self.w *= lam;
        self.x *= lam;
        self.y *= lam;
        self.xx *= lam;
        self.xy *= lam;
        self.yy *= lam;
        self.w2 *= lam * lam;
        self.w += 1.0;
        self.w2 += 1.0;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.xy += x * y;
        self.yy += y * y;
    }

    /// Overwrite the fields named in a saved `S`/`F` object; anything missing or not a number keeps its value.
    fn update_from(&mut self, obj: &Map<String, Value>) {
        let slots: [(&str, &mut f64); 7] = [
            ("w", &mut self.w),
            ("w2", &mut self.w2),
            ("x", &mut self.x),
            ("y", &mut self.y),
            ("xx", &mut self.xx),
            ("xy", &mut self.xy),
            ("yy", &mut self.yy),
        ];
        for (key, slot) in slots {
            if let Some(v) = obj.get(key).and_then(Value::as_f64) {
                *slot = v;
            }
        }
    }

    /// `(t, n_eff, e)` for this set of sums: the significance of the slope, the effective number of readings, and
    /// the effect size. Signed so that **positive means held-out loss is falling**.
    ///
    /// `t` is the slope over its own standard error. `e` is the slope over the *residual scatter*: improvement per
    /// evaluation in units of the noise it must be seen through. `t` carries a factor that grows with the amount of
    /// evidence, so it rewards patience rather than progress; `e` has no such factor.
    fn fit(&self) -> (f64, f64, f64) {
        if self.w <= 2.0 {
            return (0.0, 0.0, 0.0);
        }
        let n_eff = self.w * self.w / py_max(self.w2, 1e-12);
        let sxx = self.xx - self.x * self.x / self.w;
        let sxy = self.xy - self.x * self.y / self.w;
        let syy = self.yy - self.y * self.y / self.w;
        if sxx <= 0.0 || n_eff <= 2.0 {
            return (0.0, n_eff, 0.0);
        }
        let slope = sxy / sxx;
        let resid = py_max(syy - slope * sxy, 0.0);
        let sigma = (resid / (n_eff - 2.0)).sqrt();
        if sigma <= 0.0 {
            // A perfectly straight run has no scatter, so neither statistic is defined. Report nothing rather
            // than infinity: real held-out is never this clean.
            return (0.0, n_eff, 0.0);
        }
        let t = -slope / (sigma / sxx.sqrt());
        let e = -slope / sigma;
        (py_max(-50.0, py_min(50.0, t)), n_eff, e)
    }
}

/// A line of the controller's own history: when it noticed something and what it did.
///
/// Persisted (the last 40) in the checkpoint. Not every nudge is recorded, only drift of more than 2% since the
/// last record, and every regime change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateEvent {
    /// Optimiser step at which it happened.
    pub at: u64,
    /// `"regime"`, `"improving"`, `"settling"` or `"deteriorating"`.
    pub kind: String,
    /// The held-out value that prompted it.
    pub val: f64,
    /// The learning-rate scale after the change.
    pub scale: f64,
    /// The evidence, rounded to two places (absent on regime changes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
}

/// What one held-out evaluation did to the controller.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// `false` when the reading was not a finite number and was ignored (nothing changed).
    pub counted: bool,
    /// The learning-rate scale after this evaluation (before the warmup in [`Plasticity::factor`]).
    pub scale: f64,
    /// The evidence the scale moves on, `t` (clamped to +-50). Positive means held-out is improving.
    pub t: f64,
    /// The effect size of the slow fit: improvement per evaluation in units of the scatter.
    pub effect: f64,
    /// Effective number of evaluations behind the slow fit.
    pub n_eff: f64,
    /// A regime change was confirmed on this evaluation and the scale was stepped back up.
    pub jumped: bool,
    /// The Python's note, exact text, present only when the rate moved notably (a regime change, or a drift of more
    /// than 2% since the last note). Meant for logs.
    pub note: Option<String>,
    /// What happened, in plain English, for every evaluation. Meant for the UI.
    pub reason: String,
}

impl Observation {
    /// The event the UI timeline shows for this evaluation.
    pub fn to_event(&self) -> PlasticityEvent {
        PlasticityEvent { rate_scale: self.scale, evidence_t: self.t, reason: self.reason.clone(), jumped: self.jumped }
    }
}

/// The evidence behind the rate right now, without observing anything.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evidence {
    /// The verdict `t` the rate moves on.
    pub t: f64,
    /// Effect size of the slow fit.
    pub effect: f64,
    /// Effective number of evaluations behind the slow fit.
    pub n_eff: f64,
}

/// The held-out-driven learning-rate controller. See the module docs for what it does and why.
#[derive(Debug, Clone)]
pub struct Plasticity {
    scale: f64,
    /// The slow fit: it is the evidence.
    slow: Sums,
    /// The fast fit: it exists only to notice sooner that improvement has stopped.
    fast: Sums,
    /// Evaluation counter, the x axis of both fits.
    i: f64,
    /// Recent standard errors of the evaluation (at most [`Self::SE_KEPT`]); their median is the noise level.
    se_hist: VecDeque<f64>,
    prev: Option<f64>,
    /// A candidate regime change, waiting to be confirmed or discarded at the next evaluation.
    jump_from: Option<f64>,
    events: Vec<RateEvent>,
    step: u64,
}

impl Default for Plasticity {
    fn default() -> Self {
        Self::new()
    }
}

impl Plasticity {
    /// The rate is never allowed to reach zero.
    pub const FLOOR: f64 = 0.05;
    /// Full rate.
    pub const CEIL: f64 = 1.0;
    /// Decay per evaluation of the slow fit; its effective sample size tends to 65.7.
    pub const LAM: f64 = 0.97;
    /// Decay of the fast fit; effective sample size tends to 12.3. Not lower: below `MIN_EFF` it would never engage.
    pub const LAM_FAST: f64 = 0.85;
    /// Converts the effect size into the units of `t` so the two can be compared.
    pub const EFFECT: f64 = 45.0;
    /// Effective observations needed before the controller acts at all.
    pub const MIN_EFF: f64 = 12.0;
    /// Log-scale gain per evaluation going up.
    pub const NUDGE: f64 = 0.005;
    /// Log-scale gain per evaluation going down. Larger on purpose: overshooting costs far more than waiting.
    pub const NUDGE_DOWN: f64 = 0.025;
    /// Width of the response on the way down (the verdict reaches about -12).
    pub const T_W_DOWN: f64 = 6.0;
    /// The verdict at which the rate neither eases up nor down. Deliberately above the middle.
    pub const T_MID: f64 = 2.2;
    /// Width of the response above `T_MID`.
    pub const T_W: f64 = 0.75;
    /// Standard errors that count as a regime change.
    pub const JUMP_SE: f64 = 4.0;
    /// ...and it has to still be up by this many next time.
    pub const HOLD_SE: f64 = 2.0;
    /// A jump this large is a regime change whatever the noise.
    pub const FLOOR_JUMP: f64 = 0.25;
    /// What a confirmed regime change multiplies the scale by.
    pub const UP: f64 = 2.0;
    /// Optimiser steps over which the rate ramps up from nearly zero.
    pub const WARMUP: u64 = 100;
    /// Standard errors remembered for the noise estimate.
    pub const SE_KEPT: usize = 64;
    /// Events kept in the saved state.
    pub const EVENTS_KEPT: usize = 40;

    /// A fresh controller at full rate.
    pub fn new() -> Self {
        Self {
            scale: 1.0,
            slow: Sums::default(),
            fast: Sums::default(),
            i: 0.0,
            se_hist: VecDeque::with_capacity(Self::SE_KEPT),
            prev: None,
            jump_from: None,
            events: Vec::new(),
            step: 0,
        }
    }

    /// The scale alone, without the warmup.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Restore full learning rate (`--lr-reset`). The evidence gathered so far is kept.
    pub fn reset_scale(&mut self) {
        self.scale = 1.0;
    }

    /// Optimiser steps counted so far (survives checkpoints).
    pub fn step(&self) -> u64 {
        self.step
    }

    /// What to multiply every parameter group's base learning rate by, right now: the scale, ramped in over the
    /// first [`Self::WARMUP`] steps as `min(1, (step + 1) / WARMUP)`.
    ///
    /// The warmup is only a safety net in case the optimiser moments were not restored, and the counter *is*
    /// restored, so a resume does not repeat it.
    pub fn factor(&self) -> f64 {
        let warm =
            if self.step < Self::WARMUP { (((self.step + 1) as f64) / Self::WARMUP as f64).min(1.0) } else { 1.0 };
        self.scale * warm
    }

    /// Count one optimiser step. Call it after every step, not only after evaluations.
    pub fn tick(&mut self) {
        self.step = self.step.saturating_add(1);
    }

    /// The noise level: the median of the recent standard errors of the held-out evaluation (0 when none yet).
    pub fn noise(&self) -> f64 {
        if self.se_hist.is_empty() {
            return 0.0;
        }
        let mut s: Vec<f64> = self.se_hist.iter().copied().collect();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
    }

    /// Add one reading to both fits; they share the x axis.
    fn accumulate(&mut self, y: f64) {
        self.i += 1.0;
        let x = self.i;
        self.slow.accumulate(Self::LAM, x, y);
        self.fast.accumulate(Self::LAM_FAST, x, y);
    }

    /// The number the rate actually moves on, with the slow fit's effect size and effective sample size.
    ///
    /// The minimum of three readings, because each can only say "less than you think" and none may say "more": the
    /// slow `t` (is it distinguishable from noise), `EFFECT * e_slow` (is it big enough to matter) and
    /// `EFFECT * e_fast` (is it still happening now). Only the slow `t` appears: `t` depends on the length of the
    /// window it was measured over, so the deliberately short fast fit would veto everything on its `t`; it
    /// contributes its effect size instead, which does not know the window length.
    fn verdict(&self) -> (f64, f64, f64) {
        let (t_s, n_s, e_s) = self.slow.fit();
        let (_, n_f, e_f) = self.fast.fit();
        let mut t = py_min(t_s, Self::EFFECT * e_s);
        if n_f >= Self::MIN_EFF {
            t = py_min(t, Self::EFFECT * e_f);
        }
        (py_max(-50.0, py_min(50.0, t)), n_s, e_s)
    }

    /// The evidence behind the rate right now.
    pub fn evidence(&self) -> Evidence {
        let (t, n_eff, effect) = self.verdict();
        Evidence { t, effect, n_eff }
    }

    /// The controller's recent history (the last 40 notable changes).
    pub fn events(&self) -> &[RateEvent] {
        &self.events
    }

    /// Feed one held-out evaluation: its mean loss `val` and the standard error `se` of that mean (pass `None`, or
    /// any non-finite or non-positive number, when unknown).
    ///
    /// The rate is nudged every time by an amount that varies smoothly with the evidence,
    /// `exp(gain * tanh((t - T_MID) / width))`, so it drifts rather than staircases. Going up and coming down are
    /// different questions: a rate that is too high *shows* (held-out turns and keeps turning), so deterioration is
    /// corrected in proportion to how bad it is; a rate that is merely working says nothing about whether a higher
    /// one would be better, so upward is a slow probe. They therefore differ in both gain and width.
    ///
    /// A reading that is not finite is ignored entirely ([`Observation::counted`] is `false`).
    pub fn observe(&mut self, val: f64, se: impl Into<Option<f64>>) -> Observation {
        if !val.is_finite() {
            let e = self.evidence();
            return Observation {
                counted: false,
                scale: self.scale,
                t: e.t,
                effect: e.effect,
                n_eff: e.n_eff,
                jumped: false,
                note: None,
                reason: "Ignored a held-out reading that was not a finite number; the learning rate is unchanged."
                    .to_string(),
            };
        }
        if let Some(se) = se.into().filter(|se| se.is_finite() && *se > 0.0) {
            if self.se_hist.len() == Self::SE_KEPT {
                self.se_hist.pop_front();
            }
            self.se_hist.push_back(se);
        }
        let s = self.noise();
        let mut note: Option<String> = None;
        let mut jumped = false;
        let mut jumped_from = 0.0;

        // REGIME: a jump, confirmed on the following evaluation.
        if let Some(from) = self.jump_from.take() {
            if val > from + Self::HOLD_SE * py_max(s, 1e-9) {
                let before = self.scale;
                self.scale = py_min(Self::CEIL, self.scale * Self::UP);
                note = Some(format!(
                    "held-out moved to {val:.4} from {from:.4} and stayed - new regime, rate {before:.3} -> {:.3}",
                    self.scale
                ));
                self.push_event(RateEvent { at: self.step, kind: "regime".into(), val, scale: self.scale, t: None });
                self.slow = Sums::default();
                self.fast = Sums::default();
                self.i = 0.0;
                jumped = true;
                jumped_from = from;
            }
        } else if let Some(prev) = self.prev {
            let bar = py_max(Self::JUMP_SE * s, Self::FLOOR_JUMP);
            if val - prev > bar {
                self.jump_from = Some(prev); // confirm or discard next time
            }
        }

        self.prev = Some(val);
        self.accumulate(val);
        let (t, n_eff, e) = self.verdict();

        // The nudge, every time, sized by the evidence; skipped on the evaluation that confirmed a regime change.
        let mut nudged = None;
        if note.is_none() && n_eff >= Self::MIN_EFF {
            let before = self.scale;
            let up = t >= Self::T_MID;
            let (gain, width) = if up { (Self::NUDGE, Self::T_W) } else { (Self::NUDGE_DOWN, Self::T_W_DOWN) };
            let f = (gain * ((t - Self::T_MID) / width).tanh()).exp();
            self.scale = py_max(Self::FLOOR, py_min(Self::CEIL, self.scale * f));
            // One line per evaluation would be noise: record only when the rate has drifted a full 2% since the
            // last thing recorded.
            let last = self.events.last().map_or(1.0, |ev| ev.scale);
            if (self.scale / py_max(last, 1e-9)).ln().abs() > 0.02 {
                let kind = if t > Self::T_MID {
                    "improving"
                } else if t < -Self::T_MID {
                    "deteriorating"
                } else {
                    "settling"
                };
                note = Some(format!(
                    "{kind}: t={t:+.2} (effect {e:+.3} per evaluation) over {n_eff:.0} effective evaluations, \
                     rate {before:.3} -> {:.3}",
                    self.scale
                ));
                self.push_event(RateEvent {
                    at: self.step,
                    kind: kind.into(),
                    val,
                    scale: self.scale,
                    t: Some(py_round(t, 2)),
                });
            }
            nudged = Some(t);
        }

        let reason = self.explain(jumped.then_some((jumped_from, val)), nudged, n_eff);
        Observation { counted: true, scale: self.scale, t, effect: e, n_eff, jumped, note, reason }
    }

    fn push_event(&mut self, ev: RateEvent) {
        // Only the tail is ever read or saved, so the history is bounded here instead of growing for the whole run.
        if self.events.len() >= Self::EVENTS_KEPT {
            self.events.remove(0);
        }
        self.events.push(ev);
    }

    /// The plain-English reading of one evaluation. `jump` is `(from, to)` when a regime change was just confirmed;
    /// `nudged` is the verdict when the controller was allowed to act.
    fn explain(&self, jump: Option<(f64, f64)>, nudged: Option<f64>, n_eff: f64) -> String {
        let scale = self.scale;
        if let Some((from, to)) = jump {
            return format!(
                "Held-out loss jumped from {from:.3} to {to:.3} and stayed there, so the data has probably changed. \
                 The learning rate was raised to x{scale:.2} of its base value and the evidence starts afresh."
            );
        }
        let Some(t) = nudged else {
            return format!(
                "Still gathering evidence ({n_eff:.0} of {:.0} evaluations needed), so the learning rate is held at \
                 x{scale:.2}.",
                Self::MIN_EFF
            );
        };
        let limit = if scale <= Self::FLOOR {
            " It is at its floor."
        } else if scale >= Self::CEIL {
            " It is at its ceiling."
        } else {
            ""
        };
        let pending = if self.jump_from.is_some() {
            " Held-out loss just rose sharply; if it is still up at the next evaluation the data has changed."
        } else {
            ""
        };
        if t >= Self::T_MID {
            format!(
                "Held-out loss is clearly improving, so the learning rate is easing up (x{scale:.2}).{limit}{pending}"
            )
        } else if t < -Self::T_MID {
            format!(
                "Held-out loss is getting worse, so the learning rate is being lowered (x{scale:.2}).{limit}{pending}"
            )
        } else {
            format!(
                "Progress is slow or flat, so the learning rate is easing down gently (x{scale:.2}).{limit}{pending}"
            )
        }
    }

    /// The state to write into `manifest.json` under `plasticity`: the same keys the Python wrote.
    ///
    /// `n`, `t` and `e` are informational (the current evidence, rounded) and are not read back.
    pub fn state(&self) -> Value {
        let (t, n_eff, e) = self.verdict();
        let from = self.events.len().saturating_sub(Self::EVENTS_KEPT);
        json!({
            "scale": self.scale,
            "prev": self.prev,
            "S": self.slow,
            "F": self.fast,
            "i": self.i,
            "se": self.se_hist.iter().collect::<Vec<_>>(),
            "step": self.step,
            "n": py_round(n_eff, 1),
            "t": py_round(t, 3),
            "e": py_round(e, 4),
            "events": &self.events[from..],
        })
    }

    /// Rebuild a controller from a saved [`Self::state`], whichever implementation wrote it.
    ///
    /// Never fails: `null`, `{}`, a non-object or a partial object give the defaults for whatever is missing.
    /// What the Python did is kept, including its quirks:
    ///
    /// * the counter `i` is only read when the slow sums are present;
    /// * a checkpoint without the fast sums leaves the fast fit empty (seeding it from the slow one would make the
    ///   cap never bind; the slow fit keeps steering meanwhile);
    /// * an older checkpoint that kept a plain window (`hist`) is replayed into the fit;
    /// * a pending regime-change candidate is not saved, so one in flight at checkpoint time is forgotten.
    ///
    /// The one deliberate difference: a scale that is not finite is read as 1 and the scale is clamped to
    /// `[FLOOR, CEIL]`, which no state written by either implementation ever leaves.
    pub fn restore(d: &Value) -> Self {
        let mut p = Self::new();
        let Some(obj) = d.as_object() else { return p };
        if obj.is_empty() {
            return p;
        }
        p.scale = match obj.get("scale").and_then(Value::as_f64) {
            Some(s) if s.is_finite() => s.clamp(Self::FLOOR, Self::CEIL),
            _ => 1.0,
        };
        p.prev = obj.get("prev").and_then(Value::as_f64);
        if let Some(s) = obj.get("S").and_then(Value::as_object) {
            p.slow.update_from(s);
            p.i = obj.get("i").and_then(Value::as_f64).unwrap_or(0.0);
            if let Some(f) = obj.get("F").and_then(Value::as_object) {
                p.fast.update_from(f);
            }
        } else if let Some(hist) = obj.get("hist").and_then(Value::as_array) {
            for v in hist.iter().filter_map(Value::as_f64) {
                p.accumulate(v);
            }
        }
        p.step = obj.get("step").and_then(Value::as_f64).map_or(0, |s| if s > 0.0 { s as u64 } else { 0 });
        if let Some(se) = obj.get("se").and_then(Value::as_array) {
            for v in se.iter().filter_map(Value::as_f64) {
                if p.se_hist.len() == Self::SE_KEPT {
                    p.se_hist.pop_front();
                }
                p.se_hist.push_back(v);
            }
        }
        if let Some(events) = obj.get("events").and_then(Value::as_array) {
            let all: Vec<RateEvent> = events.iter().filter_map(|v| serde_json::from_value(v.clone()).ok()).collect();
            let from = all.len().saturating_sub(Self::EVENTS_KEPT);
            p.events = all[from..].to_vec();
        }
        p
    }

    /// One line describing the controller, printed when a run starts.
    pub fn describe(&self) -> String {
        let s = self.noise();
        let head = format!(
            "learning rate is governed by held-out, not by a horizon: rate x{:.3}, floor x{}",
            self.scale,
            Self::FLOOR
        );
        if s > 0.0 { format!("{head}, noise {s:.4}") } else { head }
    }
}

/// Python's `min(a, b)`: `b` only when it is strictly smaller, so signed zeros and NaNs come out as they do there
/// (`f64::min` may return either zero, and ignores a NaN).
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python's `max(a, b)`: `b` only when it is strictly larger. See [`py_min`].
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's `round(x, digits)` for the numbers written into the checkpoint (round half to even on the exact value).
fn py_round(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// A meter for how much of the gradient is signal, measured every step it is fed.
///
/// Keeps an exponential average of the gradient and an average of its squared norm. If successive gradients agree,
/// the average keeps its length and the ratio `||mean||^2 / mean(||g||^2)` approaches one; if they are independent
/// noise the average shrinks towards zero and so does the ratio. It is the gradient noise scale of McCandlish et
/// al. 2018 in its cheapest form: one vector and two scalars.
///
/// The held-out signal the learning rate steers by needs over a hundred readings to say anything, so this asks the
/// same question of something available on every step. **Reported, not acted on.**
#[derive(Debug, Clone)]
pub struct GradSnr {
    beta: f64,
    /// Bias-uncorrected exponential average of the flattened gradient (`f32`, like the gradient itself).
    m: Vec<f32>,
    sq: f64,
    n: u64,
}

impl Default for GradSnr {
    fn default() -> Self {
        Self::new(0.98)
    }
}

impl GradSnr {
    /// Readings needed before [`Self::ratio`] says anything.
    pub const MIN_READINGS: u64 = 8;

    /// A meter averaging over roughly `1 / (1 - beta)` readings.
    pub fn new(beta: f64) -> Self {
        Self { beta, m: Vec::new(), sq: 0.0, n: 0 }
    }

    /// Feed one step's gradient, given as the gradient of each parameter that has one (flattened, in a fixed
    /// order). Returns the ratio, or `None` while there is nothing to say.
    ///
    /// No gradients at all returns `None` and changes nothing. If the total length differs from the previous call
    /// (the Python would crash) the meter starts over.
    pub fn observe(&mut self, grads: &[&[f32]]) -> Option<f64> {
        let len: usize = grads.iter().map(|g| g.len()).sum();
        if len == 0 {
            return None;
        }
        if self.n > 0 && self.m.len() != len {
            self.m.clear();
            self.n = 0;
            self.sq = 0.0;
        }
        let keep = self.beta as f32;
        let add = (1.0 - self.beta) as f32;
        if self.m.is_empty() {
            self.m.extend(grads.iter().flat_map(|g| g.iter().copied()));
        } else {
            for (m, &g) in self.m.iter_mut().zip(grads.iter().flat_map(|g| g.iter())) {
                *m = *m * keep + g * add;
            }
        }
        let s: f64 = grads.iter().flat_map(|g| g.iter()).map(|&g| f64::from(g) * f64::from(g)).sum();
        self.sq = if self.n == 0 { s } else { self.beta * self.sq + (1.0 - self.beta) * s };
        self.n += 1;
        self.ratio()
    }

    /// 0 = pure noise, 1 = every step pointing the same way. `None` until [`Self::MIN_READINGS`] readings.
    pub fn ratio(&self) -> Option<f64> {
        if self.m.is_empty() || self.sq <= 0.0 || self.n < Self::MIN_READINGS {
            return None;
        }
        let c = 1.0 - self.beta.powf(self.n as f64); // bias correction
        let mean_sq: f64 = self.m.iter().map(|&m| (f64::from(m) / c).powi(2)).sum();
        Some(mean_sq / (self.sq / c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(p: &mut Plasticity, vals: impl Iterator<Item = f64>, se: f64) -> Vec<Observation> {
        vals.map(|v| {
            for _ in 0..10 {
                p.tick();
            }
            p.observe(v, se)
        })
        .collect()
    }

    /// A deterministic wobble in [-1, 1] so the tests need no random number generator.
    fn wobble(i: usize) -> f64 {
        ((i as f64) * 12.9898).sin()
    }

    #[test]
    fn warmup_ramps_over_the_first_hundred_steps() {
        let mut p = Plasticity::new();
        assert!((p.factor() - 0.01).abs() < 1e-15);
        for _ in 0..49 {
            p.tick();
        }
        assert!((p.factor() - 0.5).abs() < 1e-15);
        for _ in 0..50 {
            p.tick();
        }
        assert_eq!(p.step(), 99);
        assert!((p.factor() - 1.0).abs() < 1e-15);
        p.tick();
        assert_eq!(p.factor(), 1.0);
    }

    #[test]
    fn factor_multiplies_the_scale() {
        let mut p = Plasticity::new();
        p.scale = 0.4;
        p.step = 1000;
        assert_eq!(p.factor(), 0.4);
        p.step = 9;
        assert!((p.factor() - 0.4 * 0.1).abs() < 1e-15);
    }

    #[test]
    fn empty_controller_has_no_evidence_and_a_quiet_state() {
        let p = Plasticity::new();
        let e = p.evidence();
        assert_eq!((e.t, e.effect, e.n_eff), (0.0, 0.0, 0.0));
        assert_eq!(p.noise(), 0.0);
        let s = p.state();
        assert_eq!(s["scale"], json!(1.0));
        assert!(s["prev"].is_null());
        assert_eq!(s["se"], json!([]));
        assert_eq!(s["events"], json!([]));
        assert_eq!(s["step"], json!(0));
        assert_eq!(p.describe(), "learning rate is governed by held-out, not by a horizon: rate x1.000, floor x0.05");
    }

    #[test]
    fn nan_and_infinite_readings_change_nothing() {
        let mut p = Plasticity::new();
        feed(&mut p, (0..20).map(|i| 3.0 - 0.02 * i as f64 + 0.01 * wobble(i)), 0.01);
        let before = p.state();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let o = p.observe(bad, 0.01);
            assert!(!o.counted && !o.jumped && o.note.is_none());
            assert!(!o.reason.is_empty());
            assert_eq!(p.state(), before);
        }
    }

    #[test]
    fn unusable_standard_errors_are_not_remembered() {
        let mut p = Plasticity::new();
        p.observe(2.0, None);
        p.observe(2.0, f64::NAN);
        p.observe(2.0, 0.0);
        p.observe(2.0, -0.5);
        p.observe(2.0, f64::INFINITY);
        assert_eq!(p.noise(), 0.0);
        p.observe(2.0, 0.02);
        p.observe(2.0, 0.04);
        assert!((p.noise() - 0.03).abs() < 1e-15, "median of two is their mean");
        p.observe(2.0, 0.5);
        assert!((p.noise() - 0.04).abs() < 1e-15, "median of three is the middle one");
    }

    #[test]
    fn only_the_last_64_standard_errors_are_kept() {
        let mut p = Plasticity::new();
        for k in 0..100 {
            p.observe(2.0, 0.01 + 0.001 * k as f64);
        }
        let se = p.state()["se"].as_array().cloned().unwrap_or_default();
        assert_eq!(se.len(), 64);
        assert_eq!(se[0].as_f64(), Some(0.01 + 0.001 * 36.0));
    }

    #[test]
    fn it_does_nothing_until_it_has_twelve_effective_evaluations() {
        let mut p = Plasticity::new();
        let obs = feed(&mut p, (0..8).map(|i| 2.0 + 0.001 * wobble(i)), 0.02);
        assert!(obs.iter().all(|o| o.scale == 1.0 && o.note.is_none()));
        assert!(obs[3].reason.contains("Still gathering evidence"));
    }

    #[test]
    fn a_plateau_eases_the_rate_down_but_never_below_the_floor() {
        let mut p = Plasticity::new();
        let obs = feed(&mut p, (0..400).map(|i| 2.0 + 0.02 * wobble(i)), 0.02);
        let last = &obs[obs.len() - 1];
        assert!(last.scale < 0.5, "should have eased down, got {}", last.scale);
        assert!(obs.iter().all(|o| o.scale >= Plasticity::FLOOR));
        assert!(obs.windows(2).all(|w| w[1].scale <= w[0].scale + 1e-12), "monotone down on a plateau");
    }

    #[test]
    fn deterioration_hits_the_floor_and_says_so() {
        let mut p = Plasticity::new();
        let obs = feed(&mut p, (0..300).map(|i| 2.0 + 0.03 * i as f64 + 0.01 * wobble(i)), 0.01);
        let last = &obs[obs.len() - 1];
        assert_eq!(last.scale, Plasticity::FLOOR);
        assert!(last.reason.contains("getting worse") && last.reason.contains("floor"), "{}", last.reason);
    }

    #[test]
    fn steady_improvement_climbs_back_to_the_ceiling_and_stops_there() {
        let mut p = Plasticity::new();
        p.scale = 0.3;
        let obs = feed(&mut p, (0..400).map(|i| 8.0 - 0.02 * i as f64 + 0.01 * wobble(i)), 0.01);
        assert_eq!(obs[obs.len() - 1].scale, Plasticity::CEIL);
        assert!(obs.iter().all(|o| o.scale <= Plasticity::CEIL));
        assert!(obs[obs.len() - 1].reason.contains("ceiling"));
    }

    #[test]
    fn a_regime_change_must_persist_to_count() {
        // improving, then one high reading that comes straight back: discarded
        let mut p = Plasticity::new();
        p.scale = 0.3;
        let mut vals: Vec<f64> = (0..60).map(|i| 3.0 - 0.015 * i as f64 + 0.01 * wobble(i)).collect();
        vals[40] += 1.5;
        let obs = feed(&mut p, vals.into_iter(), 0.012);
        assert!(obs.iter().all(|o| !o.jumped));

        // improving, then a level shift that stays: confirmed on the evaluation after the shift
        let mut q = Plasticity::new();
        q.scale = 0.3;
        let vals = (0..80).map(|i| if i < 50 { 2.0 } else { 3.2 } + 0.01 * wobble(i));
        let obs = feed(&mut q, vals, 0.012);
        let fired: Vec<usize> = obs.iter().enumerate().filter(|(_, o)| o.jumped).map(|(i, _)| i).collect();
        assert_eq!(fired, vec![51]);
        let o = &obs[51];
        assert!((o.scale - 2.0 * obs[50].scale).abs() < 1e-12, "the rate doubles and no nudge is added on top");
        assert!(o.note.as_deref().is_some_and(|n| n.contains("new regime")));
        assert!(o.reason.contains("probably changed"));
        assert_eq!(q.events().iter().filter(|e| e.kind == "regime").count(), 1);
    }

    #[test]
    fn a_regime_change_never_raises_the_scale_past_the_ceiling() {
        let mut p = Plasticity::new();
        let obs = feed(&mut p, (0..60).map(|i| if i < 30 { 2.0 } else { 4.0 }), 0.01);
        assert!(obs.iter().all(|o| o.scale <= 1.0));
    }

    #[test]
    fn state_round_trips_and_resumes_identically() {
        let mut a = Plasticity::new();
        feed(&mut a, (0..120).map(|i| 3.0 - 0.01 * i as f64 + 0.02 * wobble(i)), 0.02);
        let mut b = Plasticity::restore(&a.state());
        assert_eq!(a.state(), b.state());
        assert_eq!(a.factor(), b.factor());
        for i in 120..180 {
            let v = 1.8 + 0.02 * wobble(i);
            let (x, y) = (a.observe(v, 0.02), b.observe(v, 0.02));
            assert_eq!(x, y);
        }
        assert_eq!(a.state(), b.state());
    }

    /// Compare two JSON values, numbers to a relative `tol`, everything else exactly.
    fn json_close(a: &Value, b: &Value, tol: f64) -> bool {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() <= tol * x.abs().max(y.abs()).max(1e-300),
                _ => false,
            },
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_close(p, q, tol))
            }
            (Value::Object(x), Value::Object(y)) => {
                x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_close(v, w, tol)))
            }
            _ => a == b,
        }
    }

    #[test]
    fn state_survives_a_json_text_round_trip() {
        let mut a = Plasticity::new();
        feed(&mut a, (0..150).map(|i| 2.0 + 0.02 * wobble(i)), 0.02);
        let text = serde_json::to_string(&a.state()).expect("state serialises");
        let value: Value = serde_json::from_str(&text).expect("state parses");
        let b = Plasticity::restore(&value);
        // serde_json parses a decimal to within one ulp unless its `float_roundtrip` feature is on
        assert!(json_close(&a.state(), &b.state(), 1e-14));
    }

    #[test]
    fn restore_from_nothing_or_nonsense_gives_a_fresh_controller() {
        let fresh = Plasticity::new().state();
        for d in [json!(null), json!({}), json!([]), json!("x"), json!(7), json!([1, 2, 3])] {
            assert_eq!(Plasticity::restore(&d).state(), fresh, "{d}");
        }
    }

    #[test]
    fn restore_from_a_partial_state_fills_in_defaults() {
        let p = Plasticity::restore(&json!({"scale": 0.4}));
        assert_eq!((p.scale(), p.step()), (0.4, 0));
        let p = Plasticity::restore(&json!({"scale": 0.4, "step": 250, "prev": 2.5}));
        assert_eq!((p.scale(), p.step()), (0.4, 250));
        assert_eq!(p.factor(), 0.4);
        // junk in the right places is ignored rather than fatal
        let p = Plasticity::restore(&json!({
            "scale": "wide", "prev": "x", "S": 5, "F": [1], "i": "n", "se": [0.01, "a", null, 0.03],
            "step": -4, "events": [1, {"at": 3}, {"at": 5, "kind": "settling", "val": 2.0, "scale": 0.9}]
        }));
        assert_eq!(p.scale(), 1.0);
        assert_eq!(p.step(), 0);
        assert!((p.noise() - 0.02).abs() < 1e-15);
        assert_eq!(p.events().len(), 1);
    }

    #[test]
    fn restore_clamps_an_impossible_scale() {
        assert_eq!(Plasticity::restore(&json!({"scale": 0.0})).scale(), Plasticity::FLOOR);
        assert_eq!(Plasticity::restore(&json!({"scale": 7.0})).scale(), Plasticity::CEIL);
        assert_eq!(Plasticity::restore(&json!({"scale": -1.0})).scale(), Plasticity::FLOOR);
    }

    #[test]
    fn fast_fit_is_left_empty_when_a_checkpoint_has_none() {
        let mut a = Plasticity::new();
        feed(&mut a, (0..60).map(|i| 3.0 - 0.01 * i as f64 + 0.02 * wobble(i)), 0.02);
        let mut state = a.state();
        if let Some(o) = state.as_object_mut() {
            o.remove("F");
        }
        let b = Plasticity::restore(&state);
        assert_eq!(b.state()["F"], serde_json::to_value(Sums::default()).unwrap_or_default());
        assert_eq!(b.state()["S"], a.state()["S"]);
    }

    #[test]
    fn an_older_checkpoint_with_a_plain_window_is_replayed() {
        let p = Plasticity::restore(&json!({"scale": 0.8, "hist": [2.9, 2.8, 2.7, 2.6, 2.5]}));
        let s = p.state();
        assert_eq!(s["i"], json!(5.0));
        assert!(s["S"]["w"].as_f64().is_some_and(|w| w > 4.0));
        assert_eq!(p.scale(), 0.8);
    }

    #[test]
    fn only_forty_events_are_kept_and_saved() {
        let mut p = Plasticity::new();
        // alternate improvement and decline so the scale keeps drifting and events keep being recorded
        feed(
            &mut p,
            (0..800).map(|i| 2.0 + 0.04 * (i as f64 / 20.0).sin() * (i as f64 / 3.0) + wobble(i) * 0.02),
            0.02,
        );
        assert!(p.events().len() <= Plasticity::EVENTS_KEPT);
        let saved = p.state()["events"].as_array().map_or(0, Vec::len);
        assert!(saved <= Plasticity::EVENTS_KEPT);
    }

    #[test]
    fn observation_converts_to_the_ui_event() {
        let mut p = Plasticity::new();
        let o = p.observe(2.0, 0.01);
        let ev = o.to_event();
        assert_eq!((ev.rate_scale, ev.evidence_t, ev.jumped), (o.scale, o.t, false));
        assert_eq!(ev.reason, o.reason);
    }

    #[test]
    fn py_round_matches_python() {
        assert_eq!(py_round(65.74999, 1), 65.7);
        assert_eq!(py_round(2.675, 2), 2.67); // the double is just below 2.675
        assert_eq!(py_round(0.125, 2), 0.12); // exact tie: half to even
        assert_eq!(py_round(-1.0005, 3), -1.0);
        assert!(py_round(f64::NAN, 2).is_nan());
    }

    // ----------------------------------------------------------------- GradSnr

    #[test]
    fn grad_snr_is_silent_until_it_has_eight_readings() {
        let mut m = GradSnr::default();
        let g = [1.0f32, -2.0, 0.5];
        for _ in 0..7 {
            assert_eq!(m.observe(&[&g]), None);
        }
        assert!(m.observe(&[&g]).is_some());
    }

    #[test]
    fn grad_snr_is_high_for_identical_gradients_and_low_for_noise() {
        // The Python divides both averages by the bias correction c = 1 - beta^n even though it starts them at the
        // first reading rather than at zero, so identical gradients read 1 / c: well above 1 early, tending to 1.
        let g = [0.5f32, -1.5, 2.0, 0.25];
        for (n, tol) in [(8, 1e-5), (50, 1e-5), (800, 1e-5)] {
            let mut same = GradSnr::default();
            let mut last = None;
            for _ in 0..n {
                last = same.observe(&[&g[..2], &g[2..]]);
            }
            let want = 1.0 / (1.0 - 0.98f64.powi(n));
            assert!((last.unwrap_or(0.0) / want - 1.0).abs() < tol, "n={n}: {last:?} vs {want}");
        }

        let mut noisy = GradSnr::default();
        let mut last = None;
        for step in 0..800 {
            let g: Vec<f32> = (0..64).map(|k| wobble(step * 64 + k) as f32).collect();
            last = noisy.observe(&[&g]);
        }
        let r = last.unwrap_or(1.0);
        assert!((0.0..0.2).contains(&r), "noise should read near zero, got {r}");
    }

    #[test]
    fn grad_snr_ignores_empty_input_and_restarts_on_a_new_shape() {
        let mut m = GradSnr::default();
        assert_eq!(m.observe(&[]), None);
        assert_eq!(m.observe(&[&[], &[]]), None);
        let a = [1.0f32; 4];
        for _ in 0..10 {
            m.observe(&[&a]);
        }
        let b = [1.0f32; 5];
        assert_eq!(m.observe(&[&b]), None, "started over, so it must warm up again");
        assert_eq!(m.n, 1);
    }

    #[test]
    fn grad_snr_of_all_zero_gradients_is_none() {
        let mut m = GradSnr::default();
        for _ in 0..20 {
            assert_eq!(m.observe(&[&[0.0f32; 8]]), None);
        }
    }
}
