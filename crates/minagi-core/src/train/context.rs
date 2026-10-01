//! The context window grows while the model is still gaining from it.
//!
//! The model attends with rotary positions, which carry no learned parameters, so a longer window costs nothing to
//! represent. What a model cannot do is jump to a length it has never trained at, so the window starts small
//! (`context_start`) and grows **one `context_step` at a time**, never past `context_end` (or the length the rotary
//! tables were built for, whichever is smaller).
//!
//! Whether to grow is decided by measurement, not by schedule:
//!
//! * Every step the loss is filed under where in the visit it was measured: the **position** is the index of the
//!   chunk inside the passage being read, so position 0 sees almost no context and later positions see more.
//! * Every `context_every_chars` the evidence is read. The **gain** is the mean loss at the early positions (below
//!   50% of the window) minus the mean loss at the late ones (75% and beyond), and it needs at least 8 readings in
//!   each. A positive gain means the model really predicts better when it has more text behind it, so it would use
//!   a longer window; near zero means the far end of the window is carried for nothing.
//! * While the gain exceeds `context_gain_min` the window grows by one step every `context_grow_every_chars`.
//!
//! Gathering the evidence and acting on it are separate on purpose: tied together, the window grew one character
//! per evidence batch (29 characters in six million read in the original's measurements).
//!
//! This ports the logic that lives inside the reading command of the original's `train.py` (`context_gain`,
//! `visit_chunks`, `_in_steps` and the ramp block), plus `ramp_context` from `minagi/stream.py`, the older
//! schedule-based ramp the batch trainer used. All of it is pure arithmetic; the training loop owns the clock.
//!
//! Per-position losses are kept as a running count and sum rather than as every loss ever recorded (the original
//! keeps lists, which grow without bound once the window stops growing); the means are the same.

use minagi_types::TrainConfig;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// The smallest window the ramp will ever use or clamp to.
pub const MIN_CONTEXT: u32 = 64;

/// How many readings each end of the window needs before a gain is reported.
pub const MIN_SAMPLES: u64 = 8;

/// Where the early end of the window stops (a share of the window's chunk positions).
const EARLY_BELOW: f64 = 0.5;
/// Where the late end of the window starts.
const LATE_FROM: f64 = 0.75;

/// Convert a cadence given in characters into optimiser steps: `round(chars / chunk)` (halves round to even, as
/// Python's `round` does), at least 1.
///
/// Every cadence is configured in characters, because a step is `chunk` characters and a count in steps would
/// silently change meaning whenever the chunk changed.
pub fn chars_to_steps(chars: u64, chunk: u32) -> u64 {
    let steps = (chars as f64 / f64::from(chunk.max(1))).round_ties_even();
    (steps as u64).max(1)
}

/// How many chunks one visit to a file lasts: the passage length in chunks, but never less than one full window,
/// or the window would never fill and the model would be asked to read with less context than it has.
pub fn visit_chunks(context: u32, chunk: u32, passage: u32) -> u32 {
    let chunk = chunk.max(1);
    let one_window = context.div_ceil(chunk).max(1);
    let wanted = passage.div_ceil(chunk).max(1);
    one_window.max(wanted)
}

/// The older, schedule-based ramp (`ramp_context` in `minagi/stream.py`): the window grows geometrically from
/// `start` to `end` over the first `warm` share of the run, rounded to a multiple of `granularity`.
///
/// "Gradually" matters: rounding to powers of two turns the ramp into a staircase, and a window that doubles in one
/// step is a shock the run may not recover from. Returns `end` when `end <= start`.
pub fn ramp_schedule(step: u64, total: u64, start: u32, end: u32, warm: f64, granularity: u32) -> u32 {
    if end <= start {
        return end;
    }
    if start == 0 {
        // the original divides by zero here; a window cannot start at nothing
        return end;
    }
    let f = (step as f64 / (total as f64 * warm).max(1.0)).min(1.0);
    let ctx = f64::from(start) * (f64::from(end) / f64::from(start)).powf(f);
    let g = f64::from(granularity.max(1));
    let rounded = (ctx / g).round_ties_even() * g;
    // int(max(start, min(end, rounded)))
    let clipped = rounded.min(f64::from(end)).max(f64::from(start));
    clipped as u32
}

/// Everything the ramp needs to know, in the units the configuration uses.
#[derive(Debug, Clone, PartialEq)]
pub struct RampSettings {
    /// Characters per optimiser step.
    pub chunk: u32,
    /// Characters read from one file before moving to the next.
    pub passage: u32,
    /// How far the rotary tables reach (`model.block`): a hard ceiling.
    pub block: u32,
    /// Where a fresh model's window begins.
    pub context_start: u32,
    /// The furthest the window may ever reach (also binds on resume).
    pub context_end: u32,
    /// Characters the window grows by when the model asks.
    pub context_step: u32,
    /// Characters between growths while the evidence says the model is still gaining.
    pub grow_every_chars: u32,
    /// Characters between asking whether the model is still gaining.
    pub check_every_chars: u32,
    /// The gain (in nats) the model must show deep into the window before it grows.
    pub gain_min: f64,
}

impl RampSettings {
    /// Settings from a run's configuration and the model's rotary-table length (`block`).
    pub fn from_config(cfg: &TrainConfig, block: u32) -> Self {
        Self {
            chunk: cfg.chunk,
            passage: cfg.passage,
            block,
            context_start: cfg.context_start,
            context_end: cfg.context_end,
            context_step: cfg.context_step,
            grow_every_chars: cfg.context_grow_every_chars,
            check_every_chars: cfg.context_every_chars,
            gain_min: cfg.context_gain_min,
        }
    }
}

/// A running count and sum of the losses seen at one position.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    n: u64,
    sum: f64,
}

/// The growing context window and the evidence it grows on. See the module docs.
///
/// Loop usage, per optimiser step: read the window with [`Self::context`], train, then
/// `ramp.record(chunk_in_visit, loss)`, increment the global step and call `ramp.after_step(step)`; when that
/// returns a new size, the next visit uses it (and [`Self::visit_chunks`] has changed).
#[derive(Debug, Clone)]
pub struct ContextRamp {
    chunk: u32,
    passage: u32,
    ceiling: u32,
    context: u32,
    step_chars: u32,
    check_every: u64,
    grow_every: u64,
    gain_min: f64,
    /// Loss by chunk position within a visit.
    by_pos: BTreeMap<u32, Tally>,
    /// Whether the last gain that could be measured exceeded the minimum. Not persisted: a resumed run waits for
    /// fresh evidence before growing.
    wants_more: bool,
    /// The window the checkpoint carried when it was above today's ceiling.
    held_from: Option<u32>,
}

impl ContextRamp {
    /// A ramp for a fresh run (`saved` = `None`) or one resumed from a checkpoint whose manifest held
    /// `context_now = saved`.
    ///
    /// The ceiling is `max(64, min(block, context_end))` and binds on resume too: a checkpoint that had grown past it
    /// is held at the ceiling and does not walk back up. The window is `saved` (or `context_start` when there is
    /// none, or it is 0) clamped to `[64, ceiling]`.
    pub fn new(settings: &RampSettings, saved: Option<u32>) -> Self {
        let ceiling = settings.block.min(settings.context_end).max(MIN_CONTEXT);
        let wanted = saved.filter(|&s| s != 0).unwrap_or(settings.context_start);
        let context = wanted.min(ceiling).max(MIN_CONTEXT);
        let saved = saved.unwrap_or(0);
        Self {
            chunk: settings.chunk.max(1),
            passage: settings.passage,
            ceiling,
            context,
            step_chars: settings.context_step,
            check_every: chars_to_steps(u64::from(settings.check_every_chars), settings.chunk),
            grow_every: chars_to_steps(u64::from(settings.grow_every_chars), settings.chunk),
            gain_min: settings.gain_min,
            by_pos: BTreeMap::new(),
            wants_more: false,
            held_from: (context < saved).then_some(saved),
        }
    }

    /// Resume from a checkpoint manifest (or any JSON object holding `context_now`); anything missing or not a
    /// number means a fresh start.
    pub fn restore(settings: &RampSettings, manifest: &Value) -> Self {
        let saved = manifest
            .get("context_now")
            .and_then(Value::as_f64)
            .filter(|s| s.is_finite() && *s > 0.0)
            .map(|s| s.min(f64::from(u32::MAX)) as u32);
        Self::new(settings, saved)
    }

    /// What to store in the manifest: `{"context_now": <window>}`.
    pub fn state(&self) -> Value {
        json!({ "context_now": self.context })
    }

    /// The window the model reads at now.
    pub fn context(&self) -> u32 {
        self.context
    }

    /// The furthest the window may reach.
    pub fn ceiling(&self) -> u32 {
        self.ceiling
    }

    /// `true` once the window has reached its ceiling (it never changes again).
    pub fn at_ceiling(&self) -> bool {
        self.context >= self.ceiling
    }

    /// When a resumed checkpoint had a larger window than today's ceiling allows, the size it had (so the caller can
    /// say "window held at N by the ceiling").
    pub fn held_from(&self) -> Option<u32> {
        self.held_from
    }

    /// Steps between evidence checks.
    pub fn check_every_steps(&self) -> u64 {
        self.check_every
    }

    /// Steps between single growths while the model keeps gaining.
    pub fn grow_every_steps(&self) -> u64 {
        self.grow_every
    }

    /// Whether the evidence says the model would use more window.
    pub fn wants_more(&self) -> bool {
        self.wants_more
    }

    /// Chunks in one visit at the current window (see [`visit_chunks`]).
    pub fn visit_chunks(&self) -> u32 {
        visit_chunks(self.context, self.chunk, self.passage)
    }

    /// File one step's loss under its position: `chunk_in_visit` is the index of the chunk within the visit (0 for
    /// the chunk that opens a passage). Losses that are not finite are ignored so one bad step cannot poison the
    /// evidence.
    pub fn record(&mut self, chunk_in_visit: u32, loss: f64) {
        if loss.is_finite() {
            let t = self.by_pos.entry(chunk_in_visit).or_default();
            t.n += 1;
            t.sum += loss;
        }
    }

    /// How much better the model predicts deep into the window than early in it: mean loss at positions below 50%
    /// of the window minus mean loss at positions from 75%. Positive means it is using the distance it has.
    ///
    /// `None` until there are at least 8 readings at each end. The window is measured in chunks:
    /// `max(2, context / chunk)` positions.
    pub fn gain(&self) -> Option<f64> {
        let buckets = f64::from((self.context / self.chunk).max(2));
        let (mut early, mut late) = (Tally::default(), Tally::default());
        for (&p, t) in &self.by_pos {
            let p = f64::from(p);
            let side = if p < buckets * EARLY_BELOW {
                &mut early
            } else if p >= buckets * LATE_FROM {
                &mut late
            } else {
                continue;
            };
            side.n += t.n;
            side.sum += t.sum;
        }
        if early.n < MIN_SAMPLES || late.n < MIN_SAMPLES {
            return None;
        }
        Some(early.sum / early.n as f64 - late.sum / late.n as f64)
    }

    /// Call once per optimiser step, after the step is counted (`step` is the global step *after* incrementing).
    /// Returns the new window when it grew.
    ///
    /// Nothing happens once the window is at its ceiling (evidence keeps accumulating, as in the original). On every
    /// `check_every_steps`: if a gain can be measured, remember whether it exceeds the minimum and start the next
    /// batch of evidence (a gain that cannot be measured yet keeps accumulating). Then, on every
    /// `grow_every_steps`, if the last measured gain was big enough, grow by one `context_step`, up to the ceiling.
    pub fn after_step(&mut self, step: u64) -> Option<u32> {
        if self.context >= self.ceiling {
            return None;
        }
        if step.is_multiple_of(self.check_every)
            && let Some(gain) = self.gain()
        {
            self.wants_more = gain > self.gain_min;
            self.by_pos.clear();
        }
        if self.wants_more && step.is_multiple_of(self.grow_every) {
            let next = self.context.saturating_add(self.step_chars).min(self.ceiling);
            if next != self.context {
                self.context = next;
                return Some(next);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> RampSettings {
        RampSettings {
            chunk: 64,
            passage: 1024,
            block: 1024,
            context_start: 256,
            context_end: 640,
            context_step: 16,
            grow_every_chars: 640,
            check_every_chars: 1280,
            gain_min: 0.015,
        }
    }

    #[test]
    fn cadences_are_converted_from_characters_to_steps() {
        let r = ContextRamp::new(&settings(), None);
        assert_eq!((r.check_every_steps(), r.grow_every_steps()), (20, 10));
        // the shipped defaults: a growth every 195 steps and a check every 128 at chunk 512
        assert_eq!(chars_to_steps(100_000, 512), 195);
        assert_eq!(chars_to_steps(65_536, 512), 128);
        // at least one step, whatever the numbers
        assert_eq!(chars_to_steps(0, 512), 1);
        assert_eq!(chars_to_steps(10, 512), 1);
        // a zero chunk is treated as 1
        assert_eq!(chars_to_steps(7, 0), 7);
        // halves round to even, like Python's round()
        assert_eq!(chars_to_steps(5, 2), 2);
        assert_eq!(chars_to_steps(7, 2), 4);
        assert_eq!(chars_to_steps(3, 2), 2);
    }

    #[test]
    fn a_visit_is_the_passage_but_never_shorter_than_one_window() {
        assert_eq!(visit_chunks(1024, 512, 16384), 32);
        assert_eq!(visit_chunks(32768, 512, 16384), 64);
        assert_eq!(visit_chunks(513, 512, 1), 2);
        assert_eq!(visit_chunks(1, 512, 1), 1);
        assert_eq!(visit_chunks(100, 0, 100), 100, "a zero chunk is treated as 1");
    }

    #[test]
    fn the_window_starts_at_the_start_or_where_the_checkpoint_left_it() {
        let s = settings();
        assert_eq!(ContextRamp::new(&s, None).context(), 256);
        assert_eq!(ContextRamp::new(&s, Some(0)).context(), 256);
        assert_eq!(ContextRamp::new(&s, Some(448)).context(), 448);
        assert_eq!(ContextRamp::new(&s, Some(10)).context(), MIN_CONTEXT);
    }

    #[test]
    fn the_ceiling_binds_on_resume_and_is_the_smaller_of_block_and_context_end() {
        let s = RampSettings { context_end: 512, block: 4096, ..settings() };
        let r = ContextRamp::new(&s, Some(4000));
        assert_eq!((r.context(), r.ceiling(), r.held_from()), (512, 512, Some(4000)));
        assert!(r.at_ceiling());
        let s = RampSettings { context_end: 2048, block: 512, ..settings() };
        assert_eq!(ContextRamp::new(&s, None).ceiling(), 512);
        // a nonsensical ceiling is raised to the floor
        let s = RampSettings { context_end: 10, ..settings() };
        assert_eq!(ContextRamp::new(&s, None).ceiling(), MIN_CONTEXT);
        assert_eq!(ContextRamp::new(&settings(), Some(300)).held_from(), None);
    }

    #[test]
    fn restore_reads_context_now_from_a_manifest() {
        let s = settings();
        assert_eq!(ContextRamp::restore(&s, &json!({"context_now": 400, "step": 5})).context(), 400);
        for m in [
            json!({}),
            json!(null),
            json!({"context_now": null}),
            json!({"context_now": "x"}),
            json!({"context_now": -5}),
        ] {
            assert_eq!(ContextRamp::restore(&s, &m).context(), 256, "{m}");
        }
        assert_eq!(ContextRamp::restore(&s, &json!({"context_now": 400.0})).context(), 400);
        let r = ContextRamp::new(&s, Some(400));
        assert_eq!(ContextRamp::restore(&s, &r.state()).context(), 400);
        assert_eq!(r.state(), json!({"context_now": 400}));
    }

    #[test]
    fn gain_needs_eight_readings_at_each_end() {
        let mut r = ContextRamp::new(&RampSettings { context_start: 640, ..settings() }, None); // 10 buckets
        assert_eq!(r.gain(), None);
        for _ in 0..7 {
            r.record(0, 3.0);
            r.record(9, 2.0);
        }
        assert_eq!(r.gain(), None, "7 each is not enough");
        r.record(0, 3.0);
        assert_eq!(r.gain(), None, "8 early but 7 late");
        r.record(9, 2.0);
        assert_eq!(r.gain(), Some(1.0));
    }

    #[test]
    fn gain_uses_the_bucket_boundaries_of_the_original() {
        // 10 buckets: early is p < 5, late is p >= 7.5, so 4 is early, 5 and 7 are neither, 8 is late
        let mut r = ContextRamp::new(&RampSettings { context_start: 640, ..settings() }, None);
        for _ in 0..8 {
            r.record(4, 3.0);
            r.record(5, 9.0);
            r.record(7, 9.0);
            r.record(8, 1.0);
        }
        assert_eq!(r.gain(), Some(2.0));
    }

    #[test]
    fn a_tiny_window_still_has_two_buckets() {
        // window of one chunk: max(2, 1) = 2 buckets, early is p < 1 (position 0), late is p >= 1.5 (2 and up)
        let mut r = ContextRamp::new(&RampSettings { context_start: 64, ..settings() }, None);
        for _ in 0..8 {
            r.record(0, 2.5);
            r.record(1, 100.0);
            r.record(2, 2.0);
        }
        assert_eq!(r.gain(), Some(0.5));
    }

    #[test]
    fn non_finite_losses_are_ignored() {
        let mut r = ContextRamp::new(&RampSettings { context_start: 640, ..settings() }, None);
        for _ in 0..8 {
            r.record(0, 3.0);
            r.record(9, 2.0);
        }
        r.record(0, f64::NAN);
        r.record(9, f64::INFINITY);
        assert_eq!(r.gain(), Some(1.0));
    }

    /// Feed a constant-gain reader: early positions lose `gain` more than late ones.
    fn read(r: &mut ContextRamp, steps: u64, gain: f64) -> Vec<(u64, u32)> {
        let mut grew = Vec::new();
        let mut step = 0;
        while step < steps {
            let turn = r.visit_chunks();
            for j in 0..turn {
                let buckets = (r.context() / 64).max(2);
                let loss = if f64::from(j) < f64::from(buckets) * 0.5 { 2.0 + gain } else { 2.0 };
                r.record(j, loss);
                step += 1;
                if let Some(c) = r.after_step(step) {
                    grew.push((step, c));
                }
                if step >= steps {
                    break;
                }
            }
        }
        grew
    }

    #[test]
    fn the_window_grows_one_step_at_a_time_while_there_is_gain() {
        let mut r = ContextRamp::new(&settings(), None);
        let grew = read(&mut r, 400, 0.2);
        assert!(!grew.is_empty());
        assert!(grew.windows(2).all(|w| w[1].1 == w[0].1 + 16 && w[1].0 - w[0].0 == 10), "{grew:?}");
        assert_eq!(grew[0].1, 272);
        assert!(r.wants_more());
    }

    #[test]
    fn no_gain_means_no_growth() {
        let mut r = ContextRamp::new(&settings(), None);
        assert!(read(&mut r, 2000, 0.0).is_empty());
        assert!(!r.wants_more());
        assert_eq!(r.context(), 256);
        // a gain just under the minimum is not enough either
        let mut r = ContextRamp::new(&settings(), None);
        assert!(read(&mut r, 2000, 0.0149).is_empty());
    }

    #[test]
    fn growth_stops_at_the_ceiling_and_the_last_step_is_clipped() {
        let s = RampSettings { context_step: 100, ..settings() };
        let mut r = ContextRamp::new(&s, Some(560));
        let grew = read(&mut r, 2000, 0.5);
        assert_eq!(grew.len(), 1);
        assert_eq!(grew[0].1, 640);
        assert!(r.at_ceiling());
        // once there, it never moves again and the check no longer runs
        assert_eq!(r.after_step(10), None);
        assert_eq!(r.after_step(20), None);
    }

    #[test]
    fn evidence_that_cannot_be_measured_yet_keeps_accumulating() {
        // one reading each end per check is never enough on its own, but they pile up across checks
        let s = RampSettings { context_start: 640, context_end: 1024, check_every_chars: 64, ..settings() };
        let mut r = ContextRamp::new(&s, None);
        assert_eq!(r.check_every_steps(), 1);
        for step in 1..=7 {
            r.record(0, 3.0);
            r.record(9, 2.0);
            assert_eq!(r.after_step(step), None);
            assert!(!r.wants_more());
        }
        r.record(0, 3.0);
        r.record(9, 2.0);
        r.after_step(8);
        assert!(r.wants_more(), "eight of each accumulated, so the check could finally read a gain of 1.0");
        assert_eq!(r.gain(), None, "and starting the next batch of evidence cleared it");
    }

    #[test]
    fn a_zero_step_never_changes_the_window() {
        let mut r = ContextRamp::new(&RampSettings { context_step: 0, ..settings() }, None);
        assert!(read(&mut r, 400, 0.5).is_empty());
        assert_eq!(r.context(), 256);
    }

    #[test]
    fn the_schedule_ramp_is_geometric_rounded_and_clamped() {
        // before any progress it sits at the start, after `warm` of the run it reaches the end
        assert_eq!(ramp_schedule(0, 1000, 256, 4096, 0.35, 256), 256);
        assert_eq!(ramp_schedule(350, 1000, 256, 4096, 0.35, 256), 4096);
        assert_eq!(ramp_schedule(10_000, 1000, 256, 4096, 0.35, 256), 4096);
        // halfway through the ramp it is the geometric middle (1024), not the arithmetic one
        assert_eq!(ramp_schedule(175, 1000, 256, 4096, 0.35, 256), 1024);
        // end <= start returns end, a zero start does not divide by zero
        assert_eq!(ramp_schedule(5, 10, 2048, 1024, 0.35, 256), 1024);
        assert_eq!(ramp_schedule(5, 10, 512, 512, 0.35, 256), 512);
        assert_eq!(ramp_schedule(5, 10, 0, 512, 0.35, 256), 512);
        // never below the start even when rounding would take it there
        assert_eq!(ramp_schedule(1, 1_000_000, 300, 4096, 0.35, 256), 300);
        // monotone
        let mut last = 0;
        for s in 0..400 {
            let c = ramp_schedule(s, 1000, 256, 4096, 0.35, 64);
            assert!(c >= last);
            last = c;
        }
    }

    #[test]
    fn settings_come_from_the_training_config() {
        let cfg = TrainConfig::default();
        let s = RampSettings::from_config(&cfg, 2048);
        assert_eq!((s.chunk, s.block, s.context_step), (cfg.chunk, 2048, cfg.context_step));
        let r = ContextRamp::new(&s, None);
        assert_eq!(r.context(), cfg.context_start.min(cfg.context_end).min(2048));
    }
}
