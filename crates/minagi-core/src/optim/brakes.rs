//! The five growth brakes: when may the expert pool add another expert?
//!
//! The pool finds its own size by trying. Every `growth.every_chars` characters the training loop asks "may I
//! grow?", and the answer is yes only if **all five** brakes agree:
//!
//! 1. **Room**: the pool is under its size limit, the expert files would stay under the disk limit (an expert is
//!    ~25 MB on disk with its Adam moments, so this limit is what decides how large the model can ever get) and
//!    accelerator memory is not nearly spent.
//! 2. **Used**: the capacity already added is being asked for. Measured on *staleness* (how long it is since
//!    anything wanted an expert), never on routing share: the load-balancing loss pushes routing towards uniform on
//!    purpose, so every expert, live or dead, receives roughly its fair share and a traffic-based count would
//!    report zero idle experts forever.
//! 3. **Earning**: the experts added recently, once their trial is over, are still being asked for. (Called `KEPT`
//!    in the original.)
//! 4. **Fits**: no more than `max_in_flight` experts are inside their trial at once. Neither usage brake can see a
//!    newborn (it reads as not idle and has not been judged), so without this cap there would be a whole trial
//!    window in which nothing could refuse.
//! 5. **Honest**: training and held-out loss have not drifted apart, which is what memorising looks like. In the
//!    original this is asked by the training loop rather than the pool, because only it sees both losses.
//!
//! A new expert is born with a tiny gate so nothing already working is disturbed, is on trial for a while during
//! which pruning cannot touch it, and is then deleted unless something still asks for it. Wrongly guessing "grow"
//! therefore costs only a few experts that get deleted again.
//!
//! This is a *pure function*: [`decide`] reads a plain [`GrowthInputs`] snapshot (no pool object) and returns a
//! [`GrowthDecision`], so the same decision can be logged, shown in the UI and tested against the Python original
//! (`AutoGrow.step`, `_in_flight`, `_earning` in `minagi/pool.py`, plus the `gap_ok` rule in `train.py`).
//! Actually adding the experts is the caller's job.

use minagi_types::{Brake, BrakeReport, GrowthConfig};

/// Fallback length of a trial, in steps, when the pool does not know its own (the original's `1600`).
const DEFAULT_TRIAL_STEPS: f64 = 1600.0;

/// A snapshot of everything the brakes read.
///
/// `dying` and `born` have one entry per expert, in the same order. The pool size is `dying.len()`.
#[derive(Debug, Clone, Copy)]
pub struct GrowthInputs<'a> {
    /// The training step now (the clock `born` and `trial` are measured on).
    pub model_step: f64,
    /// The most experts the pool may ever hold.
    pub max_experts: u32,
    /// For each expert, how close it is to being pruned as a share of the survival window: 0 means something asked
    /// for it just now, 1 means it has gone as long unaddressed as pruning allows. A newborn on trial reads 0.
    pub dying: &'a [f64],
    /// The share of `dying` at which an expert counts as idle (`prune.dying_at`).
    pub dying_at: f64,
    /// For each expert, the step it was added. Experts that existed at the start read 0 and are never judged.
    pub born: &'a [f64],
    /// Length of an expert's trial, in steps (`prune.survival_chars / chunk`); 0 if unknown.
    pub trial: f64,
    /// Fraction of accelerator memory the process needs at its peak (0..=1; 0 when there is nothing to measure).
    pub mem_frac: f64,
    /// What the pool's expert files take on disk now, in bytes (0 if unknown).
    pub disk_bytes_now: u64,
    /// What they would take after adding `growth.k` experts, in bytes (0 if unknown).
    pub disk_bytes_after: u64,
    /// Held-out loss minus recent training loss, when both are known.
    pub last_gap: Option<f64>,
}

/// Which brake refused. In the order the original reports them when several refuse at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrakeKind {
    /// Training and held-out loss have separated: memorising.
    Honest,
    /// Too many experts are still on trial.
    Fits,
    /// The pool is full, the disk limit would be passed, or memory is nearly spent.
    Room,
    /// Too many existing experts are idle.
    Used,
    /// Recently added experts are not being asked for.
    Earning,
}

/// The numbers behind a decision, for logs and the UI.
#[derive(Debug, Clone, PartialEq)]
pub struct GrowthMeasures {
    /// Experts in the pool now.
    pub experts: usize,
    /// Experts whose `dying` is at or past `dying_at`.
    pub idle: usize,
    /// `idle / max(experts, 1)`.
    pub idle_frac: f64,
    /// Experts added whose trial has not ended.
    pub in_flight: usize,
    /// Experts finished with their trial and still young enough to judge.
    pub judged: usize,
    /// Of those, how many are still being asked for.
    pub earning: usize,
    /// `earning / judged`, or 1 when nothing has been judged.
    pub keep_ratio: f64,
    /// The memory fraction that was read.
    pub mem_frac: f64,
    /// Disk used now, in GB (1 GB = 1e9 bytes).
    pub disk_gb_now: f64,
    /// Disk after adding `k` experts, in GB.
    pub disk_gb_after: f64,
}

/// The verdict of one growth decision.
#[derive(Debug, Clone, PartialEq)]
pub struct GrowthDecision {
    /// How many experts to add now (0 when any brake refused or growth is switched off).
    pub grow: u32,
    /// Each brake, with a plain-English reason, ready for the UI.
    pub brakes: BrakeReport,
    /// The first brake that refused, in the original's order (`None` when it grew or growth is off).
    pub blocked_by: Option<BrakeKind>,
    /// One plain-English line saying what was decided and why.
    pub reason: String,
    /// The numbers the brakes read.
    pub measures: GrowthMeasures,
}

/// The outcome of asking whether recently added experts are being used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Earning {
    /// Experts past their trial but younger than `trial * recent_mult`.
    pub judged: usize,
    /// How many of them are not yet dying.
    pub earning: usize,
    /// `earning / judged`; 1 when there is nothing to judge ("a pool of originals has not yet failed at anything").
    pub ratio: f64,
}

/// How many experts are on trial: added (`born > 0`) and younger than `trial` steps.
///
/// Zero when `trial` is not positive (nothing is ever on trial then).
pub fn in_flight(born: &[f64], trial: f64, model_step: f64) -> usize {
    if trial <= 0.0 {
        return 0;
    }
    born.iter().filter(|&&b| b > 0.0 && model_step - b < trial).count()
}

/// Are the experts added recently being used?
///
/// The question is asked of experts that have *finished* their trial and are still young, old enough to have had
/// their fair turn and recent enough that their fate says something about whether the pool still needs more. An
/// expert is earning when it is being asked for (its `dying` is below `dying_at`), not when its gate is large.
///
/// If `born` is longer than `dying` only the experts that have both are scored, though all judged experts count in
/// the denominator; the Python would raise on that mismatch.
pub fn earning(born: &[f64], dying: &[f64], dying_at: f64, trial: f64, model_step: f64, recent_mult: f64) -> Earning {
    let trial = if trial == 0.0 { DEFAULT_TRIAL_STEPS } else { trial };
    let judged_at = |b: f64| {
        let age = model_step - b;
        b > 0.0 && age >= trial && age < trial * recent_mult
    };
    let judged = born.iter().filter(|&&b| judged_at(b)).count();
    if judged == 0 {
        return Earning { judged: 0, earning: 0, ratio: 1.0 };
    }
    let earning = born.iter().zip(dying).filter(|&(&b, &d)| judged_at(b) && d < dying_at).count();
    Earning { judged, earning, ratio: earning as f64 / judged as f64 }
}

fn experts(n: u32) -> String {
    if n == 1 { "1 expert".to_string() } else { format!("{n} experts") }
}

/// Decide whether the pool may grow, and say why or why not.
///
/// `cfg` supplies the thresholds (`k` experts per round, `dying_frac_max`, `keep_ratio_min`, `max_in_flight`,
/// `max_disk_gb`, `mem_frac`, `recent_mult`, `max_gap`); a `max_in_flight`, `max_disk_gb` or `max_gap` of zero
/// switches that brake off. `cfg.k == 0` switches growth off altogether. Every comparison is the original's
/// (`<=` where it used `<=`, so a value exactly at its ceiling still passes).
///
/// All five brakes are always evaluated so the UI can show every one. The original skips the pool-side brakes when
/// the honest brake already refuses; the resulting decision is the same.
pub fn decide(cfg: &GrowthConfig, inp: &GrowthInputs<'_>) -> GrowthDecision {
    let n = inp.dying.len();
    let k = cfg.k;
    let idle = inp.dying.iter().filter(|&&d| d >= inp.dying_at).count();
    let idle_frac = idle as f64 / n.max(1) as f64;
    let kept = earning(inp.born, inp.dying, inp.dying_at, inp.trial, inp.model_step, cfg.recent_mult);
    let in_flight = in_flight(inp.born, inp.trial, inp.model_step);
    let disk_now = inp.disk_bytes_now as f64 / 1e9;
    let disk_after = inp.disk_bytes_after as f64 / 1e9;

    // FITS: room inside the in-flight cap
    let fits_ok = cfg.max_in_flight == 0 || in_flight as u64 + u64::from(k) <= u64::from(cfg.max_in_flight);
    // ROOM: pool size, accelerator memory, disk
    let count_ok = n as u64 + u64::from(k) <= u64::from(inp.max_experts);
    let mem_ok = inp.mem_frac <= cfg.mem_frac;
    let disk_ok = cfg.max_disk_gb == 0.0 || disk_after <= cfg.max_disk_gb;
    let room_ok = count_ok && mem_ok && disk_ok;
    // USED and EARNING
    let used_ok = idle_frac <= cfg.dying_frac_max;
    let earning_ok = kept.ratio >= cfg.keep_ratio_min;
    // HONEST: no held-out reading, or the check being off, means nothing says it is memorising
    let honest_ok = inp.last_gap.is_none_or(|gap| cfg.max_gap <= 0.0 || gap < cfg.max_gap);

    let pct = |x: f64| (100.0 * x).round();

    let room = Brake {
        ok: room_ok,
        why: if room_ok {
            let disk = if cfg.max_disk_gb > 0.0 {
                format!(", {disk_after:.1} of {:.1} GB of disk", cfg.max_disk_gb)
            } else {
                String::new()
            };
            format!(
                "There is room for {}: the pool would hold {} of {} possible{disk}, and memory is {}% used.",
                if k == 1 { "another expert".to_string() } else { experts(k) },
                n as u64 + u64::from(k),
                inp.max_experts,
                pct(inp.mem_frac)
            )
        } else {
            let mut causes = Vec::new();
            if !count_ok {
                causes.push(format!("the pool is full ({n} of {} experts)", inp.max_experts));
            }
            if !disk_ok {
                causes.push(format!(
                    "the expert files would reach {disk_after:.1} GB, over the {:.1} GB limit ({disk_now:.1} GB used now)",
                    cfg.max_disk_gb
                ));
            }
            if !mem_ok {
                causes.push(format!(
                    "memory is nearly spent ({}% in use, limit {}%)",
                    pct(inp.mem_frac),
                    pct(cfg.mem_frac)
                ));
            }
            format!("There is no room to grow: {}.", causes.join("; "))
        },
    };

    let used = Brake {
        ok: used_ok,
        why: if used_ok {
            if n == 0 {
                "There are no experts yet, so none can be idle.".to_string()
            } else {
                format!(
                    "The experts that exist are being used: {idle} of {n} are going unused (up to {}% is fine).",
                    pct(cfg.dying_frac_max)
                )
            }
        } else {
            format!(
                "{idle} of {n} experts are going unused ({}%, more than the {}% allowed), so the capacity already \
                 added is not being asked for.",
                pct(idle_frac),
                pct(cfg.dying_frac_max)
            )
        },
    };

    let earning_brake = Brake {
        ok: earning_ok,
        why: if kept.judged == 0 {
            "No recently added experts have finished their trial yet, so nothing counts against growing.".to_string()
        } else if earning_ok {
            format!(
                "{} of {} newer experts are earning their place ({}%, at least {}% needed).",
                kept.earning,
                kept.judged,
                pct(kept.ratio),
                pct(cfg.keep_ratio_min)
            )
        } else {
            format!(
                "{} of {} newer experts are not yet useful ({}% are earning their place, {}% needed), so the pool \
                 has probably found its size.",
                kept.judged - kept.earning,
                kept.judged,
                pct(kept.ratio),
                pct(cfg.keep_ratio_min)
            )
        },
    };

    let fits = Brake {
        ok: fits_ok,
        why: if cfg.max_in_flight == 0 {
            "There is no limit on how many new experts may be on trial at once.".to_string()
        } else if fits_ok {
            format!("{in_flight} experts are on trial, and up to {} are allowed at once.", cfg.max_in_flight)
        } else {
            format!(
                "{in_flight} experts are still on trial, and only {} are allowed at once, so the latest additions \
                 have not been judged yet.",
                cfg.max_in_flight
            )
        },
    };

    let honest = Brake {
        ok: honest_ok,
        why: match inp.last_gap {
            None => "There is no held-out measurement yet, so nothing says the model is memorising.".to_string(),
            Some(_) if cfg.max_gap <= 0.0 => "The memorising check is switched off.".to_string(),
            Some(gap) if honest_ok => {
                format!("Held-out loss is {gap:.2} from training loss, under the {:.2} limit.", cfg.max_gap)
            }
            Some(gap) => format!(
                "Held-out loss is {gap:.2} above training loss, over the {:.2} limit: that is memorising, and more \
                 capacity would make it worse.",
                cfg.max_gap
            ),
        },
    };

    // The first refusal, in the order the original reports them.
    let blocked_by = [
        (BrakeKind::Honest, &honest),
        (BrakeKind::Fits, &fits),
        (BrakeKind::Room, &room),
        (BrakeKind::Used, &used),
        (BrakeKind::Earning, &earning_brake),
    ]
    .into_iter()
    .find(|(_, b)| !b.ok)
    .map(|(kind, _)| kind);

    let grow = if blocked_by.is_none() { k } else { 0 };
    let reason = match (k, blocked_by) {
        (0, _) => "Growth is switched off (0 experts per round).".to_string(),
        (_, None) => format!(
            "Adding {}: there is room, the existing experts are being used and recent additions are earning their \
             place.",
            experts(k)
        ),
        (_, Some(kind)) => match kind {
            BrakeKind::Honest => honest.why.clone(),
            BrakeKind::Fits => fits.why.clone(),
            BrakeKind::Room => room.why.clone(),
            BrakeKind::Used => used.why.clone(),
            BrakeKind::Earning => earning_brake.why.clone(),
        },
    };

    GrowthDecision {
        grow,
        blocked_by,
        brakes: BrakeReport { room, used, earning: earning_brake, fits, honest },
        reason,
        measures: GrowthMeasures {
            experts: n,
            idle,
            idle_frac,
            in_flight,
            judged: kept.judged,
            earning: kept.earning,
            keep_ratio: kept.ratio,
            mem_frac: inp.mem_frac,
            disk_gb_now: disk_now,
            disk_gb_after: disk_after,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool that may grow: 3 originals, 4 settled additions that are being used (ages 1000 to 3000, judged), 2 on
    /// trial.
    struct Pool {
        born: Vec<f64>,
        dying: Vec<f64>,
    }

    fn healthy() -> Pool {
        Pool {
            born: vec![0.0, 0.0, 0.0, 17_000.0, 18_000.0, 18_500.0, 19_000.0, 19_800.0, 19_900.0],
            dying: vec![0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.0, 0.0],
        }
    }

    fn inputs(p: &Pool) -> GrowthInputs<'_> {
        GrowthInputs {
            model_step: 20_000.0,
            max_experts: 64,
            dying: &p.dying,
            dying_at: 0.65,
            born: &p.born,
            trial: 1_000.0,
            mem_frac: 0.5,
            disk_bytes_now: 9 * 25_000_000,
            disk_bytes_after: 10 * 25_000_000,
            last_gap: Some(0.1),
        }
    }

    #[test]
    fn defaults_match_the_documented_growth_config() {
        let c = GrowthConfig::default();
        assert_eq!((c.k, c.max_in_flight, c.max_disk_gb), (1, 50, 10.0));
        assert_eq!((c.max_gap, c.dying_frac_max, c.keep_ratio_min), (0.4, 0.35, 0.35));
        assert_eq!((c.recent_mult, c.mem_frac, c.birth_gate), (4.0, 0.95, 0.001));
    }

    #[test]
    fn a_healthy_pool_grows_and_every_brake_says_yes() {
        let p = healthy();
        let d = decide(&GrowthConfig::default(), &inputs(&p));
        assert_eq!(d.grow, 1);
        assert!(d.brakes.all_ok());
        assert_eq!(d.blocked_by, None);
        assert!(d.reason.starts_with("Adding 1 expert"), "{}", d.reason);
        assert_eq!(d.measures.in_flight, 2);
        assert_eq!((d.measures.judged, d.measures.earning), (4, 4));
        assert_eq!(d.measures.keep_ratio, 1.0);
        for b in [&d.brakes.room, &d.brakes.used, &d.brakes.earning, &d.brakes.fits, &d.brakes.honest] {
            assert!(b.ok && !b.why.is_empty());
        }
    }

    #[test]
    fn k_experts_are_added_when_all_agree() {
        let p = healthy();
        let cfg = GrowthConfig { k: 3, ..GrowthConfig::default() };
        let d = decide(&cfg, &inputs(&p));
        assert_eq!(d.grow, 3);
        assert!(d.reason.contains("3 experts"));
    }

    #[test]
    fn growth_off_never_grows() {
        let p = healthy();
        let cfg = GrowthConfig { k: 0, ..GrowthConfig::default() };
        let d = decide(&cfg, &inputs(&p));
        assert_eq!((d.grow, d.blocked_by), (0, None));
        assert!(d.reason.contains("switched off"));
    }

    #[test]
    fn room_refuses_a_full_pool() {
        let p = healthy();
        let inp = GrowthInputs { max_experts: 9, ..inputs(&p) };
        let d = decide(&GrowthConfig::default(), &inp);
        assert_eq!((d.grow, d.blocked_by), (0, Some(BrakeKind::Room)));
        assert!(!d.brakes.room.ok && d.brakes.used.ok && d.brakes.earning.ok && d.brakes.fits.ok && d.brakes.honest.ok);
        assert!(d.brakes.room.why.contains("pool is full (9 of 9 experts)"), "{}", d.brakes.room.why);
        // exactly at the ceiling still passes
        let at = GrowthInputs { max_experts: 10, ..inputs(&p) };
        assert_eq!(decide(&GrowthConfig::default(), &at).grow, 1);
    }

    #[test]
    fn room_refuses_when_memory_or_disk_is_spent_and_names_every_cause() {
        let p = healthy();
        let inp = GrowthInputs { mem_frac: 0.96, disk_bytes_after: 11_000_000_000, ..inputs(&p) };
        let d = decide(&GrowthConfig::default(), &inp);
        assert_eq!(d.blocked_by, Some(BrakeKind::Room));
        let why = &d.brakes.room.why;
        assert!(why.contains("memory is nearly spent (96% in use, limit 95%)"), "{why}");
        assert!(why.contains("11.0 GB, over the 10.0 GB limit"), "{why}");
        // ties pass: memory exactly at the ceiling, disk exactly at the limit
        let tie = GrowthInputs { mem_frac: 0.95, disk_bytes_after: 10_000_000_000, ..inputs(&p) };
        assert_eq!(decide(&GrowthConfig::default(), &tie).grow, 1);
        // a zero disk limit means no limit
        let cfg = GrowthConfig { max_disk_gb: 0.0, ..GrowthConfig::default() };
        let big = GrowthInputs { disk_bytes_after: u64::MAX, ..inputs(&p) };
        assert_eq!(decide(&cfg, &big).grow, 1);
    }

    #[test]
    fn used_refuses_when_too_many_experts_are_idle() {
        let mut p = healthy();
        p.dying = vec![0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.0, 0.0]; // 4 of 9 idle = 0.444 > 0.35
        let d = decide(&GrowthConfig::default(), &inputs(&p));
        assert_eq!((d.grow, d.blocked_by), (0, Some(BrakeKind::Used)));
        assert_eq!(d.measures.idle, 4);
        assert!(d.brakes.used.why.starts_with("4 of 9 experts are going unused"), "{}", d.brakes.used.why);
    }

    #[test]
    fn earning_refuses_when_recent_additions_are_not_used() {
        let mut p = healthy();
        // the four settled additions are all going stale; USED is told to tolerate that so EARNING is the one to speak
        p.dying = vec![0.1, 0.2, 0.3, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0];
        let cfg = GrowthConfig { dying_frac_max: 1.0, ..GrowthConfig::default() };
        let d = decide(&cfg, &inputs(&p));
        assert_eq!((d.measures.judged, d.measures.earning), (4, 0));
        assert_eq!((d.grow, d.blocked_by), (0, Some(BrakeKind::Earning)));
        assert!(
            d.brakes.earning.why.starts_with("4 of 4 newer experts are not yet useful"),
            "{}",
            d.brakes.earning.why
        );
        // keep_ratio_min 0 never refuses
        let lax = GrowthConfig { keep_ratio_min: 0.0, ..cfg };
        assert_eq!(decide(&lax, &inputs(&p)).grow, 1);
    }

    #[test]
    fn fits_refuses_when_too_many_are_on_trial() {
        let p = healthy();
        let cfg = GrowthConfig { max_in_flight: 2, ..GrowthConfig::default() };
        let d = decide(&cfg, &inputs(&p));
        assert_eq!((d.grow, d.blocked_by), (0, Some(BrakeKind::Fits)));
        assert!(d.brakes.fits.why.starts_with("2 experts are still on trial"), "{}", d.brakes.fits.why);
        // 2 in flight + k 1 == 3 <= 3 passes
        let at = GrowthConfig { max_in_flight: 3, ..GrowthConfig::default() };
        assert_eq!(decide(&at, &inputs(&p)).grow, 1);
        // 0 means no limit
        let off = GrowthConfig { max_in_flight: 0, ..GrowthConfig::default() };
        assert_eq!(decide(&off, &inputs(&p)).grow, 1);
    }

    #[test]
    fn honest_refuses_when_held_out_has_separated_from_training() {
        let p = healthy();
        let d = decide(&GrowthConfig::default(), &GrowthInputs { last_gap: Some(0.5), ..inputs(&p) });
        assert_eq!((d.grow, d.blocked_by), (0, Some(BrakeKind::Honest)));
        assert!(d.brakes.honest.why.contains("memorising"), "{}", d.brakes.honest.why);
        // exactly at the limit is already too far (the check is strict)
        let at = decide(&GrowthConfig::default(), &GrowthInputs { last_gap: Some(0.4), ..inputs(&p) });
        assert_eq!(at.blocked_by, Some(BrakeKind::Honest));
        // no measurement, a check that is off, or a negative gap all pass
        for gap in [None, Some(-1.0)] {
            assert_eq!(decide(&GrowthConfig::default(), &GrowthInputs { last_gap: gap, ..inputs(&p) }).grow, 1);
        }
        let off = GrowthConfig { max_gap: 0.0, ..GrowthConfig::default() };
        let d = decide(&off, &GrowthInputs { last_gap: Some(9.0), ..inputs(&p) });
        assert_eq!(d.grow, 1);
        assert!(d.brakes.honest.why.contains("switched off"));
        // a gap that is not a number cannot be shown to be small
        let nan = decide(&GrowthConfig::default(), &GrowthInputs { last_gap: Some(f64::NAN), ..inputs(&p) });
        assert_eq!(nan.blocked_by, Some(BrakeKind::Honest));
    }

    #[test]
    fn the_first_refusal_follows_the_originals_order() {
        // honest, fits, room, used, earning all refuse at once
        let mut p = healthy();
        p.dying = vec![0.9; 9];
        let cfg = GrowthConfig { max_in_flight: 1, ..GrowthConfig::default() };
        let inp = GrowthInputs { max_experts: 3, mem_frac: 0.99, last_gap: Some(1.0), ..inputs(&p) };
        let d = decide(&cfg, &inp);
        assert!(!d.brakes.all_ok());
        assert_eq!(d.blocked_by, Some(BrakeKind::Honest));
        let no_gap = GrowthInputs { last_gap: None, ..inp };
        assert_eq!(decide(&cfg, &no_gap).blocked_by, Some(BrakeKind::Fits));
        let no_cap = GrowthConfig { max_in_flight: 0, ..cfg };
        assert_eq!(decide(&no_cap, &no_gap).blocked_by, Some(BrakeKind::Room));
        let roomy = GrowthInputs { max_experts: 64, mem_frac: 0.1, ..no_gap };
        assert_eq!(decide(&no_cap, &roomy).blocked_by, Some(BrakeKind::Used));
        let tolerant = GrowthConfig { dying_frac_max: 1.0, ..no_cap };
        assert_eq!(decide(&tolerant, &roomy).blocked_by, Some(BrakeKind::Earning));
    }

    #[test]
    fn an_empty_pool_has_nothing_idle_and_nothing_to_judge() {
        let d = decide(
            &GrowthConfig::default(),
            &GrowthInputs {
                model_step: 100.0,
                max_experts: 8,
                dying: &[],
                dying_at: 0.65,
                born: &[],
                trial: 10.0,
                mem_frac: 0.0,
                disk_bytes_now: 0,
                disk_bytes_after: 0,
                last_gap: None,
            },
        );
        assert_eq!(d.grow, 1);
        assert_eq!((d.measures.idle, d.measures.idle_frac, d.measures.in_flight), (0, 0.0, 0));
        assert_eq!(d.measures.keep_ratio, 1.0);
        assert!(d.brakes.used.why.contains("no experts yet"));
    }

    #[test]
    fn originals_are_never_in_flight_and_never_judged() {
        let born = [0.0; 5];
        assert_eq!(in_flight(&born, 1000.0, 5000.0), 0);
        let e = earning(&born, &[0.9; 5], 0.65, 1000.0, 5000.0, 4.0);
        assert_eq!((e.judged, e.earning, e.ratio), (0, 0, 1.0));
    }

    #[test]
    fn trial_zero_means_nothing_is_in_flight_and_a_default_window_when_judging() {
        assert_eq!(in_flight(&[100.0, 9000.0], 0.0, 9100.0), 0);
        // trial 0 judges with 1600 and 4x: ages 8900 (too old), 1000 (too young), 1700 (judged), 7000 (too old)
        let born = [100.0, 8000.0, 7300.0, 2000.0];
        let e = earning(&born, &[0.1, 0.1, 0.1, 0.1], 0.65, 0.0, 9000.0, 4.0);
        assert_eq!(e.judged, 1);
    }

    #[test]
    fn an_expert_born_in_the_future_counts_as_in_flight() {
        assert_eq!(in_flight(&[0.0, 2000.0], 500.0, 1000.0), 1);
    }

    #[test]
    fn judging_window_is_past_the_trial_and_before_recent_mult_times_it() {
        // trial 1000, mult 4: judged when 1000 <= age < 4000
        let born = [9000.0, 9001.0, 6001.0, 6000.0, 5999.0];
        let e = earning(&born, &[0.1; 5], 0.65, 1000.0, 10_000.0, 4.0);
        assert_eq!(e.judged, 2, "ages 1000 and 3999 are judged; 999, 4000 and 4001 are not");
    }

    #[test]
    fn born_longer_than_dying_scores_only_the_experts_that_have_both() {
        let born = [0.0, 3000.0, 3500.0, 4000.0, 7500.0];
        let e = earning(&born, &[0.1, 0.9, 0.1], 0.65, 1000.0, 8000.0, 4.0);
        assert_eq!(e.judged, 0, "ages 5000, 4500, 4000 are too old and 500 is too young");
        let born = [0.0, 6000.0, 6500.0, 7000.0, 7500.0];
        let e = earning(&born, &[0.1, 0.9, 0.1], 0.65, 1000.0, 8000.0, 4.0);
        assert_eq!(e.judged, 3); // ages 2000, 1500, 1000
        assert_eq!(e.earning, 1); // only index 2 has a dying value below the line among the first three
    }

    #[test]
    fn messages_are_plain_and_carry_the_numbers() {
        let mut p = healthy();
        p.dying = vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.0, 0.0];
        let d = decide(&GrowthConfig::default(), &GrowthInputs { max_experts: 9, last_gap: Some(2.0), ..inputs(&p) });
        for why in
            [&d.brakes.room.why, &d.brakes.used.why, &d.brakes.earning.why, &d.brakes.fits.why, &d.brakes.honest.why]
        {
            assert!(why.ends_with('.') && why.len() > 20, "{why}");
            assert!(!why.contains("dying") && !why.contains("VRAM") && !why.contains("keep_ratio"), "{why}");
        }
        assert_eq!(d.reason, d.brakes.honest.why);
    }
}
