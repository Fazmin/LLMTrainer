//! The pool's bookkeeping: which expert sits in which slot of the card, what the text has voted for, and how long
//! each expert has gone unaddressed. Pure host logic (no tensors), ported from `PagedPool` in the reference.
//!
//! The rule it implements, in one paragraph. Every character ranks the *whole* pool with the router and asks for its
//! top-k experts, each request carrying the router's probability. A forward's first pass adds its characters' requests
//! to the vote of its **text** (everything read since position 0) and the forward is admitted the most-voted experts
//! until the card is full. Everything admitted stays until the forward ends; the next forward starts empty and is
//! decided by the vote again. An expert that is admitted but not on the card takes a slot from an expert nothing
//! admitted this forward: an empty slot first, otherwise the one admitted longest ago.
//!
//! While training, an expert used less than its fair share of recent forwards gets a selection bonus
//! (`explore_bias * exp(-recent / fair)`) so rarely used experts are tried. The bonus only affects what is *chosen*,
//! never how much a chosen expert contributes, and the prune clock (`last_seen`) counts only what the router alone
//! would have admitted.

use std::collections::{BTreeSet, HashMap};

use crate::store::manifest::Telemetry;

/// Descending order of `v`, ties broken towards the lower index (the reference's order is unspecified on ties).
fn argsort_desc(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx
}

/// A text's vote plus one more forward's requests. The pool may have grown since the vote began; growth appends
/// experts, so their counts start at zero at the end.
fn tally(total: Option<Vec<f32>>, add: &[f32]) -> Vec<f32> {
    match total {
        None => add.to_vec(),
        Some(mut t) => {
            if t.len() < add.len() {
                t.resize(add.len(), 0.0);
            }
            for (a, b) in t.iter_mut().zip(add) {
                *a += *b;
            }
            t
        }
    }
}

/// How the slots change when a forward admits new experts.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotPlan {
    /// Slot contents before the change (expert position or -1).
    pub old: Vec<i64>,
    /// Slot contents after the change.
    pub new: Vec<i64>,
}

impl SlotPlan {
    /// Experts that were on the card and are leaving it.
    pub fn leaving(&self) -> Vec<(usize, i64)> {
        self.old.iter().enumerate().filter(|(_, e)| **e >= 0 && !self.new.contains(e)).map(|(s, e)| (s, *e)).collect()
    }

    /// `(slot, expert)` pairs whose slot gets a different expert.
    pub fn arriving(&self) -> Vec<(usize, i64)> {
        self.new.iter().enumerate().filter(|(s, e)| **e >= 0 && self.old[*s] != **e).map(|(s, e)| (s, *e)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct PoolBook {
    /// Slots on the card.
    pub resident: usize,
    /// Experts in the pool (rows of the router).
    pub n: usize,
    /// Slot -> expert position, or -1 for an empty slot.
    pub slots: Vec<i64>,
    admitted: BTreeSet<usize>,
    merited: BTreeSet<usize>,
    voted: bool,
    vote: Option<Vec<f32>>,
    merit_vote: Option<Vec<f32>>,

    // Per-expert arrays, all of length `n`. f64 so counts stay exact over very long runs.
    pub use_: Vec<f64>,
    pub age: Vec<f64>,
    /// Training step at which the expert was created.
    pub born: Vec<f64>,
    pub gate_seen: Vec<f64>,
    /// The prune clock: the text count at which the router last would have admitted the expert.
    pub last_seen: Vec<f64>,
    pub ever: Vec<bool>,
    pub admits: Vec<f64>,
    /// Share of recent training forwards that admitted the expert (decaying).
    pub recent: Vec<f64>,
    /// An expert's name is its uid, not its position: a file keeps its name across a prune.
    pub uid: Vec<u64>,
    pub next_uid: u64,

    /// Texts begun so far: the unit `last_seen` is counted in.
    pub segments: u64,
    pub swaps: u64,
    /// Experts brought onto the card, ever.
    pub loads: u64,
    /// A newborn is safe from pruning for its first `trial` steps. 0 means nothing is ever on trial.
    pub trial: f64,
    /// The training step, set by the trainer.
    pub now: f64,
    /// Share of the survival window unaddressed that counts as dying.
    pub dying_at: f64,
    pub explore_bias: f64,
    pub explore_steps: f64,
    exploring: bool,
    bias: Option<Vec<f32>>,
    /// EMA of routing mass lost to top-k (diagnostic).
    pub pressure: f64,
    /// EMA of how many experts the router would like to use (diagnostic).
    pub want_k: f64,
    /// Uids of the experts the last routed character picked (for the chat view; not saved).
    pub last_pick: Vec<u64>,
}

impl PoolBook {
    pub fn new(n: usize, resident: usize) -> Self {
        let resident = resident.min(n).max(1);
        Self {
            resident,
            n,
            slots: vec![-1; resident],
            admitted: BTreeSet::new(),
            merited: BTreeSet::new(),
            voted: false,
            vote: None,
            merit_vote: None,
            use_: vec![0.0; n],
            age: vec![0.0; n],
            born: vec![0.0; n],
            gate_seen: vec![0.0; n],
            last_seen: vec![0.0; n],
            ever: vec![false; n],
            admits: vec![0.0; n],
            recent: vec![0.0; n],
            uid: (0..n as u64).collect(),
            next_uid: n as u64,
            segments: 0,
            swaps: 0,
            loads: 0,
            trial: 0.0,
            now: 0.0,
            dying_at: 0.75,
            explore_bias: 0.0,
            explore_steps: 1000.0,
            exploring: false,
            bias: None,
            pressure: 0.0,
            want_k: 0.0,
            last_pick: Vec::new(),
        }
    }

    // ---- a forward's lifecycle ------------------------------------------------------------------------------

    /// A new forward: nothing admitted yet; the card keeps its contents. `explore` is set for a forward that trains:
    /// its selection bonus is fixed here from how much each expert has been used, and `recent` decays one step.
    pub fn begin_forward(&mut self, explore: bool) {
        self.admitted.clear();
        self.merited.clear();
        self.voted = false;
        self.bias = None;
        self.exploring = explore && self.explore_bias > 0.0;
        if self.exploring {
            let fair = self.resident as f64 / self.n.max(1) as f64;
            self.bias = Some(self.recent.iter().map(|r| (self.explore_bias * (-r / fair).exp()) as f32).collect());
            let decay = 1.0 - 1.0 / self.explore_steps;
            self.recent.iter_mut().for_each(|r| *r *= decay);
        }
    }

    /// A forward from position 0: a new text. Its vote starts empty, and the prune clock counts it.
    pub fn begin_text(&mut self, explore: bool) {
        self.vote = None;
        self.merit_vote = None;
        self.begin_forward(explore);
        self.segments += 1;
    }

    /// The bonus each expert's router score gets when choosing, or `None` when this forward is not exploring.
    pub fn selection_bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Whether this forward may still admit experts.
    pub fn admitting(&self) -> bool {
        self.admitted.len() < self.resident
    }

    pub fn n_admitted(&self) -> usize {
        self.admitted.len()
    }

    pub fn is_admitted(&self, expert: usize) -> bool {
        self.admitted.contains(&expert)
    }

    /// Which slots hold an expert this forward admitted, in slot order.
    pub fn admitted_mask(&self) -> Vec<bool> {
        self.slots.iter().map(|&e| e >= 0 && self.admitted.contains(&(e as usize))).collect()
    }

    /// Router rows owned by the resident experts, in slot order (empty slots point at row 0, as the reference does).
    pub fn resident_rows(&self) -> Vec<u32> {
        self.slots.iter().map(|&e| e.max(0) as u32).collect()
    }

    pub fn experts_in_slots(&self) -> Vec<usize> {
        self.slots.iter().filter(|&&e| e >= 0).map(|&e| e as usize).collect()
    }

    pub fn merited(&self) -> Vec<usize> {
        self.merited.iter().copied().collect()
    }

    pub fn admitted(&self) -> Vec<usize> {
        self.admitted.iter().copied().collect()
    }

    // ---- admission ---------------------------------------------------------------------------------------------

    /// Admit the most-requested experts, up to the card's capacity.
    ///
    /// `mass[e]` is the router probability this pass's requests put on expert `e`. `merit` is the same without the
    /// exploration bonus, when there is one: the experts it would have admitted are the ones the prune clock counts,
    /// so being tried does not keep an expert alive and being wanted does.
    ///
    /// Returns the change to make to the slots, or `None` when nothing needs to move. The caller moves the data and
    /// then calls [`PoolBook::set_slots`].
    pub fn admit(&mut self, mass: &[f32], merit: Option<&[f32]>) -> Option<SlotPlan> {
        let mut m: Vec<f32> = mass.to_vec();
        let mut mm: Vec<f32> = merit.unwrap_or(mass).to_vec();
        if !self.voted {
            // this forward's first pass: its requests join the text's vote, and the forward is admitted by the whole vote
            self.voted = true;
            let v = tally(self.vote.take(), &m);
            let mv = tally(self.merit_vote.take(), &mm);
            m = v.clone();
            mm = mv.clone();
            self.vote = Some(v);
            self.merit_vote = Some(mv);
        }
        let mfree = self.resident.saturating_sub(self.merited.len());
        if mfree > 0 {
            let fresh: Vec<usize> = argsort_desc(&mm)
                .into_iter()
                .filter(|&e| mm[e] > 0.0 && !self.merited.contains(&e) && e < self.n)
                .take(mfree)
                .collect();
            for e in fresh {
                self.merited.insert(e);
                self.last_seen[e] = self.segments as f64; // the prune clock
            }
        }
        let free = self.resident.saturating_sub(self.admitted.len());
        if free == 0 {
            return None;
        }
        let new: Vec<usize> = argsort_desc(&m)
            .into_iter()
            .filter(|&e| m[e] > 0.0 && !self.admitted.contains(&e) && e < self.n)
            .take(free)
            .collect();
        if new.is_empty() {
            return None;
        }
        self.admitted.extend(new.iter().copied());
        let here: HashMap<i64, usize> =
            self.slots.iter().enumerate().filter(|(_, e)| **e >= 0).map(|(s, e)| (*e, s)).collect();

        // A newcomer takes a slot holding nothing this forward admitted: an empty one first, then the one whose expert
        // was admitted longest ago (the least recently used, so the least likely to be wanted back).
        let mut victims: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, e)| **e < 0 || !self.admitted.contains(&(**e as usize)))
            .map(|(s, _)| s)
            .collect();
        victims.sort_by(|&a, &b| {
            let key = |s: usize| -> (bool, f64) {
                let e = self.slots[s];
                if e >= 0 { (true, self.last_seen[e as usize]) } else { (false, -1.0) }
            };
            let (ka, kb) = (key(a), key(b));
            ka.0.cmp(&kb.0).then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut victims = victims.into_iter();
        let mut plan = self.slots.clone();
        for &e in &new {
            if !here.contains_key(&(e as i64))
                && let Some(s) = victims.next()
            {
                plan[s] = e as i64;
            }
        }
        let loads = plan.iter().filter(|&&e| e >= 0 && !here.contains_key(&e)).count();
        let changed = plan != self.slots;
        if changed {
            for &e in &plan {
                if e >= 0 && !here.contains_key(&e) {
                    self.admits[e as usize] += 1.0;
                }
            }
            self.swaps += 1;
        }
        for &e in &new {
            self.ever[e] = true;
        }
        if self.exploring {
            for &e in &new {
                self.recent[e] += 1.0 / self.explore_steps;
            }
        }
        self.loads += loads as u64;
        changed.then(|| SlotPlan { old: self.slots.clone(), new: plan })
    }

    /// Record that the data has been moved and the card now looks like `new`.
    pub fn set_slots(&mut self, new: Vec<i64>) {
        self.slots = new;
    }

    // ---- staleness -----------------------------------------------------------------------------------------------

    /// How close each expert is to being deleted, as a share of the survival window: 0.0 means something asked for it
    /// just now; 1.0 means it has gone exactly as long unaddressed as `trial`, the line prune deletes on.
    ///
    /// A newborn inside its trial returns 0: it has not failed to be chosen, it has not finished being offered.
    pub fn dying(&self) -> Vec<f64> {
        let n = self.n;
        let zeros = vec![0.0; n];
        let (survival, step) = (self.trial.max(0.0), self.now.max(0.0));
        if n == 0 || survival <= 0.0 || step <= 0.0 {
            return zeros;
        }
        let per_step = self.segments as f64 / step.max(1.0);
        let window = survival * per_step;
        if window <= 0.0 {
            return zeros;
        }
        (0..n)
            .map(|i| {
                // An expert cannot have gone unaddressed for longer than it has existed.
                let born_seg = (step - self.born[i]).max(0.0) * per_step;
                let idle = (self.segments as f64 - self.last_seen[i]).max(0.0);
                let young = (step - self.born[i]) < survival;
                if young { 0.0 } else { idle.min(born_seg) / window }
            })
            .collect()
    }

    /// Which experts prune would delete now: not resident, not protected, past their trial and unaddressed for longer
    /// than the survival window. Returns the positions to *keep*. If everything qualifies, the most recently wanted
    /// expert stays (a pool of zero experts cannot route).
    pub fn prune_keep(&self, step: f64, survival: f64, protect: usize) -> Vec<usize> {
        let per_step = self.segments as f64 / step.max(1.0);
        let window = if per_step > 0.0 { survival * per_step } else { f64::INFINITY };
        let mut keep = Vec::new();
        for i in 0..self.n {
            if i < protect || self.slots.contains(&(i as i64)) {
                keep.push(i);
                continue;
            }
            let young = (step - self.born[i]) < survival;
            let seen = (self.segments as f64 - self.last_seen[i]) <= window;
            if young || seen {
                keep.push(i);
            }
        }
        if keep.is_empty() && self.n > 0 {
            let best = (0..self.n)
                .max_by(|&a, &b| self.last_seen[a].partial_cmp(&self.last_seen[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0);
            keep.push(best);
        }
        keep
    }

    // ---- growing and pruning the arrays ------------------------------------------------------------------------

    /// Append `k` newborn experts; returns their uids. `step` is the training step they are born at.
    pub fn grow(&mut self, k: usize, step: f64) -> Vec<u64> {
        let uids: Vec<u64> = (0..k as u64).map(|i| self.next_uid + i).collect();
        self.next_uid += k as u64;
        for _ in 0..k {
            self.use_.push(0.0);
            self.age.push(0.0);
            self.born.push(step);
            self.gate_seen.push(0.0);
            self.last_seen.push(0.0);
            self.ever.push(false);
            self.admits.push(0.0);
            self.recent.push(0.0);
        }
        self.uid.extend(&uids);
        self.n += k;
        uids
    }

    /// Keep only the experts at `keep` (ascending positions), renumbering slots and dropping the in-progress vote.
    pub fn select(&mut self, keep: &[usize]) {
        fn pick<T: Clone>(v: &[T], keep: &[usize]) -> Vec<T> {
            keep.iter().map(|&i| v[i].clone()).collect()
        }
        self.use_ = pick(&self.use_, keep);
        self.age = pick(&self.age, keep);
        self.born = pick(&self.born, keep);
        self.gate_seen = pick(&self.gate_seen, keep);
        self.last_seen = pick(&self.last_seen, keep);
        self.ever = pick(&self.ever, keep);
        self.admits = pick(&self.admits, keep);
        self.recent = pick(&self.recent, keep);
        self.uid = pick(&self.uid, keep);
        let remap: HashMap<i64, i64> = keep.iter().enumerate().map(|(new, &old)| (old as i64, new as i64)).collect();
        for s in &mut self.slots {
            *s = remap.get(s).copied().unwrap_or(-1);
        }
        // a vote is counted by position in the pool, which pruning renumbers
        self.vote = None;
        self.merit_vote = None;
        self.admitted.clear();
        self.merited.clear();
        self.n = keep.len();
    }

    /// Routing counts arrive per *slot*; usage is kept per *expert*.
    pub fn note_use(&mut self, hit_per_slot: &[f32]) {
        for (s, &h) in hit_per_slot.iter().enumerate() {
            let e = self.slots[s].max(0) as usize;
            if e < self.n {
                self.use_[e] += h as f64;
            }
        }
        self.age.iter_mut().for_each(|a| *a += 1.0);
    }

    /// Update the diagnostic EMAs after a routing pass.
    pub fn note_pressure(&mut self, kept: f64, want: f64) {
        self.pressure = 0.9 * self.pressure + 0.1 * (1.0 - kept);
        self.want_k = 0.9 * self.want_k + 0.1 * want;
    }

    /// What the growth brakes read: how entropic and how concentrated routing has been (diagnostic only).
    pub fn usage_shares(&self) -> Vec<f64> {
        let total: f64 = self.use_.iter().sum::<f64>().max(1.0);
        self.use_.iter().map(|u| u / total).collect()
    }

    // ---- manifest --------------------------------------------------------------------------------------------

    /// Per-expert history for a checkpoint; `gate` values come from the device.
    pub fn telemetry(&self, gate: &[f32]) -> Telemetry {
        let r6 = |v: f64| (v * 1e6).round() / 1e6;
        Telemetry {
            gate: gate.iter().map(|&g| r6(g as f64)).collect(),
            gate_seen: self.gate_seen.iter().map(|&g| r6(g)).collect(),
            usage: self.use_.clone(),
            admits: self.admits.clone(),
            born: self.born.clone(),
            last_seen: self.last_seen.clone(),
            recent: self.recent.iter().map(|&g| r6(g)).collect(),
            ever: self.ever.clone(),
            uid: self.uid.clone(),
            next_uid: Some(self.next_uid),
            segments: Some(self.segments),
            extra: Default::default(),
        }
    }

    /// Put back what a checkpoint carried, ignoring anything that no longer fits (the pool may have been resized).
    pub fn load_telemetry(&mut self, t: &Telemetry) {
        fn fill(dst: &mut [f64], src: &[f64]) {
            let k = dst.len().min(src.len());
            for (d, s) in dst[..k].iter_mut().zip(&src[..k]) {
                if s.is_finite() {
                    *d = *s;
                }
            }
        }
        fill(&mut self.use_, &t.usage);
        fill(&mut self.admits, &t.admits);
        fill(&mut self.born, &t.born);
        fill(&mut self.last_seen, &t.last_seen);
        fill(&mut self.gate_seen, &t.gate_seen);
        fill(&mut self.recent, &t.recent);
        let k = self.ever.len().min(t.ever.len());
        self.ever[..k].copy_from_slice(&t.ever[..k]);
        let k = self.uid.len().min(t.uid.len());
        self.uid[..k].copy_from_slice(&t.uid[..k]);
        // A directory written before ids existed has files named by position, which is what `uid = arange` gives.
        self.next_uid = match t.next_uid {
            Some(n) if n > 0 => n,
            _ => self.uid.iter().max().map(|m| m + 1).unwrap_or(0),
        };
        if let Some(sg) = t.segments.filter(|&s| s > 0) {
            self.segments = sg;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mass(n: usize, hot: &[(usize, f32)]) -> Vec<f32> {
        let mut m = vec![0.0; n];
        for &(e, v) in hot {
            m[e] = v;
        }
        m
    }

    #[test]
    fn first_admission_fills_empty_slots_by_vote() {
        let mut b = PoolBook::new(8, 4);
        b.begin_text(false);
        let plan = b.admit(&mass(8, &[(5, 3.0), (2, 2.0), (7, 1.0), (0, 0.5), (1, 0.1)]), None).unwrap();
        assert_eq!(plan.new, vec![5, 2, 7, 0]);
        assert_eq!(plan.old, vec![-1; 4]);
        b.set_slots(plan.new.clone());
        assert!(!b.admitting());
        assert_eq!(b.admitted(), vec![0, 2, 5, 7]);
        assert_eq!(b.admitted_mask(), vec![true; 4]);
        assert_eq!(b.loads, 4);
        assert_eq!(b.admits[5], 1.0);
        assert!(b.ever[5] && !b.ever[1]);
        assert_eq!(plan.arriving().len(), 4);
        assert!(plan.leaving().is_empty());
    }

    #[test]
    fn nothing_moves_when_the_card_already_holds_what_is_asked_for() {
        let mut b = PoolBook::new(8, 4);
        b.begin_text(false);
        let p = b.admit(&mass(8, &[(1, 4.0), (2, 3.0), (3, 2.0), (4, 1.0)]), None).unwrap();
        b.set_slots(p.new);
        let (swaps, loads) = (b.swaps, b.loads);
        b.begin_text(false);
        assert!(b.admit(&mass(8, &[(4, 1.0), (3, 2.0), (2, 3.0), (1, 4.0)]), None).is_none());
        assert_eq!((b.swaps, b.loads), (swaps, loads));
        // ...but the forward is still admitted those experts, in their own slots
        assert_eq!(b.admitted_mask(), vec![true; 4]);
    }

    #[test]
    fn a_newcomer_evicts_the_least_recently_wanted_expert_nobody_admitted() {
        let mut b = PoolBook::new(8, 4);
        b.begin_text(false);
        let p = b.admit(&mass(8, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]), None).unwrap();
        b.set_slots(p.new);
        // later texts keep asking for 0, 1, 3 but nothing asks for 2
        b.last_seen[0] = 9.0;
        b.last_seen[1] = 9.0;
        b.last_seen[2] = 2.0;
        b.last_seen[3] = 9.0;
        b.begin_text(false);
        let plan = b.admit(&mass(8, &[(6, 5.0), (0, 4.0), (1, 3.0), (3, 2.0)]), None).unwrap();
        assert_eq!(plan.new, vec![0, 1, 6, 3], "expert 2 (oldest last_seen) gave up its slot to 6");
        assert_eq!(plan.leaving(), vec![(2, 2)]);
        assert_eq!(plan.arriving(), vec![(2, 6)]);
    }

    #[test]
    fn the_text_vote_accumulates_across_forwards_of_one_text() {
        let mut b = PoolBook::new(6, 2);
        b.begin_text(false);
        let p = b.admit(&mass(6, &[(0, 1.0), (1, 0.9)]), None).unwrap();
        b.set_slots(p.new);
        // second forward of the same text: expert 3 is asked for a little, 0 and 1 not at all - the vote still has them
        b.begin_forward(false);
        b.admit(&mass(6, &[(3, 0.2)]), None);
        assert_eq!(b.admitted(), vec![0, 1], "the cumulative vote, not this pass alone, decides");
        // a new text starts the vote afresh
        b.begin_text(false);
        let p = b.admit(&mass(6, &[(3, 0.2), (4, 0.1)]), None).unwrap();
        assert_eq!(p.new.iter().filter(|&&e| e == 3 || e == 4).count(), 2);
    }

    #[test]
    fn exploration_bonus_decays_with_use_and_never_touches_the_prune_clock() {
        let mut b = PoolBook::new(8, 4);
        b.explore_bias = 0.65;
        b.explore_steps = 1000.0;
        b.begin_text(true);
        let bias = b.selection_bias().unwrap().to_vec();
        assert!(bias.iter().all(|&x| (x - 0.65).abs() < 1e-6), "nothing used yet: the whole bonus");
        // the bonus decided admission (expert 6), the router alone wanted 0..3
        let plan = b
            .admit(&mass(8, &[(6, 9.0), (0, 1.0)]), Some(&mass(8, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)])))
            .unwrap();
        assert!(plan.new.contains(&6));
        assert_eq!(b.merited(), vec![0, 1, 2, 3]);
        assert_eq!(b.last_seen[0], 1.0);
        assert_eq!(b.last_seen[6], 0.0, "being tried keeps nothing alive");
        assert!(b.recent[6] > 0.0);
        b.begin_forward(true);
        let bias2 = b.selection_bias().unwrap();
        assert!(bias2[6] < 0.65 && (bias2[7] - 0.65).abs() < 1e-6);
        b.begin_forward(false);
        assert!(b.selection_bias().is_none());
    }

    #[test]
    fn dying_reads_staleness_and_exempts_newborns() {
        let mut b = PoolBook::new(3, 2);
        b.trial = 100.0;
        b.now = 1000.0;
        b.segments = 1000;
        b.born = vec![0.0, 0.0, 950.0];
        b.last_seen = vec![1000.0, 500.0, 0.0];
        let d = b.dying();
        assert!(d[0].abs() < 1e-9, "wanted just now");
        assert!((d[1] - 5.0).abs() < 1e-9, "500 texts unaddressed over a 100-text window");
        assert_eq!(d[2], 0.0, "a newborn is on trial");
        b.trial = 0.0;
        assert!(b.dying().iter().all(|&x| x == 0.0), "nothing is ever on trial without a trial window");
    }

    #[test]
    fn prune_keeps_resident_protected_young_and_recently_seen() {
        let mut b = PoolBook::new(6, 2);
        b.segments = 1000;
        b.born = vec![0.0, 0.0, 0.0, 0.0, 0.0, 990.0];
        b.last_seen = vec![0.0; 6];
        b.last_seen[3] = 995.0; // recently wanted
        b.slots = vec![1, -1];
        // survival 100 steps at 1 text/step: window 100 texts
        let keep = b.prune_keep(1000.0, 100.0, 1);
        assert_eq!(keep, vec![0, 1, 3, 5], "0 protected, 1 resident, 3 recent, 5 young; 2 and 4 are deleted");
        // everything stale and nothing protecting it: the most recently wanted one survives
        let mut c = PoolBook::new(3, 1);
        c.segments = 1000;
        c.last_seen = vec![1.0, 7.0, 3.0];
        assert_eq!(c.prune_keep(1000.0, 100.0, 0), vec![1]);
    }

    #[test]
    fn grow_and_select_keep_every_array_in_step() {
        let mut b = PoolBook::new(4, 2);
        b.slots = vec![3, 1];
        b.last_seen = vec![1.0, 2.0, 3.0, 4.0];
        let uids = b.grow(2, 77.0);
        assert_eq!(uids, vec![4, 5]);
        assert_eq!(b.n, 6);
        assert_eq!(b.born[4], 77.0);
        b.select(&[1, 3, 5]);
        assert_eq!(b.n, 3);
        assert_eq!(b.uid, vec![1, 3, 5]);
        assert_eq!(b.last_seen, vec![2.0, 4.0, 0.0]);
        assert_eq!(b.slots, vec![1, 0], "slots follow their experts to the new positions");
        b.slots = vec![0, 2];
        b.select(&[2]);
        assert_eq!(b.slots, vec![-1, 0]);
    }

    #[test]
    fn telemetry_round_trips_and_tolerates_resizing() {
        let mut b = PoolBook::new(4, 2);
        b.use_ = vec![1.0, 2.0, 3.0, 4.0];
        b.segments = 12;
        b.next_uid = 9;
        b.uid = vec![0, 1, 2, 8];
        let t = b.telemetry(&[1.0, 0.5, 0.25, 0.001]);
        let json = serde_json::to_string(&t).unwrap();
        let back: Telemetry = serde_json::from_str(&json).unwrap();
        let mut c = PoolBook::new(3, 2);
        c.load_telemetry(&back);
        assert_eq!(c.use_, vec![1.0, 2.0, 3.0]);
        assert_eq!(c.segments, 12);
        assert_eq!(c.next_uid, 9);
        // an old manifest with no ids
        let mut d = PoolBook::new(3, 2);
        d.load_telemetry(&Telemetry::default());
        assert_eq!(d.next_uid, 3);
    }

    #[test]
    fn note_use_maps_slots_back_to_experts() {
        let mut b = PoolBook::new(5, 2);
        b.slots = vec![4, 2];
        b.note_use(&[3.0, 1.0]);
        assert_eq!(b.use_, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
        assert!(b.age.iter().all(|&a| a == 1.0));
    }
}
