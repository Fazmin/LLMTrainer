//! Synthetic arithmetic text, a port of mini-AGI's `corpora/arithmetic.py`.
//!
//! Every line is `"{task} {args} = {answer}\n"`, and a share of them (`notes_frac`, 30 % by default) carry a
//! `<think> ... </think> ` scratchpad between the `=` and the answer. The formats follow the original:
//!
//! * Answers are written most significant digit first, like every other number in the corpus.
//! * A scratchpad shows the *work*, never just the result: addition and subtraction walk the digits right to left
//!   with their carries (`7+8+0=5c1`) and borrows (`3-5-0=8b1`), multiplication lists the partial products and the
//!   sums that combine them, `mod` brackets the quotient, `gcd` lists the Euclid steps, `cmp` names the first
//!   differing digit, `round` shows the remainder test (half-up, not half-to-even).
//!
//! | task    | weight | digits (uniform in 1..=max) | notes                                   |
//! |---------|--------|-----------------------------|-----------------------------------------|
//! | `add`   | 0.30   | 8                           |                                         |
//! | `sub`   | 0.20   | 8                           | answer may be negative                  |
//! | `mul`   | 0.18   | 4                           | the second factor has 1 or 2 digits     |
//! | `cmp`   | 0.09   | 8                           | `<`, `>`, `==`; answer `yes` / `no`     |
//! | `mod`   | 0.08   | 6                           | modulus in 2..=99                       |
//! | `gcd`   | 0.06   | 4                           |                                         |
//! | `sum`   | 0.06   | 4                           | 2 to 5 terms                            |
//! | `round` | 0.03   | 6                           | to 10, 100 or 1000, half up             |
//!
//! Two deliberate differences from the original: the validation set is generated first, from its own seed, and any
//! training line whose *problem* (task and arguments) appears in it is dropped and redrawn, so validation never
//! measures memorisation; and the random streams are ChaCha instead of Python's Mersenne Twister, so lines are not
//! bit-identical to the original's for the same seed (the distribution and formats are).

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use minagi_types::{ArithmeticParams, JobProgress};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{DataError, DataResult};
use crate::util::{CancelFlag, PROGRESS_INTERVAL, RateMeter, Throttle};

/// Largest shard file written, in bytes.
pub const SHARD_BYTES: u64 = 8 * 1024 * 1024;
/// The lane's name (and so its folder under `train/` and `val/`).
pub const LANE: &str = "arithmetic";

/// The eight problem types, in the original's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Task {
    Add,
    Sub,
    Mul,
    Cmp,
    Mod,
    Gcd,
    Sum,
    Round,
}

impl Task {
    /// Tasks with their probability and largest digit count, in the order the original draws them.
    pub const WEIGHTS: [(Task, f64, u32); 8] = [
        (Task::Add, 0.30, 8),
        (Task::Sub, 0.20, 8),
        (Task::Mul, 0.18, 4),
        (Task::Cmp, 0.09, 8),
        (Task::Mod, 0.08, 6),
        (Task::Gcd, 0.06, 4),
        (Task::Sum, 0.06, 4),
        (Task::Round, 0.03, 6),
    ];

    /// The word a line starts with.
    pub fn word(self) -> &'static str {
        match self {
            Task::Add => "add",
            Task::Sub => "sub",
            Task::Mul => "mul",
            Task::Cmp => "cmp",
            Task::Mod => "mod",
            Task::Gcd => "gcd",
            Task::Sum => "sum",
            Task::Round => "round",
        }
    }

    pub fn from_word(word: &str) -> Option<Task> {
        Task::WEIGHTS.iter().map(|&(t, _, _)| t).find(|t| t.word() == word)
    }
}

/// A number with exactly `digits` digits (a single digit may be 0), like the original's `rnd`.
fn rnd(rng: &mut impl Rng, digits: u32) -> u64 {
    let lo = if digits > 1 { 10u64.pow(digits - 1) } else { 0 };
    rng.random_range(lo..=10u64.pow(digits) - 1)
}

/// The scratchpad wrapper: `<think> body </think> `.
fn think(body: &str) -> String {
    format!("<think> {body} </think> ")
}

/// Decimal digits of `n`, least significant first.
fn digits_rev(n: u64) -> Vec<u64> {
    n.to_string().bytes().rev().map(|b| u64::from(b - b'0')).collect()
}

fn gen_add(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d);
    let b_digits = rng.random_range(1..=d);
    let b = rnd(rng, b_digits);
    let t = if notes {
        let (da, db) = (digits_rev(a), digits_rev(b));
        let (mut steps, mut carry) = (Vec::new(), 0);
        for i in 0..da.len().max(db.len()) {
            let x = da.get(i).copied().unwrap_or(0);
            let y = db.get(i).copied().unwrap_or(0);
            let s = x + y + carry;
            steps.push(format!("{x}+{y}+{carry}={}c{}", s % 10, s / 10));
            carry = s / 10;
        }
        if carry > 0 {
            steps.push(format!("c{carry}"));
        }
        think(&steps.join(" "))
    } else {
        String::new()
    };
    format!("add {a} + {b} = {t}{}", a + b)
}

fn gen_sub(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d);
    let b_digits = rng.random_range(1..=d);
    let b = rnd(rng, b_digits);
    let t = if notes {
        // The borrow chain runs right to left over the larger magnitude; the sign is a separate fact.
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        let (dh, dl) = (digits_rev(hi), digits_rev(lo));
        let (mut steps, mut borrow) = (Vec::new(), 0i64);
        for (i, &x) in dh.iter().enumerate() {
            let x = x as i64;
            let y = dl.get(i).copied().unwrap_or(0) as i64;
            let v = x - y - borrow;
            let next = i64::from(v < 0);
            steps.push(format!("{x}-{y}-{borrow}={}b{next}", v.rem_euclid(10)));
            borrow = next;
        }
        if a < b {
            steps.push("sign -".to_string());
        }
        think(&steps.join(" "))
    } else {
        String::new()
    };
    format!("sub {a} - {b} = {t}{}", a as i64 - b as i64)
}

fn gen_mul(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d);
    let b_digits = rng.random_range(1..=d.min(2));
    let b = rnd(rng, b_digits);
    let t = if notes {
        // Every partial product, then the additions that combine them.
        let (mut vals, mut parts) = (Vec::new(), Vec::new());
        for (i, dg) in digits_rev(b).into_iter().enumerate() {
            let v = a * dg * 10u64.pow(i as u32);
            vals.push(v);
            parts.push(format!("{a}*{dg}{}={v}", "0".repeat(i)));
        }
        let mut acc = vals[0];
        for &v in &vals[1..] {
            parts.push(format!("{acc}+{v}={}", acc + v));
            acc += v;
        }
        think(&parts.join(" "))
    } else {
        String::new()
    };
    format!("mul {a} * {b} = {t}{}", a * b)
}

fn gen_cmp(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d);
    let b = rnd(rng, d);
    let op = ["<", ">", "=="][rng.random_range(0..3)];
    let truth = match op {
        "<" => a < b,
        ">" => a > b,
        _ => a == b,
    };
    let t = if notes {
        let (sa, sb) = (a.to_string(), b.to_string());
        let body = if sa.len() != sb.len() {
            format!("len {} vs {}", sa.len(), sb.len())
        } else {
            // Equal lengths are the whole difficulty; the first differing digit decides.
            match sa.bytes().zip(sb.bytes()).position(|(x, y)| x != y) {
                None => format!("len {} vs {} equal", sa.len(), sb.len()),
                Some(k) => format!("len {} vs {} digit {k} {} vs {}", sa.len(), sb.len(), &sa[k..=k], &sb[k..=k]),
            }
        };
        think(&body)
    } else {
        String::new()
    };
    format!("cmp {a} {op} {b} = {t}{}", if truth { "yes" } else { "no" })
}

fn gen_mod(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d);
    let b = rng.random_range(2..=99u64);
    let t = if notes {
        // b*q fits under a, b*(q+1) does not, and the remainder is the subtraction.
        let q = a / b;
        think(&format!("{b}*{q}={} {b}*{}={}>{a} {a}-{}={}", b * q, q + 1, b * (q + 1), b * q, a % b))
    } else {
        String::new()
    };
    format!("mod {a} % {b} = {t}{}", a % b)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn gen_gcd(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d.min(4));
    let b = rnd(rng, d.min(4));
    let t = if notes {
        let (mut steps, mut x, mut y) = (Vec::new(), a, b);
        while y != 0 && steps.len() < 8 {
            steps.push(format!("{x}%{y}={}", x % y));
            (x, y) = (y, x % y);
        }
        think(&steps.join(" "))
    } else {
        String::new()
    };
    format!("gcd {a} , {b} = {t}{}", gcd(a, b))
}

fn gen_sum(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let k = rng.random_range(2..=5);
    let xs: Vec<u64> = (0..k)
        .map(|_| {
            let digits = rng.random_range(1..=d.min(4));
            rnd(rng, digits)
        })
        .collect();
    let body = xs.iter().map(u64::to_string).collect::<Vec<_>>().join(" + ");
    let t = if notes {
        let mut run = 0;
        let steps: Vec<String> = xs
            .iter()
            .map(|x| {
                run += x;
                run.to_string()
            })
            .collect();
        think(&steps.join(" "))
    } else {
        String::new()
    };
    format!("sum {body} = {t}{}", xs.iter().sum::<u64>())
}

/// `a` rounded to the nearest multiple of `p`, ties going up.
fn round_half_up(a: u64, p: u64) -> u64 {
    let (q, r) = (a / p, a % p);
    if r * 2 >= p { (q + 1) * p } else { q * p }
}

fn gen_round(rng: &mut impl Rng, d: u32, notes: bool) -> String {
    let a = rnd(rng, d.max(2));
    let p = [10u64, 100, 1000][rng.random_range(0..3)];
    // Half up, not half to even: the working says "r >= p/2 goes up", and a rule the working contradicts on exactly
    // the hard cases is worse than none.
    let (q, r) = (a / p, a % p);
    let up = r * 2 >= p;
    let res = round_half_up(a, p);
    let t = if notes {
        think(&format!(
            "{a}={q}*{p}+{r} {r}{}{} -> {}",
            if up { ">=" } else { "<" },
            p / 2,
            if up { "up" } else { "down" }
        ))
    } else {
        String::new()
    };
    format!("round {a} to {p} = {t}{res}")
}

/// Draw one line (without the trailing newline): a task by weight, a digit count, and a scratchpad with probability
/// `notes_frac`. The random calls follow the original's order.
pub fn sample_line(rng: &mut impl Rng, notes_frac: f64) -> String {
    let r: f64 = rng.random();
    let notes = rng.random::<f64>() < notes_frac;
    let mut acc = 0.0;
    for (task, weight, max_digits) in Task::WEIGHTS {
        acc += weight;
        if r <= acc {
            let digits = rng.random_range(1..=max_digits);
            return gen_task(rng, task, digits, notes);
        }
    }
    // The weights sum to 1.0 only up to rounding; the original falls back to addition.
    let digits = rng.random_range(1..=8);
    gen_task(rng, Task::Add, digits, notes)
}

fn gen_task(rng: &mut impl Rng, task: Task, digits: u32, notes: bool) -> String {
    match task {
        Task::Add => gen_add(rng, digits, notes),
        Task::Sub => gen_sub(rng, digits, notes),
        Task::Mul => gen_mul(rng, digits, notes),
        Task::Cmp => gen_cmp(rng, digits, notes),
        Task::Mod => gen_mod(rng, digits, notes),
        Task::Gcd => gen_gcd(rng, digits, notes),
        Task::Sum => gen_sum(rng, digits, notes),
        Task::Round => gen_round(rng, digits, notes),
    }
}

/// The problem a line poses: everything before the first `" = "` (task and arguments, no scratchpad or answer).
/// Used to keep training problems out of the validation set.
pub fn problem_key(line: &str) -> &str {
    line.split_once(" = ").map_or(line, |(problem, _)| problem)
}

/// Which random stream a split uses. Different splits (and seeds) give unrelated streams.
fn stream(seed: u32, split: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(u64::from(seed) | (split << 32))
}

/// Writes lines into `part-NNNN.txt` files of at most `limit` bytes. Each shard is written as `.part` and renamed
/// when complete.
struct ShardWriter {
    dir: PathBuf,
    limit: u64,
    index: u32,
    current: Option<OpenShard>,
    bytes: u64,
}

struct OpenShard {
    out: BufWriter<File>,
    tmp: PathBuf,
    path: PathBuf,
    size: u64,
}

impl ShardWriter {
    /// Start writing into `dir`, removing shards of an earlier run first.
    fn new(dir: PathBuf, limit: u64) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("part-") && (name.ends_with(".txt") || name.ends_with(".part")) {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(Self { dir, limit, index: 0, current: None, bytes: 0 })
    }

    fn write_line(&mut self, line: &str) -> io::Result<()> {
        let len = line.len() as u64 + 1;
        if self.current.as_ref().is_some_and(|c| c.size + len > self.limit) {
            self.close_current()?;
        }
        if self.current.is_none() {
            let path = self.dir.join(format!("part-{:04}.txt", self.index));
            let tmp = self.dir.join(format!("part-{:04}.txt.part", self.index));
            let out = BufWriter::with_capacity(256 * 1024, File::create(&tmp)?);
            self.current = Some(OpenShard { out, tmp, path, size: 0 });
            self.index += 1;
        }
        if let Some(shard) = self.current.as_mut() {
            shard.out.write_all(line.as_bytes())?;
            shard.out.write_all(b"\n")?;
            shard.size += len;
        }
        self.bytes += len;
        Ok(())
    }

    fn close_current(&mut self) -> io::Result<()> {
        if let Some(mut shard) = self.current.take() {
            shard.out.flush()?;
            drop(shard.out);
            fs::rename(&shard.tmp, &shard.path)?;
        }
        Ok(())
    }

    /// Finish the last shard; returns (bytes written, shard count).
    fn finish(mut self) -> io::Result<(u64, u32)> {
        self.close_current()?;
        Ok((self.bytes, self.index))
    }

    /// Drop an unfinished shard after an error or cancellation.
    fn abandon(mut self) {
        if let Some(shard) = self.current.take() {
            drop(shard.out);
            let _ = fs::remove_file(&shard.tmp);
        }
    }
}

/// What [`generate`] wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticReport {
    pub train_problems: u64,
    pub val_problems: u64,
    pub train_bytes: u64,
    pub val_bytes: u64,
    pub train_shards: u32,
    pub val_shards: u32,
    /// Training draws thrown away because the same problem is in the validation set.
    pub leaked_draws_discarded: u64,
}

/// [`generate_with`] using 8 MiB shards.
pub fn generate(
    params: &ArithmeticParams,
    dest: &Path,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(JobProgress),
) -> DataResult<ArithmeticReport> {
    generate_with(params, dest, SHARD_BYTES, cancel, progress)
}

/// Write `dest/val/arithmetic/part-*.txt` (generated first, from its own seed) and `dest/train/arithmetic/part-*.txt`,
/// each file at most `shard_bytes` long. Same parameters, same files.
///
/// Progress counts problems. On cancellation or error unfinished shards are removed (finished ones stay).
pub fn generate_with(
    params: &ArithmeticParams,
    dest: &Path,
    shard_bytes: u64,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(JobProgress),
) -> DataResult<ArithmeticReport> {
    if params.train_problems == 0 || params.val_problems == 0 {
        return Err(DataError::invalid("the arithmetic lane needs at least one training and one validation problem"));
    }
    if !(0.0..=1.0).contains(&params.notes_frac) {
        return Err(DataError::invalid("notesFrac must be between 0 and 1"));
    }
    if shard_bytes < 1024 {
        return Err(DataError::invalid("shards must be at least 1 KiB"));
    }

    let total = f64::from(params.train_problems) + f64::from(params.val_problems);
    let mut throttle = Throttle::new(PROGRESS_INTERVAL);
    let mut meter = RateMeter::new(0.0);
    let mut emit = |done: f64, message: &str, force: bool| {
        if force || throttle.ready() {
            let rate = meter.update(done);
            progress(JobProgress {
                done,
                total: Some(total),
                unit: "problems".to_string(),
                message: message.to_string(),
                bytes_per_sec: None,
                eta_seconds: RateMeter::eta(rate, done, Some(total)),
            });
        }
    };

    // Validation first: its problems are what training must avoid.
    let val_dir = dest.join("val").join(LANE);
    let mut val_writer = ShardWriter::new(val_dir, shard_bytes).map_err(|e| DataError::io(dest, e))?;
    let mut rng = stream(params.seed, 1);
    let mut val_keys: HashSet<String> = HashSet::with_capacity(params.val_problems as usize);
    let mut written = 0u32;
    let result = (|| -> DataResult<()> {
        while written < params.val_problems {
            if written.is_multiple_of(1024) {
                cancel.check()?;
            }
            let line = sample_line(&mut rng, params.notes_frac);
            // The same problem can be drawn twice; keep the first so validation has no duplicates either.
            if !val_keys.insert(problem_key(&line).to_string()) {
                continue;
            }
            val_writer.write_line(&line).map_err(|e| DataError::io(dest, e))?;
            written += 1;
            if written.is_multiple_of(256) {
                emit(f64::from(written), "Writing validation problems", false);
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        val_writer.abandon();
        return Err(e);
    }
    let (val_bytes, val_shards) = val_writer.finish().map_err(|e| DataError::io(dest, e))?;

    let train_dir = dest.join("train").join(LANE);
    let mut train_writer = ShardWriter::new(train_dir, shard_bytes).map_err(|e| DataError::io(dest, e))?;
    let mut rng = stream(params.seed, 2);
    let (mut written, mut discarded, mut streak) = (0u32, 0u64, 0u32);
    let result = (|| -> DataResult<()> {
        while written < params.train_problems {
            if (u64::from(written) + discarded) % 1024 == 0 {
                cancel.check()?;
            }
            let line = sample_line(&mut rng, params.notes_frac);
            if val_keys.contains(problem_key(&line)) {
                discarded += 1;
                streak += 1;
                if streak > 1_000_000 {
                    return Err(DataError::invalid("could not draw new training problems that are not in validation"));
                }
                continue;
            }
            streak = 0;
            train_writer.write_line(&line).map_err(|e| DataError::io(dest, e))?;
            written += 1;
            if written % 256 == 0 {
                emit(f64::from(params.val_problems) + f64::from(written), "Writing training problems", false);
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        train_writer.abandon();
        return Err(e);
    }
    let (train_bytes, train_shards) = train_writer.finish().map_err(|e| DataError::io(dest, e))?;
    emit(total, "Arithmetic ready", true);

    Ok(ArithmeticReport {
        train_problems: u64::from(params.train_problems),
        val_problems: u64::from(params.val_problems),
        train_bytes,
        val_bytes,
        train_shards,
        val_shards,
        leaked_draws_discarded: discarded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use tempfile::TempDir;

    // ---------- an independent evaluator ----------

    /// What a line claims, parsed from its text alone.
    struct Parsed<'a> {
        task: Task,
        args: &'a str,
        /// The scratchpad body between `<think> ` and ` </think> `, when present.
        notes: Option<&'a str>,
        answer: &'a str,
    }

    fn parse(line: &str) -> Parsed<'_> {
        let (problem, rest) = line.split_once(" = ").unwrap_or_else(|| panic!("no ' = ' in {line:?}"));
        let (word, args) = problem.split_once(' ').unwrap();
        let task = Task::from_word(word).unwrap_or_else(|| panic!("unknown task in {line:?}"));
        let (notes, answer) = match rest.strip_prefix("<think> ") {
            Some(r) => {
                let (body, answer) =
                    r.split_once(" </think> ").unwrap_or_else(|| panic!("unterminated scratchpad: {line:?}"));
                (Some(body), answer)
            }
            None => (None, rest),
        };
        Parsed { task, args, notes, answer }
    }

    fn nums(s: &str, sep: &str) -> Vec<i128> {
        s.split(sep).map(|t| t.trim().parse().unwrap_or_else(|_| panic!("number expected in {s:?}"))).collect()
    }

    /// Recompute the answer from the arguments, with different code than the generator.
    fn evaluate(p: &Parsed) -> String {
        match p.task {
            Task::Add => {
                let v = nums(p.args, " + ");
                (v[0] + v[1]).to_string()
            }
            Task::Sub => {
                let v = nums(p.args, " - ");
                (v[0] - v[1]).to_string()
            }
            Task::Mul => {
                let v = nums(p.args, " * ");
                (v[0] * v[1]).to_string()
            }
            Task::Cmp => {
                let t: Vec<&str> = p.args.split(' ').collect();
                assert_eq!(t.len(), 3, "{:?}", p.args);
                let (a, b): (i128, i128) = (t[0].parse().unwrap(), t[2].parse().unwrap());
                let truth = match t[1] {
                    "<" => a < b,
                    ">" => a > b,
                    "==" => a == b,
                    other => panic!("operator {other}"),
                };
                (if truth { "yes" } else { "no" }).to_string()
            }
            Task::Mod => {
                let v = nums(p.args, " % ");
                v[0].rem_euclid(v[1]).to_string()
            }
            Task::Gcd => {
                let v = nums(p.args, " , ");
                num_gcd(v[0], v[1]).to_string()
            }
            Task::Sum => nums(p.args, " + ").iter().sum::<i128>().to_string(),
            Task::Round => {
                let (a, step) = p.args.split_once(" to ").unwrap();
                let (a, step): (i128, i128) = (a.parse().unwrap(), step.parse().unwrap());
                // Half up: add half a step and floor.
                (((a * 2 + step) / (2 * step)) * step).to_string()
            }
        }
    }

    fn num_gcd(a: i128, b: i128) -> i128 {
        if b == 0 { a } else { num_gcd(b, a % b) }
    }

    /// Check that the scratchpad's own arithmetic is right and leads to the answer.
    fn check_notes(p: &Parsed, line: &str) {
        let Some(body) = p.notes else { return };
        let steps: Vec<&str> = body.split(' ').collect();
        match p.task {
            Task::Add => {
                // "x+y+c=dcK" per digit, optionally a final "cK".
                let mut digits = String::new();
                let mut carry_out = 0;
                for s in &steps {
                    if let Some(c) = s.strip_prefix('c') {
                        assert_eq!(c.parse::<i64>().unwrap(), carry_out, "{line}");
                        digits.push_str(c);
                        continue;
                    }
                    let (lhs, rhs) = s.split_once('=').unwrap();
                    let t = nums(lhs, "+");
                    let (d, c) = rhs.split_once('c').unwrap();
                    let (d, c): (i128, i128) = (d.parse().unwrap(), c.parse().unwrap());
                    assert_eq!(t[0] + t[1] + t[2], d + 10 * c, "{line}");
                    digits.push_str(&d.to_string());
                    carry_out = c as i64;
                }
                let answer: String = digits.chars().rev().collect();
                assert_eq!(answer, p.answer, "{line}");
            }
            Task::Sub => {
                let mut digits = String::new();
                let mut negative = false;
                for s in &steps {
                    if *s == "sign" {
                        continue;
                    }
                    if *s == "-" {
                        negative = true;
                        continue;
                    }
                    let (lhs, rhs) = s.split_once('=').unwrap();
                    let t = nums(lhs, "-");
                    let (d, b) = rhs.split_once('b').unwrap();
                    let (d, b): (i128, i128) = (d.parse().unwrap(), b.parse().unwrap());
                    assert_eq!(t[0] - t[1] - t[2], d - 10 * b, "{line}");
                    digits.push_str(&d.to_string());
                }
                let magnitude: String = digits.chars().rev().collect();
                let magnitude: i128 = magnitude.parse().unwrap();
                let answer: i128 = p.answer.parse().unwrap();
                assert_eq!(answer.abs(), magnitude, "{line}");
                assert_eq!(answer < 0, negative, "{line}");
            }
            Task::Mul => {
                let v = nums(p.args, " * ");
                let (mut partials, mut last) = (Vec::new(), 0i128);
                for s in &steps {
                    let (lhs, rhs) = s.split_once('=').unwrap();
                    let rhs: i128 = rhs.parse().unwrap();
                    if let Some((a, b)) = lhs.split_once('*') {
                        // a*<digit><zeros>
                        assert_eq!(a.parse::<i128>().unwrap(), v[0], "{line}");
                        assert_eq!(b.parse::<i128>().unwrap() * v[0], rhs, "{line}");
                        partials.push(rhs);
                    } else {
                        let (x, y) = lhs.split_once('+').unwrap();
                        assert_eq!(x.parse::<i128>().unwrap() + y.parse::<i128>().unwrap(), rhs, "{line}");
                    }
                    last = rhs;
                }
                assert_eq!(last.to_string(), p.answer, "the last step is the product: {line}");
                assert_eq!(partials.iter().sum::<i128>(), v[0] * v[1], "{line}");
            }
            Task::Cmp => {
                let t: Vec<&str> = p.args.split(' ').collect();
                let (sa, sb) = (t[0], t[2]);
                let claimed = format!("len {} vs {}", sa.len(), sb.len());
                assert!(body.starts_with(&claimed), "{line}");
                if let Some(rest) = body.strip_prefix(&claimed).and_then(|r| r.strip_prefix(" digit ")) {
                    let parts: Vec<&str> = rest.split(' ').collect();
                    let k: usize = parts[0].parse().unwrap();
                    assert_eq!((&sa[..k], &sa[k..=k], &sb[k..=k]), (&sb[..k], parts[1], parts[3]), "{line}");
                    assert_ne!(parts[1], parts[3], "{line}");
                } else if body.ends_with("equal") {
                    assert_eq!(sa, sb, "{line}");
                } else {
                    assert_ne!(sa.len(), sb.len(), "{line}");
                }
            }
            Task::Mod => {
                let v = nums(p.args, " % ");
                let (a, b) = (v[0], v[1]);
                let q = a / b;
                assert_eq!(
                    body,
                    format!("{b}*{q}={} {b}*{}={}>{a} {a}-{}={}", b * q, q + 1, b * (q + 1), b * q, a % b),
                    "{line}"
                );
            }
            Task::Gcd => {
                let v = nums(p.args, " , ");
                let (mut x, mut y, mut expected) = (v[0], v[1], Vec::new());
                while y != 0 && expected.len() < 8 {
                    expected.push(format!("{x}%{y}={}", x % y));
                    (x, y) = (y, x % y);
                }
                assert_eq!(body, expected.join(" "), "{line}");
            }
            Task::Sum => {
                let v = nums(p.args, " + ");
                let mut run = 0;
                let expected: Vec<String> = v
                    .iter()
                    .map(|x| {
                        run += x;
                        run.to_string()
                    })
                    .collect();
                assert_eq!(body, expected.join(" "), "{line}");
            }
            Task::Round => {
                let (a, step) = p.args.split_once(" to ").unwrap();
                let (a, step): (i128, i128) = (a.parse().unwrap(), step.parse().unwrap());
                let (q, r) = (a / step, a % step);
                let up = r * 2 >= step;
                assert_eq!(
                    body,
                    format!(
                        "{a}={q}*{step}+{r} {r}{}{} -> {}",
                        if up { ">=" } else { "<" },
                        step / 2,
                        if up { "up" } else { "down" }
                    ),
                    "{line}"
                );
            }
        }
    }

    fn lines(seed: u64, n: usize, notes_frac: f64) -> Vec<String> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| sample_line(&mut rng, notes_frac)).collect()
    }

    fn read_shards(dir: &Path) -> Vec<(String, String)> {
        let mut shards: Vec<(String, String)> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap())
            .map(|e| (e.file_name().to_string_lossy().into_owned(), fs::read_to_string(e.path()).unwrap()))
            .collect();
        shards.sort();
        shards
    }

    fn all_lines(dir: &Path) -> Vec<String> {
        read_shards(dir)
            .into_iter()
            .flat_map(|(_, text)| text.lines().map(str::to_string).collect::<Vec<_>>())
            .collect()
    }

    // ---------- the tests ----------

    #[test]
    fn the_documented_example_has_the_documented_shape() {
        // add 4917 + 388 = <think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305
        let line = "add 4917 + 388 = <think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305";
        let p = parse(line);
        assert_eq!(evaluate(&p), "5305");
        check_notes(&p, line);
        assert_eq!(problem_key(line), "add 4917 + 388");
    }

    /// 800 lines produced by the original Python generator (`corpora/arithmetic.py`, notes fraction 0.5). The
    /// evaluator and scratchpad checker above know nothing about the Rust generator, so passing here shows that the
    /// formats the port writes are the formats the original writes.
    #[test]
    fn lines_from_the_original_python_generator_pass_the_same_checks() {
        let sample = include_str!("../../tests/data/python_arithmetic_sample.txt");
        let mut tasks = HashSet::new();
        let mut with_notes = 0;
        for line in sample.lines() {
            let p = parse(line);
            assert_eq!(evaluate(&p), p.answer, "{line}");
            check_notes(&p, line);
            tasks.insert(p.task);
            with_notes += usize::from(p.notes.is_some());
        }
        assert_eq!(tasks.len(), 8);
        assert!(with_notes > 300, "{with_notes} lines with scratchpads");
    }

    #[test]
    fn every_generated_answer_is_re_evaluated_correctly() {
        let all = lines(11, 60_000, 0.5);
        let mut seen: HashMap<Task, usize> = HashMap::new();
        for line in &all {
            let p = parse(line);
            assert_eq!(evaluate(&p), p.answer, "{line}");
            check_notes(&p, line);
            assert!(!line.contains('\n') && line.is_ascii());
            *seen.entry(p.task).or_default() += 1;
        }
        assert_eq!(seen.len(), 8, "every task shows up: {seen:?}");
    }

    #[test]
    fn edge_cases_are_covered_by_the_generator() {
        let all = lines(5, 120_000, 1.0);
        let has = |pred: &dyn Fn(&str) -> bool| all.iter().any(|l| pred(l));
        assert!(
            has(&|l| l.starts_with("sub ") && l.contains(" = <think>") && l.contains("sign -")),
            "negative subtraction"
        );
        assert!(has(&|l| l.starts_with("sub ") && l.split(" </think> ").nth(1).is_some_and(|a| a.starts_with('-'))));
        assert!(has(&|l| l.starts_with("cmp ") && l.contains(" == ") && l.ends_with("yes")), "equal comparison");
        assert!(
            has(&|l| l.starts_with("round ") && l.contains("-> up"))
                && has(&|l| l.starts_with("round ") && l.contains("-> down"))
        );
        assert!(has(&|l| l.starts_with("gcd ") && l.contains(" , 0 =")), "gcd with a zero");
        assert!(has(&|l| l.starts_with("add ") && l.contains(" c1 </think>")), "a final carry step");
    }

    #[test]
    fn rounding_is_half_up_not_half_to_even() {
        // Python's round() gives 20 for both 15 and 25; the corpus (and its scratchpad) says ties go up.
        assert_eq!((round_half_up(15, 10), round_half_up(25, 10)), (20, 30));
        assert_eq!((round_half_up(14, 10), round_half_up(24, 10)), (10, 20));
        assert_eq!((round_half_up(500, 1000), round_half_up(1500, 1000), round_half_up(499, 1000)), (1000, 2000, 0));
        assert_eq!(round_half_up(99, 100), 100);
        // The generator agrees with the helper on every line it writes.
        for line in lines(3, 20_000, 1.0).iter().filter(|l| l.starts_with("round ")) {
            let p = parse(line);
            assert_eq!(evaluate(&p), p.answer, "{line}");
        }
    }

    #[test]
    fn task_frequencies_match_the_weights() {
        for seed in [1u64, 2, 3] {
            let n = 100_000usize;
            let mut counts: HashMap<Task, f64> = HashMap::new();
            for line in lines(seed, n, 0.3) {
                *counts.entry(parse(&line).task).or_default() += 1.0;
            }
            let chi2: f64 = Task::WEIGHTS
                .iter()
                .map(|&(task, w, _)| {
                    let expected = n as f64 * w;
                    let observed = counts.get(&task).copied().unwrap_or(0.0);
                    (observed - expected).powi(2) / expected
                })
                .sum();
            // 7 degrees of freedom; 24.32 is the 0.1 % critical value.
            assert!(chi2 < 24.32, "seed {seed}: chi-square {chi2:.2}");
        }
    }

    #[test]
    fn notes_appear_in_about_notes_frac_of_the_lines() {
        let n = 50_000;
        for frac in [0.0, 0.3, 1.0] {
            let with = lines(9, n, frac).iter().filter(|l| l.contains("<think>")).count() as f64;
            let sigma = (n as f64 * frac * (1.0 - frac)).sqrt();
            assert!((with - n as f64 * frac).abs() <= 5.0 * sigma + 1.0, "frac {frac}: {with} of {n}");
        }
    }

    #[test]
    fn digit_counts_are_uniform_per_task() {
        // For `add` the first number has d digits with d uniform in 1..=8.
        let all = lines(21, 150_000, 0.0);
        let mut by_len = [0f64; 9];
        let mut n = 0f64;
        for l in all.iter().filter(|l| l.starts_with("add ")) {
            let a = l.split(' ').nth(1).unwrap();
            by_len[a.len()] += 1.0;
            n += 1.0;
        }
        let expected = n / 8.0;
        let chi2: f64 = by_len[1..].iter().map(|o| (o - expected).powi(2) / expected).sum();
        assert!(chi2 < 24.32, "chi-square {chi2:.2} over {by_len:?}");
        assert_eq!(by_len[0], 0.0);
    }

    #[test]
    fn answers_respect_the_documented_limits() {
        for line in lines(4, 100_000, 0.0) {
            let p = parse(&line);
            match p.task {
                Task::Mul => {
                    let v = nums(p.args, " * ");
                    assert!(v[0] < 10_000 && v[1] < 100, "{line}");
                }
                Task::Mod => {
                    let v = nums(p.args, " % ");
                    assert!((2..=99).contains(&v[1]) && v[0] < 1_000_000, "{line}");
                }
                Task::Gcd => assert!(nums(p.args, " , ").iter().all(|&x| x < 10_000), "{line}"),
                Task::Sum => {
                    let v = nums(p.args, " + ");
                    assert!((2..=5).contains(&v.len()) && v.iter().all(|&x| x < 10_000), "{line}");
                }
                Task::Round => assert!(["10", "100", "1000"].contains(&p.args.split(" to ").nth(1).unwrap()), "{line}"),
                Task::Add | Task::Sub | Task::Cmp => {}
            }
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_lines_and_other_seeds_differ() {
        assert_eq!(lines(7, 1000, 0.3), lines(7, 1000, 0.3));
        assert_ne!(lines(7, 1000, 0.3), lines(8, 1000, 0.3));
    }

    fn params(train: u32, val: u32, seed: u32) -> ArithmeticParams {
        ArithmeticParams { train_problems: train, val_problems: val, seed, notes_frac: 0.3 }
    }

    #[test]
    fn generate_is_deterministic_per_seed() {
        let (a, b, c) = (TempDir::new().unwrap(), TempDir::new().unwrap(), TempDir::new().unwrap());
        let cancel = CancelFlag::new();
        generate_with(&params(3000, 300, 5), a.path(), 4096, &cancel, &mut |_| {}).unwrap();
        generate_with(&params(3000, 300, 5), b.path(), 4096, &cancel, &mut |_| {}).unwrap();
        generate_with(&params(3000, 300, 6), c.path(), 4096, &cancel, &mut |_| {}).unwrap();
        for side in ["train", "val"] {
            let dir = |root: &TempDir| root.path().join(side).join(LANE);
            assert_eq!(read_shards(&dir(&a)), read_shards(&dir(&b)), "{side}");
            assert_ne!(read_shards(&dir(&a)), read_shards(&dir(&c)), "{side}");
        }
    }

    #[test]
    fn validation_problems_never_appear_in_training() {
        let dir = TempDir::new().unwrap();
        // Many draws from a small key space make collisions certain unless they are filtered out.
        let report =
            generate_with(&params(20_000, 20_000, 3), dir.path(), 64 * 1024, &CancelFlag::new(), &mut |_| {}).unwrap();
        let val = all_lines(&dir.path().join("val").join(LANE));
        let train = all_lines(&dir.path().join("train").join(LANE));
        assert_eq!((val.len(), train.len()), (20_000, 20_000));
        let val_keys: HashSet<&str> = val.iter().map(|l| problem_key(l)).collect();
        assert_eq!(val_keys.len(), val.len(), "validation has no repeated problem either");
        let leaked: Vec<&String> = train.iter().filter(|l| val_keys.contains(problem_key(l))).collect();
        assert!(leaked.is_empty(), "{} leaked, e.g. {:?}", leaked.len(), leaked.first());
        assert!(report.leaked_draws_discarded > 0, "the filter had something to do");
        // Every line in both files is still correct.
        for line in val.iter().chain(&train) {
            let p = parse(line);
            assert_eq!(evaluate(&p), p.answer, "{line}");
        }
    }

    #[test]
    fn without_the_filter_leakage_would_occur() {
        // The control for the test above: the same draws without exclusion do share problems.
        let mut val = ChaCha8Rng::seed_from_u64(1);
        let mut train = ChaCha8Rng::seed_from_u64(2);
        let val_keys: HashSet<String> =
            (0..20_000).map(|_| problem_key(&sample_line(&mut val, 0.3)).to_string()).collect();
        let leaks = (0..20_000).filter(|_| val_keys.contains(problem_key(&sample_line(&mut train, 0.3)))).count();
        assert!(leaks > 0);
    }

    #[test]
    fn shards_respect_the_size_limit_and_are_sequentially_named() {
        let dir = TempDir::new().unwrap();
        let limit = 10_000u64;
        let report = generate_with(&params(5000, 500, 1), dir.path(), limit, &CancelFlag::new(), &mut |_| {}).unwrap();
        for side in ["train", "val"] {
            let shards = read_shards(&dir.path().join(side).join(LANE));
            assert!(shards.len() > 1);
            for (i, (name, text)) in shards.iter().enumerate() {
                assert_eq!(name, &format!("part-{i:04}.txt"));
                assert!(text.len() as u64 <= limit, "{name} is {} bytes", text.len());
                assert!(text.ends_with('\n'));
            }
            // Shards are filled, not left half empty: every shard but the last is within one line of the limit.
            for (_, text) in &shards[..shards.len() - 1] {
                assert!(text.len() as u64 + 400 > limit);
            }
        }
        assert_eq!(report.train_shards as usize, read_shards(&dir.path().join("train").join(LANE)).len());
        let total: usize = read_shards(&dir.path().join("train").join(LANE)).iter().map(|(_, t)| t.len()).sum();
        assert_eq!(report.train_bytes as usize, total);
        assert!(
            fs::read_dir(dir.path().join("val").join(LANE)).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".part"))
        );
    }

    #[test]
    fn default_shards_are_eight_mebibytes() {
        assert_eq!(SHARD_BYTES, 8 * 1024 * 1024);
        let dir = TempDir::new().unwrap();
        let report = generate(&params(100, 10, 1), dir.path(), &CancelFlag::new(), &mut |_| {}).unwrap();
        assert_eq!((report.train_shards, report.val_shards), (1, 1));
        assert!(dir.path().join("train/arithmetic/part-0000.txt").is_file());
        assert!(dir.path().join("val/arithmetic/part-0000.txt").is_file());
    }

    #[test]
    fn regenerating_replaces_the_old_shards() {
        let dir = TempDir::new().unwrap();
        generate_with(&params(5000, 500, 1), dir.path(), 8192, &CancelFlag::new(), &mut |_| {}).unwrap();
        let many = read_shards(&dir.path().join("train").join(LANE)).len();
        generate_with(&params(100, 50, 1), dir.path(), 8192, &CancelFlag::new(), &mut |_| {}).unwrap();
        let few = read_shards(&dir.path().join("train").join(LANE)).len();
        assert!(many > few && few == 1, "{many} then {few}");
    }

    #[test]
    fn progress_counts_problems_and_cancellation_cleans_up() {
        let dir = TempDir::new().unwrap();
        let mut seen: Vec<JobProgress> = Vec::new();
        generate_with(&params(2000, 200, 1), dir.path(), 8192, &CancelFlag::new(), &mut |p| seen.push(p)).unwrap();
        let last = seen.last().unwrap();
        assert_eq!((last.unit.as_str(), last.done, last.total), ("problems", 2200.0, Some(2200.0)));

        let cancel = CancelFlag::new();
        let dir2 = TempDir::new().unwrap();
        let result = generate_with(&params(1_000_000, 200, 1), dir2.path(), 8192, &cancel, &mut |p| {
            if p.message.starts_with("Writing training") {
                cancel.cancel();
            }
        });
        assert!(matches!(result, Err(DataError::Cancelled)));
        let stray: Vec<_> =
            fs::read_dir(dir2.path().join("train").join(LANE)).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert!(stray.iter().all(|n| !n.to_string_lossy().ends_with(".part")), "{stray:?}");
    }

    #[test]
    fn bad_parameters_are_rejected() {
        let dir = TempDir::new().unwrap();
        let cancel = CancelFlag::new();
        for bad in [
            params(0, 10, 1),
            params(10, 0, 1),
            ArithmeticParams { notes_frac: 1.5, ..params(10, 10, 1) },
            ArithmeticParams { notes_frac: -0.1, ..params(10, 10, 1) },
        ] {
            assert!(matches!(generate(&bad, dir.path(), &cancel, &mut |_| {}), Err(DataError::Invalid(_))), "{bad:?}");
        }
        assert!(matches!(
            generate_with(&params(10, 10, 1), dir.path(), 10, &cancel, &mut |_| {}),
            Err(DataError::Invalid(_))
        ));
    }

    #[test]
    fn problem_key_drops_scratchpad_and_answer() {
        assert_eq!(problem_key("cmp 12 == 12 = <think> len 2 vs 2 equal </think> yes"), "cmp 12 == 12");
        assert_eq!(problem_key("gcd 12 , 18 = 6"), "gcd 12 , 18");
        assert_eq!(problem_key("no separator"), "no separator");
    }
}
