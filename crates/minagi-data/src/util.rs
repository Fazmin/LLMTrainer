//! Small shared helpers: a cancellation flag, a progress throttle and a transfer-rate meter.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::error::{DataError, DataResult};

/// Cooperative cancellation for the synchronous jobs (scan, materialize, generate). Cloning shares the flag.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask every holder of this flag to stop as soon as convenient.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// `Err(Cancelled)` once cancelled, so jobs can write `cancel.check()?`.
    pub fn check(&self) -> DataResult<()> {
        if self.is_cancelled() { Err(DataError::Cancelled) } else { Ok(()) }
    }
}

/// Rate limiter for progress callbacks: [`Throttle::ready`] is true at most once per interval.
#[derive(Debug)]
pub(crate) struct Throttle {
    interval: Duration,
    last: Option<Instant>,
}

/// Progress callbacks fire at most this often.
pub(crate) const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

impl Throttle {
    pub fn new(interval: Duration) -> Self {
        Self { interval, last: None }
    }

    pub fn ready(&mut self) -> bool {
        let now = Instant::now();
        match self.last {
            Some(prev) if now.duration_since(prev) < self.interval => false,
            _ => {
                self.last = Some(now);
                true
            }
        }
    }
}

/// Smoothed transfer rate and ETA from a stream of "bytes done so far" samples.
#[derive(Debug)]
pub(crate) struct RateMeter {
    last_at: Instant,
    last_done: f64,
    rate: Option<f64>,
}

impl RateMeter {
    /// `done` is the amount already complete when the meter starts (a resumed job).
    pub fn new(done: f64) -> Self {
        let now = Instant::now();
        Self { last_at: now, last_done: done, rate: None }
    }

    /// Record the current total and return the smoothed bytes-per-second, if known yet.
    pub fn update(&mut self, done: f64) -> Option<f64> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_at).as_secs_f64();
        if dt >= 0.05 {
            let inst = (done - self.last_done) / dt;
            // Exponential moving average with a ~3 s time constant.
            let alpha = 1.0 - (-dt / 3.0).exp();
            self.rate = Some(match self.rate {
                Some(r) => r + alpha * (inst - r),
                None => inst,
            });
            self.last_at = now;
            self.last_done = done;
        }
        self.rate.filter(|r| r.is_finite() && *r > 0.0)
    }

    /// Seconds until `total` at the current rate.
    pub fn eta(rate: Option<f64>, done: f64, total: Option<f64>) -> Option<f64> {
        let (rate, total) = (rate?, total?);
        Some(((total - done) / rate).max(0.0))
    }
}

/// Lower-case hex of a byte slice.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Path components joined with `/`, whatever the platform separator is.
pub(crate) fn rel_to_string(rel: &std::path::Path) -> String {
    rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")
}

/// Nanoseconds since the Unix epoch of a file's modification time (0 when unavailable).
pub(crate) fn mtime_ns(meta: &std::fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_flag_is_shared() {
        let a = CancelFlag::new();
        let b = a.clone();
        assert!(a.check().is_ok());
        b.cancel();
        assert!(a.is_cancelled());
        assert!(matches!(a.check(), Err(DataError::Cancelled)));
    }

    #[test]
    fn throttle_limits_rate() {
        let mut t = Throttle::new(Duration::from_secs(60));
        assert!(t.ready());
        assert!(!t.ready());
        let mut open = Throttle::new(Duration::ZERO);
        assert!(open.ready() && open.ready());
    }

    #[test]
    fn hex_encodes() {
        assert_eq!(hex(&[0x00, 0xab, 0xff]), "00abff");
    }

    #[test]
    fn eta_needs_rate_and_total() {
        assert_eq!(RateMeter::eta(Some(10.0), 50.0, Some(100.0)), Some(5.0));
        assert_eq!(RateMeter::eta(None, 50.0, Some(100.0)), None);
        assert_eq!(RateMeter::eta(Some(10.0), 50.0, None), None);
    }
}
