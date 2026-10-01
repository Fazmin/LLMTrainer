//! Counter newtypes and unit conversions.
//!
//! `specta-typescript` refuses `u64`/`i64` (they do not fit a JS number). Every counter in this app stays far
//! below 2^53, so the newtypes below are exported to TypeScript as `number`.

use serde::{Deserialize, Serialize};
use specta::Type;
use specta_typescript::Number;

macro_rules! counter {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Type)]
        #[serde(transparent)]
        pub struct $name(#[specta(type = Number)] pub u64);

        impl From<u64> for $name {
            fn from(v: u64) -> Self {
                Self(v)
            }
        }
        impl From<$name> for u64 {
            fn from(v: $name) -> u64 {
                v.0
            }
        }
    };
}

counter!(
    /// Optimizer step counter.
    Step
);
counter!(
    /// Count of characters (bytes) read.
    Chars
);
counter!(
    /// Unix time in milliseconds.
    UnixMs
);
counter!(
    /// A non-negative size or count (bytes, rows).
    Count
);

impl UnixMs {
    pub fn now() -> Self {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self(ms)
    }
}

/// Natural-log loss (nats per character) to bits per character.
pub fn nats_to_bits(nats: f64) -> f64 {
    nats / std::f64::consts::LN_2
}

/// Bits per character back to nats.
pub fn bits_to_nats(bits: f64) -> f64 {
    bits * std::f64::consts::LN_2
}

/// Perplexity (effective number of equally likely next characters) for a loss in nats.
pub fn perplexity(nats: f64) -> f64 {
    nats.exp()
}

/// Loss of a model that guesses uniformly over the byte vocabulary (265 symbols), in nats.
pub const UNIFORM_NATS: f64 = 5.579_729_825_986_222; // ln(265)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_round_trip() {
        let n = 1.234;
        assert!((bits_to_nats(nats_to_bits(n)) - n).abs() < 1e-12);
        assert!((perplexity(0.0) - 1.0).abs() < 1e-12);
        assert!((UNIFORM_NATS - 265f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn counters_serialize_as_plain_numbers() {
        let s = serde_json::to_string(&Step(42)).unwrap();
        assert_eq!(s, "42");
        let c: Chars = serde_json::from_str("123456789012").unwrap();
        assert_eq!(c.0, 123_456_789_012);
    }
}
