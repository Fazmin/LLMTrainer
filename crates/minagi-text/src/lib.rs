//! Text handling for the engine, with no ML dependencies so it is fast to build and easy to test:
//!
//! - [`tokenizer`]: raw bytes plus nine marker tokens, and incremental decoding for streaming output.
//! - [`stream`]: the training reader. Lanes are read round-robin; each visit takes a random passage of a random file,
//!   and every step yields the window of text ending at the next chunk.
//! - [`eval`]: sequential held-out chunks per domain.

pub mod error;
pub mod eval;
pub mod stream;
pub mod tokenizer;

pub use error::{TextError, TextResult};
pub use eval::{EvalChunk, EvalSet};
pub use stream::{Batch, StreamConfig, TrainStream};
pub use tokenizer::{ByteTokenizer, MARKER_BASE, SPECIALS, Utf8Stream, VOCAB_SIZE};
