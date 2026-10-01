//! Writing text from a trained model: decoding rules ([`decode`]), the character-by-character writer ([`writer`]),
//! learning from a conversation ([`live`]) and the generator the app talks to ([`generator`]).

pub mod decode;
pub mod generator;
pub mod live;
pub mod writer;

pub use decode::{Decode, adjust, pick_next, repeat_rate};
pub use generator::ModelGenerator;
pub use live::LiveLearner;
pub use writer::{Writer, Written};
