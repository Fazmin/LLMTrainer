//! The byte tokenizer: ids 0..=255 are raw bytes, 256..=264 are the nine marker tokens.
//!
//! Marker strings in the text become one token each; everything else is its UTF-8 bytes, so any file can be read
//! without fitting a vocabulary, and invalid UTF-8 never raises (it round-trips as bytes).

/// The marker tokens, in id order starting at [`MARKER_BASE`].
pub const SPECIALS: [&str; 9] =
    ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>", "<g>", "</g>", "<|endoftext|>"];
pub const MARKER_BASE: u16 = 256;
pub const VOCAB_SIZE: usize = 265;

#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Id of a marker string, if it is one.
    pub fn marker_id(marker: &str) -> Option<u16> {
        SPECIALS.iter().position(|m| *m == marker).map(|i| MARKER_BASE + i as u16)
    }

    /// The marker that starts at `bytes[0]`, as `(id, length)`.
    fn marker_at(bytes: &[u8]) -> Option<(u16, usize)> {
        if bytes.first() != Some(&b'<') {
            return None;
        }
        SPECIALS
            .iter()
            .enumerate()
            .find(|(_, m)| bytes.starts_with(m.as_bytes()))
            .map(|(i, m)| (MARKER_BASE + i as u16, m.len()))
    }

    /// Tokenize raw bytes (a file as it is on disk).
    pub fn encode(&self, bytes: &[u8]) -> Vec<u16> {
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            match Self::marker_at(&bytes[i..]) {
                Some((id, len)) => {
                    out.push(id);
                    i += len;
                }
                None => {
                    out.push(bytes[i] as u16);
                    i += 1;
                }
            }
        }
        out
    }

    pub fn encode_str(&self, text: &str) -> Vec<u16> {
        self.encode(text.as_bytes())
    }

    /// Decode a whole sequence. Invalid UTF-8 becomes U+FFFD; markers are written out as their literal text.
    pub fn decode(&self, ids: &[u16]) -> String {
        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        for &id in ids {
            out.push_str(&stream.push(id));
        }
        out.push_str(&stream.finish());
        out
    }
}

/// Decodes one token at a time for streaming output, holding back the bytes of a character that is not complete yet
/// (so a multi-byte character never appears as replacement characters in the middle).
#[derive(Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a token; returns the text it completes (possibly empty).
    pub fn push(&mut self, id: u16) -> String {
        if id >= MARKER_BASE {
            let mut out = self.flush_lossy();
            if let Some(m) = SPECIALS.get((id - MARKER_BASE) as usize) {
                out.push_str(m);
            }
            return out;
        }
        self.pending.push(id as u8);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        // Safe: the first `valid` bytes are valid UTF-8.
                        out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap_or(""));
                        self.pending.drain(..valid);
                    }
                    match e.error_len() {
                        // An invalid sequence: replace it and carry on with what follows.
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..n);
                            if self.pending.is_empty() {
                                break;
                            }
                        }
                        // Incomplete at the end: wait for more bytes.
                        None => break,
                    }
                }
            }
        }
        out
    }

    /// Anything still held back, as replacement characters (call when generation ends).
    pub fn finish(&mut self) -> String {
        self.flush_lossy()
    }

    fn flush_lossy(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_its_bytes() {
        let t = ByteTokenizer::new();
        assert_eq!(t.encode_str("Hi!"), vec![72, 105, 33]);
        assert_eq!(t.decode(&t.encode_str("Hello, world")), "Hello, world");
    }

    #[test]
    fn markers_become_single_tokens_and_round_trip() {
        let t = ByteTokenizer::new();
        let text = "<user>\nhi\n</user>\n<bot>\nhello\n</bot>\n<|endoftext|>";
        let ids = t.encode_str(text);
        assert!(ids.contains(&ByteTokenizer::marker_id("<user>").unwrap()));
        assert_eq!(ids.iter().filter(|&&i| i >= MARKER_BASE).count(), 5);
        assert_eq!(t.decode(&ids), text);
        assert_eq!(ByteTokenizer::marker_id("<think>"), Some(256));
        assert_eq!(ByteTokenizer::marker_id("<|endoftext|>"), Some(264));
        assert_eq!(ByteTokenizer::marker_id("<nothing>"), None);
    }

    #[test]
    fn lookalikes_are_not_markers() {
        let t = ByteTokenizer::new();
        // Chess headers like "<g 1850 1-0>" and unfinished tags are just bytes.
        for s in ["<g 1850 1-0>", "<use", "< user>", "a<b>c", "<<user>"] {
            let ids = t.encode_str(s);
            assert_eq!(t.decode(&ids), s);
        }
        assert_eq!(t.encode_str("<g 1850>").iter().filter(|&&i| i >= MARKER_BASE).count(), 0);
        // But "<<user>" contains a real marker after the first byte.
        assert_eq!(t.encode_str("<<user>").iter().filter(|&&i| i >= MARKER_BASE).count(), 1);
    }

    #[test]
    fn invalid_utf8_never_panics_and_round_trips_as_replacements() {
        let t = ByteTokenizer::new();
        let ids = t.encode(&[b'a', 0xFF, b'b', 0xC3, 0x28, b'c']);
        assert_eq!(ids.len(), 6, "every byte is a token");
        let s = t.decode(&ids);
        assert!(s.starts_with('a') && s.contains('b') && s.ends_with('c'));
        assert!(s.contains('\u{FFFD}'));
    }

    #[test]
    fn streaming_holds_back_partial_characters() {
        let t = ByteTokenizer::new();
        let mut s = Utf8Stream::new();
        let ids = t.encode_str("héllo €!");
        let mut got = String::new();
        let mut empties = 0;
        for id in ids {
            let piece = s.push(id);
            if piece.is_empty() {
                empties += 1;
            }
            assert!(!piece.contains('\u{FFFD}'), "no replacement characters mid-stream: {piece:?}");
            got.push_str(&piece);
        }
        got.push_str(&s.finish());
        assert_eq!(got, "héllo €!");
        assert_eq!(empties, 3, "é (2 bytes) and € (3 bytes) each hold back their lead bytes");
    }

    #[test]
    fn streaming_emits_markers_and_flushes_before_them() {
        let mut s = Utf8Stream::new();
        assert_eq!(s.push(0xE2), "");
        // A marker arrives while a character is incomplete: the broken bytes are flushed first.
        let out = s.push(ByteTokenizer::marker_id("</bot>").unwrap());
        assert_eq!(out, "\u{FFFD}</bot>");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn encode_handles_big_inputs() {
        let t = ByteTokenizer::new();
        let text = "<user>x</user>".repeat(20_000);
        assert_eq!(t.encode_str(&text).len(), 60_000);
    }
}
