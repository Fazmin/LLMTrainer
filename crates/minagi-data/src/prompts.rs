//! Suggested sampling prompts for each lane of a prepared dataset.
//!
//! The well-known lanes get a hand-picked prompt. Any other lane gets the opening of a passage taken from its
//! held-out text, so the model is asked to continue something it has never trained on, in the lane's own style.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use minagi_types::LaneInfo;
use walkdir::WalkDir;

use crate::scan::is_ignored;

/// Prompts for the lanes the app knows.
pub const STORIES_PROMPT: &str = "Once upon a time, there was a little boy named Tom. One day he ";
pub const ARITHMETIC_PROMPT: &str = "add 4917 + 388 = ";
pub const CODE_PROMPT: &str = "def merge_sorted(a, b):\n    ";
pub const CHAT_PROMPT: &str = "<user>\nWhat are you?\n</user>\n<bot>\n";

/// How long a prompt made from a passage is, in characters (it may be a little shorter to end at a word boundary).
pub const PASSAGE_PROMPT_CHARS: usize = 60;
/// How much of a file is read to find the start of a passage.
const WINDOW_BYTES: u64 = 8 * 1024;

/// The prompt a well-known lane always gets.
pub fn default_prompt(lane: &str) -> Option<&'static str> {
    match lane {
        "stories" => Some(STORIES_PROMPT),
        "arithmetic" => Some(ARITHMETIC_PROMPT),
        "code" => Some(CODE_PROMPT),
        "chat" => Some(CHAT_PROMPT),
        _ => None,
    }
}

/// `(lane, prompt)` for every lane of the dataset at `root` (`root/train/<lane>` and `root/val/<lane>`), sorted by
/// lane name. Lanes with no text at all are left out. The result only depends on the files, so asking twice gives the
/// same prompts.
pub fn suggest_prompts(root: &Path) -> Vec<(String, String)> {
    let mut lanes: Vec<String> = ["train", "val"]
        .iter()
        .filter_map(|side| std::fs::read_dir(root.join(side)).ok())
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with('.'))
        .collect();
    lanes.sort();
    lanes.dedup();

    lanes
        .into_iter()
        .filter_map(|lane| {
            if let Some(prompt) = default_prompt(&lane) {
                return Some((lane, prompt.to_string()));
            }
            // Held-out text first; a lane with no validation text falls back to its training text.
            let prompt = [root.join("val").join(&lane), root.join("train").join(&lane)]
                .iter()
                .find_map(|dir| passage_prompt(dir, &lane))?;
            Some((lane, prompt))
        })
        .collect()
}

/// Fill in `sample_prompt` of each lane table row from [`suggest_prompts`] on the dataset at `root`.
pub fn fill_sample_prompts(lanes: &mut [LaneInfo], root: &Path) {
    let prompts = suggest_prompts(root);
    for lane in lanes {
        lane.sample_prompt = prompts.iter().find(|(name, _)| *name == lane.name).map(|(_, p)| p.clone());
    }
}

/// About [`PASSAGE_PROMPT_CHARS`] characters from the start of a passage of the lane's files in `dir`, picked by a
/// hash of the lane name so the choice is repeatable.
fn passage_prompt(dir: &Path, lane: &str) -> Option<String> {
    let mut files: Vec<(PathBuf, u64)> = WalkDir::new(dir)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| !is_ignored(e))
        .flatten()
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok().map(|m| (e.into_path(), m.len())))
        .filter(|(_, len)| *len > 0)
        .collect();
    if files.is_empty() {
        return None;
    }
    let hash = blake3::hash(format!("prompt:{lane}").as_bytes());
    let word = |i: usize| u64::from_le_bytes(hash.as_bytes()[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
    // Try the chosen file first, then the others in order, in case it holds nothing usable (all whitespace, say).
    let first = (word(0) % files.len() as u64) as usize;
    files.rotate_left(first);
    files.iter().find_map(|(path, len)| {
        let offset = if *len > WINDOW_BYTES { word(1) % (*len - WINDOW_BYTES + 1) } else { 0 };
        let window = read_window(path, offset).ok()?;
        prompt_from_window(&window, offset == 0)
    })
}

fn read_window(path: &Path, offset: u64) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::with_capacity(WINDOW_BYTES as usize);
    file.take(WINDOW_BYTES).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Turn the bytes of a window into a prompt. When the window starts mid-file its first (partial) line is dropped so
/// the passage begins at the start of a line.
pub(crate) fn prompt_from_window(window: &[u8], at_file_start: bool) -> Option<String> {
    let mut bytes = window;
    if at_file_start {
        bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(bytes);
    } else {
        let newline = bytes.iter().position(|&b| b == b'\n')?;
        bytes = &bytes[newline + 1..];
    }
    // Keep the valid UTF-8 prefix; a window can end in the middle of a character.
    let text = match std::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => std::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap_or(""),
    };
    // Lines that only carry a separator say nothing about the lane's style.
    let flat: String = text
        .lines()
        .filter(|l| l.trim() != "<|endoftext|>")
        .flat_map(|l| l.split_whitespace())
        .take(PASSAGE_PROMPT_CHARS)
        .collect::<Vec<_>>()
        .join(" ");
    cut_at_word_boundary(&flat, PASSAGE_PROMPT_CHARS)
}

/// The first `max_chars` characters of `text`, shortened to end after a whole word, with one trailing space (so the
/// model starts a new word). Text with no spaces (Chinese, say) is cut at exactly `max_chars` characters.
fn cut_at_word_boundary(text: &str, max_chars: usize) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut chars = text.char_indices();
    let end = match chars.nth(max_chars) {
        // `end` is where character number `max_chars` starts; if text goes on there, we are inside or at a word.
        Some((end, next)) => {
            let head = &text[..end];
            if next.is_whitespace() {
                head
            } else {
                match head.rfind(char::is_whitespace) {
                    Some(space) => &head[..space],
                    None => head,
                }
            }
        }
        None => text,
    };
    let end = end.trim_end();
    (!end.is_empty()).then(|| format!("{end} "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    fn prompts(root: &Path) -> Vec<(String, String)> {
        suggest_prompts(root)
    }

    #[test]
    fn known_lanes_get_their_fixed_prompts() {
        let dir = TempDir::new().unwrap();
        for lane in ["stories", "arithmetic", "code", "chat"] {
            write(dir.path(), &format!("train/{lane}/a.txt"), "irrelevant text for this lane\n");
        }
        let got = prompts(dir.path());
        let get = |lane: &str| got.iter().find(|(l, _)| l == lane).map(|(_, p)| p.as_str());
        assert_eq!(get("stories"), Some("Once upon a time, there was a little boy named Tom. One day he "));
        assert_eq!(get("arithmetic"), Some("add 4917 + 388 = "));
        assert_eq!(get("code"), Some("def merge_sorted(a, b):\n    "));
        assert_eq!(get("chat"), Some("<user>\nWhat are you?\n</user>\n<bot>\n"));
        assert_eq!(got.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(), ["arithmetic", "chat", "code", "stories"]);
    }

    #[test]
    fn other_lanes_use_a_passage_from_held_out_text() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/poems/a.txt", &"training words only appear in training files\n".repeat(50));
        write(
            dir.path(),
            "val/poems/v.txt",
            "The river keeps its secrets under a quiet sheet of ice, and the heron waits there patiently for spring.\nSecond line.\n",
        );
        let got = prompts(dir.path());
        assert_eq!(got.len(), 1);
        let (lane, prompt) = &got[0];
        assert_eq!(lane, "poems");
        assert_eq!(prompt, "The river keeps its secrets under a quiet sheet of ice, and ");
        assert!(prompt.chars().count() <= PASSAGE_PROMPT_CHARS + 1);
        assert!(!prompt.contains("training"));
    }

    #[test]
    fn lane_tables_can_be_filled_with_prompts() {
        use minagi_types::Count;
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/stories/a.txt", "x\n");
        write(
            dir.path(),
            "train/notes/a.txt",
            "Remember to water the plants every other morning, and never on Sundays at all.\n",
        );
        let row = |name: &str| LaneInfo {
            name: name.into(),
            display_name: name.into(),
            color_slot: 0,
            enabled: true,
            n_files: Count(1),
            train_bytes: Count(1),
            val_files: Count(0),
            val_bytes: Count(0),
            sample_prompt: None,
        };
        let mut lanes = vec![row("notes"), row("stories"), row("ghost")];
        fill_sample_prompts(&mut lanes, dir.path());
        assert_eq!(lanes[1].sample_prompt.as_deref(), Some(STORIES_PROMPT));
        assert!(lanes[0].sample_prompt.as_deref().unwrap().starts_with("Remember to water"));
        assert_eq!(lanes[2].sample_prompt, None);
    }

    #[test]
    fn a_lane_without_validation_text_falls_back_to_training_text() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "train/notes/a.txt",
            "Remember to water the plants every other morning, and never on Sundays at all.\n",
        );
        let got = prompts(dir.path());
        assert_eq!(got[0].1, "Remember to water the plants every other morning, and never ");
    }

    #[test]
    fn prompts_are_deterministic_and_differ_between_lanes() {
        let dir = TempDir::new().unwrap();
        for lane in ["alpha", "beta"] {
            let text: String =
                (0..400).map(|i| format!("{lane} sentence number {i} says something rather different.\n")).collect();
            write(dir.path(), &format!("val/{lane}/a.txt"), &text);
        }
        let a = prompts(dir.path());
        assert_eq!(a, prompts(dir.path()));
        assert_ne!(a[0].1, a[1].1);
        assert!(a[0].1.starts_with("alpha sentence number") && a[1].1.starts_with("beta sentence number"), "{a:?}");
        // Cut at a word boundary: one trailing space, nothing longer than the target.
        assert!(a[0].1.ends_with(' ') && !a[0].1.ends_with("  ") && a[0].1.chars().count() <= PASSAGE_PROMPT_CHARS + 1);
    }

    #[test]
    fn passages_end_at_a_word_boundary_and_are_valid_utf8() {
        let long_word = "x".repeat(200);
        assert_eq!(cut_at_word_boundary(&format!("short {long_word}"), 60).as_deref(), Some("short "));
        assert_eq!(cut_at_word_boundary(&long_word, 60).unwrap(), format!("{} ", "x".repeat(60)));
        assert_eq!(cut_at_word_boundary("just a few words", 60).as_deref(), Some("just a few words "));
        let exact = format!("{} tail", "a".repeat(59));
        assert_eq!(cut_at_word_boundary(&exact, 60).unwrap(), format!("{} ", "a".repeat(59)));
        let cjk = "日本語".repeat(40);
        let p = cut_at_word_boundary(&cjk, 60).unwrap();
        assert_eq!(p.chars().count(), 61, "60 characters plus the trailing space");
        assert_eq!(cut_at_word_boundary("   \n  ", 60), None);
    }

    #[test]
    fn windows_that_start_mid_character_or_hold_only_separators_are_handled() {
        // A window cut in the middle of a multi-byte character at its end.
        let mut bytes = "Les élèves écoutent la maîtresse raconter une histoire très longue.\n".as_bytes().to_vec();
        bytes.extend_from_slice(&"é".as_bytes()[..1]);
        assert_eq!(
            prompt_from_window(&bytes, true).as_deref(),
            Some("Les élèves écoutent la maîtresse raconter une histoire très ")
        );
        // Separator lines and a leading BOM are ignored.
        let mut with_bom = vec![0xEF, 0xBB, 0xBF];
        with_bom.extend_from_slice(
            b"<|endoftext|>\nA story begins here and goes on for a little while longer than sixty characters.\n",
        );
        assert_eq!(
            prompt_from_window(&with_bom, true).as_deref(),
            Some("A story begins here and goes on for a little while longer ")
        );
        // Mid-file windows drop their first (partial) line.
        assert_eq!(
            prompt_from_window(b"tial line\nFresh start of the next line\n", false).as_deref(),
            Some("Fresh start of the next line ")
        );
        assert_eq!(prompt_from_window(b"no newline at all", false), None);
        assert_eq!(prompt_from_window(b"<|endoftext|>\n<|endoftext|>\n", true), None);
    }

    #[test]
    fn empty_and_hidden_lanes_are_left_out() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("train/empty")).unwrap();
        write(dir.path(), "train/blank/a.txt", "   \n\n");
        write(dir.path(), "train/.hidden/a.txt", "should never show up in the list of lanes at all\n");
        write(dir.path(), "train/real/a.txt", "A real lane with a few real words in it, enough for a prompt.\n");
        let got = prompts(dir.path());
        assert_eq!(got.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(), ["real"]);
        assert!(prompts(&dir.path().join("nothing-here")).is_empty());
    }
}
