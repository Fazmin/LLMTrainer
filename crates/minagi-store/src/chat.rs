//! Chat sessions and messages.

use minagi_types::{ChatMessage, ChatMode, ChatRole, UnixMs};
use rusqlite::{OptionalExtension, Row, params};

use crate::db::{Store, StoreError, StoreResult};

/// A chat session as stored. The app adds what only the loaded model knows (memory needed, whether it can learn).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatSessionRow {
    pub id: i64,
    pub title: String,
    pub run_id: Option<i64>,
    pub checkpoint_id: Option<i64>,
    pub model_label: String,
    pub mode: ChatMode,
    pub learn_enabled: bool,
    pub adapted_ckpt_rel: Option<String>,
    pub created_at: UnixMs,
}

#[derive(Debug, Clone)]
pub struct NewChat {
    pub title: String,
    pub run_id: Option<i64>,
    pub checkpoint_id: Option<i64>,
    pub model_label: String,
    pub mode: ChatMode,
}

const SESSION_COLUMNS: &str =
    "id, title, run_id, checkpoint_id, model_label, mode, learn_enabled, adapted_ckpt_rel, created_at";

fn session_from_row(r: &Row<'_>) -> rusqlite::Result<ChatSessionRow> {
    let mode: String = r.get(5)?;
    Ok(ChatSessionRow {
        id: r.get(0)?,
        title: r.get(1)?,
        run_id: r.get(2)?,
        checkpoint_id: r.get(3)?,
        model_label: r.get(4)?,
        mode: ChatMode::parse(&mode).unwrap_or(ChatMode::Continue),
        learn_enabled: r.get::<_, i64>(6)? != 0,
        adapted_ckpt_rel: r.get(7)?,
        created_at: UnixMs(r.get::<_, i64>(8)? as u64),
    })
}

impl Store {
    pub fn create_chat_session(&self, new: NewChat) -> StoreResult<ChatSessionRow> {
        self.write(move |conn| {
            let now = UnixMs::now().0 as i64;
            conn.execute(
                "INSERT INTO chat_sessions (title, run_id, checkpoint_id, model_label, mode, params_json, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, '{}', ?6, ?6)",
                params![new.title, new.run_id, new.checkpoint_id, new.model_label, new.mode.as_str(), now],
            )?;
            let id = conn.last_insert_rowid();
            Ok(conn.query_row(
                &format!("SELECT {SESSION_COLUMNS} FROM chat_sessions WHERE id = ?1"),
                [id],
                session_from_row,
            )?)
        })
    }

    pub fn get_chat_session(&self, id: i64) -> StoreResult<ChatSessionRow> {
        self.read(|c| {
            c.query_row(&format!("SELECT {SESSION_COLUMNS} FROM chat_sessions WHERE id = ?1"), [id], session_from_row)
                .optional()?
                .ok_or_else(|| StoreError::NotFound(format!("chat {id}")))
        })
    }

    pub fn list_chat_sessions(&self) -> StoreResult<Vec<ChatSessionRow>> {
        self.read(|c| {
            let mut stmt =
                c.prepare(&format!("SELECT {SESSION_COLUMNS} FROM chat_sessions ORDER BY updated_at DESC, id DESC"))?;
            let rows = stmt.query_map([], session_from_row)?.collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    pub fn set_chat_learn(&self, id: i64, enabled: bool) -> StoreResult<()> {
        self.write(move |conn| {
            let n =
                conn.execute("UPDATE chat_sessions SET learn_enabled = ?2 WHERE id = ?1", params![id, enabled as i64])?;
            if n == 0 { Err(StoreError::NotFound(format!("chat {id}"))) } else { Ok(()) }
        })
    }

    pub fn set_chat_adapted(&self, id: i64, rel: Option<String>) -> StoreResult<()> {
        self.write(move |conn| {
            conn.execute("UPDATE chat_sessions SET adapted_ckpt_rel = ?2 WHERE id = ?1", params![id, rel])?;
            Ok(())
        })
    }

    pub fn delete_chat_session(&self, id: i64) -> StoreResult<()> {
        self.write(move |conn| {
            let n = conn.execute("DELETE FROM chat_sessions WHERE id = ?1", [id])?;
            if n == 0 { Err(StoreError::NotFound(format!("chat {id}"))) } else { Ok(()) }
        })
    }

    /// Append a message and return its id. `rows` is the recurrent-row count per generated character.
    pub fn add_chat_message(
        &self,
        session_id: i64,
        role: ChatRole,
        content: String,
        rows: Vec<u8>,
        chars_per_sec: Option<f64>,
    ) -> StoreResult<i64> {
        self.write(move |conn| {
            let now = UnixMs::now().0 as i64;
            let role = if role == ChatRole::User { "user" } else { "model" };
            let stats = chars_per_sec.map(|c| serde_json::json!({ "charsPerSec": c }).to_string());
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, rows_blob, stats_json, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![session_id, role, content, if rows.is_empty() { None } else { Some(rows) }, stats, now],
            )?;
            let id = conn.last_insert_rowid();
            conn.execute("UPDATE chat_sessions SET updated_at = ?2 WHERE id = ?1", params![session_id, now])?;
            Ok(id)
        })
    }

    pub fn mark_chat_message_learned(&self, message_id: i64, before: f64, after: f64) -> StoreResult<()> {
        self.write(move |conn| {
            conn.execute(
                "UPDATE chat_messages SET learned = 1, learn_nats_before = ?2, learn_nats_after = ?3 WHERE id = ?1",
                params![message_id, before, after],
            )?;
            Ok(())
        })
    }

    pub fn list_chat_messages(&self, session_id: i64) -> StoreResult<Vec<ChatMessage>> {
        self.read(|c| {
            let mut stmt = c.prepare(
                "SELECT id, role, content, learned, learn_nats_before, learn_nats_after, rows_blob, stats_json, created_at \
                 FROM chat_messages WHERE session_id = ?1 ORDER BY id",
            )?;
            let rows = stmt
                .query_map([session_id], |r| {
                    let role: String = r.get(1)?;
                    let stats: Option<String> = r.get(7)?;
                    Ok(ChatMessage {
                        id: r.get(0)?,
                        role: if role == "user" { ChatRole::User } else { ChatRole::Model },
                        content: r.get(2)?,
                        learned: r.get::<_, i64>(3)? != 0,
                        learn_nats_before: r.get(4)?,
                        learn_nats_after: r.get(5)?,
                        rows: r.get::<_, Option<Vec<u8>>>(6)?.unwrap_or_default(),
                        chars_per_sec: stats
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                            .and_then(|v| v["charsPerSec"].as_f64()),
                        created_at: UnixMs(r.get::<_, i64>(8)? as u64),
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Remove every message of a session (keeps the session itself).
    pub fn clear_chat_messages(&self, session_id: i64) -> StoreResult<()> {
        self.write(move |conn| {
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?1", [session_id])?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        (dir, store)
    }

    fn new_chat(store: &Store) -> ChatSessionRow {
        store
            .create_chat_session(NewChat {
                title: "Hello".into(),
                run_id: None,
                checkpoint_id: None,
                model_label: "Tiny run 1, after 5M characters".into(),
                mode: ChatMode::Conversation,
            })
            .unwrap()
    }

    #[test]
    fn sessions_and_messages_round_trip() {
        let (_d, store) = open();
        let chat = new_chat(&store);
        assert_eq!(chat.mode, ChatMode::Conversation);
        assert!(!chat.learn_enabled);

        store.add_chat_message(chat.id, ChatRole::User, "Once upon a time".into(), vec![], None).unwrap();
        let reply = store
            .add_chat_message(chat.id, ChatRole::Model, "there was a fox".into(), vec![1, 3, 2, 4], Some(42.5))
            .unwrap();
        store.mark_chat_message_learned(reply, 2.4, 2.3).unwrap();

        let msgs = store.list_chat_messages(chat.id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, ChatRole::User);
        assert!(msgs[0].rows.is_empty() && !msgs[0].learned);
        assert_eq!(msgs[1].rows, vec![1, 3, 2, 4]);
        assert_eq!(msgs[1].chars_per_sec, Some(42.5));
        assert!(msgs[1].learned);
        assert_eq!((msgs[1].learn_nats_before, msgs[1].learn_nats_after), (Some(2.4), Some(2.3)));
    }

    #[test]
    fn learn_flag_adapted_copy_and_deletion() {
        let (_d, store) = open();
        let chat = new_chat(&store);
        store.set_chat_learn(chat.id, true).unwrap();
        store.set_chat_adapted(chat.id, Some("chat/1/adapted".into())).unwrap();
        let got = store.get_chat_session(chat.id).unwrap();
        assert!(got.learn_enabled);
        assert_eq!(got.adapted_ckpt_rel.as_deref(), Some("chat/1/adapted"));

        store.add_chat_message(chat.id, ChatRole::User, "x".into(), vec![], None).unwrap();
        store.clear_chat_messages(chat.id).unwrap();
        assert!(store.list_chat_messages(chat.id).unwrap().is_empty());

        store.delete_chat_session(chat.id).unwrap();
        assert!(matches!(store.get_chat_session(chat.id), Err(StoreError::NotFound(_))));
        assert!(store.delete_chat_session(chat.id).is_err());
        assert!(store.list_chat_sessions().unwrap().is_empty());
    }

    #[test]
    fn newest_session_lists_first_and_messages_cascade_on_delete() {
        let (_d, store) = open();
        let a = new_chat(&store);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = new_chat(&store);
        std::thread::sleep(std::time::Duration::from_millis(5)); // timestamps have millisecond resolution
        store.add_chat_message(a.id, ChatRole::User, "hi".into(), vec![], None).unwrap();
        // activity moves a session to the top
        assert_eq!(store.list_chat_sessions().unwrap()[0].id, a.id);
        store.delete_chat_session(a.id).unwrap();
        assert_eq!(store.list_chat_sessions().unwrap().len(), 1);
        assert_eq!(store.list_chat_sessions().unwrap()[0].id, b.id);
        let n: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM chat_messages", [], |r| r.get(0))?)).unwrap();
        assert_eq!(n, 0, "messages are removed with their session");
    }
}
