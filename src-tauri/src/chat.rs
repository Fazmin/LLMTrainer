//! Chat sessions: a loaded generator per open conversation, streaming replies, and optional learning from the
//! conversation on a separate copy of the model (the saved run is never modified).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use minagi_store::ChatSessionRow;
use minagi_types::{
    AppError, AppResult, ChatEvent, ChatMode, ChatParams, ChatRole, ChatSessionInfo, Count, GenChar, GenRequest,
    Generator, GeneratorInfo,
};

use crate::state::AppState;

const STOP_MARKER: &str = "</bot>";

pub struct ChatSession {
    generator: Mutex<Box<dyn Generator>>,
    pub info: GeneratorInfo,
    cancel: AtomicBool,
    busy: AtomicBool,
}

/// Open conversations. A session is "loaded" when its generator is in memory.
#[derive(Clone, Default)]
pub struct ChatHost {
    sessions: Arc<Mutex<HashMap<i64, Arc<ChatSession>>>>,
}

impl ChatHost {
    pub fn insert(&self, id: i64, generator: Box<dyn Generator>) -> Arc<ChatSession> {
        let session = Arc::new(ChatSession {
            info: generator.info(),
            generator: Mutex::new(generator),
            cancel: AtomicBool::new(false),
            busy: AtomicBool::new(false),
        });
        self.sessions.lock().unwrap().insert(id, session.clone());
        session
    }

    pub fn get(&self, id: i64) -> AppResult<Arc<ChatSession>> {
        self.sessions
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or_else(|| AppError::Invalid("This chat is not open. Open it again to continue.".into()))
    }

    pub fn remove(&self, id: i64) {
        self.sessions.lock().unwrap().remove(&id);
    }

    pub fn is_loaded(&self, id: i64) -> bool {
        self.sessions.lock().unwrap().contains_key(&id)
    }
}

/// "5.4M" style count for labels.
fn short_count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.0}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}

/// Load the generator for a run's checkpoint. Returns it with a human label like "Tiny run 1, after 5.4M characters".
pub fn load_generator(
    state: &AppState,
    run_id: i64,
    checkpoint_id: Option<i64>,
) -> AppResult<(Box<dyn Generator>, String, Option<i64>)> {
    let run = state.store.get_run(run_id)?;
    let ckpts = state.store.list_checkpoints(run_id)?;
    let ckpt = match checkpoint_id {
        Some(id) => ckpts.iter().find(|c| c.id == id).ok_or_else(|| AppError::NotFound(format!("save {id}")))?,
        None => ckpts.iter().find(|c| c.is_best).or_else(|| ckpts.first()).ok_or_else(|| {
            AppError::Invalid("This run has no saved progress yet. Let it train for a few minutes first.".into())
        })?,
    };
    let dir = state.paths.resolve(&state.store.run_dir_rel(run_id)?).join(&ckpt.path);
    let generator = state.factory.open_generator(&dir, state.hardware().selected)?;
    let label = format!("{}, after {} characters", run.name, short_count(ckpt.chars.0));
    Ok((generator, label, Some(ckpt.id)))
}

pub fn session_info(host: &ChatHost, row: &ChatSessionRow) -> ChatSessionInfo {
    let loaded = host.sessions.lock().unwrap().get(&row.id).cloned();
    ChatSessionInfo {
        id: row.id,
        title: row.title.clone(),
        run_id: row.run_id,
        model_label: row.model_label.clone(),
        mode: row.mode,
        learn_enabled: row.learn_enabled,
        supports_learn: loaded.as_ref().is_some_and(|s| s.info.supports_learn),
        needs_gb: loaded.as_ref().map_or(0.0, |s| s.info.needs_gb),
        has_adapted_copy: row.adapted_ckpt_rel.is_some(),
        created_at: row.created_at,
    }
}

pub fn cancel(host: &ChatHost, id: i64) {
    if let Ok(s) = host.get(id) {
        s.cancel.store(true, Ordering::SeqCst);
    }
}

/// The text the model sees for a user turn, and where to stop.
fn prompt_for(mode: ChatMode, text: &str) -> (String, Option<String>) {
    match mode {
        ChatMode::Continue => (text.to_string(), None),
        ChatMode::Conversation => (format!("<user>\n{text}\n</user>\n<bot>\n"), Some(STOP_MARKER.to_string())),
    }
}

/// Run one reply on its own thread: stream characters to `emit`, persist both messages, then learn if asked to.
pub fn send(
    state: &AppState,
    session_id: i64,
    text: String,
    params: ChatParams,
    emit: impl Fn(ChatEvent) + Send + 'static,
) -> AppResult<i64> {
    let session = state.chat.get(session_id)?;
    if session.busy.swap(true, Ordering::SeqCst) {
        return Err(AppError::Invalid("The model is still writing. Stop it or wait for it to finish.".into()));
    }
    let row = state.store.get_chat_session(session_id).inspect_err(|_| session.busy.store(false, Ordering::SeqCst))?;
    let user_id = state
        .store
        .add_chat_message(session_id, ChatRole::User, text.clone(), vec![], None)
        .inspect_err(|_| session.busy.store(false, Ordering::SeqCst))?;
    session.cancel.store(false, Ordering::SeqCst);

    let store = state.store.clone();
    std::thread::Builder::new()
        .name("minagi-chat".into())
        .spawn(move || {
            let finish = |e: Option<AppError>| {
                session.busy.store(false, Ordering::SeqCst);
                if let Some(error) = e {
                    emit(ChatEvent::Error { error });
                }
            };
            let (prompt, stop_at) = prompt_for(row.mode, &text);
            let req = GenRequest { prompt, max_new: params.max_new, adapt: params.adapt, stop_at };

            let mut reply = String::new();
            let mut rows: Vec<u8> = Vec::new();
            let result = {
                let mut g = session.generator.lock().unwrap();
                g.generate(&req, &mut |c: GenChar| {
                    reply.push_str(&c.text);
                    rows.push(c.rows);
                    emit(ChatEvent::Chars { text: c.text, rows: vec![c.rows], experts: c.experts });
                    !session.cancel.load(Ordering::SeqCst)
                })
            };
            let stats = match result {
                Ok(s) => s,
                Err(e) => return finish(Some(e)),
            };
            let shown = reply.strip_suffix(STOP_MARKER).unwrap_or(&reply).trim_end().to_string();
            let message_id = match store.add_chat_message(
                session_id,
                ChatRole::Model,
                shown.clone(),
                rows,
                Some(stats.chars_per_sec),
            ) {
                Ok(id) => id,
                Err(e) => return finish(Some(e.into())),
            };
            emit(ChatEvent::Done { message_id, chars: Count(stats.chars.0), chars_per_sec: stats.chars_per_sec });

            if row.learn_enabled && session.info.supports_learn && !shown.is_empty() {
                let exchange = match row.mode {
                    ChatMode::Continue => format!("{text}{shown}"),
                    ChatMode::Conversation => format!("<user>\n{text}\n</user>\n<bot>\n{shown}\n</bot>\n"),
                };
                match session.generator.lock().unwrap().learn(&exchange) {
                    Ok(l) => {
                        let _ = store.mark_chat_message_learned(message_id, l.nats_before, l.nats_after);
                        emit(ChatEvent::Learned { nats_before: l.nats_before, nats_after: l.nats_after });
                    }
                    Err(e) => return finish(Some(e)),
                }
            }
            finish(None);
        })
        .map_err(|e| AppError::Engine(e.to_string()))?;
    Ok(user_id)
}

/// Write the learned (adapted) copy of the model into the chat's own folder and remember where it is.
pub fn save_adapted(state: &AppState, session_id: i64) -> AppResult<minagi_types::CheckpointMeta> {
    let session = state.chat.get(session_id)?;
    if !session.info.supports_learn {
        return Err(AppError::Invalid("This model cannot learn from chats.".into()));
    }
    let rel = format!("chat/{session_id}/adapted");
    let meta = session.generator.lock().unwrap().save_adapted(&state.paths.resolve(&rel))?;
    state.store.set_chat_adapted(session_id, Some(rel))?;
    Ok(meta)
}
