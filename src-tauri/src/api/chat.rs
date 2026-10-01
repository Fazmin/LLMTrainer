//! Chat with a trained model.

use minagi_store::NewChat;
use minagi_types::{AppError, ChatEvent, ChatMessage, ChatOpenRequest, ChatParams, ChatSessionInfo, CheckpointMeta};
use tauri::State;
use tauri::ipc::Channel;

use crate::chat;
use crate::state::AppState;

/// Load a model from a run's saved progress and start a new conversation with it.
#[tauri::command]
#[specta::specta]
pub async fn chat_open(state: State<'_, AppState>, req: ChatOpenRequest) -> Result<ChatSessionInfo, AppError> {
    let app = state.inner().clone();
    // Loading a model can take a while: keep it off the async runtime's threads.
    let (generator, label, checkpoint_id) =
        tauri::async_runtime::spawn_blocking(move || chat::load_generator(&app, req.run_id, req.checkpoint_id))
            .await
            .map_err(|e| AppError::Engine(e.to_string()))??;
    let run = state.store.get_run(req.run_id)?;
    let row = state.store.create_chat_session(NewChat {
        title: format!("Chat with {}", run.name),
        run_id: Some(req.run_id),
        checkpoint_id,
        model_label: label,
        mode: req.mode,
    })?;
    state.chat.insert(row.id, generator);
    Ok(chat::session_info(&state.chat, &row))
}

/// Reload an earlier conversation's model so it can be continued.
#[tauri::command]
#[specta::specta]
pub async fn chat_resume(state: State<'_, AppState>, session_id: i64) -> Result<ChatSessionInfo, AppError> {
    let row = state.store.get_chat_session(session_id)?;
    if !state.chat.is_loaded(session_id) {
        let run_id = row.run_id.ok_or_else(|| AppError::NotFound("the run for this chat".into()))?;
        let app = state.inner().clone();
        let checkpoint_id = row.checkpoint_id;
        let (generator, _, _) =
            tauri::async_runtime::spawn_blocking(move || chat::load_generator(&app, run_id, checkpoint_id))
                .await
                .map_err(|e| AppError::Engine(e.to_string()))??;
        state.chat.insert(session_id, generator);
    }
    Ok(chat::session_info(&state.chat, &row))
}

/// Send a message. The reply streams over `on_event`; the returned id is the stored user message.
#[tauri::command]
#[specta::specta]
pub async fn chat_send(
    state: State<'_, AppState>,
    session_id: i64,
    text: String,
    params: ChatParams,
    on_event: Channel<ChatEvent>,
) -> Result<i64, AppError> {
    if text.trim().is_empty() {
        return Err(AppError::Invalid("Type something for the model to continue.".into()));
    }
    chat::send(&state, session_id, text, params, move |e| {
        let _ = on_event.send(e);
    })
}

#[tauri::command]
#[specta::specta]
pub async fn chat_stop(state: State<'_, AppState>, session_id: i64) -> Result<(), AppError> {
    chat::cancel(&state.chat, session_id);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn chat_reset(state: State<'_, AppState>, session_id: i64) -> Result<(), AppError> {
    chat::cancel(&state.chat, session_id);
    Ok(state.store.clear_chat_messages(session_id)?)
}

/// Turn learning from this conversation on or off. Learning changes a separate copy, never the saved run.
#[tauri::command]
#[specta::specta]
pub async fn chat_set_learn(
    state: State<'_, AppState>,
    session_id: i64,
    enabled: bool,
) -> Result<ChatSessionInfo, AppError> {
    let session = state.chat.get(session_id)?;
    if enabled && !session.info.supports_learn {
        return Err(AppError::Invalid("This model cannot learn from chats.".into()));
    }
    state.store.set_chat_learn(session_id, enabled)?;
    Ok(chat::session_info(&state.chat, &state.store.get_chat_session(session_id)?))
}

/// Keep what the model learned in this chat as its own saved copy.
#[tauri::command]
#[specta::specta]
pub async fn chat_save_adapted(state: State<'_, AppState>, session_id: i64) -> Result<CheckpointMeta, AppError> {
    let app = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || chat::save_adapted(&app, session_id))
        .await
        .map_err(|e| AppError::Engine(e.to_string()))?
}

#[tauri::command]
#[specta::specta]
pub async fn chat_close(state: State<'_, AppState>, session_id: i64) -> Result<(), AppError> {
    chat::cancel(&state.chat, session_id);
    state.chat.remove(session_id);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn chat_list_sessions(state: State<'_, AppState>) -> Result<Vec<ChatSessionInfo>, AppError> {
    Ok(state.store.list_chat_sessions()?.iter().map(|r| chat::session_info(&state.chat, r)).collect())
}

#[tauri::command]
#[specta::specta]
pub async fn chat_get_messages(state: State<'_, AppState>, session_id: i64) -> Result<Vec<ChatMessage>, AppError> {
    Ok(state.store.list_chat_messages(session_id)?)
}

#[tauri::command]
#[specta::specta]
pub async fn chat_delete_session(state: State<'_, AppState>, session_id: i64) -> Result<(), AppError> {
    state.chat.remove(session_id);
    Ok(state.store.delete_chat_session(session_id)?)
}
