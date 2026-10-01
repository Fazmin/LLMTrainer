//! Creating, starting, controlling and browsing runs.

use minagi_store::{NewRun, RunOrigin};
use minagi_types::{AppError, CreateRunRequest, Ctl, LiveMsg, LiveSnapshot, PresetConfig, RunSummary};
use tauri::State;
use tauri::ipc::Channel;

use crate::host;
use crate::state::AppState;

/// What a run remembers about its text, so its history stays meaningful if the dataset changes later.
pub(crate) fn dataset_context(
    state: &AppState,
    dataset_id: Option<i64>,
) -> Result<(serde_json::Value, Option<u64>), AppError> {
    match dataset_id {
        Some(id) => {
            let detail = state.store.get_dataset(id)?;
            if detail.summary.status != minagi_types::DatasetStatus::Ready {
                return Err(AppError::DatasetNotReady);
            }
            if detail.lanes.iter().all(|l| !l.enabled) {
                return Err(AppError::Invalid("Switch on at least one kind of text before training.".into()));
            }
            let total = crate::datasets::enabled_train_bytes(&detail);
            Ok((serde_json::to_value(&detail).unwrap_or_default(), Some(total)))
        }
        None => Ok((serde_json::json!({}), None)),
    }
}

/// Continue a run's saved model as a new run, with different settings or text. The new run starts from the save.
#[tauri::command]
#[specta::specta]
pub async fn fork_run(state: State<'_, AppState>, req: minagi_types::ForkRunRequest) -> Result<RunSummary, AppError> {
    crate::fork::fork_run(&state, req)
}

#[tauri::command]
#[specta::specta]
pub async fn create_run(state: State<'_, AppState>, req: CreateRunRequest) -> Result<RunSummary, AppError> {
    let model = req.model.unwrap_or_else(|| req.preset.model());
    let train = req.train.unwrap_or_else(|| req.preset.train());
    model.validate().map_err(AppError::Invalid)?;
    train.validate(&model).map_err(AppError::Invalid)?;

    let name = match req.name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) {
        Some(n) => n,
        None => format!("{} run {}", req.preset.display_name(), state.store.list_runs()?.len() + 1),
    };
    let hw = state.hardware();
    let (snapshot, chars_total) = dataset_context(&state, req.dataset_id)?;
    let run = state.store.create_run(NewRun {
        name,
        preset: req.preset,
        dataset_id: req.dataset_id,
        dataset_snapshot: snapshot,
        model,
        train,
        goal: req.goal,
        engine_version: state.factory.version(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        origin: RunOrigin::Trained,
        parent_run_id: None,
        backend: hw.selected_backend().map(|b| b.name.clone()),
        chars_total,
    })?;
    Ok(run)
}

/// Start a run, or continue it from its latest checkpoint when it has one.
#[tauri::command]
#[specta::specta]
pub async fn start_run(state: State<'_, AppState>, run_id: i64) -> Result<RunSummary, AppError> {
    host::start_session(&state, run_id)
}

#[tauri::command]
#[specta::specta]
pub async fn pause_run(state: State<'_, AppState>) -> Result<(), AppError> {
    host::send_ctl(&state, Ctl::Pause)
}

#[tauri::command]
#[specta::specta]
pub async fn resume_run(state: State<'_, AppState>) -> Result<(), AppError> {
    host::send_ctl(&state, Ctl::Resume)
}

/// Stop the active run. With `save`, a checkpoint is written first so nothing is lost.
#[tauri::command]
#[specta::specta]
pub async fn stop_run(state: State<'_, AppState>, save: bool) -> Result<(), AppError> {
    host::send_ctl(&state, Ctl::Stop { save })
}

#[tauri::command]
#[specta::specta]
pub async fn checkpoint_now(state: State<'_, AppState>) -> Result<(), AppError> {
    host::send_ctl(&state, Ctl::CheckpointNow)
}

#[tauri::command]
#[specta::specta]
pub async fn sample_now(state: State<'_, AppState>) -> Result<(), AppError> {
    host::send_ctl(&state, Ctl::SampleNow)
}

#[tauri::command]
#[specta::specta]
pub async fn eval_now(state: State<'_, AppState>) -> Result<(), AppError> {
    host::send_ctl(&state, Ctl::EvalNow)
}

#[tauri::command]
#[specta::specta]
pub async fn list_runs(state: State<'_, AppState>) -> Result<Vec<RunSummary>, AppError> {
    Ok(state.store.list_runs()?)
}

#[tauri::command]
#[specta::specta]
pub async fn get_run(state: State<'_, AppState>, run_id: i64) -> Result<RunSummary, AppError> {
    Ok(state.store.get_run(run_id)?)
}

/// The full configuration a run was created with (for comparing runs and for forking one).
#[tauri::command]
#[specta::specta]
pub async fn get_run_config(state: State<'_, AppState>, run_id: i64) -> Result<PresetConfig, AppError> {
    let run = state.store.get_run(run_id)?;
    let (model, train) = state.store.run_config(run_id)?;
    Ok(PresetConfig { preset: run.preset, model, train })
}

#[tauri::command]
#[specta::specta]
pub async fn get_active_run(state: State<'_, AppState>) -> Result<Option<RunSummary>, AppError> {
    match host::active_run_id(&state) {
        Some(id) => Ok(Some(state.store.get_run(id)?)),
        None => Ok(None),
    }
}

#[tauri::command]
#[specta::specta]
pub async fn rename_run(
    state: State<'_, AppState>,
    run_id: i64,
    name: String,
    notes: String,
) -> Result<RunSummary, AppError> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(AppError::Invalid("A run needs a name.".into()));
    }
    Ok(state.store.rename_run(run_id, name, notes)?)
}

/// Delete a run's history, and optionally its saved model files from disk.
#[tauri::command]
#[specta::specta]
pub async fn delete_run(state: State<'_, AppState>, run_id: i64, delete_files: bool) -> Result<(), AppError> {
    if host::active_run_id(&state) == Some(run_id) {
        return Err(AppError::RunActive);
    }
    let dir = state.paths.resolve(&state.store.run_dir_rel(run_id)?);
    state.store.delete_run(run_id)?;
    if delete_files {
        // Only ever remove a folder inside our own runs directory.
        if dir.starts_with(state.paths.root.join("runs")) {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}

/// Subscribe to the ordered live feed. Call once at startup and again after a webview reload; the returned snapshot
/// is the current state, so the UI can render immediately.
#[tauri::command]
#[specta::specta]
pub async fn subscribe_live(state: State<'_, AppState>, on_event: Channel<LiveMsg>) -> Result<LiveSnapshot, AppError> {
    let mut snapshot = state.live.subscribe(on_event);
    // If nothing has run since launch, fall back to the most recently active run in the database.
    if snapshot.run.is_none() {
        snapshot.run = state.store.list_runs()?.into_iter().next();
    }
    Ok(snapshot)
}

/// Export a saved model as a portable folder. Progress streams on `on_event`.
#[tauri::command]
#[specta::specta]
pub async fn export_model(
    state: State<'_, AppState>,
    req: minagi_types::ExportRequest,
    on_event: Channel<minagi_types::JobEvent>,
) -> Result<minagi_types::ExportResult, AppError> {
    let emit: crate::datasets::Emit = std::sync::Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    crate::export::export_model(&state, req, emit).await
}

/// Export a saved model as a safetensors file for other tools. Progress streams on `on_event`.
#[tauri::command]
#[specta::specta]
pub async fn export_safetensors(
    state: State<'_, AppState>,
    req: minagi_types::ExportRequest,
    on_event: Channel<minagi_types::JobEvent>,
) -> Result<minagi_types::ExportResult, AppError> {
    let emit: crate::datasets::Emit = std::sync::Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    crate::export::export_safetensors(&state, req, emit).await
}

/// What importing a folder would do, before anything is copied.
#[tauri::command]
#[specta::specta]
pub async fn preview_import(
    state: State<'_, AppState>,
    folder: String,
) -> Result<minagi_types::ImportPreview, AppError> {
    crate::export::preview_import(&state, &folder)
}

/// Bring in a folder made by Export (from this or another computer), or the original program's weights folder.
#[tauri::command]
#[specta::specta]
pub async fn import_model(
    state: State<'_, AppState>,
    folder: String,
    on_event: Channel<minagi_types::JobEvent>,
) -> Result<RunSummary, AppError> {
    let emit: crate::datasets::Emit = std::sync::Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    crate::export::import_model(&state, folder, emit).await
}
