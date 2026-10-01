//! App-level information, hardware, and size recommendations.

use minagi_store::SCHEMA_VERSION;
use minagi_types::{
    AppError, AppInfo, Count, DataStats, Estimate, HardwareInfo, ModelConfig, Preset, PresetConfig, Recommendation,
    RunStorage, StorageUsage, TrainConfig, recommend_preset as recommend,
};
use tauri::State;

use crate::state::AppState;

#[tauri::command]
#[specta::specta]
pub async fn app_info(state: State<'_, AppState>) -> Result<AppInfo, AppError> {
    Ok(AppInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        engine_name: state.factory.name().to_string(),
        engine_version: state.factory.version(),
        is_mock: state.factory.is_mock(),
        data_dir: state.data_dir().to_string_lossy().to_string(),
        schema_version: SCHEMA_VERSION,
    })
}

#[tauri::command]
#[specta::specta]
pub async fn hardware_info(state: State<'_, AppState>) -> Result<HardwareInfo, AppError> {
    Ok(state.hardware().clone())
}

/// Full configuration of every preset, so the Setup screen can show and edit them.
#[tauri::command]
#[specta::specta]
pub async fn preset_configs() -> Result<Vec<PresetConfig>, AppError> {
    Ok([Preset::Tiny, Preset::Small, Preset::Full]
        .into_iter()
        .map(|preset| PresetConfig { preset, model: preset.model(), train: preset.train() })
        .collect())
}

#[tauri::command]
#[specta::specta]
pub async fn recommend_preset(state: State<'_, AppState>, dataset_id: Option<i64>) -> Result<Recommendation, AppError> {
    let stats = dataset_stats(&state, dataset_id)?;
    Ok(recommend(state.hardware(), stats.as_ref()))
}

/// Size of a dataset's enabled text, for sizing recommendations and estimates.
fn dataset_stats(state: &AppState, dataset_id: Option<i64>) -> Result<Option<DataStats>, AppError> {
    let Some(id) = dataset_id else { return Ok(None) };
    let detail = state.store.get_dataset(id)?;
    Ok(Some(DataStats {
        train_bytes: Count(crate::datasets::enabled_train_bytes(&detail)),
        val_bytes: detail.summary.val_bytes,
        lanes: detail.lanes.iter().filter(|l| l.enabled).count() as u32,
    }))
}

/// Predicted cost of a configuration on this computer. Cheap enough to call on every edit.
#[tauri::command]
#[specta::specta]
pub async fn estimate_run(
    state: State<'_, AppState>,
    model: ModelConfig,
    train: TrainConfig,
    dataset_id: Option<i64>,
) -> Result<Estimate, AppError> {
    let stats = dataset_stats(&state, dataset_id)?;
    Ok(state.factory.estimate(&model, &train, state.hardware(), stats.as_ref()))
}

/// Size of everything under `dir` (0 when it does not exist). Does not follow symlinks.
fn dir_size(dir: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

/// Where the app's disk space goes: datasets, saved models, chats and the database.
#[tauri::command]
#[specta::specta]
pub async fn storage_usage(state: State<'_, AppState>) -> Result<StorageUsage, AppError> {
    let app = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let root = app.data_dir().to_path_buf();
        let mut per_run = Vec::new();
        for run in app.store.list_runs()? {
            let dir = app.paths.resolve(&app.store.run_dir_rel(run.id)?);
            per_run.push(RunStorage { run_id: run.id, name: run.name, bytes: Count(dir_size(&dir)) });
        }
        per_run.sort_by(|a, b| b.bytes.0.cmp(&a.bytes.0));
        let db = ["", "-wal", "-shm"]
            .iter()
            .map(|ext| std::fs::metadata(format!("{}{ext}", app.paths.db().display())).map_or(0, |m| m.len()))
            .sum();
        Ok(StorageUsage {
            data_dir: root.to_string_lossy().to_string(),
            datasets_bytes: Count(dir_size(&app.paths.datasets())),
            runs_bytes: Count(per_run.iter().map(|r| r.bytes.0).sum()),
            chats_bytes: Count(dir_size(&root.join("chat"))),
            database_bytes: Count(db),
            free_bytes: Count((crate::hardware::free_disk_gb(&root) * 1_073_741_824.0) as u64),
            per_run,
        })
    })
    .await
    .map_err(|e| AppError::Engine(e.to_string()))?
}
