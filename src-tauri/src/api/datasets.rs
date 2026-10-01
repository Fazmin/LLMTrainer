//! Datasets: the text a model trains on.

use std::path::PathBuf;
use std::sync::Arc;

use minagi_types::{
    AppError, ArithmeticParams, DatasetDetail, DatasetSummary, JobEvent, JobInfo, LaneMode, PathProbe, SplitConfig,
    StarterInfo, TextPreview,
};
use tauri::State;
use tauri::ipc::Channel;

use crate::datasets::{self, Emit};
use crate::state::AppState;

fn emitter(channel: Channel<JobEvent>) -> Emit {
    Arc::new(move |event| {
        let _ = channel.send(event);
    })
}

#[tauri::command]
#[specta::specta]
pub async fn list_datasets(state: State<'_, AppState>) -> Result<Vec<DatasetSummary>, AppError> {
    Ok(state.store.list_datasets()?)
}

#[tauri::command]
#[specta::specta]
pub async fn get_dataset(state: State<'_, AppState>, dataset_id: i64) -> Result<DatasetDetail, AppError> {
    Ok(state.store.get_dataset(dataset_id)?)
}

#[tauri::command]
#[specta::specta]
pub async fn list_starters() -> Result<Vec<StarterInfo>, AppError> {
    Ok(minagi_data::starters())
}

/// A quick look at what is being dragged over the window, before anything is scanned.
#[tauri::command]
#[specta::specta]
pub async fn probe_paths(paths: Vec<String>) -> Result<Vec<PathProbe>, AppError> {
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    tauri::async_runtime::spawn_blocking(move || minagi_data::probe_paths(&paths))
        .await
        .map_err(|e| AppError::Engine(e.to_string()))
}

/// Make a dataset from folders or files. Progress streams on `on_event`; the first message carries the job id.
#[tauri::command]
#[specta::specta]
pub async fn add_folders(
    state: State<'_, AppState>,
    paths: Vec<String>,
    lane_mode: LaneMode,
    on_event: Channel<JobEvent>,
) -> Result<DatasetDetail, AppError> {
    datasets::add_folders(&state, paths, lane_mode, emitter(on_event)).await
}

/// Rebuild a dataset from its folders, optionally with a new train/test split.
#[tauri::command]
#[specta::specta]
pub async fn rebuild_dataset(
    state: State<'_, AppState>,
    dataset_id: i64,
    split: Option<SplitConfig>,
    on_event: Channel<JobEvent>,
) -> Result<DatasetDetail, AppError> {
    datasets::rebuild(&state, dataset_id, split, emitter(on_event)).await
}

#[tauri::command]
#[specta::specta]
pub async fn get_split(state: State<'_, AppState>, dataset_id: i64) -> Result<SplitConfig, AppError> {
    Ok(state.store.dataset_split(dataset_id)?)
}

#[tauri::command]
#[specta::specta]
pub async fn update_lane(
    state: State<'_, AppState>,
    dataset_id: i64,
    lane: String,
    enabled: Option<bool>,
    display_name: Option<String>,
    sample_prompt: Option<String>,
) -> Result<DatasetDetail, AppError> {
    Ok(state.store.update_lane(dataset_id, lane, enabled, display_name, sample_prompt)?)
}

/// A random passage from one lane, so people can see what the model will read.
#[tauri::command]
#[specta::specta]
pub async fn preview_text(
    state: State<'_, AppState>,
    dataset_id: i64,
    lane: Option<String>,
    n_chars: u32,
    seed: u32,
) -> Result<TextPreview, AppError> {
    let detail = state.store.get_dataset(dataset_id)?;
    let lane = lane
        .or_else(|| detail.lanes.iter().find(|l| l.enabled).map(|l| l.name.clone()))
        .ok_or(AppError::DatasetNotReady)?;
    let root = datasets::dataset_root_path(&state, dataset_id)?;
    tauri::async_runtime::spawn_blocking(move || minagi_data::preview_text(&root, &lane, n_chars as usize, seed))
        .await
        .map_err(|e| AppError::Engine(e.to_string()))?
        .map_err(AppError::from)
}

/// Download or generate one of the built-in starter datasets.
#[tauri::command]
#[specta::specta]
pub async fn install_starter(
    state: State<'_, AppState>,
    starter_id: String,
    on_event: Channel<JobEvent>,
) -> Result<DatasetDetail, AppError> {
    datasets::install_starter(&state, &starter_id, emitter(on_event)).await
}

#[tauri::command]
#[specta::specta]
pub async fn generate_arithmetic(
    state: State<'_, AppState>,
    params: ArithmeticParams,
    on_event: Channel<JobEvent>,
) -> Result<DatasetDetail, AppError> {
    datasets::generate_arithmetic(&state, params, emitter(on_event)).await
}

#[tauri::command]
#[specta::specta]
pub async fn delete_dataset(state: State<'_, AppState>, dataset_id: i64, delete_files: bool) -> Result<(), AppError> {
    datasets::delete(&state, dataset_id, delete_files)
}

#[tauri::command]
#[specta::specta]
pub async fn list_jobs(state: State<'_, AppState>) -> Result<Vec<JobInfo>, AppError> {
    Ok(state.store.list_jobs()?)
}

/// Stop a running download, scan or build. Whatever was finished is kept, so a download can continue later.
#[tauri::command]
#[specta::specta]
pub async fn cancel_job(state: State<'_, AppState>, job_id: String) -> Result<bool, AppError> {
    Ok(state.jobs.cancel(&job_id))
}
