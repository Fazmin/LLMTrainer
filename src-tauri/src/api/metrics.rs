//! Read-only queries behind the charts, the samples timeline and the expert pool views.

use minagi_types::{
    AppError, CheckpointInfo, EvalPoint, PoolSnapshot, RunEvent, SampleRound, SeriesData, SeriesRequest, Step, StepAt,
};
use tauri::State;

use crate::state::AppState;

/// Chart series, downsampled on the server. Keys are metric names such as `train.nats`, or `eval.overall`,
/// `eval.train`, `eval.gap`, `eval.domain.<name>`.
#[tauri::command]
#[specta::specta]
pub async fn get_series(state: State<'_, AppState>, req: SeriesRequest) -> Result<Vec<SeriesData>, AppError> {
    Ok(state.store.get_series(&req)?)
}

#[tauri::command]
#[specta::specta]
pub async fn get_eval_points(state: State<'_, AppState>, run_id: i64) -> Result<Vec<EvalPoint>, AppError> {
    Ok(state.store.eval_points(run_id)?)
}

/// Names of the held-out domains scored during a run (one chart line each).
#[tauri::command]
#[specta::specta]
pub async fn get_eval_domains(state: State<'_, AppState>, run_id: i64) -> Result<Vec<String>, AppError> {
    Ok(state.store.eval_domains(run_id)?)
}

#[tauri::command]
#[specta::specta]
pub async fn get_events(
    state: State<'_, AppState>,
    run_id: i64,
    kinds: Option<Vec<String>>,
) -> Result<Vec<RunEvent>, AppError> {
    Ok(state.store.get_events(run_id, kinds)?)
}

/// The sample round at `step`, or the latest when `step` is omitted.
#[tauri::command]
#[specta::specta]
pub async fn get_samples(
    state: State<'_, AppState>,
    run_id: i64,
    step: Option<Step>,
) -> Result<Option<SampleRound>, AppError> {
    Ok(state.store.get_samples(run_id, step.map(|s| s.0))?)
}

#[tauri::command]
#[specta::specta]
pub async fn list_sample_steps(state: State<'_, AppState>, run_id: i64) -> Result<Vec<StepAt>, AppError> {
    Ok(state.store.list_sample_steps(run_id)?.into_iter().map(|(step, chars)| StepAt { step, chars }).collect())
}

/// The expert pool at `step` (nearest snapshot at or before it), or the latest when `step` is omitted.
#[tauri::command]
#[specta::specta]
pub async fn get_pool_snapshot(
    state: State<'_, AppState>,
    run_id: i64,
    step: Option<Step>,
) -> Result<Option<PoolSnapshot>, AppError> {
    Ok(state.store.get_pool_snapshot(run_id, step.map(|s| s.0))?)
}

#[tauri::command]
#[specta::specta]
pub async fn list_pool_steps(state: State<'_, AppState>, run_id: i64) -> Result<Vec<StepAt>, AppError> {
    Ok(state.store.list_pool_steps(run_id)?.into_iter().map(|(step, chars)| StepAt { step, chars }).collect())
}

#[tauri::command]
#[specta::specta]
pub async fn list_checkpoints(state: State<'_, AppState>, run_id: i64) -> Result<Vec<CheckpointInfo>, AppError> {
    Ok(state.store.list_checkpoints(run_id)?)
}
