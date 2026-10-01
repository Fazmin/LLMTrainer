//! Preparing datasets: scanning folders, building the `train/` + `val/` layout the engine reads, installing starter
//! datasets, and generating practice text. Each is a background job that reports progress and can be cancelled.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use minagi_data::{
    CancelFlag, DataError, ScanOptions, assign_color_slots, detect_lanes, fill_sample_prompts, find_starter,
    http_client, materialize, plan_split, scan_paths,
};
use minagi_store::{DatasetFinal, DatasetRoot, NewDataset, Store, slugify};
use minagi_types::{
    AppError, AppResult, ArithmeticParams, Count, DatasetDetail, DatasetKind, DatasetStatus, JobEvent, JobProgress,
    JobState, LaneInfo, LaneMode, SkippedSummary, SplitConfig,
};
use tokio_util::sync::CancellationToken;

use crate::state::{AppPaths, AppState};

/// How to stop a running job: some work polls a flag, some awaits a token.
#[derive(Clone, Default)]
pub struct JobCtl {
    pub flag: CancelFlag,
    pub token: CancellationToken,
}

impl JobCtl {
    pub fn cancel(&self) {
        self.flag.cancel();
        self.token.cancel();
    }
}

#[derive(Clone, Default)]
pub struct JobRegistry(Arc<Mutex<HashMap<String, JobCtl>>>);

impl JobRegistry {
    pub fn cancel(&self, id: &str) -> bool {
        match self.0.lock().unwrap().get(id) {
            Some(ctl) => {
                ctl.cancel();
                true
            }
            None => false,
        }
    }

    fn insert(&self, id: &str, ctl: JobCtl) {
        self.0.lock().unwrap().insert(id.to_string(), ctl);
    }

    fn remove(&self, id: &str) {
        self.0.lock().unwrap().remove(id);
    }
}

pub type Emit = Arc<dyn Fn(JobEvent) + Send + Sync>;

/// One running job: records progress in the database (about twice a second) and tells the UI.
pub struct Job {
    pub id: String,
    pub ctl: JobCtl,
    store: Store,
    registry: JobRegistry,
    emit: Emit,
    last_saved: Mutex<Instant>,
}

impl Job {
    pub fn start(state: &AppState, kind: &str, subject: String, emit: Emit) -> AppResult<Job> {
        let id = state.store.create_job(kind, Some(subject))?;
        let ctl = JobCtl::default();
        state.jobs.insert(&id, ctl.clone());
        state.store.update_job(&id, JobState::Running, Some(0.0), None, None);
        emit(JobEvent::State { job_id: id.clone(), state: JobState::Running, error: None });
        Ok(Job {
            id,
            ctl,
            store: state.store.clone(),
            registry: state.jobs.clone(),
            emit,
            last_saved: Mutex::new(Instant::now() - Duration::from_secs(1)),
        })
    }

    pub fn progress(&self, p: JobProgress) {
        let fraction = p.total.filter(|t| *t > 0.0).map(|t| (p.done / t).clamp(0.0, 1.0));
        let mut last = self.last_saved.lock().unwrap();
        if last.elapsed() >= Duration::from_millis(500) {
            *last = Instant::now();
            self.store.update_job(&self.id, JobState::Running, fraction, Some(p.message.clone()), None);
        }
        drop(last);
        (self.emit)(JobEvent::Progress { job_id: self.id.clone(), progress: p });
    }

    /// Record how the job ended and tell the UI.
    pub fn finish<T>(self, result: &AppResult<T>) {
        let (state, error) = match result {
            Ok(_) => (JobState::Done, None),
            Err(AppError::Cancelled) => (JobState::Cancelled, None),
            Err(e) => (JobState::Failed, Some(e.to_string())),
        };
        self.store.update_job(&self.id, state, (state == JobState::Done).then_some(1.0), None, error.clone());
        // A finished job must be visible to the very next read, not sit in the write batch.
        self.store.flush();
        self.registry.remove(&self.id);
        (self.emit)(JobEvent::State { job_id: self.id.clone(), state, error });
    }
}

fn dataset_dir_name(id: i64, name: &str) -> String {
    format!("{id}-{}", slugify(name))
}

/// Absolute folder of a prepared dataset (the engine's `data_root`).
pub fn dataset_root_path(state: &AppState, id: i64) -> AppResult<PathBuf> {
    let root = state.store.dataset_root(id)?;
    match (root.root_abs, root.root_rel) {
        (Some(abs), _) => Ok(PathBuf::from(abs)),
        (None, Some(rel)) => Ok(state.paths.resolve(&rel)),
        _ => Err(AppError::DatasetNotReady),
    }
}

fn from_data_error(e: DataError) -> AppError {
    e.into()
}

fn lane_infos_from_report(report: &[minagi_data::LaneReport]) -> Vec<LaneInfo> {
    let names: Vec<&str> = report.iter().map(|l| l.name.as_str()).collect();
    let slots = assign_color_slots(&names);
    report
        .iter()
        .map(|l| LaneInfo {
            name: l.name.clone(),
            display_name: l.name.clone(),
            color_slot: slots.get(&l.name).copied().unwrap_or(8),
            enabled: true,
            n_files: Count(l.train_files),
            train_bytes: Count(l.train_bytes),
            val_files: Count(l.val_files),
            val_bytes: Count(l.val_bytes),
            sample_prompt: None,
        })
        .collect()
}

/// Scan, split and build a dataset from the user's folders. Blocking: run it on a worker thread.
fn prepare_blocking(
    paths: &AppPaths,
    store: &Store,
    id: i64,
    name: &str,
    sources: &[PathBuf],
    mode: LaneMode,
    job: &Job,
) -> AppResult<DatasetFinal> {
    let cancel = &job.ctl.flag;
    let scan = scan_paths(sources, &ScanOptions::default(), &Default::default(), cancel, &mut |p| {
        job.progress(JobProgress {
            done: p.files_seen as f64,
            total: None,
            unit: "files".into(),
            message: format!("Looking for text: {} readable files so far", p.text_files),
            bytes_per_sec: None,
            eta_seconds: None,
        });
    });
    if scan.cancelled {
        return Err(AppError::Cancelled);
    }
    if scan.files.is_empty() {
        let skipped = scan.skipped.total();
        return Err(AppError::Invalid(if skipped > 0 {
            format!(
                "We could not find any text we can read there. {skipped} files were skipped because they are not plain text."
            )
        } else {
            "We could not find any files there.".to_string()
        }));
    }

    let layout = detect_lanes(&scan, mode);
    let split = store.dataset_split(id)?;
    let plan = plan_split(&layout, &split).map_err(from_data_error)?;

    let dest = paths.datasets().join(dataset_dir_name(id, name));
    if plan.in_place.is_none() && dest.exists() {
        // A previous build of this same dataset; it lives in our own folder, so it is safe to replace.
        std::fs::remove_dir_all(&dest)?;
    }
    let report = materialize(&plan, &dest, cancel, &mut |p| job.progress(p)).map_err(from_data_error)?;

    let mut lanes = plan.lane_infos();
    fill_sample_prompts(&mut lanes, &report.dest);
    let root = if report.in_place {
        DatasetRoot { root_rel: None, root_abs: Some(report.dest.to_string_lossy().to_string()) }
    } else {
        DatasetRoot { root_rel: Some(format!("datasets/{}", dataset_dir_name(id, name))), root_abs: None }
    };
    let mut warnings = scan.warnings;
    warnings.extend(plan.warnings);
    if report.in_place {
        warnings.push("This folder is already in the shape the trainer reads, so it is used where it is.".into());
    }
    Ok(DatasetFinal {
        root,
        train_bytes: report.train_bytes,
        val_bytes: report.val_bytes,
        lanes,
        skipped: scan.skipped,
        warnings,
    })
}

/// Run `prepare_blocking` as a job and record the result on the dataset.
async fn run_prepare(
    state: &AppState,
    id: i64,
    name: String,
    sources: Vec<PathBuf>,
    mode: LaneMode,
    emit: Emit,
) -> AppResult<DatasetDetail> {
    let job = Job::start(state, "prepare", format!("dataset:{id}"), emit)?;
    state.store.set_dataset_status(id, DatasetStatus::Scanning, None)?;
    let (paths, store) = (state.paths.clone(), state.store.clone());
    let job = Arc::new(job);
    let worker = job.clone();
    let built = tauri::async_runtime::spawn_blocking(move || {
        prepare_blocking(&paths, &store, id, &name, &sources, mode, &worker)
    })
    .await
    .map_err(|e| AppError::Engine(e.to_string()))
    .and_then(|r| r);
    finish_dataset(state, id, built, job)
}

/// Store the outcome of preparing a dataset (ready, or an error the user can read) and close the job.
fn finish_dataset(
    state: &AppState,
    id: i64,
    built: AppResult<DatasetFinal>,
    job: Arc<Job>,
) -> AppResult<DatasetDetail> {
    let outcome = match built {
        Ok(fin) => state.store.finish_dataset(id, fin).map_err(AppError::from),
        Err(e) => {
            let msg = if e == AppError::Cancelled { None } else { Some(e.to_string()) };
            let _ = state.store.set_dataset_status(id, DatasetStatus::Error, msg);
            Err(e)
        }
    };
    if let Ok(job) = Arc::try_unwrap(job) {
        job.finish(&outcome);
    }
    outcome
}

pub async fn add_folders(state: &AppState, paths: Vec<String>, mode: LaneMode, emit: Emit) -> AppResult<DatasetDetail> {
    if paths.is_empty() {
        return Err(AppError::Invalid("Choose a folder or some files first.".into()));
    }
    let sources: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    if let Some(missing) = sources.iter().find(|p| !p.exists()) {
        return Err(AppError::NotFound(format!("{}", missing.display())));
    }
    let name = match sources.as_slice() {
        [one] => one.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "My text".into()),
        many => format!("{} folders", many.len()),
    };
    let ds = state.store.create_dataset(NewDataset {
        name: name.clone(),
        kind: DatasetKind::Linked,
        starter_id: None,
        split: SplitConfig::default(),
        license: None,
        attribution: None,
        source_url: None,
    })?;
    state.store.set_dataset_sources(ds.id, paths)?;
    run_prepare(state, ds.id, name, sources, mode, emit).await
}

/// Rebuild a dataset from its original folders (after files changed) with its current split settings.
pub async fn rebuild(state: &AppState, id: i64, split: Option<SplitConfig>, emit: Emit) -> AppResult<DatasetDetail> {
    let detail = state.store.get_dataset(id)?;
    if detail.summary.kind != DatasetKind::Linked {
        return Err(AppError::Invalid("Only datasets made from your own folders can be rebuilt.".into()));
    }
    if let Some(split) = split {
        state.store.set_dataset_split(id, split)?;
    }
    let sources: Vec<PathBuf> = state.store.dataset_sources(id)?.into_iter().map(PathBuf::from).collect();
    run_prepare(state, id, detail.summary.name, sources, LaneMode::Auto, emit).await
}

pub async fn install_starter(state: &AppState, starter_id: &str, emit: Emit) -> AppResult<DatasetDetail> {
    let starter = find_starter(starter_id).ok_or_else(|| AppError::NotFound(format!("starter {starter_id}")))?;
    // Installing the same starter twice returns the one that is already there.
    if let Some(existing) = state
        .store
        .list_datasets()?
        .into_iter()
        .find(|d| d.starter_id.as_deref() == Some(starter_id) && d.status == DatasetStatus::Ready)
    {
        return Ok(state.store.get_dataset(existing.id)?);
    }
    let info = starter.info.clone();
    let ds = state.store.create_dataset(NewDataset {
        name: info.title.clone(),
        kind: starter.kind(),
        starter_id: Some(starter_id.to_string()),
        split: SplitConfig::default(),
        license: Some(info.license.clone()),
        attribution: Some(info.attribution.clone()),
        source_url: Some(info.source_url.clone()),
    })?;
    state.store.set_dataset_status(ds.id, DatasetStatus::Preparing, None)?;
    let job = Job::start(state, "download", format!("dataset:{}", ds.id), emit)?;
    let dest = state.paths.datasets().join(dataset_dir_name(ds.id, &info.title));

    let built = async {
        let client = http_client().map_err(from_data_error)?;
        let report = minagi_data::install_starter(&starter, &dest, &client, job.ctl.token.clone(), |p| job.progress(p))
            .await
            .map_err(from_data_error)?;
        let mut lanes = lane_infos_from_report(&report.lanes);
        fill_sample_prompts(&mut lanes, &dest);
        Ok::<_, AppError>(DatasetFinal {
            root: DatasetRoot {
                root_rel: Some(format!("datasets/{}", dataset_dir_name(ds.id, &info.title))),
                root_abs: None,
            },
            train_bytes: report.train_bytes,
            val_bytes: report.val_bytes,
            lanes,
            skipped: SkippedSummary::default(),
            warnings: vec![],
        })
    }
    .await;
    finish_dataset(state, ds.id, built, Arc::new(job))
}

pub async fn generate_arithmetic(state: &AppState, params: ArithmeticParams, emit: Emit) -> AppResult<DatasetDetail> {
    let ds = state.store.create_dataset(NewDataset {
        name: "Arithmetic practice".into(),
        kind: DatasetKind::Generated,
        starter_id: None,
        split: SplitConfig::default(),
        license: None,
        attribution: Some("Generated on this computer".into()),
        source_url: None,
    })?;
    state.store.set_dataset_status(ds.id, DatasetStatus::Preparing, None)?;
    let job = Arc::new(Job::start(state, "generate", format!("dataset:{}", ds.id), emit)?);
    let dest = state.paths.datasets().join(dataset_dir_name(ds.id, "Arithmetic practice"));
    let worker = job.clone();
    let rel = format!("datasets/{}", dataset_dir_name(ds.id, "Arithmetic practice"));
    let built = tauri::async_runtime::spawn_blocking(move || {
        let report =
            minagi_data::synth::arithmetic::generate(&params, &dest, &worker.ctl.flag, &mut |p| worker.progress(p))
                .map_err(from_data_error)?;
        let mut lanes = vec![LaneInfo {
            name: "arithmetic".into(),
            display_name: "arithmetic".into(),
            color_slot: 1,
            enabled: true,
            n_files: Count(report.train_shards as u64),
            train_bytes: Count(report.train_bytes),
            val_files: Count(report.val_shards as u64),
            val_bytes: Count(report.val_bytes),
            sample_prompt: None,
        }];
        fill_sample_prompts(&mut lanes, &dest);
        Ok(DatasetFinal {
            root: DatasetRoot { root_rel: Some(rel), root_abs: None },
            train_bytes: report.train_bytes,
            val_bytes: report.val_bytes,
            lanes,
            skipped: SkippedSummary::default(),
            warnings: vec![],
        })
    })
    .await
    .map_err(|e| AppError::Engine(e.to_string()))
    .and_then(|r| r);
    finish_dataset(state, ds.id, built, job)
}

/// Remove a dataset. Files are only deleted when they are inside the app's own datasets folder.
pub fn delete(state: &AppState, id: i64, delete_files: bool) -> AppResult<()> {
    let root = state.store.dataset_root(id)?;
    state.store.delete_dataset(id)?;
    if delete_files && let Some(rel) = root.root_rel {
        let dir = state.paths.resolve(&rel);
        if dir.starts_with(state.paths.datasets()) {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}

/// Lanes that will be read for a dataset, or `None` to read all of them.
pub fn enabled_lanes(detail: &DatasetDetail) -> Option<Vec<String>> {
    if detail.lanes.iter().all(|l| l.enabled) {
        None
    } else {
        Some(detail.lanes.iter().filter(|l| l.enabled).map(|l| l.name.clone()).collect())
    }
}

/// Characters in one pass over the lanes that will be read.
pub fn enabled_train_bytes(detail: &DatasetDetail) -> u64 {
    detail.lanes.iter().filter(|l| l.enabled).map(|l| l.train_bytes.0).sum()
}
