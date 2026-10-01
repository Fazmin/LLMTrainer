//! Portable export and import of a trained model.
//!
//! A portable folder holds the engine's checkpoint untouched (`model/`), a versioned manifest the app can validate
//! (`llm-trainer-export.json`) and a plain-English model card (`README.md`). The app never looks inside `model/`: only
//! the engine that wrote it can read it, which the manifest records and import enforces.

use std::path::{Path, PathBuf};

use minagi_store::{NewRun, RunOrigin, RunProgress, slugify};
use minagi_types::{
    AppError, AppResult, Chars, CheckpointKind, CheckpointMeta, Count, ExportManifest, ExportRequest, ExportResult,
    ImportIssue, ImportKind, ImportPreview, ImportSeverity, JobProgress, RunState, RunSummary, Step, UnixMs,
};

use crate::datasets::{Emit, Job};
use crate::state::AppState;

pub const MANIFEST_FILE: &str = "llm-trainer-export.json";
pub const FORMAT: u32 = 1;

fn walk_files(dir: &Path, out: &mut Vec<(PathBuf, u64)>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            walk_files(&entry.path(), out)?;
        } else if ty.is_file() {
            out.push((entry.path(), entry.metadata()?.len()));
        }
    }
    Ok(())
}

/// Copy a directory tree, reporting progress, and stop early when cancelled.
fn copy_tree(src: &Path, dst: &Path, job: &Job, label: &str) -> AppResult<(u64, u64)> {
    let mut files = Vec::new();
    walk_files(src, &mut files)?;
    let total: u64 = files.iter().map(|(_, n)| n).sum();
    let mut done = 0u64;
    for (path, size) in &files {
        if job.ctl.flag.is_cancelled() {
            return Err(AppError::Cancelled);
        }
        let rel = path.strip_prefix(src).map_err(|e| AppError::Io(e.to_string()))?;
        let target = dst.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(path, &target)?;
        done += size;
        job.progress(JobProgress {
            done: done as f64,
            total: Some(total as f64),
            unit: "bytes".into(),
            message: label.to_string(),
            bytes_per_sec: None,
            eta_seconds: None,
        });
    }
    Ok((total, files.len() as u64))
}

fn model_card(m: &ExportManifest) -> String {
    let bits = m.heldout_nats.map(|n| format!("{:.2} bits per character", n / std::f64::consts::LN_2));
    format!(
        "# {name}\n\nA small language model trained with LLM Trainer.\n\n- Size: {preset:?}\n- Text read while training: {chars} characters\n- Test score: {score}\n- Experts in its pool: {experts}\n- Made by: LLM Trainer {version} (engine {engine})\n\n## What is in this folder\n\n- `model/` holds the saved model exactly as the trainer wrote it. Only the same engine can read it.\n- `{manifest}` describes the model for LLM Trainer's Import button.\n\n## Using it\n\nOpen LLM Trainer, go to Runs, and choose Import a model. Pick this folder. You can then chat with the model or continue training it.\n\nThe model learns by reading text one character at a time. A lower test score means it predicts new text better.\n",
        name = m.name,
        preset = m.preset,
        chars = m.chars.0,
        score = bits.unwrap_or_else(|| "not measured yet".into()),
        experts = m.n_experts,
        version = m.app_version,
        engine = m.engine_name,
        manifest = MANIFEST_FILE,
    )
}

/// The run, the save to export, where its files are, and the folder to export into.
struct Chosen {
    run: minagi_types::RunSummary,
    ckpt: minagi_types::CheckpointInfo,
    src: PathBuf,
    dest_dir: PathBuf,
}

fn choose(state: &AppState, req: &ExportRequest) -> AppResult<Chosen> {
    let run = state.store.get_run(req.run_id)?;
    let ckpts = state.store.list_checkpoints(req.run_id)?;
    let ckpt = match req.checkpoint_id {
        Some(id) => ckpts.iter().find(|c| c.id == id).ok_or_else(|| AppError::NotFound(format!("save {id}")))?,
        None => ckpts
            .iter()
            .find(|c| c.is_best)
            .or_else(|| ckpts.first())
            .ok_or_else(|| AppError::Invalid("This run has no saved progress to export yet.".into()))?,
    }
    .clone();
    let dest_dir = PathBuf::from(&req.dest_dir);
    if !dest_dir.is_dir() {
        return Err(AppError::NotFound("the folder you chose".into()));
    }
    let src = state.paths.resolve(&state.store.run_dir_rel(req.run_id)?).join(&ckpt.path);
    if !src.join("COMPLETE").exists() {
        return Err(AppError::Checkpoint("this save is incomplete".into()));
    }
    Ok(Chosen { run, ckpt, src, dest_dir })
}

/// A folder name inside `dest_dir` that does not exist yet.
fn fresh_target(dest_dir: &Path, folder: &str) -> PathBuf {
    let mut target = dest_dir.join(folder);
    let mut n = 2;
    while target.exists() {
        target = dest_dir.join(format!("{folder}-{n}"));
        n += 1;
    }
    target
}

/// Export a run's save as a portable folder inside `req.dest_dir`.
pub async fn export_model(state: &AppState, req: ExportRequest, emit: Emit) -> AppResult<ExportResult> {
    let Chosen { run, ckpt, src, dest_dir } = choose(state, &req)?;
    let (model, train) = state.store.run_config(req.run_id)?;
    let manifest = ExportManifest {
        format: FORMAT,
        app_version: env!("CARGO_PKG_VERSION").into(),
        engine_name: state.factory.name().into(),
        engine_version: state.factory.version(),
        engine_format: 0,
        name: run.name.clone(),
        preset: run.preset,
        model,
        train,
        step: ckpt.step,
        chars: ckpt.chars,
        heldout_nats: ckpt.heldout_nats,
        n_experts: ckpt.n_experts.unwrap_or(0),
        created_at: UnixMs::now(),
    };

    let target = fresh_target(&dest_dir, &format!("{}-step-{}", slugify(&run.name), ckpt.step.0));
    let tmp = dest_dir.join(format!(".{}.exporting", target.file_name().unwrap_or_default().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&tmp);

    let job = Job::start(state, "export", format!("run:{}", req.run_id), emit)?;
    let result = (|| {
        std::fs::create_dir_all(&tmp)?;
        let (bytes, files) = copy_tree(&src, &tmp.join("model"), &job, "Copying the saved model")?;
        std::fs::write(
            tmp.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).map_err(|e| AppError::Io(e.to_string()))?,
        )?;
        std::fs::write(tmp.join("README.md"), model_card(&manifest))?;
        std::fs::rename(&tmp, &target)?;
        Ok(ExportResult { path: target.to_string_lossy().to_string(), bytes: Count(bytes), files: Count(files + 2) })
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    job.finish(&result);
    result
}

/// Export a run's save for other tools: `model.safetensors`, the model's settings and a note on the tensor names.
pub async fn export_safetensors(state: &AppState, req: ExportRequest, emit: Emit) -> AppResult<ExportResult> {
    let Chosen { run, ckpt, src, dest_dir } = choose(state, &req)?;
    let (model, _) = state.store.run_config(req.run_id)?;
    let target = fresh_target(&dest_dir, &format!("{}-step-{}-safetensors", slugify(&run.name), ckpt.step.0));
    let tmp = dest_dir.join(format!(".{}.exporting", target.file_name().unwrap_or_default().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&tmp);

    let job = Job::start(state, "export", format!("run:{}", req.run_id), emit)?;
    job.progress(JobProgress {
        done: 0.0,
        total: None,
        unit: "bytes".into(),
        message: "Writing the model for other tools".into(),
        bytes_per_sec: None,
        eta_seconds: None,
    });
    let result = (|| {
        std::fs::create_dir_all(&tmp)?;
        let weights = state.factory.export_safetensors(&src, &tmp)?;
        let config = serde_json::json!({
            "architecture": "mini-agi recurrent adaptive-depth transformer with a pool of experts",
            "model": model,
            "step": ckpt.step.0,
            "charactersRead": ckpt.chars.0,
            "tokenizer": { "type": "bytes", "vocabSize": model.vocab_size, "markers": ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>", "<g>", "</g>", "<|endoftext|>"], "markerBase": 256 },
        });
        std::fs::write(
            tmp.join("config.json"),
            serde_json::to_vec_pretty(&config).map_err(|e| AppError::Io(e.to_string()))?,
        )?;
        std::fs::write(tmp.join("README.md"), safetensors_card(&run.name, ckpt.step.0, ckpt.n_experts.unwrap_or(0)))?;
        std::fs::rename(&tmp, &target)?;
        Ok(ExportResult { path: target.to_string_lossy().to_string(), bytes: Count(weights), files: Count(3) })
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    job.finish(&result);
    result
}

fn safetensors_card(name: &str, step: u64, experts: u32) -> String {
    format!(
        "# {name}\n\nA small language model trained with LLM Trainer, saved as a safetensors file for use in other tools. \
It reads and writes raw bytes (265 symbols: 256 bytes plus 9 marker tokens).\n\n\
- `model.safetensors`: every weight as 32-bit floats (step {step}, {experts} experts in the pool).\n\
- `config.json`: the model's shape and how its text is split into symbols.\n\n\
## Tensor names\n\n\
- `tok_emb.weight`: the byte embedding, also used as the output layer.\n\
- `prelude.N.*` and `recur.0.*`: attention and normalisation weights (`attn.qkv.weight` is query, key and value stacked); `prelude` blocks also have `mlp.w1/w3/w2.weight`.\n\
- `adapter.weight`: mixes the running state with the embedded input at every pass.\n\
- `halt.weight`, `halt.bias`: the head that decides when a character stops thinking.\n\
- `recur.0.mlp.router.weight` (one row per expert), `recur.0.mlp.depth_emb`, `pool.gate`: how characters are sent to experts.\n\
- `pool.experts.<id>.w1`, `.w3`, `.w2`: one expert's feed-forward weights; `<id>` is the expert's permanent number.\n\n\
The model is recurrent: one block is applied up to several times per character, and each character decides for itself how many times. A generic transformer loader will not run it unchanged.\n"
    )
}

/// Read and check the manifest of a folder the user picked.
pub fn read_manifest(state: &AppState, folder: &Path) -> AppResult<ExportManifest> {
    let bytes = std::fs::read(folder.join(MANIFEST_FILE)).map_err(|_| {
        AppError::Invalid("That folder is not an export from LLM Trainer (its description file is missing).".into())
    })?;
    let m: ExportManifest = serde_json::from_slice(&bytes)
        .map_err(|_| AppError::Checkpoint("its description file could not be read".into()))?;
    if m.format != FORMAT {
        return Err(AppError::Checkpoint(format!(
            "it uses export format {}, but this version understands {FORMAT}",
            m.format
        )));
    }
    if m.engine_name != state.factory.name() {
        return Err(AppError::Invalid(format!(
            "That model was made by the {} engine, but this app is using the {} engine, so it cannot be opened here.",
            m.engine_name,
            state.factory.name()
        )));
    }
    if !folder.join("model").join("COMPLETE").exists() {
        return Err(AppError::Checkpoint("the model files are incomplete".into()));
    }
    Ok(m)
}

/// What importing `folder` would do: a folder made by Export, or the weights folder of the original Python program.
pub fn preview_import(state: &AppState, folder: &str) -> AppResult<ImportPreview> {
    let folder = PathBuf::from(folder);
    if folder.join(MANIFEST_FILE).is_file() {
        let path = folder.to_string_lossy().into_owned();
        return Ok(match read_manifest(state, &folder) {
            Ok(m) => ImportPreview {
                path,
                kind: ImportKind::Portable,
                importable: true,
                summary: format!(
                    "{}: a {} model with {} experts, after reading {} characters.",
                    m.name,
                    m.preset.display_name(),
                    m.n_experts,
                    plain_count(m.chars.0)
                ),
                issues: Vec::new(),
                name: format!("{} (imported)", m.name),
                preset: m.preset,
                model: Some(m.model),
                step: m.step,
                chars: m.chars,
                heldout_nats: m.heldout_nats,
                n_experts: m.n_experts,
                bytes: Count(dir_bytes(&folder)),
            },
            Err(e) => ImportPreview {
                path,
                kind: ImportKind::Portable,
                importable: false,
                summary: "This folder was made by LLM Trainer, but it cannot be opened here.".into(),
                issues: vec![ImportIssue { severity: ImportSeverity::Blocker, message: e.to_string() }],
                name: String::new(),
                preset: minagi_types::Preset::Custom,
                model: None,
                step: Step(0),
                chars: Chars(0),
                heldout_nats: None,
                n_experts: 0,
                bytes: Count(0),
            },
        });
    }
    if folder.join("manifest.json").is_file() {
        return state.factory.inspect_python(&folder);
    }
    Err(AppError::Invalid(
        "That folder does not look like a saved model. Choose a folder made by Export, or the weights folder of the original mini-AGI program."
            .into(),
    ))
}

/// "3.0 million", "250 thousand", "812".
fn plain_count(n: u64) -> String {
    match n {
        1_000_000.. => format!("{:.1} million", n as f64 / 1e6),
        1_000.. => format!("{} thousand", n / 1_000),
        _ => n.to_string(),
    }
}

fn dir_bytes(dir: &Path) -> u64 {
    let mut files = Vec::new();
    let _ = walk_files(dir, &mut files);
    files.iter().map(|(_, n)| n).sum()
}

/// Bring in the weights folder of the original Python program as a new run.
async fn import_python_model(state: &AppState, folder: &Path, emit: Emit) -> AppResult<RunSummary> {
    let preview = state.factory.inspect_python(folder)?;
    if !preview.importable {
        let reasons: Vec<String> = preview
            .issues
            .iter()
            .filter(|i| i.severity == ImportSeverity::Blocker)
            .map(|i| i.message.clone())
            .collect();
        return Err(AppError::Checkpoint(format!("This model cannot be imported: {}", reasons.join(" "))));
    }
    let model = preview.model.clone().ok_or_else(|| AppError::Checkpoint("its shape could not be read".into()))?;
    model.validate().map_err(AppError::Invalid)?;
    let mut train = preview.preset.train();
    // An imported model keeps its own shape: fit the training settings to the longest context it was built for.
    let block = model.block;
    train.chunk = train.chunk.min(block);
    train.context_end = train.context_end.min(block);
    train.context_start = train.context_start.min(train.context_end).max(train.chunk.min(train.context_end));
    train.passage = train.passage.max(train.chunk);
    train.resident = train.resident.max(model.pool_top_k);
    train.validate(&model).map_err(AppError::Invalid)?;

    let run = state.store.create_run(NewRun {
        name: preview.name.clone(),
        preset: preview.preset,
        dataset_id: None,
        dataset_snapshot: serde_json::json!({}),
        model,
        train,
        goal: None,
        engine_version: state.factory.version(),
        app_version: env!("CARGO_PKG_VERSION").into(),
        origin: RunOrigin::Imported,
        parent_run_id: None,
        backend: None,
        chars_total: None,
    })?;
    let run_dir = state.paths.resolve(&state.store.run_dir_rel(run.id)?);
    let rel = format!("checkpoints/step-{:09}", preview.step.0);
    let dest = run_dir.join(&rel);
    let job = Job::start(state, "import", format!("run:{}", run.id), emit)?;
    job.progress(JobProgress {
        done: 0.0,
        total: None,
        unit: "bytes".into(),
        message: "Converting the original model".into(),
        bytes_per_sec: None,
        eta_seconds: None,
    });
    let converted = (|| {
        std::fs::create_dir_all(run_dir.join("checkpoints"))?;
        state.factory.import_python(folder, &dest)
    })();
    let outcome = converted.and_then(|p| {
        state
            .store
            .record_checkpoint(
                run.id,
                CheckpointMeta {
                    step: p.step,
                    chars: p.chars,
                    kind: CheckpointKind::Imported,
                    path: rel,
                    bytes: p.bytes,
                    heldout_nats: p.heldout_nats,
                    n_experts: p.n_experts,
                    engine_format: 1,
                },
            )
            .map(|_| p)
            .map_err(AppError::from)
    });
    let p = match outcome {
        Ok(p) => p,
        Err(e) => {
            // Do not leave a half-made run behind.
            let _ = state.store.delete_run(run.id);
            let _ = std::fs::remove_dir_all(&run_dir);
            job.finish::<()>(&Err(e.clone()));
            return Err(e);
        }
    };
    state.store.update_run_progress(RunProgress {
        run_id: run.id,
        step: p.step.0,
        chars_read: p.chars.0,
        last_heldout_nats: p.heldout_nats,
        n_experts: Some(p.n_experts),
        ..Default::default()
    });
    state.store.flush();
    let done = state.store.set_run_state(run.id, RunState::Imported, None)?;
    job.finish(&Ok::<(), AppError>(()));
    Ok(done)
}

/// Bring an exported folder (or the original program's weights folder) in as a run you can chat with or continue.
pub async fn import_model(state: &AppState, folder: String, emit: Emit) -> AppResult<RunSummary> {
    let folder = PathBuf::from(folder);
    if !folder.join(MANIFEST_FILE).is_file() && folder.join("manifest.json").is_file() {
        return import_python_model(state, &folder, emit).await;
    }
    let m = read_manifest(state, &folder)?;
    let (model, train) = (m.model.clone(), m.train.clone());
    model.validate().map_err(AppError::Invalid)?;
    train.validate(&model).map_err(AppError::Invalid)?;

    let run = state.store.create_run(NewRun {
        name: format!("{} (imported)", m.name),
        preset: m.preset,
        dataset_id: None,
        dataset_snapshot: serde_json::json!({}),
        model,
        train,
        goal: None,
        engine_version: m.engine_version.clone(),
        app_version: env!("CARGO_PKG_VERSION").into(),
        origin: RunOrigin::Imported,
        parent_run_id: None,
        backend: None,
        chars_total: None,
    })?;
    let run_dir = state.paths.resolve(&state.store.run_dir_rel(run.id)?);
    let rel = format!("checkpoints/step-{:09}", m.step.0);
    let dest = run_dir.join(&rel);

    let job = Job::start(state, "import", format!("run:{}", run.id), emit)?;
    let copied = (|| {
        let tmp = run_dir.join("checkpoints").join(format!(".tmp-import-{}", m.step.0));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp)?;
        let (bytes, _) = copy_tree(&folder.join("model"), &tmp, &job, "Copying the model in")?;
        std::fs::rename(&tmp, &dest)?;
        Ok(bytes)
    })();
    let outcome = match copied {
        Ok(bytes) => state
            .store
            .record_checkpoint(
                run.id,
                CheckpointMeta {
                    step: Step(m.step.0),
                    chars: Chars(m.chars.0),
                    kind: CheckpointKind::Imported,
                    path: rel,
                    bytes: Count(bytes),
                    heldout_nats: m.heldout_nats,
                    n_experts: m.n_experts,
                    engine_format: m.engine_format,
                },
            )
            .map(|_| ())
            .map_err(AppError::from),
        Err(e) => Err(e),
    };
    if let Err(e) = outcome {
        // Do not leave a half-made run behind.
        let _ = state.store.delete_run(run.id);
        let _ = std::fs::remove_dir_all(&run_dir);
        job.finish::<()>(&Err(e.clone()));
        return Err(e);
    }
    state.store.update_run_progress(RunProgress {
        run_id: run.id,
        step: m.step.0,
        chars_read: m.chars.0,
        last_heldout_nats: m.heldout_nats,
        n_experts: Some(m.n_experts),
        ..Default::default()
    });
    state.store.flush();
    let done = state.store.set_run_state(run.id, RunState::Imported, None)?;
    job.finish(&Ok::<(), AppError>(()));
    Ok(done)
}
