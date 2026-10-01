//! Continuing a saved model as a new run.
//!
//! A fork is a new run whose history starts at a save of an older one. The save's files are *linked* into the new run
//! (hard links where the disk allows, copies otherwise), so forking a large model takes no extra space and the original is
//! never touched: the engine only ever replaces files, it never rewrites them in place.

use std::path::Path;

use minagi_store::{NewRun, RunOrigin, RunProgress};
use minagi_types::{AppError, AppResult, CheckpointKind, CheckpointMeta, Count, ForkRunRequest, RunSummary};

use crate::state::AppState;

/// Link (or copy) every file of `src` into `dst`, creating directories as needed.
fn link_tree(src: &Path, dst: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(dst)?;
    let mut bytes = 0;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            bytes += link_tree(&entry.path(), &target)?;
        } else {
            bytes += entry.metadata()?.len();
            if std::fs::hard_link(entry.path(), &target).is_err() {
                std::fs::copy(entry.path(), &target)?;
            }
        }
    }
    Ok(bytes)
}

pub fn fork_run(state: &AppState, req: ForkRunRequest) -> AppResult<RunSummary> {
    let parent = state.store.get_run(req.run_id)?;
    if parent.status.is_active() {
        return Err(AppError::Invalid("Stop this run before continuing it as a new one.".into()));
    }
    let ckpts = state.store.list_checkpoints(req.run_id)?;
    let ckpt = match req.checkpoint_id {
        Some(id) => ckpts.iter().find(|c| c.id == id).ok_or_else(|| AppError::NotFound(format!("save {id}")))?,
        None => ckpts
            .iter()
            .find(|c| c.is_best)
            .or_else(|| ckpts.first())
            .ok_or_else(|| AppError::Invalid("This run has no saved progress to continue from yet.".into()))?,
    }
    .clone();
    let src = state.paths.resolve(&state.store.run_dir_rel(req.run_id)?).join(&ckpt.path);
    if !src.join("COMPLETE").exists() {
        return Err(AppError::Checkpoint("this save is incomplete".into()));
    }

    // the model's shape is fixed by the save; how it trains may change
    let (model, parent_train) = state.store.run_config(req.run_id)?;
    let train = req.train.unwrap_or(parent_train);
    train.validate(&model).map_err(AppError::Invalid)?;
    let dataset_id = req.dataset_id.or(parent.dataset_id);
    let (snapshot, chars_total) = crate::api::runs::dataset_context(state, dataset_id)?;
    let name = match req.name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) {
        Some(n) => n,
        None => format!("{} (continued)", parent.name),
    };
    let hw = state.hardware();
    let run = state.store.create_run(NewRun {
        name,
        preset: parent.preset,
        dataset_id,
        dataset_snapshot: snapshot,
        model,
        train,
        goal: req.goal,
        engine_version: state.factory.version(),
        app_version: env!("CARGO_PKG_VERSION").into(),
        origin: RunOrigin::Forked,
        parent_run_id: Some(parent.id),
        backend: hw.selected_backend().map(|b| b.name.clone()),
        chars_total,
    })?;
    let run_dir = state.paths.resolve(&state.store.run_dir_rel(run.id)?);
    let rel = format!("checkpoints/step-{:09}", ckpt.step.0);
    let dest = run_dir.join(&rel);
    let linked = (|| {
        std::fs::create_dir_all(run_dir.join("checkpoints"))?;
        let tmp = run_dir.join("checkpoints").join(format!(".tmp-fork-{}", ckpt.step.0));
        let _ = std::fs::remove_dir_all(&tmp);
        let bytes = link_tree(&src, &tmp)?;
        std::fs::rename(&tmp, &dest)?;
        Ok::<u64, std::io::Error>(bytes)
    })();
    let bytes = match linked {
        Ok(b) => b,
        Err(e) => {
            let _ = state.store.delete_run(run.id);
            let _ = std::fs::remove_dir_all(&run_dir);
            return Err(e.into());
        }
    };
    state.store.record_checkpoint(
        run.id,
        CheckpointMeta {
            step: ckpt.step,
            chars: ckpt.chars,
            kind: CheckpointKind::Manual,
            path: rel,
            bytes: Count(bytes),
            heldout_nats: ckpt.heldout_nats,
            n_experts: ckpt.n_experts.unwrap_or(0),
            engine_format: 0,
        },
    )?;
    state.store.update_run_progress(RunProgress {
        run_id: run.id,
        step: ckpt.step.0,
        chars_read: ckpt.chars.0,
        last_heldout_nats: ckpt.heldout_nats,
        n_experts: ckpt.n_experts,
        ..Default::default()
    });
    state.store.flush();
    Ok(state.store.get_run(run.id)?)
}
