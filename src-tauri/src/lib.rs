//! LLM Trainer desktop app: Tauri shell, commands, and the recorder that connects the engine to the database and UI.

mod api;
mod chat;
mod datasets;
mod export;
mod fork;
mod hardware;
mod host;
mod hub;
mod power;
mod recorder;
mod state;

#[cfg(test)]
mod ipc_tests;

use std::path::PathBuf;
use std::sync::Arc;

use minagi_mock::MockFactory;
use minagi_store::Store;
use minagi_types::EngineFactory;
use tauri::{Emitter, Listener, Manager};
use tauri_specta::{Builder, ErrorHandlingMode, collect_commands};

use crate::state::{AppPaths, AppState};

/// Emitted to the UI when the user tries to close the window while a training is running.
pub const CLOSE_REQUESTED_EVENT: &str = "close-requested";
/// The UI emits this once any running training has been stopped and saved; the app then exits.
pub const QUIT_NOW_EVENT: &str = "quit-now";

/// Every command, registered once so the app and the bindings export agree.
///
/// 64-bit integers are exported to TypeScript as `number`. That is lossless here: row ids and counters never come
/// near 2^53 (the counters that could grow large use the `Step`/`Chars`/`UnixMs` newtypes in `minagi-types`).
pub fn specta_builder<R: tauri::Runtime>() -> Builder<R> {
    Builder::<R>::new()
        .error_handling(ErrorHandlingMode::Throw)
        .dangerously_cast_bigints_to_number()
        // JSON has no NaN: the engine's "invalid number" arrives as null. The generated bindings turn it back into NaN
        // so the UI works with plain `number` and treats NaN as "diverged".
        .semantic_types(specta_typescript::semantic::Configuration::default().enable_lossless_floats())
        .commands(collect_commands![
            api::system::app_info,
            api::system::hardware_info,
            api::system::preset_configs,
            api::system::recommend_preset,
            api::system::estimate_run,
            api::system::storage_usage,
            api::runs::create_run,
            api::runs::fork_run,
            api::runs::start_run,
            api::runs::pause_run,
            api::runs::resume_run,
            api::runs::stop_run,
            api::runs::checkpoint_now,
            api::runs::sample_now,
            api::runs::eval_now,
            api::runs::list_runs,
            api::runs::get_run,
            api::runs::get_run_config,
            api::runs::export_model,
            api::runs::export_safetensors,
            api::runs::import_model,
            api::runs::preview_import,
            api::runs::get_active_run,
            api::runs::rename_run,
            api::runs::delete_run,
            api::runs::subscribe_live,
            api::chat::chat_open,
            api::chat::chat_resume,
            api::chat::chat_send,
            api::chat::chat_stop,
            api::chat::chat_reset,
            api::chat::chat_set_learn,
            api::chat::chat_save_adapted,
            api::chat::chat_close,
            api::chat::chat_list_sessions,
            api::chat::chat_get_messages,
            api::chat::chat_delete_session,
            api::datasets::list_datasets,
            api::datasets::get_dataset,
            api::datasets::list_starters,
            api::datasets::probe_paths,
            api::datasets::add_folders,
            api::datasets::rebuild_dataset,
            api::datasets::get_split,
            api::datasets::update_lane,
            api::datasets::preview_text,
            api::datasets::install_starter,
            api::datasets::generate_arithmetic,
            api::datasets::delete_dataset,
            api::datasets::list_jobs,
            api::datasets::cancel_job,
            api::metrics::get_series,
            api::metrics::get_eval_points,
            api::metrics::get_eval_domains,
            api::metrics::get_events,
            api::metrics::get_samples,
            api::metrics::list_sample_steps,
            api::metrics::get_pool_snapshot,
            api::metrics::list_pool_steps,
            api::metrics::list_checkpoints,
        ])
}

/// Choose the engine: the real one (candle) unless `MINAGI_ENGINE=mock` asks for the simulation, or this build left
/// the real engine out. The UI shows an unmistakable "simulated" badge whenever the simulation is running.
fn make_factory() -> Arc<dyn EngineFactory> {
    let want_mock = std::env::var("MINAGI_ENGINE").is_ok_and(|v| v.eq_ignore_ascii_case("mock"));
    #[cfg(feature = "real-engine")]
    if !want_mock {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        return Arc::new(minagi_core::RealFactory::new(sys.total_memory() as f64 / 1_073_741_824.0));
    }
    let _ = want_mock;
    Arc::new(MockFactory::from_env())
}

/// Development only: `MINAGI_DEV_SCRIPT=<file.js>` runs that script in the main window a moment after start, so a
/// whole workflow (click through the welcome flow, say) can be driven and screenshotted without a human.
#[cfg(debug_assertions)]
fn run_dev_script(app: &tauri::App) {
    let Some(path) = std::env::var_os("MINAGI_DEV_SCRIPT") else { return };
    let Ok(script) = std::fs::read_to_string(&path) else {
        eprintln!("[dev] could not read {}", path.to_string_lossy());
        return;
    };
    // the script can report what it sees: `__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "dev-report", payload })`
    app.listen("dev-report", |e| {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/minagi-dev-report.log") {
            let _ = writeln!(f, "{}", e.payload());
        }
    });
    let handle = app.handle().clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(3));
        if let Some(w) = handle.get_webview_window("main") {
            let _ = w.eval(&script);
        }
    });
}

/// Data directory: `MINAGI_HOME` when set (useful for development), otherwise the OS app-data folder.
fn data_dir(app: &tauri::App) -> PathBuf {
    std::env::var_os("MINAGI_HOME")
        .map(PathBuf::from)
        .or_else(|| app.path().app_local_data_dir().ok())
        .unwrap_or_else(|| std::env::temp_dir().join("llm-trainer"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = specta_builder::<tauri::Wry>();

    #[cfg(debug_assertions)]
    builder
        .export(
            specta_typescript::Typescript::default()
                .header("// @generated by tauri-specta from the Rust types. Do not edit."),
            "../ui/src/bindings.ts",
        )
        .expect("failed to export TypeScript bindings");

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // A second launch just brings the existing window forward.
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .on_window_event(|window, event| {
            // Closing the window mid-training would end the run. Ask the UI first, and keep the window open.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && let Some(state) = window.try_state::<AppState>()
                && host::active_run_id(&state).is_some()
            {
                api.prevent_close();
                let _ = window.emit(CLOSE_REQUESTED_EVENT, ());
            }
        })
        .invoke_handler(builder.invoke_handler())
        .setup(move |app| {
            builder.mount_events(app);
            let handle = app.handle().clone();
            app.listen(QUIT_NOW_EVENT, move |_| handle.exit(0));

            let paths = AppPaths::new(data_dir(app));
            paths.ensure()?;
            let store = Store::open(&paths.db())?;
            // A run still marked active was cut off by a close or a crash.
            let interrupted = store.mark_interrupted_on_startup()?;
            store.fail_interrupted_jobs()?;
            if !interrupted.is_empty() {
                eprintln!("[startup] marked {} unfinished run(s) as interrupted", interrupted.len());
            }
            let state = AppState::new(paths, store, make_factory());
            {
                use tauri_plugin_notification::NotificationExt;
                let handle = app.handle().clone();
                let notifier: state::Notifier = Arc::new(move |title, body| {
                    let _ = handle.notification().builder().title(title).body(body).show();
                });
                let _ = state.notifier.set(notifier);
            }
            app.manage(state);
            #[cfg(debug_assertions)]
            run_dev_script(app);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running LLM Trainer");
}

#[cfg(test)]
mod tests {
    /// Regenerates `ui/src/bindings.ts`. CI runs this and fails if the committed file differs.
    #[test]
    fn export_bindings() {
        super::specta_builder::<tauri::Wry>()
            .export(
                specta_typescript::Typescript::default()
                    .header("// @generated by tauri-specta from the Rust types. Do not edit."),
                "../ui/src/bindings.ts",
            )
            .expect("failed to export TypeScript bindings");
    }
}
