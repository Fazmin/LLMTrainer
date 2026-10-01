//! Contract tests for the seam between TypeScript and Rust: the real command handlers are invoked through Tauri's IPC
//! layer (mock runtime) with the same JSON the webview sends, so argument names, result shapes and error shapes are
//! exactly what `ui/src/bindings.ts` expects.

use std::sync::Arc;
use std::time::{Duration, Instant};

use minagi_mock::{MockFactory, Scenario};
use minagi_store::Store;
use serde_json::{Value, json};
use tauri::test::{INVOKE_KEY, get_ipc_response, mock_builder, mock_context, noop_assets};
use tauri::webview::InvokeRequest;
use tauri::{Manager, WebviewWindow, ipc::CallbackFn, ipc::InvokeBody};

use crate::state::{AppPaths, AppState};

struct Harness {
    window: WebviewWindow<tauri::test::MockRuntime>,
    _dir: tempfile::TempDir,
}

fn harness(speed: f64) -> Harness {
    harness_with(Arc::new(MockFactory::new(speed, Scenario::Normal)))
}

fn harness_with(factory: Arc<dyn minagi_types::EngineFactory>) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let paths = AppPaths::new(dir.path().to_path_buf());
    paths.ensure().unwrap();
    let store = Store::open(&paths.db()).unwrap();
    let state = AppState::new(paths, store, factory);
    let builder = crate::specta_builder::<tauri::test::MockRuntime>();
    let app = mock_builder()
        .invoke_handler(builder.invoke_handler())
        .manage(state)
        .build(mock_context(noop_assets()))
        .expect("build the app");
    let window = tauri::WebviewWindowBuilder::new(&app, "main", Default::default()).build().expect("window");
    // Keep the app alive for the duration of the test by leaking it: the mock runtime has no event loop to own it.
    std::mem::forget(app);
    Harness { window, _dir: dir }
}

impl Harness {
    fn call(&self, cmd: &str, body: Value) -> Result<Value, Value> {
        let res = get_ipc_response(
            &self.window,
            InvokeRequest {
                cmd: cmd.into(),
                callback: CallbackFn(0),
                error: CallbackFn(1),
                url: "tauri://localhost".parse().unwrap(),
                body: InvokeBody::Json(body),
                headers: Default::default(),
                invoke_key: INVOKE_KEY.to_string(),
            },
        );
        res.map(|r| r.deserialize::<Value>().expect("json response"))
    }

    fn wait_status(&self, run_id: i64, want: &str) -> Value {
        let t0 = Instant::now();
        loop {
            let run = self.call("get_run", json!({ "runId": run_id })).unwrap();
            if run["status"] == want {
                return run;
            }
            assert!(t0.elapsed() < Duration::from_secs(30), "never reached {want}; at {}", run["status"]);
            std::thread::sleep(Duration::from_millis(30));
        }
    }
}

#[test]
fn presets_arrive_in_camel_case_with_all_three_sizes() {
    let h = harness(1000.0);
    let presets = h.call("preset_configs", json!({})).unwrap();
    let arr = presets.as_array().expect("array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["preset"], "tiny");
    assert_eq!(arr[0]["model"]["dModel"], 256);
    assert_eq!(arr[2]["train"]["growth"]["everyChars"], 2_000_000);
    assert_eq!(arr[0]["train"]["moeMode"], "dense_masked");
}

#[test]
fn app_info_and_hardware_describe_the_mock_engine() {
    let h = harness(1000.0);
    let info = h.call("app_info", json!({})).unwrap();
    assert_eq!(info["isMock"], true);
    assert_eq!(info["schemaVersion"], 1);
    let hw = h.call("hardware_info", json!({})).unwrap();
    assert_eq!(hw["selected"], "metal");
    assert!(hw["ramGb"].as_f64().unwrap() > 0.0);
    let rec = h.call("recommend_preset", json!({})).unwrap();
    assert!(["tiny", "small"].contains(&rec["preset"].as_str().unwrap()));
    assert_eq!(rec["fits"].as_array().unwrap().len(), 3);
}

#[test]
fn estimate_run_takes_the_preset_configuration_as_the_ui_sends_it() {
    let h = harness(1000.0);
    let presets = h.call("preset_configs", json!({})).unwrap();
    let est = h.call("estimate_run", json!({ "model": presets[0]["model"], "train": presets[0]["train"] })).unwrap();
    assert_eq!(est["fit"], "comfortable");
    assert!(est["charsPerSec"].as_f64().unwrap() > 1000.0);
}

#[test]
fn a_full_run_through_ipc_creates_trains_stops_and_serves_charts() {
    let h = harness(400.0);
    let presets = h.call("preset_configs", json!({})).unwrap();
    let mut train = presets[0]["train"].clone();
    train["sampleEveryMin"] = json!(0.05);
    train["saveEveryMin"] = json!(0.1);

    let run = h
        .call(
            "create_run",
            json!({ "req": { "name": "ipc test", "preset": "tiny", "model": presets[0]["model"], "train": train, "datasetId": null, "goal": { "type": "until_stopped" } } }),
        )
        .unwrap();
    let id = run["id"].as_i64().unwrap();
    assert_eq!(run["status"], "created");
    assert_eq!(run["name"], "ipc test");
    assert_eq!(run["goal"]["type"], "until_stopped");

    h.call("start_run", json!({ "runId": id })).unwrap();
    // Only one run at a time, reported as a typed error.
    assert_eq!(h.call("start_run", json!({ "runId": id })).unwrap_err()["kind"], "run_active");
    assert_eq!(h.call("get_active_run", json!({})).unwrap()["id"], id);

    h.wait_status(id, "running");
    std::thread::sleep(Duration::from_millis(2500));
    h.call("pause_run", json!({})).unwrap();
    h.wait_status(id, "paused");
    h.call("resume_run", json!({})).unwrap();
    h.wait_status(id, "running");
    h.call("stop_run", json!({ "save": true })).unwrap();
    let done = h.wait_status(id, "stopped");
    assert!(done["charsRead"].as_u64().unwrap() > 0);

    let series = h
        .call("get_series", json!({ "req": { "runId": id, "keys": ["train.nats", "eval.overall"], "x": "chars", "from": null, "to": null, "maxPoints": 100 } }))
        .unwrap();
    assert_eq!(series.as_array().unwrap().len(), 2);
    assert_eq!(series[0]["key"], "train.nats");
    assert!(!series[0]["x"].as_array().unwrap().is_empty(), "ticks reach the charts");

    let ckpts = h.call("list_checkpoints", json!({ "runId": id })).unwrap();
    assert!(!ckpts.as_array().unwrap().is_empty());
    assert_eq!(ckpts[0]["kind"], "stop");
    assert!(h.call("get_samples", json!({ "runId": id, "step": null })).unwrap()["items"].as_array().is_some());
    assert!(
        h.call("get_eval_domains", json!({ "runId": id })).unwrap().as_array().unwrap().contains(&json!("stories"))
    );
    assert_eq!(h.call("list_runs", json!({})).unwrap().as_array().unwrap().len(), 1);
    assert_eq!(h.call("get_active_run", json!({})).unwrap(), Value::Null);

    // Rename and delete.
    assert_eq!(
        h.call("rename_run", json!({ "runId": id, "name": "renamed", "notes": "n" })).unwrap()["name"],
        "renamed"
    );
    h.call("delete_run", json!({ "runId": id, "deleteFiles": true })).unwrap();
    assert_eq!(h.call("list_runs", json!({})).unwrap().as_array().unwrap().len(), 0);
}

#[test]
fn errors_arrive_as_typed_objects_the_ui_can_describe() {
    let h = harness(1000.0);
    assert_eq!(
        h.call("get_run", json!({ "runId": 999 })).unwrap_err(),
        json!({ "kind": "not_found", "detail": "run 999" })
    );
    assert_eq!(h.call("pause_run", json!({})).unwrap_err(), json!({ "kind": "no_active_run" }));
    let presets = h.call("preset_configs", json!({})).unwrap();
    let mut model = presets[0]["model"].clone();
    model["nHead"] = json!(3); // 256 is not divisible by 3
    let err = h
        .call("create_run", json!({ "req": { "name": null, "preset": "tiny", "model": model, "train": null, "datasetId": null, "goal": null } }))
        .unwrap_err();
    assert_eq!(err["kind"], "invalid");
    assert!(err["detail"].as_str().unwrap().contains("divisible"));
    assert_eq!(h.call("rename_run", json!({ "runId": 1, "name": "  ", "notes": "" })).unwrap_err()["kind"], "invalid");
}

#[test]
fn state_is_managed_and_the_window_exists() {
    let h = harness(1000.0);
    assert!(h.window.app_handle().try_state::<AppState>().is_some());
}

/// Train the mock briefly and stop with a save, so there is a checkpoint to chat with.
fn trained_run(h: &Harness) -> i64 {
    let presets = h.call("preset_configs", json!({})).unwrap();
    let mut train = presets[0]["train"].clone();
    train["sampleEveryMin"] = json!(0.05);
    train["saveEveryMin"] = json!(0.1);
    let run = h
        .call(
            "create_run",
            json!({ "req": { "name": "chat model", "preset": "tiny", "model": presets[0]["model"], "train": train, "datasetId": null, "goal": { "type": "until_stopped" } } }),
        )
        .unwrap();
    let id = run["id"].as_i64().unwrap();
    h.call("start_run", json!({ "runId": id })).unwrap();
    h.wait_status(id, "running");
    std::thread::sleep(Duration::from_millis(800));
    h.call("stop_run", json!({ "save": true })).unwrap();
    h.wait_status(id, "stopped");
    id
}

fn wait_messages(h: &Harness, session: i64, n: usize) -> Value {
    let t0 = Instant::now();
    loop {
        let msgs = h.call("chat_get_messages", json!({ "sessionId": session })).unwrap();
        let done = msgs.as_array().unwrap().len() >= n && msgs[n - 1]["role"] == "model";
        if done {
            return msgs;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "reply never finished: {msgs}");
        std::thread::sleep(Duration::from_millis(30));
    }
}

#[test]
fn chat_needs_a_saved_run_then_converses_learns_and_persists() {
    let h = harness(400.0);
    let presets = h.call("preset_configs", json!({})).unwrap();
    let fresh = h
        .call("create_run", json!({ "req": { "name": "never ran", "preset": "tiny", "model": presets[0]["model"], "train": null, "datasetId": null, "goal": null } }))
        .unwrap();
    let err = h
        .call("chat_open", json!({ "req": { "runId": fresh["id"], "checkpointId": null, "mode": "conversation" } }))
        .unwrap_err();
    assert_eq!(err["kind"], "invalid");
    assert!(err["detail"].as_str().unwrap().contains("no saved progress"));

    let run = trained_run(&h);
    let chat =
        h.call("chat_open", json!({ "req": { "runId": run, "checkpointId": null, "mode": "conversation" } })).unwrap();
    let id = chat["id"].as_i64().unwrap();
    assert_eq!(chat["supportsLearn"], true);
    assert_eq!(chat["learnEnabled"], false);
    assert!(chat["modelLabel"].as_str().unwrap().starts_with("chat model, after "), "{chat}");

    assert_eq!(h.call("chat_set_learn", json!({ "sessionId": id, "enabled": true })).unwrap()["learnEnabled"], true);
    let params = json!({ "maxNew": 60, "adapt": true });
    h.call(
        "chat_send",
        json!({ "sessionId": id, "text": "Hello there", "params": params, "onEvent": "__CHANNEL__:7" }),
    )
    .unwrap();
    let msgs = wait_messages(&h, id, 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "Hello there");
    assert!(!msgs[1]["content"].as_str().unwrap().is_empty());
    assert_eq!(msgs[1]["rows"].as_array().unwrap().len(), msgs[1]["content"].as_str().unwrap().chars().count());

    // Learning is recorded once the reply is finished.
    let t0 = Instant::now();
    loop {
        let m = h.call("chat_get_messages", json!({ "sessionId": id })).unwrap();
        if m[1]["learned"] == true {
            assert!(m[1]["learnNatsAfter"].as_f64().unwrap() < m[1]["learnNatsBefore"].as_f64().unwrap());
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(10), "never learned");
        std::thread::sleep(Duration::from_millis(30));
    }

    // The learned copy is saved separately from the run.
    let saved = h.call("chat_save_adapted", json!({ "sessionId": id })).unwrap();
    assert_eq!(saved["kind"], "manual");
    let listed = h.call("chat_list_sessions", json!({})).unwrap();
    assert_eq!(listed[0]["hasAdaptedCopy"], true);

    // Empty input is rejected before anything is stored.
    assert_eq!(
        h.call("chat_send", json!({ "sessionId": id, "text": "  ", "params": params, "onEvent": "__CHANNEL__:8" }))
            .unwrap_err()["kind"],
        "invalid"
    );

    h.call("chat_reset", json!({ "sessionId": id })).unwrap();
    assert!(h.call("chat_get_messages", json!({ "sessionId": id })).unwrap().as_array().unwrap().is_empty());

    // Closing unloads the model; sending then asks to reopen; resuming loads it again.
    h.call("chat_close", json!({ "sessionId": id })).unwrap();
    let err = h
        .call("chat_send", json!({ "sessionId": id, "text": "hi", "params": params, "onEvent": "__CHANNEL__:9" }))
        .unwrap_err();
    assert!(err["detail"].as_str().unwrap().contains("not open"));
    assert_eq!(h.call("chat_resume", json!({ "sessionId": id })).unwrap()["supportsLearn"], true);

    h.call("chat_delete_session", json!({ "sessionId": id })).unwrap();
    assert!(h.call("chat_list_sessions", json!({})).unwrap().as_array().unwrap().is_empty());
}

fn write_tree(root: &std::path::Path) {
    let story = "Once upon a time there was a little fox who lived by a river.\n".repeat(400);
    let code = "def add(a, b):\n    return a + b\n\n".repeat(300);
    std::fs::create_dir_all(root.join("stories")).unwrap();
    std::fs::create_dir_all(root.join("code")).unwrap();
    for i in 0..6 {
        std::fs::write(root.join("stories").join(format!("s{i}.txt")), &story).unwrap();
        std::fs::write(root.join("code").join(format!("c{i}.py")), &code).unwrap();
    }
    std::fs::write(root.join("stories").join("cover.png"), [0x89, b'P', b'N', b'G', 0, 0, 0, 1]).unwrap();
}

#[test]
fn datasets_can_be_installed_built_from_folders_edited_previewed_and_trained_on() {
    let h = harness(400.0);

    // Starters are listed, and the offline sampler installs without a network.
    let starters = h.call("list_starters", json!({})).unwrap();
    let ids: Vec<&str> = starters.as_array().unwrap().iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"tinystories-quick") && ids.contains(&"sampler"), "{ids:?}");
    let sampler = h.call("install_starter", json!({ "starterId": "sampler", "onEvent": "__CHANNEL__:1" })).unwrap();
    assert_eq!(sampler["summary"]["status"], "ready");
    assert_eq!(sampler["summary"]["starterId"], "sampler");
    let lane_names: Vec<&str> =
        sampler["lanes"].as_array().unwrap().iter().map(|l| l["name"].as_str().unwrap()).collect();
    assert!(lane_names.contains(&"stories") && lane_names.contains(&"arithmetic"), "{lane_names:?}");
    // Asking again returns the same dataset instead of making a second copy.
    let again = h.call("install_starter", json!({ "starterId": "sampler", "onEvent": "__CHANNEL__:2" })).unwrap();
    assert_eq!(again["summary"]["id"], sampler["summary"]["id"]);

    // A user's own folder: two lanes, a picture skipped, the rest split into train and test.
    let src = tempfile::tempdir().unwrap();
    write_tree(src.path());
    let path = src.path().to_string_lossy().to_string();
    let probes = h.call("probe_paths", json!({ "paths": [path] })).unwrap();
    assert_eq!(probes[0]["isDir"], true);
    assert!(probes[0]["approxFiles"].as_u64().unwrap() >= 13);

    let ds = h.call("add_folders", json!({ "paths": [path], "laneMode": "auto", "onEvent": "__CHANNEL__:3" })).unwrap();
    let id = ds["summary"]["id"].as_i64().unwrap();
    assert_eq!(ds["summary"]["status"], "ready");
    assert_eq!(ds["summary"]["kind"], "linked");
    let names: Vec<&str> = ds["lanes"].as_array().unwrap().iter().map(|l| l["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["stories", "code"], "stories keeps its fixed colour slot, so it sorts first");
    assert!(ds["skipped"]["binary"].as_u64().unwrap() >= 1, "the picture is skipped: {}", ds["skipped"]);
    assert!(ds["lanes"][0]["valBytes"].as_u64().unwrap() > 0, "some text is held out for testing");
    assert!(ds["lanes"][1]["samplePrompt"].as_str().is_some(), "code lane gets a prompt");

    // Peek inside a lane.
    let peek = h.call("preview_text", json!({ "datasetId": id, "lane": "code", "nChars": 120, "seed": 1 })).unwrap();
    assert!(peek["text"].as_str().unwrap().contains("def add"), "{peek}");

    // Turn a lane off: the dataset reports it, and a run built on it reads only what is left.
    let edited = h
        .call(
            "update_lane",
            json!({ "datasetId": id, "lane": "code", "enabled": false, "displayName": null, "samplePrompt": null }),
        )
        .unwrap();
    assert_eq!(edited["lanes"][1]["enabled"], false);
    let presets = h.call("preset_configs", json!({})).unwrap();
    let mut train = presets[0]["train"].clone();
    train["sampleEveryMin"] = json!(0.05);
    let run = h
        .call("create_run", json!({ "req": { "name": "on my text", "preset": "tiny", "model": presets[0]["model"], "train": train, "datasetId": id, "goal": { "type": "until_stopped" } } }))
        .unwrap();
    assert_eq!(
        run["datasetName"],
        "mini-agi-test-folder-placeholder"
            .replace("mini-agi-test-folder-placeholder", ds["summary"]["name"].as_str().unwrap())
    );
    let stories_bytes = ds["lanes"][0]["trainBytes"].as_u64().unwrap();
    assert_eq!(run["charsTotal"].as_u64().unwrap(), stories_bytes, "only enabled lanes count toward the total");

    let run_id = run["id"].as_i64().unwrap();
    h.call("start_run", json!({ "runId": run_id })).unwrap();
    h.wait_status(run_id, "running");
    std::thread::sleep(Duration::from_millis(2500));
    h.call("stop_run", json!({ "save": true })).unwrap();
    h.wait_status(run_id, "stopped");
    let domains = h.call("get_eval_domains", json!({ "runId": run_id })).unwrap();
    assert_eq!(domains, json!(["stories"]), "the switched-off lane is not read or tested");

    // Re-split with a bigger test share; the dataset is rebuilt from its folders.
    let rebuilt = h
        .call(
            "rebuild_dataset",
            json!({ "datasetId": id, "split": { "mode": "auto", "pct": 10.0, "seed": 9 }, "onEvent": "__CHANNEL__:4" }),
        )
        .unwrap();
    assert!(rebuilt["lanes"][0]["valBytes"].as_u64().unwrap() >= ds["lanes"][0]["valBytes"].as_u64().unwrap());
    assert_eq!(h.call("get_split", json!({ "datasetId": id })).unwrap()["pct"], 10.0);
    assert_eq!(rebuilt["lanes"][1]["enabled"], false, "a rebuild keeps the user's lane choices");

    // Jobs are recorded, and everything can be removed again.
    let jobs = h.call("list_jobs", json!({})).unwrap();
    assert!(jobs.as_array().unwrap().iter().all(|j| j["state"] == "done"), "{jobs}");
    h.call("delete_dataset", json!({ "datasetId": id, "deleteFiles": true })).unwrap();
    assert_eq!(h.call("get_dataset", json!({ "datasetId": id })).unwrap_err()["kind"], "not_found");
    let after = h.call("get_run", json!({ "runId": run_id })).unwrap();
    assert_eq!(after["status"], "stopped", "a run keeps its history when its dataset is deleted");
}

#[test]
fn bad_input_gets_plain_dataset_errors() {
    let h = harness(1000.0);
    let nothing =
        h.call("add_folders", json!({ "paths": [], "laneMode": "auto", "onEvent": "__CHANNEL__:1" })).unwrap_err();
    assert_eq!(nothing["kind"], "invalid");
    let missing = h
        .call(
            "add_folders",
            json!({ "paths": ["/definitely/not/here"], "laneMode": "auto", "onEvent": "__CHANNEL__:2" }),
        )
        .unwrap_err();
    assert_eq!(missing["kind"], "not_found");
    // A folder with only pictures has no readable text.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.png"), [0x89, b'P', b'N', b'G', 0]).unwrap();
    let err = h
        .call(
            "add_folders",
            json!({ "paths": [dir.path().to_string_lossy()], "laneMode": "auto", "onEvent": "__CHANNEL__:3" }),
        )
        .unwrap_err();
    assert_eq!(err["kind"], "invalid");
    assert!(err["detail"].as_str().unwrap().contains("could not find any"), "{err}");
    // The failed attempt is visible as a dataset in an error state, not silently lost.
    let list = h.call("list_datasets", json!({})).unwrap();
    assert_eq!(list[0]["status"], "error");
    // Training on a dataset that is not ready is refused.
    let presets = h.call("preset_configs", json!({})).unwrap();
    let id = list[0]["id"].clone();
    let refused = h
        .call("create_run", json!({ "req": { "name": null, "preset": "tiny", "model": presets[0]["model"], "train": null, "datasetId": id, "goal": null } }))
        .unwrap_err();
    assert_eq!(refused["kind"], "dataset_not_ready");
}

#[test]
fn storage_usage_adds_up_what_the_app_keeps_on_disk() {
    let h = harness(400.0);
    let run = trained_run(&h);
    let sampler = h.call("install_starter", json!({ "starterId": "sampler", "onEvent": "__CHANNEL__:1" })).unwrap();
    let usage = h.call("storage_usage", json!({})).unwrap();
    assert!(usage["datasetsBytes"].as_u64().unwrap() > 20_000, "the sampler text is on disk: {usage}");
    assert!(usage["runsBytes"].as_u64().unwrap() > 0, "the saved model is counted");
    assert!(usage["databaseBytes"].as_u64().unwrap() > 0);
    assert!(usage["freeBytes"].as_u64().unwrap() > 0);
    assert_eq!(usage["perRun"][0]["runId"], run);
    assert_eq!(sampler["summary"]["status"], "ready");
}

#[test]
fn a_model_exports_imports_back_and_can_be_chatted_with() {
    let h = harness(400.0);
    let run = trained_run(&h);
    let out = tempfile::tempdir().unwrap();
    let dest = out.path().to_string_lossy().to_string();

    let exported = h
        .call(
            "export_model",
            json!({ "req": { "runId": run, "checkpointId": null, "destDir": dest }, "onEvent": "__CHANNEL__:1" }),
        )
        .unwrap();
    let folder = std::path::PathBuf::from(exported["path"].as_str().unwrap());
    assert!(folder.join("llm-trainer-export.json").is_file() && folder.join("README.md").is_file());
    assert!(folder.join("model").join("COMPLETE").is_file());
    let card = std::fs::read_to_string(folder.join("README.md")).unwrap();
    assert!(card.contains("chat model") && card.contains("Import"), "{card}");
    assert!(exported["bytes"].as_u64().unwrap() > 0 && exported["files"].as_u64().unwrap() >= 3);
    assert!(!out.path().join(".chat-model-step-1.exporting").exists(), "no half-written folder is left behind");

    // Exporting twice never overwrites: the second folder gets a suffix.
    let again = h
        .call(
            "export_model",
            json!({ "req": { "runId": run, "checkpointId": null, "destDir": dest }, "onEvent": "__CHANNEL__:2" }),
        )
        .unwrap();
    assert_ne!(again["path"], exported["path"]);

    // Import it as a new run.
    let imported = h.call("import_model", json!({ "folder": exported["path"], "onEvent": "__CHANNEL__:3" })).unwrap();
    assert_eq!(imported["status"], "imported");
    assert_eq!(imported["name"], "chat model (imported)");
    let id = imported["id"].as_i64().unwrap();
    assert_ne!(id, run);
    assert!(imported["charsRead"].as_u64().unwrap() > 0);
    let ckpts = h.call("list_checkpoints", json!({ "runId": id })).unwrap();
    assert_eq!(ckpts[0]["kind"], "imported");

    // The imported model works like any other: open a chat on it.
    let chat =
        h.call("chat_open", json!({ "req": { "runId": id, "checkpointId": null, "mode": "continue" } })).unwrap();
    assert_eq!(chat["supportsLearn"], true);

    // Things that must be refused, with a reason a person can act on.
    let empty = tempfile::tempdir().unwrap();
    let err = h
        .call("import_model", json!({ "folder": empty.path().to_string_lossy(), "onEvent": "__CHANNEL__:4" }))
        .unwrap_err();
    assert_eq!(err["kind"], "invalid");
    assert!(err["detail"].as_str().unwrap().contains("not an export"), "{err}");

    let manifest_path = folder.join("llm-trainer-export.json");
    let mut m: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    m["engineName"] = json!("candle");
    std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
    let err = h.call("import_model", json!({ "folder": exported["path"], "onEvent": "__CHANNEL__:5" })).unwrap_err();
    assert!(err["detail"].as_str().unwrap().contains("candle engine"), "{err}");
    m["engineName"] = json!("mock");
    m["format"] = json!(99);
    std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
    assert_eq!(
        h.call("import_model", json!({ "folder": exported["path"], "onEvent": "__CHANNEL__:6" })).unwrap_err()["kind"],
        "checkpoint"
    );

    let missing = h
        .call("export_model", json!({ "req": { "runId": run, "checkpointId": null, "destDir": "/definitely/not/here" }, "onEvent": "__CHANNEL__:7" }))
        .unwrap_err();
    assert_eq!(missing["kind"], "not_found");
    // A failed import leaves no stray run behind.
    assert_eq!(
        h.call("list_runs", json!({})).unwrap().as_array().unwrap().len(),
        2,
        "the original and the one good import"
    );
}

/// The same workflows with the real engine: a user's own text becomes a dataset, a model trains on it, the saved model
/// is chatted with, learns on its own copy, and is exported and imported again. (A deliberately tiny network, so this
/// runs in seconds on any machine.)
#[cfg(feature = "real-engine")]
#[test]
fn the_real_engine_trains_on_a_users_text_and_the_result_chats_and_exports() {
    let h = harness_with(Arc::new(minagi_core::RealFactory::new(16.0)));
    assert_eq!(h.call("app_info", json!({})).unwrap()["isMock"], false);

    let src = tempfile::tempdir().unwrap();
    write_tree(src.path());
    let path = src.path().to_string_lossy().to_string();
    let ds = h.call("add_folders", json!({ "paths": [path], "laneMode": "auto", "onEvent": "__CHANNEL__:3" })).unwrap();
    let dataset = ds["summary"]["id"].as_i64().unwrap();

    let presets = h.call("preset_configs", json!({})).unwrap();
    let mut model = presets[0]["model"].clone();
    for (k, v) in [
        ("dModel", json!(32)),
        ("nHead", json!(2)),
        ("dFf", json!(64)),
        ("nPrelude", json!(1)),
        ("maxSteps", json!(4)),
        ("bpttWindow", json!(4)),
        ("trainStepsMean", json!(0.0)),
        ("haltPrior", json!(0.4)),
        ("block", json!(96)),
        ("poolExperts", json!(8)),
        ("poolMax", json!(12)),
        ("poolDFf", json!(16)),
        ("poolTopK", json!(2)),
    ] {
        model[k] = v;
    }
    let mut train = presets[0]["train"].clone();
    for (k, v) in [
        ("chunk", json!(32)),
        ("passage", json!(192)),
        ("contextStart", json!(64)),
        ("contextEnd", json!(96)),
        ("resident", json!(4)),
        ("ramCache", json!(6)),
        ("lr", json!(0.004)),
        ("sampleEveryMin", json!(0.02)),
        ("saveEveryMin", json!(0.04)),
        ("evalChars", json!(192)),
    ] {
        train[k] = v;
    }
    train["growth"]["everyChars"] = json!(160);
    let run = h
        .call("create_run", json!({ "req": { "name": "real", "preset": "custom", "model": model, "train": train, "datasetId": dataset, "goal": { "type": "until_stopped" } } }))
        .unwrap();
    let id = run["id"].as_i64().unwrap();
    h.call("start_run", json!({ "runId": id })).unwrap();
    h.wait_status(id, "running");

    // wait for a real evaluation to be recorded, then stop and save
    let t0 = Instant::now();
    loop {
        let series = h
            .call("get_series", json!({ "req": { "runId": id, "keys": ["eval.overall"], "x": "chars", "from": null, "to": null, "maxPoints": 50 } }))
            .unwrap();
        let points =
            series.as_array().and_then(|s| s.first()).and_then(|s| s["x"].as_array()).map(|p| p.len()).unwrap_or(0);
        if points >= 1 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "no evaluation was recorded: {series}");
        std::thread::sleep(Duration::from_millis(200));
    }
    h.call("stop_run", json!({ "save": true })).unwrap();
    let done = h.wait_status(id, "stopped");
    assert!(done["charsRead"].as_u64().unwrap() >= 32, "{done}");
    assert!(done["lastHeldoutNats"].as_f64().is_some(), "{done}");
    assert!(done["nExperts"].as_u64().unwrap() >= 8, "{done}");
    let ckpts = h.call("list_checkpoints", json!({ "runId": id })).unwrap();
    assert!(ckpts.as_array().unwrap().iter().any(|c| c["kind"] == "stop"), "{ckpts}");

    // chat with it, learning on its own copy
    let chat =
        h.call("chat_open", json!({ "req": { "runId": id, "checkpointId": null, "mode": "continue" } })).unwrap();
    let cid = chat["id"].as_i64().unwrap();
    h.call("chat_set_learn", json!({ "sessionId": cid, "enabled": true })).unwrap();
    h.call("chat_send", json!({ "sessionId": cid, "text": "Once upon a time", "params": { "maxNew": 30, "adapt": true }, "onEvent": "__CHANNEL__:7" })).unwrap();
    let msgs = wait_messages(&h, cid, 2);
    assert!(!msgs[1]["content"].as_str().unwrap().is_empty(), "{msgs}");
    let saved = h.call("chat_save_adapted", json!({ "sessionId": cid })).unwrap();
    assert_eq!(saved["kind"], "manual");

    // continue the saved model as a new run (the same model, a new name, a lower learning rate)
    let parent_step = done["step"].as_u64().unwrap();
    let mut slower = train.clone();
    slower["lr"] = json!(0.001);
    let fork = h
        .call("fork_run", json!({ "req": { "runId": id, "checkpointId": null, "name": "second try", "datasetId": null, "train": slower, "goal": { "type": "until_stopped" } } }))
        .unwrap();
    let fid = fork["id"].as_i64().unwrap();
    assert_ne!(fid, id);
    assert_eq!(fork["status"], "created");
    assert_eq!(fork["step"], parent_step, "the new run starts where the save was");
    assert_eq!(fork["datasetName"], run["datasetName"]);
    let fconf = h.call("get_run_config", json!({ "runId": fid })).unwrap();
    assert_eq!(fconf["train"]["lr"], 0.001);
    h.call("start_run", json!({ "runId": fid })).unwrap();
    h.wait_status(fid, "running");
    let t1 = Instant::now();
    loop {
        let r = h.call("get_run", json!({ "runId": fid })).unwrap();
        if r["step"].as_u64().unwrap_or(0) > parent_step + 3 {
            break;
        }
        assert!(t1.elapsed() < Duration::from_secs(30), "the forked run did not train on: {r}");
        std::thread::sleep(Duration::from_millis(100));
    }
    h.call("stop_run", json!({ "save": false })).unwrap();
    h.wait_status(fid, "stopped");
    // the original is untouched
    assert_eq!(h.call("get_run", json!({ "runId": id })).unwrap()["step"], parent_step);
    // a run cannot be forked while it is active, and a run with no save cannot be forked at all
    let fresh = h
        .call("create_run", json!({ "req": { "name": "never ran", "preset": "tiny", "model": model, "train": train, "datasetId": dataset, "goal": null } }))
        .unwrap();
    assert_eq!(
        h.call("fork_run", json!({ "req": { "runId": fresh["id"], "checkpointId": null, "name": null, "datasetId": null, "train": null, "goal": null } })).unwrap_err()["kind"],
        "invalid"
    );

    // export for other tools
    let st_out = tempfile::tempdir().unwrap();
    let st = h
        .call("export_safetensors", json!({ "req": { "runId": id, "checkpointId": null, "destDir": st_out.path().to_string_lossy() }, "onEvent": "__CHANNEL__:2" }))
        .unwrap();
    let st_folder = std::path::PathBuf::from(st["path"].as_str().unwrap());
    assert!(
        st_folder.join("model.safetensors").is_file()
            && st_folder.join("config.json").is_file()
            && st_folder.join("README.md").is_file()
    );
    assert!(st["bytes"].as_u64().unwrap() > 100_000, "{st}");

    // export and import the trained model
    let out = tempfile::tempdir().unwrap();
    let exported = h
        .call("export_model", json!({ "req": { "runId": id, "checkpointId": null, "destDir": out.path().to_string_lossy() }, "onEvent": "__CHANNEL__:1" }))
        .unwrap();
    let imported = h.call("import_model", json!({ "folder": exported["path"], "onEvent": "__CHANNEL__:3" })).unwrap();
    assert_eq!(imported["status"], "imported");
    let again = h
        .call("chat_open", json!({ "req": { "runId": imported["id"], "checkpointId": null, "mode": "continue" } }))
        .unwrap();
    assert_eq!(again["supportsLearn"], true);
}

/// A weights folder written by the original Python program is previewed, imported as a run, and chatted with.
#[cfg(feature = "real-engine")]
#[test]
fn a_python_checkpoint_is_previewed_imported_and_chatted_with() {
    let h = harness_with(Arc::new(minagi_core::RealFactory::new(16.0)));
    let src =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/minagi-core/tests/fixtures/store/py_weights");
    let folder = src.canonicalize().unwrap().to_string_lossy().to_string();

    let preview = h.call("preview_import", json!({ "folder": folder })).unwrap();
    assert_eq!(preview["kind"], "python");
    assert_eq!(preview["importable"], true, "{preview}");
    assert_eq!(preview["nExperts"], 10);
    assert_eq!(preview["step"], 9);
    assert!(preview["summary"].as_str().unwrap().contains("Python checkpoint"), "{preview}");

    let run = h.call("import_model", json!({ "folder": folder, "onEvent": "__CHANNEL__:3" })).unwrap();
    assert_eq!(run["status"], "imported");
    assert!(run["name"].as_str().unwrap().starts_with("Original model"), "{run}");
    let id = run["id"].as_i64().unwrap();
    let ckpts = h.call("list_checkpoints", json!({ "runId": id })).unwrap();
    assert_eq!(ckpts[0]["kind"], "imported");

    // the imported model writes
    let chat =
        h.call("chat_open", json!({ "req": { "runId": id, "checkpointId": null, "mode": "continue" } })).unwrap();
    let cid = chat["id"].as_i64().unwrap();
    h.call("chat_send", json!({ "sessionId": cid, "text": "Hello", "params": { "maxNew": 12, "adapt": true }, "onEvent": "__CHANNEL__:7" })).unwrap();
    let msgs = wait_messages(&h, cid, 2);
    assert!(!msgs[1]["content"].as_str().unwrap().is_empty());

    // a folder that is neither kind is refused in plain words
    let empty = tempfile::tempdir().unwrap();
    let err = h.call("preview_import", json!({ "folder": empty.path().to_string_lossy() })).unwrap_err();
    assert_eq!(err["kind"], "invalid");
    assert!(err["detail"].as_str().unwrap().contains("does not look like a saved model"));
}
