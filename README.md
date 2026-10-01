# LLM Trainer

A desktop app that trains a small language model on your own computer, built so anyone can do it: pick a size, point it at some text, press start, and watch it learn to write.

It is a Rust and Tauri re-implementation of [mini-AGI](https://github.com/volotat/mini-AGI) (MIT), a byte-level language model that trains from scratch on a stream of text. The model keeps learning as it reads, reuses one block several times per character (so it can "think longer" about hard characters), and routes each character through a pool of small expert networks that can grow and shrink during training.

## Status

Everything in the plan is built and tested on macOS (Apple Silicon). Windows and Linux builds are configured in CI but have not been run yet.

| Part | State |
|---|---|
| Shared types, presets, estimator, verdict/ETA logic (`minagi-types`) | done |
| SQLite store: runs, telemetry, evaluations, samples, checkpoints, chat, datasets, jobs (`minagi-store`) | done |
| Simulated engine and contract tests (`minagi-mock`) | done |
| Dataset manager: folder scan, lanes, split, TinyStories download, arithmetic generator (`minagi-data`) | done |
| Byte tokenizer and training/evaluation text streams (`minagi-text`) | done |
| **Real ML engine on candle** (`minagi-core`): recurrent adaptive-depth transformer, paged expert pool, AdamW, plasticity controller, growth and pruning, checkpoints, chat with live learning | done; Metal verified |
| Python checkpoint import, safetensors export | done |
| Tauri app and UI: welcome and quick start, text, set up, live train dashboard, chat, runs, compare, export/import, continue-as-new-run, settings | done |

### How faithful is the engine?

The engine is a port of mini-AGI, held to the original by **golden tests**: fixtures written by the Python reference (`tools/golden/`) are replayed in Rust and must match. They cover the tensor primitives, the whole recurrent forward (halting, frozen characters, the PonderNet loss, eight variants), cached inference, expert admission and routing (including capacity drops and the exploration bonus), four real AdamW steps with experts paged in and out of the card, greedy decoding with the adaptation trace, and the tokenizer. Checkpoints are written in the original's layout and load in the Python reference (and the other way round), which the interop tests check against the real Python code.

Deliberate differences from the original: capacity drops are decided in token order (the original's order is unspecified), non-finite steps are skipped instead of poisoning the weights, and a fresh model is created from the preset rather than `config.yaml`.

### Speed and memory (Apple M4 Pro, 24 GB, measured)

| Preset | Speed | Peak memory |
|---|---|---|
| Tiny | about 10,000 characters per second | about 1 GB |
| Small | about 2,500 characters per second | about 3 to 5 GB |
| Full | about 350 characters per second (rows are recomputed in the backward pass so memory does not grow with depth) | about 12 to 15 GB |

## Run it

Requirements: Rust 1.93, Node 24, pnpm 10.

```sh
pnpm install

# the desktop app with the real engine (Apple GPU on a Mac with macOS 15 or newer, otherwise the CPU)
MINAGI_HOME=$PWD/.dev-data pnpm tauri dev

# the same app with a simulated engine (curves and text are made up, and the app says so)
MINAGI_ENGINE=mock MINAGI_HOME=$PWD/.dev-data MINAGI_MOCK_SPEED=20 pnpm tauri dev

# just the UI in a browser, against an in-browser simulation (fastest way to iterate on screens)
pnpm dev:web          # http://127.0.0.1:1420
node scripts/shots.mjs   # walks the app in headless Chromium and saves screenshots to ./shots
node scripts/axe.mjs     # accessibility audit of every screen in both themes

# train from the command line, no app
cargo run --release -p minagi-cli --bin minagi_train -- --synthetic --preset tiny --backend metal --minutes 1
```

`MINAGI_MOCK_SPEED` is how many times faster than real time the simulation runs. `MINAGI_MOCK_SCENARIO` can be `plateau`, `diverge`, `overfit`, `oom`, `diskfull` or `nogpu` to exercise the error and warning screens. Build with `--no-default-features` to leave the real engine (and candle) out while working on screens.

## Layout

```
crates/minagi-types   shared contract: config and presets, events, traits, insight helpers (no ML dependencies)
crates/minagi-store   SQLite: migrations, a single writer thread, chart downsampling, crash recovery
crates/minagi-mock    simulated engine and generator, held to the same contract as the real one
crates/minagi-data    dataset manager: folder scan, lanes, split, starter downloads, arithmetic generator
crates/minagi-text    byte tokenizer, round-robin training stream, held-out evaluation chunks
crates/minagi-core    the ML engine on candle: ops, model, expert pool, optimiser, trainer, chat, checkpoints
crates/minagi-cli     command-line tools: train, benchmarks
src-tauri             the desktop app: commands, the recorder, session control
ui                    React + TypeScript frontend, charts in ECharts
tools/golden          Python reference dumps used by the parity tests (see its README)
docs                  engine spike results and design notes
```

How a run flows: the engine thread emits events; the **recorder** thread persists them to SQLite once a second, throttles them into a live stream (at most 4 updates a second) for the UI, and works out the plain-English verdict and time estimate. The UI reads history from SQLite through typed commands and live state from the stream.

## Tests

```sh
cargo test --workspace --exclude llm-trainer-app     # engine, data, store, mock, types (about 450 tests)
cargo test -p llm-trainer-app                        # the app, including full workflows with the real engine
pnpm --filter ui typecheck && pnpm --filter ui test
cargo clippy --workspace --all-targets -- -D warnings
```

The TypeScript bindings in `ui/src/bindings.ts` are generated from the Rust types (`cargo test -p llm-trainer-app export_bindings`) and checked in CI. The Python reference fixtures are regenerated with `.venv/bin/python tools/golden/dump_model.py` (needs the reference cloned into `tools/golden/ref`).

## Credits

Based on [mini-AGI](https://github.com/volotat/mini-AGI) by volotat (MIT). Starter text comes from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0).
