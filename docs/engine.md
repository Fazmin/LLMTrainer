# The engine (`crates/minagi-core`)

A port of mini-AGI's model and trainer onto [candle](https://github.com/huggingface/candle). This note is for whoever changes it next: how it is put together, what is checked against the original, and the traps that cost days to find.

## What it computes

- **Byte-level language model**, vocabulary 265 (256 bytes plus 9 marker tokens), trained from scratch with one optimiser step per chunk of text.
- **Dense prelude blocks** (`n_prelude`), then **one weight-shared recurrent block** applied for up to `max_steps` *rows*. Row `n` mixes the running state with the embedded input (`adapter([h, x])`), attends, routes through the expert pool, and reads out logits plus a halting probability. A character's chance of stopping at a row is `p_n = cum * lam` (PonderNet); training minimises `sum_n p_n * CE_n` plus a KL pull toward a geometric prior. A halted character is frozen: it asks for no experts and later characters read its final key/values. See `model/forward.rs`.
- **Expert pool** (`moe/`): every expert is a file; RAM keeps the recently used ones; a fixed number of *slots* on the accelerator hold the ones the current text voted for. Which experts are admitted is decided by the whole text so far (`moe/book.rs`), routing among the admitted slots is top-k with a gate (`moe/route.rs`), and the optimiser's moments travel with the expert (`moe/pool.rs`). The pool grows by recombining hidden units of 16 parents and shrinks by deleting experts nothing has asked for in a long time.
- **Training loop** (`train/trainer.rs`): wall-clock scheduled evaluation, sampling and checkpoints, character-scheduled growth, the plasticity controller that steers the learning rate from held-out scores, the context ramp, control messages, and events for the app.
- **Chat** (`chat/`): greedy decoding with an adaptation trace (no randomness), one character per forward through per-row KV caches, and live learning on a private copy of the model.

## Checkpoints

The directory layout is the original's (`manifest.json`, `core.npz`, `routers.npz`, `optim.npz`, `experts/eNNNNN.npz` with bf16 moments stored as int16), plus a `COMPLETE` marker written last. A run keeps a *live* experts directory; a checkpoint's expert files are hard links into it, which is safe because the tiers never rewrite a file in place (a write-back replaces the name). Python-written directories load here and Rust-written ones load in the reference (`store/python.rs`, and the interop tests that run the real Python code).

## How it is verified

`tools/golden/README.md` documents the fixtures; `crates/minagi-core/tests/golden_*.rs` replay them:

| What | Test |
|---|---|
| RMSNorm, RoPE, attention (with and without a cache), SwiGLU, decoding, tokenizer | `golden_ops.rs` |
| The whole recurrent forward, eight variants (halting, freezing, bptt, min rows, no freeze), cached and chunked inference | `golden_model.rs` (a one-expert pool is exactly the original's dense MLP) |
| Admission, eviction, the text's vote, exploration, staleness, routing, capacity drops, the balancing loss | `golden_moe.rs` |
| Four real AdamW steps with experts paged in and out through a RAM tier of three | `golden_adamw.rs` |
| A model trained and saved by this engine, scored by the original Python code | `python_loss_parity.rs` (identical to six digits without the capacity bound) |
| Plasticity, brakes, context ramp | `pure_golden.rs` |
| Checkpoint files in both directions | `store_*.rs` |

Mutation checks matter here: breaking the Adam bias correction, the eviction order or the KL prior each makes the matching golden test fail.

## Decisions and deviations

- **f32 everywhere.** On the target GPU f16/bf16 was about 6% faster and f16 produces NaNs without loss scaling.
- **Capacity drops are decided in token order.** The original's order comes from an unstable sort and is unspecified.
- **Non-finite steps are skipped**, never applied; twenty in a row stop the run with a plain-English message.
- **Evaluation runs at the context window the model is being trained at.** Scoring at the longest window the model *could* read measured positions it had never been taught, and made a healthy run look like it was overfitting.
- **Dense-masked and sparse dispatch give the same function**, including which assignments the capacity limit drops; dense is simply faster for few resident experts.
- **Row checkpointing** (`model/ckpt.rs`, used for Full): the forward keeps no graph; the backward recomputes one row at a time from its saved inputs, so memory stays flat with depth. Gradients equal the ordinary step's (`row_checkpointing_gives_the_same_gradients_as_the_ordinary_step`).

## Traps (candle, Metal)

1. **Fused ops have no gradient.** `rms_norm`, `rope_i` and `softmax_last_dim` are registered without a backward, so using one in training silently gives zero gradients. `ops/hand.rs` wraps the fast forward and supplies the backward (the `HandBwd` pattern); `ops` also has composed versions, and tests compare the two.
2. **Detach carried state.** candle has no `no_grad`: an optimiser moment built from the previous one and the gradient keeps the whole op graph alive (about 15 GB after 30 steps in the spike). Moments are detached on every update; evaluation and generation use detached views of the weights.
3. **Autorelease pool.** Wrap every Metal step, evaluation chunk and generated character in `backend::pool`, or resident memory climbs.
4. **`scatter_add` backward is wrong** in the pinned candle; differentiable paths use `index_select` with `u32::MAX` sentinels for padding (the backward ignores out-of-range indices).
5. **crates.io candle 0.11.0 is unusable here** (does not compile on this rustc and reproduces a Metal NaN bug); the workspace pins a git commit.
6. **In-place slot writes** (`slice_set`) are safe only because a slot that something already used this forward is never overwritten before the backward pass: admission only takes slots nothing admitted.

## Performance notes (Apple M4 Pro)

A step costs roughly a fixed amount (embedding, prelude, optimiser) plus a per-row amount: attention about 18 ms and the expert pool about 28 ms per row at Small size and 1,024 characters. The GPU is the bottleneck (the CPU thread is mostly waiting in `waitUntilCompleted`), and the kernels reach well under a tenth of the GEMM peak. The next real win would be fused attention and expert kernels, not host-side tuning. `minagi-cli`'s `bench_model` and `bench_parts` measure these; `MINAGI_PROFILE=1` prints where each training step's time goes.
