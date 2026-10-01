# Golden fixtures for the Rust port (`crates/minagi-core`)

Numerical-parity test data generated from the **original Python implementation of mini-AGI**
(github.com/volotat/mini-AGI, cloned read-only at `tools/golden/ref/`). Nothing in here is
hand-written data: every number comes out of the reference code (or, where a number is derived,
the script asserts it against the reference).

* Fixtures: `crates/minagi-core/tests/fixtures/golden/` (about 5.6 MB in total).
* Generator: `tools/golden/dump_model.py` (+ `golden_common.py`).
* Independent checker: `tools/golden/verify_golden.py` (see "Verification" below).

## 1. Regenerate

```sh
# from the repo root (/Users/fazmin/projects/LLMTrainer)
.venv/bin/python tools/golden/dump_model.py                    # everything, ~3 s
.venv/bin/python tools/golden/dump_model.py --only ops_rope,decode     # a subset
.venv/bin/python tools/golden/dump_model.py --out /tmp/golden  # elsewhere
.venv/bin/python tools/golden/verify_golden.py                 # re-run twice + independent checks
git -C tools/golden/ref rev-parse HEAD                         # reference commit
```

Provenance of the committed fixtures:

| item | value |
| --- | --- |
| reference commit | `6aa42ad76e558f6332f8d1e7d53bb801f511b165` ("training process update") |
| python | 3.12.12 |
| torch | 2.14.1 (CPU, `torch.set_num_threads(1)`, float32) |
| numpy | 2.5.3 |
| determinism | two fresh runs are byte-identical (checked by `verify_golden.py`) |

Seeds are fixed (`torch.Generator` objects, see `Rng` in `golden_common.py`). The *inputs* of every
fixture are stored in the fixture, so the Rust side never needs to reproduce torch's RNG. A
different torch version would regenerate different (equally valid) random inputs; the committed
files are the source of truth.

## 2. File format and conventions

* `*.npz` is written exactly like `numpy.savez` (stored zip of `.npy`; byte-identical, only the
  entry order is fixed). Keys contain `/` as a path separator (`perturbed/grad/halt.bias`); the
  zip entry is `perturbed/grad/halt.bias.npy`. Dtypes used: `f32` (`<f4`), `i64` (`<i8`),
  `u8` (`|u1`), and `i16` (`<i2`, only for raw bfloat16 bits in `adamw_paged.npz`).
  **There is no bool and no 0-d array**: booleans are `u8` 0/1, scalars have shape `[1]`
  (`f32` for floats, `i64` for ints).
* Weight keys are the reference's own `state_dict()` / `named_parameters()` names. The tied head
  is stored once, as `tok_emb.weight` (`head.weight` is the same tensor; its gradient is part of
  `grad/tok_emb.weight`). Non-persistent buffers (rope tables, pool bookkeeping) are not weights.
* Gradients are `grad/<param name>` (gradient of the fixture's loss w.r.t. that parameter,
  before any clipping), same shape as the parameter.
* PyTorch Linear weights are `[out, in]`; `y = x @ W.T` (no biases anywhere except `halt.bias`).
* `<name>` in the key tables below is a placeholder matched by `[^/]+`; `<param>` is any model
  parameter name from this list (dense model): `tok_emb.weight [265,32]`,
  `prelude.0.ln1.weight [32]`, `prelude.0.attn.qkv.weight [96,32]`, `prelude.0.attn.proj.weight [32,32]`,
  `prelude.0.ln2.weight [32]`, `prelude.0.mlp.w1.weight [64,32]`, `prelude.0.mlp.w3.weight [64,32]`,
  `prelude.0.mlp.w2.weight [32,64]`, the same seven under `recur.0.`, `adapter.weight [32,64]`,
  `ln_f.weight [32]`, `halt.weight [1,32]`, `halt.bias [1]` (19 tensors). The pooled model replaces
  `recur.0.mlp.w1/w3/w2.weight` by `recur.0.mlp.router.weight [8,32]` and
  `recur.0.mlp.depth_emb [32]` and adds `pool.gate [8]`, `pool.w1 [4,16,32]`, `pool.w3 [4,16,32]`,
  `pool.w2 [4,32,16]` (the slot tensors).
* The existing `minagi_core::store::NpzReader` reads these files (stored zip, `f32/i64/u8/i16`).
* `*.json` files hold structured data and scalars. Floats that came from float32 are written as the
  shortest decimal that round-trips to the same float32 (parse as f32, or as f64 then cast).
* `config.json` is the tiny config every model-level fixture uses (see section 5). Where a variant
  overrides a field it says so (`cfg_overrides`).
* `SHA256SUMS` lists every fixture file (written by a full `dump_model.py` run).

Tolerances that the checker found comfortable (float32 torch vs an independent float64 numpy
implementation): forward values agree to about 3e-7 relative to the array's max; gradients to about
1e-6. A float32 Rust implementation with a different accumulation order should pass with
`atol = 1e-5 + 1e-4 * max|ref|` for forwards and `atol = 1e-5 + 1e-3 * max|ref|` for gradients and
parameter updates; `ops_rope.npz` stores how far an f64-computed rope table is from the reference's
float32 one (about 7e-7).

## 3. Index

| file | what it pins down | Rust test to write |
| --- | --- | --- |
| `tokenizer.json` | encode / decode / token_to_id, markers, invalid UTF-8 | tokenizer unit test |
| `ops_rmsnorm.npz` | RMSNorm forward and backward | op test |
| `ops_rope.npz` | rope tables, `apply_rope` with position offsets, backward | op test |
| `ops_attention.npz` | fused-qkv causal attention, KV cache with offset mask, backward | op test |
| `ops_swiglu.npz` | SwiGLU forward and backward | op test |
| `dense_forward.npz/.json` | whole dense `RecurCoder` training forward + PonderNet loss + backward, 8 variants | model test |
| `dense_infer.npz/.json` | inference forward (halted logits, steps), KV-cache chunking, all-halted rows | model test |
| `moe_shared.npz` | `PooledMLP` + resident `SharedPool`: routing, aux loss, active mask, capacity | MoE test |
| `moe_admission.json` | `PagedPool` admission / eviction / vote / exploration bookkeeping, `dying()` | pager state-machine test |
| `moe_paged_forward.npz` | `PooledMLP` + `PagedPool` (4 of 8 resident): admission mass, masked routing, grads | MoE + pager test |
| `adamw_paged.npz/.json` | 4 real training steps of the paged model: AdamW groups, clip, moment parking/restore | optimiser + pager test |
| `decode.json` | `pick_next` greedy + adaptation trace | decode test |
| `train_step_dense.npz/.json` | dense training step: AdamW, warm-up, clip, weight-decay variants | optimiser test |
| `config.json` | tiny config, runtime constants, versions, commit | - |

Everything in the task list was produced; nothing was skipped. Things that are deliberately
partial are listed in section 8.

## 4. Monkeypatches and reference-state tweaks (all in `dump_model.py`)

The reference needed **no** code changes and no CUDA workarounds: with float32 on CPU,
`minagi.precision.amp()` is a no-op, `fused=False` is passed to AdamW, and no `torch.cuda` call is
reached. The following were still done in the script (never in `ref/`):

| mark | what | where | effect on the data |
| --- | --- | --- | --- |
| `LineProbe` | `sys.settrace` reads *locals* (`lam`, `p_n`, `cum`, `halted`, `active`, `ce`, `logits_n`, `h`, `x`, `logits`, ...) out of `RecurCoder.forward`, `PooledMLP._route`, `pick_next` at a fixed source line; nothing is changed | `dense_forward`, `dense_infer`, `adamw_paged`, `decode` | none (observation) |
| `spy_admit`, `spy_tiers` | instance-level wrappers that log `PagedPool.admit(mass, merit)` and `Tiers.fetch/put/_to_disk` calls and call straight through | `moe_paged_forward`, `adamw_paged` | none (observation) |
| `model.sample_depth = lambda: 2` | pins the (otherwise Poisson-sampled) depth | `dense_forward` variant `n2` only | the reference with `train_steps_mean = 0` never runs fewer than `max_steps` rows, this variant shows what happens when it does |
| `stable_argsort()` | `torch.argsort` is replaced by a **stable** argsort for the duration of a forward | `moe_shared` (`cap/out_stable`, `cap/grad_stable/*`), the whole `adamw_paged` run | torch CPU `argsort` is *not* stable for >= ~24 elements, and PooledMLP's capacity drop keeps "the first `limit` entries of each expert's sorted run". Unpatched, which assignments an over-capacity expert drops is unspecified. Stable order = ascending flat assignment index (token-major, rank-minor). Adopt that convention in Rust. |
| `grad_checkpoint=False` | argument of `PooledMLP(...)` | `moe_shared`, `moe_paged_forward` | none (checkpoint recompute is numerically identical); `adamw_paged` keeps the reference default (`True`) |
| state assignments | `pool.recent`, `pool.explore_bias/explore_steps`, `pool.born/last_seen/segments/now/trial/dying_at` are assigned directly | `moe_admission`, `moe_paged_forward`, `adamw_paged` | scenario set-up only; each scenario lists them as `set_recent` / `set_explore` ops or `config` |
| `import train` | the reference's own `train.build_paged` and `train._split_trunk_pool` are used to build the paged model and the optimiser groups (import sets an env var, `PYTORCH_CUDA_ALLOC_CONF`, harmless) | `adamw_paged` | none |

## 5. Tiny config

`config.json` -> `config` (dense, `use_pool` false) and `pool_config` (`use_pool` true):

`vocab_size 265, d_model 32, n_head 2 (head_dim 16), d_ff 64, n_prelude 1, n_recur 1, n_coda 0,
max_steps 4, min_steps 1, halt_prior 0.4, halt_thresh 0.9, halt_freeze true, ponder_beta 0.01,
bptt_window 4, train_steps_mean 0 (training always runs max_steps rows), block 64, rope_theta 10000,
tie_embeddings true`; pool: `pool_experts 8, pool_d_ff 16, pool_depth 1, pool_top_k 2,
pool_capacity_factor 1.5, pool_max 8 (= router rows), pool_aux 0.01`; runtime (outside
`RecurConfig`): `pool_resident 4, explore_bias 0.65, explore_steps 1000`.
Cache slots for inference: `n_prelude + max_steps * (n_recur + n_coda) = 5`.

Weights are randomised **after** construction in most fixtures (reference default init leaves
attention/MLP at N(0, 0.02), `halt.weight * 0.01`, `halt.bias -2` and the adapter at `[I, I]`, which
makes most bugs invisible). The default-init model is kept as variant `default_init`.
`perturbed`: `tok_emb` N(0, 0.35); norm weights `1 + N(0, 0.1)`; adapter `[I, I] + N(0, 0.05)`;
`halt.weight` N(0, 0.3); `halt.bias -0.5`; all other matrices N(0, 0.15). `halting`:
`perturbed` with `halt.weight` N(0, 0.6) and `halt.bias 0.9`.

## 6. Key tables

Each `###` heading below is a fixture file; each row is a key *pattern* (placeholders in `<>`).
`verify_golden.py` checks that every key of every `.npz` matches a row here, that every row
matches at least one key, and that numeric shapes/dtypes agree. Shapes written with letters
(`[B,T,32]`, `[N]`, ...) vary between keys of the same pattern.

### `ops_rmsnorm.npz`

`RMSNorm(32)`: `out = x * rsqrt(mean(x*x, -1, keepdim) + eps) * weight`, mean accumulated in f32,
`eps = 1e-6`, normalised over the last dimension only.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `eps` | [1] | f32 | 1e-6 |
| `weight` | [32] | f32 | the (non-trivial) scale, `1 + 0.3 * N(0,1)` |
| `<case>/x` | [B,T,32] | f32 | input. cases: `base` [1,5,32], `big` (x30), `tiny` (x1e-3, eps matters), `batch2` [2,3,32] |
| `<case>/g` | [B,T,32] | f32 | upstream gradient; the scalar loss is `sum(out * g)` |
| `<case>/out` | [B,T,32] | f32 | forward result |
| `<case>/dx` | [B,T,32] | f32 | `d loss / d x` |
| `<case>/dweight` | [32] | f32 | `d loss / d weight` (summed over batch and positions) |

### `ops_rope.npz`

`build_rope(block=64, head_dim=16, theta=10000)`: `inv = 1/(theta ** (arange(0,16,2)/16))`
(float32, `torch.pow`), `freqs = outer(arange(64) as f32, inv)`, `cos`/`sin` of that, all float32.
`apply_rope(x [B,H,T,16], cos [T,8], sin [T,8])` rotates the pairs `(x[..., 0::2], x[..., 1::2])`:
`out_even = x1*cos - x2*sin`, `out_odd = x1*sin + x2*cos`, re-interleaved. Backward is the transpose
rotation.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `cos` | [64,8] | f32 | `build_rope` cos table |
| `sin` | [64,8] | f32 | `build_rope` sin table |
| `max_abs_diff_cos_f32_vs_f64` | [1] | f32 | informational: how far the f32 table is from an f64 one |
| `max_abs_diff_sin_f32_vs_f64` | [1] | f32 | same for sin |
| `<case>/pos_offset` | [1] | i64 | cases `pos0`, `pos7`, `pos59`: rows `cos[off:off+5]` are used |
| `<case>/x` | [1,2,5,16] | f32 | input |
| `<case>/g` | [1,2,5,16] | f32 | upstream gradient (`loss = sum(out*g)`) |
| `<case>/out` | [1,2,5,16] | f32 | `apply_rope(x, cos[off:off+5], sin[off:off+5])` |
| `<case>/dx` | [1,2,5,16] | f32 | gradient w.r.t. `x` |

### `ops_attention.npz`

`Attention(d_model 32, n_head 2)`: `q,k,v = split(x @ qkv.T, 32, dim=2)` (rows `0..32` are q,
`32..64` k, `64..96` v); each is viewed `[B,T,2,16]` and transposed to `[B,2,T,16]` (head `h` owns
columns `16h..16h+16`); rope on q and k with the cos/sin rows of the *absolute* positions; the cache
(if any) stores **post-rope k and v** and the new keys are appended after the cached ones;
`y = softmax(q k^T / sqrt(16) + mask) v`; `out = merge_heads(y) @ proj.T`. Mask: without a cache,
standard causal (key `j` visible to query `i` iff `j <= i`); with `P` cached positions, query `i`
(absolute `P+i`) sees keys `j <= P + i`.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `qkv.weight` | [96,32] | f32 | fused projection weight |
| `proj.weight` | [32,32] | f32 | output projection weight |
| `a/x` | [1,8,32] | f32 | case a: no cache, T=8, rope positions 0..7 |
| `a/g` | [1,8,32] | f32 | upstream gradient |
| `a/out` | [1,8,32] | f32 | attention output |
| `a/dx` | [1,8,32] | f32 | gradient w.r.t. `x` |
| `a/dqkv.weight` | [96,32] | f32 | gradient w.r.t. `qkv.weight` |
| `a/dproj.weight` | [32,32] | f32 | gradient w.r.t. `proj.weight` |
| `a/q_rope` | [1,2,8,16] | f32 | q after split, heads and rope |
| `a/k_rope` | [1,2,8,16] | f32 | k after rope |
| `a/v` | [1,2,8,16] | f32 | v |
| `a/y_pre_proj` | [1,8,32] | f32 | attention output before `proj` (heads merged) |
| `<case>/P` | [1] | i64 | cases `b` (P=5,T=3) and `c` (P=7,T=1): number of cached positions |
| `<case>/T` | [1] | i64 | number of new positions |
| `<case>/x_prev` | [1,P,32] | f32 | the input that produced the cache (positions 0..P-1) |
| `<case>/x_new` | [1,T,32] | f32 | the new positions P..P+T-1 (rope rows `cos[P:P+T]`) |
| `<case>/g` | [1,T,32] | f32 | upstream gradient for the new positions |
| `<case>/cache_k_in` | [1,2,P,16] | f32 | cached post-rope keys, an INPUT (treated as a constant) |
| `<case>/cache_v_in` | [1,2,P,16] | f32 | cached values, an INPUT |
| `<case>/out` | [1,T,32] | f32 | output for the new positions |
| `<case>/cache_k_out` | [1,2,8,16] | f32 | returned cache keys (`cache_k_in` ++ new keys) |
| `<case>/cache_v_out` | [1,2,8,16] | f32 | returned cache values |
| `<case>/dx_new` | [1,T,32] | f32 | gradient w.r.t. `x_new` (cache constant) |
| `<case>/dqkv.weight` | [96,32] | f32 | gradient w.r.t. `qkv.weight` (new positions only) |
| `<case>/dproj.weight` | [32,32] | f32 | gradient w.r.t. `proj.weight` |
| `<case>/full_out` | [1,8,32] | f32 | sanity: uncached pass over all 8 positions; rows `P..8` equal `out` |

### `ops_swiglu.npz`

`SwiGLU`: `out = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T` (`w1` gate, `w3` value, `w2` down).

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `w1` | [64,32] | f32 | gate weight |
| `w3` | [64,32] | f32 | value weight |
| `w2` | [32,64] | f32 | down weight |
| `<case>/x` | [1,5,32] | f32 | input; cases `base` and `big` (x6) |
| `<case>/g` | [1,5,32] | f32 | upstream gradient |
| `<case>/out` | [1,5,32] | f32 | forward |
| `<case>/h` | [1,5,64] | f32 | hidden `silu(w1 x) * (w3 x)` |
| `<case>/dx` | [1,5,32] | f32 | gradient w.r.t. x |
| `<case>/dw1` | [64,32] | f32 | gradient w.r.t. w1 |
| `<case>/dw3` | [64,32] | f32 | gradient w.r.t. w3 |
| `<case>/dw2` | [32,64] | f32 | gradient w.r.t. w2 |

### `dense_forward.npz`

Training forward of the dense `RecurCoder` (`model.train()`, `targets` given, no caches, `B=1, T=16`)
followed by `loss.backward()`. `idx`, `targets` below are the inputs of **every** variant.
Variants (details and counters in `dense_forward.json`; `cfg_overrides` = fields changed from
`config.json`):

| variant | weights | what it exercises |
| --- | --- | --- |
| **default_init** | own | reference default init; nothing halts before the forced last row |
| **perturbed** | own | non-trivial weights; a few tokens halt on rows 1-2, partial `active` masks |
| **halting** | own | `perturbed` + big halt weights: tokens halt on rows 0, 1, 2 (6/2/1 tokens) |
| **halting_all** | `perturbed` except `halt.weight`, `halt.bias` (stored under this variant) | every token halts after row 0; rows 1..3 have an EMPTY active set (block skipped) |
| **bptt** | = `perturbed` | `bptt_window 2`: `h` detached at the start of rows 0 and 1 |
| **min_steps2** | = `perturbed` | `min_steps 2`: `lam` forced to 0 on row 0 (forward arrays only, no grads, to save space) |
| **n2** | = `perturbed` | only 2 rows run (`sample_depth` pinned): forced `lam = 1` on row 1, prior normalised over 2 rows |
| **no_freeze** | = `perturbed` | `halt_freeze false`: `active` always None, `halted` never updates |

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `idx` | [1,16] | i64 | input ids (positions 0, 5, 9 are marker ids 256, 262, 264) |
| `targets` | [1,16] | i64 | targets (cross-entropy per position; NOT shifted: `targets[t]` is scored against the logits at position `t`) |
| `<variant>/<param>` | per param | f32 | weights. Stored only for `default_init`, `perturbed`, `halting` (all 19 tensors) and `halting_all` (only `halt.weight`, `halt.bias`); the other variants reuse `perturbed` |
| `<variant>/grad/<param>` | per param | f32 | gradient of `loss` w.r.t. each parameter (absent for `min_steps2`) |
| `<variant>/lam/<n>` | [1,16] | f32 | `lam` of row `n` as used: `sigmoid(halt(ln_f(h)))`, then forced to 1 on the last row and to 0 if `n < min_steps-1` |
| `<variant>/p_n/<n>` | [1,16] | f32 | `p_n = cum_before * lam` (halting probability mass of row n) |
| `<variant>/cum/<n>` | [1,16] | f32 | `cum` AFTER the row update, `cum = cum_before * (1 - lam)` |
| `<variant>/ce/<n>` | [1,16] | f32 | per-position cross-entropy of row n's logits against `targets` |
| `<variant>/halted/<n>` | [1,16] | u8 | `halted` AFTER the row-n update `halted = halted OR ((1 - cum) >= halt_thresh)` (freeze variants; all 0 for `no_freeze`) |
| `<variant>/active/<n>` | [1,16] | u8 | mask used on row n: all ones for n = 0 (reference passes `None`), else `~halted` after row n-1 (all ones for `no_freeze`) |
| `<variant>/h/<n>` | [1,16,32] | f32 | latent state after row n (after the `where(active, hn, h)` freeze) |
| `<variant>/logits_n/<n>` | [1,16,265] | f32 | `head(ln_f(h))` of row n (stored for `default_init`, `perturbed`, `halting`) |
| `<variant>/x_prelude` | [1,16,32] | f32 | embedding after the prelude blocks (the `x` that is concatenated with `h` every row) |
| `<variant>/mixture_logits` | [1,16,265] | f32 | the returned logits `sum_n logits_n * p_n` (stored for `default_init`, `perturbed`, `halting`) |
| `<variant>/halt_row` | [1,16] | i64 | first row at whose end the token was `halted`, -1 if never (`no_freeze`); with freeze every token has `halt_row <= N-1` because the forced last row halts everything |
| `<variant>/steps_tok` | [1,16] | f32 | expected rows per token `sum_n p_n * (n+1)` |
| `<variant>/prior` | [N] | f32 | the normalised geometric prior over the rows that ran: `halt_prior*(1-halt_prior)**n / sum` |

`dense_forward.json` -> `variants.<name>`: `loss`, `ce_term`, `kl_term`, `ponder_beta`,
`last_steps` (= `model.last_steps` = mean of `steps_tok`), `n_steps`, `cfg_overrides`,
`active_count_per_row`, `halted_count_after_row`, `halt_row_histogram`, `rows_with_empty_active_set`,
`min_abs_distance_of_1_minus_cum_to_halt_thresh` (the closest any running token comes to the halting
threshold; >= 0.003 in every variant, so f32 noise cannot flip a halting decision), flags for which
arrays are stored. Also top-level `idx`, `targets`. **`loss = ce_term + ponder_beta * kl_term`**,
where `ce_term = mean_t(sum_n p_n[t] * ce_n[t])` and
`kl_term = mean_t(sum_n P(logP - log prior_n))` with `P = clamp_min(p_n, 1e-8)` (the clamp also zeroes
the gradient of clamped entries) and the prior normalised over the rows that actually ran.

### `dense_infer.npz`

Inference forward (`model.eval()`, `targets=None`, `collect=True`, no_grad). Weights are those of
`dense_forward.npz`: scenario `perturbed` -> variant `perturbed`, `halting` -> `halting`, `allhalt`
-> `perturbed` with `halt.weight`/`halt.bias` of variant `halting_all`. Same `idx` as
`dense_forward.npz` (stored again). Each scenario runs the 16 tokens one-shot, then again in chunks
through `empty_caches()` with `pos_offset = start of chunk`; the chunked logits equal the one-shot
logits (max diff recorded, about 4e-6). Chunk plans are in `dense_infer.json`
(`perturbed`: 8+8; `halting`: single-token chunks at a token that halts on row 0 and one still
running on the last row; `allhalt`: 10+5+1).

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `idx` | [1,16] | i64 | input ids |
| `<sc>/oneshot/logits` | [1,16,265] | f32 | the returned `halted_logits`: per token the logits of the first row where `(1-cum) >= halt_thresh` (the forced last row catches the rest) |
| `<sc>/oneshot/steps` | [1,16] | f32 | `steps_used`: 1-based row at which each token stopped |
| `<sc>/oneshot/lam/<n>` | [1,16] | f32 | `lam` of row n (only rows that reach the end of the loop body) |
| `<sc>/oneshot/cum/<n>` | [1,16] | f32 | `cum` after row n |
| `<sc>/oneshot/halted/<n>` | [1,16] | u8 | `halted` after row n (`halted = halted OR newly`) |
| `<sc>/chunk<i>/logits` | [1,T,265] | f32 | logits for chunk i |
| `<sc>/chunk<i>/steps` | [1,T] | f32 | `steps_used` for chunk i |
| `<sc>/chunk<i>/offset` | [1] | i64 | `pos_offset` passed for chunk i |
| `<sc>/chunk<i>/halted/<n>` | [1,T] | u8 | `halted` after row n, for rows that reached the end of the loop body (rows skipped because every token had halted are absent; `dense_infer.json` lists them) |
| `<sc>/cache/<slot>/k` | [1,2,16,16] | f32 | cache of slot `slot` after all chunks: slot 0 = the prelude block, slot `1+n` = row n of the recurrent block; post-rope keys, 16 positions |
| `<sc>/cache/<slot>/v` | [1,2,16,16] | f32 | values of the same slot |

### `moe_shared.npz`

`PooledMLP(top_k=2, z_weight=1e-3)` over a plain resident `SharedPool` (8 experts, `d_ff 16`,
`depth 1`). See section 7 for the routing semantics. Shared expert weights are stored once at the top
level; every case has its own router, `depth_emb`, gate and inputs.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `w1` | [8,16,32] | f32 | stacked expert gate weights (`pool.experts[i].w1.weight`) |
| `w3` | [8,16,32] | f32 | stacked expert value weights |
| `w2` | [8,32,16] | f32 | stacked expert down weights |
| `n_experts` | [1] | i64 | 8 |
| `top_k` | [1] | i64 | 2 |
| `z_weight` | [1] | f32 | 1e-3 |
| `<case>/router` | [R,32] | f32 | router weight. Cases: `base`, `active`, `cap` have 8 rows; `wide` has 12 rows (the reference router is `max_experts` rows wide, only the first `n=8` are used; rows 8..11 get exactly zero gradient). |
| `<case>/depth_emb` | [32] | f32 | added to every token before the router (not before the experts). Not stored for `wide` (equal to `base/depth_emb`) |
| `<case>/gate` | [8] | f32 | per-expert gate. Not stored for `wide` (equal to `base/gate`) |
| `<case>/x` | [1,12,32] | f32 | input tokens (`base`, `wide`, `active` share the same `x`) |
| `<case>/g` | [1,12,32] | f32 | upstream gradient; loss for `grad/*` is `sum(out * g)` |
| `<case>/out` | [1,12,32] | f32 | output (not stored for `cap`, see `cap/out_stable`) |
| `<case>/aux` | [1] | f32 | `mlp.aux` (load-balance + z-loss) |
| `<case>/grad/x` | [1,12,32] | f32 | `d sum(out*g) / d x` (not stored for `cap`) |
| `<case>/grad/router` | [R,32] | f32 | gradient w.r.t. router weight (not stored for `cap`) |
| `<case>/grad/depth_emb` | [32] | f32 | gradient w.r.t. depth_emb (not stored for `cap`) |
| `<case>/grad/gate` | [8] | f32 | gradient w.r.t. gate (not stored for `cap`) |
| `<case>/grad/w1` | [8,16,32] | f32 | gradient w.r.t. the stacked expert gate weights (`base`, `active` only) |
| `<case>/grad/w3` | [8,16,32] | f32 | same for w3 (`base`, `active`) |
| `<case>/grad/w2` | [8,32,16] | f32 | same for w2 (`base`, `active`) |
| `<case>/grad_aux/x` | [1,12,32] | f32 | `d aux / d x` (all cases; aux does not depend on gate or experts, those gradients are exactly 0 and not stored) |
| `<case>/grad_aux/router` | [R,32] | f32 | `d aux / d router` |
| `<case>/grad_aux/depth_emb` | [32] | f32 | `d aux / d depth_emb` |
| `<case>/route_idx` | [N,2] | i64 | chosen experts per token (`N` = tokens that were routed: 12, or 8 for `active`), highest router probability first |
| `<case>/route_w` | [N,2] | f32 | the final mixing weights: top-k probabilities normalised to sum 1, then multiplied by `gate[idx]` |
| `<case>/logits` | [N,8] | f32 | `router(x + depth_emb)[:, :8]` |
| `<case>/probs` | [N,8] | f32 | softmax of `logits` |
| `<case>/min_margin_top1_top2` | [1] | f32 | smallest probability gap between a token's best and second expert (decides `frac`) |
| `<case>/min_margin_topk_boundary` | [1] | f32 | smallest gap between the k-th and (k+1)-th expert (decides the selection) |
| `<case>/use` | [8] | f32 | `pool.use` after the call: how often each expert was picked (counts all k picks) |
| `<case>/age` | [8] | f32 | `pool.age` after the call (+1 per call-site invocation) |
| `<case>/pressure` | [1] | f32 | `pool.pressure` EMA after the call: `0.1 * (1 - mean_t(sum of the top-k probs before normalisation))` |
| `<case>/want_k` | [1] | f32 | `pool.want_k` EMA after the call: `0.1 * (mean_t(count(cumsum(sorted probs) < 0.9)) + 1)` |
| `<case>/last_route` | [8] | f32 | `mlp.last_route` = `use` increments / total picks |
| `<case>/dropped` | [1] | i64 | capacity-dropped assignments of this call (0 unless `cap`) |
| `<case>/routed` | [1] | i64 | assignments considered (counted BEFORE drops) = tokens routed * k |
| `active/active` | [1,12] | u8 | the `active` mask (tokens 1, 4, 5, 9 are inactive): inactive outputs are exactly 0, they take no capacity and are excluded from aux, `use`, `routed` |
| `cap/capacity_factor` | [1] | f32 | 1.5 |
| `cap/limit` | [1] | i64 | `max(1, ceil(1.5 * routed / 8))` = 5 |
| `cap/counts_before` | [8] | i64 | assignments per expert before dropping |
| `cap/kept_counts` | [8] | i64 | `min(counts_before, limit)` |
| `cap/kept_mask_stable` | [12,2] | u8 | which (token, rank) assignments are kept under the stable order convention |
| `cap/out_stable` | [1,12,32] | f32 | output under the stable convention (dropped assignments contribute nothing) |
| `cap/grad_stable/x` | [1,12,32] | f32 | gradient of `sum(out_stable * g)` w.r.t. x |
| `cap/grad_stable/router` | [8,32] | f32 | w.r.t. router |
| `cap/grad_stable/depth_emb` | [32] | f32 | w.r.t. depth_emb |
| `cap/grad_stable/gate` | [8] | f32 | w.r.t. gate |
| `cap/grad_stable/w1` | [8,16,32] | f32 | w.r.t. w1 |
| `cap/grad_stable/w3` | [8,16,32] | f32 | w.r.t. w3 |
| `cap/grad_stable/w2` | [8,32,16] | f32 | w.r.t. w2 |

### `moe_paged_forward.npz`

`PooledMLP` routing into `PagedPool(resident=4)` over 8 experts whose files were written with
`Tiers._to_disk`; fresh empty card, `pool.begin_text(explore)` then one `mlp(x)` call
(`capacity_factor 0`, `top_k 2`). All three cases share the 8 experts and the gate. Case `a`:
`explore=False`. Case `b`: `explore=True` with `pool.recent` preset (`explore_bias 0.65`,
`explore_steps 1000`); the exploration bonus changes which experts are admitted (different `slots`
than `a`). Case `c`: a router/`depth_emb` that makes every token rank experts 1 and 4 first, so only
two experts get mass > 0 and two slots stay empty (-1) and masked.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `expert/w1` | [8,16,32] | f32 | all 8 experts' weights as stored in the expert files |
| `expert/w3` | [8,16,32] | f32 | same |
| `expert/w2` | [8,32,16] | f32 | same |
| `resident` | [1] | i64 | 4 |
| `<case>/router` | [8,32] | f32 | router weight (one row per EXPERT, not per slot) |
| `<case>/depth_emb` | [32] | f32 | added to the router input |
| `<case>/gate` | [8] | f32 | per-expert gate (pool.gate) |
| `<case>/x` | [1,12,32] | f32 | input |
| `<case>/g` | [1,12,32] | f32 | upstream gradient |
| `<case>/out` | [1,12,32] | f32 | output |
| `<case>/aux` | [1] | f32 | `mlp.aux` (n = 4 slots in the formula) |
| `<case>/slots` | [4] | i64 | slot -> expert id after admission (-1 = empty) |
| `<case>/admitted_mask` | [4] | u8 | slots holding an expert this forward admitted |
| `<case>/grad/x` | [1,12,32] | f32 | gradient of `sum(out*g)` w.r.t. x |
| `<case>/grad/router` | [8,32] | f32 | w.r.t. router (rows of non-admitted experts are exactly 0) |
| `<case>/grad/depth_emb` | [32] | f32 | w.r.t. depth_emb |
| `<case>/grad/gate` | [8] | f32 | w.r.t. gate (zero for experts not on the card) |
| `<case>/grad/slot_w1` | [4,16,32] | f32 | w.r.t. the slot tensor `pool.w1` (slot order) |
| `<case>/grad/slot_w3` | [4,16,32] | f32 | w.r.t. `pool.w3` |
| `<case>/grad/slot_w2` | [4,32,16] | f32 | w.r.t. `pool.w2` |
| `<case>/grad_aux/x` | [1,12,32] | f32 | `d aux / d x` |
| `<case>/grad_aux/router` | [8,32] | f32 | `d aux / d router` |
| `<case>/grad_aux/depth_emb` | [32] | f32 | `d aux / d depth_emb` |
| `<case>/route_idx` | [12,2] | i64 | chosen EXPERT ids per token (not slots) |
| `<case>/route_slot` | [12,2] | i64 | the same choices as slot indices |
| `<case>/route_w` | [12,2] | f32 | final mixing weights: `probs.gather(idx)` normalised to sum 1, times `gate[expert]` |
| `<case>/admit_mass` | [8] | f32 | the `mass` vector the router handed to `pool.admit` (softmax over ALL 8 experts of `router(x+depth_emb [+bias])`, top-2 values accumulated per expert, summed over tokens) |
| `<case>/admit_merit` | [8] | f32 | the `merit` vector (same without the exploration bonus); equals `admit_mass` when `admit_had_merit` is 0 |
| `<case>/admit_had_merit` | [1] | i64 | 1 if the call passed a separate `merit` (exploring), else 0 |
| `<case>/probs` | [12,4] | f32 | softmax over the 4 slots with non-admitted slots masked to -inf (probability exactly 0) |
| `<case>/logits_unmasked` | [12,4] | f32 | `router.weight[slots (empty -> row 0)]` applied to `x + depth_emb`, before masking |
| `<case>/min_margin_top1_top2` | [1] | f32 | smallest gap between best and second slot probability |
| `<case>/min_margin_topk_boundary` | [1] | f32 | smallest gap between k-th and (k+1)-th slot probability |
| `<case>/bias` | [8] | f32 | exploration bonus `0.65 * exp(-recent / (resident / n))` (only `b`) |
| `<case>/recent_before` | [8] | f32 | `pool.recent` before `begin_text` |
| `<case>/recent_after` | [8] | f32 | `pool.recent` after the forward (decayed by `1 - 1/explore_steps`, plus `1/explore_steps` per admitted expert, only when exploring) |
| `<case>/last_seen` | [8] | f32 | prune clock: `segments` for the experts the router alone would have admitted |
| `<case>/admits` | [8] | f32 | times each expert was brought onto the card |
| `<case>/ever` | [8] | u8 | whether each expert was ever admitted |
| `<case>/use` | [8] | f32 | `pool.use` after the call (per expert id) |
| `<case>/age` | [8] | f32 | `pool.age` after the call |
| `<case>/last_route` | [4] | f32 | `mlp.last_route` per slot |
| `<case>/pressure` | [1] | f32 | `pool.pressure` |
| `<case>/want_k` | [1] | f32 | `pool.want_k` |
| `<case>/loads` | [1] | i64 | `pool.loads` after the call |
| `<case>/swaps` | [1] | i64 | `pool.swaps` after the call |
| `<case>/segments` | [1] | i64 | `pool.segments` after the call (1) |

### `adamw_paged.npz`

Four real training steps of the paged tiny model with the reference's optimiser set-up
(`train.py` paged trainer ~L745): `AdamW([trunk group, pool group], betas=(0.9, 0.95), eps=1e-8)`,
**weight decay 0.1 on every parameter of both groups** (norm weights, `halt.*`, gate, `depth_emb`
included), `lr_pool = 0.01`, `lr_trunk = 0.001` (= 0.1x), `pool.attach_optimiser(opt)`, then per
step: `zero_grad`, forward `loss = model(x, y)[1] + 0.01 * model.pool_aux()`, `backward`,
`clip_grad_norm_(model.parameters(), 1.0)`, `opt.step()`. Model: pooled tiny config (8 experts on
disk, 4 slots, `ram_capacity 3`, `pool_capacity_factor 1.5` with the stable-argsort convention),
trunk perturbed (halting weights as variant `halting`), router and `depth_emb` at the reference
default init, gate random in [0.6, 1.4], experts N(0, 0.25). Trunk group = the 16 tensors listed
in `config.trunk_param_names` of `adamw_paged.json`, pool group = `recur.0.mlp.depth_emb`,
`recur.0.mlp.router.weight`, `pool.w1/w3/w2`, `pool.gate`.

Steering: before each step the script sets `pool.recent` to 6 for every expert except a "favoured"
set (0), so the exploration bonus (0.65) makes the forward admit exactly that set:
step 0 `{0,1,2,3}` (cold card), step 1 `{0,1,2,4}` (expert 3 parked, 4 loaded), step 2 `{0,1,3,4}`
(expert 3 comes back: moments restored from the RAM tier, exact f32), then **two held-out forwards**
(`eval0`, `eval1`: `model.eval()`, `no_grad`, no exploration, the router's own choice; the second
evicts experts the first loaded without any step, which are parked for free), step 3 `{2,5,6,7}`
(expert 2 comes back from DISK: bf16-rounded moments; experts 5, 6, 7 arrive without moments at
Adam step 4 and start from zero moments).

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `init/<param>` | per param | f32 | trainable tensors before step 0, EXCEPT the slot tensors (the card is empty: they are zeros) |
| `expert/w1` | [8,16,32] | f32 | initial expert files |
| `expert/w3` | [8,16,32] | f32 | initial expert files |
| `expert/w2` | [8,32,16] | f32 | initial expert files |
| `s<k>/idx` | [1,16] | i64 | step k input ids |
| `s<k>/targets` | [1,16] | i64 | step k targets |
| `s<k>/recent_set` | [8] | f32 | the `pool.recent` vector assigned before the forward of step k |
| `s<k>/slots_before` | [4] | i64 | card before the forward (slot -> expert, -1 empty) |
| `s<k>/slots` | [4] | i64 | card used by this step (after admission; also at backward and optimiser step) |
| `s<k>/admit_mass` | [8] | f32 | `mass` passed to `pool.admit` by the first pass of the forward |
| `s<k>/admit_merit` | [8] | f32 | `merit` passed with it |
| `s<k>/bias` | [8] | f32 | exploration bonus used for the whole forward |
| `s<k>/loss` | [1] | f32 | total loss `loss_model + 0.01 * aux` |
| `s<k>/loss_model` | [1] | f32 | the model's PonderNet loss |
| `s<k>/aux` | [1] | f32 | `model.pool_aux()`: mean of the sites' LAST `aux` (here: the aux of the last row that ran the pooled block) |
| `s<k>/grad_norm` | [1] | f32 | `clip_grad_norm_` return value (norm BEFORE clipping, over all parameters) |
| `s<k>/clip_coef` | [1] | f32 | `min(1, 1.0 / (grad_norm + 1e-6))`; every gradient is multiplied by it (also when it is 1) |
| `s<k>/grad/<param>` | per param | f32 | gradients BEFORE clipping, all 22 trainable tensors (slot tensors in slot order) |
| `s<k>/param/<param>` | per param | f32 | every trainable tensor AFTER optimiser step k |
| `s<k>/pre_param/<param>` | [4,..] | f32 | the three slot tensors as they were when the optimiser step started (the loaded expert weights) |
| `s<k>/pre_exp_avg/<param>` | [4,..] | f32 | slot `exp_avg` AFTER the pool's `_own_moments` pre-hook, i.e. exactly what AdamW consumed (steps 1..3) |
| `s<k>/pre_exp_avg_sq/<param>` | [4,..] | f32 | slot `exp_avg_sq`, same moment |
| `s<k>/exp_avg/<param>` | per param | f32 | `exp_avg` after step k. Slot tensors at every step; every other parameter only after step 3 (the last), earlier values are the standard recurrence on the stored grads |
| `s<k>/exp_avg_sq/<param>` | per param | f32 | `exp_avg_sq`, same coverage |
| `eval<i>/idx` | [1,16] | i64 | held-out forward inputs (between step 2 and step 3) |
| `eval<i>/targets` | [1,16] | i64 | their targets |
| `eval<i>/slots_after` | [4] | i64 | card after that forward |
| `file/<e>/w1` | [16,32] | f32 | expert e's file after `pool.flush()` at the end: weights (fp32) |
| `file/<e>/w3` | [16,32] | f32 | same |
| `file/<e>/w2` | [32,16] | f32 | same |
| `file/<e>/<m>_bits` | [..] | i16 | the raw stored array: bfloat16 bits as int16; `<m>` is one of `w1_m, w1_v, w3_m, w3_v, w2_m, w2_v` (Adam first/second moment of that tensor) |
| `file/<e>/<m>` | [..] | f32 | the same moment unpacked to f32 (bf16 round-to-nearest-even of the fp32 slot moment) |

`adamw_paged.json`: `config` (hyper-parameters, group membership), and per step: `slots_before`,
`slots_used`, `own_flags_at_optimizer_step`, `adam_step_all_params` (= k+1, one global count),
`tier_log` (every `Tiers.fetch/put/_to_disk` call in order: `["fetch", expert, "ram"|"disk",
counted]`, `["put", expert, has_moments]`, `["write_back", expert, has_moments]`; a fetch with
`counted=false` is the moment fetch of `_own_moments`), `ram_order` (LRU order, oldest first),
`dirty`, `loads_total`, `swaps_total`, `recent_after`, `last_seen`, `admits`, `active_count_per_row`,
`last_pooled_row`, `dropped_routed_this_step`, and the decision margins
(`min_gap_topk_boundary_in_routing_scores` >= 1e-3, `min_gap_top1_top2_in_routing_scores`,
`min_abs_distance_of_1_minus_cum_to_halt_thresh`) that tell you how far f32 noise is from flipping a
discrete choice. `evals_between_step2_and_step3` holds the same bookkeeping for the two held-out
forwards; `flush_tier_log` the final `pool.flush()`; `files` which arrays each expert file holds.

### `train_step_dense.npz`

Dense `perturbed` weights (from `dense_forward.npz`), two training steps at `train_step_index`
98 and 100, `lr_base 0.05`, `warmup 100`, `lr = lr_base * min(1, (index+1)/100)`, trunk
`lr * 0.1` (all dense parameters are "trunk"), `betas (0.9, 0.95)`, `eps 1e-8`, AdamW's own step
counter is 1 then 2 (independent of the train step index). Two runs:

* `wd_all`: one group, weight decay 0.1 on every tensor (the paged trainer's construction),
  `clip 1.0` (clipping active, `clip_coef` about 0.09). Per-step grads and params stored.
* `wd_split`: two groups, weight decay 0.1 on tensors with `dim >= 2` and 0.0 on vectors/biases
  (the batch trainer, `train.py` ~L164), `clip 100` (inactive, `clip_coef = 1`). Only the final
  parameters are stored.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `idx/<n>` | [1,16] | i64 | input ids of step n |
| `targets/<n>` | [1,16] | i64 | targets of step n |
| `<run>/s<k>/loss` | [1] | f32 | loss of step k (dense model: no pool aux) |
| `<run>/s<k>/grad_norm` | [1] | f32 | gradient norm before clipping |
| `<run>/s<k>/clip_coef` | [1] | f32 | applied clip coefficient |
| `<run>/s<k>/lr_trunk` | [1] | f32 | learning rate used (`0.05 * factor * 0.1`) |
| `<run>/s<k>/warmup_factor` | [1] | f32 | `min(1, (train_step_index+1)/100)` |
| `<run>/s<k>/param/<param>` | per param | f32 | parameters after step k (`wd_all`: both steps; `wd_split`: step 1 only) |
| `<run>/s<k>/grad/<param>` | per param | f32 | gradients before clipping (`wd_all` only) |

### JSON fixtures (`tokenizer`, `moe_admission`, `decode`, `dense_forward`, `dense_infer`, `adamw_paged`, `train_step_dense`, `config`)

JSON schemas (no key tables, see section 7 for the semantics):

* `tokenizer.json`: `vocab_size` 265; `specials` (text, id) in the reference's SPECIALS order;
  `encode_cases[]` `{text, ids, utf8_hex, roundtrip_decode_equals_text}` (61 cases: every marker,
  look-alikes such as `<g 1850 1-0>`, `<G>`, `< g>`, `<<think>>`, `<think`, adjacent markers,
  unicode, emoji, NUL, BOM); `decode_cases[]` `{name, ids, input_bytes_hex (null when markers or
  out-of-range ids are present), text, text_utf8_hex, n_replacement_chars}` (invalid UTF-8,
  truncated sequences, markers flushing a partial sequence, ids outside 0..264 ignored);
  `token_to_id_cases[]` `{token, id|null}`.
* `moe_admission.json`: `scenarios[]` `{name, description, config {n_experts, resident,
  ram_capacity, explore_bias, explore_steps}, steps[]}`. A step is `{op, ...args, state}` with
  `op` in `init`, `begin_text {explore}`, `begin_forward {explore}`, `admit {mass, merit,
  returned_loads}`, `set_recent {values}`, `set_explore {explore_bias, explore_steps}`; `state` is
  everything observable after that op: `slots, admitted, merited, admitting, admitted_mask,
  last_seen, admits, ever, loads, swaps, segments, recent, bias, exploring, vote, merit_vote, voted,
  ram_order, tiers {reads, hits, evictions, writebacks}`. `dying_cases[]` `{name, note, born,
  last_seen, segments, now, trial, dying_at, dying[8], saturation_idle}`.
* `decode.json`: `cases[]` `{name, note, params, prefix_len, prefix, logits[265], expected_next,
  adjusted_top8, adjusted[265], unadjusted_argmax, argmax_is_a_tie}`.
* `dense_forward.json`, `dense_infer.json`, `adamw_paged.json`, `train_step_dense.json`,
  `config.json`: described above and in section 7.

## 7. Reference behaviour the Rust port must replicate

(All verified against the reference by the fixtures; "Q" numbers are just handles.)

**Tokenizer** (`minagi/tokenizer.py`)
* Q1 Vocabulary 265 = 256 bytes + 9 markers `<think> </think> <user> </user> <bot> </bot> <g> </g>
  <|endoftext|>` = ids 256..264. `encode`: one regex alternation in that order, leftmost match,
  everything else UTF-8 bytes. Matching is exact and case-sensitive: `<g 1850 1-0>`, `<G>`, `< g>`,
  `<think` are plain bytes. A marker inside another is found left to right (`<t<think>hink>`).
* Q2 `decode`: ids outside 0..264 are **silently dropped** (including negatives); bytes are buffered
  and decoded with `errors="replace"` (one U+FFFD per maximal invalid subsequence, i.e.
  `String::from_utf8_lossy`); a marker id first flushes the buffer, then appends the marker text.
  A partial UTF-8 sequence split by a marker is therefore replaced, and the continuation byte after
  the marker becomes its own U+FFFD.
* Q3 `token_to_id`: marker text -> id; a string with exactly one UTF-8 byte -> that byte; anything
  else (including `""`, multibyte chars, `"ab"`) -> `None`.

**Dense model** (`model.py`, `recur.py`)
* Q4 Pre-norm blocks: `x += attn(rms(x))`, `x += mlp(rms(x))`. RMSNorm eps is 1e-6 (inside the rsqrt).
* Q5 RoPE tables are computed in float32 (see `ops_rope.npz`); positions are absolute
  (`pos_offset + i`). Attention scale is `1/sqrt(16)`; the cached keys are stored post-rope.
* Q6 `adapter` is applied to `cat([h, x], -1)` (**h first**, then the prelude output `x`); default
  init is `[I, I]`. `h` starts at 0.
* Q7 Row loop (training with targets): `if bptt_window > 0 and n < n_steps - bptt_window: h = h.detach()`
  *at the start of row n*; `active = ~halted` only when `halt_freeze and n > 0`; if no token is active
  the block is skipped and `h` is unchanged (otherwise `h = where(active, hn, h)`). The recurrent
  block still runs attention on ALL positions and, in the **dense** model, its MLP on all tokens
  (`active` only reaches a pooled MLP); the frozen tokens' new values are discarded by the `where`.
* Q8 Then, for every row, including rows where nobody was active: `yf = ln_f(h)`, `logits_n =
  head(yf)` (tied to `tok_emb`), `lam = sigmoid(halt(yf))`; last row (`n == n_steps-1`) -> `lam = 1`;
  `n < min_steps-1` -> `lam = 0`; `p_n = cum * lam`; `cum *= 1 - lam`; CE per token in f32;
  `halted = halted OR ((1 - cum) >= halt_thresh)` (freeze only; evaluated after the last row too). A halted token
  keeps contributing `p_n * ce` on later rows with its frozen state (its `p_n` is small but not 0).
* Q9 Loss: `mean_t(sum_n p_n ce_n) + ponder_beta * mean_t(sum_n P (log P - log prior_n))`,
  `P = clamp_min(p_n, 1e-8)`, `prior` = geometric(`halt_prior`) normalised over the rows that ran.
  `last_steps = mean_t(sum_n p_n (n+1))`. The returned logits are `sum_n logits_n * p_n`.
* Q10 Inference (`targets=None`): same loop, but no CE/KL; a token keeps the logits of the first row
  where `(1 - cum) >= halt_thresh` (`halted_logits`, `steps_used`); if every token has halted the row is
  skipped entirely. **With KV caches** the skipped rows still extend their cache slot (once-computed
  `carried` keys/values of the frozen state), so every slot always holds exactly the processed
  positions. `halt_freeze` needs `n_recur == 1` and `n_coda == 0` (ValueError otherwise).
* Q11 Cache slots: index 0..n_prelude-1 prelude blocks, then `n_recur + n_coda` slots per row, row by
  row. `generate` (not covered by a fixture) re-reads the last `block // 2` tokens from offset 0 with
  empty caches when `offset + len(cur) > block`.

**MoE routing** (`pool.py`, `PooledMLP._route`)
* Q12 `logits = router(x + depth_emb)[:, :n]` (f32). `depth_emb` is added to the router input only.
  `n` = number of experts (resident pool) or number of slots (paged). `probs = softmax(logits)`;
  `idx` = top-k of `probs` (ties -> lowest index); weights = `probs[idx]` normalised to sum 1, then
  multiplied by `gate[idx]` (gate applied AFTER normalisation).
* Q13 `aux = n * sum_e(frac_e * mean_t(probs[:, e])) + 1e-3 * mean_t(logsumexp(logits)^2)` where
  `frac_e` is the share of tokens whose **top-1** is `e` (a constant: no gradient). Aux does not
  depend on gates or experts.
* Q14 Dispatch: `limit = max(1, ceil(capacity_factor * N*k / n))` per expert when `capacity_factor
  != 0`, applied only if the busiest expert exceeds it; the first `limit` entries of each expert's
  stable-sorted run are kept; `routed += N*k` before dropping, `dropped += number dropped`. For the
  paged pool `n` is the slot count (4), not the pool size. `out` = `sum_r w[t,r] * SwiGLU_e(x[t])`
  (index_add). Inactive tokens (`active=False`) are removed first: they take no capacity and no
  statistics, and their output row is exactly 0 (`N` and `n` in the formulas are then the active
  token count).
* Q15 `pool.use[e] += picks`, `pool.age += 1`, `pressure = 0.9 pressure + 0.1 (1 - kept)`,
  `want_k = 0.9 want_k + 0.1 want`, with `kept = mean_t(sum_k probs[idx])` BEFORE normalisation and
  `want = mean_t(count(cumsum(sort_desc(probs)) < 0.9)) + 1`.
* Q16 **`model.pool_aux()` is the mean over call sites of the site's LAST `aux`**. With one site and
  `halt_freeze`, only the last row on which the pooled block actually ran contributes (earlier rows'
  aux tensors are overwritten and carry no gradient; a skipped all-halted row leaves the previous
  one). `adamw_paged.json` records `last_pooled_row`.

**Paged pool** (`paged.py`, `moe_admission.json`, `moe_paged_forward.npz`)
* Q17 Router rows belong to experts. While a forward may still admit experts, every token ranks the
  WHOLE pool: `z = router.weight[:E] @ (x + depth_emb)` (f32); `mass = sum over tokens of the top-k
  entries of softmax(z + bias)` accumulated per expert (`bias` only when exploring; the bonus is
  INSIDE the softmax), `merit` = the same with `softmax(z)` (only passed when exploring). No gradient.
  Then `pool.admit(mass, merit)`. Afterwards routing uses
  `router.weight[slots]` (an empty slot reads row 0), `admitted_mask` masks non-admitted slots with -inf
  BEFORE the softmax, `gate[slots]` weights. With exploration the top-k is taken on
  `logits.detach() + bias[slots]`, but the weights are `probs.gather(idx)` (router softmax only).
* Q18 `admit(mass, merit)`: on a forward's FIRST call the pass's mass is added to the text's vote
  (`_vote`, `_merit_vote`) and the forward is admitted by the whole vote; later calls in the same
  forward use only their own mass. `merited` = top-`resident` experts of the merit vote with value > 0
  (`last_seen[e] = segments` for each; `free = resident - |merited|`). Admitted: the top
  `resident - |admitted|` experts of the vote with value > 0 not already admitted - **including experts that are already resident** -
  so `recent[new] += 1/explore_steps` (exploring only) counts them too. A newcomer takes a slot
  holding nothing admitted this forward: empty slots first, then the occupant with the smallest
  `last_seen`, ties -> lowest slot index (Python stable sort). Newcomers are placed in
  mass-descending order. `loads` counts experts that were not resident; `admits[e] += 1` only for
  those; `swaps += 1` if the slot layout changed; `ever[new] = True`. `argsort(descending)` on ties is
  unspecified, so no scenario contains ties among the ranked entries.
* Q19 `begin_text(explore)` = clear the vote + `begin_forward(explore)` + `segments += 1`.
  `begin_forward(explore)`: clears `admitted`, `merited`, `voted`, mask; if `explore and explore_bias
  > 0`: `bias = explore_bias * exp(-recent / (resident / n_experts))` computed from `recent` BEFORE it
  decays, then `recent *= 1 - 1/explore_steps`; else `bias = None`. A model forward is a `begin_text`
  when `caches is None or pos_offset == 0`, otherwise a `begin_forward`; `explore = model.training and
  grad enabled`.
* Q20 `dying()` (f32): `per_step = segments / max(now, 1)`; `window = trial * per_step`;
  `idle = clamp_min(segments - last_seen, 0)`; `born_seg = clamp_min(now - born, 0) * per_step`;
  `frac = min(idle, born_seg) / window`; experts with `now - born < trial` read 0; all zeros if
  `n == 0`, `trial <= 0`, `now <= 0` or `window <= 0`.
* Q21 Parking/moments (`adamw_paged`): an expert leaves the card -> if any optimiser step happened
  since it was loaded (`_loaded_at` compares the slot tensors' versions and the step count), it is
  `put` into the RAM tier as `{w1,w3,w2, w*_m, w*_v}` (moments from the optimiser if the slot held its own
  moments, else from its previous entry) and marked dirty; if nothing stepped it, nothing is copied
  or written. The RAM tier is an LRU of `ram_capacity` entries; the oldest is written to disk if dirty
  when the capacity is exceeded; every fetch moves the entry to most-recent. Disk files are
  `e%05d.npz` with `w1,w3,w2` fp32 and `w1_m,w1_v,w3_m,...` as **bf16 bits in int16** (round to nearest
  even); a file may have no moments (as written by `store.save`/`create.py`: the `create.py`
  docstring promises zeroed moments but the code writes none). Moments are fetched only just before an
  optimiser step (`_own_moments`): every slot whose expert is not already holding its own moments gets
  them from RAM (exact f32) or disk (bf16 rounded), or zeros when the entry has none.
* Q22 **Adam step count is global**: the slot tensors are ordinary parameters, so bias correction uses
  `t` = optimiser steps taken (1..4 here), also for an expert that just arrived with zero moments.
  AdamW also steps (momentum + decay) slots whose expert received zero gradient.

**Optimiser** (`torch.optim.AdamW`, single-tensor CPU path, f32)
* Q23 `p *= 1 - lr*wd; m += (g - m)*(1 - b1)` (lerp); `v = v*b2 + g*g*(1 - b2)`; `step_size = lr / (1 -
  b1**t)`; `denom = sqrt(v) / sqrt(1 - b2**t) + eps`; `p -= step_size * m / denom`; `b1=0.9, b2=0.95,
  eps=1e-8`. `verify_golden.py::adamw_np` is an executable numpy version (agrees to 6e-8).
* Q24 `clip_grad_norm_(all params, 1.0)`: total = sqrt(sum of squared per-tensor 2-norms) over every
  parameter with a gradient; `coef = min(1, 1.0 / (total + 1e-6))` multiplies every gradient, always.
* Q25 The paged trainer decays every parameter (including norm weights, biases, gates, `depth_emb`,
  slot tensors); the older batch trainer (`train.py` ~L164) exempts tensors with `dim < 2`. Both are
  in `train_step_dense.npz`. Warm-up factor in the task description: `min(1, (step+1)/100)`.

**Decode** (`decode.py`)
* Q26 `pick_next(logits [1,265], prev_ids [1,n], temperature=0, adapt_strength=2.5, adapt_decay=0.88,
  adapt_window=64)`: `tail = prev_ids[-min(n,64):]`; `w = 0.88 ** arange(n-1, -1, -1)` (float32,
  newest = 1); `trace = scatter_add(zeros(265), tail, w)` (accumulated in position order);
  `adjusted = logits - 2.5 * trace`; `next = argmax(adjusted)` (lowest index on exact ties, as torch
  does on CPU). Empty prefix: no adaptation. A prefix longer than 64 ignores the oldest tokens
  entirely. Marker ids take part like any other id. `rep_penalty != 1` (not used by default):
  `l > 0 ? l / p : l * p` for every unique id in the prefix, applied after the trace.

## 8. Caveats, approximations and things deliberately partial

* `ops_attention.npz` cache cases treat the cached k/v as constants; gradients are for the new
  positions only (an uncached pass over all positions has additional paths).
* `dense_forward`: `logits_n/*` and `mixture_logits` only for `default_init`, `perturbed`, `halting`;
  `min_steps2` has no gradients; `bptt` / `min_steps2` / `n2` / `no_freeze` reuse `perturbed` weights
  (all to stay under the 6 MB budget). A finite-difference check cannot see `detach`, so `bptt` is only
  checked structurally (same loss as `perturbed`, different gradients).
* `moe_shared/cap`: the reference's own `out` and gradients under a capacity drop depend on torch's
  unstable argsort and are NOT stored; use `out_stable` / `grad_stable` / `kept_mask_stable`
  (stable convention) and the order-free counters. `aux`, routes and counters are order-free.
* `adamw_paged`: moments of non-slot parameters are stored only after the last step (steps 0..2 are
  the plain recurrence on the stored gradients); the card is steered through `pool.recent`, not by
  doctoring weights; learning rates (0.01 / 0.001) are larger than config.yaml's 3e-4 so the updates are
  visible. Discrete decisions (routing, admission, halting) are far from ties (margins in the json),
  but a Rust test should still teacher-force `slots`/`halted` if it wants to isolate the optimiser.
* `decode.json` stores logits as shortest-f32 decimals; `adjusted` is the reference's own local after
  all adjustments.
* Not covered by any fixture: `PagedPool.add_experts/prune`, `AutoGrow`, `generate()`'s context
  re-read, `contrastive_generate`, sampling with temperature / top-k / top-p, store/manifest I/O
  (the `.npz` expert-file layout is documented in Q21 and exercised by `adamw_paged`), bf16 autocast
  (the reference's `precision.amp()` is a no-op on CPU float32), `train_steps_mean > 0` (Poisson depth
  sampling), `n_recur > 1` / `n_coda > 0` (the tiny config has 1 and 0, which `halt_freeze` requires),
  `pool_depth > 1` (residually stacked SwiGLU blocks inside an expert), an expert pool smaller than the
  card, and `top_k` larger than the number of admitted experts: then the masked (probability 0) slots
  are still picked as zero-weight fillers by `topk`, their stale weights multiply weight 0, and
  `use` is credited to the expert id the slot maps to (row 0 for an empty slot).
* Nothing here is nondeterministic for a given torch/numpy build. A different build may change
  float32 rounding at the 1e-7 level; regenerate with `dump_model.py` and review the diff.

## 9. Verification (`tools/golden/verify_golden.py`)

Runs in about 5 seconds and exits non-zero on any failure:

1. regenerates twice in fresh processes: byte-identical, and identical to the committed fixtures;
2. the `.npz` writer is byte-identical to `numpy.savez`;
3. float64 numpy re-implementations (no torch) of RMSNorm, RoPE, attention (+cache), SwiGLU, the
   whole MoE layer (resident and paged, with active mask and capacity), the whole dense recurrent
   model and the whole paged pooled model (halting, active masks, capacity drops, exploration
   bias, "aux of the last pooled row") reproduce the stored outputs (`out`, `aux`, loss, every
   per-row array, halting decisions, admission mass) and **finite differences** (central, eps 1e-6)
   match the stored gradients (RMSNorm, attention, SwiGLU, MoE: every gradient group; dense model: a
   sample of every parameter in 7 of the 8 variants; paged model: a sample of all 22 trainable
   tensors at every one of the 4 steps of `adamw_paged.npz`);
   a plain-python re-statement of `PagedPool.admit/begin_text/begin_forward` (README Q17-Q19) and of
   `dying()` replays every scenario of `moe_admission.json` state by state (slots, admitted, merited,
   last_seen, admits, ever, loads, swaps, segments, recent, bias, vote, RAM LRU order, tier counters);
4. a numpy float32 AdamW replay (clipping, group learning rates, restored moments) reproduces the
   stored parameter trajectories of `adamw_paged.npz` and `train_step_dense.npz`, the stored file
   moments are the bf16 rounding of the last slot moments, and the final expert files equal the last
   resident weights;
5. `decode.json`'s `adjusted` vectors are reproduced by a numpy `pick_next`;
6. every key of every `.npz` is documented above (shape and dtype checked).
