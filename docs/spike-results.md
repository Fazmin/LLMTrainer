# Engine spike results: candle on Apple Silicon (S1-S7)

Go/no-go spikes for the Rust re-implementation of `volotat/mini-AGI` (byte-level recurrent transformer with halting + paged
mixture-of-experts) on **candle**. Everything below is measured, not estimated, unless a row says "extrapolated". The code is in
`crates/minagi-core/src/spike/` (helpers) and `crates/minagi-cli/src/bin/spike_s*.rs` (one driver per spike);
`crates/minagi-cli/scripts/run_spikes.sh` regenerates the raw outputs.

* Machine: Apple M4 Pro (14 CPU cores, 20 GPU cores), 24 GB unified memory, macOS 26.6.2, Metal 4.
  `MTLDevice.recommendedMaxWorkingSetSize` = 18186 MiB (17.8 GiB).
* Toolchain: rustc 1.93.1; candle **git rev 5ba5d5b** (= 0.11.0 plus fixes from main, 2026-09-28; the root `Cargo.toml` patch applies unchanged to
  all four candle crates and `Cargo.lock` resolves a single `objc2` 0.6.4); numpy 2.5.3, ml_dtypes 0.6.0 for S7.
* Build profile: the root `release` profile (thin LTO, codegen-units 1) for every timing. **Use release for any Metal timing**: in the
  dev profile objc2's runtime signature verification makes every Metal call ~2.3x slower (3.2 vs 1.4 us per op) and the autorelease leak ~8x larger (S1).
* **How the timings were taken.** This was a shared development machine; during part of the session the 1-minute load average was 40-190
  (other builds, Python training jobs). Every timing run was gated on a quiet-machine probe (a fixed GEMM on Metal and on CPU; reference 1.1 ms / 1.8 ms),
  the first full pass was partly contended, and the affected benchmarks (S1 GEMM/launch, S3/S4 timings, the whole S5 matrix, per-buffer sweep,
  Small, read-back, checkpoint overhead) were **re-run in quiet windows, three times each** where it matters. The tables below are the quiet runs
  (ranges are over the repeats). CPU numbers are the most load-sensitive.

## 0. Decision

**GO-WITH-CHANGES: train on Metal, in f32, with the changes in section 10.** No spike found a blocker; two candle defects need workarounds
(`scatter_add` backward, fused ops without gradient) and one engineering rule is critical (detach everything carried across steps).

| Spike | Criterion | Result | Verdict |
|---|---|---|---|
| S1 Metal basics | flat memory with an `autoreleasepool` | flat over 200k iterations with a pool; without one the process leaks ~0.2 KiB per iteration in release builds (+41 MiB / 200k) and **1.8 KiB in debug builds** (+349 MiB); RSS alone hides it | PASS (pool mandatory) |
| S2 op-zoo parity | Metal vs CPU grad rel. err < 1e-4 | 41 ops + R2 + FD: worst gradient rel. err **3.3e-7**; R2 confirmed (fused ops give **absent** gradients); #3847 repro 0 NaN in 6/6 processes on the pin | PASS (candle `scatter_add` backward is wrong, workaround tested) |
| S3 MoE | dense == sparse grads, Metal == CPU | agree to ~1e-6 with and without capacity drops; sparse 3.8-4.0x faster than dense-masked at 64 resident experts, equal at 16 | PASS |
| S4 HandBwd | grads == composed ops | ~1e-7; Tiny step -16% / -21% on Metal; one allocation per output (no copy), flat memory | PASS |
| S5 Tiny step | 8-12k chars/s Metal, 2-3.5k CPU; loss < 1.0 | **14.4k chars/s Metal** (23.5k with HandBwd + dense MLP), **8.8k chars/s CPU+Accelerate** (13.6k); loss 5.93 -> 1.0 at step 210 on both devices, 0.000 on a fixed batch; full-model grad parity 1.1e-6 | PASS (above estimates) |
| S6 full-size row (T=4096, d=512, 8 heads) | fits ~14 GB | one full-attention layer peaks at **9.7 GiB**, two layers **run out of GPU memory**; query tiling alone is linear in depth; **layer-level recompute with 512-query tiles peaks at 6.0 GiB for 2 to 8 layers** (+30% time) | PASS-WITH-CHANGES |
| S7 npz / bf16 | bit-exact vs numpy, both ways | 28/28 checks incl. byte-identical `.npy` entries and a 1.3M-value bf16 corpus vs ml_dtypes; 34 + 4 Rust tests | PASS |

Surprises worth reading first (the evidence is in sections 2-9):

1. **Carried state must be `.detach()`ed.** A textbook AdamW (`m = b*m + (1-b)*g`) keeps the op graph of every previous step alive through `m`/`v`
   (and the gradient tensors behind them): the Tiny model reached **~14.8 GB after 30 steps on Metal and ~13.7 GB after 35 steps on CPU** before the one-line fix. The same rule applies to
   anything stored across steps (recurrent state across chunks, KV caches, EMA, logged tensors).
2. candle's **`scatter_add` backward is wrong**: it errors when `index.shape != init.shape`, and when it does run, the gradient of `init` is masked at the scattered positions
   (finite-difference rel. err 1.0). Both devices. Use `zeros.scatter(idx, src)` with unique indices.
3. Metal elementwise kernels are fine (~210 GB/s on large tensors); what looks like "slow Metal" in naive timings is **buffer churn**: `synchronize()` sweeps candle's pool, so a synchronised
   100 MB op pays ~3 ms for a fresh `MTLBuffer`. The real weak spots are the strided copy / broadcast kernels (~37 GB/s), `index_select` (16 GB/s), `index_add`, and
   a strided `sum` over the middle dim.
4. The Tiny step is **GPU-bound on ~2500 small ops (tensors) per step**, not launch-bound, and Metal is only ~1.6x faster than CPU+Accelerate at that size (2.7x at Small).
   Fusing ops (HandBwd) is the lever: bigger batches do not help (throughput is flat in batch).
5. crates.io `candle-core 0.11.0` does not compile on rustc 1.93/aarch64 and reproduces the #3847 NaN bug (in 6 of 15 fresh processes for the bare bf16 repro, 1/16 trials in every process for a GQA block): the git pin is required.
6. Exceeding the Metal working set (17.8 GiB here) comes back as `Err(Metal error ... kIOGPUCommandBufferCallbackErrorOutOfMemory)` on the next synchronize, not as a crash.

## 1. Method

* One binary per spike in `crates/minagi-cli/src/bin/`; helpers in `crates/minagi-core/src/spike/` (`ops`: composed ops; `handbwd`: custom ops and checkpointing; `moe`: dispatch; `model`: Tiny model + AdamW;
  `dev`: devices, per-step autorelease pool, memory probes; `stats`).
* Parity = `max|a-b| / max|reference|` ("rel"); inputs are generated on the host with a seeded RNG and uploaded to both devices.
* Memory: `phys_footprint` from `task_info(TASK_VM_INFO)` (what Activity Monitor shows; includes Metal allocations *and* buffers parked in candle's pool), RSS from `task_info`/`ps`, and
  on Metal `MTLDevice.currentAllocatedSize` ("MTLBuffer allocated": what the GPU actually holds).
* Timing: medians; "queue kept full" = launch many, synchronize once; "sync per op" = synchronize after each call.

## 2. S1 - Metal basics

### 2.1 GEMM throughput (queue kept full)

**GEMM throughput (async queue kept full, one sync at the end)**

| device | dtype | shape | time/op | TFLOPS |
|---|---|---|---|---|
| cpu | F32 | [4096, 512] @ [512, 1536] | 1776.2 us | 3.63 |
| cpu | F32 | [32, 64, 512] @ [32, 512, 2048] | 6150.0 us | 0.70 |
| cpu | F32 | [32, 128, 512] @ [32, 512, 2048] | 6021.9 us | 1.43 |
| cpu | F32 | [32, 256, 512] @ [32, 512, 2048] | 9189.3 us | 1.87 |
| cpu | F32 | [512, 256] @ [256, 256] | 41.1 us | 1.63 |
| metal | F32 | [4096, 512] @ [512, 1536] | 978.9 us | 6.58 |
| metal | F32 | [32, 64, 512] @ [32, 512, 2048] | 726.0 us | 5.92 |
| metal | F32 | [32, 128, 512] @ [32, 512, 2048] | 1348.3 us | 6.37 |
| metal | F32 | [32, 256, 512] @ [32, 512, 2048] | 2623.1 us | 6.55 |
| metal | F32 | [512, 256] @ [256, 256] | 20.0 us | 3.36 |
| metal | F16 | [4096, 512] @ [512, 1536] | 915.2 us | 7.04 |
| metal | F16 | [32, 64, 512] @ [32, 512, 2048] | 644.3 us | 6.67 |
| metal | F16 | [32, 128, 512] @ [32, 512, 2048] | 1237.3 us | 6.94 |
| metal | F16 | [32, 256, 512] @ [32, 512, 2048] | 2482.8 us | 6.92 |
| metal | F16 | [512, 256] @ [256, 256] | 21.4 us | 3.14 |
| metal | BF16 | [4096, 512] @ [512, 1536] | 925.6 us | 6.96 |
| metal | BF16 | [32, 64, 512] @ [32, 512, 2048] | 654.6 us | 6.56 |
| metal | BF16 | [32, 128, 512] @ [32, 512, 2048] | 1266.5 us | 6.78 |
| metal | BF16 | [32, 256, 512] @ [32, 512, 2048] | 2482.3 us | 6.92 |
| metal | BF16 | [512, 256] @ [256, 256] | 21.0 us | 3.20 |

Metal f32 reaches ~6.5 TFLOPS; **f16 / bf16 are only ~5-7% faster** (7.0 vs 6.6 TFLOPS). Accelerate on the M4 is strong for plain 2-D GEMM (3.5 TFLOPS) but batched matmul on CPU is much worse
(0.7-1.9 TFLOPS), and **CPU+Accelerate has no f16 matmul** (`the accelerate backend does not support f16 matmul`), so half precision exists only on Metal.

### 2.2 Per-op overhead and latencies (release build)

**Per-op launch overhead (tiny [16] / [16,16] tensors)**

| device | measurement | us/op |
|---|---|---|
| cpu | add [16], independent, 1 sync per 100000 | 0.12 |
| cpu | add [16], dependent chain | 0.12 |
| cpu | matmul [16,16]@[16,16] | 0.51 |
| cpu | 1 op + synchronize round trip (median / p99) | 0.1 / 0.2 |
| cpu | empty synchronize() | 0.00 |
| cpu | Tensor::from_vec upload [512] f32 | 0.11 |
| cpu | sum_all + to_scalar readback (first / median / mean of 500) | 50.1 / 4.9 / 5.0 |
| metal | add [16], independent, 1 sync per 100000 | 1.40 |
| metal | add [16], dependent chain | 1.31 |
| metal | matmul [16,16]@[16,16] | 2.72 |
| metal | 1 op + synchronize round trip (median / p99) | 108.3 / 157.5 |
| metal | empty synchronize() | 0.01 |
| metal | Tensor::from_vec upload [512] f32 | 2.77 |
| metal | sum_all + to_scalar readback (first / median / mean of 500) | 4961.7 / 141.7 / 165.9 |

Launch overhead is ~1.3-1.4 us per tiny op with the queue kept full (3.2 us in a debug build), a synchronize round trip ~110 us, a host upload ~3 us, a scalar read-back ~140 us.
(The first read-back in a process is slow, ~5 ms, because it sweeps the pool of all earlier uploads.)

### 2.3 Kernel throughput at Tiny-model sizes (queue kept full)

**Kernel throughput at Tiny-model tensor sizes (us per op, effective GB/s counting reads+writes)**

| device | op | shape | us/op | GB/s |
|---|---|---|---|---|
| cpu | add (x+y) | [512, 256] | 6.2 | 252 |
| cpu | mul by scalar (affine) | [512, 256] | 6.7 | 156 |
| cpu | silu | [512, 256] | 24.6 | 43 |
| cpu | exp | [512, 256] | 17.8 | 59 |
| cpu | sum_keepdim(last) | [512, 256] | 5.4 | 96 |
| cpu | max_keepdim(last) | [512, 256] | 53.8 | 10 |
| cpu | contiguous() of transposed view (copy) | [512, 256] | 236.8 | 4 |
| cpu | broadcast_mul by [last] | [512, 256] | 4.9 | 214 |
| cpu | add (x+y) | [512, 2048] | 20.5 | 614 |
| cpu | mul by scalar (affine) | [512, 2048] | 51.8 | 162 |
| cpu | silu | [512, 2048] | 61.6 | 136 |
| cpu | exp | [512, 2048] | 44.5 | 189 |
| cpu | sum_keepdim(last) | [512, 2048] | 39.7 | 106 |
| cpu | max_keepdim(last) | [512, 2048] | 519.1 | 8 |
| cpu | contiguous() of transposed view (copy) | [512, 2048] | 1988.8 | 4 |
| cpu | broadcast_mul by [last] | [512, 2048] | 19.0 | 441 |
| cpu | add (x+y) | [1, 4, 512, 512] | 19.7 | 638 |
| cpu | mul by scalar (affine) | [1, 4, 512, 512] | 50.4 | 166 |
| cpu | silu | [1, 4, 512, 512] | 62.4 | 134 |
| cpu | exp | [1, 4, 512, 512] | 43.2 | 194 |
| cpu | sum_keepdim(last) | [1, 4, 512, 512] | 39.6 | 106 |
| cpu | max_keepdim(last) | [1, 4, 512, 512] | 474.6 | 9 |
| cpu | contiguous() of transposed view (copy) | [1, 4, 512, 512] | 1692.6 | 5 |
| cpu | broadcast_mul by [last] | [1, 4, 512, 512] | 16.3 | 515 |
| cpu | add (x+y) | [4096, 4096] | 1430.4 | 141 |
| cpu | mul by scalar (affine) | [4096, 4096] | 1127.0 | 119 |
| cpu | silu | [4096, 4096] | 1016.7 | 132 |
| cpu | exp | [4096, 4096] | 660.3 | 203 |
| cpu | sum_keepdim(last) | [4096, 4096] | 743.9 | 90 |
| cpu | max_keepdim(last) | [4096, 4096] | 8274.0 | 8 |
| cpu | contiguous() of transposed view (copy) | [4096, 4096] | 63749.5 | 2 |
| cpu | broadcast_mul by [last] | [4096, 4096] | 594.1 | 226 |
| cpu | matmul f32 | [512,256]@[256,256] | 41.2 | 1.63 TFLOPS |
| cpu | matmul f32 | [512,256]@[256,2048] | 227.2 | 2.36 TFLOPS |
| cpu | matmul f32 | [512,2048]@[2048,256] | 212.5 | 2.53 TFLOPS |
| metal | add (x+y) | [512, 256] | 13.5 | 116 |
| metal | mul by scalar (affine) | [512, 256] | 12.2 | 86 |
| metal | silu | [512, 256] | 12.7 | 83 |
| metal | exp | [512, 256] | 12.2 | 86 |
| metal | sum_keepdim(last) | [512, 256] | 25.3 | 21 |
| metal | max_keepdim(last) | [512, 256] | 25.2 | 21 |
| metal | contiguous() of transposed view (copy) | [512, 256] | 38.8 | 27 |
| metal | broadcast_mul by [last] | [512, 256] | 31.3 | 34 |
| metal | add (x+y) | [512, 2048] | 29.7 | 424 |
| metal | mul by scalar (affine) | [512, 2048] | 21.8 | 384 |
| metal | silu | [512, 2048] | 22.1 | 379 |
| metal | exp | [512, 2048] | 21.8 | 385 |
| metal | sum_keepdim(last) | [512, 2048] | 18.1 | 231 |
| metal | max_keepdim(last) | [512, 2048] | 18.3 | 230 |
| metal | contiguous() of transposed view (copy) | [512, 2048] | 215.9 | 39 |
| metal | broadcast_mul by [last] | [512, 2048] | 184.6 | 45 |
| metal | add (x+y) | [1, 4, 512, 512] | 29.2 | 431 |
| metal | mul by scalar (affine) | [1, 4, 512, 512] | 21.9 | 383 |
| metal | silu | [1, 4, 512, 512] | 22.0 | 381 |
| metal | exp | [1, 4, 512, 512] | 22.0 | 382 |
| metal | sum_keepdim(last) | [1, 4, 512, 512] | 49.5 | 85 |
| metal | max_keepdim(last) | [1, 4, 512, 512] | 49.3 | 85 |
| metal | contiguous() of transposed view (copy) | [1, 4, 512, 512] | 275.9 | 30 |
| metal | broadcast_mul by [last] | [1, 4, 512, 512] | 277.2 | 30 |
| metal | add (x+y) | [4096, 4096] | 935.8 | 215 |
| metal | mul by scalar (affine) | [4096, 4096] | 629.3 | 213 |
| metal | silu | [4096, 4096] | 629.5 | 213 |
| metal | exp | [4096, 4096] | 654.5 | 205 |
| metal | sum_keepdim(last) | [4096, 4096] | 295.3 | 227 |
| metal | max_keepdim(last) | [4096, 4096] | 284.1 | 236 |
| metal | contiguous() of transposed view (copy) | [4096, 4096] | 3663.7 | 37 |
| metal | broadcast_mul by [last] | [4096, 4096] | 3663.6 | 37 |
| metal | matmul f32 | [512,256]@[256,256] | 21.3 | 3.15 TFLOPS |
| metal | matmul f32 | [512,256]@[256,2048] | 93.3 | 5.75 TFLOPS |
| metal | matmul f32 | [512,2048]@[2048,256] | 124.1 | 4.33 TFLOPS |

* Large tensors: unary/binary kernels run at ~210 GB/s, reductions at ~230 GB/s; 4 MB tensors (L2/SLC resident) at 380-430 GB/s.
* Small tensors (`[512,256]`, 0.5 MB): every kernel costs 12-14 us (elementwise) to 25 us (reductions), i.e. a **fixed cost of ~12 us per dispatch** on the GPU side. A Tiny step creates ~1600-2500 tensors (ops and views), most of them dispatches.
* The slow ones: `contiguous()` of a transposed view and `broadcast_mul` run at **27-45 GB/s** (strided-index path), CPU `max_keepdim` at 8-10 GB/s and CPU strided copy at 2-5 GB/s.

**Reduction timings (op + sync round trip, median us)**

| device | shape | op | median us |
|---|---|---|---|
| cpu | [512, 256] | sum_all | 4.9 |
| cpu | [512, 256] | sum_keepdim(last) | 5.1 |
| cpu | [512, 256] | max_keepdim(last) | 47.8 |
| cpu | [512, 256] | sqr | 3.4 |
| cpu | [512, 256] | flatten+sum(0) | 5.1 |
| cpu | [512, 256] | to_scalar of existing scalar | 0.0 |
| cpu | [512, 265] | sum_all | 5.0 |
| cpu | [512, 265] | sum_keepdim(last) | 6.0 |
| cpu | [512, 265] | max_keepdim(last) | 53.5 |
| cpu | [512, 265] | sqr | 3.0 |
| cpu | [512, 265] | flatten+sum(0) | 5.3 |
| cpu | [512, 265] | to_scalar of existing scalar | 0.0 |
| cpu | [4096, 512] | sum_all | 75.6 |
| cpu | [4096, 512] | sum_keepdim(last) | 77.2 |
| cpu | [4096, 512] | max_keepdim(last) | 894.0 |
| cpu | [4096, 512] | sqr | 33.5 |
| cpu | [4096, 512] | flatten+sum(0) | 75.7 |
| cpu | [4096, 512] | to_scalar of existing scalar | 0.0 |
| cpu | [1048576] | sum_all | 37.5 |
| cpu | [1048576] | sum_keepdim(last) | 37.5 |
| cpu | [1048576] | max_keepdim(last) | 487.4 |
| cpu | [1048576] | sqr | 17.3 |
| cpu | [1048576] | flatten+sum(0) | 37.5 |
| cpu | [1048576] | to_scalar of existing scalar | 0.0 |
| metal | [512, 256] | sum_all | 260.5 |
| metal | [512, 256] | sum_keepdim(last) | 137.4 |
| metal | [512, 256] | max_keepdim(last) | 132.6 |
| metal | [512, 256] | sqr | 125.1 |
| metal | [512, 256] | flatten+sum(0) | 158.3 |
| metal | [512, 256] | to_scalar of existing scalar | 114.7 |
| metal | [512, 265] | sum_all | 152.5 |
| metal | [512, 265] | sum_keepdim(last) | 138.2 |
| metal | [512, 265] | max_keepdim(last) | 138.5 |
| metal | [512, 265] | sqr | 132.1 |
| metal | [512, 265] | flatten+sum(0) | 152.2 |
| metal | [512, 265] | to_scalar of existing scalar | 106.1 |
| metal | [4096, 512] | sum_all | 510.2 |
| metal | [4096, 512] | sum_keepdim(last) | 214.2 |
| metal | [4096, 512] | max_keepdim(last) | 213.3 |
| metal | [4096, 512] | sqr | 271.8 |
| metal | [4096, 512] | flatten+sum(0) | 510.4 |
| metal | [4096, 512] | to_scalar of existing scalar | 115.3 |
| metal | [1048576] | sum_all | 318.2 |
| metal | [1048576] | sum_keepdim(last) | 319.0 |
| metal | [1048576] | max_keepdim(last) | 319.3 |
| metal | [1048576] | sqr | 198.2 |
| metal | [1048576] | flatten+sum(0) | 318.5 |
| metal | [1048576] | to_scalar of existing scalar | 112.0 |

### 2.4 Memory behaviour over 200,000 iterations (release build)

`mm` = one `[512,256]@[256,256]` per iteration; `step` = host upload of ids + `index_select` + matmul + `sqr().sum_all()` (4 kernels, 1 upload); one process per configuration.

| scenario | autoreleasepool | sync per iter | iterations | RSS MiB (start -> end) | phys_footprint MiB (delta) | footprint KiB per iteration |
|---|---|---|---|---|---|---|
| mm | no | no | 200000 | 58 -> 72 | 136 -> 150 (+14) | +0.07 |
| mm | no | yes | 200000 | 28 -> 69 | 97 -> 138 (+41) | +0.21 |
| mm | yes | no | 200000 | 57 -> 57 | 135 -> 135 (+0) | +0.00 |
| mm | yes | yes | 200000 | 23 -> 23 | 91 -> 92 (+1) | +0.01 |
| step | no | no | 200000 | 82 -> 67 | 161 -> 179 (+18) | +0.09 |
| step | no | yes | 200000 | 35 -> 76 | 101 -> 142 (+41) | +0.21 |
| step | yes | no | 200000 | 80 -> 81 | 159 -> 160 (+1) | +0.01 |
| step | yes | yes | 200000 | 30 -> 30 | 96 -> 96 (+0) | +0.00 |

Same experiment in a **debug build** (dev profile; measured earlier in a quiet window):

| scenario | autoreleasepool | sync per iter | iterations | RSS MiB (start -> end) | phys_footprint MiB (delta) | footprint KiB per iteration |
|---|---|---|---|---|---|---|
| mm | no | no | 200000 | 60 -> 91 | 138 -> 169 (+31) | +0.16 |
| mm | no | yes | 200000 | 61 -> 61 | 130 -> 479 (+349) | +1.79 |
| mm | yes | no | 200000 | 57 -> 57 | 135 -> 135 (+0) | +0.00 |
| mm | yes | yes | 200000 | 23 -> 17 | 91 -> 91 (+0) | +0.00 |
| step | no | no | 200000 | 84 -> 135 | 163 -> 214 (+51) | +0.26 |
| step | no | yes | 200000 | 69 -> 193 | 135 -> 483 (+348) | +1.78 |
| step | yes | no | 200000 | 81 -> 81 | 159 -> 160 (+1) | +0.01 |
| step | yes | yes | 200000 | 30 -> 21 | 96 -> 96 (+0) | +0.00 |

Reading: without an `autoreleasepool` the process leaks autoreleased Objective-C objects (command buffers, encoders, completion handlers) at **~0.2 KiB per iteration in release builds and ~1.8 KiB in debug
builds**; RSS under-reports it (flat for `mm`+sync in debug builds), `phys_footprint` shows it. With one pool per iteration both are flat, with or without `synchronize()`. A real training step is ~2000-2500
dispatches, i.e. ~0.2 MiB per step (section 6.7). `synchronize()` per iteration costs ~100-150 us of latency, negligible per training step. **The pool is mandatory for long runs, in every build.**

## 3. S2 - gradient parity, R2, #3847

### 3.1 Op zoo, Metal vs CPU (f32)

| op | fwd rel | worst grad rel | grad max abs err | grads non-zero | pass (tol) | note |
|---|---|---|---|---|---|---|
| embedding (index_select dim0, dup ids) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-5) |  |
| narrow dim1 + dim0 (strided views) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| cat dim1 and dim0 | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| rope_i_slow (candle, rank-5 internals) T=128 Dh=64 | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-5) |  |
| rope_i_composed (ours, rank<=4) T=128 Dh=64 | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-5) |  |
| rms_norm_slow (grad x and w) | 1.4e-7 | 2.2e-7 | 5.7e-6 | yes | PASS (1e-5) |  |
| causal attention composed B=2 H=4 T=128 Dh=64 | 7.7e-8 | 2.7e-7 | 7.2e-7 | yes | PASS (1e-5) |  |
| causal attention tiled (q_block 32) T=128 | 7.7e-8 | 2.2e-7 | 7.2e-7 | yes | PASS (1e-5) |  |
| where_cond (both branches) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| gather dim1 (idx [64,2] from [64,16]) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| scatter_add, idx shape == init shape (row permutations) | 0.0e0 | 0.0e0 | 0.0e0 | no | PASS (1e-6) | works only when idx.shape == init.shape |
| scatter_add, idx [64,2] into init [64,16] (MoE gains pattern) | - | - | - | - | XFAIL (known candle bug) | candle bug: ScatterAdd bwd builds mask via scatter(idx, zeros_like(init)); shape mismatch cpu: shape mismatch in scatter (indexes, src), lhs: [64, 2], rhs: [64, 16] / metal: shape mismatch in scatter (indexes, src), lhs: [64, 2], rhs: [64, 16] |
| scatter (overwrite) onto zeros, unique idx [64,2] -> [64,16] (workaround) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) | workaround for scatter_add gains pattern |
| index_select(u32::MAX sentinel) + index_add(sentinel) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) | sentinel rows zero-filled / skipped |
| bmm a @ w.transpose(1,2) [4,32,64]x[4,48,64] | 0.0e0 | 2.6e-7 | 8.6e-6 | yes | PASS (1e-5) |  |
| 2-D x @ w.t() [128,64]x[48,64] (Linear) | 0.0e0 | 3.3e-7 | 9.5e-6 | yes | PASS (1e-5) |  |
| matmul, lhs broadcast_as [1,32,64]->[4,32,64] (stride 0) | 0.0e0 | 8.2e-8 | 3.8e-6 | yes | PASS (1e-5) | R5 |
| matmul, rhs broadcast_as [1,64,48]->[4,64,48] (stride 0) | 0.0e0 | 9.1e-8 | 3.8e-6 | yes | PASS (1e-5) | R5 |
| GQA: q [1,4,64,32] @ expand(k [1,2,64,32] 2->4 heads)^T, rank 5 repeat | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-5) | k.unsqueeze(2).expand -> reshape (copy) path |
| cumsum dim 1 / last dim (halting-style cumulative sums) | 0.0e0 | 1.0e-7 | 1.1e-6 | yes | PASS (1e-5) |  |
| slice_scatter / narrow-concat (state update) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| silu | 8.3e-8 | 1.0e-7 | 3.6e-7 | yes | PASS (1e-5) |  |
| sigmoid | 6.3e-8 | 1.4e-7 | 1.2e-7 | yes | PASS (1e-5) |  |
| tanh | 1.2e-7 | 1.5e-7 | 4.8e-7 | yes | PASS (1e-5) |  |
| gelu (tanh approx) | 7.9e-8 | 1.4e-7 | 4.8e-7 | yes | PASS (1e-5) |  |
| gelu_erf | 1.6e-7 | 1.4e-7 | 4.8e-7 | yes | PASS (1e-5) |  |
| relu | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| exp | 9.3e-8 | 1.2e-7 | 3.8e-6 | yes | PASS (1e-5) |  |
| log(|x|+1) | 1.5e-7 | 8.3e-8 | 2.4e-7 | yes | PASS (1e-5) |  |
| sqrt(x^2+1) | 1.1e-7 | 1.1e-7 | 3.6e-7 | yes | PASS (1e-5) |  |
| powf(|x|+0.5, 1.7) | 1.4e-7 | 1.7e-7 | 1.9e-6 | yes | PASS (1e-5) |  |
| clamp(-0.5, 0.5) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| affine(2x+1) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| neg, recip(|x|+1) | 6.0e-8 | 4.7e-8 | 1.2e-7 | yes | PASS (1e-5) |  |
| sum_keepdim / mean_keepdim / sum(dim0) on rank 3 | 1.9e-7 | 0.0e0 | 0.0e0 | yes | PASS (1e-5) |  |
| max_keepdim / max(dim) / min_keepdim on rank 3 | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| softmax composed (detached max) vs candle softmax | 1.1e-7 | 2.5e-7 | 2.1e-7 | yes | PASS (1e-5) |  |
| to_dtype f32->f16->f32 (cast only) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| to_dtype f32->bf16->f32 (cast only) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (1e-6) |  |
| f16 matmul + cast back (CPU emulated in f32 w/ f16 rounding) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (5e-3) | CPU+Accelerate has no f16 matmul; reference = f16-rounded inputs/outputs, f32 accumulate |
| bf16 matmul + cast back (mixed precision) | 0.0e0 | 0.0e0 | 0.0e0 | yes | PASS (3e-2) | bf16 accumulate order differs (informational) |
| topk via arg_sort_last_dim + gather + softmax | 6.5e-8 | 1.1e-7 | 1.2e-7 | yes | PASS (1e-6) |  |

top-2 indices: cpu==metal for all 256 rows: true; metal == host-sorted reference: true

zoo failures: 0

41 cases pass with 30-3000x margin (worst gradient rel. err **3.3e-7**), 1 expected failure (candle bug, below). Highlights:

* **`u32::MAX` sentinel ids** behave as designed on both backends: `index_select` yields a zero row, `index_add` skips the id (and the gradient of a sentinel row is zero).
* `rope_i_slow` (rank-5 internals), my rank <= 4 `rope_i_composed`, composed causal attention (`T=128`, 4 heads, `Dh=64`, plain and query-tiled), `where_cond` (both branches), `gather`, `cumsum`
  (matmul with `triu`), `slice_scatter`, bf16/f16 round trips and `arg_sort_last_dim` top-k (indices identical on both devices and equal to a host sort) all match.
  **Stride-0 (`broadcast_as`) matmul operands and GQA-style `unsqueeze().expand()` are correct on both devices at this commit** (R5 holds at the pin); I still keep every graph tensor at rank <= 4.
* **candle bug (both devices): `scatter_add` backward.** (a) It builds its mask with `scatter(idx, zeros_like(init))`, which fails with a shape mismatch unless `index.shape == init.shape`
  (the MoE "gains" pattern `zeros[N,S].scatter_add(idx[N,k], w[N,k])` hits this). (b) When it does run, `init`'s gradient is multiplied by that mask, i.e. zeroed at every scattered position (see 3.3).
  **Tested workaround:** `Tensor::zeros(..).scatter(&idx, &src, dim)` (valid because top-k slots are unique per row), row "scatter (overwrite) onto zeros".
* CPU+Accelerate cannot do f16 matmul; the f16 row emulates it on CPU with f16-rounded inputs/outputs and f32 accumulation.

### 3.2 R2: fused candle-nn ops have no gradient

| device | op | grad to x | grad to w/params | verdict |
|---|---|---|---|---|
| cpu | candle_nn::ops::rms_norm (fused) | ABSENT | ABSENT | NO GRADIENT (silent) |
| cpu | ops::rms_norm_slow | present, |g|_1 = 4.335e4 | present, |g|_1 = 3.267e4 | ok |
| cpu | rotary_emb::rope_i (fused) | ABSENT | n/a | NO GRADIENT (silent) |
| cpu | rotary_emb::rope_i_slow | present, |g|_1 = 2.618e4 | n/a | ok |
| cpu | ops::softmax_last_dim (fused) | ABSENT | n/a | NO GRADIENT (silent) |
| cpu | ops::softmax (composed) | present, |g|_1 = 7.675e1 | n/a | ok |
| metal | candle_nn::ops::rms_norm (fused) | ABSENT | ABSENT | NO GRADIENT (silent) |
| metal | ops::rms_norm_slow | present, |g|_1 = 4.335e4 | present, |g|_1 = 3.267e4 | ok |
| metal | rotary_emb::rope_i (fused) | ABSENT | n/a | NO GRADIENT (silent) |
| metal | rotary_emb::rope_i_slow | present, |g|_1 = 2.618e4 | n/a | ok |
| metal | ops::softmax_last_dim (fused) | ABSENT | n/a | NO GRADIENT (silent) |
| metal | ops::softmax (composed) | present, |g|_1 = 7.675e1 | n/a | ok |

Confirmed on both devices: `ops::rms_norm`, `rotary_emb::rope_i` and `ops::softmax_last_dim` are registered with `apply_opN_no_bwd`; `loss.backward()` succeeds and `grads.get(x)` is **`None`** (absent, not zero), while a Var placed *after*
the op still receives a gradient, so a model built on them trains silently wrong. The slow/composed variants give non-zero gradients, and 3.3 shows they are *correct*.

### 3.3 Composed ops vs finite differences (CPU f64)

**S2c. Finite-difference check of composed ops (CPU, f64, eps=1e-6; rel = max|autograd-fd|/max|fd|)**

| op | rel err | pass (<1e-6) |
|---|---|---|
| rms_norm_slow (x,w) | 1.6e-9 | PASS |
| rope_i_composed | 1.3e-9 | PASS |
| attention composed (q,k,v) | 2.3e-9 | PASS |
| cross_entropy | 6.3e-9 | PASS |
| matmul with broadcast_as operands (lhs [1,6,5]->[3,6,5], rhs [1,5,4]->[3,5,4]) | 5.9e-10 | PASS |
| scatter_add, idx.shape == init.shape (row permutations) [candle bug: init grad is masked] | 1.0e0 | FAIL (expected: candle bug) |
| scatter (overwrite) onto zeros, unique idx (workaround) | 1.9e-10 | PASS |
| swiglu | 3.2e-10 | PASS |

The composed ops (`rms_norm_slow`, `rope_i_composed`, attention, cross-entropy, SwiGLU, stride-0 matmul) match central finite differences to ~1e-9. The `scatter_add` row fails with rel. err 1.0, confirming candle's masked
`init` gradient (it is identical on Metal and CPU, so parity tests alone would not catch it); the `scatter` workaround is exact.

### 3.4 Issue #3847 (Metal NaN gradients from the backward of a rank-5 `expand`)

On the pinned commit (the suite run; six more fresh processes are tabulated below):

| variant | trials with non-finite grad | iterations run |
|---|---|---|
| bare expand, bf16 (issue repro) | 0/16 | 1280 |
| bare expand, f32 | 0/16 | 1280 |
| GQA block (expand 1->2 kv heads), f32 | 0/16 | 1280 |
| GQA block, bf16 | 0/16 | 1280 |
| 200x identical rank-5 expand backward, f32 (fixed input) | 0 non-finite values total | 200 |

expand-backward gradient value check vs closed form: max abs err 0.00e0, rel 0.00e0

The root cause (PR #3873) is a rank-5+ strided reduce reading out of bounds in `reduce.metal`; the pin contains the fix. To make sure the repro detects the bug at all, the same test
was built against **crates.io `candle-core 0.11.0`** in a scratch crate (needs `RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zcrate-attr=feature(stdarch_neon_f16)"` because 0.11.0 does not compile on rustc 1.93 for aarch64; the crate is `crates/minagi-cli/scripts/candle0110_repro/`); fresh processes, each row is one process (non-finite trials out of 16):

| fresh process | bare expand bf16 | bare expand f32 | GQA block f32 | GQA block bf16 |
|---|---|---|---|---|
| 1 | 14/16 | 0/16 | 0/16 | 1/16 |
| 2 | 0/16 | 0/16 | 0/16 | 1/16 |
| 3 | 0/16 | 0/16 | 0/16 | 1/16 |
| 4 | 0/16 | 0/16 | 0/16 | 1/16 |
| 5 | 0/16 | 0/16 | 0/16 | 1/16 |
| 6 | 0/16 | 0/16 | 0/16 | 1/16 |
| 7 | 16/16 | 0/16 | 0/16 | 1/16 |
| 8 | 0/16 | 0/16 | 0/16 | 1/16 |

Because the bug is an out-of-bounds *read*, whether it shows depends on what lies next to the buffer. Over 15 fresh processes (the table is the last 8) the bare bf16 repro on **0.11.0 produced non-finite gradients in 6 processes (14-16 of 16 trials in each) and none in the other 9**;
the interior-expand GQA block in bf16 produced 1/16 in every process (14 of 14 that ran it). The pin produced none in any of 7 processes (6 repeats below plus the suite run). Keeping all graph tensors at rank <= 4 avoids the code path regardless.

The pinned commit, six fresh processes, all variants:

| fresh process | bare expand bf16 | bare expand f32 | GQA block f32 | GQA block bf16 | 200x identical backward (f32) |
|---|---|---|---|---|---|
| 1 | 0/16 | 0/16 | 0/16 | 0/16 | 0 non-finite values total |
| 2 | 0/16 | 0/16 | 0/16 | 0/16 | 0 non-finite values total |
| 3 | 0/16 | 0/16 | 0/16 | 0/16 | 0 non-finite values total |
| 4 | 0/16 | 0/16 | 0/16 | 0/16 | 0 non-finite values total |
| 5 | 0/16 | 0/16 | 0/16 | 0/16 | 0 non-finite values total |
| 6 | 0/16 | 0/16 | 0/16 | 0/16 | 0 non-finite values total |

## 4. S3 - MoE forward + backward

Code: `crates/minagi-core/src/spike/moe.rs`. **Dense-masked** = one wide SwiGLU over all resident experts scaled by per-(token, expert) gains. **Sparse** = host-side stable bucketing with a per-expert capacity,
a padded `[S*cap, D]` buffer via `index_select` with `u32::MAX` sentinels, three `bmm`s over `[S, cap, *]`, an inverse-map `index_select` back to `[N, k, D]`, gate-weighted sum. "Fast bwd" gives both gathers a hand-written
backward that is itself an `index_select` through the inverse map (the maps are permutations with sentinels) instead of `index_add`.

### 4.1 Parity (N=512, D=256, S=8 resident of 16, F=256, top_k=2)

rel = max abs diff / max |reference|. Pass threshold 1e-4.

| scenario | comparison | fwd rel | grad rel (per tensor) | dropped assignments | pass |
|---|---|---|---|---|---|
| cap = N (no drops) | cpu: sparse(stock) vs dense | 9.9e-7 | x 7.5e-7, router 5.6e-7, w1 5.4e-7, w3 6.0e-7, w2 2.0e-7 | 0 | PASS |
| cap = N (no drops) | cpu: sparse(fast bwd) vs dense | 9.9e-7 | x 7.5e-7, router 5.6e-7, w1 5.4e-7, w3 6.0e-7, w2 2.0e-7 | 0 | PASS |
| cap = N (no drops) | cpu: sparse(fast bwd) vs sparse(stock) | 0.0e0 | x 7.1e-8, router 0.0e0, w1 0.0e0, w3 0.0e0, w2 0.0e0 | 0 | PASS |
| cap = N (no drops) | metal: sparse(stock) vs dense | 1.0e-6 | x 7.1e-7, router 4.7e-7, w1 5.4e-7, w3 5.4e-7, w2 1.6e-7 | 0 | PASS |
| cap = N (no drops) | metal: sparse(fast bwd) vs dense | 1.0e-6 | x 7.1e-7, router 4.7e-7, w1 5.4e-7, w3 5.4e-7, w2 1.6e-7 | 0 | PASS |
| cap = N (no drops) | metal: sparse(fast bwd) vs sparse(stock) | 0.0e0 | x 7.1e-8, router 0.0e0, w1 0.0e0, w3 0.0e0, w2 0.0e0 | 0 | PASS |
| cap = N (no drops) | metal vs cpu: Dense | 3.3e-7 | x 2.1e-7, router 2.9e-7, w1 1.6e-7, w3 1.6e-7, w2 2.0e-7 | 0 | PASS |
| cap = N (no drops) | metal vs cpu: SparseStock | 2.2e-7 | x 2.5e-7, router 2.6e-7, w1 2.3e-7, w3 3.2e-7, w2 2.0e-7 | 0 | PASS |
| cap = N (no drops) | metal vs cpu: SparseFast | 2.2e-7 | x 2.5e-7, router 2.6e-7, w1 2.3e-7, w3 3.2e-7, w2 2.0e-7 | 0 | PASS |
| cap = N (no drops) | routing indices identical cpu vs metal | - | - | - | PASS |
| cap = 0.8*N*k/S (forced drops) | cpu: sparse(stock) vs dense | 1.0e-6 | x 7.5e-7, router 5.2e-7, w1 5.1e-7, w3 6.0e-7, w2 1.9e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | cpu: sparse(fast bwd) vs dense | 1.0e-6 | x 7.5e-7, router 5.2e-7, w1 5.1e-7, w3 6.0e-7, w2 1.9e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | cpu: sparse(fast bwd) vs sparse(stock) | 0.0e0 | x 7.1e-8, router 0.0e0, w1 0.0e0, w3 0.0e0, w2 0.0e0 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | metal: sparse(stock) vs dense | 1.1e-6 | x 7.1e-7, router 6.1e-7, w1 5.1e-7, w3 6.0e-7, w2 4.8e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | metal: sparse(fast bwd) vs dense | 1.1e-6 | x 7.1e-7, router 6.1e-7, w1 5.1e-7, w3 6.0e-7, w2 4.8e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | metal: sparse(fast bwd) vs sparse(stock) | 0.0e0 | x 7.1e-8, router 0.0e0, w1 0.0e0, w3 0.0e0, w2 0.0e0 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | metal vs cpu: Dense | 3.4e-7 | x 2.1e-7, router 3.0e-7, w1 1.6e-7, w3 1.7e-7, w2 1.9e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | metal vs cpu: SparseStock | 2.2e-7 | x 2.5e-7, router 3.0e-7, w1 3.1e-7, w3 4.3e-7, w2 4.3e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | metal vs cpu: SparseFast | 2.2e-7 | x 2.2e-7, router 3.0e-7, w1 3.1e-7, w3 4.3e-7, w2 4.3e-7 | 200 | PASS |
| cap = 0.8*N*k/S (forced drops) | routing indices identical cpu vs metal | - | - | - | PASS |

Dense == sparse(stock) == sparse(fast bwd) and Metal == CPU for the forward output and **every** gradient (x, router, w1, w3, w2), with and without dropped assignments (200 of 1024 dropped in the forced-drop scenario;
the dense reference needs the same `keep` mask). Routing indices are identical on both devices.

### 4.2 Timings (N=4096, k=8, D=512, F=512; Tiny-sized block first)

Note: the block-level rows reuse a pre-built host plan (planning itself = one device->host copy of idx + a host loop, measured separately below).

**cpu: N=512 k=2 D=256 F=256 S=8 cap=192 (padded rows 1536 = 1.50x assignments)**

dropped 0 of 1024 assignments (0.0%), max expert load 154 (cap 192), fill 67%

Each cell is `sync-per-op / queue-full` ms (see the note on buffer churn).

| primitive | forward ms | backward ms |
|---|---|---|
| gather: index_select [512,256] by 1536 ids | 0.05 / 0.05 | stock index_add 0.05 / 0.05 ; fast (index_select via inverse map + sum) 0.38 / 0.35 |
| inverse-map: index_select [1536,256] by 1024 ids | 0.02 / 0.02 | stock index_add 0.07 / 0.07 ; fast (index_select via assign map) 0.04 / 0.04 |
| bmm up/gate [8,192,256]x[8,256,256] (x2 in block) | 0.13 / 0.13 (1.6 TFLOPS queue-full) | 0.26 / 0.25 (dA + dW) |
| bmm down [8,192,256]x[8,256,256] | 0.13 / 0.12 (1.6 TFLOPS queue-full) | 0.25 / 0.25 (dA + dW) |
| elementwise silu on [8,192,256] (2 MB) | 0.03 / 0.03 (120 GB/s queue-full) | |
| elementwise mul (2 inputs), same size | 0.01 / 0.01 (814 GB/s queue-full) | |
| combine: broadcast_mul gates + sum(1) on [512,2,256] | 0.32 / 0.31 | |

| full block | forward ms | forward+backward ms |
|---|---|---|
| sparse, stock autograd | 1.1 | 3.7 |
| sparse, permutation-aware bwd | 1.2 | 3.7 |
| dense-masked | 1.5 | 6.2 |

**cpu: N=4096 k=8 D=512 F=512 S=16 cap=3072 (padded rows 49152 = 1.50x assignments)**

dropped 0 of 32768 assignments (0.0%), max expert load 2081 (cap 3072), fill 67%

Each cell is `sync-per-op / queue-full` ms (see the note on buffer churn).

| primitive | forward ms | backward ms |
|---|---|---|
| gather: index_select [4096,512] by 49152 ids | 1.94 / 1.92 | stock index_add 1.70 / 1.68 ; fast (index_select via inverse map + sum) 18.93 / 18.46 |
| inverse-map: index_select [49152,512] by 32768 ids | 1.15 / 1.25 | stock index_add 3.80 / 4.09 ; fast (index_select via assign map) 8.19 / 8.24 |
| bmm up/gate [16,3072,512]x[16,512,512] (x2 in block) | 8.99 / 9.02 (2.9 TFLOPS queue-full) | 19.18 / 19.15 (dA + dW) |
| bmm down [16,3072,512]x[16,512,512] | 8.99 / 8.98 (2.9 TFLOPS queue-full) | 19.05 / 19.08 (dA + dW) |
| elementwise silu on [16,3072,512] (101 MB) | 1.54 / 1.53 (131 GB/s queue-full) | |
| elementwise mul (2 inputs), same size | 1.37 / 1.37 (220 GB/s queue-full) | |
| combine: broadcast_mul gates + sum(1) on [4096,8,512] | 17.85 / 17.82 | |

| full block | forward ms | forward+backward ms |
|---|---|---|
| sparse, stock autograd | 53.0 | 157.2 |
| sparse, permutation-aware bwd | 55.5 | 182.7 |
| dense-masked | 40.5 | 169.2 |

**metal: N=512 k=2 D=256 F=256 S=8 cap=192 (padded rows 1536 = 1.50x assignments)**

dropped 0 of 1024 assignments (0.0%), max expert load 154 (cap 192), fill 67%

Each cell is `sync-per-op / queue-full` ms (see the note on buffer churn).

| primitive | forward ms | backward ms |
|---|---|---|
| gather: index_select [512,256] by 1536 ids | 0.55 / 0.36 | stock index_add 1.57 / 0.45 ; fast (index_select via inverse map + sum) 0.93 / 0.66 |
| inverse-map: index_select [1536,256] by 1024 ids | 0.24 / 0.07 | stock index_add 0.60 / 0.37 ; fast (index_select via assign map) 0.26 / 0.09 |
| bmm up/gate [8,192,256]x[8,256,256] (x2 in block) | 0.21 / 0.05 (3.7 TFLOPS queue-full) | 0.26 / 0.09 (dA + dW) |
| bmm down [8,192,256]x[8,256,256] | 0.22 / 0.05 (3.9 TFLOPS queue-full) | 0.25 / 0.09 (dA + dW) |
| elementwise silu on [8,192,256] (2 MB) | 0.17 / 0.01 (214 GB/s queue-full) | |
| elementwise mul (2 inputs), same size | 0.17 / 0.02 (248 GB/s queue-full) | |
| combine: broadcast_mul gates + sum(1) on [512,2,256] | 0.69 / 0.51 | |

| full block | forward ms | forward+backward ms |
|---|---|---|
| sparse, stock autograd | 1.3 | 3.5 |
| sparse, permutation-aware bwd | 1.3 | 3.5 |
| dense-masked | 1.8 | 5.9 |

**metal: N=4096 k=8 D=512 F=512 S=16 cap=3072 (padded rows 49152 = 1.50x assignments)**

dropped 0 of 32768 assignments (0.0%), max expert load 2081 (cap 3072), fill 67%

Each cell is `sync-per-op / queue-full` ms (see the note on buffer churn).

| primitive | forward ms | backward ms |
|---|---|---|
| gather: index_select [4096,512] by 49152 ids | 8.78 / 6.06 | stock index_add 13.96 / 13.53 ; fast (index_select via inverse map + sum) 14.88 / 13.68 |
| inverse-map: index_select [49152,512] by 32768 ids | 5.58 / 4.50 | stock index_add 20.31 / 15.14 ; fast (index_select via assign map) 8.71 / 6.07 |
| bmm up/gate [16,3072,512]x[16,512,512] (x2 in block) | 6.76 / 4.01 (6.4 TFLOPS queue-full) | 10.69 / 7.99 (dA + dW) |
| bmm down [16,3072,512]x[16,512,512] | 6.81 / 4.02 (6.4 TFLOPS queue-full) | 10.70 / 7.92 (dA + dW) |
| elementwise silu on [16,3072,512] (101 MB) | 3.96 / 0.96 (209 GB/s queue-full) | |
| elementwise mul (2 inputs), same size | 4.32 / 1.42 (212 GB/s queue-full) | |
| combine: broadcast_mul gates + sum(1) on [4096,8,512] | 18.17 / 16.94 | |

| full block | forward ms | forward+backward ms |
|---|---|---|
| sparse, stock autograd | 59.4 | 180.0 |
| sparse, permutation-aware bwd | 60.7 | 174.0 |
| dense-masked | 52.1 | 168.9 |

**metal: N=4096 k=8 D=512 F=512 S=64 cap=768 (padded rows 49152 = 1.50x assignments)**

dropped 0 of 32768 assignments (0.0%), max expert load 620 (cap 768), fill 67%

Each cell is `sync-per-op / queue-full` ms (see the note on buffer churn).

| primitive | forward ms | backward ms |
|---|---|---|
| gather: index_select [4096,512] by 49152 ids | 8.95 / 6.10 | stock index_add 13.96 / 13.54 ; fast (index_select via inverse map + sum) 14.89 / 13.67 |
| inverse-map: index_select [49152,512] by 32768 ids | 5.63 / 4.50 | stock index_add 20.59 / 15.19 ; fast (index_select via assign map) 8.78 / 6.08 |
| bmm up/gate [64,768,512]x[64,512,512] (x2 in block) | 6.88 / 4.03 (6.4 TFLOPS queue-full) | 10.74 / 7.91 (dA + dW) |
| bmm down [64,768,512]x[64,512,512] | 6.91 / 4.04 (6.4 TFLOPS queue-full) | 10.77 / 7.92 (dA + dW) |
| elementwise silu on [64,768,512] (101 MB) | 3.99 / 0.97 (207 GB/s queue-full) | |
| elementwise mul (2 inputs), same size | 4.39 / 1.39 (216 GB/s queue-full) | |
| combine: broadcast_mul gates + sum(1) on [4096,8,512] | 18.19 / 16.94 | |

| full block | forward ms | forward+backward ms |
|---|---|---|
| sparse, stock autograd | 59.7 | 181.6 |
| sparse, permutation-aware bwd | 60.5 | 172.5 |
| dense-masked | 224.5 | 691.8 |

routing + readback + host plan + upload (N=4096,k=8,S=64): 1.38 ms median

* **Measurement artifact worth knowing:** cells are `sync-per-op / queue-full`. Synchronising after a 100 MB-output op costs ~3 ms extra because `synchronize()` sweeps candle's buffer pool and the next op has to create a fresh 128 MiB
  `MTLBuffer` (silu: 3.96 ms vs 0.97 ms; 207-216 GB/s queue-full). Training pays this once per step, not per op (section 6.6: <= 5%).
* bmm runs at **6.4 TFLOPS** (queue-full) for `[16,3072,512]x[16,512,512]` and the same for 64 experts.
* **Gathers are the weak kernels:** `index_select` of 100 MB takes 6.1 ms (16 GB/s effective, one thread per output element with 64-bit index math), the inverse-map select 4.5 ms; stock `index_add` backward 13.5 ms / 15.2 ms.
  The permutation-aware backward turns the second one into a 6.1 ms `index_select` (2.5x) but is no better for the first gather; at block level it is a 0-5% effect. `broadcast_mul + sum(1)` over `[4096,8,512]` costs 17 ms (strided reduce).
* **Dense-masked vs sparse:** at 64 resident experts sparse is **3.8x (forward) / 3.8-4.0x (forward+backward) faster**; at 16 it is a tie (dense slightly ahead); at Tiny size (S=8, k=2, N=512) the sparse *block* is faster on Metal
  (1.3 / 3.5 ms vs 1.8 / 5.9 ms) but needs a device->host read-back of the routing (+0.9-1.4 ms and a pipeline drain per MoE layer), so dense-masked stays the right Tiny/Small choice.
* Host routing + read-back + plan + upload: 1.4 ms at N=4096, S=64.
* If sparse dispatch is needed at full size, a custom Metal row-gather / row-scatter-add kernel is the obvious optimisation (not attempted).

## 5. S4 - HandBwd custom ops

`HandBwd` = `CustomOp1/2/3` whose forward hands over an output **precomputed from detached inputs with the fused kernel** (`rms_norm`, `rope_i`) and whose backward is ordinary tensor math
(RoPE: the same rotation with `-sin`; RMSNorm: `dx = r*g*w - x*r^3*mean(x*g*w)`, `dw = sum(g*x*r)`). On Metal the output buffer is shared (the `MetalStorage` clone is an `Arc<Buffer>` clone), on CPU it is one memcpy.
Code: `spike/handbwd.rs`.

### 5.1 Gradient parity

rel = max|a-b| / max|b|; reference b is the composed op on the same device unless noted.

| device | op | fwd rel | dx rel | dw rel | pass (<1e-4) |
|---|---|---|---|---|---|
| cpu | rms_norm_hand [512, 256] | 3.2e-7 | 9.6e-8 | 2.6e-7 | PASS |
| cpu | rms_norm_hand [2, 256, 256] | 3.2e-7 | 9.6e-8 | 2.6e-7 | PASS |
| cpu | rope_i_hand [2,4,512,64] | 0.0e0 | 0.0e0 | n/a | PASS |
| metal | rms_norm_hand [512, 256] | 1.6e-7 | 4.8e-8 | 0.0e0 | PASS |
| metal | rms_norm_hand [2, 256, 256] | 1.6e-7 | 4.8e-8 | 0.0e0 | PASS |
| metal | rope_i_hand [2,4,512,64] | 1.1e-7 | 9.9e-8 | n/a | PASS |
| metal vs cpu | rms_hand grads | - | 1.4e-7 | 5.7e-7 | PASS |
| metal vs cpu | rope_hand grads | - | 9.9e-8 | n/a | PASS |

| device | worst grad rel (hand vs composed) over emb/w/proj | pass |
|---|---|---|
| cpu | 3.4e-7 | PASS |
| metal | 3.0e-7 | PASS |

### 5.2 fwd+bwd latency (synchronised, median of 100; second of two runs)

| device | op | shape | composed (autograd) | HandBwd | speedup |
|---|---|---|---|---|---|
| cpu | rms_norm | [512, 256] | 551 us | 581 us | 0.95x |
| cpu | rms_norm | [2048, 256] | 2182 us | 1290 us | 1.69x |
| cpu | rms_norm | [4096, 512] | 13771 us | 6791 us | 2.03x |
| cpu | rope_i | [1,4,512,64] | 1704 us | 520 us | 3.28x |
| cpu | rope_i | [1,4,1024,64] | 3160 us | 741 us | 4.26x |
| metal | rms_norm | [512, 256] | 1126 us | 989 us | 1.14x |
| metal | rms_norm | [2048, 256] | 2137 us | 1689 us | 1.27x |
| metal | rms_norm | [4096, 512] | 8221 us | 6062 us | 1.36x |
| metal | rope_i | [1,4,512,64] | 1344 us | 486 us | 2.77x |
| metal | rope_i | [1,4,1024,64] | 1957 us | 629 us | 3.11x |

### 5.3 No extra copy, no leak on Metal

| step | MTLBuffer allocated MiB | delta MiB |
|---|---|---|
| baseline (x 128 MiB + w) | 129 | |
| + fused rms_norm output | 257 | +128 |
| (output dropped, Var copy of x created) | 257 | |
| + rms_norm_hand output (graph attached) | 385 | +128 |

Expected one 128 MiB allocation for each output; a second copy would show as ~+256.

| iter | phys_footprint MiB | rss MiB | elapsed s |
|---|---|---|---|
| 2000 | 382 | 391 | 4.3 |
| 4000 | 262 | 271 | 8.6 |
| 6000 | 262 | 271 | 12.9 |
| 8000 | 262 | 271 | 17.9 |
| 10000 | 118 | 143 | 22.2 |

footprint growth after first sample: -264.2 MiB over 8000 iterations

(`phys_footprint` is flat or falling over 10,000 fwd+bwd iterations with pool + sync; the first sample includes warm-up allocations.)

* HandBwd matches the composed ops to ~1e-7 on both devices, also in a graph where the custom node has two consumers (gradient accumulation through it) and inside the full model (section 6.10).
* End to end in the Tiny model HandBwd saves **16% (MoE) / 21% (dense MLP) of the step on Metal and 10% / 12% on CPU**; the Small model's peak footprint drops from 7.1 to 5.7 GiB (7241 -> 5854 MiB) because the composed norm/RoPE intermediates are no longer retained.
* The same mechanism gives `attention_tile_ckpt` and `checkpoint1/2` (layer-level recompute with a parameter-gradient side channel, `CkptCtx`), used by S6; both are unit-tested against plain autograd (`cargo test -p minagi-core`).
  A custom op can only return gradients for its <= 3 tensor arguments, so a checkpointed layer's *weights* cannot receive gradients through autograd; they are parked in `CkptCtx` and merged into the `GradStore` after `backward()` (`Model::backward`).

## 6. S5 - end-to-end Tiny / Small training steps

Model (`spike/model.rs`): tied byte embedding (vocab 265); one dense prelude block; ONE shared recurrent block applied for 3 rows (adapter `Linear(512->256)`, causal attention with RoPE, MLP = **dense-masked MoE with 8 resident
experts, d_ff 256, top-2** or a dense SwiGLU of 512); RMSNorm; 4 heads x 64; chunk 512; 2.69M parameters (1.51M with the dense MLP); plain per-Var AdamW over the `GradStore`. Each step is wrapped in an `autoreleasepool` and ends
with the loss read-back and `synchronize()`. No halting/ACT, no paging, no auxiliary losses: this is a stand-in for timing, not the real model.

### 6.1 Step time and chars/s (chunk 512, batch 1, f32, 30 steps after 5 warm-up, three repeats)

| device | MLP | norms / RoPE | step median (3 runs: median, range) | chars/s | tensors per step | fwd graph nodes | params |
|---|---|---|---|---|---|---|---|
| metal | dense SwiGLU (512) | HandBwd | 21.8 ms (21.8-21.8) | **23516** (23510-23538) | 1623 | 190 | 1.51M |
| metal | dense SwiGLU (512) | composed | 27.7 ms (27.6-28.1) | **18473** (18244-18553) | 2232 | 342 | 1.51M |
| metal | dense-masked MoE (8x256, top-2) | HandBwd | 30.0 ms (29.8-30.1) | **17068** (16989-17177) | 1887 | 233 | 2.69M |
| metal | dense-masked MoE (8x256, top-2) | composed | 35.5 ms (35.5-35.5) | **14425** (14423-14434) | 2496 | 385 | 2.69M |
| cpu | dense SwiGLU (512) | HandBwd | 37.8 ms (37.5-38.0) | **13558** (13480-13643) | 1623 | 190 | 1.51M |
| cpu | dense SwiGLU (512) | composed | 42.9 ms (42.7-43.0) | **11945** (11900-12001) | 2232 | 342 | 1.51M |
| cpu | dense-masked MoE (8x256, top-2) | HandBwd | 52.3 ms (52.0-52.5) | **9787** (9761-9851) | 1887 | 233 | 2.69M |
| cpu | dense-masked MoE (8x256, top-2) | composed | 58.4 ms (58.2-58.4) | **8771** (8765-8793) | 2496 | 385 | 2.69M |

Estimates from the plan: 8-12k chars/s (Metal), 2-3.5k (CPU+Accelerate). **Measured: 14.4k-23.5k on Metal and 8.8k-13.6k on CPU.** "Tensors per step" counts every op result and view created in one step (fwd + bwd + optimizer); "fwd graph nodes" are the
forward autograd nodes (HandBwd collapses many composed nodes into a few custom nodes).

### 6.2 Where the time goes (Metal vs CPU)

Phase times with a `synchronize()` between phases (= GPU/CPU time per phase):

| device | MLP | norms / RoPE | forward ms | backward ms | optimizer ms | sum (synced between phases) |
|---|---|---|---|---|---|---|
| metal | dense SwiGLU (512) | HandBwd | 6.4 | 14.8 | 1.2 | 22.4 |
| metal | dense SwiGLU (512) | composed | 8.4 | 18.9 | 1.3 | 28.6 |
| metal | dense-masked MoE (8x256, top-2) | HandBwd | 8.6 | 20.3 | 1.6 | 30.5 |
| metal | dense-masked MoE (8x256, top-2) | composed | 10.5 | 23.7 | 1.7 | 35.9 |
| cpu | dense SwiGLU (512) | HandBwd | 12.7 | 23.3 | 1.6 | 37.6 |
| cpu | dense SwiGLU (512) | composed | 10.4 | 30.5 | 1.6 | 42.5 |
| cpu | dense-masked MoE (8x256, top-2) | HandBwd | 16.0 | 33.7 | 2.6 | 52.3 |
| cpu | dense-masked MoE (8x256, top-2) | composed | 15.0 | 41.2 | 2.6 | 58.8 |

And on Metal with the queue kept full, how much of the step the CPU spends encoding (the rest is waiting for the GPU):

| MLP | norms / RoPE | CPU encode fwd ms | bwd ms | opt ms | then waiting for the GPU ms | step ms |
|---|---|---|---|---|---|---|
| dense-masked MoE (8x256, top-2) | composed | 0.6 | 17.4 | 1.2 | 16.3 | 35.5 |
| dense SwiGLU (512) | composed | 0.4 | 14.8 | 0.9 | 11.7 | 27.7 |
| dense-masked MoE (8x256, top-2) | HandBwd | 0.4 | 11.4 | 0.9 | 17.4 | 30.1 |
| dense SwiGLU (512) | HandBwd | 0.2 | 7.8 | 0.7 | 13.0 | 21.8 |

* Metal is **GPU-bound**: the CPU finishes encoding in ~19 ms and then waits ~16 ms. Matmuls are only ~10-20% of the GPU time (estimated from the FLOP count: ~25 GFLOP per step is ~4-8 ms at 3-6.5 TFLOPS); the rest is ~2000 small elementwise / reduction /
  copy ops at ~12-25 us each plus the dense-masked MoE's `[512,2048]` tensors.
* HandBwd norms/RoPE: **-16% (MoE) / -21% (dense) on Metal**, -10% / -12% on CPU. The dense-masked MoE costs +8 ms per step over a dense SwiGLU at this size.
* Metal is only **~1.6x** faster than CPU+Accelerate at Tiny size (14.4k vs 8.8k chars/s), 2.7x at Small (6.8).

### 6.3 Batch size does not help on Metal

| device | batch (sequences of 512) | tokens/step | step ms | chars/s |
|---|---|---|---|---|
| metal | 1 | 512 | 35.2 | 14548 |
| metal | 2 | 1024 | 68.9 | 14865 |
| metal | 4 | 2048 | 142.4 | 14383 |
| metal | 8 | 4096 | 297.0 | 13790 |
| cpu | 4 | 2048 | 408.6 | 5012 |

Throughput is flat at ~14k chars/s from batch 1 to 8: the GPU is saturated by the small kernels, not underused. **Keep batch 1 (one 512-token chunk) for Tiny.**

### 6.4 dtype

| dtype (everything incl. optimizer state) | step ms | chars/s | loss after 35 steps | note |
|---|---|---|---|---|
| f32 | 35.5 | 14425 | 2.057 | stable |
| bf16 | 33.5 | 15265 | 2.230 | trains (pure bf16, no master weights) |
| f16 | 33.8 | 15147 | NaN | NaN without loss scaling / master weights |

bf16 is only ~6% faster and f16 NaNs without loss scaling / f32 master weights. **Stay in f32.**

### 6.5 `CANDLE_METAL_COMPUTE_PER_BUFFER` sweep

| CANDLE_METAL_COMPUTE_PER_BUFFER | step median (3 runs) | chars/s |
|---|---|---|
| 1 | 42.8 ms (42.6-43.2) | 11972 |
| 5 | 36.4 ms (36.2-36.9) | 14081 |
| 10 | 35.6 ms (35.5-35.6) | 14401 |
| 25 | 35.8 ms (35.8-36.2) | 14295 |
| 50 | 35.4 ms (35.2-35.4) | 14473 |
| 100 | 36.6 ms (36.3-37.0) | 14005 |
| 200 | 38.0 ms (37.7-38.2) | 13456 |
| 1000 | 48.0 ms (47.4-49.9) | 10665 |

Flat between 10 and 100; the default **50 is the best (or tied)**; 1 is 21% slower and 1000 is 36% slower. **Do not override it.**

### 6.6 Loss read-back / synchronize frequency

`loss.to_scalar()` and `synchronize()` both make candle sweep its Metal buffer pool, so the next step re-creates its buffers. Reading the loss only every N steps:

| MLP | norms / RoPE | loss readback + synchronize every N steps | wall-clock step (mean of 2 runs) | chars/s |
|---|---|---|---|---|
| dense SwiGLU (512) | HandBwd | 1 | 22.1 ms | 23261 |
| dense SwiGLU (512) | HandBwd | 10 | 21.0 ms | 24401 |
| dense SwiGLU (512) | HandBwd | 50 | 20.9 ms | 24532 |
| dense-masked MoE (8x256, top-2) | composed | 1 | 35.5 ms | 14444 |
| dense-masked MoE (8x256, top-2) | composed | 10 | 33.7 ms | 15202 |
| dense-masked MoE (8x256, top-2) | composed | 50 | 33.5 ms | 15276 |

Per-step read-back + synchronize costs **~5%**; for a 10-step cadence of loss logging the gain is about 1.5 ms. Not worth complicating the loop; keep the per-step read-back (it also bounds in-flight work).

### 6.7 Autoreleasepool / synchronize inside the training loop, and the optimizer-state leak

300 steps are too short to separate the four variants (step-time and slope differences between rows are noise; RSS 123 vs 67 MiB is the visible difference); the 2500-step runs are the evidence:

| autoreleasepool | explicit synchronize | step ms | chars/s | footprint MiB (300 steps) | RSS MiB | slope over 2nd half MiB/step |
|---|---|---|---|---|---|---|
| yes | yes | 39.4 | 13006 | 378 -> 774 (peak 817) | 67 | +0.04 |
| no | yes | 35.2 | 14562 | 378 -> 812 (peak 861) | 123 | +0.12 |
| yes | no | 39.9 | 12819 | 378 -> 763 (peak 864) | 68 | +0.33 |
| no | no | 39.0 | 13113 | 378 -> 822 (peak 862) | 124 | +0.62 |

| autoreleasepool | phys_footprint MiB at steps 500 / 1000 / ... / 2500 | slope over 2nd half MiB/step | final RSS MiB | final loss |
|---|---|---|---|---|
| yes | 500: 549, 1000: 560, 1500: 496, 2000: 571, 2500: 773 | +0.277 | 49 | 0.036 |
| no | 500: 823, 1000: 940, 1500: 1054, 2000: 1146, 2500: 1234 | +0.180 | 537 | 0.036 |

In a real training step the loss read-back retires the command buffers every step, yet **without the pool the loop still leaks ~0.2 MiB per step** (monotonic footprint growth, RSS 537 MiB vs 49 MiB after 2500 steps); with the pool the footprint stays within
about +/-60 MiB of ~550 MiB for 2000 steps (then one unexplained +220 MiB sample at step 2500, RSS unchanged at 49 MiB). The pool costs nothing measurable.

**The big one, the optimizer-state leak.** My first AdamW kept `m = m*b1 + g*(1-b1)` as the next state. Each such tensor keeps an op-graph reference to the previous state *and to the gradient tensors*, so every step retained the entire
history. `spike_s5 bench --leaky-optimizer` reproduces it (30 steps, release, Metal):

| step | 6 | 12 | 18 | 24 | 30 |
|---|---|---|---|---|---|
| phys_footprint MiB | 3948 | 6667 | 9379 | 12075 | 14820 |

(+450 MiB per step; the same on CPU: 244 -> 13.7 GB in 35 steps in the first dev-build observation.) The fix is `m.detach()` / `v.detach()`; it applies to **every tensor stored across steps**.

### 6.8 Small-ish point (d_model 384, 6 heads, 6 rows, 1024-token chunk, MoE 8x384)

| device | norms / RoPE | step ms (2 runs) | chars/s (best run) | tensors per step | peak phys_footprint MiB | RSS MiB |
|---|---|---|---|---|---|---|
| metal | HandBwd | 247, 246 | **4157** | 3039 | 5854 | 76 |
| metal | composed | 281, 277 | **3691** | 4092 | 7241 | 89 |
| cpu | HandBwd | 545, 547 | **1878** | 3039 | 3329 | 1939 |
| cpu | composed | 1000, 739 | **1385** | 4092 | 3749 | 2355 |

Memory at Small: peak footprint 7.1 GiB with composed ops, 5.7 GiB with HandBwd (7 attention layers with full `[1,6,1024,1024]` scores). Metal is 2.7x (composed) / 2.2x (HandBwd) faster than CPU+Accelerate here.

### 6.9 Overfit: loss trajectories

Synthetic 10 KB pseudo-prose (intrinsic entropy ~0.8 nats/char), random 512-token windows, lr 3e-3, warmup 20, 300 steps (step times in these runs include the batch generation and are not the benchmark numbers):

overfit: device=cpu text=10240 bytes (synthetic), chunk=512 steps=300 lr=0.003 warmup=20 params=2.69M ln(265)=5.580

| step | loss | elapsed s |
|---|---|---|
| 1 | 5.9307 | 0.1 |
| 25 | 2.2343 | 1.6 |
| 50 | 1.9598 | 3.2 |
| 75 | 1.8813 | 4.8 |
| 100 | 1.5799 | 6.3 |
| 125 | 1.3901 | 8.0 |
| 150 | 1.4896 | 10.5 |
| 175 | 1.2853 | 12.8 |
| 200 | 1.3113 | 15.0 |
| 225 | 1.2490 | 17.4 |
| 250 | 1.0430 | 19.8 |
| 275 | 1.1366 | 22.4 |
| 300 | 1.0256 | 25.2 |

RESULT overfit device=cpu: first 5.931, last 1.026, mean(last10) 0.916, min 0.789; first step below 2.0 / 1.0: Some(49) / Some(210); steady step median 89.3 ms (5736 chars/s); total 25.2s

overfit: device=metal text=10240 bytes (synthetic), chunk=512 steps=300 lr=0.003 warmup=20 params=2.69M ln(265)=5.580

| step | loss | elapsed s |
|---|---|---|
| 1 | 5.9307 | 0.1 |
| 25 | 2.2343 | 2.2 |
| 50 | 1.9596 | 4.3 |
| 75 | 1.9010 | 6.4 |
| 100 | 1.5359 | 8.4 |
| 125 | 1.4132 | 10.5 |
| 150 | 1.5078 | 12.6 |
| 175 | 1.3615 | 14.7 |
| 200 | 1.2532 | 16.8 |
| 225 | 1.1990 | 18.9 |
| 250 | 1.0564 | 20.9 |
| 275 | 1.0794 | 23.0 |
| 300 | 1.0465 | 25.1 |

RESULT overfit device=metal: first 5.931, last 1.046, mean(last10) 0.963, min 0.760; first step below 2.0 / 1.0: Some(49) / Some(210); steady step median 85.1 ms (6016 chars/s); total 25.1s

Single fixed 512-token batch (classic overfit sanity check), lr 1e-3:

overfit: device=cpu text=10240 bytes (synthetic), chunk=512 steps=300 lr=0.001 warmup=20 params=2.69M ln(265)=5.580

| step | loss | elapsed s |
|---|---|---|
| 1 | 6.0103 | 0.1 |
| 25 | 1.3829 | 1.6 |
| 50 | 0.0162 | 3.2 |
| 75 | 0.0014 | 4.7 |
| 100 | 0.1471 | 6.3 |
| 125 | 0.0018 | 7.8 |
| 150 | 0.0005 | 9.4 |
| 175 | 0.0003 | 11.1 |
| 200 | 0.0002 | 13.3 |
| 225 | 0.0001 | 15.3 |
| 250 | 0.0001 | 17.3 |
| 275 | 0.0001 | 19.4 |
| 300 | 0.0000 | 21.8 |

RESULT overfit device=cpu: first 6.010, last 0.000, mean(last10) 0.000, min 0.000; first step below 2.0 / 1.0: Some(19) / Some(30); steady step median 65.0 ms (7879 chars/s); total 21.8s

overfit: device=metal text=10240 bytes (synthetic), chunk=512 steps=300 lr=0.001 warmup=20 params=2.69M ln(265)=5.580

| step | loss | elapsed s |
|---|---|---|
| 1 | 6.0103 | 0.1 |
| 25 | 1.3829 | 3.2 |
| 50 | 0.0162 | 6.8 |
| 75 | 0.0014 | 9.2 |
| 100 | 0.0068 | 11.7 |
| 125 | 0.0014 | 13.4 |
| 150 | 0.0204 | 15.2 |
| 175 | 0.0082 | 16.1 |
| 200 | 0.0006 | 17.1 |
| 225 | 0.0002 | 18.2 |
| 250 | 0.0001 | 19.9 |
| 275 | 0.0001 | 21.3 |
| 300 | 0.0001 | 22.6 |

RESULT overfit device=metal: first 6.010, last 0.000, mean(last10) 0.000, min 0.000; first step below 2.0 / 1.0: Some(19) / Some(30); steady step median 51.4 ms (9956 chars/s); total 22.6s

Both devices follow the same trajectory (step-1 loss identical to 4 digits; they drift apart only by float summation order), cross 2.0 at step 49 and **1.0 at step 210**, and drive the single batch to 0.000 (below 1.0 at step 30).
The synthetic text has an entropy floor of ~0.8 nats/char, so mean(last 10) ~0.92-0.96 is "learned the distribution"; the 2500-step run above (random windows, constant lr 1e-3) reaches 0.036, i.e. it memorises the 10 KB.
On real prose (candle's README, 10 KB of markdown/code) loss was ~2.0 after 300 steps at lr 3e-3 (noisy, not memorised); that one-off run is not in the final suite.

### 6.10 Full-model gradient parity, Metal vs CPU (same init, same batch)

**S5 full-model gradient parity, Metal vs CPU (same init, same batch, f32, hand_ops=false)**

loss cpu 5.930692 / metal 5.930692 (rel diff 0.0e0)

| parameter | grad rel err (max abs diff / max abs cpu) |
|---|---|
| emb | 5.9e-7 |
| prelude.attn.ln | 7.3e-7 |
| prelude.attn.wq | 1.1e-6 |
| prelude.attn.wk | 7.4e-7 |
| prelude.attn.wv | 4.0e-7 |
| prelude.attn.wo | 5.1e-7 |
| prelude.mlp.ln | 3.5e-7 |
| prelude.mlp.w1 | 4.7e-7 |
| prelude.mlp.w3 | 4.5e-7 |
| prelude.mlp.w2 | 6.2e-7 |
| adapter | 5.7e-7 |
| recur.attn.ln | 6.9e-7 |
| recur.attn.wq | 8.4e-7 |
| recur.attn.wk | 8.9e-7 |
| recur.attn.wv | 5.0e-7 |
| recur.attn.wo | 6.7e-7 |
| recur.mlp.ln | 5.6e-7 |
| recur.mlp.router | 6.6e-7 |
| recur.mlp.wg | 5.4e-7 |
| recur.mlp.wu | 6.2e-7 |
| recur.mlp.wd | 3.6e-7 |
| ln_f | 4.4e-7 |

worst parameter gradient rel err: 1.1e-6 -> PASS (<1e-3, expected for a 3-row 20-op deep f32 graph; ops individually < 1e-4)
**S5 full-model gradient parity, Metal vs CPU (same init, same batch, f32, hand_ops=true)**

loss cpu 5.930692 / metal 5.930692 (rel diff 0.0e0)

| parameter | grad rel err (max abs diff / max abs cpu) |
|---|---|
| emb | 7.0e-7 |
| prelude.attn.ln | 7.3e-7 |
| prelude.attn.wq | 8.3e-7 |
| prelude.attn.wk | 7.1e-7 |
| prelude.attn.wv | 5.2e-7 |
| prelude.attn.wo | 6.5e-7 |
| prelude.mlp.ln | 6.2e-7 |
| prelude.mlp.w1 | 1.1e-6 |
| prelude.mlp.w3 | 5.4e-7 |
| prelude.mlp.w2 | 8.5e-7 |
| adapter | 5.7e-7 |
| recur.attn.ln | 9.9e-7 |
| recur.attn.wq | 1.0e-6 |
| recur.attn.wk | 1.2e-6 |
| recur.attn.wv | 6.2e-7 |
| recur.attn.wo | 1.1e-6 |
| recur.mlp.ln | 6.7e-7 |
| recur.mlp.router | 7.6e-7 |
| recur.mlp.wg | 8.1e-7 |
| recur.mlp.wu | 1.2e-6 |
| recur.mlp.wd | 5.3e-7 |
| ln_f | 4.7e-7 |

worst parameter gradient rel err: 1.2e-6 -> PASS (<1e-3, expected for a 3-row 20-op deep f32 graph; ops individually < 1e-4)
**S5 full-model gradient parity, Metal vs CPU (same init, same batch, f32, hand_ops=false)**

loss cpu 6.193407 / metal 6.193407 (rel diff 0.0e0)

| parameter | grad rel err (max abs diff / max abs cpu) |
|---|---|
| emb | 6.8e-7 |
| prelude.attn.ln | 8.0e-7 |
| prelude.attn.wq | 8.2e-7 |
| prelude.attn.wk | 7.1e-7 |
| prelude.attn.wv | 4.6e-7 |
| prelude.attn.wo | 5.9e-7 |
| prelude.mlp.ln | 9.0e-7 |
| prelude.mlp.w1 | 5.8e-7 |
| prelude.mlp.w3 | 4.9e-7 |
| prelude.mlp.w2 | 9.3e-7 |
| adapter | 5.0e-7 |
| recur.attn.ln | 8.1e-7 |
| recur.attn.wq | 1.1e-6 |
| recur.attn.wk | 9.8e-7 |
| recur.attn.wv | 5.6e-7 |
| recur.attn.wo | 7.3e-7 |
| recur.mlp.ln | 5.3e-7 |
| recur.mlp.w1 | 4.5e-7 |
| recur.mlp.w3 | 6.7e-7 |
| recur.mlp.w2 | 5.0e-7 |
| ln_f | 4.0e-7 |

worst parameter gradient rel err: 1.1e-6 -> PASS (<1e-3, expected for a 3-row 20-op deep f32 graph; ops individually < 1e-4)

Every parameter's gradient agrees to ~1e-6 through the whole 4-layer model (prelude + 3 shared rows, MoE router included); the same check passes for HandBwd ops and the dense MLP (appended above).

### 6.11 Cost of layer-level recompute on the Tiny model

| MLP | norms / RoPE | step ms without recompute | with layer recompute | cost | tensors per step | peak footprint MiB |
|---|---|---|---|---|---|---|
| dense-masked MoE (8x256, top-2) | composed | 35.5 | 46.8 | +32% | 3092 | 539 |
| dense SwiGLU (512) | HandBwd | 21.8 | 29.2 | +34% | 2007 | 330 |

Recompute adds ~one forward per layer (+32-34% step time) and cuts the live graph memory (S6).

## 7. S6 - full-size single recurrent row memory (T=4096, d_model 512, 8 heads)

`spike_s6`: prelude + `rows` recurrent rows (layers = rows + 1), dense-masked MoE (8 slots x 512) unless `--moe 0`, forward + backward of one 4096-token chunk (no optimizer), one process per configuration. Modes:
`full` = one `[1,8,T,T]` score matrix per layer; `tiled` = 512-query tiles, autograd retains every tile; `ckpt` = tiles each recomputed in backward; `lckpt` = whole layers (prelude and each row) recomputed in backward with tiled attention inside.
`peak` = lifetime peak of `phys_footprint` (includes candle's cached free buffers, so it overstates live memory); "MTLBuffer after fwd" = bytes the GPU holds after the forward with the graph retained.

| layers | mode | peak footprint | MTLBuffer after fwd | fwd+bwd (2nd step) |
|---|---|---|---|---|
| 1 | `full` | 9.65 GiB | 3.5 GiB | 776 ms |
| 1 | `tiled` | 4.89 GiB | 2.5 GiB | 331 ms |
| 1 | `ckpt` (per tile) | 6.72 GiB | 0.44 GiB | 434 ms |
| 1 | `lckpt` | 4.99 GiB | 0.17 GiB | 411 ms |
| 2 | `full` | **out of GPU memory** (7.2 GiB allocated after the forward, error in the backward) | 7.0 GiB | - |
| 2 | `tiled` | 9.94 GiB | 5.0 GiB | 657 ms |
| 2 | `ckpt` (per tile) | 8.93 GiB (6.76 GiB with a `synchronize()` after every tile, +22% time) | 1.0 GiB | 819 ms |
| 2 | `lckpt` | **5.86 GiB** | 0.18 GiB | 869 ms |
| 4 | `lckpt` | **5.97 GiB** | 0.20 GiB | 1739 ms |
| 6 | `lckpt` | **5.99 GiB** | 0.21 GiB | 2663 ms |
| 8 | `lckpt` | **6.01 GiB** | 0.23 GiB | 4080 ms (1.0k chars/s fwd+bwd) |
| 2 | `tiled`, no MoE | 9.28 GiB | 4.8 GiB | 603 ms |
| 4 | `tiled`, no MoE | 17.76 GiB (at the 17.8 GiB limit; an earlier dev-build run of this configuration failed with out-of-memory) | 9.4 GiB | 1440 ms |

Scaling of one full-attention layer with T: 0.92 GiB at T=1024, 2.77 GiB at T=2048, 9.7 GiB at T=4096 (x3-3.5 per doubling).

Raw output (all configurations, including the failed run):

**T=4096, d=512, 8 heads, prelude + rows recurrent rows (layers = rows + 1), dense-masked MoE**
S6 mode=full device=metal d_model=512 heads=8 T=4096 attention layers=1 q_block=0 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 285 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 3776 MiB (+3491), MTLBuffer allocated 3572 MiB
  step 0: loss 6.2624, fwd+bwd 3148 ms, footprint 9882 MiB, lifetime peak 9882 MiB
  step 1: after forward (graph retained) footprint 9139 MiB (+8854), MTLBuffer allocated 3516 MiB
  step 1: loss 6.2624, fwd+bwd 776 ms, footprint 9829 MiB, lifetime peak 9882 MiB
RESULT mode=full T=4096 d=512 heads=8 layers=1: peak phys_footprint 9.65 GiB, steady footprint 9.60 GiB, fwd+bwd median 3148 ms (1301 chars/s fwd+bwd only)
S6 mode=tiled device=metal d_model=512 heads=8 T=4096 attention layers=1 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 2850 MiB (+2491), MTLBuffer allocated 2572 MiB
  step 0: loss 6.2624, fwd+bwd 502 ms, footprint 5011 MiB, lifetime peak 5011 MiB
  step 1: after forward (graph retained) footprint 4647 MiB (+4288), MTLBuffer allocated 2516 MiB
  step 1: loss 6.2624, fwd+bwd 331 ms, footprint 4955 MiB, lifetime peak 5011 MiB
RESULT mode=tiled T=4096 d=512 heads=8 layers=1: peak phys_footprint 4.89 GiB, steady footprint 4.84 GiB, fwd+bwd median 502 ms (8157 chars/s fwd+bwd only)
S6 mode=ckpt device=metal d_model=512 heads=8 T=4096 attention layers=1 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 1114 MiB (+755), MTLBuffer allocated 835 MiB
  step 0: loss 6.2624, fwd+bwd 422 ms, footprint 6381 MiB, lifetime peak 6885 MiB
  step 1: after forward (graph retained) footprint 1531 MiB (+1172), MTLBuffer allocated 455 MiB
  step 1: loss 6.2624, fwd+bwd 434 ms, footprint 5979 MiB, lifetime peak 6885 MiB
RESULT mode=ckpt T=4096 d=512 heads=8 layers=1: peak phys_footprint 6.72 GiB, steady footprint 5.84 GiB, fwd+bwd median 434 ms (9440 chars/s fwd+bwd only)
S6 mode=lckpt device=metal d_model=512 heads=8 T=4096 attention layers=1 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 2681 MiB (+2322), MTLBuffer allocated 235 MiB
  step 0: loss 6.2624, fwd+bwd 421 ms, footprint 4704 MiB, lifetime peak 5115 MiB
  step 1: after forward (graph retained) footprint 2646 MiB (+2287), MTLBuffer allocated 179 MiB
  step 1: loss 6.2624, fwd+bwd 411 ms, footprint 4644 MiB, lifetime peak 5115 MiB
RESULT mode=lckpt T=4096 d=512 heads=8 layers=1: peak phys_footprint 4.99 GiB, steady footprint 4.54 GiB, fwd+bwd median 421 ms (9740 chars/s fwd+bwd only)
S6 mode=full device=metal d_model=512 heads=8 T=4096 attention layers=2 q_block=0 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 7483 MiB (+7124), MTLBuffer allocated 7197 MiB
Error: Metal error Command buffer had following error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)

Caused by:
    Command buffer had following error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
S6 mode=tiled device=metal d_model=512 heads=8 T=4096 attention layers=2 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 5484 MiB (+5125), MTLBuffer allocated 5197 MiB
  step 0: loss 6.2282, fwd+bwd 683 ms, footprint 10183 MiB, lifetime peak 10183 MiB
  step 1: after forward (graph retained) footprint 9544 MiB (+9185), MTLBuffer allocated 5141 MiB
  step 1: loss 6.2282, fwd+bwd 657 ms, footprint 10129 MiB, lifetime peak 10183 MiB
RESULT mode=tiled T=4096 d=512 heads=8 layers=2: peak phys_footprint 9.94 GiB, steady footprint 9.89 GiB, fwd+bwd median 683 ms (5997 chars/s fwd+bwd only)
S6 mode=ckpt device=metal d_model=512 heads=8 T=4096 attention layers=2 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 2012 MiB (+1654), MTLBuffer allocated 1724 MiB
  step 0: loss 6.2282, fwd+bwd 849 ms, footprint 8434 MiB, lifetime peak 9146 MiB
  step 1: after forward (graph retained) footprint 3116 MiB (+2757), MTLBuffer allocated 1014 MiB
  step 1: loss 6.2282, fwd+bwd 819 ms, footprint 7789 MiB, lifetime peak 9146 MiB
RESULT mode=ckpt T=4096 d=512 heads=8 layers=2: peak phys_footprint 8.93 GiB, steady footprint 7.61 GiB, fwd+bwd median 849 ms (4823 chars/s fwd+bwd only)
S6 mode=lckpt device=metal d_model=512 heads=8 T=4096 attention layers=2 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 3043 MiB (+2684), MTLBuffer allocated 243 MiB
  step 0: loss 6.2282, fwd+bwd 876 ms, footprint 5468 MiB, lifetime peak 6005 MiB
  step 1: after forward (graph retained) footprint 3092 MiB (+2733), MTLBuffer allocated 187 MiB
  step 1: loss 6.2282, fwd+bwd 869 ms, footprint 5405 MiB, lifetime peak 6005 MiB
RESULT mode=lckpt T=4096 d=512 heads=8 layers=2: peak phys_footprint 5.86 GiB, steady footprint 5.28 GiB, fwd+bwd median 876 ms (4677 chars/s fwd+bwd only)
**layer-level recompute at depth (the configuration that has to fit under ~14 GB)**
S6 mode=lckpt device=metal d_model=512 heads=8 T=4096 attention layers=4 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 2934 MiB (+2575), MTLBuffer allocated 259 MiB
  step 0: loss 6.2452, fwd+bwd 1745 ms, footprint 5504 MiB, lifetime peak 6118 MiB
  step 1: after forward (graph retained) footprint 2903 MiB (+2544), MTLBuffer allocated 203 MiB
  step 1: loss 6.2452, fwd+bwd 1739 ms, footprint 5127 MiB, lifetime peak 6118 MiB
RESULT mode=lckpt T=4096 d=512 heads=8 layers=4: peak phys_footprint 5.97 GiB, steady footprint 5.01 GiB, fwd+bwd median 1745 ms (2347 chars/s fwd+bwd only)
S6 mode=lckpt device=metal d_model=512 heads=8 T=4096 attention layers=6 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 3077 MiB (+2718), MTLBuffer allocated 275 MiB
  step 0: loss 6.2068, fwd+bwd 2612 ms, footprint 5522 MiB, lifetime peak 6134 MiB
  step 1: after forward (graph retained) footprint 2956 MiB (+2597), MTLBuffer allocated 219 MiB
  step 1: loss 6.2068, fwd+bwd 2663 ms, footprint 5438 MiB, lifetime peak 6134 MiB
RESULT mode=lckpt T=4096 d=512 heads=8 layers=6: peak phys_footprint 5.99 GiB, steady footprint 5.28 GiB, fwd+bwd median 2663 ms (1538 chars/s fwd+bwd only)
S6 mode=lckpt device=metal d_model=512 heads=8 T=4096 attention layers=8 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 3037 MiB (+2678), MTLBuffer allocated 291 MiB
  step 0: loss 6.1623, fwd+bwd 3946 ms, footprint 5530 MiB, lifetime peak 6150 MiB
  step 1: after forward (graph retained) footprint 3138 MiB (+2780), MTLBuffer allocated 235 MiB
  step 1: loss 6.1623, fwd+bwd 4080 ms, footprint 5354 MiB, lifetime peak 6150 MiB
RESULT mode=lckpt T=4096 d=512 heads=8 layers=8: peak phys_footprint 6.01 GiB, steady footprint 5.23 GiB, fwd+bwd median 4080 ms (1004 chars/s fwd+bwd only)
**per-tile pool trim (MINAGI_SYNC_TILES=1) for per-tile checkpointing**
S6 mode=ckpt device=metal d_model=512 heads=8 T=4096 attention layers=2 q_block=512 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 1921 MiB (+1562), MTLBuffer allocated 1740 MiB
  step 0: loss 6.2282, fwd+bwd 1005 ms, footprint 6440 MiB, lifetime peak 6920 MiB
  step 1: after forward (graph retained) footprint 1874 MiB (+1515), MTLBuffer allocated 1677 MiB
  step 1: loss 6.2282, fwd+bwd 1000 ms, footprint 6385 MiB, lifetime peak 6920 MiB
RESULT mode=ckpt T=4096 d=512 heads=8 layers=2: peak phys_footprint 6.76 GiB, steady footprint 6.22 GiB, fwd+bwd median 1005 ms (4076 chars/s fwd+bwd only)
**scaling of full T x T attention with T (1 layer)**
S6 mode=full device=metal d_model=512 heads=8 T=1024 attention layers=1 q_block=0 params=10.6M; one [1,8,1024,1024] f32 score tensor = 32 MiB
before: footprint 115 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 480 MiB (+365), MTLBuffer allocated 324 MiB
  step 0: loss 6.2833, fwd+bwd 65 ms, footprint 943 MiB, lifetime peak 943 MiB
  step 1: after forward (graph retained) footprint 932 MiB (+815), MTLBuffer allocated 322 MiB
  step 1: loss 6.2833, fwd+bwd 45 ms, footprint 941 MiB, lifetime peak 943 MiB
RESULT mode=full T=1024 d=512 heads=8 layers=1: peak phys_footprint 0.92 GiB, steady footprint 0.92 GiB, fwd+bwd median 65 ms (15830 chars/s fwd+bwd only)
S6 mode=full device=metal d_model=512 heads=8 T=2048 attention layers=1 q_block=0 params=10.6M; one [1,8,2048,2048] f32 score tensor = 128 MiB
before: footprint 166 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 1188 MiB (+1022), MTLBuffer allocated 1007 MiB
  step 0: loss 6.3096, fwd+bwd 173 ms, footprint 2837 MiB, lifetime peak 2837 MiB
  step 1: after forward (graph retained) footprint 2774 MiB (+2608), MTLBuffer allocated 995 MiB
  step 1: loss 6.3096, fwd+bwd 160 ms, footprint 2825 MiB, lifetime peak 2837 MiB
RESULT mode=full T=2048 d=512 heads=8 layers=1: peak phys_footprint 2.77 GiB, steady footprint 2.76 GiB, fwd+bwd median 173 ms (11805 chars/s fwd+bwd only)
S6 mode=full device=metal d_model=512 heads=8 T=4096 attention layers=1 q_block=0 params=10.6M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 359 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 3849 MiB (+3490), MTLBuffer allocated 3572 MiB
  step 0: loss 6.2624, fwd+bwd 680 ms, footprint 9958 MiB, lifetime peak 9958 MiB
  step 1: after forward (graph retained) footprint 9225 MiB (+8866), MTLBuffer allocated 3516 MiB
  step 1: loss 6.2624, fwd+bwd 707 ms, footprint 9902 MiB, lifetime peak 9958 MiB
RESULT mode=full T=4096 d=512 heads=8 layers=1: peak phys_footprint 9.72 GiB, steady footprint 9.67 GiB, fwd+bwd median 707 ms (5798 chars/s fwd+bwd only)
**without MoE (isolate attention); tiled without recompute at depth hits the Metal working-set limit**
S6 mode=tiled device=metal d_model=512 heads=8 T=4096 attention layers=2 q_block=512 params=5.9M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 315 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 5175 MiB (+4860), MTLBuffer allocated 4922 MiB
  step 0: loss 6.2132, fwd+bwd 611 ms, footprint 9502 MiB, lifetime peak 9502 MiB
  step 1: after forward (graph retained) footprint 8759 MiB (+8444), MTLBuffer allocated 4866 MiB
  step 1: loss 6.2132, fwd+bwd 603 ms, footprint 9449 MiB, lifetime peak 9502 MiB
RESULT mode=tiled T=4096 d=512 heads=8 layers=2: peak phys_footprint 9.28 GiB, steady footprint 9.23 GiB, fwd+bwd median 611 ms (6704 chars/s fwd+bwd only)
S6 mode=tiled device=metal d_model=512 heads=8 T=4096 attention layers=4 q_block=512 params=5.9M; one [1,8,4096,4096] f32 score tensor = 512 MiB
before: footprint 315 MiB; recommendedMaxWorkingSetSize 18186 MiB
  step 0: after forward (graph retained) footprint 9914 MiB (+9599), MTLBuffer allocated 9659 MiB
  step 0: loss 6.1937, fwd+bwd 1494 ms, footprint 18158 MiB, lifetime peak 18158 MiB
  step 1: after forward (graph retained) footprint 17187 MiB (+16872), MTLBuffer allocated 9603 MiB
  step 1: loss 6.1937, fwd+bwd 1440 ms, footprint 18182 MiB, lifetime peak 18182 MiB
RESULT mode=tiled T=4096 d=512 heads=8 layers=4: peak phys_footprint 17.76 GiB, steady footprint 17.76 GiB, fwd+bwd median 1494 ms (2742 chars/s fwd+bwd only)

Conclusions:

* **The T x T problem is real**: one `[1,8,4096,4096]` f32 score tensor is 512 MiB and autograd keeps ~6-8 of them per layer. One full-attention layer peaks at 9.7 GiB; two layers fail with
  `Error: Metal error Command buffer had following error: Insufficient Memory (...kIOGPUCommandBufferCallbackErrorOutOfMemory)`, returned as an ordinary `Err` at the next synchronize (the process survives).
* **Tiling by 512 queries alone is not enough**: it halves the cost (causal skip) and is 2.3x *faster* than full attention at 1 layer, but autograd still retains every tile, so memory grows ~4-5 GiB per layer and runs into the working-set limit at 4 layers.
* **Per-tile recompute (`ckpt`)** shrinks what the graph retains (0.4-1 GiB after the forward) but the peak is dominated by the backward transients plus candle's cached buffers.
* **Layer-level recompute (`lckpt`) with 512-query tiles is the configuration that fits**: the peak is flat (~6.0 GiB) from 2 to 8 layers because only one layer's transients are alive at a time; the cost is one extra forward per layer
  (~30%, ~0.5 s per layer at T=4096 with this dense-masked MoE). Well under the ~14 GB target and the 17.8 GiB limit.
* `phys_footprint` overstates live memory: after the forward `lckpt` holds 0.2 GiB of `MTLBuffer`s while the process footprint reads ~2.6 GiB (cached/free buffers and Metal's own allocations). candle's pool rounds buffer sizes **up to a power of two**
  and is only swept on `synchronize()`.

## 8. S7 - npz / bf16 round trip

Code (production quality, doc comments, 19 unit tests + 4 numpy-fixture tests): `crates/minagi-core/src/store/{mod,npy,npz,bf16}.rs`; fixtures `crates/minagi-core/tests/fixtures/numpy_{stored,deflate}.npz` (written by numpy 2.5.3);
interop scripts `crates/minagi-cli/scripts/spike_s7_{make_numpy,verify}.py` driving `spike_s7`.

* `.npy` v1.0 writer (64-byte aligned header, numpy's exact dict text and padding rule), reader for v1.0/2.0/3.0, dtypes `<f4 <f8 <i2 <i4 <i8 |u1`, 0-d arrays, empty arrays, C order only (Fortran order, big-endian and other dtypes are rejected with an error).
* `.npz` writer: *stored* entries, deterministic timestamps (byte-reproducible output), zip64 only for entries >= 4 GiB; reader: stored and deflate (`numpy.savez_compressed`). Keys are arbitrary strings
  (`recur.0.mlp.router.weight`, `adam|m|...`, spaces and unicode are tested).
* bf16: `f32 -> u16` round-to-nearest-even with NaN handling (a NaN whose payload sits only in the low 16 bits stays a NaN; overflow rounds to inf), stored as `<i2` (`pack`); `unpack` is `f32::from_bits((x as u16 as u32) << 16)`.

numpy <-> Rust verification (`spike_s7_verify.py`; numpy writes -> Rust reads and re-writes -> numpy compares, and Rust-constructed arrays -> numpy):

PASS np_stored.npz->rust->rust_from_stored.npz: same key set/order  [['recur.0.mlp.router.weight', 'bf16|w.0', 'step', 'special', 'ids', 'empty', 'bytes', 'i4', 'adam|v|recur.1.attn.wq.weight']]
PASS np_stored.npz->rust->rust_from_stored.npz: all arrays bit exact (dtype, shape, bytes incl. NaN payloads)
PASS rust_from_stored.npz: every .npy entry byte-identical to numpy's (header padding, dict text)
PASS rust_from_stored.npz: entries stored (ZIP_STORED) like np.savez
PASS rust_from_stored.npz: zip CRCs valid (testzip)
PASS np_deflate.npz->rust->rust_from_deflate.npz: same key set/order  [['recur.0.mlp.router.weight', 'bf16|w.0', 'step', 'special', 'ids', 'empty', 'bytes', 'i4', 'adam|v|recur.1.attn.wq.weight']]
PASS np_deflate.npz->rust->rust_from_deflate.npz: all arrays bit exact (dtype, shape, bytes incl. NaN payloads)
PASS rust_from_deflate.npz: every .npy entry byte-identical to numpy's (header padding, dict text)
PASS rust_from_deflate.npz: entries stored (ZIP_STORED) like np.savez
PASS rust_from_deflate.npz: zip CRCs valid (testzip)
PASS bf16 packed dtype is <i2  [int16]
PASS bf16 pack == ml_dtypes RNE for all 1310729 values (non-NaN bit exact)  [0 mismatches]
PASS bf16 pack: NaN inputs stay NaN  [4190 NaNs]
PASS bf16 unpack == ml_dtypes widening, bit exact
PASS bf16 unpack NaN stays NaN
PASS rust_made: key set  [['recur.0.mlp.router.weight', 'bf16|weights.0', 'train|step', 'lr', 'tokens|i2', 'big', 'empty.rank2', 'ünï key with spaces|and.dots']]
PASS rust_made['recur.0.mlp.router.weight'] dtype=float32 shape=(64, 32) bit exact
PASS rust_made['bf16|weights.0'] dtype=int16 shape=(64, 32) bit exact
PASS rust_made['train|step'] dtype=float64 shape=() bit exact
PASS rust_made['lr'] dtype=float64 shape=() bit exact
PASS rust_made['tokens|i2'] dtype=int16 shape=(1000,) bit exact
PASS rust_made['big'] dtype=float32 shape=(4194304,) bit exact
PASS rust_made['empty.rank2'] dtype=float32 shape=(0, 7) bit exact
PASS rust_made['ünï key with spaces|and.dots'] dtype=int32 shape=(3,) bit exact
PASS rust_made: 0-d scalar has ndim 0
PASS rust_made: stored entries, CRC ok
PASS rust_made: each entry byte-identical to np.save()

OVERALL: PASS

Rust tests (`cargo test -p minagi-core`, dev profile; the first group is the 34 library tests, including the non-S7 spike tests):

test store::npy::tests::rejects_bad_input ... ok
test store::npy::tests::scalar_header_uses_empty_tuple ... ok
test spike::ops::tests::rope_composed_matches_candle_slow_rank5 ... ok
test store::npy::tests::roundtrip_all_dtypes_and_shapes ... ok
test store::npy::tests::shape_data_mismatch_is_an_error ... ok
test spike::handbwd::tests::hand_op_in_a_graph_with_two_consumers_accumulates ... ok
test spike::ops::tests::tiled_attention_equals_full_and_is_causal ... ok
test store::npz::tests::missing_key_and_bad_archive ... ok
test spike::handbwd::tests::rope_hand_matches_composed_value_and_grads ... ok
test store::npz::tests::entries_are_stored_not_compressed_and_deterministic ... ok
test store::npz::tests::reads_deflated_archives ... ok
test store::npz::tests::roundtrip_in_memory_with_odd_keys ... ok
test store::npz::tests::file_roundtrip ... ok
test spike::handbwd::tests::rms_norm_hand_matches_composed_value_and_grads ... ok
test spike::model::tests::model_is_causal ... ok
test spike::handbwd::tests::checkpointed_tiled_attention_matches_full_attention_grads ... ok
test spike::moe::tests::dense_sparse_stock_sparse_fast_agree_on_cpu ... ok
test spike::model::tests::hand_ops_model_matches_composed_model ... ok
test spike::model::tests::layer_checkpointing_matches_plain_autograd ... ok
test store::bf16::tests::matches_half_crate_on_random_and_special_values ... ok
test spike::model::tests::all_params_receive_gradients_and_adamw_overfits_one_batch ... ok

test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.11s

     Running tests/numpy_fixtures.rs (target-spike/debug/deps/numpy_fixtures-6b6edd8ce26b5b3f)

running 4 tests
test bf16_pack_matches_ml_dtypes_for_fixture_values ... ok
test reads_numpy_savez_stored ... ok
test rust_written_archive_has_the_same_arrays_as_numpy_fixture ... ok
test reads_numpy_savez_compressed ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests minagi_core

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

Not verified: files > 4 GiB (the zip64 path is enabled by size but untested), structured dtypes / object arrays / Fortran order (rejected on purpose), and NaN *payload* equality with ml_dtypes (only NaN-ness is compared).

## 9. Candle rules after the spikes

| Rule | Status |
|---|---|
| R1 pin candle to a git rev | **Confirmed and strengthened**: crates.io 0.11.0 does not build on rustc 1.93 / aarch64 (unstable `stdarch_neon_f16`) and has the #3847 NaN bug. |
| R2 fused ops have no gradient | **Confirmed** (`rms_norm`, `rope_i`, `softmax_last_dim`: gradient absent on CPU and Metal). Use composed ops or HandBwd. |
| R3 no `no_grad`, anything derived from a Var builds a graph | **Confirmed, with a nasty corollary (new R8):** every tensor carried across steps must be `.detach()`ed (optimizer state, recurrent state, KV cache, EMA). Missing it leaked ~15 GB in 30 steps. A forward computed for checkpointing must also be `.detach()`ed to free its intermediates. |
| R5 rank <= 4, contiguous ids, no broadcast views in matmul | Stride-0 matmul operands, GQA expand and rank-5 `rope_i_slow` are **correct at the pin** (parity + FD); rank-5 strided reduces were the #3847 bug. Keep rank <= 4 anyway (cheap; the composed ops here are rank <= 4). |
| R6 `autoreleasepool` + `synchronize` per step | **Confirmed**: ~0.2 KiB per iteration leaked in release (1.8 KiB in debug) without the pool, flat with it, no speed cost. RSS under-reports; monitor `phys_footprint`. |
| R7 Metal needs macOS >= 15 | Not testable here (macOS 26.6). |
| new R9 | `scatter_add` backward is wrong (shape error and masked `init` gradient); use `scatter` on zeros with unique indices (or a one-hot matmul). |
| new R10 | candle's Metal pool is power-of-two bucketed and swept only on `synchronize()`; peak footprint can be 1.5-2x the live set; exceeding `recommendedMaxWorkingSetSize` (17.8 GiB here) returns `Err(...OutOfMemory)`. |
| new R11 | CPU+Accelerate has no f16 matmul; Metal half precision gives <= 7% on GEMM and ~6% on a step. |
| new R12 | Metal kernel quality: gathers 16 GB/s, strided copy/broadcast 30-45 GB/s, strided middle-dim `sum` slow; elementwise ~210 GB/s is fine. Tiny steps are bound by ~12 us per dispatch. Fuse (HandBwd, custom kernels later). |
| new R13 | Use release builds for any Metal measurement; objc2's debug-assertion signature checks slow Metal calls ~2.3x and enlarge the autorelease leak ~8x. |

## 10. Recommended defaults

| Setting | Recommendation | Evidence |
|---|---|---|
| Training device | **Metal when available** for Tiny and up (1.6x CPU at Tiny, 2.7x at Small); CPU+Accelerate is a fully working fallback (8.8k chars/s Tiny, 1.4k Small) | 6.1, 6.8 |
| dtype | **f32** parameters, activations and optimizer state; no f16 (NaN without loss scaling), bf16 only ~6% | 2.1, 6.4 |
| Chunk / batch | Tiny: 512 tokens, batch 1 (throughput is flat in batch); Small: 1024; for T >= 2048 use 512-query tiles **and layer-level recompute** | 6.3, 7 |
| `CANDLE_METAL_COMPUTE_PER_BUFFER` | leave the default (50) | 6.5 |
| Step wrapper | `autoreleasepool` around every step in every build + loss read-back and `synchronize()` per step | 2.4, 6.6, 6.7 |
| Optimizer | own per-var AdamW over `GradStore`; **detach carried state** | 6.7 |
| Norm / RoPE | composed versions as the reference/test oracle, **HandBwd** (fused fwd, hand bwd) in production behind the parity gate | 5, 6.1 |
| MoE | dense-masked for <= ~16 resident experts (Tiny/Small); sparse dispatch (capacity + `u32::MAX` sentinels) from ~64 resident experts; never `scatter_add` for gains | 4 |
| Attention | composed softmax attention at rank 4 with a detached max; 512-query tiles for T >= 2048, with `checkpoint` per layer | 3, 7 |
| Checkpoints | `.npz` (stored): bf16 weights as `<i2`, f32 optimizer state, 0-d `<f8` scalars | 8 |
| Dependency pin | candle git rev `5ba5d5b` for all four crates; revisit when a release contains #3873/#3960/#3973 *and* builds on the project's rustc | 3.4 |

## 11. What was not verified / limitations

* **Timing environment:** shared machine, see the note at the top; the quiet-run repeats agree within ~1-3% for Metal, but GPU contention from other Metal jobs cannot be excluded for any single run, and CPU numbers are the most sensitive.
* No Metal GPU profiler (Xcode is not installed): "GPU-bound" is inferred from phase timings, CPU-encode vs wait time and batch scaling, not from a capture.
* The Tiny/Small models are stand-ins (no halting/ACT, no paging, no aux losses, no real data pipeline). Single seed everywhere.
* Real-prose overfit was only run once for 300 steps (not memorised, not repeated in the final suite).
* S6 peaks are `phys_footprint` (they include cached free buffers); S6 timings are two steps in a fresh process, no optimizer. The "fits under ~14 GB" statement refers to that metric.
* The S3 sparse block timings use a pre-built host plan; the end-to-end sparse step (with the per-layer host round trip) was not built.
* Not tested: CUDA/MKL features, Intel macs, macOS < 15, multi-GPU, bf16 CPU matmul speed, f16/bf16 master-weight training schemes, S7 files > 4 GiB.
* The root `Cargo.toml` needed no change. I removed the (stub) `minagi-types` dependency from `minagi-core` and `minagi-cli` so their builds do not depend on another process's work in progress; add it back when that crate is stable.
  `Cargo.lock` at the repo root did not exist before this work and was created by cargo.
* `target-spike/` (my scratch `CARGO_TARGET_DIR`, several GB) lives in the repo root and is not covered by `.gitignore`; delete it or ignore it.

## 12. Reproduce

```sh
# everything (release; ~1 h; each benchmark waits for a quiet machine and tags its output with load/probe)
crates/minagi-cli/scripts/run_spikes.sh /tmp/spike-out
# single spikes
cargo run --release -p minagi-cli --bin spike_s5 -- bench --device metal --hand --dense-mlp
cargo run --release -p minagi-cli --bin spike_s6 -- --mode lckpt --chunk 4096 --qblock 512 --rows 7
cargo test -p minagi-core
# S7 numpy side: python3 -m venv v && v/bin/pip install numpy ml_dtypes
v/bin/python crates/minagi-cli/scripts/spike_s7_make_numpy.py OUT/np crates/minagi-core/tests/fixtures
target/release/spike_s7 rewrite OUT/np OUT/rs && v/bin/python crates/minagi-cli/scripts/spike_s7_verify.py OUT/np OUT/rs
```

Quality gates (all green at the end): `cargo build -p minagi-core -p minagi-cli`, `cargo test -p minagi-core` (34 + 4 tests, dev profile), `cargo clippy -p minagi-core -p minagi-cli --all-targets` (no warnings), `cargo fmt` with the repo's `rustfmt.toml`.

Files created (everything under `crates/minagi-core` and `crates/minagi-cli`, plus this report):

* `minagi-core/Cargo.toml` (deps), `src/lib.rs` (re-exports candle), `src/spike/{mod,dev,rng,stats,ops,handbwd,moe,model}.rs`, `src/store/{mod,npy,npz,bf16}.rs`, `tests/numpy_fixtures.rs`, `tests/fixtures/numpy_{stored,deflate}.npz`
* `minagi-cli/Cargo.toml`, `src/main.rs`, `src/bin/spike_s{1,2,3,4,5,6,7}.rs`, `scripts/{run_spikes.sh,spike_s7_make_numpy.py,spike_s7_verify.py}`, `scripts/candle0110_repro/{Cargo.toml,src/main.rs}` (standalone #3847 repro on crates.io 0.11.0)
* `docs/spike-results.md`
