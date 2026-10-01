#!/usr/bin/env bash
# Re-run the engine go/no-go spikes (S1-S7) and collect raw outputs.
#   usage: run_spikes.sh <out_dir> [target_dir]
# Builds with the root release profile. Timings are only meaningful on an otherwise idle machine:
# every benchmark first waits (up to $GATE_SECONDS, default 90 s) for a quiet-machine probe and records the load
# average and probe result before/after as HTML comments in its output.
set -uo pipefail
OUT=${1:?usage: run_spikes.sh <out_dir> [target_dir]}
export CARGO_TARGET_DIR=${2:-$PWD/target-spike}
mkdir -p "$OUT"

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
# probe: "<metal_us> <cpu_us>" for a fixed GEMM (quiet reference ~980 / ~1760 us); the machine counts as
# quiet when both are within 25% of the reference. Waits at most $GATE_SECONDS (default 90) and then
# runs anyway, tagging the output so noisy runs can be repeated.
probe() { "$CARGO_TARGET_DIR/release/spike_s1" probe 2>/dev/null || echo "9999 9999"; }
quiet() { local p m c; p=$(probe); m=${p% *}; c=${p#* }; [ "$m" -lt 1230 ] && [ "$c" -lt 2200 ]; }
gate() {
  local start=$SECONDS
  while [ $((SECONDS - start)) -lt "${GATE_SECONDS:-90}" ]; do
    quiet && return 0
    sleep 3
  done
  return 0
}
# run <outfile> <cmd...>: gated, appends stdout/stderr to outfile, tags it with probe/load before and after
run() {
  local f=$1; shift
  gate
  echo "<!-- $(date +%T) load1=$(load1) probe=[$(probe)] :: $* -->" >> "$f"
  "$@" >> "$f" 2>&1 || echo "<!-- exit status $? -->" >> "$f"
  echo "<!-- end load1=$(load1) probe=[$(probe)] -->" >> "$f"
}

cargo build --release -p minagi-core -p minagi-cli || exit 1
BIN="$CARGO_TARGET_DIR/release"
uptime | tee "$OUT/load.txt"

# S1: GEMM / launch / reductions / kernel throughput / leak matrix (one process per leak config)
run "$OUT/s1_gemm.txt"   "$BIN/spike_s1" gemm
run "$OUT/s1_launch.txt" "$BIN/spike_s1" launch
run "$OUT/s1_reduce.txt" "$BIN/spike_s1" reduce
run "$OUT/s1_ops.txt"    "$BIN/spike_s1" ops
: > "$OUT/s1_leak.txt"
for sc in mm step; do for pool in 0 1; do for sync in 0 1; do
  run "$OUT/s1_leak.txt" "$BIN/spike_s1" leak $sc $pool $sync 200000 20000
done; done; done

# S2: op zoo, R2, finite differences, issue 3847 repro
run "$OUT/s2.txt" "$BIN/spike_s2" all

# S3: MoE parity and full-size timings
run "$OUT/s3.txt" "$BIN/spike_s3" all

# S4: HandBwd ops
run "$OUT/s4.txt" "$BIN/spike_s4" all

# S5: Tiny/Small steps, sweeps, overfit (the matrix is run 3x per configuration; the report uses the repeats' median/range)
: > "$OUT/s5_matrix.txt"
for dev in metal cpu; do for hand in "" "--hand"; do for mlp in "" "--dense-mlp"; do
  for rep in 1 2 3; do run "$OUT/s5_matrix.txt" "$BIN/spike_s5" bench --device $dev --steps 30 --warm 5 $hand $mlp; done
  run "$OUT/s5_matrix.txt" "$BIN/spike_s5" bench --device $dev --steps 30 --warm 5 $hand $mlp --profile
done; done; done
: > "$OUT/s5_bench.txt"
for b in 1 2 4 8; do run "$OUT/s5_batch.txt" "$BIN/spike_s5" bench --device metal --steps 15 --warm 4 --batch $b; done
run "$OUT/s5_batch.txt" "$BIN/spike_s5" bench --device cpu --steps 10 --warm 3 --batch 4
# loss read-back / synchronize cadence
for ev in 1 10 50; do for rep in 1 2; do
  run "$OUT/s5_readevery.txt" "$BIN/spike_s5" bench --device metal --steps 100 --warm 10 --read-every $ev
  run "$OUT/s5_readevery.txt" "$BIN/spike_s5" bench --device metal --steps 100 --warm 10 --read-every $ev --hand --dense-mlp
done; done
for dt in bf16 f16; do run "$OUT/s5_bench.txt" "$BIN/spike_s5" bench --device metal --steps 20 --warm 5 --dtype $dt; done
# optimizer-state graph retention (the leak): 30 steps without detaching the carried state
run "$OUT/s5_bench.txt" "$BIN/spike_s5" bench --device metal --steps 30 --warm 2 --leaky-optimizer
for pb in 1 5 10 25 50 100 200 1000; do for rep in 1 2 3; do
  echo "== CANDLE_METAL_COMPUTE_PER_BUFFER=$pb" >> "$OUT/s5_perbuffer.txt"
  CANDLE_METAL_COMPUTE_PER_BUFFER=$pb run "$OUT/s5_perbuffer.txt" "$BIN/spike_s5" bench --device metal --steps 30 --warm 5
done; done
for dev in metal cpu; do for hand in "" "--hand"; do for rep in 1 2; do
  run "$OUT/s5_small.txt" "$BIN/spike_s5" bench --device $dev --preset small --steps 8 --warm 3 $hand
done; done; done
for flags in "" "--no-pool" "--no-sync" "--no-pool --no-sync"; do
  run "$OUT/s5_poolsync.txt" "$BIN/spike_s5" bench --device metal --steps 300 --warm 5 $flags
done
for flags in "" "--no-pool"; do
  run "$OUT/s5_longrun.txt" "$BIN/spike_s5" bench --device metal --steps 2500 --warm 5 $flags
done
run "$OUT/s5_ckpt_tiny.txt" "$BIN/spike_s5" bench --device metal --steps 20 --warm 4 --layer-ckpt
run "$OUT/s5_ckpt_tiny.txt" "$BIN/spike_s5" bench --device metal --steps 20 --warm 4 --layer-ckpt --hand --dense-mlp
: > "$OUT/s5_parity.txt"
run "$OUT/s5_parity.txt" "$BIN/spike_s5" parity
run "$OUT/s5_parity.txt" "$BIN/spike_s5" parity --hand
run "$OUT/s5_parity.txt" "$BIN/spike_s5" parity --dense-mlp
for dev in cpu metal; do
  run "$OUT/s5_overfit_${dev}_synthetic.txt" "$BIN/spike_s5" overfit --device $dev --steps 300
  run "$OUT/s5_overfit_${dev}_fixed.txt"     "$BIN/spike_s5" overfit --device $dev --steps 300 --fixed-batch --lr 1e-3
done

# S6: single-row memory (one process per config; the lifetime footprint peak needs a fresh process)
: > "$OUT/s6.txt"
echo "## T=4096, d=512, 8 heads, prelude + rows recurrent rows (layers = rows + 1), dense-masked MoE" >> "$OUT/s6.txt"
for rows in 0 1; do for mode in full tiled ckpt lckpt; do
  run "$OUT/s6.txt" "$BIN/spike_s6" --mode $mode --chunk 4096 --qblock 512 --rows $rows --steps 2
done; done
echo "## layer-level recompute at depth (the configuration that has to fit under ~14 GB)" >> "$OUT/s6.txt"
for rows in 3 5 7; do run "$OUT/s6.txt" "$BIN/spike_s6" --mode lckpt --chunk 4096 --qblock 512 --rows $rows --steps 2; done
echo "## per-tile pool trim (MINAGI_SYNC_TILES=1) for per-tile checkpointing" >> "$OUT/s6.txt"
MINAGI_SYNC_TILES=1 run "$OUT/s6.txt" "$BIN/spike_s6" --mode ckpt --chunk 4096 --qblock 512 --rows 1 --steps 2
echo "## scaling of full T x T attention with T (1 layer)" >> "$OUT/s6.txt"
for c in 1024 2048 4096; do run "$OUT/s6.txt" "$BIN/spike_s6" --mode full --chunk $c --rows 0 --steps 2; done
echo "## without MoE (isolate attention); tiled without recompute at depth hits the Metal working-set limit" >> "$OUT/s6.txt"
for rows in 1 3; do run "$OUT/s6.txt" "$BIN/spike_s6" --mode tiled --chunk 4096 --qblock 512 --rows $rows --moe 0 --steps 2; done

# S7: unit + interop tests (python venv needed for the numpy side, see scripts/spike_s7_*.py)
cargo test -p minagi-core 2>&1 | tail -40 > "$OUT/s7_cargo_test.txt"
echo "done -> $OUT"
