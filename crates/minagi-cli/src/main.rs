//! `minagi-cli`: command line tools for the engine. Each is its own binary:
//!
//! - `minagi_train`: the same engine the app runs, from the command line (`--synthetic` makes a made-up dataset).
//! - `bench_model`: times whole training steps at fixed row counts, to separate per-row cost from fixed cost.
//! - `bench_parts`: times the building blocks of one recurrent row (attention, the expert pool, ...).
//!
//! Run, for example: `cargo run --release -p minagi-cli --bin minagi_train -- --synthetic --preset tiny --backend metal --minutes 1`.

fn main() {
    println!(
        "minagi-cli: the tools are separate binaries:\n\
         \n  minagi_train  train a model from the command line (--help is this list: --data DIR | --synthetic, --preset, --backend, --minutes, --steps)\
         \n  bench_model   whole-step timing at 1, 2, 4, 6 and 12 rows\
         \n  bench_parts   timing of attention, the expert pool and the other parts of a row\
         \n\nrun e.g.: cargo run --release -p minagi-cli --bin minagi_train -- --synthetic --preset tiny --backend metal --minutes 1"
    );
}
