-- LLM Trainer schema v1. Metadata only: weights, experts and checkpoints live as files owned by the engine.
-- Times are unix milliseconds. Paths under the data directory are stored relative so the folder is relocatable.

CREATE TABLE app_settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

-- ───────── datasets ─────────
CREATE TABLE datasets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('starter','linked','generated','bundled')),
  starter_id TEXT,
  status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','scanning','preparing','ready','error')),
  root_rel TEXT,            -- datasets/<id>-<slug>/ holds train/<lane>/... and val/<domain>/...
  root_abs TEXT,            -- a user folder already shaped train/ + val/ (used in place, read-only)
  split_mode TEXT NOT NULL DEFAULT 'auto' CHECK (split_mode IN ('auto','folder')),
  split_pct REAL NOT NULL DEFAULT 2.0,
  split_seed INTEGER NOT NULL DEFAULT 1337,
  train_bytes INTEGER NOT NULL DEFAULT 0,
  val_bytes INTEGER NOT NULL DEFAULT 0,
  skipped_json TEXT NOT NULL DEFAULT '{}',
  license TEXT,
  attribution TEXT,
  source_url TEXT,
  error TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  scanned_at INTEGER
) STRICT;

CREATE TABLE dataset_sources (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('folder','file')),
  lane_hint TEXT,
  UNIQUE (dataset_id, path)
) STRICT;

CREATE TABLE lanes (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  display_name TEXT NOT NULL,
  color_slot INTEGER NOT NULL,          -- 0..7, never reassigned
  enabled INTEGER NOT NULL DEFAULT 1,
  n_files INTEGER NOT NULL DEFAULT 0,
  train_bytes INTEGER NOT NULL DEFAULT 0,
  val_files INTEGER NOT NULL DEFAULT 0,
  val_bytes INTEGER NOT NULL DEFAULT 0,
  sample_prompt TEXT,
  UNIQUE (dataset_id, name)
) STRICT;

CREATE TABLE dataset_files (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  lane_id INTEGER REFERENCES lanes(id) ON DELETE SET NULL,
  path TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  chars INTEGER,
  split TEXT NOT NULL DEFAULT 'train' CHECK (split IN ('train','val','excluded')),
  UNIQUE (dataset_id, path)
) STRICT;
CREATE INDEX idx_files_lane ON dataset_files (dataset_id, lane_id, split);

-- ───────── jobs: download | scan | prepare | import_python | export_model | compact_metrics | delete_run ─────────
CREATE TABLE jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('queued','running','paused','done','failed','cancelled')),
  subject TEXT,
  progress REAL,
  message TEXT,
  detail_json TEXT,
  error TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  finished_at INTEGER
) STRICT, WITHOUT ROWID;

-- ───────── runs ─────────
CREATE TABLE run_configs (
  id INTEGER PRIMARY KEY,
  preset TEXT NOT NULL,
  config_json TEXT NOT NULL,            -- canonical {model, train}
  config_hash TEXT NOT NULL UNIQUE,
  schema_version INTEGER NOT NULL,
  created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE runs (
  id INTEGER PRIMARY KEY,
  uid TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  origin TEXT NOT NULL DEFAULT 'trained' CHECK (origin IN ('trained','imported','forked')),
  dataset_id INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
  dataset_snapshot_json TEXT NOT NULL DEFAULT '{}',
  parent_run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
  parent_ckpt_id INTEGER,
  preset TEXT NOT NULL,
  config_id INTEGER NOT NULL REFERENCES run_configs(id),
  dir_rel TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN
    ('created','preparing','running','paused','stopping','completed','stopped','failed','interrupted','imported')),
  stage TEXT,
  backend TEXT,
  device_name TEXT,
  engine_version TEXT NOT NULL,
  app_version TEXT NOT NULL,
  goal_json TEXT,
  step INTEGER NOT NULL DEFAULT 0,
  chars_read INTEGER NOT NULL DEFAULT 0,
  chars_total INTEGER,
  active_ms INTEGER NOT NULL DEFAULT 0,
  best_heldout_nats REAL,
  last_heldout_nats REAL,
  last_train_nats REAL,
  n_experts INTEGER,
  verdict TEXT,
  error TEXT,
  created_at INTEGER NOT NULL,
  started_at INTEGER,
  ended_at INTEGER,
  updated_at INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_runs_created ON runs (created_at DESC);

-- ───────── time series ─────────
CREATE TABLE metric_keys (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  unit TEXT
) STRICT;

INSERT INTO metric_keys (id, name, unit) VALUES
  (1,  'train.nats',          'nats/char'),
  (2,  'grad.norm',           NULL),
  (3,  'lr.scale',            'x'),
  (4,  'lr.effective',        NULL),
  (5,  'speed.read_cps',      'chars/s'),
  (6,  'speed.write_cps',     'chars/s'),
  (7,  'ctx.now',             'chars'),
  (8,  'pool.n_experts',      NULL),
  (9,  'halt.avg_rows',       'rows'),
  (10, 'text.rep8_pct',       '%'),
  (11, 'mem.resident_gb',     'GB'),
  (12, 'mem.ram_gb',          'GB'),
  (13, 'mem.disk_gb',         'GB'),
  (14, 'gap.nats',            'nats/char'),
  (15, 'plasticity.evidence', NULL),
  (16, 'pool.dying_pct',      '%'),
  (17, 'ctx.gain',            NULL);

-- Universal x-axis registry: one row per logged moment.
CREATE TABLE ticks (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  chars INTEGER NOT NULL,
  t_ms INTEGER NOT NULL,                -- active training ms (pauses excluded)
  wall_ts INTEGER NOT NULL,
  PRIMARY KEY (run_id, step)
) STRICT, WITHOUT ROWID;
CREATE INDEX idx_ticks_chars ON ticks (run_id, chars);

-- Narrow, clustered by (run, key, step): one series is one contiguous range scan.
-- Deliberately no composite FK to ticks: a cascade would scan the run per row (O(n^2)).
CREATE TABLE metrics (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  key_id INTEGER NOT NULL,
  step INTEGER NOT NULL,
  value REAL,                           -- NULL = non-finite
  PRIMARY KEY (run_id, key_id, step)
) STRICT, WITHOUT ROWID;

-- ───────── evaluation (held-out) ─────────
CREATE TABLE eval_rounds (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  chars INTEGER NOT NULL,
  t_ms INTEGER NOT NULL,
  wall_ts INTEGER NOT NULL,
  overall_nats REAL,
  overall_se REAL,
  train_nats_ema REAL,
  UNIQUE (run_id, step)
) STRICT;

CREATE TABLE eval_domain (
  round_id INTEGER NOT NULL REFERENCES eval_rounds(id) ON DELETE CASCADE,
  domain TEXT NOT NULL,
  nats REAL,
  se REAL,
  n_chars INTEGER,
  PRIMARY KEY (round_id, domain)
) STRICT, WITHOUT ROWID;

-- ───────── samples: prompts x (raw, adapted) per round ─────────
CREATE TABLE samples (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  chars INTEGER NOT NULL DEFAULT 0,
  domain TEXT NOT NULL,
  prompt TEXT NOT NULL,
  raw TEXT NOT NULL,
  adapted TEXT NOT NULL,
  raw_rep8_pct REAL,
  adapted_rep8_pct REAL,
  note TEXT,
  PRIMARY KEY (run_id, step, idx)
) STRICT, WITHOUT ROWID;

-- ───────── timeline and experts ─────────
CREATE TABLE run_events (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  chars INTEGER NOT NULL,
  wall_ts INTEGER NOT NULL,
  kind TEXT NOT NULL,                   -- open-ended: stage, expert_born, expert_pruned, growth_blocked, ...
  expert_uid INTEGER,
  brake TEXT,
  payload_json TEXT
) STRICT;
CREATE INDEX idx_events_run ON run_events (run_id, step);

CREATE TABLE pool_snapshots (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  chars INTEGER NOT NULL,
  n_experts INTEGER NOT NULL,
  resident_json TEXT NOT NULL,
  usage_b64 TEXT NOT NULL,
  brakes_json TEXT NOT NULL,
  halting_hist_json TEXT NOT NULL,
  PRIMARY KEY (run_id, step)
) STRICT, WITHOUT ROWID;

-- ───────── checkpoints ─────────
CREATE TABLE checkpoints (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  chars INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('auto','manual','best','final','stop','imported')),
  path_rel TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  heldout_nats REAL,
  n_experts INTEGER,
  is_best INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0,
  engine_format INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'ok' CHECK (state IN ('ok','missing','corrupt','deleting')),
  created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_ckpt_run ON checkpoints (run_id, step DESC);
CREATE UNIQUE INDEX idx_ckpt_best ON checkpoints (run_id) WHERE is_best = 1;

-- ───────── chat ─────────
CREATE TABLE chat_sessions (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
  checkpoint_id INTEGER REFERENCES checkpoints(id) ON DELETE SET NULL,
  model_label TEXT NOT NULL,
  mode TEXT NOT NULL DEFAULT 'continue' CHECK (mode IN ('continue','conversation')),
  params_json TEXT NOT NULL,
  learn_enabled INTEGER NOT NULL DEFAULT 0,
  adapted_ckpt_rel TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
) STRICT;

CREATE TABLE chat_messages (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user','model')),
  content TEXT NOT NULL,
  learned INTEGER NOT NULL DEFAULT 0,
  learn_nats_before REAL,
  learn_nats_after REAL,
  rows_blob BLOB,
  experts_json TEXT,
  stats_json TEXT,
  created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX idx_chat_msgs ON chat_messages (session_id, id);

-- ───────── import / export / calibration ─────────
CREATE TABLE exports (
  id INTEGER PRIMARY KEY,
  run_id INTEGER,
  checkpoint_id INTEGER,
  format TEXT NOT NULL CHECK (format IN ('safetensors','portable')),
  dest_path TEXT NOT NULL,
  bytes INTEGER,
  sha256 TEXT,
  created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE imports (
  id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL,
  kind TEXT NOT NULL,
  created_run_id INTEGER,
  report_json TEXT,
  created_at INTEGER NOT NULL
) STRICT;

-- Measured speeds improve future estimates on this machine.
CREATE TABLE bench_results (
  device_key TEXT NOT NULL,
  preset TEXT NOT NULL,
  read_cps REAL NOT NULL,
  measured_at INTEGER NOT NULL,
  PRIMARY KEY (device_key, preset)
) STRICT, WITHOUT ROWID;
