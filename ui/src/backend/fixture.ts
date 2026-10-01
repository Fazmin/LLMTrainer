/**
 * In-browser stand-in for the desktop backend, used by `pnpm dev:web`, the component tests and the screenshot script.
 * It simulates a run (learning curve, held-out evaluations, samples that improve, a growing expert pool) behind the
 * same `Backend` interface, so every screen can be built and checked without launching the app.
 */
import type {
  ArithmeticParams,
  DatasetDetail,
  DatasetSummary,
  JobEvent,
  JobProgress,
  LaneInfo,
  LaneMode,
  PathProbe,
  SplitConfig,
  StarterInfo,
  TextPreview,
  ChatEvent,
  ChatMessage,
  ChatParams,
  ChatSessionInfo,
  CheckpointInfo,
  CreateRunRequest,
  EvalResult,
  Goal,
  Insight,
  LiveMsg,
  LiveSnapshot,
  ModelConfig,
  PoolSnapshot,
  Preset,
  PresetConfig,
  RunEvent,
  RunState,
  RunSummary,
  SampleRound,
  SeriesData,
  SeriesRequest,
  Stage,
  StageInfo,
  TrainConfig,
} from "../bindings";
import type { Backend } from "./types";

const LN2 = Math.LN2;
/** Simulated seconds per real second. */
const SIM = 24;
const NOW = () => Date.now();

// ── configuration (mirrors the Rust presets closely enough for the UI) ─────────────────────────────────────────────
function modelFor(p: "tiny" | "small" | "full"): ModelConfig {
  const t = { tiny: [256, 4, 704, 1, 6, 0.29, 6, 3.2, 1024, 16, 64, 256, 2], small: [384, 6, 1024, 2, 12, 0.2, 12, 4.0, 2048, 32, 256, 768, 4], full: [512, 8, 1408, 2, 24, 0.072, 24, 12.8, 4096, 64, 1024, 2048, 8] }[p];
  return {
    dModel: t[0]!, nHead: t[1]!, dFf: t[2]!, nPrelude: t[3]!, nRecur: 1, nCoda: 0, maxSteps: t[4]!, minSteps: 1,
    haltPrior: t[5]!, haltThresh: 0.9, haltFreeze: true, ponderBeta: 0.01, bpttWindow: t[6]!, trainStepsMean: t[7]!,
    block: t[8]!, vocabSize: 265, poolExperts: t[9]!, poolMax: t[10]!, poolDFf: t[11]!, poolDepth: 1, poolTopK: t[12]!,
  };
}

function trainFor(p: "tiny" | "small" | "full"): TrainConfig {
  const big = p === "full";
  const tiny = p === "tiny";
  return {
    lr: tiny ? 2e-3 : p === "small" ? 1e-3 : 3e-4, trunkLrMult: tiny ? 0.5 : p === "small" ? 0.25 : 0.1, weightDecay: 0.1,
    beta1: 0.9, beta2: 0.95, clip: 1, chunk: big ? 2048 : 512, passage: tiny ? 8192 : big ? 32768 : 16384, shuffleSeed: 0,
    poolAux: 0.01, resident: tiny ? 8 : p === "small" ? 16 : 32, ramCache: tiny ? 16 : p === "small" ? 32 : 96, capacityFactor: 1.5,
    exploreBias: 0.65, exploreSteps: 1000, contextStart: tiny ? 512 : p === "small" ? 1024 : 2048,
    contextEnd: tiny ? 1024 : p === "small" ? 2048 : 4096, contextStep: 1, contextGrowEveryChars: 100000, contextGainMin: 0.015,
    contextEveryChars: 65536,
    growth: { everyChars: 2000000, k: 1, maxGap: 0.4, dyingFracMax: 0.35, maxInFlight: 50, keepRatioMin: 0.35, recentMult: 4, maxDiskGb: 10, memFrac: 0.95, birthGate: 0.001 },
    prune: { survivalChars: 100000000, dyingAt: 0.65 },
    decoding: { adaptStrength: 2.5, adaptDecay: 0.88, repPenalty: 1 },
    sampleEveryMin: tiny ? 1 : p === "small" ? 5 : 10, saveEveryMin: 5, evalChars: tiny ? 30720 : 122880, precision: "f32",
    moeMode: tiny ? "dense_masked" : "sparse_dispatch", rowCheckpointing: big,
  };
}

const presetCfgs = (): PresetConfig[] =>
  (["tiny", "small", "full"] as const).map((p) => ({ preset: p, model: modelFor(p), train: trainFor(p) }));

// ── deterministic noise ──────────────────────────────────────────────────────────────────────────────────────────────
function rng(seed: number) {
  let s = (seed ^ 0x9e3779b9) >>> 0;
  const f = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(Math.max(f(), 1e-12))) * Math.cos(2 * Math.PI * f());
  return { f, normal, int: (a: number, b: number) => a + Math.floor(f() * (b - a)) };
}

// ── the learning curve and text ──────────────────────────────────────────────────────────────────────────────────────
const START = 5.58;
const FLOOR = 0.9;
const heldout = (chars: number, off: number) =>
  FLOOR + (START - FLOOR) / Math.pow(1 + chars / 2e5, 0.45) + off * Math.max(0.05, 1 - Math.exp(-chars / 4e6));
const trainLoss = (chars: number, off: number) => Math.max(0.05, heldout(chars, off) - (0.04 + 0.08 * Math.min(1, chars / 2e7)));

const STORY = "Once upon a time, there was a little boy named Tom. One day he found a shiny red ball in the garden. He picked it up and ran to show his mom. She smiled and said, \"What a lovely ball!\"";
const MATH = "add 4917 + 388 = <think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305";
const PROMPTS: Record<string, string> = {
  stories: "Once upon a time, there was a little boy named Tom. One day he ",
  arithmetic: "add 4917 + 388 = ",
};

function writeText(r: ReturnType<typeof rng>, domain: string, nats: number, adapted: boolean, n: number): string {
  const canon = domain === "arithmetic" ? MATH : STORY;
  let p = Math.min(1, Math.max(0, (nats - 1.1) / 4.2));
  if (adapted) p *= 0.9;
  const alphabet = "abcdefghijklmnopqrstuvwxyz      eeetaoinsh";
  let out = "";
  let i = r.int(0, canon.length);
  for (let k = 0; k < n; k++) {
    out += r.f() < p ? alphabet[r.int(0, alphabet.length)]! : canon[i % canon.length]!;
    i++;
    if (!adapted && p > 0.25 && r.f() < 0.02) i = Math.max(0, i - r.int(3, 12));
  }
  return out;
}

const rep8 = (s: string) => {
  if (s.length < 9) return 0;
  const seen = new Set<string>();
  let rep = 0;
  for (let i = 0; i + 8 <= s.length; i++) {
    const w = s.slice(i, i + 8);
    if (seen.has(w)) rep++;
    else seen.add(w);
  }
  return (100 * rep) / (s.length - 7);
};

// ── one simulated run ────────────────────────────────────────────────────────────────────────────────────────────────
interface Tick { step: number; chars: number; tMs: number; train: number; grad: number; lr: number; cps: number; ctx: number; experts: number; rows: number }

const DOMAINS = ["stories", "arithmetic"];
const OFFSETS = [-0.1, 0.45];

class RunSim {
  step = 0; chars = 0; simMs = 0; experts: number; ctx: number; lrScale = 1;
  ticks: Tick[] = []; evals: EvalResult[] = []; samples: SampleRound[] = []; pools: PoolSnapshot[] = [];
  events: RunEvent[] = []; ckpts: CheckpointInfo[] = []; trainEma = START;
  private nextTick = 0; private nextEval = 0; private nextCkpt = 0; private nextGrow = 0;
  private r: ReturnType<typeof rng>;
  readonly cps: number; readonly chunk: number;

  constructor(readonly id: number, readonly model: ModelConfig, readonly train: TrainConfig) {
    this.r = rng(id * 7919);
    this.experts = model.poolExperts; this.ctx = train.contextStart; this.chunk = train.chunk;
    this.cps = (train.chunk >= 2048 ? 900 : 9000) * (0.9 + 0.2 * this.r.f());
    this.nextTick = SIM * 1000; this.nextEval = SIM * 1000 * 4; this.nextCkpt = SIM * 1000 * 12; this.nextGrow = 1e6;
  }

  rows() { return 1.2 + (this.model.maxSteps * 0.45 - 1.2) * (1 - Math.exp(-this.chars / 2e6)); }

  /** Advance by `dtMs` of simulated time. Calls `emit` for things worth announcing live. */
  advance(dtMs: number, emit: (m: LiveMsg) => void, stageHook?: (s: Stage) => void) {
    for (let t = 0; t < dtMs; t += 1000) {
      this.simMs += 1000;
      const d = Math.round(this.cps / this.chunk);
      this.step += d; this.chars += d * this.chunk;
      this.trainEma += 0.2 * (trainLoss(this.chars, OFFSETS[0]!) - this.trainEma);
      if (this.chars > 5e5 && this.ctx < this.train.contextEnd && this.step % 3 === 0) this.ctx = Math.min(this.train.contextEnd, this.ctx + 8);
      if (this.simMs >= this.nextTick) {
        this.nextTick += SIM * 1000;
        const tick: Tick = {
          step: this.step, chars: this.chars, tMs: this.simMs,
          train: trainLoss(this.chars, OFFSETS[0]!) + 0.04 * this.r.normal() * Math.exp(-this.chars / 4e6),
          grad: 0.9 + 0.4 * this.r.f() + 3 * Math.exp(-this.chars / 4e5), lr: this.lrScale, cps: this.cps * (1 + 0.03 * this.r.normal()),
          ctx: this.ctx, experts: this.experts, rows: this.rows(),
        };
        this.ticks.push(tick);
        emit({ type: "ticks", runId: this.id, points: [{ step: tick.step, chars: tick.chars, tMs: tick.tMs, trainNats: tick.train, gradNorm: tick.grad, lrScale: tick.lr, lrEffective: this.train.lr * tick.lr, readCps: tick.cps, writeCps: null, contextNow: tick.ctx, nExperts: tick.experts, avgRows: tick.rows, rep8Pct: null }] });
      }
      if (this.simMs >= this.nextEval) {
        this.nextEval += SIM * 1000 * 4;
        stageHook?.("evaluating");
        const ev = this.evaluate();
        emit({ type: "eval", runId: this.id, result: ev });
        const samples = this.sample();
        stageHook?.("sampling");
        emit({ type: "samples", runId: this.id, round: samples });
        const pool = this.poolSnapshot();
        emit({ type: "pool", runId: this.id, snapshot: pool });
        const old = this.lrScale;
        this.lrScale = Math.min(1, Math.max(0.05, this.lrScale * (this.r.f() < 0.12 ? 2 : 0.93)));
        if (this.lrScale > old * 1.5) this.event("plasticity_jump");
        emit({ type: "insight", runId: this.id, insight: this.insight() });
        stageHook?.("reading");
      }
      if (this.chars >= this.nextGrow) {
        this.nextGrow += 1.2e6;
        if (this.experts < this.model.poolMax) { this.experts++; this.event("expert_born"); }
      }
      if (this.simMs >= this.nextCkpt) {
        this.nextCkpt += SIM * 1000 * 12;
        stageHook?.("checkpointing");
        const c: CheckpointInfo = { id: this.ckpts.length + 1, step: this.step, chars: this.chars, kind: "auto", path: `checkpoints/step-${String(this.step).padStart(9, "0")}`, bytes: 18_400_000 + this.experts * 3_100_000, heldoutNats: this.evals.at(-1)?.overallNats ?? null, nExperts: this.experts, isBest: false, pinned: false, createdAt: NOW() };
        this.ckpts.unshift(c);
        this.event("checkpoint");
        emit({ type: "checkpoint", runId: this.id, checkpoint: { step: c.step, chars: c.chars, kind: "auto", path: c.path, bytes: c.bytes, heldoutNats: c.heldoutNats, nExperts: this.experts, engineFormat: 0 } });
        stageHook?.("reading");
      }
    }
  }

  event(kind: string) {
    this.events.push({ id: this.events.length + 1, step: this.step, chars: this.chars, at: NOW(), kind, expertUid: null, brake: null, payload: null });
  }

  evaluate(): EvalResult {
    const domains = DOMAINS.map((d, i) => ({ domain: d, nats: heldout(this.chars, OFFSETS[i]!) + 0.012 * this.r.normal(), se: 0.01 + 0.03 / (1 + this.chars / 1e6), nChars: 30720 }));
    const overall = domains.reduce((a, d) => a + d.nats, 0) / domains.length;
    const ev: EvalResult = { step: this.step, chars: this.chars, overallNats: overall, overallSe: 0.012, trainNatsEma: this.trainEma, domains };
    this.evals.push(ev);
    return ev;
  }

  sample(): SampleRound {
    const items = DOMAINS.map((d, i) => {
      const nats = heldout(this.chars, OFFSETS[i]!);
      const raw = writeText(this.r, d, nats, false, 140);
      const adapted = writeText(this.r, d, nats, true, 140);
      return { domain: d, prompt: PROMPTS[d] ?? "", raw, adapted, rawRep8: rep8(raw), adaptedRep8: rep8(adapted), note: null };
    });
    const round = { step: this.step, chars: this.chars, items };
    this.samples.push(round);
    return round;
  }

  poolSnapshot(): PoolSnapshot {
    const n = this.experts;
    let shares = Array.from({ length: n }, (_, i) => (1 / (1 + i * 0.35)) * (0.7 + 0.6 * this.r.f()));
    const sum = shares.reduce((a, b) => a + b, 0);
    shares = shares.map((s) => s / sum).sort((a, b) => b - a);
    const max = shares[0]!;
    const resident = Math.min(this.train.resident, n);
    const b = (ok: boolean, good: string, bad: string) => ({ ok, why: ok ? good : bad });
    const snap: PoolSnapshot = {
      step: this.step, chars: this.chars, nExperts: n,
      resident: Array.from({ length: resident }, (_, i) => ({ uid: i + 1, slot: i, gate: 0.8 + 0.4 * this.r.f(), useShare: shares[i]!, admits: this.r.int(5, 900), ageChars: Math.max(0, this.chars - this.r.int(0, 1e6)), onTrial: i + 3 >= resident, staleness: i + 3 >= resident ? 0.4 + 0.5 * this.r.f() : 0.05 * this.r.f(), dying: i + 1 === resident && n > 20 })),
      usage: shares.map((s) => Math.floor((s / max) * 255)),
      brakes: {
        room: b(n < this.model.poolMax, "There is room for another expert.", "The expert pool is at its size limit."),
        used: b(true, "Most experts are being used.", "Too many experts sit idle."),
        earning: b(this.r.f() > 0.35, "Recent newcomers are earning their keep.", "3 of 8 newer experts are not yet useful."),
        fits: b(true, "Few newcomers are still on trial.", "Too many newcomers are still on trial."),
        honest: b(true, "It does about as well on new text as on training text.", "It does much better on training text than on new text."),
      },
      haltingHist: (() => { const c = Math.max(1, this.rows()); const h = Array.from({ length: this.model.maxSteps }, (_, i) => Math.exp(-0.5 * ((i + 1 - c) / (c * 0.5 + 0.8)) ** 2)); const s = h.reduce((a, v) => a + v, 0); return h.map((v) => v / s); })(),
    };
    this.pools.push(snap);
    return snap;
  }

  insight(): Insight {
    const pts = this.evals.map((e) => e.overallNats);
    const n = pts.length;
    const recent = pts.slice(-5);
    const change = recent.length >= 2 ? (recent[0]! - recent[recent.length - 1]!) / recent[0]! : 0;
    const verdict = n < 3 ? "warming_up" : change > 0.01 ? "learning" : "plateau";
    const bits = pts.length ? pts[pts.length - 1]! / LN2 : null;
    const idx = bits == null ? 0 : bits <= 1.3 ? 5 : bits <= 1.8 ? 4 : bits <= 2.5 ? 3 : bits <= 3.5 ? 2 : bits <= 5 ? 1 : 0;
    const keys = ["random_guessing", "letter_frequencies", "common_words", "short_phrases", "simple_sentences", "fluent_sentences"];
    const remainingChars = bits != null && bits > 2 ? (heldoutInverse(2 * LN2) - this.chars) : 0;
    return { report: { verdict, changePct: change * 100, advice: [verdict === "learning" ? "keep_going" : verdict === "plateau" ? "read_more_text" : "wait_for_first_check"] }, eta: remainingChars > 0 ? { seconds: remainingChars / this.cps, chars: remainingChars } : null, milestone: { index: idx, total: 6, key: keys[idx]! }, bitsPerChar: bits };
  }
}

/** Characters at which the (no-offset) held-out curve reaches `nats`. */
function heldoutInverse(nats: number): number {
  const x = (START - FLOOR) / Math.max(1e-6, nats - FLOOR);
  return 2e5 * (Math.pow(x, 1 / 0.45) - 1);
}

// ── the fake backend ─────────────────────────────────────────────────────────────────────────────────────────────────
const subscribers = new Set<(m: LiveMsg) => void>();
const broadcast = (m: LiveMsg) => subscribers.forEach((s) => s(m));

const runs = new Map<number, RunSummary>();
const sims = new Map<number, RunSim>();
let nextId = 1;
let active: { id: number; paused: boolean; timer: ReturnType<typeof setInterval>; goal: Goal | null } | null = null;
const snapshot: LiveSnapshot = { run: null, stage: null, lastEval: null, lastPool: null, insight: null, schedule: null };

function summary(id: number): RunSummary {
  const r = runs.get(id);
  if (!r) throw { kind: "not_found", detail: `run ${id}` };
  const sim = sims.get(id);
  if (sim) {
    r.step = sim.step; r.charsRead = sim.chars; r.activeMs = sim.simMs; r.nExperts = sim.experts;
    r.lastTrainNats = sim.trainEma; r.lastHeldoutNats = sim.evals.at(-1)?.overallNats ?? null;
    r.bestHeldoutNats = sim.evals.length ? Math.min(...sim.evals.map((e) => e.overallNats)) : null;
    r.verdict = sim.evals.length ? sim.insight().report.verdict : null;
  }
  return { ...r };
}

function setState(id: number, state: RunState) {
  const r = runs.get(id)!;
  r.status = state;
  if (state === "running" && !r.startedAt) r.startedAt = NOW();
  if (["completed", "stopped", "failed", "interrupted"].includes(state)) r.endedAt = NOW();
  snapshot.run = summary(id);
  broadcast({ type: "state", runId: id, state, reason: null });
}

function stage(id: number, s: Stage) {
  const info: StageInfo = { stage: s, detail: null, progress: null };
  snapshot.stage = info;
  broadcast({ type: "stage", runId: id, info, at: NOW() });
}

function createRunRecord(req: CreateRunRequest, offlineMinutes = 0): RunSummary {
  const id = nextId++;
  const p = (req.preset === "tiny" || req.preset === "small" || req.preset === "full" ? req.preset : "small") as "tiny" | "small" | "full";
  const model = req.model ?? modelFor(p);
  const train = req.train ?? trainFor(p);
  const label = p[0]!.toUpperCase() + p.slice(1);
  const ds = req.datasetId != null ? datasets.get(req.datasetId) : undefined;
  if (req.datasetId != null && !ds) throw { kind: "not_found", detail: `dataset ${req.datasetId}` };
  if (ds && ds.summary.status !== "ready") throw { kind: "dataset_not_ready" };
  const r: RunSummary = {
    id, uid: `fixture-${id}`, name: req.name?.trim() || `${label} run ${id}`, notes: "", status: "created", preset: req.preset,
    datasetId: req.datasetId, datasetName: ds ? ds.summary.name : "Stories and arithmetic (sample)", step: 0, charsRead: 0, charsTotal: ds ? ds.lanes.filter((l) => l.enabled).reduce((a, l) => a + l.trainBytes, 0) : null, activeMs: 0,
    bestHeldoutNats: null, lastHeldoutNats: null, lastTrainNats: null, nExperts: null, verdict: null, stage: null, backend: "Simulated GPU",
    goal: req.goal, error: null, createdAt: NOW() - offlineMinutes * 60000, startedAt: null, endedAt: null,
  };
  runs.set(id, r);
  sims.set(id, new RunSim(id, model, train));
  if (offlineMinutes > 0) {
    const sim = sims.get(id)!;
    sim.advance(offlineMinutes * 60 * 1000, () => {});
    r.status = "completed"; r.startedAt = r.createdAt; r.endedAt = r.createdAt + offlineMinutes * 30000;
  }
  return r;
}

// A fresh install (`?fresh=1`) starts empty, like a first launch. Otherwise there is one dataset and two finished runs.
const FRESH = typeof location !== "undefined" && new URLSearchParams(location.search).has("fresh");
function seedFixture() {
  if (FRESH) return;
  addDataset("Offline sampler", "bundled", "sampler", [laneOf("stories", 0, 31_000_000, PREVIEWS.stories!), laneOf("arithmetic", 1, 9_000_000, PREVIEWS.arithmetic!)], { license: "MIT", attribution: "Written for this app" });
  createRunRecord({ name: "Tiny stories, first try", preset: "tiny", model: null, train: null, datasetId: null, goal: { type: "minutes", value: 10 } }, 12);
  createRunRecord({ name: "Small, higher learning rate", preset: "custom", model: modelFor("small"), train: { ...trainFor("small"), lr: 3e-3, chunk: 512 }, datasetId: null, goal: { type: "minutes", value: 30 } }, 9);
}

function downsample(xs: number[], ys: (number | null)[], max: number): SeriesData["x"] extends number[] ? { x: number[]; y: (number | null)[]; lo: (number | null)[]; hi: (number | null)[]; bucket: number } : never {
  if (xs.length <= max) return { x: xs, y: ys, lo: ys, hi: ys, bucket: 1 } as never;
  const per = Math.ceil(xs.length / max);
  const out = { x: [] as number[], y: [] as (number | null)[], lo: [] as (number | null)[], hi: [] as (number | null)[], bucket: per };
  for (let i = 0; i < xs.length; i += per) {
    const vs = ys.slice(i, i + per).filter((v): v is number => v != null);
    out.x.push(xs[Math.min(xs.length - 1, i + per - 1)]!);
    out.y.push(vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null);
    out.lo.push(vs.length ? Math.min(...vs) : null);
    out.hi.push(vs.length ? Math.max(...vs) : null);
  }
  return out as never;
}

const xOf = (t: { step: number; chars: number; tMs: number }, ax: SeriesRequest["x"]) => (ax === "chars" ? t.chars : ax === "step" ? t.step : t.tMs / 1000);

// ── text the model reads ─────────────────────────────────────────────────────────────────────────────────────────────
const STARTERS: StarterInfo[] = [
  { id: "tinystories-quick", title: "Simple stories (TinyStories)", description: "About 80,000 short children's stories in plain English. A good first thing to learn from.", sizeBytes: 87_000_000, license: "CDLA-Sharing-1.0", attribution: "Eldan and Li, TinyStories (2023)", sourceUrl: "https://huggingface.co/datasets/roneneldan/TinyStories", offline: false },
  { id: "sampler", title: "Offline sampler", description: "A small bundle of stories and sums. Works without internet.", sizeBytes: 120_000, license: "MIT", attribution: "Written for this app", sourceUrl: "", offline: true },
  { id: "arithmetic", title: "Arithmetic practice", description: "Sums with worked steps, made on this computer.", sizeBytes: 36_000_000, license: "MIT", attribution: "Generated on this computer", sourceUrl: "", offline: true },
  { id: "tinystories-full", title: "All the stories (TinyStories, full)", description: "The complete collection of about 2 million stories. Large.", sizeBytes: 1_940_000_000, license: "CDLA-Sharing-1.0", attribution: "Eldan and Li, TinyStories (2023)", sourceUrl: "https://huggingface.co/datasets/roneneldan/TinyStories", offline: false },
];
const datasets = new Map<number, DatasetDetail>();
const splits = new Map<number, SplitConfig>();
let nextDatasetId = 1;
const jobCancel = new Map<string, boolean>();
let nextJob = 1;

const laneOf = (name: string, slot: number, bytes: number, prompt: string | null): LaneInfo => ({
  name, displayName: name, colorSlot: slot, enabled: true, nFiles: Math.max(1, Math.round(bytes / 40000)), trainBytes: bytes,
  valFiles: 1, valBytes: Math.round(bytes / 50), samplePrompt: prompt,
});

function addDataset(name: string, kind: DatasetSummary["kind"], starterId: string | null, lanes: LaneInfo[], extras: Partial<DatasetSummary> = {}): DatasetDetail {
  const id = nextDatasetId++;
  const detail: DatasetDetail = {
    summary: { id, name, kind, starterId, status: "ready", trainBytes: lanes.reduce((a, l) => a + l.trainBytes, 0), valBytes: lanes.reduce((a, l) => a + l.valBytes, 0), lanes: lanes.length, license: null, attribution: null, sourceUrl: null, error: null, createdAt: NOW(), ...extras },
    lanes, skipped: { binary: 0, empty: 0, utf16: 0, tooLarge: 0, unreadable: 0, examples: [] }, warnings: [],
  };
  datasets.set(id, detail);
  splits.set(id, { mode: "auto", pct: 2, seed: 42 });
  return detail;
}

/** Pretend to work for `ms`, streaming progress and honouring cancel. */
async function fakeJob(onEvent: (e: JobEvent) => void, unit: JobProgress["unit"], message: string, total: number, ms: number): Promise<void> {
  const jobId = `job-${nextJob++}`;
  jobCancel.set(jobId, false);
  onEvent({ type: "state", jobId, state: "running", error: null });
  const steps = Math.max(4, Math.round(ms / 150));
  for (let i = 1; i <= steps; i++) {
    await new Promise((r) => setTimeout(r, ms / steps));
    if (jobCancel.get(jobId)) {
      onEvent({ type: "state", jobId, state: "cancelled", error: null });
      jobCancel.delete(jobId);
      throw { kind: "cancelled" };
    }
    onEvent({ type: "progress", jobId, progress: { done: (total * i) / steps, total, unit, message, bytesPerSec: unit === "bytes" ? 9_500_000 : null, etaSeconds: unit === "bytes" ? ((steps - i) * ms) / steps / 1000 : null } });
  }
  onEvent({ type: "state", jobId, state: "done", error: null });
  jobCancel.delete(jobId);
}

const PREVIEWS: Record<string, string> = {
  stories: "Once upon a time, there was a little girl named Lily. She loved to play in the garden with her cat, Max. One sunny day, they found a shiny blue stone under the old oak tree.",
  arithmetic: "add 4917 + 388 = <think> 7+8+0=5c1 1+8+1=0c1 9+3+1=3c1 4+0+1=5c0 </think> 5305",
  code: "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])",
  notes: "Meeting notes: we agreed to move the launch to the second week of March. Priya will check the budget and send the numbers on Friday.",
};

const datasetOrThrow = (id: number) => {
  const d = datasets.get(id);
  if (!d) throw { kind: "not_found", detail: `dataset ${id}` };
  return d;
};

const refreshSummary = (d: DatasetDetail) => {
  d.summary.trainBytes = d.lanes.reduce((a, l) => a + l.trainBytes, 0);
  d.summary.valBytes = d.lanes.reduce((a, l) => a + l.valBytes, 0);
  d.summary.lanes = d.lanes.length;
};

// ── chat ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
interface FixtureChat { info: ChatSessionInfo; messages: ChatMessage[]; cancel: boolean; busy: boolean; quality: number }
const chats = new Map<number, FixtureChat>();
let nextChatId = 1;
let nextMessageId = 1;
const REPLIES = [
  "The little fox looked up at the moon and smiled. \"Tomorrow,\" he said, \"I will find my way home.\"",
  "I am a small language model. I learn by reading text one character at a time, and I am still learning.",
  "Once upon a time, a kind old woman planted a tiny seed. Every day she watered it, and slowly it grew.",
];
const chatOf = (id: number) => {
  const c = chats.get(id);
  if (!c) throw { kind: "invalid", detail: "This chat is not open. Open it again to continue." };
  return c;
};

export const fixtureBackend: Backend = {
  kind: "fixture",
  appInfo: async () => ({ version: "0.1.0", engineName: "fixture", engineVersion: "fixture-0.1.0", isMock: true, dataDir: "(in-browser preview)", schemaVersion: 1 }),
  hardwareInfo: async () => ({ os: "Preview", cpu: "Browser preview", cores: 8, ramGb: 24, ramFreeGb: 14, diskFreeGb: 300, backends: [{ kind: "metal", name: "Simulated GPU", available: true, reason: null, memBudgetGb: 14, bf16Ok: true }, { kind: "cpu", name: "CPU", available: true, reason: null, memBudgetGb: 12, bf16Ok: false }], selected: "metal" }),
  presetConfigs: async () => presetCfgs(),
  recommendPreset: async (datasetId) => {
    const fits = (["tiny", "small", "full"] as const).map((p) => ({ preset: p as Preset, estimate: estimate(p) }));
    const mb = datasetId != null ? (datasets.get(datasetId)?.summary.trainBytes ?? 0) / 1e6 : null;
    const small = mb != null && mb < 5;
    return { preset: "tiny", reasons: [small ? "Your text is small (under 5 MB), so a bigger model would just memorise it." : "Tiny gives your first readable results in about ten minutes."], fits };
  },
  estimateRun: async (model, train, _datasetId) => estimateFor(model, train),

  createRun: async (req) => summary(createRunRecord(req).id),
  startRun: async (id) => {
    if (active) throw { kind: "run_active" };
    const sim = sims.get(id);
    if (!sim) throw { kind: "not_found", detail: `run ${id}` };
    const goal = runs.get(id)!.goal;
    setState(id, "preparing");
    snapshot.run = summary(id);
    stage(id, "preparing_data");
    let phase = 0;
    let lastPulse = 0;
    const timer = setInterval(() => {
      if (!active || active.id !== id || active.paused) return;
      phase += 100;
      if (phase === 800) stage(id, "creating_model");
      if (phase === 1600) { stage(id, "reading"); setState(id, "running"); }
      if (phase < 1700) return;
      sim.advance(100 * SIM, (m) => { if (m.type === "eval") snapshot.lastEval = m.result; if (m.type === "pool") snapshot.lastPool = m.snapshot; if (m.type === "insight") snapshot.insight = m.insight; broadcast(m); }, (s) => stage(id, s));
      if (phase - lastPulse >= 250) {
        lastPulse = phase;
        broadcast({ type: "pulse", runId: id, step: sim.step, chars: sim.chars, charsTotal: null, trainNatsEma: sim.trainEma, readCps: sim.cps, activeMs: sim.simMs, schedule: { nextEvalChars: null, nextSampleMs: 30000, nextCheckpointMs: 120000, nextGrowthChars: null } });
        const reached = goal?.type === "minutes" ? sim.simMs >= goal.value * 60000 : goal?.type === "chars" ? sim.chars >= goal.value : false;
        if (reached) void fixtureBackend.stopRun(true);
      }
    }, 100);
    active = { id, paused: false, timer, goal };
    return summary(id);
  },
  pauseRun: async () => { if (active) { active.paused = true; stage(active.id, "paused"); setState(active.id, "paused"); } },
  resumeRun: async () => { if (active) { active.paused = false; stage(active.id, "reading"); setState(active.id, "running"); } },
  stopRun: async () => {
    if (!active) throw { kind: "no_active_run" };
    const { id, timer, goal } = active;
    clearInterval(timer);
    active = null;
    stage(id, "stopping");
    const sim = sims.get(id)!;
    const reached = goal?.type === "minutes" ? sim.simMs >= goal.value * 60000 : goal?.type === "chars" ? sim.chars >= goal.value : false;
    setTimeout(() => { stage(id, "finished"); setState(id, reached ? "completed" : "stopped"); }, 300);
  },
  checkpointNow: async () => {},
  sampleNow: async () => {},
  evalNow: async () => {},
  listRuns: async () => [...runs.keys()].sort((a, b) => b - a).map(summary),
  getRun: async (id) => summary(id),
  getRunConfig: async (id) => {
    const sim = sims.get(id);
    if (!sim) throw { kind: "not_found", detail: `run ${id}` };
    return { preset: runs.get(id)!.preset, model: sim.model, train: sim.train };
  },
  getActiveRun: async () => (active ? summary(active.id) : null),
  renameRun: async (id, name, notes) => { const r = runs.get(id)!; r.name = name; r.notes = notes; return summary(id); },
  deleteRun: async (id) => { if (active?.id === id) throw { kind: "run_active" }; runs.delete(id); sims.delete(id); },

  subscribeLive: async (onMessage) => {
    subscribers.add(onMessage);
    return { ...snapshot, run: snapshot.run ?? (runs.size ? summary(Math.max(...runs.keys())) : null) };
  },

  getSeries: async (req) => {
    const sim = sims.get(req.runId);
    if (!sim) return [];
    const max = req.maxPoints || 1000;
    const lo = req.from ?? -Infinity, hi = req.to ?? Infinity;
    return req.keys.map((key): SeriesData => {
      if (key.startsWith("eval.")) {
        const rest = key.slice(5);
        const pts = sim.evals.map((e) => {
          const x = xOf({ step: e.step, chars: e.chars, tMs: e.step }, req.x);
          let v: number | null;
          if (rest === "overall") v = e.overallNats;
          else if (rest === "train") v = e.trainNatsEma;
          else if (rest === "gap") v = e.overallNats - e.trainNatsEma;
          else v = e.domains.find((d) => d.domain === rest.replace("domain.", ""))?.nats ?? null;
          return { x, v, se: e.overallSe };
        }).filter((p) => p.x >= lo && p.x <= hi);
        return { key, x: pts.map((p) => p.x), y: pts.map((p) => p.v), yLo: pts.map((p) => (p.v == null ? null : p.v - p.se)), yHi: pts.map((p) => (p.v == null ? null : p.v + p.se)), bucket: 1 };
      }
      const pick: Record<string, (t: Tick) => number | null> = {
        "train.nats": (t) => t.train, "grad.norm": (t) => t.grad, "lr.scale": (t) => t.lr, "speed.read_cps": (t) => t.cps,
        "ctx.now": (t) => t.ctx, "pool.n_experts": (t) => t.experts, "halt.avg_rows": (t) => t.rows,
      };
      const f = pick[key];
      if (!f) return { key, x: [], y: [], yLo: [], yHi: [], bucket: 1 };
      const ticks = sim.ticks.filter((t) => { const x = xOf(t, req.x); return x >= lo && x <= hi; });
      const d = downsample(ticks.map((t) => xOf(t, req.x)), ticks.map(f), max);
      return { key, x: d.x, y: d.y, yLo: d.lo, yHi: d.hi, bucket: d.bucket };
    });
  },
  getEvalPoints: async (id) => (sims.get(id)?.evals ?? []).map((e) => ({ chars: e.chars, nats: e.overallNats, se: e.overallSe, trainNats: e.trainNatsEma })),
  getEvalDomains: async (id) => (sims.get(id)?.evals.length ? [...DOMAINS].sort() : []),
  getEvents: async (id, kinds) => (sims.get(id)?.events ?? []).filter((e) => !kinds?.length || kinds.includes(e.kind)),
  getSamples: async (id, step) => { const s = sims.get(id)?.samples ?? []; return (step == null ? s.at(-1) : s.find((r) => r.step === step)) ?? null; },
  listSampleSteps: async (id) => (sims.get(id)?.samples ?? []).map((s) => ({ step: s.step, chars: s.chars })),
  getPoolSnapshot: async (id, step) => { const p = sims.get(id)?.pools ?? []; return (step == null ? p.at(-1) : [...p].reverse().find((s) => s.step <= step)) ?? null; },
  listPoolSteps: async (id) => (sims.get(id)?.pools ?? []).map((s) => ({ step: s.step, chars: s.chars })),
  listCheckpoints: async (id) => sims.get(id)?.ckpts ?? [],

  listDatasets: async () => [...datasets.values()].map((d) => ({ ...d.summary })).sort((a, b) => b.id - a.id),
  getDataset: async (id) => structuredClone(datasetOrThrow(id)),
  listStarters: async () => STARTERS.map((x) => ({ ...x })),
  probePaths: async (paths) => paths.map((p): PathProbe => ({ path: p, isDir: true, approxFiles: 214, exact: true, looksLikeMiniAgiLayout: false })),
  addFolders: async (paths: string[], _mode: LaneMode, onEvent) => {
    if (paths.length === 0) throw { kind: "invalid", detail: "Choose a folder or some files first." };
    await fakeJob(onEvent, "files", "Looking for text", 214, 1500);
    const name = paths.length === 1 ? (paths[0]!.split(/[\\/]/).filter(Boolean).pop() ?? "My text") : `${paths.length} folders`;
    const d = addDataset(name, "linked", null, [laneOf("notes", 8, 3_200_000, PREVIEWS.notes!)]);
    d.skipped.binary = 12;
    d.warnings = [];
    return structuredClone(d);
  },
  rebuildDataset: async (id, split, onEvent) => {
    const d = datasetOrThrow(id);
    if (split) splits.set(id, split);
    await fakeJob(onEvent, "files", "Rebuilding", 100, 900);
    const pct = splits.get(id)!.pct;
    d.lanes.forEach((l) => (l.valBytes = Math.round((l.trainBytes * pct) / 100)));
    refreshSummary(d);
    return structuredClone(d);
  },
  getSplit: async (id) => ({ ...splits.get(id)! }),
  updateLane: async (id, lane, patch) => {
    const d = datasetOrThrow(id);
    const l = d.lanes.find((x) => x.name === lane);
    if (!l) throw { kind: "not_found", detail: `lane ${lane}` };
    if (patch.enabled != null) l.enabled = patch.enabled;
    if (patch.displayName != null) l.displayName = patch.displayName;
    if (patch.samplePrompt != null) l.samplePrompt = patch.samplePrompt;
    return structuredClone(d);
  },
  previewText: async (id, lane, _n, seed): Promise<TextPreview> => {
    const d = datasetOrThrow(id);
    const name = lane ?? d.lanes.find((l) => l.enabled)?.name ?? "stories";
    const text = PREVIEWS[name] ?? PREVIEWS.notes!;
    return { file: `${name}/part-0000.txt`, offset: (seed % 7) * 4096, text };
  },
  installStarter: async (starterId, onEvent) => {
    const info = STARTERS.find((x) => x.id === starterId);
    if (!info) throw { kind: "not_found", detail: `starter ${starterId}` };
    const existing = [...datasets.values()].find((d) => d.summary.starterId === starterId);
    if (existing) return structuredClone(existing);
    await fakeJob(onEvent, "bytes", info.offline ? "Writing the text" : "Downloading the stories", info.sizeBytes, info.offline ? 900 : 4200);
    const lanes =
      starterId === "sampler"
        ? [laneOf("stories", 0, 33_000, PREVIEWS.stories!), laneOf("arithmetic", 1, 90_000, PREVIEWS.arithmetic!)]
        : starterId === "arithmetic"
          ? [laneOf("arithmetic", 1, info.sizeBytes, PREVIEWS.arithmetic!)]
          : [laneOf("stories", 0, info.sizeBytes, PREVIEWS.stories!)];
    return structuredClone(addDataset(info.title, info.id === "arithmetic" ? "generated" : info.offline ? "bundled" : "starter", starterId, lanes, { license: info.license, attribution: info.attribution, sourceUrl: info.sourceUrl || null }));
  },
  generateArithmetic: async (params: ArithmeticParams, onEvent) => {
    await fakeJob(onEvent, "problems", "Writing problems", params.trainProblems, 1500);
    return structuredClone(addDataset("Arithmetic practice", "generated", null, [laneOf("arithmetic", 1, params.trainProblems * 60, PREVIEWS.arithmetic!)], { attribution: "Generated on this computer" }));
  },
  deleteDataset: async (id) => {
    datasetOrThrow(id);
    datasets.delete(id);
    for (const r of runs.values()) if (r.datasetId === id) { r.datasetId = null; r.datasetName = null; }
  },
  cancelJob: async (jobId) => {
    if (!jobCancel.has(jobId)) return false;
    jobCancel.set(jobId, true);
    return true;
  },
  storageUsage: async () => ({
    dataDir: "/Users/you/Library/Application Support/LLM Trainer",
    datasetsBytes: [...datasets.values()].reduce((a, d) => a + d.summary.trainBytes + d.summary.valBytes, 0),
    runsBytes: [...sims.values()].reduce((a, sm) => a + sm.ckpts.reduce((x, c) => x + c.bytes, 0), 0),
    chatsBytes: 0,
    databaseBytes: 2_400_000,
    freeBytes: 312_000_000_000,
    perRun: [...runs.values()].map((r) => ({ runId: r.id, name: r.name, bytes: (sims.get(r.id)?.ckpts ?? []).reduce((a, c) => a + c.bytes, 0) })).sort((a, b) => b.bytes - a.bytes),
  }),
  revealInFolder: async () => {},
  openLink: async (url) => { window.open(url, "_blank", "noopener"); },
  exportModel: async (req, onEvent) => {
    const run = runs.get(req.runId);
    const sim = sims.get(req.runId);
    if (!run || !sim || sim.ckpts.length === 0) throw { kind: "invalid", detail: "This run has no saved progress to export yet." };
    await fakeJob(onEvent, "bytes", "Copying the saved model", 24_000_000, 1200);
    return { path: `${req.destDir}/${run.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-step-${sim.step}`, bytes: 24_000_000, files: 9 };
  },
  exportSafetensors: async (req, onEvent) => {
    const run = runs.get(req.runId);
    const sim = sims.get(req.runId);
    if (!run || !sim || sim.ckpts.length === 0) throw { kind: "invalid", detail: "This run has no saved progress to export yet." };
    await fakeJob(onEvent, "bytes", "Writing the model for other tools", 24_000_000, 1000);
    return { path: `${req.destDir}/${run.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-step-${sim.step}-safetensors`, bytes: 24_000_000, files: 3 };
  },
  forkRun: async (req) => {
    const src = runs.get(req.runId);
    const sim = sims.get(req.runId);
    if (!src || !sim || sim.ckpts.length === 0) throw { kind: "invalid", detail: "This run has no saved progress to continue from yet." };
    const r = createRunRecord({ name: req.name?.trim() || `${src.name} (continued)`, preset: src.preset, model: null, train: req.train, datasetId: req.datasetId ?? src.datasetId, goal: req.goal }, 0);
    r.step = src.step; r.charsRead = src.charsRead; r.lastHeldoutNats = src.lastHeldoutNats; r.bestHeldoutNats = src.bestHeldoutNats;
    return { ...r };
  },
  previewImport: async (folder) => {
    const original = /original|mini-?agi|weights/i.test(folder);
    if (!original && runs.size === 0) throw { kind: "invalid", detail: "That folder does not look like a saved model. Choose a folder made by Export, or the weights folder of the original mini-AGI program." };
    const name = folder.split("/").pop() ?? "Model";
    return {
      path: folder,
      kind: original ? "python" : "portable",
      importable: !/blocked/i.test(folder),
      summary: original ? "Python checkpoint at step 458000: 174 experts, 512-wide, 8 heads, 2 dense + 1 recurrent block(s), up to 24 rows, experts of 2048 units with 8 per character. Closest preset: Full." : `${name}: a Tiny model with 16 experts, after reading 3.0 million characters.`,
      issues: /blocked/i.test(folder)
        ? [{ severity: "blocker", message: "This model has more than one recurrent block per row; this app only runs models with one." }]
        : original
          ? [{ severity: "note", message: "Adam's optimiser history for the expert slots is not carried over; each expert keeps its own." }, { severity: "warning", message: "The model was saved with halting freeze off; it will run with it on, as the original trainer does." }]
          : [],
      name: original ? `Original model (${name})` : `${name} (imported)`,
      preset: original ? "full" : "tiny",
      model: null,
      step: original ? 458000 : 3000,
      chars: original ? 900_000_000 : 3_000_000,
      heldoutNats: original ? 1.19 : 2.1,
      nExperts: original ? 174 : 16,
      bytes: original ? 4_400_000_000 : 24_000_000,
    };
  },
  importModel: async (folder, onEvent) => {
    await fakeJob(onEvent, "bytes", "Copying the model in", 24_000_000, 1000);
    const src = [...runs.values()][0];
    if (!src) throw { kind: "invalid", detail: "That folder is not an export from LLM Trainer (its description file is missing)." };
    const r = createRunRecord({ name: `${folder.split("/").pop() ?? "Model"} (imported)`, preset: src.preset, model: null, train: null, datasetId: null, goal: null }, 3);
    r.status = "imported";
    return summary(r.id);
  },
  pickFolder: async () => "/Users/you/Desktop",
  pickFolders: async () => ["/Users/you/Documents/my notes"],
  watchDrops: (handler) => {
    const on = (e: Event) => handler((e as CustomEvent<{ type: "enter" | "leave" | "drop"; paths: string[] }>).detail);
    window.addEventListener("fixture-drop", on);
    return () => window.removeEventListener("fixture-drop", on);
  },

  chatOpen: async (req) => {
    const run = runs.get(req.runId);
    const sim = sims.get(req.runId);
    if (!run || !sim || sim.ckpts.length === 0) throw { kind: "invalid", detail: "This run has no saved progress yet. Let it train for a few minutes first." };
    const info: ChatSessionInfo = {
      id: nextChatId++, title: `Chat with ${run.name}`, runId: run.id, modelLabel: `${run.name}, after ${(sim.chars / 1e6).toFixed(1)}M characters`,
      mode: req.mode, learnEnabled: false, supportsLearn: true, needsGb: 0.5, hasAdaptedCopy: false, createdAt: NOW(),
    };
    chats.set(info.id, { info, messages: [], cancel: false, busy: false, quality: Math.min(1, sim.chars / 3e6) });
    return { ...info };
  },
  chatResume: async (id) => ({ ...chatOf(id).info }),
  chatSend: async (id, text, params: ChatParams, onEvent: (e: ChatEvent) => void) => {
    const chat = chatOf(id);
    if (!text.trim()) throw { kind: "invalid", detail: "Type something for the model to continue." };
    if (chat.busy) throw { kind: "invalid", detail: "The model is still writing. Stop it or wait for it to finish." };
    chat.busy = true; chat.cancel = false;
    const userId = nextMessageId++;
    chat.messages.push({ id: userId, role: "user", content: text, learned: false, learnNatsBefore: null, learnNatsAfter: null, rows: [], charsPerSec: null, createdAt: NOW() });
    const reply = REPLIES[(userId + id) % REPLIES.length]!.slice(0, params.maxNew);
    const rows: number[] = [];
    let shown = "";
    void (async () => {
      const t0 = performance.now();
      for (const ch of reply) {
        if (chat.cancel) break;
        const c = Math.random() > 0.55 + 0.45 * chat.quality ? "e" : ch;
        const r = 1 + Math.floor(Math.random() * 6);
        shown += c; rows.push(r);
        onEvent({ type: "chars", text: c, rows: [r], experts: [1 + Math.floor(Math.random() * 29), 1 + Math.floor(Math.random() * 29), 1 + Math.floor(Math.random() * 29)] });
        await new Promise((res) => setTimeout(res, 18));
      }
      const mid = nextMessageId++;
      const cps = shown.length / Math.max(0.001, (performance.now() - t0) / 1000);
      const msg: ChatMessage = { id: mid, role: "model", content: shown, learned: false, learnNatsBefore: null, learnNatsAfter: null, rows, charsPerSec: cps, createdAt: NOW() };
      chat.messages.push(msg);
      onEvent({ type: "done", messageId: mid, chars: shown.length, charsPerSec: cps });
      if (chat.info.learnEnabled && shown) {
        const learned = chat.messages.filter((m) => m.learned).length;
        msg.learned = true; msg.learnNatsBefore = 2.4 - 0.1 * learned; msg.learnNatsAfter = msg.learnNatsBefore - 0.1;
        onEvent({ type: "learned", natsBefore: msg.learnNatsBefore, natsAfter: msg.learnNatsAfter });
      }
      chat.busy = false;
    })();
    return userId;
  },
  chatStop: async (id) => { chatOf(id).cancel = true; },
  chatReset: async (id) => { const c = chatOf(id); c.cancel = true; c.messages = []; },
  chatSetLearn: async (id, enabled) => { const c = chatOf(id); c.info.learnEnabled = enabled; return { ...c.info }; },
  chatSaveAdapted: async (id) => { chatOf(id).info.hasAdaptedCopy = true; return { step: 0, chars: 0, kind: "manual", path: `chat/${id}/adapted`, bytes: 24, heldoutNats: null, nExperts: 16, engineFormat: 0 }; },
  chatClose: async (id) => { chats.delete(id); },
  chatListSessions: async () => [...chats.values()].map((c) => ({ ...c.info })).reverse(),
  chatGetMessages: async (id) => [...chatOf(id).messages],
  chatDeleteSession: async (id) => { chats.delete(id); },
};

function estimateFor(model: ModelConfig, train: TrainConfig) {
  const params = (n: number) => 8e6 * (model.dModel / 512) ** 2 + n * 3 * model.dModel * model.poolDFf;
  const slots = Math.min(train.resident, model.poolExperts);
  const gpu = (params(0) * 16 + slots * 3 * model.dModel * model.poolDFf * 12) / 1073741824 + (train.chunk >= 2048 ? 6 : 0.4);
  const cps = train.chunk >= 2048 ? 700 : model.dModel >= 384 ? 3200 : 9500;
  return { paramsStart: params(model.poolExperts), paramsFullPool: params(model.poolMax), gpuGb: gpu, ramGb: 0.6 + train.ramCache * 0.03, diskGb: 0.3 + model.poolMax * 0.01, charsPerSec: cps, measured: false, fit: gpu > 14 ? ("wont_fit" as const) : gpu > 11 ? ("tight" as const) : ("comfortable" as const), notes: gpu > 14 ? ["Needs more memory than this computer can offer."] : [] };
}
const estimate = (p: "tiny" | "small" | "full") => estimateFor(modelFor(p), trainFor(p));

seedFixture();
