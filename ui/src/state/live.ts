import { create } from "zustand";
import type {
  AppError,
  CheckpointMeta,
  EvalResult,
  Insight,
  LiveMsg,
  LiveSnapshot,
  PoolSnapshot,
  RunState,
  RunSummary,
  SampleRound,
  Schedule,
  StageInfo,
} from "../bindings";

export interface Pulse {
  step: number;
  chars: number;
  charsTotal: number | null;
  trainNatsEma: number | null;
  readCps: number | null;
  activeMs: number;
  schedule: Schedule;
}

export interface Warning {
  code: string;
  message: string;
  hint: string | null;
  at: number;
}

/** Everything the dashboard shows "right now", folded from the ordered live message stream. */
export interface LiveState {
  runId: number | null;
  run: RunSummary | null;
  runState: RunState | null;
  stage: StageInfo | null;
  stageAt: number;
  pulse: Pulse | null;
  lastEval: EvalResult | null;
  samples: SampleRound | null;
  pool: PoolSnapshot | null;
  insight: Insight | null;
  lastCheckpoint: CheckpointMeta | null;
  warnings: Warning[];
  error: { error: AppError; fatal: boolean } | null;
  /** Bumped on every state-changing message so queries can refresh. */
  version: number;
  apply: (msg: LiveMsg) => void;
  setSnapshot: (s: LiveSnapshot) => void;
}

const EMPTY = {
  runState: null,
  stage: null,
  stageAt: 0,
  pulse: null,
  lastEval: null,
  samples: null,
  pool: null,
  insight: null,
  lastCheckpoint: null,
  warnings: [] as Warning[],
  error: null,
};

export const useLive = create<LiveState>((set, get) => ({
  runId: null,
  run: null,
  ...EMPTY,
  version: 0,

  setSnapshot: (s) =>
    set({
      ...EMPTY,
      runId: s.run?.id ?? null,
      run: s.run,
      runState: s.run?.status ?? null,
      stage: s.stage,
      stageAt: Date.now(),
      lastEval: s.lastEval,
      pool: s.lastPool,
      insight: s.insight,
      pulse: null,
      version: get().version + 1,
    }),

  apply: (msg) => {
    const cur = get();
    // A message about a different run starts a fresh feed.
    const base = msg.runId !== cur.runId ? { ...EMPTY, runId: msg.runId, run: null } : {};
    const bump = { version: cur.version + 1 };
    switch (msg.type) {
      case "stage":
        return set({ ...base, stage: msg.info, stageAt: msg.at, ...bump });
      case "state":
        return set({ ...base, runState: msg.state, ...bump });
      case "pulse":
        return set({
          ...base,
          pulse: {
            step: msg.step,
            chars: msg.chars,
            charsTotal: msg.charsTotal,
            trainNatsEma: msg.trainNatsEma,
            readCps: msg.readCps,
            activeMs: msg.activeMs,
            schedule: msg.schedule,
          },
        });
      case "ticks":
        return set({ ...base });
      case "eval":
        return set({ ...base, lastEval: msg.result, ...bump });
      case "samples":
        return set({ ...base, samples: msg.round, ...bump });
      case "pool":
        return set({ ...base, pool: msg.snapshot, ...bump });
      case "insight":
        return set({ ...base, insight: msg.insight, ...bump });
      case "checkpoint":
        return set({ ...base, lastCheckpoint: msg.checkpoint, ...bump });
      case "warning": {
        const earlier = msg.runId !== cur.runId ? [] : cur.warnings;
        return set({
          ...base,
          warnings: [{ code: msg.code, message: msg.message, hint: msg.hint, at: Date.now() }, ...earlier].slice(0, 20),
        });
      }
      case "error":
        return set({ ...base, error: { error: msg.error, fatal: msg.fatal }, ...bump });
      case "growth":
      case "plasticity":
        return set({ ...base, ...bump });
    }
  },
}));
