import { beforeEach, describe, expect, it } from "vitest";
import type { LiveMsg, StageInfo } from "../src/bindings";
import { useLive } from "../src/state/live";

const stage = (runId: number, s: StageInfo["stage"]): LiveMsg => ({ type: "stage", runId, info: { stage: s, detail: null, progress: null }, at: 1 });

beforeEach(() => {
  useLive.setState({ runId: null, run: null, runState: null, stage: null, pulse: null, lastEval: null, samples: null, pool: null, insight: null, lastCheckpoint: null, warnings: [], error: null, version: 0 });
});

describe("live store", () => {
  it("folds stage and state messages for the current run", () => {
    const { apply } = useLive.getState();
    apply(stage(1, "reading"));
    apply({ type: "state", runId: 1, state: "running", reason: null });
    const s = useLive.getState();
    expect(s.runId).toBe(1);
    expect(s.stage?.stage).toBe("reading");
    expect(s.runState).toBe("running");
  });

  it("a message about a different run starts a fresh feed", () => {
    const { apply } = useLive.getState();
    apply(stage(1, "reading"));
    apply({ type: "warning", runId: 1, code: "x", message: "old warning", hint: null });
    apply(stage(2, "preparing_data"));
    const s = useLive.getState();
    expect(s.runId).toBe(2);
    expect(s.stage?.stage).toBe("preparing_data");
    expect(s.warnings).toEqual([]);
  });

  it("keeps the newest warnings first and caps the list", () => {
    const { apply } = useLive.getState();
    for (let i = 0; i < 25; i++) apply({ type: "warning", runId: 1, code: `w${i}`, message: `m${i}`, hint: null });
    const w = useLive.getState().warnings;
    expect(w).toHaveLength(20);
    expect(w[0]?.code).toBe("w24");
  });

  it("pulses update the numbers without bumping the version", () => {
    const { apply } = useLive.getState();
    apply({ type: "state", runId: 1, state: "running", reason: null });
    const v = useLive.getState().version;
    apply({ type: "pulse", runId: 1, step: 5, chars: 2560, charsTotal: null, trainNatsEma: 4.2, readCps: 9000, activeMs: 1000, schedule: { nextEvalChars: null, nextSampleMs: 5000, nextCheckpointMs: null, nextGrowthChars: null } });
    const s = useLive.getState();
    expect(s.pulse?.chars).toBe(2560);
    expect(s.version).toBe(v);
  });

  it("recovers from a snapshot after a reload", () => {
    useLive.getState().setSnapshot({ run: { id: 7, status: "running" } as never, stage: { stage: "paused", detail: null, progress: null }, lastEval: null, lastPool: null, insight: null, schedule: null });
    const s = useLive.getState();
    expect(s.runId).toBe(7);
    expect(s.runState).toBe("running");
    expect(s.stage?.stage).toBe("paused");
  });
});
