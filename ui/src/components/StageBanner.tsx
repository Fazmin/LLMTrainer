import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Pause, Play, Square } from "lucide-react";
import type { ReactNode } from "react";
import { backend } from "../backend";
import type { RunState, RunSummary } from "../bindings";
import { RUN_STATES, STAGES } from "../content/stages";
import { describeError } from "../content/errors";
import { fmtDuration } from "../lib/format";
import { useLive } from "../state/live";
import { Button } from "./Button";

type Tone = "active" | "paused" | "bad" | "done" | "idle";

const TONE: Record<Tone, { bar: string; dot: string; wash: string }> = {
  active: { bar: "bg-accent", dot: "bg-accent", wash: "bg-accent-soft" },
  paused: { bar: "bg-warning", dot: "bg-warning", wash: "bg-surface" },
  bad: { bar: "bg-critical", dot: "bg-critical", wash: "bg-surface" },
  done: { bar: "bg-good", dot: "bg-good", wash: "bg-surface" },
  idle: { bar: "bg-axis", dot: "bg-muted", wash: "bg-surface" },
};

function toneFor(state: RunState | null): Tone {
  switch (state) {
    case "running": case "preparing": case "stopping": return "active";
    case "paused": return "paused";
    case "failed": case "interrupted": return "bad";
    case "completed": case "stopped": case "imported": return "done";
    default: return "idle";
  }
}

const isLiveState = (s: RunState | null) => s === "preparing" || s === "running" || s === "paused" || s === "stopping";

/**
 * The persistent "What's happening now" band: which stage the run is in, what that means in plain words, how far along
 * it is, and the controls that apply right now.
 */
export function StageBanner({ run, onResume }: { run: RunSummary | null; onResume?: (run: RunSummary) => void }) {
  const live = useLive();
  const isLiveRun = run != null && live.runId === run.id;
  const state: RunState | null = isLiveRun ? (live.runState ?? run.status) : (run?.status ?? null);
  const active = isLiveState(state);
  const tone = toneFor(state);
  const stageKey = isLiveRun && active ? live.stage?.stage : undefined;
  const copy = stageKey ? STAGES[stageKey] : state ? RUN_STATES[state] : { title: "No training yet", blurb: "Choose a size in Setup and start. This is where you will see what the model is doing." };
  const reading = state === "running" && (!stageKey || stageKey === "reading");

  const pulse = isLiveRun ? live.pulse : null;
  const goal = run?.goal ?? null;
  const elapsed = pulse?.activeMs ?? run?.activeMs ?? 0;
  let progress: number | null = null;
  if (goal?.type === "minutes") progress = Math.min(1, elapsed / (goal.value * 60000));
  else if (goal?.type === "chars") progress = Math.min(1, (pulse?.chars ?? run?.charsRead ?? 0) / goal.value);
  const nextCheck = pulse?.schedule.nextSampleMs;

  const fail = isLiveRun && live.error?.fatal ? describeError(live.error.error) : null;
  const stateErr = !fail && (state === "failed") && run?.error ? { title: "", body: run.error } : null;

  const run_ = async (fn: () => Promise<unknown>) => {
    try { await fn(); } catch (e) { console.error(e); }
  };

  return (
    <div role="status" aria-live="polite" className={clsx("relative border-b border-hairline", TONE[tone].wash)}>
      <span aria-hidden className={clsx("absolute inset-y-0 left-0 w-1", TONE[tone].bar)} />
      <div className="mx-auto flex max-w-[1180px] flex-wrap items-center gap-x-6 gap-y-3 px-8 py-4">
        <div className="min-w-0 flex-1 basis-[420px]">
          <div className="flex items-center gap-2.5">
            {tone === "bad" ? (
              <AlertTriangle size={16} className="text-critical-ink" />
            ) : tone === "done" ? (
              <CheckCircle2 size={16} className="text-good-ink" />
            ) : (
              <span aria-hidden className={clsx("inline-block h-2.5 w-2.5 rounded-full", TONE[tone].dot, reading && "pulse-dot")} />
            )}
            <h2 className="m-0 text-[17px] font-semibold leading-tight">{copy.title}</h2>
            {isLiveRun && active && live.stage?.detail && <span className="text-[13px] text-muted">{live.stage.detail}</span>}
          </div>
          <p className="mt-1 max-w-[68ch] text-[13.5px] leading-snug text-ink-2">{fail ? `${fail.title}. ${fail.body}` : stateErr ? stateErr.body : copy.blurb}</p>
          {(active || progress != null) && (
            <div className="mt-2.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-[13px] text-ink-2">
              {progress != null && (
                <span className="flex items-center gap-2">
                  <span className="h-1.5 w-40 overflow-hidden rounded-full bg-surface-2" aria-hidden>
                    <span className="block h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${Math.round(progress * 100)}%` }} />
                  </span>
                  <span>
                    {goal?.type === "minutes" ? `${fmtDuration(elapsed)} of ${goal.value} minutes` : `${Math.round(progress * 100)}% of the goal`}
                  </span>
                </span>
              )}
              {active && progress == null && <span>Training for {fmtDuration(elapsed)}</span>}
              {active && nextCheck != null && state === "running" && <span>Next progress check in {fmtDuration(nextCheck)}</span>}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {isLiveRun && state === "running" && (
            <Button icon={<Pause size={15} />} onClick={() => run_(() => backend.pauseRun())}>Pause</Button>
          )}
          {isLiveRun && state === "paused" && (
            <Button variant="primary" icon={<Play size={15} />} onClick={() => run_(() => backend.resumeRun())}>Resume</Button>
          )}
          {isLiveRun && (state === "running" || state === "paused") && (
            <Button variant="secondary" icon={<Square size={14} />} onClick={() => run_(() => backend.stopRun(true))}>Stop and save</Button>
          )}
          {run && !active && onResume && (state === "stopped" || state === "interrupted" || state === "created" || state === "failed" || state === "completed") && (
            <Button variant="primary" icon={<Play size={15} />} onClick={() => onResume(run)}>
              {state === "created" ? "Start" : "Continue training"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline gap-3">
      <h2 className="m-0 text-lg font-semibold">{children}</h2>
      {aside}
    </div>
  );
}
