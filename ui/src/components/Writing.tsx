import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";
import { fmtCount, isCodeLike, laneColor, laneSlot } from "../lib/format";
import { useSampleSteps, useSamples } from "../state/queries";
import { Term } from "./explain";

/**
 * "Watch it learn to write." The model's own text, typeset as prose, with a slider through every sample round so the
 * progress from noise to words is visible. When the slider is at the end it follows the live run.
 */
export function Writing({ runId, version, live }: { runId: number | null; version: number; live: boolean }) {
  const steps = useSampleSteps(runId, version);
  const list = steps.data ?? [];
  const [pick, setPick] = useState<number | null>(null); // index into `list`; null follows the newest round
  const [domain, setDomain] = useState<string | null>(null);
  const [guard, setGuard] = useState(true);

  // A different run starts again from the newest round.
  useEffect(() => {
    setPick(null);
    setDomain(null);
  }, [runId]);

  const idx = pick == null ? list.length - 1 : Math.min(pick, list.length - 1);
  const step = idx >= 0 ? list[idx]?.step : undefined;
  const samples = useSamples(runId, pick == null ? undefined : step, version);
  const round = samples.data;
  const domains = useMemo(() => round?.items.map((i) => i.domain) ?? [], [round]);
  const active = round?.items.find((i) => i.domain === domain) ?? round?.items[0];

  if (!round || !active) {
    return (
      <div className="flex min-h-[240px] flex-col justify-center">
        <p className="m-0 text-[15px] font-medium">Its writing will appear here.</p>
        <p className="mt-1 max-w-[52ch] text-[13.5px] text-ink-2">
          After each progress check the model is asked to continue a few openings. Early on it writes nonsense. Watch it turn into words.
        </p>
      </div>
    );
  }

  const text = guard ? active.adapted : active.raw;
  const code = isCodeLike(active.domain);
  const following = pick == null || idx === list.length - 1;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {domains.length > 1 && (
          <div role="tablist" aria-label="Kind of text" className="flex flex-wrap gap-1">
            {domains.map((d) => (
              <button
                key={d}
                role="tab"
                aria-selected={d === active.domain}
                onClick={() => setDomain(d)}
                className={clsx("inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[13px] font-medium transition-colors", d === active.domain ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2")}
              >
                <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: laneColor(laneSlot(d, domains)) }} />
                {d}
              </button>
            ))}
          </div>
        )}
        <label className="ml-auto flex items-center gap-2 text-[13px] text-ink-2">
          <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} className="accent-[var(--accent)]" />
          Repetition guard
        </label>
      </div>

      <div className="mt-4 min-h-[170px]">
        <p className={clsx("model-text m-0 text-[13.5px] text-muted", code && "font-mono")}>{active.prompt}</p>
        <p
          className={clsx("model-text mt-1 text-ink", code ? "font-mono text-[15px] leading-relaxed" : "font-serif text-[21px] leading-[1.55]")}
          aria-label="Text written by the model"
        >
          {text}
        </p>
      </div>

      {list.length > 1 && (
        <div className="mt-5">
          <input
            type="range"
            min={0}
            max={list.length - 1}
            value={idx}
            onChange={(e) => {
              const v = Number(e.target.value);
              setPick(v >= list.length - 1 ? null : v);
            }}
            aria-label="Move through time"
            aria-valuetext={`After ${fmtCount(round.chars)} characters read`}
            className="h-1.5 w-full cursor-pointer accent-[var(--accent)]"
          />
          <div className="mt-1.5 flex items-center justify-between text-[13px] text-ink-2">
            <span>
              After <span className="font-medium text-ink">{fmtCount(round.chars)}</span> characters read
              {active.rawRep8 != null && guard === false && active.rawRep8 > 15 && <> It is repeating itself a lot; <Term k="held_out">the guard</Term> helps with that.</>}
            </span>
            {live && following ? <span className="font-medium text-accent-ink">Following live</span> : <button className="text-accent-ink underline underline-offset-2" onClick={() => setPick(null)}>Jump to latest</button>}
          </div>
        </div>
      )}
    </div>
  );
}
