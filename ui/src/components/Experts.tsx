import clsx from "clsx";
import { Check, X } from "lucide-react";
import type { PoolSnapshot } from "../bindings";
import { Term } from "./explain";

const BRAKES: { key: keyof PoolSnapshot["brakes"]; label: string }[] = [
  { key: "room", label: "There is room" },
  { key: "used", label: "Experts are used" },
  { key: "earning", label: "Newcomers earn their place" },
  { key: "fits", label: "Few are on trial" },
  { key: "honest", label: "Not just memorising" },
];

/** The expert pool: who is on the GPU now, how busy every expert is, and why the pool may or may not grow. */
export function ExpertsPanel({ pool }: { pool: PoolSnapshot | null }) {
  if (!pool) {
    return <p className="py-10 text-center text-[13.5px] text-muted">The expert pool appears after the first progress check.</p>;
  }
  const max = Math.max(1e-6, ...pool.resident.map((e) => e.useShare));
  return (
    <div className="grid gap-x-12 gap-y-8 lg:grid-cols-[1fr_1fr]">
      <div>
        <h3 className="m-0 text-[15px] font-semibold">Working now ({pool.resident.length} of {pool.nExperts})</h3>
        <p className="mt-1 max-w-[52ch] text-[13px] text-ink-2">
          These <Term k="experts" /> are loaded and doing the work. Darker means busier. A ring marks a newcomer still on trial; a dashed outline marks one that is barely used.
        </p>
        <ul className="m-0 mt-4 grid list-none grid-cols-8 gap-1.5 p-0" aria-label="Experts working now">
          {pool.resident.map((e) => (
            <li
              key={e.uid}
              tabIndex={0}
              title={`Expert ${e.uid}: ${(e.useShare * 100).toFixed(1)}% of recent work`}
              aria-label={`Expert ${e.uid}, ${(e.useShare * 100).toFixed(1)} percent of recent work${e.onTrial ? ", on trial" : ""}${e.dying ? ", barely used" : ""}`}
              className={clsx("relative aspect-square rounded-[6px] bg-accent", e.onTrial && "ring-2 ring-accent ring-offset-2 ring-offset-page", e.dying && "outline-dashed outline-2 -outline-offset-2 outline-ink-2")}
              style={{ opacity: 0.2 + 0.8 * (e.useShare / max) }}
            />
          ))}
        </ul>
      </div>

      <div>
        <h3 className="m-0 text-[15px] font-semibold"><Term k="growing">Can the pool grow?</Term></h3>
        <p className="mt-1 max-w-[52ch] text-[13px] text-ink-2">
          {pool.brakes.room.ok && pool.brakes.used.ok && pool.brakes.earning.ok && pool.brakes.fits.ok && pool.brakes.honest.ok
            ? "All five checks pass, so a new expert may be added soon."
            : "A new expert is added only when all five checks pass."}
        </p>
        <ul className="m-0 mt-3 list-none space-y-2 p-0">
          {BRAKES.map(({ key, label }) => {
            const b = pool.brakes[key];
            return (
              <li key={key} className="flex items-start gap-2.5 text-[13.5px]">
                {b.ok ? <Check size={16} className="mt-0.5 shrink-0 text-good-ink" aria-label="Passing" /> : <X size={16} className="mt-0.5 shrink-0 text-critical-ink" aria-label="Not passing" />}
                <span>
                  <span className="font-medium">{label}.</span> <span className="text-ink-2">{b.why}</span>
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="lg:col-span-2">
        <h3 className="m-0 text-[15px] font-semibold">Everyone in the pool, busiest first</h3>
        <p className="mt-1 text-[13px] text-ink-2">{pool.nExperts} experts. A few do most of the work, which is normal.</p>
        <div className="mt-3 flex h-16 items-end gap-[2px]" role="img" aria-label={`Usage of all ${pool.nExperts} experts, busiest first`}>
          {pool.usage.map((u, i) => (
            <span
              key={i}
              title={`Expert #${i + 1} by use: ${Math.round((u / 255) * 100)}% of the busiest`}
              className={clsx("min-w-[3px] flex-1 rounded-t-[2px] bg-accent", i < pool.resident.length ? "opacity-100" : "opacity-45")}
              style={{ height: `${Math.max(4, (u / 255) * 100)}%` }}
            />
          ))}
        </div>
        <p className="mt-1.5 text-xs text-muted">Solid bars are loaded now; pale bars are waiting on disk.</p>
      </div>
    </div>
  );
}

/** How many characters take 1 step, 2 steps, and so on. */
export function HaltingBars({ hist }: { hist: number[] | undefined }) {
  if (!hist || hist.length === 0) return <p className="py-6 text-[13.5px] text-muted">Appears after the first progress check.</p>;
  const max = Math.max(...hist, 1e-6);
  const mean = hist.reduce((a, v, i) => a + v * (i + 1), 0) / Math.max(1e-6, hist.reduce((a, v) => a + v, 0));
  return (
    <div>
      <p className="m-0 text-[13px] text-ink-2">On average it now takes <span className="font-semibold text-ink">{mean.toFixed(1)}</span> <Term k="thinking_depth">thinking steps</Term> per character.</p>
      <div className="mt-3 flex h-36 items-end gap-1.5" role="img" aria-label="Share of characters by number of thinking steps">
        {hist.map((v, i) => (
          <div key={i} className="flex h-full flex-1 flex-col items-center justify-end gap-1">
            <span className="w-full rounded-t-[3px] bg-accent" title={`${(v * 100).toFixed(1)}% of characters take ${i + 1} step${i ? "s" : ""}`} style={{ height: `${Math.max(2, (v / max) * 100)}%` }} />
            <span className="text-[11px] text-muted">{i + 1}</span>
          </div>
        ))}
      </div>
      <p className="mt-1 text-center text-xs text-muted">Thinking steps per character</p>
    </div>
  );
}
