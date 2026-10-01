import clsx from "clsx";
import { AlertTriangle, Clock, Copy, Minus, TrendingDown } from "lucide-react";
import type { Insight, Verdict } from "../bindings";
import { ADVICE, MILESTONES, VERDICTS } from "../content/verdict";
import { choicesGloss, fmtBits, fmtCount, fmtDuration, fmtRate, fmtRoughTime, readingGloss } from "../lib/format";
import { Term } from "./explain";

const ICONS = { clock: Clock, "trending-down": TrendingDown, minus: Minus, copy: Copy, alert: AlertTriangle } as const;
const TONE = {
  neutral: "text-ink-2",
  good: "text-good-ink",
  warning: "text-warning-ink",
  critical: "text-critical-ink",
} as const;

/** "Is it learning?" in plain words, with an icon and label so colour is never the only signal. */
export function VerdictCard({ verdict, insight }: { verdict: Verdict | null; insight: Insight | null }) {
  const v = VERDICTS[verdict ?? "warming_up"];
  const Icon = ICONS[v.icon];
  const advice = insight?.report.advice[0];
  const change = insight?.report.changePct;
  return (
    <div>
      <p className="m-0 text-[13px] text-ink-2">Is it learning?</p>
      <p className={clsx("mt-1 flex items-center gap-2 text-[17px] font-semibold leading-tight", TONE[v.tone])}>
        <Icon size={18} aria-hidden />
        <span className="text-ink">{v.label}</span>
      </p>
      <p className="mt-1.5 text-[13.5px] leading-snug text-ink-2">
        {v.headline}
        {verdict === "learning" && change != null && change > 0 && <> It improved {change.toFixed(change < 10 ? 1 : 0)}% over the last few checks.</>}
      </p>
      {advice && <p className="mt-1.5 text-[13.5px] leading-snug text-ink">{ADVICE[advice]}</p>}
    </div>
  );
}

interface HeroProps {
  insight: Insight | null;
  lastNats: number | null;
  prevBits: number | null;
  chars: number | null;
  cps: number | null;
  activeMs: number | null;
  goalMinutes: number | null;
}

const Stat = ({ label, value, unit, gloss }: { label: React.ReactNode; value: React.ReactNode; unit?: string; gloss?: React.ReactNode }) => (
  <div className="min-w-0 px-6 first:pl-0 last:pr-0">
    <p className="m-0 text-[13px] text-ink-2">{label}</p>
    <p className="mt-1 flex items-baseline gap-1.5 text-[30px] font-semibold leading-none tracking-tight">
      {value}
      {unit && <span className="text-[13px] font-normal tracking-normal text-muted">{unit}</span>}
    </p>
    {gloss && <p className="mt-2 text-[13px] leading-snug text-ink-2">{gloss}</p>}
  </div>
);

/** The four numbers that matter, separated by hairlines rather than boxed into cards. */
export function HeroStats({ insight, lastNats, prevBits, chars, cps, activeMs, goalMinutes }: HeroProps) {
  const bits = insight?.bitsPerChar ?? (lastNats != null ? lastNats / Math.LN2 : null);
  const delta = bits != null && prevBits != null ? bits - prevBits : null;
  const eta = insight?.eta;
  return (
    <div className="grid grid-cols-2 gap-y-6 divide-hairline md:grid-cols-4 md:divide-x">
      <Stat
        label={<>Test score <Term k="bits_per_char">(bits per character)</Term></>}
        value={bits != null ? fmtBits(bits) : "–"}
        gloss={
          bits == null ? "Appears after the first progress check." : (
            <>
              {delta != null && Math.abs(delta) >= 0.005 && (
                <span className={clsx("mr-1.5 font-medium", delta < 0 ? "text-good-ink" : "text-critical-ink")}>
                  {delta < 0 ? "▼" : "▲"} {Math.abs(delta).toFixed(2)}
                </span>
              )}
              {choicesGloss(bits * Math.LN2)}
            </>
          )
        }
      />
      <Stat label="Text read" value={fmtCount(chars)} unit="characters" gloss={`That is ${readingGloss(chars)}.`} />
      <Stat label="Reading speed" value={fmtRate(cps)} unit="characters per second" gloss="Steady is good." />
      <Stat
        label="Time"
        value={fmtDuration(activeMs)}
        gloss={
          eta && eta.seconds > 0
            ? `${fmtRoughTime(eta.seconds)} to reach a score of 2.0.`
            : goalMinutes
              ? `Goal: ${goalMinutes} minutes.`
              : insight?.milestone
                ? MILESTONES[insight.milestone.key]
                : undefined
        }
      />
    </div>
  );
}

/** Where the model is on the road from random guessing to fluent writing. */
export function MilestoneStrip({ insight }: { insight: Insight | null }) {
  const m = insight?.milestone ?? { index: 0, total: 6, key: "random_guessing" };
  const keys = Object.keys(MILESTONES);
  return (
    <div>
      <ol className="m-0 flex list-none items-center gap-1 p-0" aria-label="Progress milestones">
        {keys.map((k, i) => (
          <li key={k} aria-current={i === m.index ? "step" : undefined} className="flex flex-1 items-center">
            <span className={clsx("h-1.5 flex-1 rounded-full", i <= m.index ? "bg-accent" : "bg-surface-2")} title={MILESTONES[k]} />
          </li>
        ))}
      </ol>
      <p className="mt-2 text-[13px] text-ink-2">
        Right now: <span className="font-medium text-ink">{MILESTONES[m.key] ?? m.key}</span>
        {m.index + 1 < m.total && <span className="text-muted"> Next: {MILESTONES[keys[m.index + 1]!]?.toLowerCase()}.</span>}
      </p>
    </div>
  );
}
