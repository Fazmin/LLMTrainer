import type { Insight, RunSummary } from "../bindings";

/** Same milestone ladder as the engine, for runs that are not live (history view has no stored insight). */
const LADDER: [string, number][] = [
  ["fluent_sentences", 1.3],
  ["simple_sentences", 1.8],
  ["short_phrases", 2.5],
  ["common_words", 3.5],
  ["letter_frequencies", 5.0],
];
const KEYS = ["random_guessing", "letter_frequencies", "common_words", "short_phrases", "simple_sentences", "fluent_sentences"];

export function milestoneFor(bits: number | null) {
  let key = "random_guessing";
  if (bits != null) for (const [k, t] of LADDER) if (bits <= t) { key = k; break; }
  return { index: KEYS.indexOf(key), total: KEYS.length, key };
}

/** An insight for a finished or stopped run, from what the run row stores. */
export function insightFromRun(run: RunSummary): Insight {
  const bits = run.lastHeldoutNats != null && Number.isFinite(run.lastHeldoutNats) ? run.lastHeldoutNats / Math.LN2 : null;
  return {
    report: { verdict: run.verdict ?? "warming_up", changePct: 0, advice: [] },
    eta: null,
    milestone: milestoneFor(bits),
    bitsPerChar: bits,
  };
}
