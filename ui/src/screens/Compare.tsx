import { useQueries } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router";
import { backend } from "../backend";
import { CHART_INFO } from "../content/charts";
import { VERDICTS } from "../content/verdict";
import { configDiff, formatValue, keyLabel } from "../lib/diff";
import { fmtBits, fmtCount, fmtDuration, natsToBits } from "../lib/format";
import { isCodeLike } from "../lib/format";
import { type ChartLine, TimeSeriesChart } from "../components/TimeSeriesChart";
import clsx from "clsx";

/** Several runs side by side: their test-score curves on one chart, final results, what differed, and what they wrote. */
export function Compare() {
  const [params] = useSearchParams();
  const ids = useMemo(
    () => (params.get("ids") ?? "").split(",").map(Number).filter((n) => Number.isFinite(n) && n > 0).slice(0, 6),
    [params],
  );

  const runs = useQueries({ queries: ids.map((id) => ({ queryKey: ["run", id], queryFn: () => backend.getRun(id) })) });
  const series = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["series", { compare: id }],
      queryFn: () => backend.getSeries({ runId: id, keys: ["eval.overall"], x: "chars", from: null, to: null, maxPoints: 600 }),
    })),
  });
  const configs = useQueries({ queries: ids.map((id) => ({ queryKey: ["runConfig", id], queryFn: () => backend.getRunConfig(id) })) });
  const samples = useQueries({ queries: ids.map((id) => ({ queryKey: ["samples", id, "latest", 0], queryFn: () => backend.getSamples(id) })) });

  const ready = runs.every((r) => r.data) && configs.every((c) => c.data);
  if (ids.length < 2) {
    return (
      <div className="mx-auto max-w-[760px] px-8 py-14">
        <h1 className="m-0 text-[26px] font-semibold tracking-tight">Compare runs</h1>
        <p className="mt-3 text-[15px] text-ink-2">Choose two or more runs on the <Link to="/runs" className="text-accent-ink underline underline-offset-2">Runs</Link> page to compare them here.</p>
      </div>
    );
  }
  if (!ready) return <div className="mx-auto max-w-[1060px] px-8 py-14 text-ink-2">Loading…</div>;

  const list = runs.map((r) => r.data!);
  const lines: ChartLine[] = list.map((run, i) => {
    const s = series[i]?.data?.[0];
    return {
      key: String(run.id),
      name: run.name,
      color: `var(--lane-${i + 1})`,
      points: true,
      x: s?.x ?? [],
      y: (s?.y ?? []).map((v) => natsToBits(v)),
    };
  });

  const diff = configDiff(configs.map((c) => ({ model: c.data!.model, train: c.data!.train })));
  const modelDiff = diff.filter((d) => d.key.startsWith("model."));
  const trainDiff = diff.filter((d) => !d.key.startsWith("model."));
  const sizeNames = configs.map((c) => (c.data!.preset[0]!.toUpperCase() + c.data!.preset.slice(1)));
  const bestBits = Math.min(...list.map((r) => natsToBits(r.bestHeldoutNats) ?? Infinity));
  const firstSamples = samples.map((s) => s.data?.items[0]);

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-10">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">Compare runs</h1>
      <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-ink-2">The lower line is the better model. Below, only the settings that were different between these runs.</p>

      <section className="mt-8">
        <TimeSeriesChart
          title={CHART_INFO.score.title}
          chart="score"
          caption="Bits per character. Lower is better."
          lines={lines}
          yFormat={(v) => v.toFixed(1)}
          group="compare"
          height={300}
        />
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="final-h">
        <h2 id="final-h" className="m-0 text-lg font-semibold">How each one ended</h2>
        <table className="mt-4 w-full border-collapse text-sm">
          <thead>
            <tr className="text-left text-[13px] text-ink-2">
              <th className="py-2 pr-4 font-medium">Run</th>
              <th className="py-2 pr-4 font-medium">Size</th>
              <th className="py-2 pr-4 text-right font-medium">Text read</th>
              <th className="py-2 pr-4 text-right font-medium">Best score</th>
              <th className="py-2 pr-4 text-right font-medium">Last score</th>
              <th className="py-2 pr-4 text-right font-medium">Trained for</th>
              <th className="py-2 font-medium">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {list.map((r, i) => {
              const best = natsToBits(r.bestHeldoutNats);
              const isBest = best != null && best === bestBits;
              return (
                <tr key={r.id} className="border-t border-hairline">
                  <td className="py-3 pr-4">
                    <span className="inline-flex items-center gap-2">
                      <span aria-hidden className="inline-block h-[3px] w-4 rounded-sm" style={{ background: `var(--lane-${i + 1})` }} />
                      <Link to={`/train/${r.id}`} className="font-medium underline-offset-2 hover:underline">{r.name}</Link>
                    </span>
                  </td>
                  <td className="py-3 pr-4 text-ink-2">{r.preset[0]!.toUpperCase() + r.preset.slice(1)}</td>
                  <td className="py-3 pr-4 text-right">{fmtCount(r.charsRead)}</td>
                  <td className={clsx("py-3 pr-4 text-right", isBest && "font-semibold")}>{fmtBits(best)}{isBest && <span className="ml-1.5 text-xs font-medium text-good-ink">best</span>}</td>
                  <td className="py-3 pr-4 text-right">{fmtBits(natsToBits(r.lastHeldoutNats))}</td>
                  <td className="py-3 pr-4 text-right">{fmtDuration(r.activeMs)}</td>
                  <td className="py-3 text-ink-2">{r.verdict ? VERDICTS[r.verdict].label : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="diff-h">
        <h2 id="diff-h" className="m-0 text-lg font-semibold">What was different</h2>
        {diff.length === 0 ? (
          <p className="mt-3 text-[14px] text-ink-2">These runs used exactly the same settings.</p>
        ) : (
          <>
            <table className="mt-4 w-full border-collapse text-sm">
              <thead>
                <tr className="text-left text-[13px] text-ink-2">
                  <th className="py-2 pr-4 font-medium">Setting</th>
                  {list.map((r) => <th key={r.id} className="py-2 pr-4 font-medium">{r.name}</th>)}
                </tr>
              </thead>
              <tbody>
                {modelDiff.length > 0 && (
                  <tr className="border-t border-hairline">
                    <th scope="row" className="py-2.5 pr-4 text-left font-normal text-ink-2">Model size</th>
                    {sizeNames.map((n, i) => <td key={i} className="py-2.5 pr-4">{n}</td>)}
                  </tr>
                )}
                {trainDiff.map((row) => (
                  <tr key={row.key} className="border-t border-hairline">
                    <th scope="row" className="py-2.5 pr-4 text-left font-normal text-ink-2">{keyLabel(row.key)}</th>
                    {row.values.map((v, i) => <td key={i} className="py-2.5 pr-4">{formatValue(v)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
            {modelDiff.length > 0 && (
              <details className="mt-4">
                <summary className="cursor-pointer text-[13.5px] text-accent-ink">Show the {modelDiff.length} model settings that differ</summary>
                <table className="mt-3 w-full border-collapse text-sm">
                  <tbody>
                    {modelDiff.map((row) => (
                      <tr key={row.key} className="border-t border-hairline">
                        <th scope="row" className="py-2 pr-4 text-left font-normal text-ink-2">{keyLabel(row.key)}</th>
                        {row.values.map((v, i) => <td key={i} className="py-2 pr-4">{formatValue(v)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
          </>
        )}
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="write-h">
        <h2 id="write-h" className="m-0 text-lg font-semibold">What each one writes</h2>
        <p className="mt-1 text-[13.5px] text-ink-2">The same opening, continued by each model at the end of its run.</p>
        <div className="mt-5 grid gap-x-8 gap-y-6 md:grid-cols-2">
          {list.map((r, i) => {
            const s = firstSamples[i];
            return (
              <div key={r.id}>
                <p className="m-0 flex items-center gap-2 text-sm font-medium">
                  <span aria-hidden className="inline-block h-[3px] w-4 rounded-sm" style={{ background: `var(--lane-${i + 1})` }} />
                  {r.name}
                </p>
                {s ? (
                  <>
                    <p className={clsx("model-text mt-2 text-[13px] text-muted", isCodeLike(s.domain) && "font-mono")}>{s.prompt}</p>
                    <p className={clsx("model-text mt-1 text-ink", isCodeLike(s.domain) ? "font-mono text-[14px]" : "font-serif text-[18px] leading-[1.55]")}>{s.adapted}</p>
                  </>
                ) : (
                  <p className="mt-2 text-[13.5px] text-muted">No sample was written yet.</p>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
