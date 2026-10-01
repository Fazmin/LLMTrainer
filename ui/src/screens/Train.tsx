import { Activity, Brain, Cpu, Layers } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { backend } from "../backend";
import type { RunSummary, SeriesData } from "../bindings";
import { Button } from "../components/Button";
import { ForkDialog } from "../components/ForkDialog";
import { ExpertsPanel, HaltingBars } from "../components/Experts";
import { StageBanner } from "../components/StageBanner";
import { TabStrip } from "../components/Tabs";
import { type Annotation, type ChartLine, TimeSeriesChart } from "../components/TimeSeriesChart";
import { HeroStats, MilestoneStrip, VerdictCard } from "../components/Verdict";
import { Writing } from "../components/Writing";
import { CHART_INFO } from "../content/charts";
import { describeError } from "../content/errors";
import { fmtBits, fmtCount, fmtWhen, laneColor, laneSlot, natsToBits } from "../lib/format";
import { insightFromRun } from "../lib/insight";
import { useLive } from "../state/live";
import { usePrefs } from "../state/prefs";
import { useCheckpoints, useEvalDomains, useEvents, useHardware, usePool, useRun, useRuns, useSeries } from "../state/queries";

type Tab = "progress" | "experts" | "thinking" | "system";

const toBits = (s: SeriesData | undefined): Pick<ChartLine, "x" | "y" | "lo" | "hi"> =>
  s
    ? { x: s.x, y: s.y.map((v) => natsToBits(v)), lo: s.yLo.map((v) => natsToBits(v)), hi: s.yHi.map((v) => natsToBits(v)) }
    : { x: [], y: [] };

const plain = (s: SeriesData | undefined): Pick<ChartLine, "x" | "y" | "lo" | "hi"> =>
  s ? { x: s.x, y: s.y, lo: s.yLo, hi: s.yHi } : { x: [], y: [] };

function useCurrentRun(): { run: RunSummary | null; loading: boolean } {
  const { runId } = useParams();
  const live = useLive();
  const runs = useRuns();
  const explicit = runId ? Number(runId) : null;
  const id = explicit ?? live.runId ?? runs.data?.[0]?.id ?? null;
  const q = useRun(id);
  // While live, the run row in the store is the freshest source; the query catches up on state changes.
  const run = q.data ?? (live.run && live.run.id === id ? live.run : null);
  return { run, loading: q.isLoading || runs.isLoading };
}

export function Train() {
  const navigate = useNavigate();
  const { run, loading } = useCurrentRun();
  const live = useLive();
  const level = usePrefs((s) => s.level);
  const [tab, setTab] = useState<Tab>("progress");
  const runId = run?.id ?? null;
  const isLiveRun = run != null && live.runId === run.id;
  const liveState = isLiveRun ? live.runState : null;
  const active = liveState === "running" || liveState === "paused" || liveState === "preparing" || liveState === "stopping";
  const version = isLiveRun ? live.version : 0;

  const startAgain = async (r: RunSummary) => {
    try {
      await backend.startRun(r.id);
    } catch (e) {
      const d = describeError(e);
      window.alert(`${d.title}\n\n${d.body}`);
    }
  };

  if (!run) {
    return (
      <div>
        <StageBanner run={null} />
        <div className="mx-auto max-w-[620px] px-8 py-20 text-center">
          {loading ? null : (
            <>
              <h1 className="m-0 text-2xl font-semibold tracking-tight">Nothing is training yet</h1>
              <p className="mx-auto mt-3 max-w-[48ch] text-[15px] leading-relaxed text-ink-2">
                When you start, this page shows what the model is doing, how well it is learning, and what it can write.
              </p>
              <Button className="mt-7" variant="primary" size="lg" onClick={() => navigate("/setup")}>Set up a training</Button>
            </>
          )}
        </div>
      </div>
    );
  }

  const insight = isLiveRun && live.insight ? live.insight : insightFromRun(run);
  const verdict = insight.report.verdict;

  return (
    <div>
      <StageBanner run={run} onResume={startAgain} />
      <div className="mx-auto max-w-[1180px] px-8 pb-16 pt-6">
        <RunHeader run={run} />

        <section className="mt-6 grid gap-x-14 gap-y-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]" aria-label="What the model can do now">
          <div>
            <h2 className="m-0 mb-3 text-lg font-semibold">What it writes</h2>
            <Writing runId={runId} version={version} live={active} />
          </div>
          <div className="flex flex-col gap-8 lg:border-l lg:border-hairline lg:pl-10">
            <VerdictCard verdict={verdict} insight={insight} />
            <div>
              <h3 className="m-0 mb-2.5 text-[15px] font-semibold">How far along</h3>
              <MilestoneStrip insight={insight} />
            </div>
          </div>
        </section>

        <HeadlineNumbers run={run} insight={insight} isLiveRun={isLiveRun} />

        <div className="mt-10">
          <TabStrip
            tabs={[
              { id: "progress", label: "Progress", icon: <Activity size={15} /> },
              { id: "experts", label: "Experts", icon: <Layers size={15} /> },
              { id: "thinking", label: "Thinking depth", icon: <Brain size={15} /> },
              { id: "system", label: "This computer", icon: <Cpu size={15} /> },
            ]}
            value={tab}
            onChange={setTab}
          />
          <div className="pt-7" role="tabpanel">
            {tab === "progress" && <ProgressCharts runId={run.id} live={active} version={version} advanced={level === "advanced"} />}
            {tab === "experts" && <ExpertsTab runId={run.id} version={version} />}
            {tab === "thinking" && <ThinkingTab runId={run.id} version={version} live={active} />}
            {tab === "system" && <SystemTab run={run} version={version} />}
          </div>
        </div>
      </div>
    </div>
  );
}

function RunHeader({ run }: { run: RunSummary }) {
  const ckpts = useCheckpoints(run.id, run.charsRead);
  const live = useLive();
  const [forking, setForking] = useState(false);
  const busy = live.runId === run.id && (live.runState === "running" || live.runState === "paused" || live.runState === "preparing" || live.runState === "stopping");
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <h1 className="m-0 text-[22px] font-semibold tracking-tight">{run.name}</h1>
      <span className="text-sm text-ink-2">{run.preset[0]!.toUpperCase() + run.preset.slice(1)} model</span>
      {run.datasetName && <span className="text-sm text-ink-2">Reading: {run.datasetName}</span>}
      <span className="ml-auto flex items-center gap-5 text-sm">
        {(ckpts.data?.length ?? 0) > 0 && (
          <>
            <Link to={`/chat?run=${run.id}`} className="text-accent-ink underline underline-offset-2">Chat with this model</Link>
            <Link to={`/export?run=${run.id}`} className="text-accent-ink underline underline-offset-2">Export</Link>
            {!busy && (
              <>
                <button type="button" onClick={() => setForking(true)} className="cursor-pointer border-0 bg-transparent p-0 text-sm text-accent-ink underline underline-offset-2">Continue as a new run…</button>
                <ForkDialog run={run} open={forking} onOpenChange={setForking} />
              </>
            )}
          </>
        )}
        <Link to="/runs" className="text-accent-ink underline underline-offset-2">All runs</Link>
      </span>
    </div>
  );
}

function HeadlineNumbers({ run, insight, isLiveRun }: { run: RunSummary; insight: ReturnType<typeof insightFromRun>; isLiveRun: boolean }) {
  const live = useLive();
  const evalSeries = useSeries({ runId: run.id, keys: ["eval.overall"], x: "chars", from: null, to: null, maxPoints: 500 }, isLiveRun);
  const ys = (evalSeries.data?.[0]?.y ?? []).filter((v): v is number => v != null);
  const prevBits = ys.length >= 2 ? natsToBits(ys[ys.length - 2]!) : null;
  const pulse = isLiveRun ? live.pulse : null;
  return (
    <section className="mt-10 border-t border-hairline pt-7" aria-label="Headline numbers">
      <HeroStats
        insight={insight}
        lastNats={(isLiveRun ? live.lastEval?.overallNats : null) ?? run.lastHeldoutNats}
        prevBits={prevBits}
        chars={pulse?.chars ?? run.charsRead}
        cps={pulse?.readCps ?? (run.activeMs > 0 ? run.charsRead / (run.activeMs / 1000) : null)}
        activeMs={pulse?.activeMs ?? run.activeMs}
        goalMinutes={run.goal?.type === "minutes" ? run.goal.value : null}
      />
    </section>
  );
}

function ProgressCharts({ runId, live, version, advanced }: { runId: number; live: boolean; version: number; advanced: boolean }) {
  const domainsQ = useEvalDomains(runId, version);
  const domains = domainsQ.data ?? [];
  const events = useEvents(runId, version);

  const scoreKeys = useMemo(() => ["eval.overall", ...(domains.length > 1 ? domains.map((d) => `eval.domain.${d}`) : [])], [domains]);
  const score = useSeries({ runId, keys: scoreKeys, x: "chars", from: null, to: null, maxPoints: 800 }, live);
  const ticks = useSeries({ runId, keys: ["train.nats", "lr.scale", "speed.read_cps", "halt.avg_rows", "ctx.now", "grad.norm"], x: "chars", from: null, to: null, maxPoints: 800 }, live);

  const by = (list: SeriesData[] | undefined, key: string) => list?.find((s) => s.key === key);
  const annotations = useMemo<{ jumps: Annotation[]; all: Annotation[] }>(() => {
    const ev = events.data ?? [];
    const jumps = ev.filter((e) => e.kind === "plasticity_jump").map((e) => ({ x: e.chars, kind: "jump" as const, label: "Learning rate raised" }));
    const saves = ev.filter((e) => e.kind === "checkpoint").map((e) => ({ x: e.chars, kind: "checkpoint" as const, label: "Progress saved" }));
    return { jumps, all: [...jumps, ...saves] };
  }, [events.data]);

  const scoreLines: ChartLine[] = [];
  const overall = by(score.data, "eval.overall");
  if (domains.length > 1) {
    domains.forEach((d) => {
      const s = by(score.data, `eval.domain.${d}`);
      scoreLines.push({ key: `domain.${d}`, name: d, color: laneColor(laneSlot(d, domains)), points: true, ...toBits(s), lo: undefined, hi: undefined });
    });
    scoreLines.push({ key: "overall", name: "Overall", color: "var(--ink)", points: true, ...toBits(overall), lo: undefined, hi: undefined });
  } else {
    scoreLines.push({ key: "overall", name: "Test score", color: "var(--lane-1)", points: true, ...toBits(overall) });
  }
  scoreLines.push({ key: "train", name: "Training score", color: "var(--muted)", dashed: true, faint: true, ...toBits(by(ticks.data, "train.nats")), lo: undefined, hi: undefined });

  const loading = (score.isFetching && !score.isLoading) || (ticks.isFetching && !ticks.isLoading);

  return (
    <div className="space-y-9">
      <TimeSeriesChart
        title={CHART_INFO.score.title}
        chart="score"
        caption="Bits per character. Lower is better."
        lines={scoreLines}
        yFormat={(v) => v.toFixed(1)}
        group="progress"
        height={280}
        annotations={annotations.all}
        loading={loading}
      />
      <div className="grid gap-x-10 gap-y-9 md:grid-cols-2">
        <TimeSeriesChart
          title={CHART_INFO.lr.title}
          chart="lr"
          caption="Times the starting rate."
          lines={[{ key: "lr", name: "Learning rate", color: "var(--lane-1)", ...plain(by(ticks.data, "lr.scale")), lo: undefined, hi: undefined }]}
          yFormat={(v) => `${v.toFixed(2)}×`}
          yMin={0}
          yMax={1.05}
          group="progress"
          annotations={annotations.jumps}
          loading={loading}
        />
        <TimeSeriesChart
          title={CHART_INFO.speed.title}
          chart="speed"
          caption="Characters per second."
          lines={[{ key: "speed", name: "Reading speed", color: "var(--lane-3)", ...plain(by(ticks.data, "speed.read_cps")) }]}
          yFormat={(v) => fmtCount(v)}
          group="progress"
          loading={loading}
        />
        <TimeSeriesChart
          title={CHART_INFO.depth.title}
          chart="depth"
          caption="Thinking steps per character."
          lines={[{ key: "rows", name: "Average steps", color: "var(--lane-7)", ...plain(by(ticks.data, "halt.avg_rows")), lo: undefined, hi: undefined }]}
          yFormat={(v) => v.toFixed(2)}
          group="progress"
          loading={loading}
        />
        {advanced && (
          <TimeSeriesChart
            title={CHART_INFO.context.title}
            chart="context"
            caption="Characters it can look back over."
            lines={[{ key: "ctx", name: "Context window", color: "var(--lane-2)", ...plain(by(ticks.data, "ctx.now")), lo: undefined, hi: undefined }]}
            yFormat={(v) => fmtCount(v)}
            group="progress"
            loading={loading}
          />
        )}
      </div>
    </div>
  );
}

function ExpertsTab({ runId, version }: { runId: number; version: number }) {
  const live = useLive();
  const pool = usePool(runId, version);
  return <ExpertsPanel pool={(live.runId === runId ? live.pool : null) ?? pool.data ?? null} />;
}

function ThinkingTab({ runId, version, live }: { runId: number; version: number; live: boolean }) {
  const store = useLive();
  const pool = usePool(runId, version);
  const hist = ((store.runId === runId ? store.pool : null) ?? pool.data)?.haltingHist;
  const depth = useSeries({ runId, keys: ["halt.avg_rows"], x: "chars", from: null, to: null, maxPoints: 800 }, live);
  const s = depth.data?.[0];
  return (
    <div className="grid gap-x-12 gap-y-9 md:grid-cols-2">
      <div>
        <h3 className="m-0 mb-1 text-[15px] font-semibold">{CHART_INFO.halting.title}</h3>
        <p className="mb-4 max-w-[48ch] text-[13px] text-ink-2">{CHART_INFO.halting.how}</p>
        <HaltingBars hist={hist} />
      </div>
      <TimeSeriesChart
        title={CHART_INFO.depth.title}
        chart="depth"
        caption="Average steps per character."
        lines={[{ key: "rows", name: "Average steps", color: "var(--lane-7)", ...plain(s) }]}
        yFormat={(v) => v.toFixed(2)}
        group="thinking"
      />
    </div>
  );
}

function SystemTab({ run, version }: { run: RunSummary; version: number }) {
  const hw = useHardware();
  const live = useLive();
  const ckpts = useCheckpoints(run.id, version);
  const warnings = live.runId === run.id ? live.warnings : [];
  const b = hw.data?.backends.find((x) => x.kind === hw.data?.selected);
  return (
    <div className="grid gap-x-12 gap-y-9 md:grid-cols-2">
      <div>
        <h3 className="m-0 text-[15px] font-semibold">This computer</h3>
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-[13.5px]">
          <dt className="text-ink-2">Processor</dt><dd className="m-0 text-right">{hw.data?.cpu ?? "…"}</dd>
          <dt className="text-ink-2">Memory</dt><dd className="m-0 text-right">{hw.data ? `${hw.data.ramGb.toFixed(0)} GB` : "…"}</dd>
          <dt className="text-ink-2">Training on</dt><dd className="m-0 text-right">{b?.name ?? "…"}</dd>
          <dt className="text-ink-2">Memory for training</dt><dd className="m-0 text-right">{b ? `${b.memBudgetGb.toFixed(0)} GB` : "…"}</dd>
          <dt className="text-ink-2">Free disk space</dt><dd className="m-0 text-right">{hw.data ? `${hw.data.diskFreeGb.toFixed(0)} GB` : "…"}</dd>
        </dl>
        {warnings.length > 0 && (
          <div className="mt-6">
            <h3 className="m-0 text-[15px] font-semibold">Warnings</h3>
            <ul className="m-0 mt-2 list-none space-y-2 p-0 text-[13.5px]">
              {warnings.map((w) => (
                <li key={w.at + w.code}>
                  <span className="font-medium">{w.message}</span>
                  {w.hint && <span className="text-ink-2"> {w.hint}</span>}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div>
        <h3 className="m-0 text-[15px] font-semibold">Saved progress</h3>
        {(ckpts.data?.length ?? 0) === 0 ? (
          <p className="mt-2 text-[13.5px] text-muted">Nothing saved yet. It saves automatically every few minutes.</p>
        ) : (
          <table className="mt-3 w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="text-left text-ink-2">
                <th className="py-1.5 pr-3 font-medium">Saved</th>
                <th className="py-1.5 pr-3 font-medium">Text read</th>
                <th className="py-1.5 pr-3 text-right font-medium">Score</th>
              </tr>
            </thead>
            <tbody>
              {ckpts.data!.slice(0, 8).map((c) => (
                <tr key={c.id} className="border-t border-hairline">
                  <td className="py-1.5 pr-3">{fmtWhen(c.createdAt)}</td>
                  <td className="py-1.5 pr-3">{fmtCount(c.chars)}</td>
                  <td className="py-1.5 pr-3 text-right">{fmtBits(natsToBits(c.heldoutNats))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
