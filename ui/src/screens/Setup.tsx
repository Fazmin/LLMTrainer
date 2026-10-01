import clsx from "clsx";
import { AlertTriangle, Check, Play } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { backend } from "../backend";
import type { Goal, Preset, PresetConfig, TrainConfig } from "../bindings";
import { Button } from "../components/Button";
import { Term } from "../components/explain";
import { describeError } from "../content/errors";
import { PRESET_COPY } from "../content/presets";
import { fmtCount } from "../lib/format";
import { useLive } from "../state/live";
import { usePrefs } from "../state/prefs";
import { useDatasets, useEstimate, useHardware, usePresets, useRecommendation } from "../state/queries";

type SizedPreset = "tiny" | "small" | "full";

const GOALS: { id: string; label: string; goal: Goal }[] = [
  { id: "10", label: "10 minutes", goal: { type: "minutes", value: 10 } },
  { id: "30", label: "30 minutes", goal: { type: "minutes", value: 30 } },
  { id: "60", label: "1 hour", goal: { type: "minutes", value: 60 } },
  { id: "stop", label: "Until I stop it", goal: { type: "until_stopped" } },
];

function PresetOption({ cfg, datasetId, selected, recommended, onSelect }: { cfg: PresetConfig; datasetId: number | null; selected: boolean; recommended: boolean; onSelect: () => void }) {
  const est = useEstimate(cfg.model, cfg.train, datasetId);
  const copy = PRESET_COPY[cfg.preset as SizedPreset];
  const e = est.data;
  const blocked = e?.fit === "wont_fit";
  return (
    <label
      className={clsx(
        "relative flex cursor-pointer flex-col rounded-[var(--radius-panel)] border bg-surface p-4 transition-colors",
        selected ? "border-accent ring-1 ring-accent" : "border-hairline hover:border-axis",
        blocked && "cursor-not-allowed opacity-60",
      )}
    >
      <input type="radio" name="preset" value={cfg.preset} checked={selected} disabled={blocked} onChange={onSelect} className="sr-only" />
      <span className="flex items-center gap-2">
        <span className="text-base font-semibold">{copy.title}</span>
        {recommended && <span className="rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent-ink">Recommended</span>}
        {selected && <Check size={16} className="ml-auto text-accent" aria-label="Selected" />}
      </span>
      <span className="mt-1.5 text-[13.5px] leading-snug">{copy.tagline}</span>
      <span className="mt-1 text-[13px] leading-snug text-ink-2">{copy.writes}</span>
      <span className="mt-1 text-[13px] leading-snug text-muted">Good for: {copy.for}</span>
      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 border-t border-hairline pt-3 text-[13px]">
        <dt className="text-ink-2">Speed</dt>
        <dd className="m-0 text-right">{e ? `${fmtCount(e.charsPerSec)} characters/sec` : "…"}</dd>
        <dt className="text-ink-2">Memory</dt>
        <dd className="m-0 text-right">{e ? `${e.gpuGb < 10 ? e.gpuGb.toFixed(1) : Math.round(e.gpuGb)} GB` : "…"}</dd>
        <dt className="text-ink-2">Disk</dt>
        <dd className="m-0 text-right">{e ? `up to ${e.diskGb < 10 ? e.diskGb.toFixed(1) : Math.round(e.diskGb)} GB` : "…"}</dd>
      </dl>
      {e?.fit === "tight" && <p className="mt-2 text-[12.5px] text-warning-ink">Close to this computer's memory limit.</p>}
      {blocked && <p className="mt-2 text-[12.5px] text-critical-ink">{e?.notes[0] ?? "This computer cannot run this size."}</p>}
    </label>
  );
}

function NumberField({ label, help, value, onChange, step, min }: { label: React.ReactNode; help: string; value: number; onChange: (v: number) => void; step?: number; min?: number }) {
  return (
    <label className="block">
      <span className="text-sm font-medium">{label}</span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : ""}
        step={step}
        min={min}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 block h-9 w-full rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-sm"
      />
      <span className="mt-1 block text-[12.5px] leading-snug text-muted">{help}</span>
    </label>
  );
}

export function Setup() {
  const navigate = useNavigate();
  const presets = usePresets();
  const hw = useHardware();
  const datasets = useDatasets();
  const [params] = useSearchParams();
  const [pickedDataset, setPickedDataset] = useState<number | null>(null);
  const ready = (datasets.data ?? []).filter((d) => d.status === "ready");
  const wanted = Number(params.get("dataset"));
  const datasetId = pickedDataset ?? (ready.find((d) => d.id === wanted)?.id ?? ready[0]?.id ?? null);
  const dataset = ready.find((d) => d.id === datasetId) ?? null;
  const rec = useRecommendation(datasetId);
  const level = usePrefs((s) => s.level);
  const activeRun = useLive((s) => (s.runState === "running" || s.runState === "paused" || s.runState === "preparing" ? s.runId : null));

  const [chosen, setChosen] = useState<SizedPreset | null>(null);
  const [goalId, setGoalId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [edits, setEdits] = useState<Partial<TrainConfig>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ title: string; body: string } | null>(null);

  const selected = (chosen ?? (rec.data?.preset as SizedPreset | undefined) ?? "tiny") as SizedPreset;
  const cfg = presets.data?.find((p) => p.preset === selected);
  const goal = GOALS.find((g) => g.id === (goalId ?? (selected === "tiny" ? "10" : "60")))!;
  const train = useMemo(() => (cfg ? { ...cfg.train, ...edits } : undefined), [cfg, edits]);
  const est = useEstimate(cfg?.model, train, datasetId);
  const edited = Object.keys(edits).length > 0;
  const cpuOnly = hw.data?.selected === "cpu";
  const blocked = est.data?.fit === "wont_fit";

  const start = async () => {
    if (!cfg || !train) return;
    setBusy(true);
    setError(null);
    try {
      const run = await backend.createRun({
        name: name.trim() || null,
        preset: (edited ? "custom" : selected) as Preset,
        model: cfg.model,
        train,
        datasetId,
        goal: goal.goal,
      });
      await backend.startRun(run.id);
      navigate(`/train/${run.id}`);
    } catch (e) {
      setError(describeError(e));
      setBusy(false);
    }
  };

  const checks = [
    { ok: !cpuOnly, text: cpuOnly ? "No GPU found. It will run on the CPU, which is much slower." : `Graphics processor: ${hw.data?.backends.find((b) => b.kind === hw.data?.selected)?.name ?? "…"}` },
    { ok: est.data ? est.data.fit !== "wont_fit" : true, text: est.data ? `Memory: needs about ${est.data.gpuGb.toFixed(1)} GB of ${hw.data?.backends.find((b) => b.kind === hw.data?.selected)?.memBudgetGb.toFixed(0) ?? "…"} GB available` : "Memory: checking…" },
    { ok: hw.data ? hw.data.diskFreeGb > (est.data?.diskGb ?? 1) * 1.2 : true, text: hw.data ? `Disk space: ${hw.data.diskFreeGb.toFixed(0)} GB free` : "Disk space: checking…" },
    dataset
      ? { ok: true, text: `Training text: ${dataset.name} (${(Number(dataset.trainBytes) / 1e6).toFixed(1)} MB)` }
      : { ok: false, text: "Training text: none yet. Add some on the Text page." },
  ];

  return (
    <div className="mx-auto max-w-[940px] px-8 py-10">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">Set up a training</h1>
      <p className="mt-2 max-w-[60ch] text-[15px] leading-relaxed text-ink-2">
        Pick a size, say how long it should train, and press start. You can stop whenever you like and continue later.
      </p>

      {activeRun != null && (
        <p className="mt-5 flex items-center gap-2 rounded-[var(--radius-control)] border border-hairline bg-surface px-3.5 py-2.5 text-sm">
          <AlertTriangle size={16} className="text-warning-ink" aria-hidden />
          A training is already running. <Link to={`/train/${activeRun}`} className="font-medium text-accent-ink underline underline-offset-2">Open it</Link>, or stop it before starting another.
        </p>
      )}

      <section className="mt-9" aria-labelledby="text-h">
        <h2 id="text-h" className="m-0 text-lg font-semibold">What should it read?</h2>
        {ready.length === 0 ? (
          <div className="mt-3 rounded-[var(--radius-panel)] border border-hairline bg-surface p-5">
            <p className="m-0 text-[15px] font-medium">You need some text to learn from first.</p>
            <p className="mt-1 text-sm text-ink-2">Pick a ready-made set (about a minute) or add a folder of your own.</p>
            <Button className="mt-4" variant="primary" onClick={() => navigate("/data")}>Get some text</Button>
          </div>
        ) : (
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
            <select
              value={datasetId ?? ""}
              onChange={(e) => { setPickedDataset(Number(e.target.value)); setEdits({}); }}
              aria-label="Training text"
              className="h-10 min-w-[260px] rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-[15px]"
            >
              {ready.map((d) => <option key={d.id} value={d.id}>{d.name} ({(Number(d.trainBytes) / 1e6).toFixed(1)} MB)</option>)}
            </select>
            <Link to="/data" className="text-sm text-accent-ink underline underline-offset-2">Add or change text</Link>
          </div>
        )}
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="size-h">
        <h2 id="size-h" className="m-0 text-lg font-semibold">How big a model?</h2>
        <p className="mt-1 text-sm text-ink-2">Bigger models write better but need more memory and time. The numbers below are for this computer.</p>
        <div role="radiogroup" aria-labelledby="size-h" className="mt-4 grid gap-4 md:grid-cols-3">
          {(presets.data ?? []).map((p) => (
            <PresetOption key={p.preset} cfg={p} datasetId={datasetId} selected={selected === p.preset} recommended={rec.data?.preset === p.preset} onSelect={() => { setChosen(p.preset as SizedPreset); setEdits({}); }} />
          ))}
        </div>
        {rec.data && rec.data.preset === selected && rec.data.reasons[0] && <p className="mt-3 text-[13.5px] text-ink-2">{rec.data.reasons[0]}</p>}
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="goal-h">
        <h2 id="goal-h" className="m-0 text-lg font-semibold">How long should it train?</h2>
        <div role="radiogroup" aria-labelledby="goal-h" className="mt-3 flex flex-wrap gap-2">
          {GOALS.map((g) => (
            <label key={g.id} className={clsx("cursor-pointer rounded-[var(--radius-control)] border px-4 py-2 text-sm font-medium transition-colors", g.id === goal.id ? "border-accent bg-accent-soft text-accent-ink" : "border-hairline bg-surface text-ink-2 hover:border-axis")}>
              <input type="radio" name="goal" className="sr-only" checked={g.id === goal.id} onChange={() => setGoalId(g.id)} />
              {g.label}
            </label>
          ))}
        </div>
        <p className="mt-3 max-w-[60ch] text-[13.5px] text-ink-2">
          It saves its progress as it goes, so even a short run is not wasted. You can <Term k="checkpoint">continue from a save</Term> any time.
        </p>
      </section>

      <section className="mt-10 border-t border-hairline pt-8">
        <label className="block max-w-[420px]">
          <span className="text-lg font-semibold">Name this training</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={`${PRESET_COPY[selected].title} run`}
            className="mt-2 block h-10 w-full rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-[15px]"
          />
          <span className="mt-1 block text-[13px] text-muted">Optional. It helps you find this run later.</span>
        </label>
      </section>

      {level === "advanced" && train && (
        <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="adv-h">
          <h2 id="adv-h" className="m-0 text-lg font-semibold">Advanced settings</h2>
          <p className="mt-1 text-sm text-ink-2">The defaults suit most people. Changing these makes this a custom run.</p>
          <div className="mt-4 grid gap-5 sm:grid-cols-2">
            <NumberField label={<Term k="learning_rate">Learning rate</Term>} help="How big a step it takes when it learns. The app adjusts it as it goes." value={train.lr} step={0.0001} min={0} onChange={(v) => setEdits((e) => ({ ...e, lr: v }))} />
            <NumberField label="Characters per step" help="How much text it reads before each round of learning." value={train.chunk} step={64} min={64} onChange={(v) => setEdits((e) => ({ ...e, chunk: v }))} />
            <NumberField label="Minutes between progress checks" help="How often it is tested on new text and asked to write a sample." value={train.sampleEveryMin} step={0.5} min={0.1} onChange={(v) => setEdits((e) => ({ ...e, sampleEveryMin: v }))} />
            <NumberField label="Minutes between saves" help="How often progress is written to disk." value={train.saveEveryMin} step={0.5} min={0.1} onChange={(v) => setEdits((e) => ({ ...e, saveEveryMin: v }))} />
          </div>
        </section>
      )}

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="pre-h">
        <h2 id="pre-h" className="m-0 text-lg font-semibold">Before you start</h2>
        <ul className="m-0 mt-3 list-none space-y-2 p-0">
          {checks.map((c) => (
            <li key={c.text} className="flex items-start gap-2.5 text-[14px]">
              {c.ok ? <Check size={16} className="mt-0.5 shrink-0 text-good-ink" aria-label="Fine" /> : <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warning-ink" aria-label="Needs attention" />}
              <span>{c.text}</span>
            </li>
          ))}
        </ul>

        {error && (
          <div role="alert" className="mt-5 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-3">
            <p className="m-0 font-semibold">{error.title}</p>
            <p className="mt-1 text-sm text-ink-2">{error.body}</p>
          </div>
        )}

        <div className="mt-6 flex items-center gap-4">
          <Button variant="primary" size="lg" icon={<Play size={16} />} disabled={busy || !cfg || blocked || activeRun != null || datasetId == null} onClick={start}>
            {busy ? "Starting…" : "Start training"}
          </Button>
          <span className="text-[13px] text-muted">{goal.goal.type === "until_stopped" ? "It keeps going until you stop it." : `It stops by itself after ${goal.label}.`}</span>
        </div>
      </section>
    </div>
  );
}
