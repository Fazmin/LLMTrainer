import { Dialog } from "radix-ui";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { backend } from "../backend";
import type { Goal, RunSummary } from "../bindings";
import { describeError } from "../content/errors";
import { Button } from "./Button";

const GOALS: { id: string; label: string; goal: Goal }[] = [
  { id: "stop", label: "Until I stop it", goal: { type: "until_stopped" } },
  { id: "10", label: "10 minutes", goal: { type: "minutes", value: 10 } },
  { id: "30", label: "30 minutes", goal: { type: "minutes", value: 30 } },
  { id: "60", label: "1 hour", goal: { type: "minutes", value: 60 } },
  { id: "180", label: "3 hours", goal: { type: "minutes", value: 180 } },
];

const RATES = [
  { id: "same", label: "As before", mult: 1 },
  { id: "slower", label: "Slower (half as fast)", mult: 0.5 },
  { id: "faster", label: "Faster (twice as fast)", mult: 2 },
];

/**
 * Continue a saved model as a new run. The model itself (its size and everything it has learned) carries over; what the
 * person can change is what it reads next and how it learns. The original run is never touched.
 */
export function ForkDialog({ run, open, onOpenChange }: { run: RunSummary; open: boolean; onOpenChange: (open: boolean) => void }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const config = useQuery({ queryKey: ["run-config", run.id], queryFn: () => backend.getRunConfig(run.id), enabled: open });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => backend.listDatasets(), enabled: open });
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState<number | null>(run.datasetId);
  const [rate, setRate] = useState("same");
  const [goalId, setGoalId] = useState("stop");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName(`${run.name} (continued)`);
      setDatasetId(run.datasetId);
      setRate("same");
      setGoalId("stop");
      setError(null);
    }
  }, [open, run.id, run.name, run.datasetId]);

  const ready = (datasets.data ?? []).filter((d) => d.status === "ready");

  const submit = async () => {
    if (!config.data) return;
    if (datasetId == null) {
      setError("Choose some text for it to read next.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const mult = RATES.find((r) => r.id === rate)?.mult ?? 1;
      const train = { ...config.data.train, lr: config.data.train.lr * mult };
      const goal = GOALS.find((g) => g.id === goalId)?.goal ?? null;
      const forked = await backend.forkRun({ runId: run.id, checkpointId: null, name, datasetId, train, goal });
      await qc.invalidateQueries({ queryKey: ["runs"] });
      onOpenChange(false);
      navigate(`/train/${forked.id}`);
      await backend.startRun(forked.id);
    } catch (e) {
      const d = describeError(e);
      setError(`${d.title}. ${d.body}`);
    } finally {
      setBusy(false);
    }
  };

  const select = "mt-1.5 block h-10 w-full rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-[15px]";
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 max-h-[90vh] w-[min(520px,92vw)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-[14px] border border-hairline bg-surface p-6 shadow-2xl">
          <Dialog.Title className="m-0 text-lg font-semibold">Continue as a new run</Dialog.Title>
          <Dialog.Description className="mt-2 text-sm leading-relaxed text-ink-2">
            The model carries on from its latest save, with everything it has learned. “{run.name}” stays exactly as it is, so you can compare the two.
          </Dialog.Description>
          <div className="mt-5 space-y-4">
            <label className="block text-sm font-medium">
              Name
              <input value={name} onChange={(e) => setName(e.target.value)} className={select} />
            </label>
            <label className="block text-sm font-medium">
              Text to read next
              <select value={datasetId ?? ""} onChange={(e) => setDatasetId(e.target.value === "" ? null : Number(e.target.value))} className={select}>
                {datasetId == null && <option value="">Choose…</option>}
                {ready.map((d) => <option key={d.id} value={d.id}>{d.id === run.datasetId ? `Same as before: ${d.name}` : d.name}</option>)}
              </select>
            </label>
            <label className="block text-sm font-medium">
              How fast it learns
              <select value={rate} onChange={(e) => setRate(e.target.value)} className={select}>
                {RATES.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
            </label>
            <label className="block text-sm font-medium">
              Train for
              <select value={goalId} onChange={(e) => setGoalId(e.target.value)} className={select}>
                {GOALS.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
              </select>
            </label>
          </div>
          {error && <p role="alert" className="mt-4 text-sm text-critical-ink">{error}</p>}
          <div className="mt-6 flex justify-end gap-2">
            <Dialog.Close asChild><Button>Cancel</Button></Dialog.Close>
            <Button variant="primary" disabled={busy || !config.data} onClick={() => void submit()}>Start</Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
