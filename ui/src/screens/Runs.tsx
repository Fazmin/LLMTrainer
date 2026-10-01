import clsx from "clsx";
import { Download, GitCompare, Info, TriangleAlert, Trash2 } from "lucide-react";
import { Dialog } from "radix-ui";
import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { backend } from "../backend";
import type { ImportPreview, RunState, RunSummary } from "../bindings";
import { Button } from "../components/Button";
import { RUN_STATES } from "../content/stages";
import { describeError } from "../content/errors";
import { fmtBits, fmtCount, fmtDuration, fmtWhen, natsToBits } from "../lib/format";
import { JobBar } from "../components/JobBar";
import { useJobRunner } from "../state/jobs";
import { useRuns } from "../state/queries";

const TONE: Record<RunState, string> = {
  created: "text-ink-2",
  preparing: "text-accent-ink",
  running: "text-accent-ink",
  paused: "text-warning-ink",
  stopping: "text-accent-ink",
  completed: "text-good-ink",
  stopped: "text-ink-2",
  failed: "text-critical-ink",
  interrupted: "text-critical-ink",
  imported: "text-ink-2",
};

function StatusChip({ state }: { state: RunState }) {
  return <span className={clsx("inline-flex items-center gap-1.5 text-[13px] font-medium", TONE[state])}>
    <span aria-hidden className="h-2 w-2 rounded-full bg-current" />
    {RUN_STATES[state].title}
  </span>;
}

export function Runs() {
  const runs = useRuns();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [toDelete, setToDelete] = useState<RunSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const importer = useJobRunner();
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);
  const toggle = (id: number) =>
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : p.length < 6 ? [...p, id] : p));

  const remove = async (run: RunSummary) => {
    try {
      await backend.deleteRun(run.id, true);
      await qc.invalidateQueries({ queryKey: ["runs"] });
      setToDelete(null);
    } catch (e) {
      const d = describeError(e);
      setError(`${d.title}. ${d.body}`);
      setToDelete(null);
    }
  };

  return (
    <div className="mx-auto max-w-[1060px] px-8 py-10">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">Your runs</h1>
      <p className="mt-2 max-w-[60ch] text-[15px] leading-relaxed text-ink-2">Every training you have started. Open one to see how it went, or continue it from where it stopped.</p>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          icon={<Download size={14} />}
          disabled={!!importer.job}
          onClick={async () => {
            setPickError(null);
            const folder = await backend.pickFolder("Choose a saved model folder");
            if (!folder) return;
            try {
              setPreview(await backend.previewImport(folder));
            } catch (e) {
              const d = describeError(e);
              setPickError(`${d.title}. ${d.body}`);
            }
          }}
        >
          Import a model…
        </Button>
        <span className="text-[13px] text-muted">From a folder made by Export, or the weights folder of the original mini-AGI program.</span>
      </div>
      {importer.job && <div className="mt-4"><JobBar job={importer.job} onCancel={importer.cancel} /></div>}
      {importer.error && (
        <div role="alert" className="mt-4 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-3">
          <p className="m-0 font-semibold">{importer.error.title}</p>
          <p className="mt-1 text-sm text-ink-2">{importer.error.body}</p>
        </div>
      )}
      {error && <p role="alert" className="mt-4 text-sm text-critical-ink">{error}</p>}
      {pickError && <p role="alert" className="mt-4 text-sm text-critical-ink">{pickError}</p>}
      {(runs.data?.length ?? 0) > 1 && (
        <div className="mt-6 flex items-center gap-3">
          <Button
            variant={picked.length >= 2 ? "primary" : "secondary"}
            size="sm"
            icon={<GitCompare size={15} />}
            disabled={picked.length < 2}
            onClick={() => navigate(`/compare?ids=${picked.join(",")}`)}
          >
            Compare {picked.length >= 2 ? `${picked.length} runs` : "runs"}
          </Button>
          <span className="text-[13px] text-muted">{picked.length < 2 ? "Tick two or more runs to compare them." : "Up to six at a time."}</span>
        </div>
      )}

      {runs.data && runs.data.length === 0 ? (
        <div className="mt-16 text-center">
          <p className="text-[15px] text-ink-2">No runs yet.</p>
          <Button className="mt-4" variant="primary" onClick={() => navigate("/setup")}>Set up a training</Button>
        </div>
      ) : (
        <table className="mt-8 w-full border-collapse text-sm">
          <thead>
            <tr className="text-left text-[13px] text-ink-2">
              <th className="w-8 py-2 pr-2"><span className="sr-only">Select to compare</span></th>
              <th className="py-2 pr-4 font-medium">Name</th>
              <th className="py-2 pr-4 font-medium">Status</th>
              <th className="py-2 pr-4 font-medium">Size</th>
              <th className="py-2 pr-4 text-right font-medium">Text read</th>
              <th className="py-2 pr-4 text-right font-medium">Best score</th>
              <th className="py-2 pr-4 text-right font-medium">Trained for</th>
              <th className="py-2 pr-4 font-medium">Started</th>
              <th className="w-10" />
            </tr>
          </thead>
          <tbody>
            {(runs.data ?? []).map((r) => (
              <tr key={r.id} className="border-t border-hairline hover:bg-surface">
                <td className="py-3 pr-2">
                  <input type="checkbox" aria-label={`Select ${r.name} to compare`} checked={picked.includes(r.id)} onChange={() => toggle(r.id)} className="accent-[var(--accent)]" />
                </td>
                <td className="py-3 pr-4">
                  <Link to={`/train/${r.id}`} className="font-medium text-ink underline-offset-2 hover:underline">{r.name}</Link>
                </td>
                <td className="py-3 pr-4"><StatusChip state={r.status} /></td>
                <td className="py-3 pr-4 text-ink-2">{r.preset[0]!.toUpperCase() + r.preset.slice(1)}</td>
                <td className="py-3 pr-4 text-right">{fmtCount(r.charsRead)}</td>
                <td className="py-3 pr-4 text-right">{fmtBits(natsToBits(r.bestHeldoutNats))}</td>
                <td className="py-3 pr-4 text-right">{fmtDuration(r.activeMs)}</td>
                <td className="py-3 pr-4 text-ink-2">{fmtWhen(r.startedAt ?? r.createdAt)}</td>
                <td className="py-3 text-right">
                  <button aria-label={`Delete ${r.name}`} disabled={r.status === "running" || r.status === "paused" || r.status === "preparing"} onClick={() => setToDelete(r)} className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-critical-ink disabled:opacity-30">
                    <Trash2 size={15} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <Dialog.Root open={preview != null} onOpenChange={(o) => !o && setPreview(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/40" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-[min(560px,92vw)] -translate-x-1/2 -translate-y-1/2 rounded-[14px] border border-hairline bg-surface p-6 shadow-2xl">
            <Dialog.Title className="m-0 text-lg font-semibold">{preview?.importable ? "Import this model?" : "This model cannot be imported"}</Dialog.Title>
            <Dialog.Description className="mt-2 text-sm leading-relaxed text-ink-2">{preview?.summary}</Dialog.Description>
            {preview && preview.kind === "python" && preview.importable && (
              <p className="mt-3 text-[13px] leading-relaxed text-ink-2">
                It was made by the original program. It is converted into a copy that this app can chat with and keep training; the folder you picked is not changed.
              </p>
            )}
            {preview && preview.issues.length > 0 && (
              <ul className="mt-4 space-y-2 p-0">
                {preview.issues.map((i, k) => (
                  <li key={k} className="flex list-none gap-2 text-[13.5px] leading-snug">
                    {i.severity === "note"
                      ? <Info size={16} className="mt-0.5 shrink-0 text-muted" aria-label="Note" />
                      : <TriangleAlert size={16} className={clsx("mt-0.5 shrink-0", i.severity === "blocker" ? "text-critical-ink" : "text-warning-ink")} aria-label={i.severity === "blocker" ? "Problem" : "Warning"} />}
                    <span>{i.message}</span>
                  </li>
                ))}
              </ul>
            )}
            <div className="mt-6 flex justify-end gap-2">
              <Dialog.Close asChild><Button>{preview?.importable ? "Cancel" : "Close"}</Button></Dialog.Close>
              {preview?.importable && (
                <Button
                  variant="primary"
                  onClick={async () => {
                    const folder = preview.path;
                    setPreview(null);
                    const run = await importer.run(preview.kind === "python" ? "Converting the original model" : "Copying the model in", (cb) => backend.importModel(folder, cb));
                    if (run) {
                      await qc.invalidateQueries({ queryKey: ["runs"] });
                      navigate(`/train/${run.id}`);
                    }
                  }}
                >
                  Import
                </Button>
              )}
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root open={toDelete != null} onOpenChange={(o) => !o && setToDelete(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/40" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-[min(440px,92vw)] -translate-x-1/2 -translate-y-1/2 rounded-[14px] border border-hairline bg-surface p-6 shadow-2xl">
            <Dialog.Title className="m-0 text-lg font-semibold">Delete “{toDelete?.name}”?</Dialog.Title>
            <Dialog.Description className="mt-2 text-sm leading-relaxed text-ink-2">
              Its history and its saved model are removed from this computer. This cannot be undone.
            </Dialog.Description>
            <div className="mt-6 flex justify-end gap-2">
              <Dialog.Close asChild><Button>Keep it</Button></Dialog.Close>
              <Button variant="danger" onClick={() => toDelete && remove(toDelete)}>Delete</Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
