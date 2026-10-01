import { Check, FolderOpen, Share2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { backend } from "../backend";
import type { ExportResult } from "../bindings";
import { Button } from "../components/Button";
import { JobBar } from "../components/JobBar";
import { fmtBits, fmtBytes, fmtCount, fmtWhen, natsToBits } from "../lib/format";
import { useJobRunner } from "../state/jobs";
import { useCheckpoints, useRuns } from "../state/queries";

/** Copy a trained model into a folder you can keep, move to another computer, or share. */
export function Export() {
  const [params] = useSearchParams();
  const runs = useRuns();
  const { job, error, run, cancel } = useJobRunner();
  const choices = useMemo(() => (runs.data ?? []).filter((r) => r.charsRead > 0 && r.status !== "created"), [runs.data]);
  const [runId, setRunId] = useState<number | null>(null);
  const [saveId, setSaveId] = useState<number | "best">("best");
  const [done, setDone] = useState<ExportResult | null>(null);

  useEffect(() => {
    if (runId != null || choices.length === 0) return;
    const wanted = Number(params.get("run"));
    setRunId(choices.find((c) => c.id === wanted)?.id ?? choices[0]!.id);
  }, [choices, params, runId]);

  const saves = useCheckpoints(runId, 0);
  const list = saves.data ?? [];
  const chosen = saveId === "best" ? (list.find((c) => c.isBest) ?? list[0]) : list.find((c) => c.id === saveId);

  const exportNow = async (kind: "portable" | "safetensors") => {
    if (runId == null) return;
    const dest = await backend.pickFolder("Where should the model be saved?");
    if (!dest) return;
    setDone(null);
    const req = { runId, checkpointId: saveId === "best" ? null : saveId, destDir: dest };
    const result = kind === "portable"
      ? await run("Copying the saved model", (cb) => backend.exportModel(req, cb))
      : await run("Writing the model for other tools", (cb) => backend.exportSafetensors(req, cb));
    if (result) setDone(result);
  };

  return (
    <div className="mx-auto max-w-[760px] px-8 py-10">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">Export a model</h1>
      <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-ink-2">
        Copy a trained model into a folder you can keep, move to another computer, or give to someone else. They can bring it in with Import on the Runs page.
      </p>

      {choices.length === 0 ? (
        <p className="mt-10 text-[15px] text-ink-2">There is nothing to export yet. Train a model and let it save its progress first.</p>
      ) : (
        <>
          <section className="mt-9">
            <label className="block">
              <span className="text-lg font-semibold">Which model?</span>
              <select value={runId ?? ""} onChange={(e) => { setRunId(Number(e.target.value)); setSaveId("best"); setDone(null); }} className="mt-2 block h-10 w-full max-w-[460px] rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-[15px]">
                {choices.map((r) => <option key={r.id} value={r.id}>{r.name} ({fmtCount(r.charsRead)} characters read)</option>)}
              </select>
            </label>
          </section>

          <section className="mt-8">
            <label className="block">
              <span className="text-lg font-semibold">Which saved version?</span>
              <select value={saveId} onChange={(e) => setSaveId(e.target.value === "best" ? "best" : Number(e.target.value))} className="mt-2 block h-10 w-full max-w-[460px] rounded-[var(--radius-control)] border border-hairline bg-surface px-3 text-[15px]">
                <option value="best">The best one (recommended)</option>
                {list.map((c) => <option key={c.id} value={c.id}>{fmtWhen(c.createdAt)}, {fmtCount(c.chars)} characters, score {fmtBits(natsToBits(c.heldoutNats))}</option>)}
              </select>
            </label>
            {chosen && <p className="mt-2 text-[13.5px] text-ink-2">This one is about {fmtBytes(Number(chosen.bytes))} and scored {fmtBits(natsToBits(chosen.heldoutNats))} bits per character.</p>}
          </section>

          <section className="mt-8 rounded-[var(--radius-panel)] border border-hairline bg-surface p-5">
            <h2 className="m-0 text-[15px] font-semibold">What you get</h2>
            <ul className="m-0 mt-2 list-disc space-y-1 pl-5 text-[14px] text-ink-2">
              <li>The saved model, exactly as it is.</li>
              <li>A short description file, so the app knows what it is.</li>
              <li>A plain-English note about the model.</li>
            </ul>
          </section>

          {job && <div className="mt-6"><JobBar job={job} onCancel={cancel} /></div>}
          {error && (
            <div role="alert" className="mt-6 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-3">
              <p className="m-0 font-semibold">{error.title}</p>
              <p className="mt-1 text-sm text-ink-2">{error.body}</p>
            </div>
          )}
          {done && (
            <div role="status" className="mt-6 rounded-[var(--radius-panel)] border border-hairline bg-surface p-5">
              <p className="m-0 flex items-center gap-2 text-[15px] font-semibold"><Check size={17} className="text-good-ink" aria-hidden /> Saved</p>
              <p className="mt-1 break-all text-[13.5px] text-ink-2">{done.path} ({fmtBytes(Number(done.bytes))})</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button size="sm" icon={<FolderOpen size={14} />} onClick={() => void backend.revealInFolder(done.path)}>Show in folder</Button>
                {runId != null && <Link to={`/chat?run=${runId}`} className="inline-flex h-8 items-center rounded-[var(--radius-control)] px-3 text-[13px] text-accent-ink underline underline-offset-2">Chat with it</Link>}
              </div>
            </div>
          )}

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Button variant="primary" size="lg" icon={<Share2 size={16} />} disabled={!!job || runId == null || list.length === 0} onClick={() => void exportNow("portable")}>
              Choose where to save…
            </Button>
            <Button size="lg" disabled={!!job || runId == null || list.length === 0} onClick={() => void exportNow("safetensors")}>
              Save for other tools…
            </Button>
          </div>
          <p className="mt-3 max-w-[62ch] text-[13px] leading-relaxed text-muted">
            “Save for other tools” writes the weights as a single safetensors file, which programmers can open with common libraries. LLM Trainer cannot import that kind of file; use the first button to move a model between computers.
          </p>
          {list.length === 0 && runId != null && <p className="mt-2 text-[13px] text-muted">This run has not saved any progress yet.</p>}
        </>
      )}
    </div>
  );
}
