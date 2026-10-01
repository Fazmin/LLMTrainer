import clsx from "clsx";
import { Check, Download, FolderOpen, Play, Trash2 } from "lucide-react";
import { Dialog } from "radix-ui";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { backend } from "../backend";
import type { DatasetDetail, DatasetSummary, StarterInfo } from "../bindings";
import { Button } from "../components/Button";
import { Term } from "../components/explain";
import { JobBar } from "../components/JobBar";
import { fmtBytes, fmtCount, isCodeLike, laneColor } from "../lib/format";
import { useJobRunner } from "../state/jobs";
import { useDataset, useDatasets, useSplit, useStarters } from "../state/queries";

const mb = fmtBytes;

const KIND_LABEL: Record<DatasetSummary["kind"], string> = {
  starter: "Ready-made",
  bundled: "Ready-made",
  generated: "Made on this computer",
  linked: "Your own text",
};

function StarterCard({ info, dataset, busy, anyBusy, onGet }: { info: StarterInfo; dataset?: DatasetSummary; busy: boolean; anyBusy: boolean; onGet: () => void }) {
  const ready = dataset?.status === "ready";
  return (
    <div className="flex flex-col rounded-[var(--radius-panel)] border border-hairline bg-surface p-4">
      <div className="flex items-baseline gap-2">
        <h3 className="m-0 text-[15px] font-semibold">{info.title}</h3>
        {info.id === "tinystories-quick" && <span className="shrink-0 whitespace-nowrap rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent-ink">Start here</span>}
      </div>
      <p className="mt-1.5 flex-1 text-[13.5px] leading-snug text-ink-2">{info.description}</p>
      <p className="mt-3 text-[13px] text-muted">{info.offline ? "No internet needed. " : "Needs internet once. "}About {mb(Number(info.sizeBytes))}.</p>
      <div className="mt-3">
        {ready ? (
          <p className="m-0 flex items-center gap-1.5 text-[13.5px] font-medium text-good-ink"><Check size={15} aria-hidden /> Ready</p>
        ) : (
          <Button size="sm" variant={info.id === "tinystories-quick" ? "primary" : "secondary"} icon={<Download size={14} />} disabled={anyBusy} onClick={onGet}>
            {busy ? "Getting it…" : info.offline ? "Get it" : "Download"}
          </Button>
        )}
      </div>
    </div>
  );
}

function LaneTable({ detail, onToggle }: { detail: DatasetDetail; onToggle: (lane: string, enabled: boolean) => void }) {
  const total = Math.max(1, detail.lanes.filter((l) => l.enabled).reduce((a, l) => a + l.trainBytes, 0));
  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="text-left text-[13px] text-ink-2">
          <th className="w-8 py-2 pr-2"><span className="sr-only">Read this</span></th>
          <th className="py-2 pr-4 font-medium"><Term k="lane">Kind of text</Term></th>
          <th className="py-2 pr-4 text-right font-medium">Files</th>
          <th className="py-2 pr-4 text-right font-medium">Size</th>
          <th className="w-[28%] py-2 font-medium">Share of what it reads</th>
        </tr>
      </thead>
      <tbody>
        {detail.lanes.map((l) => (
          <tr key={l.name} className={clsx("border-t border-hairline", !l.enabled && "opacity-55")}>
            <td className="py-3 pr-2">
              <input type="checkbox" checked={l.enabled} aria-label={`Read ${l.displayName}`} onChange={(e) => onToggle(l.name, e.target.checked)} className="accent-[var(--accent)]" />
            </td>
            <td className="py-3 pr-4">
              <span className="inline-flex items-center gap-2 font-medium">
                <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: laneColor(l.colorSlot) }} />
                {l.displayName}
              </span>
            </td>
            <td className="py-3 pr-4 text-right text-ink-2">{fmtCount(Number(l.nFiles))}</td>
            <td className="py-3 pr-4 text-right">{mb(Number(l.trainBytes))}</td>
            <td className="py-3">
              <div className="h-2 overflow-hidden rounded-full bg-surface-2" aria-hidden>
                <div className="h-full rounded-full" style={{ width: l.enabled ? `${(Number(l.trainBytes) / total) * 100}%` : "0%", background: laneColor(l.colorSlot) }} />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Peek({ datasetId, lanes }: { datasetId: number; lanes: string[] }) {
  const [lane, setLane] = useState<string>(lanes[0] ?? "");
  const [seed, setSeed] = useState(1);
  const [text, setText] = useState<{ file: string; text: string } | null>(null);
  useEffect(() => setLane((l) => (lanes.includes(l) ? l : (lanes[0] ?? ""))), [lanes]);
  useEffect(() => {
    let cancelled = false;
    if (!lane) return;
    backend.previewText(datasetId, lane, 700, seed).then((p) => !cancelled && setText({ file: p.file, text: p.text })).catch(() => !cancelled && setText(null));
    return () => {
      cancelled = true;
    };
  }, [datasetId, lane, seed]);
  if (!lane) return null;
  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="m-0 text-[15px] font-semibold">A peek at what it will read</h3>
        {lanes.length > 1 && (
          <select value={lane} onChange={(e) => setLane(e.target.value)} aria-label="Kind of text to look at" className="h-8 rounded-[var(--radius-control)] border border-hairline bg-surface px-2 text-[13px]">
            {lanes.map((l) => <option key={l}>{l}</option>)}
          </select>
        )}
        <Button size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>Show another</Button>
      </div>
      {text && (
        <>
          <p className={clsx("model-text mt-3 max-h-48 overflow-y-auto rounded-[var(--radius-control)] bg-surface-2 px-4 py-3 text-ink", isCodeLike(lane) ? "font-mono text-[13px]" : "font-serif text-[16px] leading-relaxed")}>{text.text}</p>
          <p className="mt-1.5 text-xs text-muted">From {text.file}</p>
        </>
      )}
    </div>
  );
}

export function Data() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const datasets = useDatasets();
  const starters = useStarters();
  const { job, error, run, cancel, clearError } = useJobRunner();
  const [selected, setSelected] = useState<number | null>(null);
  const [startingId, setStartingId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pct, setPct] = useState<number | null>(null);

  const list = datasets.data ?? [];
  const current = selected != null && list.some((d) => d.id === selected) ? selected : (list[0]?.id ?? null);
  const detail = useDataset(current);
  const split = useSplit(current);

  useEffect(() => setPct(null), [current]);

  const choose = (d: DatasetDetail | null) => d && setSelected(d.summary.id);

  const addPaths = async (paths: string[]) => {
    if (paths.length === 0) return;
    clearError();
    choose(await run("Adding your text", (cb) => backend.addFolders(paths, "auto", cb)));
  };

  useEffect(
    () =>
      backend.watchDrops((e) => {
        if (e.type === "enter") setDragging(true);
        else if (e.type === "leave") setDragging(false);
        else {
          setDragging(false);
          void addPaths(e.paths);
        }
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const get = async (info: StarterInfo) => {
    setStartingId(info.id);
    choose(await run(`Getting ${info.title}`, (cb) => backend.installStarter(info.id, cb)));
    setStartingId(null);
  };

  const pick = async () => {
    const paths = await backend.pickFolders();
    if (paths) await addPaths(paths);
  };

  const toggle = async (lane: string, enabled: boolean) => {
    if (current == null) return;
    await backend.updateLane(current, lane, { enabled });
    void qc.invalidateQueries({ queryKey: ["dataset", current] });
  };

  const applySplit = async () => {
    if (current == null || pct == null || !split.data) return;
    await run("Setting some text aside for testing", (cb) => backend.rebuildDataset(current, { ...split.data!, pct }, cb));
    setPct(null);
    void qc.invalidateQueries({ queryKey: ["split", current] });
  };

  const remove = async () => {
    if (current == null) return;
    await backend.deleteDataset(current, true);
    setConfirmDelete(false);
    setSelected(null);
    void qc.invalidateQueries({ queryKey: ["datasets"] });
  };

  const d = detail.data;
  const enabledLanes = useMemo(() => (d?.lanes ?? []).filter((l) => l.enabled).map((l) => l.name), [d]);
  const skippedTotal = d ? d.skipped.binary + d.skipped.empty + d.skipped.utf16 + d.skipped.tooLarge + d.skipped.unreadable : 0;
  const pctShown = pct ?? split.data?.pct ?? 2;

  return (
    <div className="relative mx-auto max-w-[1060px] px-8 py-10">
      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-accent-soft/80">
          <p className="rounded-[var(--radius-panel)] border-2 border-dashed border-accent bg-surface px-10 py-8 text-xl font-semibold">Drop to add this text</p>
        </div>
      )}

      <h1 className="m-0 text-[26px] font-semibold tracking-tight">Your text</h1>
      <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-ink-2">
        This is what the model reads to learn. Start with one of ours, or add a folder of your own. Everything stays on this computer.
      </p>

      {job && <div className="mt-6"><JobBar job={job} onCancel={cancel} /></div>}
      {error && (
        <div role="alert" className="mt-6 rounded-[var(--radius-control)] border border-critical/50 bg-surface px-4 py-3">
          <p className="m-0 font-semibold">{error.title}</p>
          <p className="mt-1 text-sm text-ink-2">{error.body}</p>
        </div>
      )}

      <section className="mt-9" aria-labelledby="start-h">
        <h2 id="start-h" className="m-0 text-lg font-semibold">Start with something ready</h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {(starters.data ?? []).filter((s) => s.id !== "tinystories-full").map((s) => (
            <StarterCard key={s.id} info={s} dataset={list.find((x) => x.starterId === s.id)} busy={startingId === s.id} anyBusy={!!job} onGet={() => get(s)} />
          ))}
        </div>
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="own-h">
        <h2 id="own-h" className="m-0 text-lg font-semibold">Use your own text</h2>
        <div className="mt-4 flex flex-col items-center rounded-[var(--radius-panel)] border-2 border-dashed border-axis px-6 py-9 text-center">
          <FolderOpen size={26} className="text-muted" aria-hidden />
          <p className="mt-3 text-[15px] font-medium">Drop folders or files here</p>
          <p className="mt-1 max-w-[52ch] text-[13.5px] text-ink-2">
            Books, notes, code, anything written in plain text. Each top-level folder becomes its own kind of text, so the model learns all of them.
          </p>
          <Button className="mt-4" icon={<FolderOpen size={15} />} disabled={!!job} onClick={pick}>Choose folders…</Button>
        </div>
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="mine-h">
        <h2 id="mine-h" className="m-0 text-lg font-semibold">What you have</h2>
        {list.length === 0 ? (
          <p className="mt-3 text-[14px] text-ink-2">Nothing yet. Pick a ready-made set above, or add your own folder.</p>
        ) : (
          <div className="mt-4 grid gap-x-10 gap-y-6 lg:grid-cols-[250px_minmax(0,1fr)]">
            <ul className="m-0 flex list-none flex-row flex-wrap gap-1.5 p-0 lg:flex-col" aria-label="Your text collections">
              {list.map((x) => (
                <li key={x.id}>
                  <button
                    onClick={() => setSelected(x.id)}
                    aria-current={x.id === current}
                    className={clsx("w-full rounded-[var(--radius-control)] px-3 py-2 text-left transition-colors", x.id === current ? "bg-surface-2" : "hover:bg-surface")}
                  >
                    <span className="block text-sm font-medium">{x.name}</span>
                    <span className="block text-[12.5px] text-ink-2">
                      {KIND_LABEL[x.kind]}, {mb(Number(x.trainBytes))}
                      {x.status !== "ready" && <span className={x.status === "error" ? " text-critical-ink" : ""}> ({x.status === "error" ? "problem" : "working…"})</span>}
                    </span>
                  </button>
                </li>
              ))}
            </ul>

            {d && (
              <div className="min-w-0 space-y-8">
                <div className="flex flex-wrap items-start gap-4">
                  <div className="min-w-0 flex-1">
                    <h3 className="m-0 text-xl font-semibold">{d.summary.name}</h3>
                    <p className="mt-1 text-[13.5px] text-ink-2">
                      {KIND_LABEL[d.summary.kind]}. {mb(Number(d.summary.trainBytes))} to learn from, {mb(Number(d.summary.valBytes))} kept aside to test it.
                    </p>
                    {d.summary.attribution && <p className="mt-1 text-xs text-muted">{d.summary.attribution}{d.summary.license ? `, licence ${d.summary.license}` : ""}.</p>}
                  </div>
                  {d.summary.status === "ready" && (
                    <Button variant="primary" icon={<Play size={15} />} disabled={enabledLanes.length === 0} onClick={() => navigate(`/setup?dataset=${d.summary.id}`)}>
                      Train on this text
                    </Button>
                  )}
                </div>

                {d.summary.status === "error" && d.summary.error && (
                  <p role="alert" className="m-0 text-[14px] text-critical-ink">{d.summary.error}</p>
                )}

                {d.lanes.length > 0 && <LaneTable detail={d} onToggle={toggle} />}
                {d.lanes.length > 0 && enabledLanes.length === 0 && <p className="m-0 text-[13.5px] text-critical-ink">Switch on at least one kind of text to train.</p>}

                {d.summary.kind === "linked" && d.summary.status === "ready" && split.data && (
                  <div>
                    <h3 className="m-0 text-[15px] font-semibold">Keep some aside for testing</h3>
                    <p className="mt-1 max-w-[60ch] text-[13px] text-ink-2">
                      This share is never learned from. It is what <Term k="held_out">tests the model</Term> on text it has not seen. A few percent is plenty.
                    </p>
                    <div className="mt-3 flex items-center gap-4">
                      <input type="range" min={0.5} max={10} step={0.5} value={pctShown} onChange={(e) => setPct(Number(e.target.value))} aria-label="Share kept for testing" aria-valuetext={`${pctShown} percent`} className="h-1.5 w-56 accent-[var(--accent)]" />
                      <span className="w-12 text-sm font-medium">{pctShown}%</span>
                      {pct != null && pct !== split.data.pct && <Button size="sm" onClick={applySplit} disabled={!!job}>Apply</Button>}
                    </div>
                  </div>
                )}

                {d.summary.status === "ready" && enabledLanes.length > 0 && <Peek datasetId={d.summary.id} lanes={enabledLanes} />}

                {(skippedTotal > 0 || d.warnings.length > 0) && (
                  <details className="text-[13.5px]">
                    <summary className="cursor-pointer text-ink-2">
                      {skippedTotal > 0 && <>{skippedTotal} files were left out because they are not plain text.</>}
                      {skippedTotal === 0 && <>{d.warnings.length} note{d.warnings.length === 1 ? "" : "s"} about this text.</>}
                    </summary>
                    <ul className="mt-2 list-disc pl-5 text-ink-2">
                      {d.warnings.map((w) => <li key={w}>{w}</li>)}
                      {d.skipped.examples.slice(0, 8).map((ex) => <li key={ex.path}><span className="font-mono text-[12.5px]">{ex.path}</span>: {ex.reason}</li>)}
                    </ul>
                  </details>
                )}

                <div>
                  <Button variant="danger" size="sm" icon={<Trash2 size={14} />} disabled={!!job} onClick={() => setConfirmDelete(true)}>Remove this text</Button>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      <p className="mt-12 text-[13px] text-muted">Starting fresh? <Link to="/setup" className="text-accent-ink underline underline-offset-2">Go to set up</Link> once you have some text.</p>

      <Dialog.Root open={confirmDelete} onOpenChange={setConfirmDelete}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/40" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-[min(440px,92vw)] -translate-x-1/2 -translate-y-1/2 rounded-[14px] border border-hairline bg-surface p-6 shadow-2xl">
            <Dialog.Title className="m-0 text-lg font-semibold">Remove “{d?.summary.name}”?</Dialog.Title>
            <Dialog.Description className="mt-2 text-sm leading-relaxed text-ink-2">
              The prepared copy is deleted from this computer. Your original folders are not touched, and past trainings keep their history.
            </Dialog.Description>
            <div className="mt-6 flex justify-end gap-2">
              <Dialog.Close asChild><Button>Keep it</Button></Dialog.Close>
              <Button variant="danger" onClick={remove}>Remove</Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
