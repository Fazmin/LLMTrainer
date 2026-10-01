import { useQuery } from "@tanstack/react-query";
import { FolderOpen } from "lucide-react";
import { Link } from "react-router";
import { backend } from "../backend";
import { Button } from "../components/Button";
import { fmtBytes } from "../lib/format";
import { usePrefs, type ThemeChoice } from "../state/prefs";
import { useAppInfo, useHardware } from "../state/queries";

const THEMES: { id: ThemeChoice; label: string }[] = [
  { id: "system", label: "Match my computer" },
  { id: "light", label: "Light" },
  { id: "dark", label: "Dark" },
];

function Row({ label, value, action }: { label: string; value: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-4 border-t border-hairline py-3 text-sm first:border-t-0">
      <span className="w-48 shrink-0 text-ink-2">{label}</span>
      <span className="min-w-0 flex-1 break-words">{value}</span>
      {action}
    </div>
  );
}

const open = (url: string) => (e: React.MouseEvent) => {
  e.preventDefault();
  void backend.openLink(url);
};

export function Settings() {
  const { theme, setTheme, level, setLevel } = usePrefs();
  const info = useAppInfo();
  const hw = useHardware();
  const usage = useQuery({ queryKey: ["storage"], queryFn: () => backend.storageUsage(), staleTime: 10_000 });
  const u = usage.data;
  const b = hw.data?.backends.find((x) => x.kind === hw.data?.selected);

  return (
    <div className="mx-auto max-w-[780px] px-8 py-10">
      <h1 className="m-0 text-[26px] font-semibold tracking-tight">Settings</h1>

      <section className="mt-9" aria-labelledby="look-h">
        <h2 id="look-h" className="m-0 text-lg font-semibold">How it looks</h2>
        <div role="radiogroup" aria-label="Theme" className="mt-3 flex flex-wrap gap-2">
          {THEMES.map((t) => (
            <label key={t.id} className={`cursor-pointer rounded-[var(--radius-control)] border px-4 py-2 text-sm font-medium ${theme === t.id ? "border-accent bg-accent-soft text-accent-ink" : "border-hairline bg-surface text-ink-2 hover:border-axis"}`}>
              <input type="radio" name="theme" className="sr-only" checked={theme === t.id} onChange={() => setTheme(t.id)} />
              {t.label}
            </label>
          ))}
        </div>
        <label className="mt-5 flex cursor-pointer items-start gap-3">
          <input type="checkbox" checked={level === "advanced"} onChange={(e) => setLevel(e.target.checked ? "advanced" : "beginner")} className="mt-1 accent-[var(--accent)]" />
          <span>
            <span className="block text-[15px] font-medium">Show advanced settings</span>
            <span className="block text-[13.5px] text-ink-2">Adds the learning rate, step size and other controls to Set up, and a few extra charts to Train.</span>
          </span>
        </label>
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="disk-h">
        <h2 id="disk-h" className="m-0 text-lg font-semibold">Disk space</h2>
        <p className="mt-1 text-sm text-ink-2">Everything the app keeps lives in one folder on this computer.</p>
        <div className="mt-4">
          <Row
            label="Folder"
            value={u?.dataDir ?? "…"}
            action={u && <Button size="sm" icon={<FolderOpen size={14} />} onClick={() => void backend.revealInFolder(u.dataDir)}>Show in folder</Button>}
          />
          <Row label="Text" value={u ? <>{fmtBytes(Number(u.datasetsBytes))} <Link to="/data" className="ml-2 text-accent-ink underline underline-offset-2">Manage</Link></> : "…"} />
          <Row label="Saved models" value={u ? <>{fmtBytes(Number(u.runsBytes))} <Link to="/runs" className="ml-2 text-accent-ink underline underline-offset-2">Manage</Link></> : "…"} />
          <Row label="Chats" value={u ? fmtBytes(Number(u.chatsBytes)) : "…"} />
          <Row label="History and charts" value={u ? fmtBytes(Number(u.databaseBytes)) : "…"} />
          <Row label="Free on this disk" value={u ? fmtBytes(Number(u.freeBytes)) : "…"} />
        </div>
        {u && u.perRun.length > 0 && (
          <details className="mt-4 text-sm">
            <summary className="cursor-pointer text-ink-2">Which trainings use the most</summary>
            <ul className="m-0 mt-2 list-none space-y-1 p-0">
              {u.perRun.slice(0, 8).map((r) => (
                <li key={r.runId} className="flex justify-between gap-4 border-t border-hairline py-1.5 first:border-t-0">
                  <Link to={`/train/${r.runId}`} className="underline-offset-2 hover:underline">{r.name}</Link>
                  <span className="text-ink-2">{fmtBytes(Number(r.bytes))}</span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="pc-h">
        <h2 id="pc-h" className="m-0 text-lg font-semibold">This computer</h2>
        <div className="mt-4">
          <Row label="Processor" value={hw.data?.cpu ?? "…"} />
          <Row label="Memory" value={hw.data ? `${hw.data.ramGb.toFixed(0)} GB` : "…"} />
          <Row label="Trains on" value={b ? `${b.name} (up to ${b.memBudgetGb.toFixed(0)} GB of memory)` : "…"} />
        </div>
      </section>

      <section className="mt-10 border-t border-hairline pt-8" aria-labelledby="about-h">
        <h2 id="about-h" className="m-0 text-lg font-semibold">About</h2>
        <div className="mt-4">
          <Row label="Version" value={info.data ? `LLM Trainer ${info.data.version}` : "…"} />
          <Row label="Engine" value={info.data ? (info.data.isMock ? "Simulated (numbers are not from a real model)" : info.data.engineVersion) : "…"} />
        </div>
        <p className="mt-5 max-w-[64ch] text-[13.5px] leading-relaxed text-ink-2">
          The model and the way it learns are a re-implementation of{" "}
          <a href="https://github.com/volotat/mini-AGI" onClick={open("https://github.com/volotat/mini-AGI")} className="text-accent-ink underline underline-offset-2">mini-AGI</a>{" "}
          by Alexey Borsky (MIT licence). The practice stories come from{" "}
          <a href="https://huggingface.co/datasets/roneneldan/TinyStories" onClick={open("https://huggingface.co/datasets/roneneldan/TinyStories")} className="text-accent-ink underline underline-offset-2">TinyStories</a>{" "}
          by Eldan and Li (CDLA-Sharing-1.0 licence), and are downloaded only when you ask.
        </p>
      </section>
    </div>
  );
}
