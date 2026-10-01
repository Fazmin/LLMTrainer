import clsx from "clsx";
import { Activity, Database, History, MessageSquare, Monitor, Moon, Settings as SettingsIcon, Share2, SlidersHorizontal, Sun, TriangleAlert } from "lucide-react";
import { NavLink } from "react-router";
import { useAppInfo, useHardware } from "../state/queries";
import { useLive } from "../state/live";
import { usePrefs, type ThemeChoice } from "../state/prefs";

const NAV = [
  { to: "/data", label: "Text", icon: Database },
  { to: "/setup", label: "Set up", icon: SlidersHorizontal },
  { to: "/train", label: "Train", icon: Activity },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/export", label: "Export", icon: Share2 },
  { to: "/runs", label: "Runs", icon: History },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
] as const;

export function Sidebar() {
  const info = useAppInfo();
  const hw = useHardware();
  const live = useLive((s) => s.runState);
  const { theme, setTheme, level, setLevel } = usePrefs();
  const backend = hw.data?.backends.find((b) => b.kind === hw.data?.selected);
  const cpuOnly = hw.data?.selected === "cpu";
  const running = live === "running" || live === "preparing" || live === "paused";

  return (
    <aside className="flex h-full w-[216px] shrink-0 flex-col border-r border-hairline bg-surface px-3 py-5">
      <div className="flex items-center gap-2.5 px-2 pb-6">
        <img src="/app-icon.png" alt="" width={28} height={28} className="rounded-[7px]" />
        <span className="text-[15px] font-semibold tracking-tight">LLM Trainer</span>
      </div>

      <nav aria-label="Main" className="flex flex-col gap-0.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-2.5 rounded-[var(--radius-control)] px-2.5 py-2 text-sm font-medium transition-colors",
                isActive ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink",
              )
            }
          >
            <Icon size={17} strokeWidth={1.9} />
            {label}
            {to === "/train" && running && <span aria-label="A training is running" className="pulse-dot ml-auto h-2 w-2 rounded-full bg-accent" />}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto flex flex-col gap-3 px-1">
        {info.data?.isMock && (
          <p className="m-0 rounded-md border border-critical/40 px-2.5 py-2 text-[12px] leading-snug text-ink-2" title="The numbers on screen come from a simulation, not a real model.">
            <strong className="font-semibold text-critical-ink">Simulated.</strong> These numbers are not from a real model.
          </p>
        )}
        <p className={clsx("m-0 flex items-start gap-2 text-[12.5px] leading-snug", cpuOnly ? "text-ink" : "text-ink-2")}>
          {cpuOnly && <TriangleAlert size={14} className="mt-0.5 shrink-0 text-serious" aria-hidden />}
          <span>
            {cpuOnly ? "No GPU found. Training runs on the CPU and is much slower." : `Training on ${backend?.name ?? "…"}`}
          </span>
        </p>

        <div className="flex items-center justify-between">
          <div role="group" aria-label="Theme" className="flex rounded-md border border-hairline p-0.5">
            {(["system", "light", "dark"] as ThemeChoice[]).map((t) => {
              const Icon = t === "system" ? Monitor : t === "light" ? Sun : Moon;
              return (
                <button key={t} aria-label={`${t} theme`} aria-pressed={theme === t} onClick={() => setTheme(t)} className={clsx("rounded p-1.5", theme === t ? "bg-surface-2 text-ink" : "text-muted hover:text-ink")}>
                  <Icon size={14} />
                </button>
              );
            })}
          </div>
          <label className="flex cursor-pointer items-center gap-1.5 text-[12.5px] text-ink-2">
            <input type="checkbox" checked={level === "advanced"} onChange={(e) => setLevel(e.target.checked ? "advanced" : "beginner")} className="accent-[var(--accent)]" />
            Advanced
          </label>
        </div>
      </div>
    </aside>
  );
}
