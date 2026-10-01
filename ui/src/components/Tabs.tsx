import clsx from "clsx";
import type { ReactNode } from "react";

export interface TabDef<T extends string> {
  id: T;
  label: string;
  icon?: ReactNode;
}

/** Underlined tab strip. Keyboard accessible: arrow keys move between tabs. */
export function TabStrip<T extends string>({ tabs, value, onChange }: { tabs: TabDef<T>[]; value: T; onChange: (id: T) => void }) {
  return (
    <div role="tablist" className="flex gap-1 border-b border-hairline" onKeyDown={(e) => {
      const i = tabs.findIndex((t) => t.id === value);
      if (e.key === "ArrowRight") onChange(tabs[(i + 1) % tabs.length]!.id);
      if (e.key === "ArrowLeft") onChange(tabs[(i - 1 + tabs.length) % tabs.length]!.id);
    }}>
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === value}
          tabIndex={t.id === value ? 0 : -1}
          onClick={() => onChange(t.id)}
          className={clsx(
            "-mb-px inline-flex items-center gap-2 border-b-2 px-3.5 py-2.5 text-sm font-medium transition-colors",
            t.id === value ? "border-accent text-ink" : "border-transparent text-ink-2 hover:text-ink",
          )}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );
}
