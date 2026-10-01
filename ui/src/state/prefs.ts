import { create } from "zustand";

export type ThemeChoice = "system" | "light" | "dark";
export type Level = "beginner" | "advanced";

interface Prefs {
  theme: ThemeChoice;
  level: Level;
  /** Bumped when the resolved colours change, so canvas charts can re-read their tokens. */
  themeVersion: number;
  setTheme: (t: ThemeChoice) => void;
  setLevel: (l: Level) => void;
}

const KEY = "llm-trainer:prefs";

// localStorage can be missing or throw (private windows, blocked storage): preferences are a convenience only.
function load(): Partial<Pick<Prefs, "theme" | "level">> {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Partial<Pick<Prefs, "theme" | "level">>) : {};
  } catch {
    return {};
  }
}

function save(p: Pick<Prefs, "theme" | "level">) {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* ignore */
  }
}

export function applyTheme(theme: ThemeChoice) {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}

const initial = load();

export const usePrefs = create<Prefs>((set, get) => ({
  theme: initial.theme ?? "system",
  level: initial.level ?? "beginner",
  themeVersion: 0,
  setTheme: (theme) => {
    applyTheme(theme);
    set({ theme, themeVersion: get().themeVersion + 1 });
    save({ theme, level: get().level });
  },
  setLevel: (level) => {
    set({ level });
    save({ theme: get().theme, level });
  },
}));

/** Apply the saved theme and follow OS changes while the choice is "system". Call once at startup. */
export function initTheme() {
  applyTheme(usePrefs.getState().theme);
  const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
  mq?.addEventListener("change", () => usePrefs.setState((s) => ({ themeVersion: s.themeVersion + 1 })));
}
