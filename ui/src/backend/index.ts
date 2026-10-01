import type { Backend } from "./types";
import { tauriBackend } from "./tauri";
import { fixtureBackend } from "./fixture";

/** Inside the desktop app the real commands exist; in a plain browser (dev preview, tests) a simulation stands in. */
export const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export const backend: Backend = isTauri ? tauriBackend : fixtureBackend;
export type { Backend } from "./types";
