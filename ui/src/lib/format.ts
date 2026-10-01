/** Number and time formatting shared across the UI. All functions accept `null` (the engine's "not a number"). */

const LN2 = Math.LN2;

export const natsToBits = (nats: number | null | undefined): number | null =>
  nats == null || !Number.isFinite(nats) ? null : nats / LN2;

export const bitsToNats = (bits: number) => bits * LN2;

/** 12_400_000 -> "12.4M", 950 -> "950", 12_345 -> "12.3k". */
export function fmtCount(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "–";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(abs >= 1e11 ? 0 : 1)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(abs >= 1e8 ? 0 : 1)}M`;
  if (abs >= 1e4) return `${(n / 1e3).toFixed(abs >= 1e5 ? 0 : 1)}k`;
  return Math.round(n).toLocaleString("en-US");
}

/** A size in bytes as KB, MB or GB: 87_000_000 -> "87 MB". */
export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "–";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

/** Speed in characters per second: 9_040 -> "9.0k". */
export const fmtRate = (n: number | null | undefined) => (n == null ? "–" : fmtCount(n));

export function fmtBits(bits: number | null | undefined, digits = 2): string {
  return bits == null || !Number.isFinite(bits) ? "–" : bits.toFixed(digits);
}

/** Milliseconds as m:ss or h:mm:ss. */
export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "–";
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (v: number) => String(v).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** Seconds as a rough human phrase for time estimates: "about 22 minutes". */
export function fmtRoughTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  if (seconds < 45) return "under a minute";
  const min = seconds / 60;
  if (min < 90) return `about ${Math.max(1, Math.round(min))} minutes`;
  const h = min / 60;
  if (h < 36) return `about ${h < 10 ? h.toFixed(1).replace(/\.0$/, "") : Math.round(h)} hours`;
  return `about ${Math.round(h / 24)} days`;
}

/** A human-scale sense of how much text has been read. A novel is roughly half a million characters. */
export function readingGloss(chars: number | null | undefined): string {
  if (chars == null || chars <= 0) return "nothing yet";
  const novels = chars / 500_000;
  if (novels >= 2) return `about ${novels >= 10 ? Math.round(novels) : novels.toFixed(1).replace(/\.0$/, "")} novels`;
  const pages = chars / 2_000;
  if (pages >= 2) return `about ${Math.round(pages)} pages`;
  return "a few lines";
}

/**
 * Plain-language meaning of a held-out score: how many characters the model is, on average, "choosing between"
 * for each next character (perplexity). 5.6 nats (~8 bits) is the uniform-guess starting point.
 */
export function choicesGloss(nats: number | null | undefined): string {
  if (nats == null || !Number.isFinite(nats)) return "";
  const choices = Math.exp(nats);
  if (choices >= 100) return "It is still close to guessing at random.";
  return `On average it is choosing between about ${choices < 10 ? choices.toFixed(1) : Math.round(choices)} characters for each next one.`;
}

export function fmtPercent(v: number | null | undefined, digits = 0): string {
  return v == null || !Number.isFinite(v) ? "–" : `${v.toFixed(digits)}%`;
}

export function pluralize(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
}

/** Date as "Sep 30, 21:49". */
export function fmtWhen(ms: number | null | undefined): string {
  if (ms == null) return "–";
  const d = new Date(ms);
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

/** Lane colour slot (0..7) -> CSS variable. Lanes beyond the eighth fold into a neutral. */
export function laneColor(slot: number): string {
  return slot >= 0 && slot < 8 ? `var(--lane-${slot + 1})` : "var(--muted)";
}

const NAMED_SLOTS: Record<string, number> = {
  stories: 0,
  arithmetic: 1,
  wikipedia: 2,
  code: 3,
  chat: 4,
  reasoning: 5,
  chess: 6,
  "self-knowledge": 7,
};

/** Stable slot for a domain name: the original eight keep fixed slots, anything else takes the next free one. */
export function laneSlot(name: string, known: string[]): number {
  const fixed = NAMED_SLOTS[name.toLowerCase()];
  if (fixed !== undefined) return fixed;
  const taken = new Set(known.map((k) => NAMED_SLOTS[k.toLowerCase()]).filter((v) => v !== undefined));
  const others = known.filter((k) => NAMED_SLOTS[k.toLowerCase()] === undefined).sort();
  const free = [0, 1, 2, 3, 4, 5, 6, 7].filter((s) => !taken.has(s));
  const idx = others.indexOf(name);
  return idx >= 0 && idx < free.length ? free[idx]! : 8;
}

/** Whether a lane's samples read better in a monospace face. */
export const isCodeLike = (domain: string) => /code|arith|math|chess|json|log/i.test(domain);
