import type { Preset } from "../bindings";

/** What to expect from each size, in plain words. The numbers (time, memory) come from the live estimate. */
export const PRESET_COPY: Record<"tiny" | "small" | "full", { title: string; tagline: string; writes: string; for: string }> = {
  tiny: {
    title: "Tiny",
    tagline: "Your first results in about ten minutes.",
    writes: "Learns spelling and simple sentences.",
    for: "Trying things out, and small amounts of text.",
  },
  small: {
    title: "Small",
    tagline: "A few hours for noticeably better writing.",
    writes: "Writes short, mostly sensible passages.",
    for: "A bigger collection of text on a computer with a good GPU.",
  },
  full: {
    title: "Full",
    tagline: "The size of the original research model. Days, not hours.",
    writes: "The best quality, given enough text and time.",
    for: "A powerful GPU and a lot of patience.",
  },
};

export const isSizedPreset = (p: Preset): p is "tiny" | "small" | "full" => p === "tiny" || p === "small" || p === "full";
