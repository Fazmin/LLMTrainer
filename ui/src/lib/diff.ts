/** Flatten nested configuration into dotted keys: `{ growth: { k: 1 } }` -> `{ "growth.k": 1 }`. */
export function flatten(value: unknown, prefix = ""): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      Object.assign(out, flatten(v, prefix ? `${prefix}.${k}` : k));
    }
  } else if (prefix) {
    out[prefix] = value;
  }
  return out;
}

export interface DiffRow {
  key: string;
  values: unknown[];
}

/** Keys whose value is not identical across all configurations, in a stable order. */
export function configDiff(configs: unknown[]): DiffRow[] {
  const flat = configs.map((c) => flatten(c));
  const keys = [...new Set(flat.flatMap((f) => Object.keys(f)))].sort();
  return keys
    .map((key) => ({ key, values: flat.map((f) => f[key]) }))
    .filter((row) => row.values.some((v) => JSON.stringify(v) !== JSON.stringify(row.values[0])));
}

const LABELS: Record<string, string> = {
  "train.lr": "Learning rate",
  "train.trunkLrMult": "Learning-rate multiplier for the core",
  "train.chunk": "Characters per step",
  "train.passage": "Characters per passage",
  "train.sampleEveryMin": "Minutes between progress checks",
  "train.saveEveryMin": "Minutes between saves",
  "train.contextStart": "Starting context window",
  "train.contextEnd": "Largest context window",
  "train.resident": "Experts loaded at once",
  "train.weightDecay": "Weight decay",
  "train.clip": "Gradient clip",
  "model.dModel": "Model width",
  "model.nHead": "Attention heads",
  "model.maxSteps": "Most thinking steps",
  "model.poolExperts": "Starting experts",
  "model.poolMax": "Most experts",
  "model.poolTopK": "Experts per character",
  "model.poolDFf": "Expert size",
  "model.dFf": "Core feed-forward width",
  "model.nPrelude": "Dense blocks before the thinking block",
  "model.minSteps": "Fewest thinking steps",
  "model.bpttWindow": "Steps remembered when learning",
  "model.trainStepsMean": "Average thinking steps while learning",
  "model.haltPrior": "Starting chance of stopping to think",
  "model.haltThresh": "Confidence needed to stop thinking",
  "model.ponderBeta": "Penalty for thinking too long",
  "model.block": "Longest context it can handle",
  "model.poolDepth": "Layers in each expert",
  "train.contextStep": "Context growth per step",
  "train.growth.everyChars": "Characters between growth checks",
  "train.growth.maxDiskGb": "Disk limit for experts (GB)",
  "train.prune.survivalChars": "Characters before an unused expert is removed",
  "train.moeMode": "How experts are computed",
  "train.rowCheckpointing": "Recompute to save memory",
  "train.capacityFactor": "Expert capacity",
  "train.evalChars": "Characters scored per progress check",
  "train.ramCache": "Experts kept in memory",
};

/** A readable name for a configuration key; unknown keys become spaced words. */
export function keyLabel(key: string): string {
  if (LABELS[key]) return LABELS[key];
  const last = key.split(".").pop() ?? key;
  const words = last.replace(/([a-z0-9])([A-Z])/g, "$1 $2").toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString("en-US") : String(Number(v.toPrecision(4)));
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v).replace(/_/g, " ");
}
