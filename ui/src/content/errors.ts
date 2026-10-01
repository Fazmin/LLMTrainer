import type { AppError } from "../bindings";

export type ErrorCard = { title: string; body: string };

/** Turn any error from the backend into a message that says what happened and what to do next. */
export function describeError(e: unknown): ErrorCard {
  const err = e as Partial<AppError> | string | null;
  if (typeof err === "string") return { title: "Something went wrong", body: err };
  if (!err || typeof err !== "object" || !("kind" in err)) {
    return { title: "Something went wrong", body: e instanceof Error ? e.message : "An unexpected error occurred." };
  }
  switch (err.kind) {
    case "run_active":
      return { title: "A training is already running", body: "Stop it first, or wait for it to finish, before starting another." };
    case "no_active_run":
      return { title: "Nothing is running", body: "There is no active training to control." };
    case "dataset_not_ready":
      return { title: "Your text is not ready yet", body: "Wait for the text to finish preparing, then try again." };
    case "disk_full":
      return {
        title: "Not enough disk space",
        body: "Free up some space, or choose a smaller model, then try again.",
      };
    case "out_of_memory":
      return {
        title: "This model is too big for this computer's memory",
        body: err.detail?.suggestedPreset
          ? `Try the ${err.detail.suggestedPreset} size instead.`
          : "Try a smaller model size.",
      };
    case "backend_unavailable":
      return { title: "The graphics processor is not available", body: `${err.detail} Training will run on the CPU, which is much slower.` };
    case "network":
      return { title: "The download stopped", body: "Your progress is saved. Check your connection and try again." };
    case "checkpoint":
      return { title: "That saved model cannot be loaded", body: "It may be damaged or from an incompatible version. Try an earlier save." };
    case "not_found":
      return { title: "We could not find that", body: err.detail ?? "It may have been deleted." };
    case "invalid":
      return { title: "Please check your settings", body: err.detail ?? "One of the settings is not valid." };
    case "cancelled":
      return { title: "Cancelled", body: "Nothing was changed." };
    default:
      return { title: "Something went wrong", body: ("detail" in err && typeof err.detail === "string" ? err.detail : "An unexpected error occurred.") };
  }
}
