import type { RunState, Stage } from "../bindings";

/** Plain-English copy for every stage the engine reports. The banner shows `title` and `blurb`. */
export const STAGES: Record<Stage, { title: string; blurb: string }> = {
  idle: { title: "Ready", blurb: "Nothing is running. Start a training from Setup." },
  preparing_data: {
    title: "Getting your text ready",
    blurb: "Finding the text files and checking that they can be read.",
  },
  creating_model: {
    title: "Building the model",
    blurb: "Setting up a new model with random starting values. At this point it knows nothing.",
  },
  loading_checkpoint: { title: "Loading saved progress", blurb: "Picking up where you left off." },
  reading: {
    title: "Reading",
    blurb: "It reads a chunk of text, guesses each next character, and learns from its mistakes. Then it moves on to the next chunk.",
  },
  evaluating: {
    title: "Checking its progress",
    blurb: "Testing it on text it has never trained on. This is the score that tells you whether it is really learning.",
  },
  sampling: {
    title: "Writing a sample",
    blurb: "Asking it to continue a few prompts, so you can see how it writes right now.",
  },
  grow_prune: {
    title: "Tuning its experts",
    blurb: "Deciding whether to add a new expert or retire one that nothing uses.",
  },
  checkpointing: {
    title: "Saving progress",
    blurb: "Writing everything to disk so you can stop and continue later.",
  },
  paused: { title: "Paused", blurb: "Nothing is being read right now. Resume when you are ready." },
  stopping: { title: "Stopping", blurb: "Finishing the current step and saving your progress." },
  finished: { title: "Finished", blurb: "The run has ended. Your progress is saved." },
  failed: { title: "Stopped by a problem", blurb: "Something went wrong. Your last saved progress is safe." },
};

/** Copy for a run that is not live (history view, or before it starts). */
export const RUN_STATES: Record<RunState, { title: string; blurb: string }> = {
  created: { title: "Ready to start", blurb: "This run has been set up but has not started reading yet." },
  preparing: { title: "Getting ready", blurb: "Preparing to read." },
  running: { title: "Reading", blurb: STAGES.reading.blurb },
  paused: { title: "Paused", blurb: STAGES.paused.blurb },
  stopping: { title: "Stopping", blurb: STAGES.stopping.blurb },
  completed: { title: "Finished", blurb: "The goal was reached and the model was saved." },
  stopped: { title: "Stopped", blurb: "You stopped this run. Its progress is saved, and you can continue it." },
  failed: { title: "Stopped by a problem", blurb: "Something went wrong. Your last saved progress is safe." },
  interrupted: {
    title: "Interrupted",
    blurb: "The app closed while this run was active. You can continue from the last saved progress.",
  },
  imported: { title: "Imported", blurb: "This model was imported. You can chat with it or continue training it." },
};
