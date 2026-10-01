import type { AdviceKey, Verdict } from "../bindings";

/** The "Is it learning?" answer. Colour is never the only signal: each verdict has an icon name and a label. */
export const VERDICTS: Record<Verdict, { label: string; headline: string; tone: "neutral" | "good" | "warning" | "critical"; icon: "clock" | "trending-down" | "minus" | "copy" | "alert" }> = {
  warming_up: {
    label: "Too early to tell",
    headline: "It needs a few more checks before we can say.",
    tone: "neutral",
    icon: "clock",
  },
  learning: {
    label: "Yes, it is learning",
    headline: "Its score on new text keeps getting better.",
    tone: "good",
    icon: "trending-down",
  },
  plateau: {
    label: "Progress has slowed",
    headline: "The score has stopped improving for now.",
    tone: "warning",
    icon: "minus",
  },
  overfitting: {
    label: "It is memorising",
    headline: "It does much better on text it trained on than on new text.",
    tone: "warning",
    icon: "copy",
  },
  diverging: {
    label: "Something went wrong",
    headline: "The score jumped the wrong way.",
    tone: "critical",
    icon: "alert",
  },
};

export const ADVICE: Record<AdviceKey, string> = {
  keep_going: "Keep going.",
  wait_for_first_check: "The first check comes after a few minutes of reading.",
  read_more_text: "Give it time; slow stretches are normal. It may also need more text.",
  add_more_varied_text: "Add more, and more varied, text.",
  stop_and_keep_best: "Stop here and keep the best saved version.",
  resume_from_best: "Stop, then continue from the last good save.",
  lower_learning_rate: "Try a lower learning rate next time.",
  try_tiny_model: "Try a smaller model, which memorises less.",
};

/** Where the model is on the road from random guessing to fluent writing. Keys match the engine's milestones. */
export const MILESTONES: Record<string, string> = {
  random_guessing: "Guessing at random",
  letter_frequencies: "Learning which letters are common",
  common_words: "Spelling common words",
  short_phrases: "Writing short phrases",
  simple_sentences: "Writing simple sentences",
  fluent_sentences: "Writing fluent sentences",
};
