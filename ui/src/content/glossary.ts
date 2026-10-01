/** Short definitions for the words the interface cannot avoid. Shown in popovers wherever a term appears. */
export type TermKey =
  | "bits_per_char"
  | "held_out"
  | "experts"
  | "thinking_depth"
  | "learning_rate"
  | "checkpoint"
  | "lane"
  | "gap"
  | "context"
  | "growing";

export const GLOSSARY: Record<TermKey, { label: string; text: string }> = {
  bits_per_char: {
    label: "bits per character",
    text: "How surprised the model is by each next character in text it has never seen. Lower is better. Around 8 means pure guessing; a good small model reaches about 1.5.",
  },
  held_out: {
    label: "held-out text",
    text: "Text kept aside that the model never learns from. Scoring it on this text shows real learning rather than memorising.",
  },
  experts: {
    label: "experts",
    text: "Small specialist networks inside the model. For each character it picks a few of them. The pool can grow when a new one would help, and shrink when one is never used.",
  },
  thinking_depth: {
    label: "thinking depth",
    text: "The model can reuse one block several times per character, like thinking for longer. Harder characters get more steps.",
  },
  learning_rate: {
    label: "learning rate",
    text: "How big a step the model takes each time it learns from a mistake. The app raises and lowers it for you based on how the score is moving.",
  },
  checkpoint: {
    label: "checkpoint",
    text: "A saved copy of the model at one moment. You can continue training from it, chat with it, or export it.",
  },
  lane: {
    label: "lane",
    text: "A folder holding one kind of text. The model reads the lanes in turn, so it learns all of them instead of only the last.",
  },
  gap: {
    label: "gap",
    text: "The difference between the score on text it has not seen and on text it trains on. A big gap means it is memorising instead of learning.",
  },
  context: {
    label: "context window",
    text: "How many characters back the model can look when guessing the next one. It starts small and grows as the model improves.",
  },
  growing: {
    label: "growing",
    text: "Every so often the model may add a new expert, if five checks pass: there is room, experts are being used, newcomers are earning their place, not too many are on trial, and it is not just memorising.",
  },
};
