/** "What is this?" text for each chart. Every chart in the dashboard has one. */
export const CHART_INFO = {
  score: {
    title: "Test score",
    what: "How well the model predicts text it has never trained on, in bits per character. Lower is better.",
    how: "A falling line means it is learning. The pale line is how it does on text it is training on; if the two pull far apart, it is memorising.",
  },
  lr: {
    title: "Learning rate",
    what: "How big a step the model takes when it learns. 1× is the starting rate.",
    how: "The app lowers it as learning slows and raises it again if progress stalls. Triangles mark those raises.",
  },
  speed: {
    title: "Reading speed",
    what: "How many characters the model reads each second.",
    how: "It should stay fairly steady. Dips usually mean your computer is busy with something else.",
  },
  depth: {
    title: "Thinking depth",
    what: "The average number of thinking steps the model takes for each character.",
    how: "It usually starts low and rises as the model learns that some characters are worth more thought.",
  },
  context: {
    title: "Context window",
    what: "How many characters back the model can look.",
    how: "It grows in small steps once the model has learned enough to use more.",
  },
  halting: {
    title: "How long it thinks",
    what: "How many characters take 1 step, 2 steps, and so on.",
    how: "Taller bars further right mean more characters get extra thought.",
  },
} as const;

export type ChartId = keyof typeof CHART_INFO;
