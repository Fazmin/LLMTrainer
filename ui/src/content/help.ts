/**
 * "What is this?" text for every section of every screen, shown by the info button beside its heading. Written for
 * someone who has never trained a model: `what` says what the section shows, `how` says how to read or use it.
 */
export const SECTION_INFO = {
  // ── sidebar ────────────────────────────────────────────────────────────────────────────────────────────────────
  "sidebar.hardware": {
    title: "Where training runs",
    what: "The part of your computer that does the training. A graphics processor (GPU) does this kind of maths many times faster than the main processor (CPU).",
    how: "If no GPU was found, training still works, only more slowly. The Tiny size is the best choice then.",
  },

  // ── Train ──────────────────────────────────────────────────────────────────────────────────────────────────────
  "train.stage": {
    title: "What's happening now",
    what: "What the training is doing at this moment. It changes as the model reads, checks its progress, writes a sample and saves.",
    how: "The coloured edge shows the state at a glance: blue is working, yellow is paused, green is finished and red means a problem stopped it. Pausing or stopping keeps your progress.",
  },
  "train.writing": {
    title: "What it writes",
    what: "After each progress check, the model is given the start of a passage and asked to carry on. This is its own writing, unedited.",
    how: "The small grey text is the opening it was given; the large text is what it wrote. Drag the slider back to watch it go from gibberish to letters, words and sentences.",
  },
  "train.guard": {
    title: "Repetition guard",
    what: "Small models easily get stuck writing the same few words over and over. The guard nudges it away from repeating itself.",
    how: "Leave it on to see its best writing. Turn it off to see exactly what the model writes on its own.",
  },
  "train.verdict": {
    title: "Is it learning?",
    what: "A plain answer worked out from the test score, so you do not have to read the charts.",
    how: "“Yes” means its score on new text keeps improving. Slow patches are normal. “It is memorising” means it is learning its practice text by heart instead of learning to write. The line underneath says what to do.",
  },
  "train.milestones": {
    title: "How far along",
    what: "Six steps on the way from random letters to fluent writing, based on the test score.",
    how: "Filled segments are steps it has reached. Hover a segment to see its name. The line below says where it is now and what comes next.",
  },
  "train.score": {
    title: "Test score",
    what: "How well the model guesses the next character of text it has never seen, in bits per character. Lower is better.",
    how: "Random guessing scores about 8. Below 3.5 it spells common words, and below 1.8 it writes simple sentences. The arrow shows the change since the last check.",
  },
  "train.textRead": {
    title: "Text read",
    what: "How many characters (letters, spaces and punctuation) the model has read so far.",
    how: "The line below compares it with something familiar. The more it reads, the better it usually writes.",
  },
  "train.speed": {
    title: "Reading speed",
    what: "How many characters the model reads each second on this computer.",
    how: "Steady is good. A drop usually means another app is busy. Bigger models read more slowly.",
  },
  "train.time": {
    title: "Time",
    what: "How long the model has spent training. Paused time is not counted.",
    how: "Below it you see your time goal, or a rough guess of how long until the score reaches 2.0.",
  },

  // ── Train: Experts tab ─────────────────────────────────────────────────────────────────────────────────────────
  "experts.working": {
    title: "Working now",
    what: "The model is made of many small specialist networks called experts. These are the ones loaded in memory and doing the work right now.",
    how: "Each square is one expert; darker means busier. A ring marks a newcomer still on trial. A dashed outline marks one that is barely used and may be retired.",
  },
  "experts.grow": {
    title: "Can the pool grow?",
    what: "The model can add a new expert when it needs more room to learn, but only when five safety checks all pass.",
    how: "A tick means a check passes; a cross means it does not yet. Crosses are normal and are not a problem. When all five pass, a new expert may be added.",
  },
  "experts.all": {
    title: "Everyone in the pool",
    what: "Every expert the model has, from busiest to least busy.",
    how: "Solid bars are loaded in memory now; pale bars wait on disk until needed. It is normal for a few experts to do most of the work.",
  },

  // ── Train: This computer tab ───────────────────────────────────────────────────────────────────────────────────
  "system.computer": {
    title: "This computer",
    what: "The hardware this training is using.",
    how: "“Memory for training” is how much memory the app lets the model use, leaving the rest for your other apps. Saves go to disk, so keep some space free.",
  },
  "system.warnings": {
    title: "Warnings",
    what: "Things the app noticed during this training that you may want to know.",
    how: "Training carries on. Each warning says what happened and, where it can, what to do about it.",
  },
  "system.saves": {
    title: "Saved progress",
    what: "Copies of the model saved to disk every few minutes while it trains.",
    how: "If you stop, or something goes wrong, training continues from a save. The score is the test score at that moment; lower is better. Chat and Export use these saves.",
  },

  // ── Set up ─────────────────────────────────────────────────────────────────────────────────────────────────────
  "setup.text": {
    title: "What should it read?",
    what: "The text the model learns from. It learns to write like whatever it reads.",
    how: "Stories teach it stories; code teaches it code. Add or change text on the Text page.",
  },
  "setup.size": {
    title: "How big a model?",
    what: "Bigger models can learn more and write better, but they need more memory and much more time.",
    how: "The recommended size is a safe start. The speed, memory and disk figures on each card are estimates for this computer. A size it cannot run is greyed out.",
  },
  "setup.goal": {
    title: "How long should it train?",
    what: "When the training stops by itself.",
    how: "Ten minutes is enough to watch the Tiny model learn words. Longer runs keep improving, though more slowly. You can stop early or continue later.",
  },
  "setup.advanced": {
    title: "Advanced settings",
    what: "Finer control over how the model learns. The defaults work well for most text.",
    how: "Change them only to experiment. Any change marks the run as Custom, so you can compare it with a standard run later.",
  },
  "setup.checks": {
    title: "Before you start",
    what: "A quick check that this computer and your text are ready for this training.",
    how: "A tick means all good. A warning sign points at something to look at, such as no GPU (training will be slow) or no text yet (you need some before you can start).",
  },

  // ── Text ───────────────────────────────────────────────────────────────────────────────────────────────────────
  "data.starters": {
    title: "Start with something ready",
    what: "Text collections prepared for you, so you can start without finding your own.",
    how: "Simple stories is the best first choice. Some are downloaded once; the others are made on this computer. Get one, then train on it.",
  },
  "data.own": {
    title: "Use your own text",
    what: "Any plain-text files: books, notes, articles or code. They are read on this computer and never uploaded.",
    how: "Drop folders here or choose them. Each top-level folder becomes its own kind of text. Files that are not plain text, such as images or PDFs, are left out.",
  },
  "data.mine": {
    title: "What you have",
    what: "Every text collection you have added.",
    how: "Pick one from the list to see what is inside, switch parts of it on or off, peek at it, or train on it.",
  },
  "data.detail": {
    title: "About this text",
    what: "Most of the text is for learning. A small part is kept aside and never learned from, so the model can be tested fairly.",
    how: "Each row in the table is one kind of text. Untick a row to leave it out. The bar shows how much of the reading each kind gets.",
  },
  "data.split": {
    title: "Keep some aside for testing",
    what: "A share of your text that the model never learns from. It is used only to test it.",
    how: "Testing on unseen text shows whether it is really learning to write, not just memorising. A few percent is plenty. Press Apply after moving the slider.",
  },
  "data.peek": {
    title: "A peek at what it will read",
    what: "A random passage from your text, exactly as the model will see it.",
    how: "Press Show another for a different passage. If the text looks garbled here, the model will learn it garbled too.",
  },

  // ── Chat ───────────────────────────────────────────────────────────────────────────────────────────────────────
  "chat.model": {
    title: "Which model?",
    what: "Any training that has saved some progress. The chat uses its best save.",
    how: "Models that have read more usually write better. Chatting never changes the original training.",
  },
  "chat.mode": {
    title: "How should it respond?",
    what: "Two ways to use the model.",
    how: "“Continue my text”: you write the start and it carries on. This suits most models. “Conversation”: you ask and it answers, which works well only if it was trained on conversations.",
  },
  "chat.thinking": {
    title: "How it thinks",
    what: "The model can think over a character for several steps before writing it. Harder characters get more steps.",
    how: "The line shows the steps for each recent character; peaks are where it thought hardest. Tick the box to shade the reply, darker where it thought more.",
  },
  "chat.learn": {
    title: "Learn from this chat",
    what: "When it is on, the model keeps learning from what you write together, a little after each message.",
    how: "Only a copy changes, never your original training. Press “Keep what it learned” to save that copy. It may get slightly worse at other things.",
  },

  // ── Export ─────────────────────────────────────────────────────────────────────────────────────────────────────
  "export.model": {
    title: "Which model?",
    what: "The training you want to save to a folder you can keep or share.",
    how: "Only trainings that have saved some progress are listed.",
  },
  "export.version": {
    title: "Which saved version?",
    what: "A training saves itself many times along the way. Each save is the model as it was at that moment.",
    how: "“The best one” is the save with the lowest test score, which is usually the one you want.",
  },
  "export.contents": {
    title: "What you get",
    what: "A folder holding the model and a short description of it.",
    how: "Copy the folder to another computer and use Import on its Runs page to bring the model in.",
  },

  // ── Runs ───────────────────────────────────────────────────────────────────────────────────────────────────────
  "runs.status": {
    title: "Status",
    what: "Where each training stands: reading, paused, finished, stopped and so on.",
    how: "Open a stopped, finished or interrupted run to continue it from its last save.",
  },
  "runs.size": {
    title: "Size",
    what: "The model size chosen in Set up: Tiny, Small or Full.",
    how: "Custom means some advanced settings were changed. Imported means it came from a folder.",
  },
  "runs.best": {
    title: "Best score",
    what: "The best test score the run reached, in bits per character. Lower is better.",
    how: "Tick two or more runs and press Compare to see their progress side by side.",
  },

  // ── Compare ────────────────────────────────────────────────────────────────────────────────────────────────────
  "compare.results": {
    title: "How each one ended",
    what: "The final numbers for each run, side by side.",
    how: "Best score is the lowest test score it reached; last score is where it ended. Lower is better, and the best run is marked. The verdict is the “Is it learning?” answer at the end.",
  },
  "compare.diff": {
    title: "What was different",
    what: "Only the settings that differ between these runs. Anything not listed was the same.",
    how: "Use it to see what may have made one run do better than another.",
  },
  "compare.samples": {
    title: "What each one writes",
    what: "Each model's latest writing sample, continuing the same opening.",
    how: "The scores say which run did better; reading these shows what that difference looks like.",
  },

  // ── Settings ───────────────────────────────────────────────────────────────────────────────────────────────────
  "settings.look": {
    title: "How it looks",
    what: "Light or dark colours, and how much detail the app shows.",
    how: "“Show advanced settings” adds extra controls to Set up and extra charts to Train. Leave it off until you want to experiment.",
  },
  "settings.disk": {
    title: "Disk space",
    what: "How much room the app's text, saved models, chats and history take on this computer.",
    how: "To free space, delete old runs on the Runs page or text you no longer need on the Text page.",
  },
  "settings.computer": {
    title: "This computer",
    what: "The hardware the app found and trains on.",
    how: "A graphics processor (GPU) trains much faster than the main processor. The memory figure is how much the app allows itself, so the rest of your computer stays responsive.",
  },
  "settings.about": {
    title: "About",
    what: "Which version of the app you have, and where the ideas and practice text come from.",
    how: "The engine is the part that builds and trains the model. If it says Simulated, the numbers are made up for testing and do not come from a real model.",
  },

  // ── Continue as a new run ──────────────────────────────────────────────────────────────────────────────────────
  "fork.rate": {
    title: "How fast it learns",
    what: "How big a step the model takes each time it learns from a mistake.",
    how: "Keep “As before” unless you have a reason. Slower is steadier, which suits new text. Faster can save time but may make the score jump around.",
  },
} as const satisfies Record<string, { title: string; what: string; how: string }>;

export type SectionId = keyof typeof SECTION_INFO;
