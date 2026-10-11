// Dev-only (MINAGI_DEV_SCRIPT): a guided tour of the real app for the README screenshots and video.
// It clicks through the first-run Quick Start (real TinyStories download, real training on this machine's GPU),
// visits each screen, and announces `SHOT <name>` on the dev-report log at the moments worth keeping. The capture
// loop in scripts/media/capture.sh turns those announcements into window-only screenshots.
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const emit = (payload) => window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "dev-report", payload });
  const shot = async (name, wait = 900) => {
    await sleep(wait); // let the layout and charts settle before the picture is taken
    await emit(`SHOT ${name}`);
    await sleep(1800); // the capture loop polls about once a second
  };
  const find = (text, selector = "button") =>
    [...document.querySelectorAll(selector)].find((b) => (b.textContent || "").includes(text) || b.getAttribute("aria-label") === text);
  const click = async (text, selector = "button", tries = 300) => {
    for (let i = 0; i < tries; i++) {
      const el = find(text, selector);
      if (el && !el.disabled) { el.click(); return true; }
      await sleep(200);
    }
    await emit(`NOTE never found "${text}"`);
    return false;
  };
  const waitFor = async (test, tries = 3000, every = 500) => {
    for (let i = 0; i < tries; i++) { if (test()) return true; await sleep(every); }
    return false;
  };
  const go = async (hash) => { location.hash = hash; await sleep(900); };
  const seen = (text) => document.body.innerText.includes(text);

  await sleep(1500);

  // 1. First launch.
  await shot("welcome-intro");
  await click("Get started");
  await sleep(700);
  await shot("welcome-choose");
  await click("Quick start");

  // 2. Getting the text ready (download, prepare, build the model).
  await waitFor(() => seen("Getting the text") || location.hash.startsWith("#/train"), 100, 200);
  await shot("quickstart-progress", 2500);

  // 3. Training, live. The clock starts when the dashboard shows a run, not while the welcome overlay is still up.
  await waitFor(() => location.hash.startsWith("#/train") && !seen("Getting the text") && seen("What it writes"), 3000, 500);
  const trainStart = Date.now();
  await emit("NOTE training started");
  await shot("train-early", 25000);
  const at = async (seconds) => { const left = trainStart + seconds * 1000 - Date.now(); if (left > 0) await sleep(left); };
  // the window is short, so bring the tab strip (and the chart under it) into view, or back to the top
  const toTabs = async () => { (find("Progress", "[role=tab]") || document.body).scrollIntoView({ block: "start" }); await sleep(500); };
  const toTop = async () => {
    document.querySelectorAll("div").forEach((d) => { if (d.scrollTop > 0 && getComputedStyle(d).overflowY === "auto") d.scrollTo({ top: 0 }); });
    await sleep(500);
  };
  await at(110);
  await shot("train-mid");
  await toTabs();
  await shot("train-mid-charts");
  await toTop();
  await at(270);
  await shot("train-late");
  await toTabs();
  await shot("train-charts");
  await click("Experts", "[role=tab]");
  await toTabs();
  await shot("experts");
  await click("Thinking depth", "[role=tab]");
  await toTabs();
  await shot("thinking");
  await click("This computer", "[role=tab]");
  await toTabs();
  await shot("system");
  await click("Progress", "[role=tab]");
  await toTop();
  await sleep(600);
  await click("Stop and save");
  await waitFor(() => seen("Stopped") || seen("stopped"), 60, 500);
  await shot("train-stopped", 2500);

  // 4. Chat with what it learned.
  await go("#/chat");
  await waitFor(() => seen("Chat with your model"), 40, 250);
  await shot("chat-start");
  await click("Continue my text", "label");
  await click("Start chatting");
  await waitFor(() => seen("Once upon a time, there was a little boy named Tom"), 40, 250);
  await click("Once upon a time, there was a little boy named Tom. One day he ");
  await sleep(500);
  await click("Send");
  await sleep(2500);
  await shot("chat-writing", 1500);
  await waitFor(() => !find("Stop"), 120, 500);
  const shade = document.querySelector('input[aria-label="Shade each character by how hard it thought"]');
  if (shade && !shade.checked) shade.click();
  await shot("chat-reply", 2500);

  // 5. The other screens.
  await go("#/runs");
  await waitFor(() => seen("Your runs"), 40, 250);
  await shot("runs");
  await go("#/data");
  await waitFor(() => seen("Start with something ready"), 40, 250);
  await shot("text");
  await go("#/setup");
  await waitFor(() => seen("How big a model?"), 40, 250);
  await shot("setup");
  await emit("NOTE tour finished");
  await emit("DONE");
})();
