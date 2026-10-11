// Dev-only (MINAGI_DEV_SCRIPT): the second half of the README tour, run against the data folder that
// readme_tour.js left behind (a finished Tiny run). Chats with the model, visits the other screens by clicking the
// sidebar like a person would, then shows the dashboard in the dark theme. See scripts/media/capture.sh.
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const emit = (payload) => window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "dev-report", payload });
  const shot = async (name, wait = 900) => {
    await sleep(wait);
    await emit(`SHOT ${name}`);
    await sleep(1800);
  };
  const find = (text, selector = "button") =>
    [...document.querySelectorAll(selector)].find((b) => (b.textContent || "").trim().startsWith(text) || b.getAttribute("aria-label") === text);
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
  const seen = (text) => document.body.innerText.includes(text);
  const nav = async (label) => { await click(label, "nav a"); await sleep(900); };
  const toTabs = async () => { (find("Progress", "[role=tab]") || document.body).scrollIntoView({ block: "start" }); await sleep(500); };
  const toTop = async () => {
    document.querySelectorAll("div").forEach((d) => { if (d.scrollTop > 0 && getComputedStyle(d).overflowY === "auto") d.scrollTo({ top: 0 }); });
    await sleep(500);
  };

  // The real mouse pointer may be resting over the window; make nothing hoverable so no item looks highlighted by it.
  const style = document.createElement("style");
  style.textContent = "*, *::before, *::after { pointer-events: none !important; }";
  document.head.appendChild(style);

  await sleep(1500);

  // Chat with what the model learned.
  await nav("Chat");
  await waitFor(() => seen("Chat with your model"), 40, 250);
  await shot("chat-start");
  await click("Continue my text", "label");
  await click("Start chatting");
  await waitFor(() => !!find("Once upon a time"), 40, 250);
  await click("Once upon a time");
  await sleep(500);
  await click("Send");
  await shot("chat-writing", 1200);
  await waitFor(() => !find("Stop"), 240, 500);
  await shot("chat-reply", 1500);
  const shade = document.querySelector('input[aria-label="Shade each character by how hard it thought"]') ||
    [...document.querySelectorAll("label")].find((l) => l.textContent.includes("Shade each character"))?.querySelector("input");
  if (shade && !shade.checked) shade.click();
  await shot("chat-shaded", 1200);

  // The other screens.
  await nav("Runs");
  await waitFor(() => seen("Your runs"), 40, 250);
  await shot("runs");
  await nav("Text");
  await waitFor(() => seen("Start with something ready"), 40, 250);
  await shot("text");
  await nav("Set up");
  await waitFor(() => seen("How big a model?"), 40, 250);
  await shot("setup");

  // Dark theme on the finished run.
  await click("dark theme");
  await nav("Train");
  await waitFor(() => seen("What it writes"), 40, 250);
  await shot("train-dark");
  await toTabs();
  await shot("charts-dark");
  await toTop();
  await click("system theme");
  await sleep(500);
  await emit("NOTE tour finished");
  await emit("DONE");
})();
