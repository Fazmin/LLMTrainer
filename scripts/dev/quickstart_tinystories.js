// Dev-only (MINAGI_DEV_SCRIPT): first-run flow with the real TinyStories download, then report what the window shows.
window.__devReport = async (rounds, every) => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const emit = (payload) => window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "dev-report", payload });
  for (let i = 0; i < rounds; i++) {
    const t = document.body.innerText;
    const writes = t.indexOf("What it writes");
    await emit(`--- ${new Date().toISOString()} ${location.hash}\n` + (location.hash.startsWith("#/train") ? t.slice(writes, writes + 1400) + "\n[numbers] " + t.slice(t.indexOf("Test score (bits"), t.indexOf("Test score (bits") + 600) : t.slice(0, 900)));
    await sleep(every);
  }
};
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const click = async (text, tries = 150) => {
    for (let i = 0; i < tries; i++) {
      const el = [...document.querySelectorAll("button")].find((b) => b.textContent.includes(text));
      if (el) { el.click(); return true; }
      await sleep(200);
    }
    return false;
  };
  await sleep(1000);
  await click("Get started");
  await sleep(600);
  await click("Quick start");
  window.__devReport(60, 20000);
})();
