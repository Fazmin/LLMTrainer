// Dev-only: report the visible text of the window every few seconds, so a run can be checked without a screenshot.
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const emit = (payload) => window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "dev-report", payload });
  for (let i = 0; i < 12; i++) {
    await emit(`--- ${new Date().toISOString()} ${location.hash || location.pathname}\n` + document.body.innerText.slice(0, 2500));
    await sleep(8000);
  }
})();
