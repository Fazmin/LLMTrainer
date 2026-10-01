// Dev-only: report the visible text of the window every few seconds, so a run can be checked without a screenshot.
window.__devReport = async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const emit = (payload) => window.__TAURI_INTERNALS__.invoke("plugin:event|emit", { event: "dev-report", payload });
  for (let i = 0; i < 12; i++) {
    await emit(`--- ${new Date().toISOString()} ${location.hash || location.pathname}\n` + document.body.innerText.slice(0, 2500));
    await sleep(8000);
  }
};
// Dev-only (MINAGI_DEV_SCRIPT): the first-run flow with the offline sample, as a person would click through it.
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const click = async (text, tries = 150) => {
    for (let i = 0; i < tries; i++) {
      const el = [...document.querySelectorAll("button")].find((b) => b.textContent.includes(text));
      if (el) { el.click(); console.log("[dev] clicked", text); return true; }
      await sleep(200);
    }
    console.log("[dev] never found", text);
    return false;
  };
  await sleep(1000);
  await click("Get started");
  await sleep(600);
  await click("Just look around");
  window.__devReport && window.__devReport();
})();
