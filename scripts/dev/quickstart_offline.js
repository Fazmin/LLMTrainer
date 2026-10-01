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
})();
