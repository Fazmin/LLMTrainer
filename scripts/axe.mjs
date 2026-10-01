// Runs axe-core on every screen, in both themes, against the in-browser fixture backend.
// Usage: pnpm dev:web (other terminal), then: node scripts/axe.mjs [baseUrl]
import AxeBuilder from "@axe-core/playwright";
import { chromium } from "@playwright/test";

const base = process.argv[2] ?? "http://127.0.0.1:1420";
const browser = await chromium.launch();
let total = 0;

const screens = [
  ["train (finished run)", "/#/train", "text=What it writes"],
  ["setup", "/#/setup", "text=How big a model?"],
  ["runs", "/#/runs", "text=Your runs"],
  ["chat (start)", "/#/chat", "text=Chat with your model"],
  ["compare", "/#/compare?ids=1,2", "text=What was different"],
  ["text", "/#/data", "text=Start with something ready"],
  ["settings", "/#/settings", "text=Disk space"],
  ["export", "/#/export", "text=Which saved version?"],
  ["welcome", "/?fresh=1#/welcome", "text=Teach a small computer program to write."],
];

for (const scheme of ["light", "dark"]) {
  for (const [name, route, ready] of screens) {
    const ctx = await browser.newContext({ viewport: { width: 1360, height: 1200 }, colorScheme: scheme });
    const page = await ctx.newPage();
    await page.addInitScript(() => localStorage.setItem("llm-trainer:prefs", JSON.stringify({ theme: "system", level: "advanced" })));
    await page.goto(base + route);
    await page.waitForSelector(ready, { timeout: 15000 });
    await page.waitForTimeout(900);
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    for (const v of results.violations) {
      total += v.nodes.length;
      console.log(`[${scheme}] ${name}: ${v.id} (${v.impact}) x${v.nodes.length} — ${v.help}`);
      for (const n of v.nodes.slice(0, 3)) console.log(`    ${n.target.join(" ")}  ${(n.failureSummary ?? "").split("\n")[1] ?? ""}`);
    }
    await ctx.close();
  }
}
await browser.close();
console.log(total === 0 ? "axe: no violations" : `axe: ${total} violating node(s)`);
process.exit(total === 0 ? 0 : 1);
