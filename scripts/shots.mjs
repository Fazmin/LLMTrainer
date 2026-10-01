// Walks the app in a real browser (using the in-browser fixture backend) and saves screenshots to ./shots.
// Usage: pnpm dev:web  (in another terminal), then: node scripts/shots.mjs [baseUrl]
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:1420";
mkdirSync("shots", { recursive: true });
const browser = await chromium.launch();

async function page(theme, level = "beginner", size = { width: 1360, height: 1750 }) {
  const ctx = await browser.newContext({ viewport: size, colorScheme: theme, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  const errors = [];
  p.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  p.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
  await p.addInitScript((lvl) => localStorage.setItem("llm-trainer:prefs", JSON.stringify({ theme: "system", level: lvl })), level);
  return { p, errors, ctx };
}

const shot = (p, name, full = true) => p.screenshot({ path: `shots/${name}.png`, fullPage: full });
let problems = 0;

// 1. A finished run, light theme.
{
  const { p, errors, ctx } = await page("light");
  await p.goto(base + "/#/train");
  await p.waitForSelector("text=What it writes");
  await p.waitForTimeout(1200);
  await shot(p, "01-train-finished-light");
  await p.getByRole("button", { name: /Continue as a new run/ }).click();
  await p.waitForSelector("text=Text to read next");
  await p.waitForTimeout(300);
  await shot(p, "01b-fork-dialog-light", false);
  await p.keyboard.press("Escape");
  await p.getByRole("tab", { name: "Experts" }).click();
  await p.waitForTimeout(500);
  await shot(p, "02-experts-light");
  await p.getByRole("tab", { name: "Thinking depth" }).click();
  await p.waitForTimeout(500);
  await shot(p, "03-thinking-light");
  problems += errors.length;
  errors.forEach((e) => console.log("light finished:", e));
  await ctx.close();
}

// 2. Setup, both themes, advanced.
{
  const { p, errors, ctx } = await page("light", "advanced");
  await p.goto(base + "/#/setup");
  await p.waitForSelector("text=How big a model?");
  await p.waitForTimeout(800);
  await shot(p, "04-setup-light-advanced");
  problems += errors.length;
  errors.forEach((e) => console.log("setup:", e));
  await ctx.close();
}

// 3. Start a run and watch it live, dark theme.
{
  const { p, errors, ctx } = await page("dark");
  await p.goto(base + "/#/setup");
  await p.waitForSelector("text=Start training");
  await p.getByRole("button", { name: "Start training" }).click();
  await p.waitForSelector("text=Reading", { timeout: 15000 });
  await p.waitForTimeout(6000);
  await shot(p, "05-train-live-early-dark");
  await p.waitForTimeout(16000);
  await shot(p, "06-train-live-later-dark");
  await p.getByRole("tab", { name: "Experts" }).click();
  await p.waitForTimeout(700);
  await shot(p, "07-experts-live-dark");
  problems += errors.length;
  errors.forEach((e) => console.log("live:", e));
  await p.goto(base + "/#/runs");
  await p.waitForSelector("text=Your runs");
  await p.waitForTimeout(500);
  await shot(p, "08-runs-dark");
  await p.getByRole("button", { name: /Import a model/ }).click();
  await p.waitForSelector("text=Import this model?");
  await p.waitForTimeout(300);
  await shot(p, "08b-import-dialog-dark", false);
  await ctx.close();
}

// 4. Chat with the seeded finished run.
{
  const { p, errors, ctx } = await page("light", "beginner", { width: 1360, height: 900 });
  await p.goto(base + "/#/chat");
  await p.waitForSelector("text=Chat with your model");
  await p.waitForTimeout(500);
  await shot(p, "09-chat-start-light", false);
  await p.getByRole("radio", { name: /Conversation/ }).check({ force: true });
  await p.getByRole("button", { name: "Start chatting" }).click();
  await p.waitForSelector("text=Ask it something.");
  await p.getByRole("switch", { name: "Learn from this chat" }).click();
  await p.getByRole("button", { name: "What are you?" }).click();
  await p.getByRole("button", { name: "Send" }).click();
  await p.waitForTimeout(900);
  await shot(p, "10-chat-streaming-light", false);
  await p.waitForSelector("text=Learned from this exchange", { timeout: 15000 });
  await p.getByLabel("Shade each character by how hard it thought").check();
  await p.waitForTimeout(300);
  await shot(p, "11-chat-learned-light", false);
  problems += errors.length;
  errors.forEach((e) => console.log("chat:", e));
  await ctx.close();
}

// 5. Compare two seeded runs.
{
  const { p, errors, ctx } = await page("light", "beginner", { width: 1360, height: 1500 });
  await p.goto(base + "/#/runs");
  await p.waitForSelector("text=Your runs");
  await p.getByLabel(/Select Tiny stories, first try/).check();
  await p.getByLabel(/Select Small, higher learning rate/).check();
  await p.getByRole("button", { name: /Compare 2 runs/ }).click();
  await p.waitForSelector("text=What was different");
  await p.waitForTimeout(700);
  await shot(p, "12-compare-light");
  problems += errors.length;
  errors.forEach((e) => console.log("compare:", e));
  await ctx.close();
}

// 6. Text: a download in flight, then the dataset detail.
{
  const { p, errors, ctx } = await page("light", "beginner", { width: 1360, height: 1500 });
  await p.goto(base + "/#/data");
  await p.waitForSelector("text=Start with something ready");
  await p.waitForTimeout(500);
  await p.getByRole("button", { name: "Download" }).first().click();
  await p.waitForSelector("text=Downloading the stories");
  await p.waitForTimeout(1500);
  await shot(p, "13-data-downloading-light");
  await p.waitForSelector("text=Ready", { timeout: 15000 });
  await p.waitForTimeout(600);
  await shot(p, "14-data-ready-light");
  problems += errors.length;
  errors.forEach((e) => console.log("data:", e));
  await ctx.close();
}

// 7. First launch.
{
  const { p, errors, ctx } = await page("light", "beginner", { width: 1360, height: 900 });
  await p.goto(base + "/?fresh=1#/");
  await p.waitForSelector("text=Teach a small computer program to write.");
  await p.waitForTimeout(600);
  await shot(p, "15-welcome-intro-light", false);
  await p.getByRole("button", { name: "Get started" }).click();
  await p.waitForSelector("text=How would you like to begin?");
  await shot(p, "16-welcome-choose-light", false);
  await p.getByRole("button", { name: /Quick start/ }).click();
  await p.waitForSelector("text=Getting the text");
  await p.waitForTimeout(1800);
  await shot(p, "17-welcome-working-light", false);
  await p.waitForSelector("text=What it writes", { timeout: 25000 });
  problems += errors.length;
  errors.forEach((e) => console.log("welcome:", e));
  await ctx.close();
}

// 8. Settings.
{
  const { p, errors, ctx } = await page("dark", "beginner", { width: 1360, height: 1000 });
  await p.goto(base + "/#/settings");
  await p.waitForSelector("text=Disk space");
  await p.waitForTimeout(600);
  await shot(p, "18-settings-dark");
  problems += errors.length;
  errors.forEach((e) => console.log("settings:", e));
  await ctx.close();
}

// 9. Export a model.
{
  const { p, errors, ctx } = await page("light", "beginner", { width: 1360, height: 900 });
  await p.goto(base + "/#/export");
  await p.waitForSelector("text=Which saved version?");
  await p.getByRole("button", { name: /Choose where to save/ }).click();
  await p.waitForSelector("text=Saved", { timeout: 15000 });
  await p.waitForTimeout(400);
  await shot(p, "19-export-done-light", false);
  problems += errors.length;
  errors.forEach((e) => console.log("export:", e));
  await ctx.close();
}

await browser.close();
console.log(problems === 0 ? "no browser errors" : `${problems} browser error(s) — see above`);
