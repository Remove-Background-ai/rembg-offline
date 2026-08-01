// Cross-engine smoke test: loads the demo, runs background removal, and
// asserts a valid (non-degenerate) mask is produced. WebKit is the target
// engine (Safari); Chromium is the baseline.
//
// Usage: node scripts/test-browsers.mjs [webkit|webkit-ios|chromium]
import { webkit, chromium, devices } from "playwright";

const BASE = process.env.DEMO_URL || "http://localhost:5199";
const only = process.argv[2];
const TIMEOUT = 8 * 60 * 1000; // model download + wasm inference can be slow

const scenarios = [
  { name: "auto", query: "" },
  { name: "forced-wasm-q8", query: "?device=wasm&dtype=q8" },
];

const engines = [
  ["webkit", webkit, null],
  ["webkit-ios", webkit, devices["iPhone 15 Pro"]],
  ["chromium", chromium, null],
].filter(([name]) => !only || name === only);

let failed = false;

for (const [name, engine, device] of engines) {
  const browser = await engine.launch();
  const context = device ? await browser.newContext(device) : await browser.newContext();
  for (const scenario of scenarios) {
    const page = await context.newPage();
    page.on("console", (m) => {
      const t = m.text();
      if (t.startsWith("[rembg]") || m.type() === "error") console.log(`  [${name}:${scenario.name}] console: ${t}`);
    });
    page.on("pageerror", (e) => console.log(`  [${name}:${scenario.name}] pageerror: ${e.message}`));

    try {
      await page.goto(`${BASE}/${scenario.query}`, { waitUntil: "domcontentloaded" });
      await page.waitForFunction(() => !document.getElementById("cap").textContent.includes("detecting"), { timeout: 30000 });
      const ladder = await page.textContent("#cap");
      console.log(`[${name}:${scenario.name}] ladder: ${ladder}`);

      await page.click("#run-sample");
      await page.waitForSelector("#status.done, #status.error", { timeout: TIMEOUT });
      const cls = await page.getAttribute("#status", "class");
      const text = await page.textContent("#status");

      if (cls === "done") {
        const result = JSON.parse(text);
        // A valid mask keeps the subject (nonzero alpha) without keeping everything
        const maskOk = result.nonzeroAlphaPct > 5 && result.nonzeroAlphaPct < 95;
        console.log(`[${name}:${scenario.name}] ${maskOk ? "PASS" : "FAIL (degenerate mask)"}: ${text}`);
        if (!maskOk) failed = true;
      } else {
        console.log(`[${name}:${scenario.name}] FAIL: ${text}`);
        failed = true;
      }
    } catch (e) {
      console.log(`[${name}:${scenario.name}] FAIL: ${e.message}`);
      failed = true;
    } finally {
      await page.close();
    }
  }
  await browser.close();
}

process.exit(failed ? 1 : 0);
