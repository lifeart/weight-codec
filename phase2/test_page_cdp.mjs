// Drive the REAL demo page (docs/index.html + app.js + worker) in headless Chromium over the DevTools protocol and
// exercise the user flow: pick a configuration -> "Download + decode + verify" -> read the verification list and the
// per-file SHA table -> "Generate" a preset prompt and compare with the precomputed PyTorch answer.  Collects page
// exceptions / console errors.  Works against a local copy or the live URLs.
// usage: node phase2/test_page_cdp.mjs <page url> [--render-only] [--chains family,delta,indep] [--gen family:instruct,...]
//   local: node phase2/test_page_cdp.mjs local          (serves docs/ on 127.0.0.1 with ?data=bitstreams/)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const renderOnly = args.includes("--render-only");
const chains = opt("--chains", "family,delta,indep").split(",");
const gens = opt("--gen", "family:instruct,family:base,delta:instruct").split(",").filter(Boolean).map((s) => s.split(":"));
const CHROME = process.env.CHROME || path.join(os.homedir(), "Library/Caches/ms-playwright/chromium_headless_shell-1255/chrome-headless-shell-mac-arm64/chrome-headless-shell");
let url = args[0], server = null;
if (!url) { console.error("usage: node phase2/test_page_cdp.mjs <url|local> [...]"); process.exit(2); }
if (url === "local") {
  const DOCS = path.join(ROOT, "docs");
  server = http.createServer((req, res) => {
    const p = path.join(DOCS, decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/\/$/, "/index.html"));
    if (!p.startsWith(DOCS) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    const ct = p.endsWith(".html") ? "text/html" : p.endsWith(".js") ? "text/javascript" : p.endsWith(".css") ? "text/css" : p.endsWith(".json") ? "application/json" : "application/octet-stream";
    res.writeHead(200, { "content-length": fs.statSync(p).size, "content-type": ct }); fs.createReadStream(p).pipe(res);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  url = `http://127.0.0.1:${server.address().port}/index.html?data=bitstreams/`;
}
let fails = 0;
const check = (ok, msg) => { console.log((ok ? "  ok   " : "  FAIL ") + msg); if (!ok) fails++; };

// ---- launch Chromium (fresh profile: nothing cached, like a first-time visitor) ----
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "wc_cdp_"));
const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--remote-debugging-port=0", `--user-data-dir=${prof}`, "about:blank"], { stdio: "ignore" });
let port = null;
for (let i = 0; i < 100 && !port; i++) {
  await new Promise((r) => setTimeout(r, 200));
  const f = path.join(prof, "DevToolsActivePort");
  if (fs.existsSync(f)) port = Number(fs.readFileSync(f, "utf8").split("\n")[0]);
}
if (!port) { console.error("Chromium did not start"); chrome.kill(); process.exit(2); }
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let seq = 0; const pend = new Map(); const errors = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); return; }
  if (m.method === "Runtime.exceptionThrown") errors.push("exception: " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") errors.push("console.error: " + m.params.args.map((a) => a.value ?? a.description).join(" "));
  if (m.method === "Log.entryAdded" && m.params.entry.level === "error") errors.push("log: " + m.params.entry.text + " " + (m.params.entry.url || ""));
};
const send = (method, params = {}) => new Promise((r) => { const id = ++seq; pend.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const ev = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error("evaluate failed: " + JSON.stringify(r.result.exceptionDetails).slice(0, 300));
  return r.result?.result?.value;
};
const until = async (expr, ms, step = 500) => {
  const t0 = Date.now();
  for (;;) { const v = await ev(expr); if (v) return v; if (Date.now() - t0 > ms) return null; await new Promise((r) => setTimeout(r, step)); }
};
await send("Runtime.enable"); await send("Log.enable"); await send("Page.enable");
console.log("page:", url);
await send("Page.navigate", { url });
const ready = await until(`document.readyState === "complete" && typeof MAN !== "undefined" && !!MAN && !document.querySelector("#go").disabled`, 60000);
check(!!ready, "page loaded, data JSON parsed, worker ready (Download button enabled)");
const note = await ev(`(() => { const n = document.querySelector("#data-note"); return n.hidden ? "" : n.textContent; })()`);
check(!note, "no 'live decoding unavailable' note" + (note ? ": " + note.slice(0, 200) : ""));
const cards = await ev(`Array.from(document.querySelectorAll("#chains .card")).map((c) => c.innerText.replace(/\\s+/g, " "))`);
check(cards && cards.length === 3, `3 configuration cards rendered`);
for (const c of cards || []) console.log("     card:", c.slice(0, 260));
const nums = await ev(`document.querySelector("#numbers-body").innerText.length`);
check(nums > 500, `numbers section rendered (${nums} chars)`);
const ex = await ev(`document.querySelectorAll("#extable tbody tr").length`);
check(ex >= 5, `precomputed answers table rendered (${ex} rows)`);
const refline = await ev(`document.querySelector("#refline").innerText`);
console.log("     refline:", (refline || "").slice(0, 300));

const ORDER = await ev(`CHAIN_ORDER`);
if (!renderOnly) {
  for (const c of chains) {
    const idx = ORDER.indexOf(c);
    console.log(`chain ${c}`);
    await ev(`document.querySelectorAll("#chains .card")[${idx}].click(); true`);
    const sel = await ev(`chain`);
    check(sel === c, `card click selects chain ${c}`);
    const t0 = Date.now();
    await ev(`document.querySelector("#go").click(); true`);
    const done = await until(`(!busy && decoded && decoded.chain === ${JSON.stringify(c)}) || document.querySelector("#demo-log").classList.contains("err") ? document.querySelector("#demo-log").innerText : null`, 600000, 1000);
    console.log(`     ${((Date.now() - t0) / 1000).toFixed(1)} s: ${(done || "TIMEOUT").slice(0, 160)}`);
    check(done && done.startsWith("✓"), `${c}: download + decode + verify reported success`);
    const checks = await ev(`Array.from(document.querySelectorAll("#checks li")).map((li) => [li.querySelector("span").className, li.innerText.slice(0, 120)])`);
    for (const [cls, txt] of checks || []) check(cls === "ok" || cls === "approx", `${c}: ${txt.replace(/\s+/g, " ")}`);
    const rows = await ev(`Array.from(document.querySelectorAll("#filetable tbody tr")).map((tr) => Array.from(tr.children).map((td) => td.innerText))`);
    for (const r of rows || []) check(r[8] === "✓" && r[9] === "✓", `${c}: ${r[0]} (${r[2]} MB, ${r[3]}) symbols ${r[8]} weights ${r[9]} vs-original ${r[10]}`);
    for (const [gc, which] of gens.filter(([gc]) => gc === c)) {
      const hasBase = await ev(`!document.querySelector('input[name="which"][value="base"]').disabled`);
      if (which === "base" && !hasBase) { check(false, `${c}: base radio disabled`); continue; }
      await ev(`document.querySelector('input[name="which"][value="${which}"]').checked = true; document.querySelector("#prompt").value = GEN.prompts[0]; document.querySelector("#maxnew").value = GEN.max_new_tokens; document.querySelector("#gen-go").click(); true`);
      const g = await until(`!genRun && /tok\\/s/.test(document.querySelector("#gen-stat").innerText) ? [document.querySelector("#gen-stat").innerText, document.querySelector("#gen-cmp").innerText, document.querySelector("#gen-out").innerText] : null`, 300000, 1000);
      if (!g) { check(false, `${c}/${which}: generation timed out`); continue; }
      console.log(`     ${which}: ${g[0]} | ${g[2].replace(/\s+/g, " ").slice(0, 100)}`);
      check(g[1].startsWith("✓"), `${c}/${which}: live answer == offline PyTorch answer (${g[1].slice(0, 90)})`);
    }
  }
}
check(errors.length === 0, `no page exceptions / console errors` + (errors.length ? ": " + errors.slice(0, 5).join(" | ") : ""));
ws.close(); if (server) server.close();
await new Promise((r) => { chrome.once("exit", r); chrome.kill(); });
try { fs.rmSync(prof, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); }
catch (e) { console.log(`  note: temporary browser profile ${prof} not removed (${e.code}); delete it by hand`); }
console.log(fails ? `${fails} FAILED` : "ALL OK");
process.exit(fails ? 1 : 0);
