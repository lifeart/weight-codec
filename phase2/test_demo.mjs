// End-to-end test of the browser demo code in Node (>= 18): serves docs/ over HTTP, runs the REAL
// docs/js/worker.js (importScripts / postMessage emulated), decodes every chain from the staged
// bitstreams, and checks
//   * symbol + rebuilt-weight SHA-256 == Python decoder (manifest), lossless parts == original checkpoint
//   * the JS tokenizer == Python `tokenizers` (test vectors) and the chat template == apply_chat_template
//   * the JS Llama forward pass on the rebuilt weights reproduces PyTorch: first-step top-10 logits and
//     the 32-token greedy continuation (phase2/demo_testvec.json, written by phase2/demo_build.py)
// Usage: node phase2/test_demo.mjs [chain ...]      (needs docs/bitstreams staged by demo_build.py)
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DOCS = path.join(ROOT, "docs");
const TV = JSON.parse(fs.readFileSync(path.join(ROOT, "phase2", "demo_testvec.json"), "utf8"));
const MAN = JSON.parse(fs.readFileSync(path.join(DOCS, "data", "manifest.json"), "utf8"));
const chains = process.argv.slice(2).length ? process.argv.slice(2) : ["family", "delta", "indep"];
let fails = 0;
const check = (ok, msg) => { console.log((ok ? "  ok   " : "  FAIL ") + msg); if (!ok) fails++; };

// ---- static server for docs/ ----
const server = http.createServer((req, res) => {
  const p = path.join(DOCS, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!p.startsWith(DOCS) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-length": fs.statSync(p).size }); fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const BASE = `http://127.0.0.1:${server.address().port}/`;

// ---- emulate the worker global scope ----
globalThis.self = globalThis;
globalThis.importScripts = (...fs_) => { for (const f of fs_) vm.runInThisContext(fs.readFileSync(path.join(DOCS, "js", f), "utf8"), { filename: f }); };
let waiters = [];
const log = [];
globalThis.postMessage = (m) => { log.push(m); if (m.type === "error" || m.type === "warn") console.log("  [worker " + m.type + "] " + m.message); waiters = waiters.filter((w) => !w(m)); };
const until = (pred) => new Promise((res) => waiters.push((m) => (pred(m) ? (res(m), true) : false)));
importScripts("worker.js");
const send = (data) => self.onmessage({ data });

send({ cmd: "init", manifest: MAN, dataBase: BASE + "bitstreams/" });
await until((m) => m.type === "ready");

// ---- tokenizer ----
console.log("tokenizer");
const tj = JSON.parse(fs.readFileSync(path.join(DOCS, "bitstreams", "tokenizer.json"), "utf8"));
const T = new self.WCTokenizer.BPETokenizer(tj);
for (const t of TV.tokenizer) {
  const got = T.encode(t.text);
  check(JSON.stringify(got) === JSON.stringify(t.ids), `encode ${JSON.stringify(t.text).slice(0, 40)} (${t.ids.length} ids)`);
  check(T.decode(t.ids) === t.text, `decode round trip ${JSON.stringify(t.text).slice(0, 30)}`);
}
TV.prompts.forEach((p, i) => check(JSON.stringify(T.chatIds(p)) === JSON.stringify(TV.chat_ids[i]), `chat template == apply_chat_template: ${p.slice(0, 40)}`));

// ---- chains ----
const LABEL = { family: { instruct: "instruct_family", base: "base_family_iframe" }, delta: { instruct: "instruct_delta" }, indep: {} };
for (const c of chains) {
  console.log(`chain ${c}: ${MAN.chains[c].title}`);
  const t0 = performance.now();
  const done = until((m) => m.type === "chainDone" || m.type === "error");
  send({ cmd: "run", chain: c });
  const r = await done;
  if (r.type === "error") { check(false, "decode error: " + r.message); continue; }
  console.log(`  decoded in ${(r.decMs / 1000).toFixed(2)} s (download+decode ${((performance.now() - t0) / 1000).toFixed(2)} s)`);
  for (const f of r.results) {
    check(f.symOk && f.wOk, `${f.name}: symbols + rebuilt fp16 weights bit-exact vs Python decoder`);
    if (f.lossless) check(f.origOk, `${f.name}: lossless -> identical to the original checkpoint tensors`);
  }
  for (const [which, label] of Object.entries(LABEL[c])) {
    for (const [k, v] of (TV.forward[label] || []).entries()) {
      // first-step logits through the worker's own model
      const net = new self.WCLlama.Llama(vm.runInThisContext("models")[which]);
      let lg = null;
      for (let i = 0; i < v.prompt_ids.length; i++) lg = net.step(v.prompt_ids[i], i === v.prompt_ids.length - 1);
      const order = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]).slice(0, 10);
      const maxd = Math.max(...v.top_ids.map((id, j) => Math.abs(lg[id] - v.top_vals[j])));
      check(order[0] === v.top_ids[0] && maxd < 2e-3, `${label} #${k}: top-1 ${order[0]} (py ${v.top_ids[0]}), max |Δlogit| top-10 = ${maxd.toExponential(2)}`);
      const prompt = TV.prompts[TV.chat_ids.findIndex((ids) => JSON.stringify(ids) === JSON.stringify(v.prompt_ids))];
      const g = until((m) => m.type === "genDone" || m.type === "error");
      send({ cmd: "generate", which, prompt, maxNew: 32 });
      const res = await g;
      if (res.type === "error") { check(false, "generate error: " + res.message); continue; }
      const ref = v.greedy.slice(0, (v.greedy.findIndex((t) => t === 2 || t === 0) + 1 || v.greedy.length + 1) - 1);  // drop EOS
      const same = JSON.stringify(res.ids) === JSON.stringify(ref);
      let agree = 0; while (agree < ref.length && res.ids[agree] === ref[agree]) agree++;
      check(same, `${label} #${k}: worker greedy == PyTorch greedy (${agree}/${ref.length} tokens agree), ${res.tokPerSec.toFixed(1)} tok/s, prefill ${res.prefillMs.toFixed(0)} ms`);
    }
  }
}
server.close();
console.log(fails ? `${fails} FAILED` : "ALL OK");
process.exit(fails ? 1 : 0);
