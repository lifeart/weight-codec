// Demo worker: download (with Cache Storage = "base downloaded once") -> rans16 decode -> fp16 weight
// reconstruction -> SHA-256 verification against the Python decoder -> text generation with the
// rebuilt weights.  Everything runs off the UI thread; progress is posted back as messages.
"use strict";
importScripts("codec.js", "tokenizer.js", "llama.js");
const { parseWCDC, decodeRecord, sha256, digestOfDigests } = self.WCCodec;
const CACHE = "wcodec-demo-v1";
const TOKENIZER_FALLBACK = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json";
const SYS_PREFIX = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";

let MAN = null, DATA = "";
let models = null;            // { chain, base: {name: Uint16Array}, instruct: {...} }
let nets = {};                // model key -> { net: Llama, sys: snapshot }
let tok = null, tokSource = "";
let stopFlag = false;

const post = (m) => self.postMessage(m);
const tick = () => new Promise((r) => setTimeout(r, 0));

async function openCache() {
  if (!self.caches) return null;
  try { return await caches.open(CACHE); }
  catch (e) { post({ type: "warn", message: "Cache Storage unavailable (" + e.message + "): files will be downloaded every time." }); return null; }
}

async function fetchFile(url, expect, onProgress) {
  const cache = await openCache();
  const t0 = performance.now();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const ab = await hit.arrayBuffer();
      if (ab.byteLength === expect) return { ab, fromCache: true, ms: performance.now() - t0 };
      await cache.delete(url);           // stale entry: re-download
    }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  const total = expect || Number(r.headers.get("content-length")) || 0;
  let buf = new Uint8Array(total || 1 << 20), got = 0;
  const rd = r.body.getReader();
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    if (got + value.length > buf.length) { const nb = new Uint8Array(Math.max(buf.length * 2, got + value.length)); nb.set(buf.subarray(0, got)); buf = nb; }
    buf.set(value, got); got += value.length;
    onProgress(got, total);
  }
  if (expect && got !== expect) throw new Error(`${url}: got ${got} bytes, manifest says ${expect} (stale or wrong file?)`);
  const ab = got === buf.length ? buf.buffer : buf.slice(0, got).buffer;
  const ms = performance.now() - t0;
  if (cache) {
    try { await cache.put(url, new Response(new Blob([ab]), { headers: { "content-type": "application/octet-stream" } })); }
    catch (e) { post({ type: "warn", message: `could not store ${url.split("/").pop()} in Cache Storage (${e.message}); it will be downloaded again next time.` }); }
  }
  return { ab, fromCache: false, ms };
}

async function cacheStatus() {
  const cache = await openCache();
  const out = {};
  for (const nm of Object.keys(MAN.files)) {
    let hit = false;
    if (cache) { const r = await cache.match(DATA + nm + ".wcdc"); hit = !!r; }
    out[nm] = hit;
  }
  post({ type: "cacheStatus", cached: out, available: !!cache });
}

function chainOrder(ch) {                       // base files first, then the delta / standalone files
  const all = ch.groups.flat();
  return all.filter((n) => MAN.files[n].role === "base").concat(all.filter((n) => MAN.files[n].role !== "base"));
}

async function runChain(key) {
  const ch = MAN.chains[key];
  if (!ch) throw new Error("unknown chain " + key);
  models = null; nets = {};                     // free the previous model(s)
  const files = {};
  for (const nm of chainOrder(ch)) {
    const exp = MAN.files[nm].bytes;
    post({ type: "dl", name: nm, loaded: 0, total: exp, state: "start" });
    let last = 0;
    const r = await fetchFile(DATA + nm + ".wcdc", exp, (loaded, total) => {
      const now = performance.now();
      if (now - last > 80) { last = now; post({ type: "dl", name: nm, loaded, total, state: "progress" }); }
    });
    files[nm] = { f: await parseWCDC(r.ab), dlMs: r.ms, fromCache: r.fromCache, bytes: r.ab.byteLength };
    post({ type: "dl", name: nm, loaded: exp, total: exp, state: "done", fromCache: r.fromCache, ms: r.ms });
  }
  const W = { base: {}, instruct: {} };
  const totalW = ch.groups.reduce((a, g) => a + MAN.files[g[0]].n_weights, 0);
  let doneW = 0, lastPost = 0;
  const tDec0 = performance.now();
  const results = [];
  for (const grp of ch.groups) {
    const recs = files[grp[0]].f.records;
    const st = {};
    for (const nm of grp) st[nm] = { entropy: 0, reconstruct: 0, sha: 0, sym: [], w: [] };
    for (const r0 of recs) {
      let R = null;
      for (const nm of grp) {
        const f = files[nm].f, rec = f.byName[r0.name];
        if (!rec) throw new Error(`${nm}: record ${r0.name} missing`);
        const s = decodeRecord(f, rec, R, st[nm]);
        const t = performance.now();
        st[nm].sym.push(await sha256(s.symBytes));
        st[nm].w.push(await sha256(new Uint8Array(s.bits.buffer)));
        st[nm].sha += performance.now() - t;
        W[MAN.files[nm].model][r0.name] = s.bits;
        R = s;
      }
      doneW += r0.shape.reduce((a, b) => a * b, 1);
      const now = performance.now();
      if (now - lastPost > 100) { lastPost = now; post({ type: "dec", done: doneW, total: totalW }); await tick(); }
    }
    for (const nm of grp) {
      const m = MAN.files[nm], s = st[nm];
      const symHex = await digestOfDigests(s.sym), wHex = await digestOfDigests(s.w);
      const res = {
        name: nm, role: m.role, model: m.model, lossless: m.lossless, bytes: files[nm].bytes,
        fromCache: files[nm].fromCache, dlMs: files[nm].dlMs,
        entropyMs: s.entropy, reconstructMs: s.reconstruct, shaMs: s.sha,
        symOk: symHex === m.sha256, wOk: wHex === m.w_sha256,
        origOk: m.lossless ? wHex === m.orig_sha256 : null, symHex, wHex,
      };
      results.push(res);
      post({ type: "fileResult", res });
    }
  }
  post({ type: "dec", done: totalW, total: totalW });
  const decMs = performance.now() - tDec0;
  for (const k in files) files[k] = null;      // drop the compressed buffers
  models = { chain: key, base: W.base, instruct: W.instruct };
  post({ type: "chainDone", chain: key, decMs, results, hasBase: Object.keys(W.base).length > 0 });
}

async function loadTokenizer() {
  if (tok) return tok;
  let tj = null, src = DATA + "tokenizer.json";
  try {
    const r = await fetch(src);
    if (!r.ok) throw new Error("HTTP " + r.status);
    tj = await r.json();
  } catch (e) {
    post({ type: "warn", message: `tokenizer.json not found next to the bitstreams (${e.message}); using the official copy from the Hugging Face Hub.` });
    src = TOKENIZER_FALLBACK;
    const r = await fetch(src);
    if (!r.ok) throw new Error("tokenizer download failed: HTTP " + r.status);
    tj = await r.json();
  }
  tok = new self.WCTokenizer.BPETokenizer(tj); tokSource = src;
  return tok;
}

function getNet(which) {
  if (!models || !Object.keys(models[which]).length) throw new Error("decode a chain with a " + which + " model first");
  if (!nets[which]) nets[which] = { net: new self.WCLlama.Llama(models[which]), sys: null };
  return nets[which];
}

async function generate({ which, prompt, maxNew }) {
  stopFlag = false;
  const T = await loadTokenizer();
  const N = getNet(which), net = N.net;
  const ids = T.chatIds(prompt), sys = T.encode(SYS_PREFIX);
  const t0 = performance.now();
  let reused = false;
  if (N.sys && sys.every((v, i) => ids[i] === v)) { net.restore(N.sys); reused = true; }
  else {
    net.reset();
    for (const id of sys) net.step(id, false);
    N.sys = net.snapshot();
  }
  let logits = null;
  for (let i = sys.length; i < ids.length; i++) logits = net.step(ids[i], i === ids.length - 1);
  const prefillMs = performance.now() - t0;
  post({ type: "genStart", promptTokens: ids.length, prefillMs, sysReused: reused, tokSource });
  const out = [], dec = new TextDecoder();
  let text = "";
  const t1 = performance.now();
  for (let k = 0; k < maxNew; k++) {
    const id = self.WCLlama.argmax(logits);
    if (id === 2 || id === 0) break;                     // <|im_end|> / <|endoftext|>
    out.push(id);
    const piece = dec.decode(new Uint8Array(T.tokenBytes(id)), { stream: true });
    text += piece;
    post({ type: "tok", piece, n: out.length, ms: performance.now() - t1 });
    await tick();
    if (stopFlag) break;
    if (k + 1 < maxNew) logits = net.step(id, true);
  }
  text += dec.decode();
  const genMs = performance.now() - t1;
  post({ type: "genDone", ids: out, text, genMs, prefillMs, tokPerSec: out.length / (genMs / 1000), stopped: stopFlag });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === "init") { MAN = m.manifest; DATA = m.dataBase; await cacheStatus(); post({ type: "ready" }); }
    else if (m.cmd === "cacheStatus") await cacheStatus();
    else if (m.cmd === "run") await runChain(m.chain);
    else if (m.cmd === "generate") await generate(m);
    else if (m.cmd === "stop") stopFlag = true;
    else if (m.cmd === "clearCache") { if (self.caches) await caches.delete(CACHE); await cacheStatus(); }
    else throw new Error("unknown command " + m.cmd);
  } catch (err) {
    post({ type: "error", cmd: m.cmd, message: err && err.message ? err.message : String(err) });
  }
};
