// weight-codec demo page (UI thread).  All heavy work happens in js/worker.js.
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const MB = (b) => (b / 1e6).toFixed(1) + " MB";
const fx = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pct = (x) => (100 * x).toFixed(1) + " %";

const CHAIN_ORDER = ["delta", "family", "indep"];
const GEN_LABEL = { delta: { instruct: "instruct_delta", base: "base_original" },
                    family: { instruct: "instruct_family", base: "base_family_iframe" }, indep: {} };
// card text; Q = measured quality of this chain's lossy Instruct file, P = the file it replaced (previous coder, REPORT §14.3)
const ratio = (Q, P) => (P && P.chat_kl && Q.chat_kl) ? (P.chat_kl / Q.chat_kl).toFixed(1) + "×" : "—";
const CARD_TEXT = {
  delta: { tag: "near-lossless · recommended",
           q: (Q, P) => `Embeddings/norms bit-exact; Instruct linears approximate: chat-KL ${fx(Q.chat_kl, 4)}, top-1 agreement ${pct(Q.chat_top1)} with the original Instruct ` +
             `(previous delta coder: chat-KL ${fx(P.chat_kl, 4)} at a similar size — ${ratio(Q, P)} lower now).` },
  family: { tag: "smallest",
            q: (Q, P) => `Both models lossy (int4-class): Instruct chat-KL ${fx(Q.chat_kl, 3)}, top-1 ${pct(Q.chat_top1)} — ${ratio(Q, P)} lower than the previous ` +
              `row-grid coder (chat-KL ${fx(P.chat_kl, 2)}) at about the same size. Still visibly below the original on a model this small (see examples).` },
  indep: { tag: "control",
           q: (Q) => `Nothing reused from a base, same coder: chat-KL ${fx(Q.chat_kl, 3)}, top-1 ${pct(Q.chat_top1)}. Shows what reusing the base saves.` },
};

let MAN, GEN, RES, SUM, worker, DATA;
let chain = "delta", busy = false, decoded = null, cached = {}, cacheAvailable = false;
let session = { net: 0, cache: 0 };
let genRun = null;

function dataBase() {
  const q = new URLSearchParams(location.search).get("data");
  const b = q || (window.WC_CONFIG && window.WC_CONFIG.dataBase) || "bitstreams/";
  return new URL(b.endsWith("/") ? b : b + "/", location.href).href;
}

function chainFiles(c) { return MAN.chains[c].groups.flat(); }
function ordered(c) { const a = chainFiles(c); return a.filter((n) => MAN.files[n].role === "base").concat(a.filter((n) => MAN.files[n].role !== "base")); }
function sumBytes(c, pred) { return chainFiles(c).filter((n) => pred(MAN.files[n])).reduce((a, n) => a + MAN.files[n].bytes, 0); }

// ------------------------------------------------------------------ cards + tiles
function renderCards() {
  const root = $("#chains"); root.innerHTML = "";
  for (const c of CHAIN_ORDER) {
    const ch = MAN.chains[c], T = CARD_TEXT[c], Q = ch.quality;
    const baseB = ch.base_bytes, deltaB = ch.delta_bytes || ch.standalone_bytes;
    const baseCached = c !== "indep" && chainFiles(c).filter((n) => MAN.files[n].role === "base").every((n) => cached[n]);
    const el = document.createElement("div");
    el.className = "card"; el.tabIndex = 0; el.setAttribute("role", "radio"); el.setAttribute("aria-checked", String(c === chain));
    el.innerHTML = `<div class="c-tag">${esc(T.tag)}</div><div class="c-title">${esc(ch.title)}</div>
      <div class="c-sizes">${c === "indep" ? `standalone Instruct <b>${MB(deltaB)}</b>` : `base <b>${MB(baseB)}</b> + Instruct delta <b>${MB(deltaB)}</b>`}</div>
      <div class="c-q">${esc(T.q(Q, ch.prev_quality || {}))}</div>
      <div class="c-cached ${baseCached ? "ok" : "muted"}">${c === "indep" ? "" : baseCached ? "✓ base already cached in this browser" : "base not cached yet"}</div>`;
    const pick = () => { if (busy) return; chain = c; renderCards(); renderTiles(); };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); pick(); } });
    root.appendChild(el);
  }
}

function renderTiles() {
  const ch = MAN.chains[chain], files = chainFiles(chain);
  const baseFiles = files.filter((n) => MAN.files[n].role === "base");
  const delFiles = files.filter((n) => MAN.files[n].role !== "base");
  const allCached = (fs) => fs.length && fs.every((n) => cached[n]);
  if (chain === "indep") {
    $("#t-base").textContent = "0 MB"; $("#t-base-sub").textContent = "no base is reused in the control";
  } else {
    $("#t-base").textContent = MB(ch.base_bytes);
    $("#t-base-sub").innerHTML = (allCached(baseFiles) ? `<span class="ok">✓ already in this browser's cache — 0 MB to download</span>`
      : `not cached yet · ${MAN.chains[chain].base_lossless ? "lossless fp16 base" : "lossy int4-class I-frame"}`) +
      `<br>${baseFiles.map((n) => `${n} ${MB(MAN.files[n].bytes)}`).join(" + ")}`;
  }
  const dB = delFiles.reduce((a, n) => a + MAN.files[n].bytes, 0);
  $("#t-delta").textContent = MB(dB);
  $("#t-delta-sub").innerHTML = delFiles.map((n) => `${n} ${MB(MAN.files[n].bytes)}${MAN.files[n].lossless ? " (lossless)" : ""}`).join(" + ");
  const Q = ch.quality;
  $("#t-out-sub").innerHTML = `fp16, ${MB(2 * 134515008)} in memory · lossless parts bit-exact · linears approximate (chat-KL ${fx(Q.chat_kl)})`;
  const off = MAN.original.instruct_bf16_safetensors_bytes;
  const int4 = RES.files.T_gptq4_packed.MB + RES.files.T_rest_raw16.MB;
  const pFam = RES.pairs.find((p) => p.kind === "default" && p.scenario.startsWith("family"));
  const pInt4 = RES.pairs.find((p) => p.scenario.startsWith("two standard GPTQ-int4"));
  $("#refline").innerHTML = `For scale: the official Instruct checkpoint is <b>${MB(off)}</b> (bf16 safetensors); a standard GPTQ-int4 g64 file of it ` +
    `(rest fp16) is <b>${int4.toFixed(1)} MB</b> at chat-KL ${fx(RES.files.T_gptq4_packed.chat_kl, 2)}. Storing <i>both</i> models: family ` +
    `<b>${pFam.pair_MB.toFixed(1)} MB</b> (chat-KL ${fx(pFam.ft_chat_kl, 3)}) vs two standard int4 files ${pInt4.pair_MB.toFixed(1)} MB (chat-KL ${fx(pInt4.ft_chat_kl, 2)}): ` +
    `<b>${(100 * (pFam.pair_MB / pInt4.pair_MB - 1)).toFixed(0)} %</b> bytes.`;
}

function renderWhatsNew() {          // measured numbers of this build vs the files it replaced (manifest: quality / prev_quality)
  const f = MAN.chains.family, d = MAN.chains.delta;
  $("#whatsnew").innerHTML = `<b>New coder (REPORT §17):</b> quantization steps are now set per output row <i>and per input column</i> ` +
    `(Fisher-weighted for the base, Hessian-weighted for the delta). At about the same size, the smallest configuration's Instruct ` +
    `chat-KL fell from ${fx(f.prev_quality.chat_kl, 2)} to <b>${fx(f.quality.chat_kl, 3)}</b> and the near-lossless delta's from ` +
    `${fx(d.prev_quality.chat_kl, 3)} to <b>${fx(d.quality.chat_kl, 3)}</b> (lower = closer to the original).`;
}

function renderSession() {
  $("#session").textContent = `this session: ${MB(session.net)} from the network · ${MB(session.cache)} from your browser cache` +
    (cacheAvailable ? "" : " (Cache Storage unavailable here)");
}

// ------------------------------------------------------------------ download / decode
function buildProgress() {
  const root = $("#progress"); root.innerHTML = "";
  for (const n of ordered(chain)) {
    const f = MAN.files[n];
    const row = document.createElement("div"); row.className = "prow"; row.id = "p-" + n.replace(/\./g, "_");
    row.innerHTML = `<span class="pname" title="${esc(f.desc)}">${esc(n)}</span><span><span class="badge ${f.role === "delta" ? "delta" : ""}">${f.role}</span></span>
      <span class="bar"><i></i></span><span class="pstat">${MB(f.bytes)} · waiting</span>`;
    root.appendChild(row);
  }
  const row = document.createElement("div"); row.className = "prow"; row.id = "p-decode";
  row.innerHTML = `<span class="pname">decode + rebuild + SHA-256</span><span></span><span class="bar"><i></i></span><span class="pstat">waiting</span>`;
  root.appendChild(row);
}
const prow = (n) => document.getElementById("p-" + n.replace(/\./g, "_"));

function onDl(m) {
  const r = prow(m.name); if (!r) return;
  const bar = $(".bar", r), st = $(".pstat", r);
  $("i", bar).style.width = (100 * m.loaded / m.total).toFixed(1) + "%";
  if (m.state === "start") st.textContent = `${MB(m.total)} · connecting…`;
  else if (m.state === "progress") st.textContent = `${MB(m.loaded)} / ${MB(m.total)}`;
  else if (m.state === "done") {
    if (m.fromCache) { bar.classList.add("cached"); st.textContent = `${MB(m.total)} · from browser cache`; session.cache += m.total; cached[m.name] = true; }
    else { st.textContent = `${MB(m.total)} · network ${(m.total / 1e6 / (m.ms / 1000)).toFixed(1)} MB/s`; session.net += m.total; cached[m.name] = cacheAvailable; }
    renderSession();
  }
}

function onDec(m) {
  const r = prow("decode"); if (!r) return;
  $("i", $(".bar", r)).style.width = (100 * m.done / m.total).toFixed(1) + "%";
  $(".pstat", r).textContent = `${(m.done / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} M weights`;
}

function onChainDone(m) {
  busy = false; decoded = { chain: m.chain, hasBase: m.hasBase };
  const ch = MAN.chains[m.chain], R = m.results, Q = ch.quality;
  const nW = R.reduce((a, r) => a + MAN.files[r.name].n_weights, 0);
  const tb = $("#filetable tbody"); tb.innerHTML = "";
  let entropy = 0, rebuild = 0, shams = 0;
  for (const r of R) {
    entropy += r.entropyMs; rebuild += r.reconstructMs; shams += r.shaMs;
    const tr = document.createElement("tr");
    const orig = r.lossless ? (r.origOk ? `<span class="ok">✓ identical</span>` : `<span class="bad">✗ differs</span>`) : `<span class="approx">≈ lossy (see above)</span>`;
    tr.innerHTML = `<td title="${esc(MAN.files[r.name].desc)}">${esc(r.name)}</td><td>${r.role}</td><td>${(r.bytes / 1e6).toFixed(2)}</td>
      <td>${r.fromCache ? "cache" : "network"}</td><td>${r.dlMs.toFixed(0)}</td><td>${r.entropyMs.toFixed(0)}</td><td>${r.reconstructMs.toFixed(0)}</td>
      <td>${r.shaMs.toFixed(0)}</td><td class="${r.symOk ? "ok" : "bad"}">${r.symOk ? "✓" : "✗"}</td><td class="${r.wOk ? "ok" : "bad"}">${r.wOk ? "✓" : "✗"}</td><td>${orig}</td>`;
    tb.appendChild(tr);
  }
  const allSym = R.every((r) => r.symOk), allW = R.every((r) => r.wOk);
  const ll = R.filter((r) => r.lossless), lossy = R.filter((r) => !r.lossless);
  const allOrig = ll.every((r) => r.origOk);
  const bc = MAN.original.base_cast.n_inexact_bf16_to_fp16, tc = MAN.original.instruct_cast.n_inexact_bf16_to_fp16;
  const items = [
    [allSym, "Entropy decoding is bit-exact.", `SHA-256 of the decoded symbols of all ${R.length} files equals the Python decoder's (lattice indices for quantized tensors, fp16 bit patterns for lossless ones).`],
    [allW, "Rebuilt fp16 weights = the Python decoder's, bit for bit.", `SHA-256 of every rebuilt fp16 tensor (${(nW / 1e6).toFixed(1)} M weights across the chain) matches the reference digests written by <code>phase2/demo_build.py</code>.`],
    [allOrig, "Lossless parts are identical to the original checkpoint.", `${ll.map((r) => `<code>${esc(r.name)}</code>`).join(", ")}: the rebuilt tensors have the same SHA-256 as the original tensors (computed from the official checkpoint, not from a decode). “Original” = the official bf16 checkpoint cast to fp16, the codec's reference; that cast itself changes ${bc.toLocaleString("en")} (base) / ${tc.toLocaleString("en")} (Instruct) of ${(MAN.original.base_cast.n_params / 1e6).toFixed(1)} M values, all |w| &lt; 7.6·10⁻⁶.`],
  ];
  const lossyNames = lossy.map((r) => `<code>${esc(r.name)}</code>`).join(", ");
  let approxTxt = `${lossyNames} ${lossy.length > 1 ? "are" : "is"} quantized: the rebuilt Instruct is an <b>approximation</b> of the original — chat-KL <b>${fx(Q.chat_kl, 4)}</b>, top-1 agreement ${pct(Q.chat_top1)} on the original's own answers (48 prompts), WikiText KL ${fx(Q.kl, 4)} — measured offline on models decoded from these files (REPORT §17).`;
  if (m.chain === "family") approxTxt += ` The base is lossy too (int4-class I-frame: WikiText PPL ${fx(RES.files[RES.names.iframe].ppl, 2)} vs ${fx(RES.files.B_fp16.ppl, 2)} for fp16).`;
  if (m.chain === "delta") approxTxt += ` The base here is lossless, so the rebuilt base is bit-identical to the fp16 base.`;
  const ul = $("#checks"); ul.innerHTML = "";
  for (const [ok, k, txt] of items) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"}</span> <span class="k">${k}</span> ${txt}`; ul.appendChild(li);
  }
  const li = document.createElement("li");
  li.innerHTML = `<span class="approx">≈</span> <span class="k">Lossy parts are NOT identical.</span> ${approxTxt}`; ul.appendChild(li);
  $("#timing-note").innerHTML = `All timings were measured in <b>your</b> browser just now (one JavaScript thread in a Web Worker): entropy decode ${(entropy / 1000).toFixed(2)} s, ` +
    `rebuild ${(rebuild / 1000).toFixed(2)} s, SHA-256 ${(shams / 1000).toFixed(2)} s; total decode ${(m.decMs / 1000).toFixed(2)} s for ${(nW / 1e6).toFixed(1)} M weights. ` +
    `Native multi-threaded / GPU decoders of the same container are much faster (REPORT §14.6).`;
  $("#verify").hidden = false;
  const ok = allSym && allW && allOrig;
  $("#demo-log").className = "log " + (ok ? "" : "err");
  $("#demo-log").innerHTML = `${ok ? "✓" : "✗"} <b>${esc(ch.title)}</b>: decoded and verified in ${(m.decMs / 1000).toFixed(2)} s. ` +
    (ok ? "Generate with the rebuilt weights in step 4." : "MISMATCH — see the table.");
  renderCards(); renderTiles(); setButtons();
  $("#gen-out").textContent = "Ready: pick a prompt and press Generate.";
  const bRadio = document.querySelector('input[name="which"][value="base"]');
  bRadio.disabled = !m.hasBase; if (!m.hasBase) document.querySelector('input[name="which"][value="instruct"]').checked = true;
}

function setButtons() {
  $("#go").disabled = busy;
  $("#clear").disabled = busy || !cacheAvailable;
  $("#gen-go").disabled = busy || !decoded || !!genRun;
  $("#gen-stop").disabled = !genRun;
}

// ------------------------------------------------------------------ generation
function genStart() {
  const which = document.querySelector('input[name="which"]:checked').value;
  const prompt = $("#prompt").value.trim();
  const maxNew = Math.max(1, Math.min(200, parseInt($("#maxnew").value, 10) || 64));
  if (!prompt) return;
  genRun = { which, prompt, maxNew, text: "", chain: decoded.chain };
  $("#gen-out").textContent = ""; $("#gen-cmp").textContent = ""; $("#gen-stat").textContent = "prefill…";
  setButtons();
  worker.postMessage({ cmd: "generate", which, prompt, maxNew });
}

function onGenDone(m) {
  const g = genRun; genRun = null; setButtons();
  $("#gen-stat").textContent = `${m.ids.length} tokens · ${m.tokPerSec.toFixed(1)} tok/s · prefill ${(m.prefillMs / 1000).toFixed(2)} s${m.stopped ? " · stopped" : ""}`;
  const label = (GEN_LABEL[g.chain] || {})[g.which];
  const pi = GEN.prompts.indexOf(g.prompt);
  const cmp = $("#gen-cmp");
  if (label && pi >= 0 && g.maxNew === GEN.max_new_tokens && !m.stopped) {
    const ref = GEN.outputs[label][pi];
    if (ref === m.text) cmp.innerHTML = `<span class="ok">✓ identical</span> to the offline PyTorch answer for the same rebuilt weights (<code>${label}</code> column below).`;
    else {
      let k = 0; while (k < ref.length && ref[k] === m.text[k]) k++;
      cmp.innerHTML = `<span class="approx">≠</span> differs from the offline PyTorch answer (<code>${label}</code>) from character ${k}: fp32 summation order differs between JS and PyTorch, which can flip a near-tie in greedy decoding.`;
    }
  } else if (!label) cmp.innerHTML = `<span class="muted">No offline reference for this configuration (the precomputed table covers the original models and the two base + delta configurations).</span>`;
  else cmp.innerHTML = `<span class="muted">No offline reference to compare with: pick a preset prompt with ${GEN.max_new_tokens} new tokens (and let it finish) for a check.</span>`;
}

// ------------------------------------------------------------------ static content
function renderExamples() {
  $("#ex-how").textContent = "Precomputed once, offline — " + GEN.how + ` ${GEN.max_new_tokens} new tokens. Not live: the live path is step 4.`;
  const cols = [["base_original", "base (original fp16)"], ["instruct_original", "Instruct (original fp16)"],
                ["instruct_delta", "Instruct rebuilt: fp16 base + Δ (column grid, c=1.6)"], ["instruct_family", "Instruct rebuilt: int4-class family (Fisher-grid P-frame)"]];
  $("#extable thead").innerHTML = "<tr><th>prompt</th>" + cols.map(([, t]) => `<th>${esc(t)}</th>`).join("") + "</tr>";
  $("#extable tbody").innerHTML = GEN.prompts.map((p, i) => "<tr><td>" + esc(p) + "</td>" +
    cols.map(([k]) => `<td>${esc(GEN.outputs[k][i])}</td>`).join("") + "</tr>").join("");
}

function renderNumbers() {
  const F = RES.files, P = RES.pairs, N = RES.names;
  const rows = P.map((p) => [p.scenario, p, p.kind === "default"]);
  const max = Math.max(...rows.map((r) => r[1].pair_MB));
  let h = `<h3>Storing the pair {base, Instruct} — every tensor of both models, MB on disk</h3><div class="hbars" role="img" aria-label="pair storage in MB">`;
  for (const [lab, p, hi] of rows) {
    const w = (100 * p.pair_MB / max).toFixed(2);
    const l = hi ? `<b>${esc(lab)}</b>` : esc(lab);
    h += `<div class="hb ${hi ? "hi" : ""}" title="${esc(lab)}: ${p.pair_MB.toFixed(1)} MB · base PPL ${fx(p.base_ppl, 2)} · Instruct chat-KL ${fx(p.ft_chat_kl, 4)}">
      <span class="lab">${l}</span><span class="track"><span class="fill" style="width:${w}%"></span><span class="val" style="left:calc(${w}% + 6px)">${p.pair_MB.toFixed(1)} MB · chat-KL ${fx(p.ft_chat_kl, 3)}</span></span></div>`;
  }
  const fam = P.find((p) => p.kind === "default" && p.scenario.startsWith("family"));
  const famPrev = P.find((p) => p.kind === "prev" && p.scenario.includes("family"));
  const int4 = P.find((p) => p.scenario.startsWith("two standard GPTQ-int4"));
  const indep = P.find((p) => p.kind === "control");
  h += `</div><p class="small muted">Highlighted = this demo's files (the codec's default coders). Family vs two standard int4 files: <b>${(100 * (fam.pair_MB / int4.pair_MB - 1)).toFixed(0)} %</b> bytes ` +
    `at chat-KL ${fx(fam.ft_chat_kl, 3)} vs ${fx(int4.ft_chat_kl, 3)}; vs both models coded standalone with the same coder: ${(100 * (fam.pair_MB / indep.pair_MB - 1)).toFixed(0)} %. ` +
    `Same pair size as the previous row-grid family (${famPrev.pair_MB.toFixed(1)} MB), ${(famPrev.ft_chat_kl / fam.ft_chat_kl).toFixed(1)}× lower Instruct chat-KL. ` +
    `Base PPL (WikiText-2): fp16 ${fx(F.B_fp16.ppl, 2)}, Fisher-grid I-frame ${fx(F[N.iframe].ppl, 2)}, previous row-grid I-frame ${fx(F[N.prev.iframe].ppl, 2)}, int4 ${fx(F.B_gptq4_packed.ppl, 2)}.</p>`;

  const off = MAN.original.instruct_bf16_safetensors_bytes / 1e6;
  const tr = (cells) => "<tr>" + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>";
  const row = (lab, lin, rest) => tr([lab, (F[lin].MB + F[rest].MB).toFixed(1), fx(F[lin].real_bpw, 2), fx(F[lin].chat_kl, 4), pct(F[lin].chat_top1)]);
  h += `<h3>What a client that already has the base downloads to get the Instruct model</h3><div class="tw"><table>
    <thead><tr><th>what is downloaded</th><th>MB</th><th>linears bpw</th><th>Instruct chat-KL</th><th>top-1 agreement</th></tr></thead><tbody>
    ${tr(["official Instruct checkpoint (bf16 safetensors)", off.toFixed(1), "16", "0 (it is the original)", "100 %"])}
    ${row("standard GPTQ-int4 g64 file + fp16 rest", "T_gptq4_packed", "T_rest_raw16")}
    ${row("standalone Fisher-grid c=0.35 + lossless rest (control: no base reuse)", N.indep, "T_rest_ll16")}
    ${row("<b>P-frame + XOR rest</b> (int4-class Fisher-grid base I-frame cached)", N.pframe, "T_rest_xor_base")}
    ${row("<span class='muted'>previous coder: row-grid P-frame + XOR rest</span>", N.prev.pframe, "T_rest_xor_base")}
    ${row("<b>full-rank Δ, column grid c=1.6 + XOR rest</b> (fp16 base cached)", N.delta, "T_rest_xor_base")}
    ${row("<span class='muted'>previous coder: full-rank Δ, GPTQ row grid c=0.12 + XOR rest</span>", N.prev.delta, "T_rest_xor_base")}
    </tbody></table></div>
    <p class="small muted">Linears alone: the P-frame is ${F[N.pframe].MB.toFixed(1)} MB (${fx(F[N.pframe].real_bpw, 2)} bpw) vs ${F.T_gptq4_packed.MB.toFixed(1)} MB for GPTQ-int4
    (${(100 * (F[N.pframe].MB / F.T_gptq4_packed.MB - 1)).toFixed(0)} %). The ${F.T_rest_xor_base.MB.toFixed(1)} MB lossless embedding/norm XOR stream dominates both deltas — lossy
    embedding coding was not evaluated. What changed vs the previous coder: quantization steps are now set per row <i>and per input column</i> (Fisher-weighted for the base,
    Hessian-weighted for the delta; REPORT §16–§17) — the file format only gained one small fp16 section per tensor.</p>`;

  h += `<h3>Full-rank Δ + entropy coding vs a LoRA-style low-rank Δ (same fp16 base)</h3><div class="tw"><table>
    <thead><tr><th>full-rank Δ</th><th>MB</th><th>chat-KL</th><th>LoRA-style</th><th>MB</th><th>chat-KL</th><th>KL ratio</th><th>bytes ratio</th></tr></thead><tbody>`;
  for (const [d, l] of RES.delta_vs_lora) {
    h += tr([d, F[d].MB.toFixed(1), fx(F[d].chat_kl, 4), l, F[l].MB.toFixed(1), fx(F[l].chat_kl, 4), (F[l].chat_kl / F[d].chat_kl).toFixed(1) + "× lower", pct(F[d].MB / F[l].MB)]);
  }
  h += `</tbody></table></div><p class="small muted">Source: ${esc(RES.source)}. ${esc(RES.chat_set)}.</p>`;
  $("#numbers-body").innerHTML = h;
}

function renderAllResults() {
  const root = $("#results");
  for (const sec of SUM.sections) {
    const div = document.createElement("section");
    let h = `<h3>${esc(sec.title)}</h3><p class="small">${esc(sec.md || "")}</p>`;
    if (sec.table) {
      h += "<div class='tw'><table><thead><tr>" + sec.table.headers.map((x) => `<th>${esc(x)}</th>`).join("") + "</tr></thead><tbody>";
      for (const r of sec.table.rows) h += "<tr>" + r.map((x) => `<td>${x === null ? "—" : esc(x)}</td>`).join("") + "</tr>";
      h += "</tbody></table></div>";
    }
    for (const img of sec.images || []) h += `<img src="figures/${esc(img)}" alt="${esc(sec.title)}" loading="lazy">`;
    div.innerHTML = h; root.appendChild(div);
  }
}

// ------------------------------------------------------------------ data host check
async function checkData() {
  const note = $("#data-note");
  const problems = [];
  if (!window.isSecureContext || !(window.crypto && crypto.subtle)) problems.push("this page is not in a secure context (https or localhost), so SHA-256 (WebCrypto) is unavailable");
  if (typeof DecompressionStream === "undefined") problems.push("this browser lacks DecompressionStream (needs Chrome 80+, Firefox 113+, Safari 16.4+)");
  if (typeof Worker === "undefined") problems.push("Web Workers are unavailable");
  let reachable = true;
  try {
    const r = await fetch(DATA + "B_rest_ll16.wcdc", { method: "HEAD" });
    if (!r.ok) { reachable = false; problems.push(`the compressed files are not reachable at <code>${esc(DATA)}</code> (HTTP ${r.status})`); }
  } catch (e) { reachable = false; problems.push(`the compressed files are not reachable at <code>${esc(DATA)}</code> (${esc(e.message)})`); }
  if (problems.length) {
    note.hidden = false;
    note.innerHTML = `Live decoding is unavailable here: ${problems.join("; ")}. ` + (reachable ? "" :
      `The bitstreams are not stored in git; this copy of the page needs a data host (see “Browser demo” in the project README: ` +
      `stage them next to the page, or point <code>config.js</code> / <code>?data=</code> at a public host with CORS). `) +
      `The numbers and the precomputed answers below do not depend on it.`;
  }
  return reachable;
}

// ------------------------------------------------------------------ boot
window.addEventListener("load", async () => {
  try {
    [MAN, GEN, RES, SUM] = await Promise.all(["data/manifest.json", "data/generations.json", "data/results.json", "data/summary.json"]
      .map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + ": HTTP " + r.status); return r.json(); })));
  } catch (e) {
    $("#demo-log").className = "log err"; $("#demo-log").textContent = "Failed to load the page data: " + e.message; return;
  }
  DATA = dataBase();
  renderCards(); renderTiles(); renderSession(); renderWhatsNew(); renderExamples(); renderNumbers(); renderAllResults();
  for (const p of GEN.prompts) {
    const b = document.createElement("button"); b.className = "chip"; b.textContent = p;
    b.addEventListener("click", () => { $("#prompt").value = p; $("#maxnew").value = GEN.max_new_tokens; });
    $("#presets").appendChild(b);
  }
  const reachable = await checkData();
  worker = new Worker("js/worker.js");
  worker.onerror = (e) => { busy = false; genRun = null; setButtons(); $("#demo-log").className = "log err"; $("#demo-log").textContent = "Worker failed: " + (e.message || "script error"); };
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { setButtons(); if (!reachable) $("#go").title = "data host not reachable — see the note at the top"; }
    else if (m.type === "cacheStatus") { cached = m.cached; cacheAvailable = m.available; renderCards(); renderTiles(); renderSession(); setButtons(); }
    else if (m.type === "dl") onDl(m);
    else if (m.type === "dec") onDec(m);
    else if (m.type === "fileResult") { /* rendered at chainDone */ }
    else if (m.type === "chainDone") onChainDone(m);
    else if (m.type === "genStart") $("#gen-stat").textContent = `prompt ${m.promptTokens} tokens · prefill ${(m.prefillMs / 1000).toFixed(2)} s${m.sysReused ? " (system prompt KV reused)" : ""} · generating…`;
    else if (m.type === "tok") { genRun.text += m.piece; $("#gen-out").textContent = genRun.text; $("#gen-stat").textContent = `${m.n} tokens · ${(m.n / (m.ms / 1000)).toFixed(1)} tok/s`; }
    else if (m.type === "genDone") onGenDone(m);
    else if (m.type === "warn") { const l = $("#demo-log"); l.innerHTML += `<br><span class="approx">note:</span> ${esc(m.message)}`; }
    else if (m.type === "error") {
      busy = false; if (m.cmd === "generate") { genRun = null; $("#gen-out").textContent = "Error: " + m.message; }
      setButtons(); $("#demo-log").className = "log err"; $("#demo-log").textContent = `Error (${m.cmd}): ${m.message}`;
    }
  };
  worker.postMessage({ cmd: "init", manifest: MAN, dataBase: DATA });
  $("#go").addEventListener("click", () => {
    busy = true; decoded = null; setButtons(); buildProgress(); $("#verify").hidden = true;
    $("#demo-log").className = "log"; $("#demo-log").textContent = "downloading (base first, then the delta)…";
    worker.postMessage({ cmd: "run", chain });
  });
  $("#clear").addEventListener("click", () => { worker.postMessage({ cmd: "clearCache" }); $("#demo-log").textContent = "Cached files removed from this browser."; });
  $("#gen-go").addEventListener("click", genStart);
  $("#gen-stop").addEventListener("click", () => worker.postMessage({ cmd: "stop" }));
});
