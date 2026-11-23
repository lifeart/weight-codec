// JS half of phase2/test_codec_js.py: decode every record of the fixture files with the browser decoder
// (docs/js/codec.js), in the given reference chains, and print per-file symbol / weight SHA-256 (same digest
// definition as phase2/demo_build.py) or the error message, as JSON on stdout.
// usage: node phase2/test_codec_js.mjs <fixture dir>     (fixture.json lists files, chains and the expected-error records)
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
globalThis.self = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(ROOT, "docs", "js", "codec.js"), "utf8"), { filename: "codec.js" });
const { parseWCDC, decodeRecord, sha256, digestOfDigests } = self.WCCodec;
const dir = process.argv[2];
const fx = JSON.parse(fs.readFileSync(path.join(dir, "fixture.json"), "utf8"));
const load = async (nm) => {
  const b = fs.readFileSync(path.join(dir, nm + ".wcdc"));
  return parseWCDC(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const out = { files: {}, errors: {} };
for (const chain of fx.chains) {                 // [file, file, ...]: each file's records reference the previous file's
  let prev = null;
  for (const nm of chain) {
    const f = await load(nm), st = {}, sym = [], w = [];
    const tim = { entropy: 0, reconstruct: 0 };
    for (const rec of f.records) {
      const s = decodeRecord(f, rec, prev ? prev[rec.name] : null, tim);
      sym.push(await sha256(s.symBytes)); w.push(await sha256(new Uint8Array(s.bits.buffer)));
      st[rec.name] = s;
    }
    out.files[nm] = { sha256: await digestOfDigests(sym), w_sha256: await digestOfDigests(w) };
    prev = st;
  }
}
for (const [nm, refnm] of fx.bad) {               // malformed records: must throw, never return weights
  const f = await load(nm);
  let ref = null;
  if (refnm) { const fr = await load(refnm); ref = {}; for (const r of fr.records) ref[r.name] = decodeRecord(fr, r, null, { entropy: 0, reconstruct: 0 }); }
  for (const rec of f.records) {
    try { decodeRecord(f, rec, ref ? ref[rec.name] : null, { entropy: 0, reconstruct: 0 }); out.errors[`${nm}/${rec.name}`] = null; }
    catch (e) { out.errors[`${nm}/${rec.name}`] = String(e.message || e); }
  }
}
console.log(JSON.stringify(out));
