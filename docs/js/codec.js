// weight-codec browser decoder: WCDC container + rans16 entropy decoder + record reconstruction.
// Mirrors wcodec/container.py, wcodec/rans16.py and wcodec/codecs.py::decode_record bit for bit:
// the reconstructed fp16 weights are computed with the same fp32 arithmetic and the same
// round-to-nearest-even fp32 -> fp16 cast as PyTorch, so their SHA-256 can be compared with the
// Python decoder's.  Plain script: works in a Web Worker (importScripts), a page, or Node (vm).
"use strict";
(function (root) {

  // ---------------------------------------------------------------- container
  async function inflateText(u8) {
    const ds = new DecompressionStream("deflate");            // zlib (RFC 1950) stream
    const buf = await new Response(new Blob([u8]).stream().pipeThrough(ds)).arrayBuffer();
    return new TextDecoder().decode(buf);
  }

  async function parseWCDC(ab) {
    const u8 = new Uint8Array(ab), dv = new DataView(ab);
    if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "WCDC") throw new Error("not a WCDC file");
    if (u8[4] !== 1) throw new Error("unsupported WCDC version " + u8[4]);
    const hl = dv.getUint32(5, true);
    const hdr = JSON.parse(await inflateText(u8.subarray(9, 9 + hl)));
    hdr.base = 9 + hl; hdr.ab = ab; hdr.dv = dv; hdr.u8 = u8;
    hdr.byName = Object.create(null);
    for (const r of hdr.records) hdr.byName[r.name] = r;
    return hdr;
  }

  // ---------------------------------------------------------------- rans16 (32-bit state, 16-bit words, 15-bit freqs)
  function readVarints(u8, pos, n) {
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let v = 0, shift = 0, b;
      do { b = u8[pos++]; v += (b & 0x7f) * 2 ** shift; shift += 7; } while (b >= 0x80);
      out[i] = v;
    }
    return [out, pos];
  }

  // Returns Int32Array of symbols.  off = absolute byte offset of the section in the file.
  function decodeRans16(f, off) {
    const dv = f.dv, u8 = f.u8;
    const lo = dv.getUint32(off, true), hi = dv.getUint32(off + 4, true);
    if ((hi >>> 24) !== 1) throw new Error("section is not rans16 (coder id " + (hi >>> 24) + ")");
    const n = lo + (hi & 0xffffff) * 4294967296;
    const smin = dv.getInt32(off + 8, true), nalpha = dv.getUint32(off + 12, true);
    const chunk = dv.getUint32(off + 16, true), nch = dv.getUint32(off + 20, true);
    let [freq, pos] = readVarints(u8, off + 24, nalpha);
    const out = new Int32Array(n);
    if (n === 0) return out;
    if (nalpha === 1) { out.fill(smin); return out; }
    const states = new Uint32Array(f.ab.slice(pos, pos + 4 * nch)); pos += 4 * nch;
    const nws = new Uint32Array(f.ab.slice(pos, pos + 4 * nch)); pos += 4 * nch;
    let tot = 0; for (let c = 0; c < nch; c++) tot += nws[c];
    const words = new Uint16Array(f.ab.slice(pos, pos + 2 * tot));
    const lut = new Uint16Array(32768), fc = new Uint32Array(nalpha);
    let cum = 0;
    for (let s = 0; s < nalpha; s++) { lut.fill(s, cum, cum + freq[s]); fc[s] = (freq[s] & 0xffff) | (cum << 16); cum += freq[s]; }
    if (cum !== 32768) throw new Error("bad rans16 frequency table");
    let p = 0;
    for (let c = 0; c < nch; c++) {
      let st = states[c];
      const a = c * chunk, b = Math.min(a + chunk, n);
      for (let i = a; i < b; i++) {
        const slot = st & 32767, s = lut[slot], v = fc[s];
        st = (v & 0xffff) * (st >>> 15) + slot - (v >>> 16);
        if (st < 65536) st = ((st * 65536) + words[p++]);
        out[i] = s + smin;
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- fp16 <-> fp32
  const H2F = (() => {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
    }
    return t;
  })();

  const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
  // fp32 -> fp16 bits, round to nearest even (IEEE, same as torch .half() / numpy astype(float16)).
  function f32ToF16(x) {
    _f[0] = x;
    const b = _u[0], sign = (b >>> 16) & 0x8000;
    const e8 = (b >>> 23) & 0xff, m = b & 0x7fffff;
    if (e8 === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    const e = e8 - 112;                                         // fp16 biased exponent (127 - 15)
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
      if (e < -10) return sign;
      const mm = m | 0x800000, sh = 14 - e;
      let r = mm >>> sh;
      const rem = mm & ((1 << sh) - 1), half = 1 << (sh - 1);
      if (rem > half || (rem === half && (r & 1))) r++;
      return sign | r;
    }
    let r = (e << 10) | (m >>> 13);
    const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (r & 1))) r++;        // carry into the exponent is correct (-> inf)
    return sign | r;
  }

  // ---------------------------------------------------------------- record reconstruction
  // State of a decoded record:  bits (Uint16Array fp16 pattern of W), and for qrow also Q (Int16Array) + step (Float32Array).
  // symBytes = bytes whose SHA-256 is the "symbol digest" (Q as int16 LE for qrow, fp16 bits for ll16).
  const secOff = (f, rec, k) => f.base + rec.sec[k][0];

  function decodeRecord(f, rec, R, tim) {
    const n = rec.shape.reduce((a, b) => a * b, 1);
    const p = rec.params || {};
    const t0 = performance.now();
    if (rec.kind === "ll16") {
      const hi = decodeRans16(f, secOff(f, rec, "hi"));
      const parts = new Array(16), k = new Int32Array(16);
      for (const c of p.cls) parts[c] = decodeRans16(f, secOff(f, rec, "lo" + c));
      const bits = new Uint16Array(n);
      for (let i = 0; i < n; i++) { const h = hi[i], c = h >> 4; bits[i] = (h << 8) | parts[c][k[c]++]; }
      const t1 = performance.now();
      if (p.xor) {
        if (!R || !R.bits) throw new Error(rec.name + ": XOR record without reference");
        const rb = R.bits; for (let i = 0; i < n; i++) bits[i] ^= rb[i];
      }
      tim.entropy += t1 - t0; tim.reconstruct += performance.now() - t1;
      return { bits, symBytes: new Uint8Array(bits.buffer) };
    }
    if (rec.kind === "qrow") {
      if (p.ctx) throw new Error("context-coded records are not supported in the browser demo");
      const q = decodeRans16(f, secOff(f, rec, "sym"));
      const t1 = performance.now();
      let step;
      if (p.step === "own") {
        const [o, nb] = rec.sec.step;
        const h = new Uint16Array(f.ab.slice(f.base + o, f.base + o + nb));
        step = new Float32Array(h.length); for (let i = 0; i < h.length; i++) step[i] = H2F[h[i]];
      } else if (p.step === "ref") {
        if (!R || !R.step) throw new Error(rec.name + ": step=ref without reference");
        step = R.step;
      } else throw new Error("step mode " + p.step + " not supported in the browser demo");
      const Q = new Int16Array(n);
      if (p.acc === "add") { const rq = R.Q; for (let i = 0; i < n; i++) Q[i] = rq[i] + q[i]; }
      else if (p.acc === "none") { for (let i = 0; i < n; i++) Q[i] = q[i]; }
      else throw new Error("acc " + p.acc + " not supported in the browser demo");
      // W = fp16( fp32(Q * step_row) [+ fp32(W_ref)] )   -- exactly the PyTorch decoder's arithmetic
      const rows = rec.shape[0], cols = n / rows, bits = new Uint16Array(n);
      const fr = Math.fround;
      if (p.base === "fp16ref") {
        if (!R || !R.bits) throw new Error(rec.name + ": fp16ref record without reference");
        const rb = R.bits;
        for (let r = 0, i = 0; r < rows; r++) {
          const s = step[r];
          for (let j = 0; j < cols; j++, i++) bits[i] = f32ToF16(fr(H2F[rb[i]] + fr(Q[i] * s)));
        }
      } else if (p.base === "none" || !p.base) {
        for (let r = 0, i = 0; r < rows; r++) {
          const s = step[r];
          for (let j = 0; j < cols; j++, i++) bits[i] = f32ToF16(fr(Q[i] * s));
        }
      } else throw new Error("base mode " + p.base);
      tim.entropy += t1 - t0; tim.reconstruct += performance.now() - t1;
      return { Q, step, bits, symBytes: new Uint8Array(Q.buffer) };
    }
    throw new Error("record kind " + rec.kind + " not supported in the browser demo");
  }

  const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
  const sha256 = (u8) => crypto.subtle.digest("SHA-256", u8);
  async function digestOfDigests(list) {            // list of ArrayBuffer(32) in record order
    const cat = new Uint8Array(32 * list.length);
    list.forEach((d, i) => cat.set(new Uint8Array(d), 32 * i));
    return hex(await sha256(cat));
  }

  root.WCCodec = { parseWCDC, decodeRans16, decodeRecord, H2F, f32ToF16, hex, sha256, digestOfDigests };
})(typeof self !== "undefined" ? self : globalThis);
