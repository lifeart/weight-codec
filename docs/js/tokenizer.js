// Minimal byte-level BPE tokenizer for SmolLM2 (reads the official tokenizer.json).
// Pre-tokenizer = Sequence[Digits(individual_digits), ByteLevel(GPT-2 regex, no prefix space)], no normalizer,
// special tokens split first.  Checked against the Python `tokenizers` output in phase2/test_demo.mjs.
"use strict";
(function (root) {
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const enc = new Array(256), dec = new Map();
    bs.forEach((b, i) => { enc[b] = String.fromCodePoint(cs[i]); dec.set(String.fromCodePoint(cs[i]), b); });
    return [enc, dec];
  }

  class BPETokenizer {
    constructor(tj) {
      const m = tj.model;
      if (m.type !== "BPE") throw new Error("expected a BPE tokenizer");
      this.vocab = new Map(Object.entries(m.vocab));
      this.inv = [];
      for (const [t, id] of this.vocab) this.inv[id] = t;
      this.ranks = new Map();
      m.merges.forEach((mg, i) => this.ranks.set(Array.isArray(mg) ? mg.join(" ") : mg, i));
      this.special = new Map();
      for (const a of tj.added_tokens || []) { this.inv[a.id] = a.content; if (a.special) this.special.set(a.content, a.id); }
      this.specialIds = new Set(this.special.values());
      [this.b2u, this.u2b] = bytesToUnicode();
      this.cache = new Map();
      const esc = [...this.special.keys()].map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      this.splitRe = esc.length ? new RegExp("(" + esc.join("|") + ")") : null;
      this.utf8 = new TextEncoder();
    }

    bpe(word) {
      const c = this.cache.get(word);
      if (c) return c;
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const a = parts[best], b = parts[best + 1], out = [];
        for (let i = 0; i < parts.length;) {
          if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { out.push(a + b); i += 2; }
          else { out.push(parts[i]); i++; }
        }
        parts = out;
      }
      const ids = parts.map((p) => {
        const id = this.vocab.get(p);
        if (id === undefined) throw new Error("token not in vocab: " + p);
        return id;
      });
      this.cache.set(word, ids);
      return ids;
    }

    encodeOrdinary(text) {                         // no special-token parsing
      const ids = [];
      // Digits(individual_digits=True): every digit becomes its own piece, the rest is kept as is
      const pieces = text.split(/(\p{N})/u).filter((s) => s.length);
      for (const piece of pieces) {
        for (const m of piece.matchAll(PAT)) {
          let w = "";
          for (const b of this.utf8.encode(m[0])) w += this.b2u[b];
          for (const id of this.bpe(w)) ids.push(id);
        }
      }
      return ids;
    }

    encode(text) {                                 // special tokens like <|im_start|> are recognised
      if (!this.splitRe) return this.encodeOrdinary(text);
      const ids = [];
      for (const seg of text.split(this.splitRe)) {
        if (!seg) continue;
        if (this.special.has(seg)) ids.push(this.special.get(seg));
        else ids.push(...this.encodeOrdinary(seg));
      }
      return ids;
    }

    // SmolLM2-135M-Instruct chat template (single user turn, default system prompt, generation prompt).
    chatIds(userText) {
      return this.encode("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n" +
                         "<|im_start|>user\n" + userText + "<|im_end|>\n<|im_start|>assistant\n");
    }

    tokenBytes(id) {                               // raw bytes of one token (special tokens -> empty)
      if (this.specialIds.has(id)) return [];
      const t = this.inv[id];
      if (t === undefined) return [];
      const out = [];
      for (const ch of t) { const b = this.u2b.get(ch); if (b !== undefined) out.push(b); }
      return out;
    }

    decode(ids) {
      const bytes = [];
      for (const id of ids) bytes.push(...this.tokenBytes(id));
      return new TextDecoder().decode(new Uint8Array(bytes));
    }
  }

  root.WCTokenizer = { BPETokenizer };
})(typeof self !== "undefined" ? self : globalThis);
