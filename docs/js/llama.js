// Tiny Llama forward pass in plain JavaScript (single thread, CPU) for SmolLM2-135M.
// Runs directly on the fp16 weights the codec reconstructed (Uint16Array bit patterns, converted
// to fp32 on the fly through a 64 K lookup table); all activations / accumulations in fp32/fp64.
// Same math as transformers' LlamaForCausalLM (RMSNorm, RoPE rotate_half, GQA, SiLU MLP, tied head).
// Summation order differs from PyTorch, so logits agree to ~1e-4, not bit-exactly.
"use strict";
(function (root) {
  const H2F = root.WCCodec.H2F;

  // y[O] = W[O, I] (fp16 bits) @ x[I]
  function matvec(w, x, y, O, I) {
    for (let r = 0, o = 0; r < O; r++, o += I) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let j = 0; j < I; j += 4) {
        s0 += H2F[w[o + j]] * x[j]; s1 += H2F[w[o + j + 1]] * x[j + 1];
        s2 += H2F[w[o + j + 2]] * x[j + 2]; s3 += H2F[w[o + j + 3]] * x[j + 3];
      }
      y[r] = s0 + s1 + s2 + s3;
    }
  }

  class Llama {
    // W: {name -> Uint16Array}  names as in the WCDC files: "model.embed_tokens.weight", "layers.0.self_attn.q_proj", ...
    constructor(W, cfg) {
      this.cfg = cfg = Object.assign({ hidden: 576, inter: 1536, layers: 30, heads: 9, kvHeads: 3, vocab: 49152,
                                       eps: 1e-5, theta: 100000, maxSeq: 512 }, cfg || {});
      const need = (k) => { if (!W[k]) throw new Error("missing weight " + k); return W[k]; };
      this.emb = need("model.embed_tokens.weight");
      this.normF = Float32Array.from(need("model.norm.weight"), (h) => H2F[h]);
      this.L = [];
      for (let i = 0; i < cfg.layers; i++) {
        const p = `layers.${i}.`, m = `model.layers.${i}.`;
        this.L.push({
          q: need(p + "self_attn.q_proj"), k: need(p + "self_attn.k_proj"), v: need(p + "self_attn.v_proj"),
          o: need(p + "self_attn.o_proj"), g: need(p + "mlp.gate_proj"), u: need(p + "mlp.up_proj"),
          d: need(p + "mlp.down_proj"),
          n1: Float32Array.from(need(m + "input_layernorm.weight"), (h) => H2F[h]),
          n2: Float32Array.from(need(m + "post_attention_layernorm.weight"), (h) => H2F[h]),
        });
      }
      const D = cfg.hidden, hd = D / cfg.heads, kvd = hd * cfg.kvHeads;
      this.hd = hd; this.kvd = kvd;
      // inverse frequencies exactly as transformers computes them (fp32)
      const fr = Math.fround;
      this.invf = new Float32Array(hd / 2);
      for (let i = 0; i < hd / 2; i++) this.invf[i] = fr(1 / fr(Math.pow(cfg.theta, fr((2 * i) / hd))));
      this.x = new Float32Array(D); this.h = new Float32Array(D);
      this.qb = new Float32Array(D); this.kb = new Float32Array(kvd); this.vb = new Float32Array(kvd);
      this.att = new Float32Array(D); this.ob = new Float32Array(D);
      this.gb = new Float32Array(cfg.inter); this.ub = new Float32Array(cfg.inter);
      this.logits = new Float32Array(cfg.vocab);
      this.scores = new Float64Array(cfg.maxSeq);
      this.reset();
    }

    reset() {
      const c = this.cfg;
      this.K = []; this.V = [];
      for (let i = 0; i < c.layers; i++) { this.K.push(new Float32Array(c.maxSeq * this.kvd)); this.V.push(new Float32Array(c.maxSeq * this.kvd)); }
      this.pos = 0;
    }

    // snapshot / restore of the KV cache prefix (prefix caching for the shared system prompt)
    snapshot() { return { pos: this.pos, K: this.K.map((k) => k.slice(0, this.pos * this.kvd)), V: this.V.map((v) => v.slice(0, this.pos * this.kvd)) }; }
    restore(s) { for (let i = 0; i < this.cfg.layers; i++) { this.K[i].set(s.K[i]); this.V[i].set(s.V[i]); } this.pos = s.pos; }

    rmsnorm(x, w, out) {
      let ss = 0; for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
      const r = 1 / Math.sqrt(Math.fround(ss / x.length) + this.cfg.eps);
      for (let i = 0; i < x.length; i++) out[i] = w[i] * Math.fround(x[i] * r);
    }

    rope(v, nh, pos) {
      const hd = this.hd, half = hd / 2, fr = Math.fround;
      for (let i = 0; i < half; i++) {
        const a = fr(pos * this.invf[i]), c = fr(Math.cos(a)), s = fr(Math.sin(a));
        for (let h = 0; h < nh; h++) {
          const o = h * hd, x1 = v[o + i], x2 = v[o + i + half];
          v[o + i] = x1 * c - x2 * s; v[o + i + half] = x2 * c + x1 * s;
        }
      }
    }

    // one token -> (optionally) logits over the vocabulary
    step(token, wantLogits = true) {
      const c = this.cfg, D = c.hidden, hd = this.hd, kvd = this.kvd, pos = this.pos;
      if (pos >= c.maxSeq) throw new Error("context full");
      const x = this.x, h = this.h, e = this.emb, off = token * D;
      for (let i = 0; i < D; i++) x[i] = H2F[e[off + i]];
      const grp = c.heads / c.kvHeads, scale = 1 / Math.sqrt(hd), sc = this.scores;
      for (let l = 0; l < c.layers; l++) {
        const L = this.L[l];
        this.rmsnorm(x, L.n1, h);
        matvec(L.q, h, this.qb, D, D); matvec(L.k, h, this.kb, kvd, D); matvec(L.v, h, this.vb, kvd, D);
        this.rope(this.qb, c.heads, pos); this.rope(this.kb, c.kvHeads, pos);
        const K = this.K[l], V = this.V[l];
        K.set(this.kb, pos * kvd); V.set(this.vb, pos * kvd);
        const att = this.att; att.fill(0);
        for (let hh = 0; hh < c.heads; hh++) {
          const qo = hh * hd, ko = Math.floor(hh / grp) * hd;
          let mx = -Infinity;
          for (let t = 0; t <= pos; t++) {
            let s = 0; const kb = t * kvd + ko;
            for (let j = 0; j < hd; j++) s += this.qb[qo + j] * K[kb + j];
            s *= scale; sc[t] = s; if (s > mx) mx = s;
          }
          let den = 0;
          for (let t = 0; t <= pos; t++) { sc[t] = Math.exp(sc[t] - mx); den += sc[t]; }
          for (let t = 0; t <= pos; t++) {
            const p = sc[t] / den, vb = t * kvd + ko;
            for (let j = 0; j < hd; j++) att[qo + j] += p * V[vb + j];
          }
        }
        matvec(L.o, att, this.ob, D, D);
        for (let i = 0; i < D; i++) x[i] += this.ob[i];
        this.rmsnorm(x, L.n2, h);
        matvec(L.g, h, this.gb, c.inter, D); matvec(L.u, h, this.ub, c.inter, D);
        const g = this.gb, u = this.ub;
        for (let i = 0; i < c.inter; i++) { const a = g[i]; g[i] = (a / (1 + Math.exp(-a))) * u[i]; }
        matvec(L.d, g, this.ob, D, c.inter);
        for (let i = 0; i < D; i++) x[i] += this.ob[i];
      }
      this.pos++;
      if (!wantLogits) return null;
      this.rmsnorm(x, this.normF, h);
      matvec(e, h, this.logits, c.vocab, D);                         // tied embedding head
      return this.logits;
    }
  }

  function argmax(a) { let b = 0, m = a[0]; for (let i = 1; i < a.length; i++) if (a[i] > m) { m = a[i]; b = i; } return b; }

  root.WCLlama = { Llama, argmax, matvec };
})(typeof self !== "undefined" ? self : globalThis);
