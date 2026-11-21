# weight-codec — REPORT

*"Compress weights like JPEG (intra), families / layers / checkpoints like video (inter)."*
Falsification-first study on real ≤1 GB models. Phase 1: 2026-11-20; phase 2 (real bitstreams, §14): 2026-11-21.
Hardware: an Apple Silicon laptop, MPS (plus x86-64 CPU Hugging Face Jobs in phase 2). Author: Aleksandr Kanunnikov (lifeart).

The experiments were run and written up by a Claude Code agent under the author's direction; "I" / "my" in this
report refers to that agent. Process deviations, reruns and bugs are kept in the text on purpose (§12, §14.9).

Legend: **FACT** = measured here, number in `results/`. **HYPOTHESIS / INTERPRETATION** = my reading, not measured.

## 0. TL;DR

> **Phase 2 (2026-11-21): real ANS bitstreams, decode benchmark, Hugging Face runs → §14.** Real file bytes match the phase-1
> entropy estimates to ±0.02 %; all phase-1 winners hold on real bytes.


| hypothesis | verdict on these models | headline number (FACT) |
|---|---|---|
| A. JPEG **transform** (block-DCT, ± reorder / Hadamard) | **falsified** | DCT changes rate by **+0.007 bit/weight** (best reorder −0.036 vs null −0.010); positive control −2.17 |
| A. JPEG **quantization + entropy** (ECSQ, A2) | **confirmed** | int4 indices: 4.16 → 3.76 bpw for free; GPTQ on an unclipped entropy-coded grid: GPTQ-int4 quality at **3.47 bpw** |
| A. **sensitivity-weighted steps** (A3, Fisher) | **confirmed** | GPTQ-int4 quality at **3.05 bpw (−27 %)**; 3.65 bpw → PPL 13.42 vs fp16 13.07; holds out-of-domain (C4) with a smaller margin |
| B. layers as frames (+ motion compensation) | **falsified** | adjacent-layer residual 0.99998 raw, 0.986 after neuron alignment (null 0.996) → 0.01 bit/weight |
| C. family as GOP (SFT/RL fine-tune) | **confirmed, largest win** | instruct given an int4-class base: **1.05–1.11 bpw at the quality of a 3.58-bpw independent coder**; given an fp16 base: **1.69 bpw → chat-KL 0.011** (independent GPTQ-int4: 3.76 bpw → 0.148) |
| C. "the delta is low-rank (LoRA-like)" | **falsified** | Δ is small (7.7 %) but full-rank; rank-128 Δ is 13× worse in KL than an entropy-coded Δ at fewer bits |
| C. continued-pretraining relatives (Coder) | **mostly falsified** | ‖Δ‖/‖W‖ = 0.88 → P-frame saves only 8.6 % |
| D. training checkpoints as video | **confirmed late in training** | 1000-step P-frame = **9.5–27 %** of an I-frame; D2 GOP of 8 late checkpoints: **−79 % bits** (IPPP, no drift: PPL 23.16 vs 23.08), mid-training −56 % |
| E. one progressive stream (SVC) | **confirmed with nested quantizers** | every layer reproduces the single-shot quantizer exactly, but ratio-2 refinement costs +0.5…1.6 bpw; with a **nested ×3 lattice + context coding the overhead is +0.004…0.016 bpw** (≈ free) |

Bottom line (INTERPRETATION): the *video* half of the analogy works where real temporal redundancy
exists — between a base model and its fine-tunes and between nearby training checkpoints — and the
"motion" is trivial (identity alignment). The *JPEG* half works only through its quantizer + entropy coder +
perceptual (sensitivity) weighting; the transform part, which is the heart of JPEG, has nothing to compact.
Inside a single model there is no spatial or inter-layer redundancy to exploit.

## 1. Measurement spine

**Protocol (FACT).** Distortion = WikiText-2 *test* PPL, non-overlapping 2048-token windows
(146 windows, 299 k tokens, GPTQ-paper protocol) + exact KL(fp16 ‖ q) and top-1 agreement on
the first 16 windows (32 k tokens). Rate = bits/weight over the *quantized set* = all 168
`nn.Linear` of the 24 blocks (357.8 M of 494.0 M params); the tied embedding (136.2 M) and
norms/biases stay fp16 in **every** config. Entropy-coded rate = per-tensor empirical order-0
entropy of the integer symbols + frequency table (32 bit/used symbol) + all side info at stored
precision. **This is a proxy** for a static-model rANS coder, not an implemented coder.
fp16 on MPS == fp32 to 1e-5 in PPL (sanity: 11.5922 vs 11.5921 on 8 windows).

**Baselines, Qwen2.5-0.5B (FACT, `results/spine_qwen05b.json`)**

| config | fixed-rate bpw | entropy-coded bpw (A2) | PPL | KL | top-1 |
|---|---|---|---|---|---|
| fp16 | 16 | — | **13.071** | 0 | 1.000 |
| RTN int8 per-channel | 8.020 | 7.270 | 13.090 | 0.0017 | 0.977 |
| RTN int4 g128 | 4.156 | 3.756 | 15.542 | 0.188 | 0.779 |
| RTN int4 per-channel | 4.017 | 3.207 | 23.795 | 0.617 | 0.614 |
| RTN int3 g128 | 3.148 | 2.710 | 51.63 | 1.349 | 0.466 |
| RTN int2 g128 | 2.141 | 1.824 | 4.5e5 (collapse) | 10.1 | 0.012 |
| GPTQ int4 g128 | 4.156 | 3.757 | **14.033** | 0.075 | 0.860 |
| GPTQ int3 g128 | 3.148 | 2.712 | 19.574 | 0.390 | 0.702 |
| GPTQ int2 g128 | 2.141 | 1.831 | 1.3e5 (collapse) | 9.15 | 0.019 |

bpw accounting for g128: b + (16 + b)/128 (fp16 scale + b-bit zero per group of 128).
Qwen2.5-0.5B is very quantization-sensitive here (FACT: int3 RTN → PPL 51.6, any 2-bit point collapses); that small Qwen models are hard to quantize is also the common experience (INTERPRETATION).

## 2. Probe verdicts (all probes run BEFORE any codec; criteria pre-registered in each script docstring)

| probe | question | key number(s) | pre-registered line | verdict | codec built? |
|---|---|---|---|---|---|
| **A1** | does 8×8 DCT (± reorder / Hadamard) compact weight energy? | median Δbits DCT−raw = **+0.007** (raw), −0.036 (best: TSP) vs null −0.010; positive control −2.17 | ≤ −0.10 on majority & ≥0.05 beyond null | **FAIL** | no (transform dead) |
| A3-probe | is sensitivity dispersed enough for step allocation? | predicted ½log2(AM/GM): 0.58 (per matrix) … 0.91 bpw (separable) | > 0.10 bpw at ≤ 0.05 bpw side info | **PASS** | yes |
| **B1** (+B3-lite) | are adjacent layers predictable (after neuron alignment)? | raw residual **0.99998**; aligned **0.986** vs null 0.996 | aligned residual ≤ 0.70 and ≥ 0.10 below null | **FAIL** | no |
| **C1** small | is Δ(instruct−base) small? | ‖Δ‖/‖W‖ = **0.077**; int4-grid index-diff entropy 0.87 vs 3.54 bit | < 0.30 | **PASS** | yes (C2/C4) |
| C1 low-rank | is Δ low-rank (LoRA-like)? | top-5 %-rank energy 0.152 vs null 0.112; r90 = 67 % of rank | ≥ 0.5 and ≥ 2× null on > 50 % matrices | **FAIL** | low-rank coder built only as a control |
| C1 coder | is Coder a small Δ from base? | ‖Δ‖/‖W‖ = 0.88 (Coder), 1.05 (Coder-Instruct vs Coder) | < 0.30 | **FAIL** | rate-only (C4) |
| **D1** | are late checkpoint deltas cheap P-frames? | P/I bits (int4 grid) at 64k→65k 0.27, 142k→143k **0.096**; mean 0.18 | ≤ 0.5 | **PASS** | yes (D2) |


## 3. A — "JPEG intra"

### A1 probe: block-DCT energy compaction — **FAIL** (FACT, `results/probe_a1.json`, `figures/probe_a1_dct.png`)

12 matrices (blocks 2/12/22 × q/o/gate/down), row-RMS normalised (the baseline coder already has
a per-row step). Decisive metric: entropy of uniformly quantized symbols at the *same step*
(orthonormal ⇒ same MSE), DCT with 64 per-position histograms (JPEG-style) vs no transform.

| variant | median Δbits DCT−raw (real) | matrices ≤ −0.10 | Gaussian null | positive control (smooth AR(1), shuffled) |
|---|---|---|---|---|
| raw order | **+0.007** | 0/12 | −0.000 | −0.007 (raw order destroyed by shuffle) |
| sort by norm | +0.005 | 0/12 | 0.000 | −0.034 |
| spectral (Fiedler, kNN graph) | +0.005 | 0/12 | +0.000 | **−2.155** |
| greedy TSP on \|corr\| (+sign flips) | −0.036 | 1/12 (L22.q_proj −0.16) | −0.010 | **−2.128** |
| block-Hadamard rotation | +0.000 | 0/12 | +0.000 | 0.000 |

Top-8-of-64 energy: raw 0.50–0.54 vs DCT 0.49–0.50 (DCT *spreads* heavy-tailed outliers).
Transform coding gain (AM/GM of coefficient variances): ≤0.01 dB raw/norm/spectral/Hadamard, 0.3–0.4 dB after TSP
(the null gets 0.06–0.07 dB from TSP alone). The same pipeline gives 17.3 dB / −2.17 bit on the positive control
and recovers −2.13 bit after the control's rows/cols are shuffled, so the probe *can* see compaction.

**Verdict:** weight matrices have no 2-D spatial smoothness for a DCT to exploit, and none can be
manufactured by reordering (best case ≈ 0.03 bit/weight over null). Hadamard rotation is, by
construction, the *opposite* of compaction (it whitens / spreads) — it helps fixed-rate grids by
killing outliers, not by compaction. **The transform half of the JPEG analogy is dead; no DCT codec was built.**

### A2: entropy coding of the int indices — **confirmed** (FACT)

Free lunch exactly as expected: int4 g128 indices have 3.76 bit entropy (RTN and GPTQ alike;
fixed-rate 4.156 incl. side info) ⇒ **−0.40 bpw at identical PPL**; int3: 3.148 → 2.71 (−0.44);
int8 per-channel: 8.02 → 7.27. Per-channel int4: 4.02 → 3.21 (wider dynamic range ⇒ more peaked histogram).

### A2+: ECSQ — uniform step without clipping + entropy coding (the quantizer JPEG actually uses)

Replacing min-max grids by a uniform step `s = c·rms(row)` with *unbounded* levels and entropy coding
(no groups, one fp16 step per row) — and running GPTQ on that grid — is the single largest
intra-model win in this study (FACT, `figures/rd_spine_ppl_zoom.png`):

| bits needed to reach … (interpolated on each RD curve) | PPL 14.03 (=GPTQ-int4) | PPL 15.54 (=RTN-int4) | PPL 19.57 (=GPTQ-int3) |
|---|---|---|---|
| RTN g128 fixed-rate | — | 4.16 | 3.96 |
| GPTQ g128 fixed-rate | 4.16 | 3.85 | 3.15 |
| GPTQ g128 + entropy (A2) | 3.76 | 3.44 | 2.71 |
| ECSQ row-step (RTN) | (≈4.06) | 3.53 | 3.03 |
| **GPTQ-ECSQ row-step** | **3.47** | **2.90** | **2.45** |

⇒ at GPTQ-int4 quality, GPTQ-ECSQ needs **3.47 bpw instead of 4.16 (−17 %)**; at GPTQ-int3 quality
**2.45 instead of 3.15 (−22 %)**. Measured points: GPTQ-ECSQ c=0.35 → 3.578 bpw, PPL 13.84, KL 0.060
(beats GPTQ-int4 g128 at 4.156 bpw: 14.03 / 0.075 on *both* axes).
Decode cost (qualitative): variable-length symbols ⇒ an entropy-decode pass (rANS, GB/s on CPU/GPU) to a
fixed-width int8 buffer at load time, or a fused ANS-decode-in-GEMM kernel (DFloat11-style) for
on-the-fly decode. Not a drop-in for existing int4 kernels.

### A3: sensitivity-weighted steps (Fisher) — probe **PASS** (prediction), codec: see §3.A3-codec

Probe (FACT, `results/probe_a3_qwen05b.json`): empirical Fisher diagonal from 64×512 WikiText-2
train tokens. High-rate theory for ECSQ with per-weight distortion weights F: optimal step
s ∝ F^-1/2; predicted saving vs the row-RMS step = ½·log2(AM/GM) of x = F·s0².

| allocation granularity | predicted saving | extra side info | net |
|---|---|---|---|
| per matrix (cross-layer/type only) | 0.58 bpw | ~0 | 0.58 |
| per row | 0.81 | 0 | 0.81 |
| separable row × column | 0.92 | 0.011 | 0.91 |
| per row × 128-col block | 0.94 | 0.125 | 0.81 |

Sensitivity per unit of (row-RMS)² differs ~50× between matrix types (v_proj 6.6e-9, k_proj 2.9e-9 vs
gate 1.3e-10, q 1.8e-10). **Caveat (INTERPRETATION):** these are high-rate predictions from a noisy
64-sequence Fisher; estimation noise inflates AM/GM, and GPTQ already captures part of the within-layer
effect. The codec result below is what counts.

### A3 codec: Fisher-allocated ECSQ steps — **confirmed** (FACT, `results/a3_codec_qwen05b.json`, `figures/rd_spine_ppl_zoom.png`)

Steps `log s_ij = log λ + (1−γ)·log rms_i + γ·(u_ij + shift)` with the separable Fisher model
u_ij = −½(log F_row,i + log F_col,j − log F̄); γ=0 is exactly the row-RMS ECSQ of the spine, γ=1 the pure
high-rate-optimal Fisher allocation (global across all 168 matrices); side info = fp16 row + column factors
(0.011 bpw). RTN rounding (no calibration) and GPTQ rounding on that grid.

| config | γ | λ | entropy bpw | PPL | KL |
|---|---|---|---|---|---|
| fisher_g1.0_lam0.35 | 1.0 | 0.35 | 3.792 | 13.569 | 0.0408 |
| fisher_g1.0_lam0.5 | 1.0 | 0.5 | 3.286 | 14.241 | 0.0869 |
| fisher_g1.0_lam0.7 | 1.0 | 0.7 | 2.815 | 15.632 | 0.1818 |
| fisher_g0.5_lam0.35 | 0.5 | 0.35 | 3.644 | 13.686 | 0.0469 |
| fisher_g0.5_lam0.5 | 0.5 | 0.5 | 3.137 | 14.353 | 0.0999 |
| fisher_g0.5_lam0.7 | 0.5 | 0.7 | 2.666 | 16.006 | 0.2065 |
| gptq_fisher_g1.0_lam0.5 | 1.0 | 0.5 | 3.291 | 13.845 | 0.0560 |
| gptq_fisher_g1.0_lam0.7 | 1.0 | 0.7 | 2.825 | 14.764 | 0.1151 |
| gptq_fisher_g0.5_lam0.35 | 0.5 | 0.35 | 3.647 | 13.423 | 0.0271 |
| gptq_fisher_g0.5_lam0.5 | 0.5 | 0.5 | 3.142 | 13.855 | 0.0559 |
| gptq_fisher_g0.5_lam0.7 | 0.5 | 0.7 | 2.676 | 14.804 | 0.1203 |

| bits needed to reach … | PPL 14.03 (=GPTQ-int4 g128) | PPL 14.5 | KL 0.05 |
|---|---|---|---|
| GPTQ int4 g128, fixed-rate | 4.16 | 4.06 | — |
| GPTQ g128 + entropy (A2) | 3.76 | 3.65 | — |
| GPTQ-ECSQ row-step | 3.47 | 3.21 | — |
| Fisher-step ECSQ, RTN rounding (γ=0.5) | 3.38 | 3.09 | 3.60 |
| Fisher-step GPTQ-ECSQ (γ=1) | 3.19 | 2.96 | — |
| **Fisher-step GPTQ-ECSQ (γ=0.5)** | **3.05** | **2.82** | **3.22** |

* Sensitivity-weighted steps alone (no GPTQ) already beat GPTQ-ECSQ; combined with GPTQ the codec reaches
  GPTQ-int4 quality at **3.05 bpw (−27 % vs 4.16 fixed-rate GPTQ-int4, −0.42 bpw vs GPTQ-ECSQ)**.
  Best measured point: 3.647 bpw → PPL **13.42** / KL 0.027 (fp16: 13.07).
* Measured gain is smaller than the 0.9 bpw high-rate prediction (≈0.4 bpw over GPTQ-ECSQ, ≈0.6 over row-ECSQ
  at PPL 15.5), and γ=0.5 beats γ=1: the 64-sequence Fisher is noisy and full allocation over-steers
  (INTERPRETATION, consistent with both observations).
* Caveat: Fisher and GPTQ Hessians are calibrated on WikiText-2 train; evaluation is WikiText-2 test.
  Out-of-domain check on C4: see §8.

## 4. B — layers as frames

### B1 probe (+ B3-lite motion compensation) — **FAIL** (FACT, `results/probe_b1.json`, `figures/probe_b1_layers.png`)

Same-named matrices of adjacent blocks (23 pairs × 7 types):

* raw: |cos| ≤ 0.044 for **every** pair; optimal-α residual energy 1−cos² = **0.99998** (mid blocks 6–17).
* motion-compensated with the network's exact symmetries — Hungarian permutation of the 4864 MLP
  neurons on [gate; up; downᵀ] and per-KV-group orthogonal Procrustes for V/O:
  residual **0.986** (mid blocks), vs **0.996** when the same alignment is fitted to a Gaussian matrix
  with identical norms (what matching manufactures out of nothing).
* ⇒ predicted saving 0.010 bit/weight (0.007 over null); pre-registered pass line was residual ≤ 0.70.
* Most "similar" spot: v_proj of the last blocks after Procrustes (residual 0.906 at 22→23) — still
  only 0.07 bit/weight.

**Verdict:** adjacent layers are, to first order, independent random-looking matrices even after
permutation/rotation alignment. Layer-to-layer "motion compensation" does not exist at the weight
level in this model. B2/B3 codecs not built.

## 5. C — the family as a GOP

### C1 probe — **PASS on "small", FAIL on "low-rank"** (FACT, `results/probe_c1_*.json`, `figures/probe_c1_*.png`)

| pair | ‖Δ‖/‖W‖ (weighted median, block linears) | embedding ‖Δ‖/‖W‖ | top-5%-rank energy of Δ (null) | r90 / full rank | H(q_ft) → H(q_ft − q_base) on int4-class grid | indices changed |
|---|---|---|---|---|---|---|
| base → instruct | **0.077** | 0.112 | 0.152 (0.112) | 0.67 | 3.54 → **0.87** bit | 18 % |
| base → coder | 0.88 | 1.02 | 0.199 (0.112) | — | 3.53 → 3.40 bit | 83 % |
| coder → coder-instruct | 1.05 | 1.08 | 0.191 (0.113) | — | 3.65 → 3.64 bit | 86 % |

* **Small: yes** for base→instruct (pre-registered threshold 0.30). **Low-rank: no** — Δ of a full
  SFT+RL fine-tune is a full-rank, small-amplitude perturbation (0.6 % of matrices pass the low-rank
  test; rank for 90 % energy ≈ 67 % of full rank). LoRA-style structure is *not* present when the
  fine-tune itself was not LoRA.
* **Surprise (FACT):** Qwen2.5-Coder-0.5B is *not* a small delta from Qwen2.5-0.5B (cos 0.60,
  ‖Δ‖/‖W‖ 0.88 — continued pre-training on 5.5 T tokens moves weights as far as training from scratch
  moves them in the late phase), and Coder-Instruct is not a small delta from Coder (cos 0.48,
  ‖Δ‖/‖W‖ 1.05). Pairwise cosines of block linears (`results/probe_c_pairwise.json`): base–instruct
  0.997, base–coder 0.598, coder–coder-instruct 0.485, base–coder-instruct 0.416 — coder-instruct's closest
  relative is coder, but it is far from everything. (Qwen2.5-Math has no 0.5B variant → not tested.)

### C3: alignment base ↔ fine-tune (FACT, `results/probe_c3_*.json`)

* base→instruct: Hungarian matching of MLP neurons returns the **identity for 100 %** of neurons in all
  24 blocks; V/O Procrustes rotation distance from I = 0.003. No permutation, no rotation.
* base→coder: identity for **99.96 %** of neurons (min 99.7 % per block) even though weights moved
  ‖Δ‖/‖W‖ = 0.88; V/O rotation distance 0.28 (largest in block 0: 0.83/0.70). Continued pre-training
  keeps the neuron basis; alignment ≈ identity. Motion compensation would buy nothing here either.

### C4 (rate side): closed-loop P-frames inside the family (FACT, `results/c4_gop_entropy.json`)

Uniform grid s = c·rms_row(target); every variant has the **same** quantization error (relative weight
MSE 0.0102 at c=0.35 for all I-, P- and chain variants — measured), so rates are compared at equal
distortion. Per-row least-squares prediction gain α (fp16 side info) included.

| target (int4-class, c=0.35) | I-frame | P from best ref (α=1) | P with per-row α |
|---|---|---|---|
| instruct ← base | 3.575 | 0.911 | **0.875** (−76 %) |
| coder ← base | 3.575 | 3.452 | 3.267 (−8.6 %) |
| coder-instruct ← coder (chain, closed loop) | 3.575 | 3.563 | 3.380 (−5.5 %) |
| coder-instruct ← base (star) | 3.575 | 3.678 | 3.445 (−3.6 %) |

* GOP for all 4 models: 4 × 3.575 = 14.30 → 3.575 + 0.875 + 3.267 + 3.380 = **11.10 bpw-sum (−22 %)**;
  for {base, instruct} alone: 7.15 → **4.45 (−38 %)**.
* **Open-loop drift (FACT):** predicting coder-instruct from the *original* coder while the decoder only
  has reconstructions raises relative MSE 0.0102 → 0.0141 (+38 %) at equal rate. Closed loop is mandatory.
* Random access: star topology (every P-frame references the base) costs one extra decode; the chain
  base→coder→coder-instruct saves 0.065 bpw over star for coder-instruct but needs two decodes.

### C2 codec: instruct coded given base — **large win, confirmed** (FACT, `results/c2_qwen05b_instruct.json`, `figures/rd_c2_instruct_*.png`)

Distortion is measured against the fp16 **instruct** model: KL / top-1 / NLL on the instruct model's own
greedy answers to 48 fixed prompts (5 203 answer tokens, "chat") and on WikiText-2 (KL on 16 windows,
PPL on all 146). **WikiText PPL is a misleading metric for an instruct model** (FACT): reconstructions that
drift back toward the base get a *lower* PPL than the instruct model itself (instruct 14.24; low-rank-Δ
reconstructions 13.64–13.67; base-only linears 19.30 because embeddings/norms stay instruct). KL to the
instruct model is the honest distortion; PPL is reported but not used for verdicts.
Rate = bits/weight of the instruct stream over the 168 block linears (base cost reported separately).

Anchors: instruct fp16 → chat-KL 0; "instruct := base" (0 bits for block linears) → chat-KL **0.277**,
wiki-KL 0.430. So the 7.7 %-norm delta carries a lot of behaviour.

| config | instruct bpw | chat KL | chat top-1 | chat NLL | wiki KL | wiki top-1 | wiki PPL |
|---|---|---|---|---|---|---|---|
| fp16 | 16.000 | 0.0000 | 1.000 | 0.519 | 0.0000 | 1.000 | 14.24 |
| zero_bits_base | 0.000 | 0.2767 | 0.828 | 0.857 | 0.4296 | 0.705 | 19.30 |
| indep_gptq_ecsq_c0.35 | 3.579 | 0.1173 | 0.865 | 0.686 | 0.0645 | 0.872 | 15.09 |
| indep_gptq_ecsq_c0.5 | 3.075 | 0.2562 | 0.802 | 0.842 | 0.1254 | 0.826 | 16.06 |
| indep_gptq_ecsq_c0.7 | 2.610 | 0.5121 | 0.729 | 1.146 | 0.2753 | 0.747 | 18.71 |
| indep_gptq_ecsq_c1.0 | 2.136 | 1.3625 | 0.540 | 2.140 | 0.6782 | 0.621 | 28.49 |
| indep_gptq_int3_g128 | 2.713 | 0.7968 | 0.642 | 1.499 | 0.4191 | 0.694 | 21.77 |
| indep_gptq_int4_g128 | 3.757 | 0.1482 | 0.849 | 0.711 | 0.0799 | 0.858 | 15.31 |
| indep_rtn_int3_g128 | 2.711 | 1.1997 | 0.547 | 1.948 | 1.5133 | 0.428 | 66.30 |
| indep_rtn_int4_g128 | 3.757 | 0.1731 | 0.833 | 0.728 | 0.2033 | 0.769 | 17.05 |
| delta_ecsq_c0.03 | 3.529 | 0.0010 | 0.990 | 0.522 | 0.0018 | 0.979 | 14.23 |
| delta_ecsq_c0.06 | 2.553 | 0.0039 | 0.978 | 0.520 | 0.0065 | 0.961 | 14.17 |
| delta_ecsq_c0.12 | 1.637 | 0.0164 | 0.949 | 0.540 | 0.0377 | 0.901 | 14.35 |
| delta_ecsq_c0.25 | 0.692 | 0.0825 | 0.887 | 0.645 | 0.1898 | 0.791 | 15.84 |
| delta_ecsq_c0.7 | 0.032 | 0.2678 | 0.832 | 0.846 | 0.4128 | 0.707 | 18.98 |
| delta_gptq_c0.06 | 2.568 | 0.0031 | 0.978 | 0.520 | 0.0018 | 0.978 | 14.25 |
| delta_gptq_c0.12 | 1.686 | 0.0105 | 0.960 | 0.529 | 0.0058 | 0.959 | 14.27 |
| delta_gptq_c0.25 | 0.843 | 0.0375 | 0.929 | 0.557 | 0.0205 | 0.922 | 14.24 |
| delta_gptq_c0.5 | 0.197 | 0.0971 | 0.889 | 0.614 | 0.0582 | 0.874 | 13.83 |
| delta_lowrank_r128 | 3.147 | 0.0522 | 0.918 | 0.568 | 0.0608 | 0.870 | 13.67 |
| delta_lowrank_r32 | 0.787 | 0.0874 | 0.892 | 0.612 | 0.0880 | 0.851 | 13.64 |
| pframe_gptq_c0.35 | 1.046 | 0.1280 | 0.855 | 0.674 | 0.0657 | 0.872 | 15.07 |
| pframe_gptq_c0.5 | 0.921 | 0.2558 | 0.808 | 0.824 | 0.1319 | 0.822 | 16.05 |
| pframe_gptq_c0.7 | 0.841 | 0.5433 | 0.712 | 1.202 | 0.2864 | 0.746 | 19.01 |
| pframe_rtn_c0.35 | 0.883 | 0.1477 | 0.846 | 0.702 | 0.1757 | 0.786 | 16.70 |
| pframe_rtn_c0.7 | 0.545 | 0.7243 | 0.657 | 1.418 | 1.0219 | 0.517 | 38.74 |
| pframe_rtn_c1.4 | 0.325 | 7.8785 | 0.062 | 8.988 | 6.5492 | 0.069 | 16584.89 |

**Scenario 1 — base available losslessly (hub stores fp16 base; instruct as Δ):**
* GPTQ on Δ (instruct Hessian) is the best Δ coder: **0.84 bpw → chat-KL 0.038, wiki-KL 0.021**;
  **1.69 bpw → chat-KL 0.011, wiki-KL 0.006**; 2.57 bpw → chat-KL 0.003. ECSQ(Δ) is close behind
  (3.53 bpw → chat-KL 0.001 ≈ int8-class fidelity).
* vs storing the instruct model on its own with GPTQ-int4: 3.76 bpw → chat-KL 0.148. The Δ coder reaches the
  same chat-KL at **≈0.36 bpw (−90 %)** and gives **14× lower chat-KL at 45 % of the bits** (1.69 bpw).
* {base fp16, instruct}: 16 + 1.69 = 17.7 bpw-sum instead of 32 (both fp16) for an instruct copy at chat-KL 0.011.
* **LoRA-style low-rank Δ is falsified as a representation** of a full fine-tune: rank 32 (0.79 bpw) →
  chat-KL 0.087 (GPTQ-Δ at 0.84 bpw: 0.038); rank 128 (3.15 bpw) → 0.052 (ECSQ-Δ at 2.55 bpw: 0.004, 13× better).

**Scenario 2 — quantized family (base and instruct both stored at int4-class quality, closed loop):**
* Instruct quantized on the *base's* grid and sent as index differences (RTN, c=0.35): **0.883 bpw**,
  chat-KL 0.148 / wiki-KL 0.176 — the same chat-KL as independent GPTQ-int4 (3.757 bpw) and better than
  independent RTN-int4 (0.173 / 0.203 at 3.757 bpw).
* GPTQ on the residual against the decoded RTN-ECSQ base (c=0.35): **1.046 bpw → wiki-KL 0.0657, chat-KL 0.128,
  PPL 15.07**, vs the best independent coder, GPTQ-ECSQ c=0.35: **3.579 bpw → 0.0645 / 0.117 / 15.09**.
  Same quality for **29 % of the bits**. Same pattern at c=0.5 (0.921 vs 3.075 bpw, chat-KL 0.256 vs 0.256).
* {base, instruct} at int4-class: 3.575 + 1.046 = **4.62 bpw-sum vs 7.16 (−35 %)**.
* With the *better* base (GPTQ-ECSQ, base PPL 13.84) the instruct residual costs **1.114 bpw** and the
  instruct reconstruction is **identical** (wiki-KL 0.0657, chat-KL 0.1279) to the RTN-base variant.
  Explanation (verified algebraically, consistent with the numbers): both bases lie on the same lattice
  s·ℤ, and GPTQ on a residual against an on-lattice point is exactly GPTQ of the instruct weights on that
  lattice. **On a shared lattice, closed-loop inter-model coding is free in quality; the whole saving is the
  conditional entropy H(q_inst − q_base).** (The 0.001–0.01 KL gap to `indep_gptq_ecsq_c0.35` comes from the
  step being c·rms(base row) instead of c·rms(instruct row).)
  ⇒ {GPTQ-ECSQ base, instruct}: 3.578 + 1.114 = **4.69 bpw-sum vs 7.16 (−34 %)**, both models at their
  best int4-class quality.
* Embedding (tied, 136 M params, kept fp16 in all runs above; rate only, `results/c2_embedding_qwen05b_instruct.json`):
  on the int4-class grid I-frame 3.55 bpw → P-frame **1.10 bpw** (25 % of indices change; not a single
  embedding row is bit-identical between base and instruct).
* Decode cost (qualitative): one extra entropy-decode + integer add per weight at load time; random access to
  the instruct model needs the base decoded first (star GOP = depth 1).

## 6. D — training checkpoints as frames (Pythia-410M)

### D1 probe — **PASS** (FACT, `results/probe_d1.json`, `figures/probe_d1_*.png`)

13 revisions streamed from the Hub straight into RAM (never written to disk): steps 1, 128, 512, 1k, 2k,
8k, 32k, 33k, 64k, 65k, 128k, 142k, 143k. Over the 96 block linears:

| pair | gap | ‖Δ‖/‖W‖ | top-5%-rank energy (null 0.11) | P/I bits, int4-class grid | indices changed | lossless fp16: H(bits) → H(bits⊕prev) |
|---|---|---|---|---|---|---|
| 1 → 128 | 127 | 0.051 | **0.74** | 0.13 | 9 % | 13.44 → 9.37 |
| 128 → 512 | 384 | 0.180 | 0.62 | 0.33 | 29 % | 13.44 → 11.21 |
| 512 → 1k | 488 | 0.379 | 0.50 | 0.52 | 48 % | 13.45 → 12.18 |
| 1k → 2k | 1000 | 0.548 | 0.31 | 0.71 | 69 % | 13.46 → 13.14 |
| 2k → 8k | 6000 | 0.766 | 0.17 | 0.91 | 87 % | 13.47 → 13.84 |
| 8k → 32k | 24000 | 0.722 | 0.15 | 0.89 | 86 % | 13.48 → 13.74 |
| 32k → 33k | 1000 | 0.131 | 0.14 | 0.33 | 30 % | 13.48 → 11.57 |
| 33k → 64k | 31000 | 0.474 | 0.16 | 0.71 | 70 % | 13.49 → 13.15 |
| 64k → 65k | 1000 | 0.093 | 0.15 | **0.27** | 21 % | 13.49 → 11.11 |
| 65k → 128k | 63000 | 0.445 | 0.21 | 0.66 | 62 % | 13.51 → 12.90 |
| 128k → 142k | 14000 | 0.071 | 0.20 | 0.22 | 16 % | 13.52 → 10.65 |
| 142k → 143k | 1000 | **0.022** | 0.20 | **0.096** | 5 % | 13.52 → 9.12 |

* Pre-registered criterion (late 1000-step pairs, P/I ≤ 0.5): mean **0.18** → PASS.
* The 1000-step delta shrinks monotonically through training: 0.548 (1k) → 0.131 (32k) → 0.093 (64k) →
  **0.022** (142k) — consistent with the cosine LR decay (INTERPRETATION).
* **Surprise (FACT):** checkpoint deltas are low-rank *only at the very start* (step 1→128: 74 % of energy
  in top-5 % of ranks); from ~2k steps on they are as full-rank as fine-tune deltas (0.14–0.21 vs null 0.11).
* Lossless: XOR with the previous checkpoint removes 1.4–4.4 bits of the 13.5-bit fp16 entropy for close
  checkpoints, but almost nothing for far ones (gap ≥ 6k: XOR is *worse* than raw).

### D2 codec: training as video — **confirmed** (FACT, `results/d2_pythia.json`, `figures/d2_pythia_frames.png`)

Consecutive 1000-step checkpoints streamed one by one (RAM only). Lossy: ECSQ grid with the I-frame's
step s = c·rms(row) fixed for the GOP; P-frames are closed-loop residuals against the decoder state
(kept in fp16). Equal step ⇒ equal weight-MSE (measured: identical to 5 digits for I vs IPPP).

**Late training, steps 136k…143k (8 frames)**

| scheme | c=0.35 (int4-class): mean bpw per P-frame | GOP total (8 frames, bpw-sum) | c=0.7: per P-frame | GOP total |
|---|---|---|---|---|
| all-I (independent) | 3.570 | 28.56 | 2.592 | 20.74 |
| I every 4 + P | 0.799 (P ≈ 0.338, I ≈ 3.57) | 9.17 (−68 %) | 0.536 | 6.36 (−69 %) |
| **I + 7 P (IPPP…)** | **0.338** | **5.94 (−79 %)** | **0.194** | **3.96 (−81 %)** |

* P-frame = **9.5 % of an I-frame** at int4-class, 7.5 % at int3-class; constant across the window.
* Drift check (WikiText-2, 40 windows, Pythia tokenizer): original step143000 PPL 15.83; I-frame c=0.35
  23.08; **end of the 7-P-frame closed-loop chain 23.16** (c=0.7: 67.42 vs 67.45). No meaningful drift.
  (Plain RTN-ECSQ without GPTQ/Fisher degrades Pythia a lot — irrelevant here: I and P have the same distortion.)
* **Lossless archive** (bit-exact fp16): byte-split entropy 13.54 → **9.31 bit/weight** with XOR-to-previous
  (−31 %); 16-bit-symbol bound 13.52 → 9.12.
* Random access: frame t in an IPPP GOP needs I + (t mod GOP) P-frame decodes; K=4 caps it at 3 extra
  decodes for +54 % bits vs IPPP — the classic video trade-off, now for checkpoints.

**Mid training, steps 32k…37k (6 frames)**: P-frame **1.175 bpw vs I 3.590** (33 %) at c=0.35, 0.745 vs 2.611 at
c=0.7; GOP total all-I 21.54 → IPPP **9.46 (−56 %)**, K=4 11.88 (−45 %). Drift check: original step37000 PPL 18.00,
I-frame 19.19, end of 5-P chain 19.20 (c=0.7: 24.02 vs 23.99). Lossless 13.51 → 11.58 bit (−14 %).
The P-frame cost tracks the learning-rate schedule: ~33 % of an I-frame mid-training, ~9.5 % at the end.

## 7. E — SVC / progressive bitstream (FACT, `results/svc_e_qwen05b.json`, `results/svc_direct_rates_qwen05b.json`, `figures/rd_svc_e.png`)

One stream: base layer = ECSQ with step s0 = 1.4·rms(row) (int2-class), each enhancement layer re-quantizes
the residual with step s0/2^k. Decoding layers 0..k gives a usable model at each k.

| layers decoded | equiv. step c | RTN layers: cum bpw (ctx-coded) | PPL / KL | single-shot ECSQ at same step | GPTQ layers: cum bpw (ctx) | PPL / KL | single-shot GPTQ-ECSQ |
|---|---|---|---|---|---|---|---|
| L0 | 1.4 | 1.675 | 18 489 / 6.90 (collapse) | 1.675 | 1.720 | 294 / 2.79 | 1.720, PPL 294 |
| L0+1 | 0.7 | 3.171 (3.081) | 31.25 / 0.878 | 2.597, PPL 31.25 | 3.415 (3.295) | **16.95** / 0.249 | 2.610, PPL 16.95 |
| L0..2 | 0.35 | 4.671 (4.332) | 15.23 / 0.153 | 3.575, PPL 15.23 | 5.122 (4.841) | **13.84** / 0.060 | 3.578, PPL 13.84 |
| L0..3 | 0.175 | 6.171 (5.582) | 13.50 / 0.037 | 4.570 | 6.832 (6.389) | **13.25** / 0.016 | — |

* **Quality: free.** Every cumulative layer reconstructs *exactly* the single-shot quantizer at that step
  (identical PPL and KL to 3–4 digits — the step-halving lattices contain each other, and GPTQ on an
  on-lattice residual equals GPTQ on the finer lattice, same argument as in C2).
* **Rate: not free with step ratio 2.** Overhead vs single-shot: +0.57 / +1.10 / +1.60 bpw after 1/2/3
  enhancement layers (RTN; +0.48 / +0.76 / +1.01 with the previous layer's symbol as context), larger with
  GPTQ layers (+0.81 at L1). Cause: a mid-tread quantizer refined by 2 has *non-nested cells* — fine points
  on a coarse cell boundary belong to two coarse cells, so each refinement symbol carries ~1.5 bit for a 1-bit
  step halving. This is exactly why JPEG2000 uses a dead-zone quantizer (ratio-2 bit-planes nest) and why
  successive-refinement theory prefers nested cells.
* Model-specific caveat: Qwen2.5-0.5B collapses at any 2-bit-class point (also GPTQ int2: PPL 1.3e5), so the
  base layer is not a usable model on its own here; the first usable SVC point is L0+1 with GPTQ layers.

**Fix: nested lattice (step ratio 3).** With mid-tread cells refined by 3, every coarse cell contains exactly
3 fine cells, so H(fine) = H(coarse) + H(refinement | coarse) holds exactly (FACT, `r3_*` in the same json):

| layers decoded | equiv. step c | cum bpw (plain) | **cum bpw (ctx-coded)** | single-shot ECSQ at same step | overhead (ctx) | PPL / KL |
|---|---|---|---|---|---|---|
| L0 | 1.4 | 1.675 | 1.675 | 1.675 | 0 | 18 489 / 6.90 |
| L0+1 | 0.467 | 3.264 | **3.169** | 3.166 | **+0.004** | 18.61 / 0.368 |
| L0..2 | 0.156 | 4.854 | **4.756** | 4.740 | **+0.016** | 13.45 / 0.030 |

⇒ **One stream, several operating points, essentially zero rate penalty** once the quantizer is nested and the
refinement symbols are coded with the coarser symbol as context. The price is a coarser ladder (×3 steps ≈
1.58 bit per layer) — or use a dead-zone quantizer for a ×2 ladder (not run).

## 8. Out-of-domain check of the intra-model coders (FACT, `results/crossdomain_c4_qwen05b.json`)

All calibration (GPTQ Hessians, Fisher) used WikiText-2 *train*. Re-evaluated on C4 validation (allenai/c4 en
shard 0, first 64×2048 tokens; KL on 16 windows), quantizers re-run identically:

| config | bpw | WikiText-2 PPL (in-domain) | WikiText KL | **C4 PPL (out-of-domain)** | C4 KL | C4 top-1 |
|---|---|---|---|---|---|---|
| fp16 | 16.000 | 13.071 | 0.0000 | 19.276 | 0.0000 | 1.000 |
| rtn_int4_g128 | 3.756 | 15.542 | 0.1878 | 22.883 | 0.1903 | 0.755 |
| gptq_int4_g128 | 3.757 | 14.033 | 0.0746 | 21.250 | 0.1143 | 0.816 |
| ecsq_row_c0.35 | 3.575 | 15.229 | 0.1527 | 22.308 | 0.1613 | 0.774 |
| gptq_ecsq_row_c0.35 | 3.578 | 13.838 | 0.0599 | 20.828 | 0.0892 | 0.835 |
| gptq_ecsq_row_c0.5 | 3.075 | 14.740 | 0.1215 | 22.545 | 0.1800 | 0.772 |
| fisher_g0.5_lam0.5 | 3.137 | 14.353 | 0.0999 | 21.190 | 0.1074 | 0.820 |
| gptq_fisher_g0.5_lam0.5 | 3.142 | 13.855 | 0.0559 | 20.893 | 0.0996 | 0.837 |
| gptq_fisher_g0.5_lam0.35 | 3.647 | 13.423 | 0.0271 | 19.993 | 0.0465 | 0.887 |

* The ranking survives out of domain: GPTQ-ECSQ (3.58 bpw) beats GPTQ-int4 g128 (3.76 entropy / 4.16 fixed) on C4
  PPL and KL; Fisher-step GPTQ-ECSQ at **3.14 bpw** still beats GPTQ-int4 (C4 PPL 20.89 vs 21.25, KL 0.100 vs 0.114).
* Part of A3's in-domain edge is domain fit: at ~equal C4 PPL, Fisher-GPTQ (3.14 bpw) vs GPTQ-ECSQ (3.58 bpw) is
  20.89 vs 20.83, but KL flips (0.100 vs 0.089), whereas in-domain Fisher-GPTQ had the lower KL (0.056 vs 0.060).
  So the robust out-of-domain A3 gain is ≈0.3–0.4 bpw rather than the in-domain 0.42.

## 9. Surprises (FACT unless marked)

1. **Coder ≠ small delta.** Qwen2.5-Coder-0.5B is as far from Qwen2.5-0.5B (‖Δ‖/‖W‖ 0.88, cos 0.60) as two
   Pythia checkpoints 30–60 k steps apart, and Coder-Instruct is even farther from Coder (1.05). Yet the neuron
   basis is preserved (99.96 % identity matching). "Same architecture + same init" does not imply "cheap P-frame".
2. **Instruct delta is small but carries a lot**: 7.7 % of the norm, but zeroing it costs chat-KL 0.28
   (more than int4-quantizing the instruct model, 0.15).
3. **Shared-lattice equivalence**: closed-loop P-frame coding on the base's lattice reproduces *exactly* the
   independent quantization (RTN or GPTQ) of the fine-tune; inter-model coding is a pure entropy-coding gain.
   The same identity makes SVC layers reproduce single-shot quantizers exactly.
4. **DCT is slightly *harmful*** on weights (+0.007 bit) — it spreads heavy-tailed outliers; Hadamard is
   exactly neutral for entropy coding (it helps only fixed-rate grids).
5. **Low-rank only at the very beginning of training** (step 1→128: 74 % energy in 5 % of ranks), never for
   fine-tunes or late checkpoints.
6. **WikiText PPL rewards forgetting the fine-tune**: partial reversion toward base lowers the instruct
   model's WikiText PPL below its own fp16 value (13.64 vs 14.24).
7. A quantizer with smart steps but plain rounding (Fisher-ECSQ, no calibration-based rounding) beats GPTQ
   on a naive grid (3.38 vs 3.47 bpw at GPTQ-int4 quality).

## 10. Limitations (read before quoting numbers)

Phase-1 limitations; the first and the last item were addressed in phase 2 (§14), whose own limitations are in §14.10.

* **Rate is an entropy proxy** (order-0, per tensor, static table cost included) — no rANS bitstream was
  produced; real coders land within ~0.1–1 % of it. Context modelling beyond the SVC experiment was not tried.
  *(Phase 2: real rANS files match the proxy to ±0.02 %, §14.2.)*
* **One main model family** (Qwen2.5-0.5B, notoriously quantization-sensitive) + one checkpoint series
  (Pythia-410M). Numbers will shift on bigger / more robust models; int2 results are dominated by this model's
  collapse.
* **MPS fp16 inference** (checked equal to fp32 within 1e-5 PPL on 8 windows); single seed; KL on 16 windows
  (32 k tokens) and 48-prompt chat set — no confidence intervals were computed.
* **Calibration domain = evaluation domain** for GPTQ/Fisher (WikiText-2 train → test); see the C4 check (§8).
  The instruct model's Hessians were also WikiText-calibrated (no chat calibration), which favours wiki-KL over chat-KL.
* GPTQ is the non-sequential variant (Hessians from the fp model), no act-order, no MSE-optimal clipping —
  stronger int4 baselines (AWQ/OmniQuant/QuIP#) would narrow the A-results; they would *not* change the C/D
  results, which are relative to the same quantizer.
* Embeddings and norms were kept fp16 in all RD curves (the embedding is 27.6 % of Qwen2.5-0.5B's parameters);
  the embedding delta was measured rate-only.
* Math variant of Qwen2.5 does not exist at 0.5B — the family test used {base, instruct, coder, coder-instruct}.
* No decode-speed benchmarks; decode cost is discussed qualitatively. *(Phase 2: measured, §14.6.)*

## 11. What next

As written after phase 1. Item 1 and parts of items 2–4 were done in phase 2 (§14); the updated list is §14.11.

1. Real entropy coder (rANS, per-tensor static tables) + measured decode throughput; fused ANS→int8 GEMM.
2. Fisher-step + GPTQ + family P-frames combined into one container ("hub codec"): base I-frame + per-variant
   P-frames, with chat-calibrated Hessians for instruct variants.
3. Dead-zone / ratio-3 nested quantizers for SVC with conditional coding (measure true successive-refinement gap).
4. Training-checkpoint archive codec at full temporal resolution (all 143 Pythia checkpoints, closed-loop, GOP
   sweep) and on LoRA/full-FT model zoos (many fine-tunes of one base → one I-frame + many sub-1-bpw P-frames).
5. Repeat on a larger, more robust model (e.g. Qwen2.5-1.5B / Llama-3.2-1B) and add downstream task metrics.

## 12. Process notes / deviations (for honesty)

* **One mis-gridded sweep, rerun:** the first `delta_ecsq`/`delta_gptq` sweep in C2 used steps 0.7–11.2·rms(W),
  far coarser than Δ (≈0.077·rms) — everything rounded to 0. Caught after one point (kept: `delta_ecsq_c0.7`,
  0.03 bpw ≈ "instruct := base"), grid fixed to 0.03–0.5 and rerun. Documented in `c2_family_codec.py`.
* **Spectral reordering (A1) initially failed the positive control** (dense |corr| graph); switched to a kNN graph
  *before* looking at real-weight results; the probe was rerun from scratch.
* D1 was restarted once (download-to-cache tripped the disk guard; switched to in-memory streaming; finished pairs
  were kept). D2 crashed once on a 1-D LayerNorm tensor (`post_attention_layernorm` matched the linear-name filter;
  D1 was unaffected because it only used 2-D tensors — verified: 96 linears counted) and was restarted.
* Two runs were killed by me because two model processes running at once overloaded the machine's memory:
  first spine run (resumed, finished configs kept), first C2 and D2 attempts. After that,
  strictly one heavy process at a time.
* Interpolated "bits at matched quality" numbers are linear in log-metric between the two bracketing measured points.

## 13. Environment, downloads, disk

* Python 3.11.4 project venv `.venv` (created with `--system-site-packages`, 24 MB) on top of the existing
  torch 2.7.0 / transformers 5.2.0 / datasets 4.5.0 / safetensors 0.8.0 / numpy 2.2.6 / scipy 1.15.3 /
  matplotlib 3.10.3 — **no packages were installed or changed** in the global environment.
* Downloaded (public HF): Qwen2.5-0.5B, -Instruct, -Coder-0.5B, -Coder-0.5B-Instruct (≈1 GB each, safetensors);
  Pythia-410M `pytorch_model.bin` (fp16, 0.9 GB) for 13 (D1) + 14 (D2) revisions, ≈30 GB of network incl. restarts;
  the first 3 D1 revisions went through a per-revision temp dir that was deleted right after loading, all later ones
  were streamed **into RAM only** (0 bytes on disk); always loaded with `torch.load(weights_only=True)`.
  WikiText-2 raw; one C4 validation shard (39 MB).
* Cache hygiene: Coder models deleted right after C1/C3/C4; Qwen base/instruct, C4 shard and Pythia configs deleted
  after all series finished. Project-local `data/` (146 MB,
  git-ignored) keeps tokenized eval sets, reference hidden states and Fisher statistics for fast re-runs.
* No weights, caches or `data/` are in git (`.gitignore`); git holds scripts, `results/*.json`, `figures/*.png`, docs.

## 14. Phase 2 — real ANS bitstream & decode benchmark (2026-11-21)

Phase 1 only *estimated* rates (order-0 entropy + table + side info, §1). Phase 2 turns the winners into a real
codec: every rate below is **bytes of a file on disk** (header, tables, steps/scales, ANS words — everything),
`real bpw = file bytes · 8 / weights`, and every quality number is measured on a model **decoded from that file**.
Legend as before: **FACT** = measured (number in `results/phase2_*`), **INTERPRETATION** = my reading.

### 14.0 TL;DR (phase 2)

| phase-1 claim / question | phase-1 estimate | phase 2 — real files (FACT) | verdict |
|---|---|---|---|
| entropy proxy ≈ what a real coder writes | "within ~0.1 %" | Qwen, 13 identical configs: real rANS bpw = estimate **±0.02 %**; P-frames **−1.15 %** (they inherit the I-frame's steps: 0 side info) | **confirmed** |
| quality of the coded models | — | models decoded from the files reproduce the phase-1 PPL / KL / chat-KL **to 4 digits** in 12 of 13 configs (P-frame c=0.5: 0.2556 → 0.2558, the base I-frame now uses the base's own Hessian) | **confirmed** |
| family codec {base, Instruct}, int4-class | 4.69 vs 7.16 bpw-sum (−34 %) | Qwen linears: 160.0 + **49.2 MB** vs 160.0 + 160.1 MB (**−35 %**); **whole pair incl. lossless embeddings: 532 MB vs 678 MB independent (−22 %) vs 917 MB as two standard GPTQ-int4 files (−42 %)**, same fine-tune chat-KL (0.128 vs 0.117 / 0.148). SmolLM2-360M pair: **295 MB** vs 406 MB (−27 %) vs 528 MB (−44 %), chat-KL 0.138 / 0.138 / 0.118; SmolLM2-135M: 131 vs 171 vs 228 MB | **confirmed** on 3 base+SFT pairs (full-file gain smaller: embeddings are coded losslessly) |
| full-rank Δ + entropy ≫ LoRA-style low-rank Δ | 2–13× lower KL | Qwen, matched real MB: **2.3×** (37.7 vs 35.2 MB), **6.7×** (75.4 vs 70.4 MB), **17×** (114.9 vs 140.8 MB); SmolLM2-135M: **2.2×** at 59 % of the bytes, **14×** at 88 %, **7×** at 44 %; SmolLM2-360M: **2.8×** at 88 % / **10.8×** at 91 % / **25.2×** at 70 % | **confirmed on real bytes, on 2–3 families** |
| training checkpoints as video | P-frame 9.5 % of I | late window: P-frame **0.339 bpw = 9.5 % of the I-frame**; 8-frame GOP **224.7 MB vs 1081 MB all-I (−79 %)**; mid −56 %; P-chain end bit-identical to the lattice quantizer (no drift); x86-encoded files decode bit-exactly on the Mac (sha256 match on CPU and GPU) | **confirmed** (encoded off-Mac on an HF Job) |
| SVC with nested ×3 lattice ≈ free | +0.004 / +0.016 bpw | ratio-3 nested lattice + context: **+0.0047 / +0.0192 bpw (+0.15 / +0.40 %)** over single-shot files; ratio 2: +0.49 / +0.76 bpw | **confirmed (≈ free)** |
| decode cost "GB/s on CPU/GPU" (qualitative) | — | Qwen linears (358 M weights, warm page cache): constriction rANS **7–9.5 s** (1 thread, GIL-bound); own `rans16` **1.1 s** (8 cores) / **0.6 s** Metal GPU; GPTQ-int4 packed 0.6 s; fp16 safetensors 0.13 s. P-frame chain (base + P) 15.4 → 2.3 → **0.96 s** | **ANS decode is a real cost**; with a parallel / GPU decoder it is ~1 s per 0.5 B model |
| generic coder instead of ANS? | not asked | zstd-19 on the same int8 symbols: **+0.8–2.8 %** bytes on int4-class I-frames (decodes ≈ 1.5 GB/s, 1 thread) but **+6–23 %** on P-frame / Δ streams | zstd is fine for I-frames; **ANS is needed for the family streams** |
| run it on Hugging Face | — | HF Jobs **available**: 8 CPU jobs launched — Pythia trajectory **completed** off-Mac (plus a cancelled first attempt), SmolLM2-360M base stage done off-Mac before two jobs were **cancelled externally**, 4 short starts killed/failed by my own bugs; private dataset repo for bitstreams; **private static Space** with results + in-browser decoding (a Gradio Space needs paid hardware: HTTP 402) | done (static) |

### 14.1 What was built and installed (FACT)

* **Installed:** `constriction 0.5.0` (pip, into the project `.venv` only). Already present and used:
  `zstandard 0.19`, `numba 0.65.1`, `gradio 5.31` (system site-packages via the venv). Nothing else changed.
* **`wcodec/` — the codec.** One `.wcdc` file = magic + zlib-compressed JSON header + binary sections; every
  byte of the file (header, frequency tables, fp16 steps / scales, 4-bit zeros, ANS words) is counted.
  Record kinds: `raw16`, `ll16` (lossless fp16: hi byte + lo byte coded with 16 hi-class contexts, optional
  XOR with a reference tensor = lossless P-frame), `qrow` (per-row-step lattice indices; own / inherited /
  nested step; accumulation `Q = Q_ref + q` for closed-loop P-frames and `Q = 3·Q_ref + q` for SVC layers;
  optional context streams), `g128` (GPTQ min-max groups; ANS or fixed-rate nibble packing), `lowrank`.
* **Entropy coders (all produce real bytes):** (1) `constriction` rANS, static `Categorical` model rebuilt by the
  decoder from stored counts, 2^20-symbol independent chunks; (2) **`rans16`** — own rANS (32-bit state,
  16-bit words, 15-bit frequencies), numba encoder/decoder, 2^16-symbol chunks decoded in parallel (lossless
  transcode of any constriction file, +0.05 % bytes); (3) a **Metal GPU** decoder of `rans16`
  (`torch.mps.compile_shader`, one GPU thread per 4096-symbol chunk, +0.4 % bytes); (4) a **JavaScript**
  decoder of `rans16` used by the Space (runs in the visitor's browser).
* **Tests:** `phase2/test_wcodec.py` — every record kind round-trips bit-exactly (CPU, MPS, and x86 at the start
  of every HF Job); GPTQ symbols identical to the phase-1 implementation; packed int4 = 4.1562 bpw (formula).
  Every quality number below is measured on a model whose weights were **decoded from the written file**.

### 14.2 Real bytes vs the phase-1 estimate (FACT, `results/phase2_family_qwen05b.json`, `results/phase2_tables.md`)

Same quantizers, calibration seeds and Hessians as phase 1 (deterministic), so the decoded models should be the very
reconstructions phase 1 evaluated — the identical metrics confirm it; the new thing is the byte count. (Exception: the base
I-frame now uses the base's own Hessian instead of the Instruct one, which moves P-frame lattice points slightly.)

| file (Qwen2.5-0.5B linears, 357.8 M weights) | phase-1 config | phase-1 estimate bpw | **real bpw** | real vs est. | MB | quality: phase 1 → phase 2 (decoded file) |
|---|---|---|---|---|---|---|
| base GPTQ-int4 g128 + rANS | gptq_int4_g128 | 3.7568 | **3.7570** | +0.01 % | 168.0 | wiki-KL 0.0746 → 0.0746, PPL 14.033 → 14.033 |
| base GPTQ-ECSQ c=0.35 (I-frame) | gptq_ecsq_row_c0.35 | 3.5784 | **3.5782** | −0.01 % | 160.0 | 0.0599 → 0.0599, PPL 13.838 → 13.838 |
| Instruct GPTQ-int4 g128 + rANS | indep_gptq_int4_g128 | 3.7571 | **3.7573** | +0.01 % | 168.1 | chat-KL 0.1482 → 0.1482 |
| Instruct GPTQ-ECSQ c=0.35 | indep_gptq_ecsq_c0.35 | 3.5785 | **3.5783** | −0.01 % | 160.1 | 0.1173 → 0.1173 |
| Instruct **P-frame** on the base lattice, c=0.35 | pframe_gptq2_c0.35 | 1.1137 | **1.1009** | **−1.15 %** | **49.2** | 0.1279 → 0.1279 |
| Instruct P-frame, c=0.5 | pframe_gptq2_c0.5 | 0.9781 | **0.9670** | −1.13 % | 43.3 | 0.2556 → 0.2558 |
| Instruct full-rank Δ (fp16 base), c=0.06 / 0.12 / 0.25 | delta_gptq_c* | 2.5685 / 1.6865 / 0.8425 | **2.5685 / 1.6866 / 0.8427** | ≤ +0.02 % | 114.9 / 75.4 / 37.7 | 0.0031 / 0.0105 / 0.0375 — identical |
| Instruct LoRA-style Δ r=32 / 128 (fp16 factors) | delta_lowrank_r* | 0.7868 / 3.1473 | **0.7869 / 3.1473** | 0.00 % | 35.2 / 140.8 | 0.0874 / 0.0522 — identical |

* The order-0 proxy charged 32 bit per used symbol for the table; the real file stores LEB128 counts (≈1–3 B),
  which is why real ECSQ files are a hair *below* the proxy. P-frames are 1.1 % below because the phase-1 proxy
  charged an fp16 step per row; the real P-frame **inherits the I-frame's steps** (`step=ref`).
* `rans16` (own coder, 15-bit frequencies) costs **+0.05–0.06 %** bytes vs constriction; its GPU variant with
  4096-symbol chunks **+0.4 %**; both decode bit-identically (sha256-checked).
* SmolLM2-135M and SmolLM2-360M show the same agreement (real vs proxy within ±0.07 %, `real vs proxy` column of
  `results/phase2_tables.md`).

### 14.3 Family codec on real files: format → MB → decode → quality

**Qwen2.5-0.5B → Qwen2.5-0.5B-Instruct** (‖Δ‖/‖W‖ = 0.079). Linears only (357.8 M weights); decode = time to
rebuild all fp16 linears from the file(s), page cache warm, median of 2 runs, Apple Silicon laptop (8 CPU cores).
P-frame rows decode the chain *base I-frame + P-frame*; Δ/LoRA rows assume the fp16 base is already in RAM.

| fine-tune format | MB | real bpw | chat-KL | wiki-KL | PPL | decode CPU constriction 1 thr | CPU `rans16` 8 thr | GPU (Metal rANS) |
|---|---|---|---|---|---|---|---|---|
| fp16 safetensors (whole model, 988 MB) | — | 16 | 0 | 0 | 14.24 | 0.13 s (mmap load) | — | 0.25 s (to MPS) |
| GPTQ-int4 g128, fixed-rate packed | 185.9 | 4.156 | 0.1482 | 0.0799 | 15.31 | 0.58 s (unpack+dequant) | — | 0.55 s |
| GPTQ-int4 g128 + rANS | 168.1 | 3.757 | 0.1482 | 0.0799 | 15.31 | 6.85 s | 1.05 s | 0.69 s |
| GPTQ-ECSQ c=0.35 + rANS | 160.1 | 3.578 | 0.1173 | 0.0645 | 15.09 | 9.16 s | 1.13 s | 0.62 s |
| **P-frame c=0.35** (+ base I-frame 160.0 MB) | **49.2** | **1.101** | 0.1279 | 0.0657 | 15.07 | 15.35 s (chain) / 6.19 s (P only) | 2.35 s / 1.11 s | **0.96 s / 0.43 s** |
| full-rank Δ c=0.25 (fp16 base in RAM) | 37.7 | 0.843 | 0.0375 | 0.0205 | 14.24 | 6.25 s | 1.17 s | 0.58 s |
| full-rank Δ c=0.12 | 75.4 | 1.687 | **0.0105** | 0.0058 | 14.27 | 7.17 s | 1.20 s | 0.61 s |
| full-rank Δ c=0.06 | 114.9 | 2.568 | **0.0031** | 0.0018 | 14.25 | — | — | — |
| LoRA-style r=32 (fp16 factors) | 35.2 | 0.787 | 0.0874 | 0.0880 | 13.64 | 0.38 s (A·B matmul) | — | 0.22 s |
| LoRA-style r=64 | 70.4 | 1.574 | 0.0706 | 0.0773 | 13.62 | — | — | — |
| LoRA-style r=128 | 140.8 | 3.147 | 0.0522 | 0.0608 | 13.67 | 0.45 s | — | 0.29 s |

**The pair {base, Instruct}, every tensor on disk** (embeddings / norms / biases = 136.2 M params per model; lossless:
16 → 10.51 bit/param for the base, 8.42 bit/param for the Instruct rest as XOR-with-base):

| how both models are stored | pair MB | base PPL | Instruct chat-KL |
|---|---|---|---|
| both fp16 | 1976 | 13.07 | 0 |
| two standard GPTQ-int4 g128 files (rest fp16) | 917 | 14.03 | 0.148 |
| both GPTQ-int4 + rANS, rest lossless | 694 | 14.03 | 0.148 |
| both independent GPTQ-ECSQ c=0.35, rest lossless | 678 | 13.84 | 0.117 |
| **family: base I-frame + Instruct P-frame (c=0.35) + XOR rest** | **532** | 13.84 | 0.128 |
| family, c=0.5 | 503 | 14.74 | 0.256 |
| lossless fp16 base (linears raw) + Instruct Δ c=0.12 | 1114 | 13.07 | 0.0105 |

* Family vs two standard int4 files: **−42 %**; vs the best independent entropy-coded pair: **−22 %**. In full-file terms the
  gain is smaller than the phase-1 linears-only −34 % because the embedding is coded **losslessly**: for the Instruct
  model the 143 MB lossless XOR rest is now 3× its 49 MB P-frame. INTERPRETATION: a lossy embedding P-frame (phase 1
  measured 1.10 bpw rate-only, quality never measured) is the next obvious step.
* **"Full Δ + entropy ≫ LoRA" holds on real bytes:** at matched MB, chat-KL is **2.3×** lower (37.7 vs 35.2 MB),
  **6.7×** lower (75.4 vs 70.4 MB) and **17×** lower at 18 % *fewer* bytes (114.9 vs 140.8 MB). Even if the LoRA factors
  were stored at int8 (half the bytes), r=64 at ≈35 MB would still be ≥ 0.07 vs 0.0375 for Δ c=0.25 (INTERPRETATION: the
  low-rank error is truncation-limited, not precision-limited).
* WikiText PPL keeps misleading (phase-1 §5): low-rank reconstructions get PPL 13.6 < the Instruct model's own 14.24.

**SmolLM2-135M → SmolLM2-135M-Instruct** (‖Δ‖/‖W‖ = 0.081; GPTQ group 64 because 576/1536 are not multiples of 128;
the Space demo pair). This model is very fragile at int4-class rates (every ≈3.6–4.3 bpw format: chat-KL 0.40–0.54):

| fine-tune format | MB | real bpw | chat-KL | decode CPU constriction | CPU rans16 | GPU |
|---|---|---|---|---|---|---|
| GPTQ-int4 g64 fixed-rate packed | 57.2 | 4.313 | 0.400 | 0.23 s | — | 0.31 s |
| GPTQ-int4 g64 + rANS | 53.8 | 4.051 | 0.400 | 2.13 s | 0.44 s | 0.39 s |
| GPTQ-ECSQ c=0.35 | 47.7 | 3.594 | 0.505 | 2.67 s | 0.40 s | 0.30 s |
| P-frame c=0.35 (+ base I-frame 47.7 MB) | **14.6** | 1.098 | 0.538 | 4.50 s (chain) | 0.76 s | 0.48 s |
| full-rank Δ c=0.25 / 0.12 / 0.06 | 11.5 / 22.6 / 34.5 | 0.87 / 1.71 / 2.60 | 0.102 / **0.030** / **0.010** | ≈2.0–2.4 s | ≈0.48 s | ≈0.30 s |
| LoRA-style r=32 / 64 / 128 | 19.5 / 39.1 / 78.2 | 1.47 / 2.94 / 5.89 | 0.220 / 0.144 / 0.071 | 0.21–0.23 s | — | 0.21–0.25 s |

Full Δ vs LoRA, SmolLM2-135M: Δ c=0.25 (11.5 MB) beats r=32 (19.5 MB) **2.2×** at 59 % of the bytes; Δ c=0.06 (34.5 MB)
beats r=64 (39.1 MB) **14×** and r=128 (78.2 MB) **7×** at 44 % of the bytes. Pair: family 130.5 MB vs independent
170.5 MB (−23 %) vs two GPTQ-int4 files 227.9 MB (−43 %). Quality caveat (FACT): at the int4 class the P-frame fine-tune
(chat-KL 0.54) is close to independent GPTQ-ECSQ (0.505) but clearly worse than GPTQ-int4 g64 (0.40, which pays 4.3 bpw incl.
per-64 scales). INTERPRETATION: on this fragile model the per-row ECSQ step (inherited from the base) is the weak point, not the
inter-model coding — the full-rank Δ over an fp16 base has no such problem (22.6 MB → 0.030).

### 14.4 Training trajectory as video — Pythia-410M on an HF Job (FACT, `results/phase2_trajectory_pythia410m.json`, `results/phase2_xcheck_trajectory.json`, `figures/p2_pythia_frames_real.png`)

Run **off-Mac as an HF Job** (CPU, 8 vCPU / 32 GB, x86 Linux): 14 checkpoints streamed from the Hub into RAM,
real WCDC files for every frame and scheme (all-I with own steps; IPPP = closed-loop P-frames on the I-frame's lattice with
integer state `Q_t = Q_{t-1} + q_t`; K=4 GOP; lossless XOR-to-previous archive), uploaded to the private dataset
`lifeart/weight-codec-phase2` (`runs/pythia410m_traj_v2/`). Linears = 302.0 M weights; rest (embed_in/out, norms, biases) = 103.3 M.

| window (frames) | scheme, c=0.35 (int4-class) | GOP MB (linears) | mean bpw | c=0.7 GOP MB | c=0.7 mean bpw |
|---|---|---|---|---|---|
| late 136k…143k (8) | all-I | 1081.2 | 3.580 | 785.7 | 2.602 |
| | I every 4 + P | 347.3 (−68 %) | — | 240.8 (−69 %) | — |
| | **I + 7 P** | **224.7 (−79 %)** | P = **0.339** (9.5 % of I) | **149.8 (−81 %)** | P = 0.195 |
| mid 32k…37k (6) | all-I | 811.0 | 3.581 | 589.4 | 2.602 |
| | **I + 5 P** | **357.0 (−56 %)** | P = **1.176** (33 % of I) | 239.1 (−59 %) | P = 0.746 |
| lossless archive (linears) | I → XOR-P | late 13.53 → **9.31** bit/weight; mid 13.50 → 11.57 | | | |

* **Phase-1 estimate vs real bytes:** P-frame 0.338 → **0.339** bpw (late), 1.175 → **1.176** (mid); GOP totals 5.94 → 5.95 and
  9.46 → 9.46 bpw-sum; lossless 13.54 → 9.31 → **13.53 → 9.31**. Real = estimate within 0.3 %.
* **No drift, by construction (FACT):** with the integer decoder state the end of the 7-P chain is *bit-identical* to the ECSQ
  quantizer of the last checkpoint on frame 0's lattice (`chain_end_equals_lattice_quantizer: true`, both windows, both c).
  INTERPRETATION: phase 1's tiny drift (23.08 → 23.16 PPL, same lattice) came from keeping the decoder state in fp16.
* **Cross-machine decode (FACT, `phase2/xcheck_trajectory.py`):** the x86-encoded chain files were downloaded from the dataset and
  decoded on the Mac — **sha256 of the decoded chain end matches the job's for every window and c (CPU and Metal GPU paths)**.
  Decoding the whole 8-frame late chain (8 × 302 M symbols, 224.7 MB): constriction 1 thread **38.7 s**, rans16 8 threads **7.4 s**,
  Metal GPU **2.9 s** (on the 8-vCPU job: 173 s / 189 s — see limitations).
* **Drift check on the decoded files (MPS fp16, 40 windows, phase-1 protocol):**

  | window | original last checkpoint | its own I-frame (decoded) | end of the closed-loop P chain (decoded) |
  |---|---|---|---|
  | late, c=0.35 | 15.829 | 23.272 | **23.102** |
  | late, c=0.7 | 15.829 | 66.216 | 67.578 |
  | mid, c=0.35 | 18.002 | 19.182 | **19.193** |
  | mid, c=0.7 | 18.002 | 23.993 | 24.025 |

  The P-chain end is bit-identical to quantizing the last checkpoint on frame 0's lattice; its PPL differs from the last frame's
  own I-frame only because the two lattices differ (frame-0 vs own row-RMS step) — there is no accumulation error. Phase 1
  evaluated that same frame-0-lattice quantizer: mid window **identical** (19.193 vs 19.1929; 24.025 vs 24.0243), late window
  0.1–0.2 % apart (23.102 vs 23.078; 67.58 vs 67.42) — not explained (the decode itself is sha-verified). Plain RTN-ECSQ hurts
  Pythia a lot at both steps (as in phase 1); I and P share the distortion, which is the point of the comparison.

### 14.5 Progressive (SVC) stream in real bytes (FACT, `results/phase2_svc_qwen05b.json`, `figures/p2_svc_tax_real.png`)

Qwen2.5-0.5B base, RTN layers, one WCDC file per quality layer; layer k ≥ 1 is coded in 7 context streams
(ctx = previous layer's symbol clipped to ±3) and only stores `step = previous/ratio` (0 side info). The single-shot
reference at each layer's lattice stores the same fp16 base step once. **Decoding L0..k is asserted bit-identical to
decoding the single-shot file** (true for every layer), so quality is shared and the tax is pure bytes.

| ratio | layers decoded | layer file MB | cumulative MB | **cumulative real bpw** | single-shot real bpw | **tax** | PPL / KL (decoded) |
|---|---|---|---|---|---|---|---|
| 3 (nested) | L0 | 74.9 | 74.9 | 1.6752 | 1.6752 | 0 | 18 489 / 6.90 (collapse) |
| 3 | L0+1 | 66.9 | 141.8 | **3.1701** | 3.1654 | **+0.0047 bpw (+0.15 %)** | 18.61 / 0.368 |
| 3 | L0..2 | 71.0 | 212.8 | **4.7582** | 4.7390 | **+0.0192 bpw (+0.40 %)** | 13.44 / 0.030 |
| 2 (not nested) | L0+1 | 62.9 | 137.9 | 3.0823 | 2.5966 | +0.486 (+18.7 %) | (= single-shot c=0.7) |
| 2 | L0..2 | 56.0 | 193.8 | 4.3338 | 3.5747 | +0.759 (+21.2 %) | (= single-shot c=0.35) |

Phase-1 estimate → real: ratio-3 tax +0.004 / +0.016 → **+0.0047 / +0.0192** bpw; ratio-2 (ctx) +0.48 / +0.76 → +0.486 / +0.759;
PPL/KL identical (18.61 / 13.45, KL 0.368 / 0.030). **Confirmed in bytes: one nested stream ≈ free (≤ 0.4 %).**

### 14.6 Decode-speed benchmark (FACT, `results/phase2_decode_*_Darwin-arm64-8cpu.json`, `results/phase2_zstd_ref_*.json`)

Time to rebuild fp16 weights from the file(s) in RAM (page cache warm; download modelled separately), Apple Silicon
laptop, 8 CPU cores, MPS GPU; median of 2–3 runs after a warm-up. Entropy decoders: **constriction** (1 thread: its Python binding
holds the GIL, threads give 1.0×; "pool" = 8 worker processes on chunks via shared memory), **rans16** (own, numba, 1 or 8
threads), **Metal** (own, GPU, then GPU reconstruction). Raw entropy throughput (Qwen ECSQ, 358 M symbols): constriction
≈ 40 Msym/s per thread, pool ≈ 250 Msym/s; rans16 ≈ 120 Msym/s per core, ≈ 500 Msym/s on 8 cores; Metal ≈ 1 Gsym/s;
JavaScript (V8, 1 thread) ≈ 55 Msym/s; zstd-19 Huffman ≈ 1.5 G bytes/s (1 thread).

**Qwen2.5-0.5B-Instruct, linears (357.8 M weights)**

| format | MB | constriction 1 thr | constriction pool (8 proc) | rans16 1 thr | rans16 8 thr | rans16 8 thr + MPS recon | Metal GPU |
|---|---|---|---|---|---|---|---|
| GPTQ-int4 packed (no entropy decode) | 185.9 | 0.58 s | — | — | — | 0.54 s | 0.55 s |
| GPTQ-int4 + rANS | 168.1 | 6.85 s | 1.47 s | 3.33 s | 1.05 s | 1.12 s | 0.69 s |
| GPTQ-ECSQ c=0.35 | 160.1 | 9.16 s | 1.83 s | 3.37 s | 1.13 s | 1.07 s | 0.62 s |
| P-frame chain (base 160.0 + P 49.2 MB) | 209.2 | 15.35 s | 3.89 s | 6.66 s | 2.34 s | 2.15 s | **0.96 s** |
| P-frame only (base state in RAM) | 49.2 | 6.19 s | 1.45 s | 3.21 s | 1.11 s | 0.97 s | **0.43 s** |
| full-rank Δ c=0.12 (fp16 base in RAM) | 75.4 | 7.17 s | 1.58 s | 3.37 s | 1.20 s | 1.13 s | 0.61 s |
| LoRA r=32 (A·B matmul) | 35.2 | 0.38 s | — | — | — | 0.22 s | 0.22 s |
| fp16 safetensors, whole model | 988.1 | 0.13 s load | | | | 0.25 s to MPS | |

**Lossless rest (embeddings + norms + biases, 136.2 M params for Qwen)** — after replacing my numpy 16-context scatter by a
one-pass numba merge (the first benchmark of these files was implementation-bound; old timings kept in
`results/phase2_decode_*_rest_before_numba_scatter.json`):

| Qwen rest file | MB | constriction 1 thr | rans16 1 thr | rans16 8 thr |
|---|---|---|---|---|
| base / Instruct rest, lossless (10.5 bit/param) | 179.0 | 7.56 s | 2.63 s | 0.90 s |
| Instruct rest as lossless XOR-P vs base (8.4 bit/param) | 143.4 | 7.80 s | 2.99 s | 1.08 s |

**"Download + decode" for delivering Qwen2.5-0.5B-Instruct (whole model)** — MB measured, download *modelled* as MB / bandwidth,
decode measured (CPU, rans16 8 threads unless noted):

| what the client fetches | MB | decode | 50 Mbit/s | 200 Mbit/s | 1 Gbit/s | Instruct chat-KL |
|---|---|---|---|---|---|---|
| fp16 safetensors | 988 | 0.13 s | 158 s | 40 s | 8.0 s | 0 |
| standard GPTQ-int4 g128 file (rest fp16) | 458 | 0.6 s | 74 s | 19 s | 4.3 s | 0.148 |
| GPTQ-ECSQ c=0.35 + rANS, rest lossless | 339 | 2.0 s (17 s constriction) | 56 s | 16 s | 4.7 s | 0.117 |
| **P-frame + XOR rest, base already cached** | **193** | 2.2 s (1.5 s with GPU P-frame) | **33 s** | **9.9 s** | 3.7 s | 0.128 |
| full-rank Δ c=0.12 + XOR rest, fp16 base cached | 219 | 2.3 s | 37 s | 11 s | 4.0 s | **0.0105** |

INTERPRETATION: below ~1 Gbit/s the bytes dominate and the family/Δ streams win clearly; at LAN speeds the decode cost of entropy
coding becomes visible and a fast (parallel/GPU) decoder is required — with constriction's single-threaded path the
entropy-coded formats would *lose* to plain int4 files at ≥ 200 Mbit/s.

**Generic coder as a reference** (same symbols, each tensor as int8/int16, one zstd-19 / gzip-9 stream):

| symbol stream (side info excluded) | rANS symbol MB | zstd-19 MB | zstd vs rANS | gzip-9 vs rANS | zstd-19 decode (1 thread) |
|---|---|---|---|---|---|
| Qwen B_gptq4_ans | 161.1 | 165.6 | **+2.8 %** | +14.6 % | 319 ms |
| Qwen B_ecsq_c0.35 | 159.4 | 161.6 | **+1.4 %** | +15.5 % | 233 ms |
| Qwen T_ecsq_c0.35 | 159.4 | 161.7 | **+1.4 %** | — | 249 ms |
| Qwen T_pframe_c0.35 | 49.2 | 57.3 | **+16.3 %** | — | 242 ms |
| Qwen T_pframe_c0.5 | 43.3 | 50.6 | **+17.1 %** | — | 216 ms |
| Qwen T_delta_c0.06 | 114.3 | 121.2 | **+6.0 %** | +18.2 % | 324 ms |
| Qwen T_delta_c0.12 | 74.8 | 82.3 | **+10.0 %** | +22.2 % | 285 ms |
| Qwen T_delta_c0.25 | 37.1 | 45.4 | **+22.5 %** | — | 214 ms |
| SmolLM2-135M B_gptq4_ans | 49.6 | 50.1 | **+1.0 %** | +14.2 % | 62 ms |
| SmolLM2-135M B_ecsq_c0.35 | 47.4 | 47.8 | **+0.8 %** | +15.2 % | 64 ms |
| SmolLM2-135M T_ecsq_c0.35 | 47.4 | 47.8 | **+0.8 %** | +15.2 % | 64 ms |
| SmolLM2-135M T_pframe_c0.35 | 14.6 | 16.7 | **+14.9 %** | +28.2 % | 68 ms |
| SmolLM2-135M T_pframe_c0.5 | 12.7 | 14.9 | **+17.1 %** | +30.5 % | 61 ms |
| SmolLM2-135M T_delta_c0.06 | 34.1 | 36.2 | **+6.0 %** | +18.4 % | 93 ms |
| SmolLM2-135M T_delta_c0.12 | 22.3 | 24.5 | **+9.9 %** | +22.4 % | 80 ms |
| SmolLM2-135M T_delta_c0.25 | 11.2 | 13.4 | **+19.6 %** | +30.5 % | 60 ms |

INTERPRETATION:
* Entropy decoding is **not free** and the off-the-shelf Python path is slow: constriction's 40 Msym/s/thread makes a 0.5 B
  model a 7–15 s decode. A chunked format fixes it: own rANS on 8 cores ≈ 1 s, on the GPU ≈ 0.6 s (≈ 1 s for base + P-frame,
  0.4 s for a P-frame against a cached base) — i.e. the same order as an int4 unpack + dequant (0.55 s here). The rest of the
  GPU time is my per-tensor reconstruction (several small kernels per tensor), not the entropy decoder.
* For **I-frames at int4-class rates a generic zstd-19 is within 1–3 % of rANS and decodes ~5–40× faster** (Huffman, 4 streams).
  For the **low-entropy family streams (P-frames, Δ) zstd costs +6–23 %** — exactly where the family codec gets its gain — so the
  ANS coder earns its keep there; gzip-9 is +14–30 % everywhere.

### 14.7 Second and third family + the "scene cut" control

**SmolLM2-360M → SmolLM2-360M-Instruct** (‖Δ‖/‖W‖ = 0.076, small: PASS; GPTQ group 64). Planned as an HF Job; the
two long HF jobs were cancelled externally (§14.8) after the base stage, so the family was re-run **locally with the Qwen protocol**
(MPS fp16, all WikiText windows, 48-prompt chat set). The job's base stage (x86, fp32, 40 windows) is an off-Mac cross-check:
its four base files have **the same sizes as the local ones to 0.01 MB** (GPTQ-int4 g64 packed 169.58 / rANS 159.35, GPTQ-ECSQ
c=0.35 140.87, c=0.5 121.05 MB) and the same KL (GPTQ-int4: 0.0860 x86-fp32 vs 0.0856 MPS-fp16; PPL differs only by window count:
11.94 on 40 windows vs 12.51 on all).

| fine-tune format | MB | real bpw | chat-KL | wiki-KL | PPL | decode CPU constriction | CPU rans16 8 thr | GPU |
|---|---|---|---|---|---|---|---|---|
| GPTQ-int4 g64 fixed-rate packed | 169.6 | 4.313 | 0.1184 | 0.0974 | 13.90 | 0.42 s | 0.42 s | 0.56 s |
| GPTQ-int4 g64 + rANS | 159.4 | 4.053 | 0.1184 | 0.0974 | 13.90 | 5.91 s | 0.90 s | 0.59 s |
| GPTQ-ECSQ c=0.35 | 140.9 | 3.582 | 0.1375 | 0.1080 | 14.08 | 7.78 s | 0.90 s | 0.51 s |
| GPTQ-ECSQ c=0.5 | 121.0 | 3.078 | 0.3029 | 0.2343 | 15.82 | — | — | — |
| P-frame c=0.35 (+ base I-frame) | 41.5 | 1.056 | 0.1379 | 0.1071 | 14.04 | 13.15 s | 1.86 s | 0.93 s |
| P-frame c=0.5 | 36.3 | 0.922 | 0.3018 | 0.2287 | 15.77 | — | — | — |
| full-rank Δ c=0.06 (fp16 base) | 97.6 | 2.482 | 0.0035 | 0.0029 | 12.75 | — | — | — |
| full-rank Δ c=0.12 | 63.0 | 1.602 | 0.0134 | 0.0097 | 12.77 | 5.97 s | 1.05 s | 0.50 s |
| full-rank Δ c=0.25 | 30.6 | 0.777 | 0.0681 | 0.0274 | 12.63 | — | — | — |
| LoRA-style r=32 | 34.7 | 0.883 | 0.1874 | 0.0915 | 12.34 | 0.33 s | 0.31 s | 0.25 s |
| LoRA-style r=64 | 69.5 | 1.767 | 0.1448 | 0.0751 | 12.33 | — | — | — |
| LoRA-style r=128 | 138.9 | 3.533 | 0.0885 | 0.0541 | 12.37 | — | — | — |

base: fp16 PPL 11.47; GPTQ-int4 g64 169.6 MB packed / 159.4 MB rANS (PPL 12.51); I-frame GPTQ-ECSQ c=0.35 140.9 MB (PPL 12.66). Pair, every tensor: two GPTQ-int4 files (rest fp16) 528 MB; independent GPTQ-ECSQ + lossless rest 406 MB; **family I+P 295 MB** (Instruct chat-KL 0.118 / 0.138 / 0.138).

Full Δ vs LoRA, SmolLM2-360M: Δ c=0.25 (30.6 MB) vs r=32 (34.7 MB): **2.8×** lower chat-KL at 88 % of the bytes; Δ c=0.12 (63.0 MB) vs r=64 (69.5 MB): **10.8×** lower chat-KL at 91 % of the bytes; Δ c=0.06 (97.6 MB) vs r=128 (138.9 MB): **25.2×** lower chat-KL at 70 % of the bytes.


**Scene-cut control (rate only, real files, `results/phase2_scene_cut_qwen05b.json`)**: everything on the *base's* RTN lattice
(c = 0.35), I-frame = indices of the model, P-frame = indices − base indices (closed loop on a shared lattice ⇒ identical
reconstruction, asserted):

| relative of Qwen2.5-0.5B (base lattice, c=0.35) | ‖Δ‖/‖W‖ | I-frame MB (bpw) | P-frame MB (bpw) | P / I |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.079 | 158.1 (3.536) | 38.9 (0.869) | **0.246** |
| Qwen2.5-Coder-0.5B | 0.907 | 157.8 (3.527) | 152.0 (3.398) | **0.964** |

Base I-frame on its own lattice: 159.9 MB (3.5747 bpw). Phase-1 estimate (C4, α=1): Instruct P 0.911 vs I 3.575 (0.25), Coder P 3.452 vs I 3.575 (0.97).


### 14.8 Hugging Face: Jobs, dataset, Space (FACT)

* **Account:** the author's (`lifeart`); the token was never printed, logged or committed.
* **HF Jobs: available and used** — CPU only (by design). Code is shipped as a base64 tarball in an env var (same mechanism as
  `hf jobs uv run`), CPU torch is pip-installed, `phase2/test_wcodec.py` runs first (bit-exact round trips on x86 — passed in every
  job), results are pushed **incrementally** to the private dataset after every config, bitstreams at the end. The HF token reaches
  the job only through the jobs-secrets mechanism.

| run | outcome |
|---|---|
| pythia410m_traj | **cancelled by me** after the late window: x86 fp32 PPL evals took ~27 min each → would hit the 5 h timeout, and that bundle uploaded only at the end. Log kept: `results/phase2_job_pythia410m_traj_v1_log.txt` (its late-window bytes and c=0.35 PPL agree with v2 / the Mac check) |
| pythia410m_traj_v2 | **completed** (~1.6 h): encode + bytes + x86 decode timing, no PPL (PPL done on the Mac from the job's files, §14.4) |
| smol360_family | cancelled by me after 10 min (my bug: missing import, would have crashed at the P-frame) |
| smol360_family_v2 | cancelled by me after 45 min (≈15 min per eval on 8 vCPU → over the timeout; no incremental upload yet) |
| smol360_family_v3 | **CANCELED externally** after ≈3.8 h (not by the agent or the launcher; the API returns no reason). The whole base stage was saved by the incremental upload (`results/hfjob_partial/`) |
| smol360_delta | cancelled by me at start (my bug: empty CLI arg not quoted) |
| smol360_delta2 | **failed** (my bug: target-only stage never downloaded the base) |
| smol360_delta3 | **CANCELED externally** at the same time, after its 69-min Hessian pass (no config finished). I did not relaunch the jobs; SmolLM2-360M was completed **locally** (the planned fallback) |

  CPU fp32 evaluation of 0.4–0.5 B models is the bottleneck on Jobs (15–27 min per PPL/KL/chat
  eval vs ≈2 min on the local Apple-Silicon GPU); encoding / decoding takes little time there.
* **Private dataset** `lifeart/weight-codec-phase2` (private=True verified): `runs/<run>/results/*.json` and `runs/<run>/bitstreams/…`.
* **Space — private, static:** <https://huggingface.co/spaces/lifeart/weight-codec> (private=True verified; served pages return
  HTTP 200 with the author's credentials and **401 anonymously**). Creating a **Gradio** Space failed with **HTTP 402**
  (a Gradio Space requires paid Space hardware). So the Space is a
  static page: results tables + figures, and a **live demo that downloads the real WCDC/rans16 bitstreams of
  SmolLM2-135M → Instruct and decodes them in the visitor's browser** (plain JavaScript, one thread), checking SHA-256 of every
  decoded tensor against the Python decoder. Verified offline with Node 22 on the staged files: all 6 files ✓ bit-exact,
  ≈55 M symbols/s in V8 (the family chain of 130.6 MB decodes in ≈6.3 s on one JS thread). The Gradio app (rebuild the model
  from bitstreams + generate) is in `spaces/weight-codec-demo/`, tested locally end-to-end, ready for Space hardware that supports Gradio.

### 14.9 Surprises / deviations (phase 2)

1. **The entropy estimate was right to 0.02 %** — the whole phase-1 rate story survives real bytes unchanged; the only systematic
   difference is in the codec's favour (P-frames inherit steps: −1.1 %).
2. **Decode speed is the real cost of entropy coding**, and the off-the-shelf path is the slow one: constriction's Python binding
   holds the GIL (threads 1.0×) and decodes ≈ 40 Msym/s → 7–15 s for a 0.5 B model. Own chunked rANS: 8–10× faster on CPU, ≈ 15×
   on the GPU.
3. **zstd-19 is a strong baseline for I-frames** (+0.8–2.8 % bytes, ~1.5 GB/s decode) but **not for the family streams**
   (+6–23 % on P-frames / Δ). An honest "ANS vs generic coder" result: ANS matters most exactly where the video idea pays.
4. **The embedding dominates the full-file story**: Qwen's tied 136 M-param embedding coded losslessly (10.5 bit/param; 8.4 as an
   XOR P-frame) is 143–179 MB — 3× the Instruct P-frame of all 168 linears. The "bf16 → fp16" round trip leaves 3 zero mantissa bits,
   which the lossless coder finds (16 → 10.5 bit/param; Pythia, natively fp16: 13.5).
5. **SmolLM2-135M is fragile at int4-class rates** (chat-KL ≥ 0.40 for every ≈ 4-bit format) — the full-rank Δ over an fp16 base is the
   only format that keeps it usable at small size (22.6 MB, chat-KL 0.030); the Space demo shows this live.
6. CPU-fp32 evaluation on HF Jobs is **10–15× slower** than the local Apple-Silicon GPU (15–27 min per eval) — Jobs are great for encode/decode/streaming
   checkpoints, poor for many evals. Two long jobs were cancelled externally without a reason.
7. Bugs I introduced and caught in this phase (all before any number was reported): `spawn` pool re-running the benchmark in every
   worker (fixed with a `main()` guard, polluted timings deleted); an empty-dict reference treated as "no reference"; missing import;
   unquoted empty CLI argument in the job launcher; base not downloaded in a target-only job. The Qwen run's memory footprint
   (2.3 GB Hessians + MPS cache + multi-threaded zstd-19 reference coding; later the decoded int32 state kept on the GPU) was too large
   for the machine: twice I stopped the (resumable) run, cut the footprint and resumed.

### 14.10 Limitations (phase 2)

* Decode times are **warm page-cache** numbers on one machine (an Apple Silicon laptop, 8 CPU cores + MPS); download time is not measured, only
  modelled from MB (§14.6). The x86 job CPU decode timings (constriction 173 s, rans16 189 s for an 8-frame Pythia chain) were
  dominated by something other than the entropy decoder (not investigated; thread oversubscription torch × numba is my guess).
* GPU path = Metal only (no CUDA build); its reconstruction step is unfused (several small kernels per tensor).
* Embeddings / norms are coded **losslessly** in all full-file numbers; a lossy embedding (and its P-frame) was not evaluated.
* SmolLM2-360M's off-Mac base stage (x86 CPU fp32, 40 windows) and its local re-run (MPS fp16, all windows) use different precisions / windows; only the local run is used in the tables.
* The static Space cannot run the rebuilt model (no server-side Python on a static Space); generation from the decoded model was
  verified locally with the Gradio app (`spaces/weight-codec-demo/app.py`).
* Single seed, no confidence intervals (as phase 1).

### 14.11 What next

1. Lossy embedding coding + embedding P-frames (would cut the Instruct delivery from ≈ 193 MB to ≈ 70 MB for Qwen if the
   phase-1 rate estimate of 1.1 bpw holds at acceptable chat-KL — unmeasured).
2. Fused GPU decode (rANS → dequant → fp16 in one kernel; or ANS-in-GEMM, DFloat11-style) and a WASM/WebGPU decoder for the browser.
3. A hub container: one base I-frame + many fine-tune P-frames / Δ-streams with random access per tensor (the format already supports it).
4. zstd-for-I-frames / rANS-for-P-frames hybrid (best of both: decode speed where entropy is high, bytes where it is low).
5. Same protocol on a ≥ 1.5 B model and GPU Jobs for evaluation.

### 14.12 Environment, downloads, disk (phase 2)

* Installed: `constriction 0.5.0` into `.venv` only. Downloads (public HF): Qwen2.5-0.5B / -Instruct (again; deleted in phase 1),
  Qwen2.5-Coder-0.5B (scene-cut control only, deleted right after), SmolLM2-135M / -Instruct, SmolLM2-360M / -Instruct; Pythia
  checkpoints were streamed into RAM on the HF Job (14 × 0.9 GB) and on the Mac only for the drift check (2 checkpoints, RAM only);
  the job's bitstreams (≈ 1.3 GB) went to the private dataset, the Mac downloaded only the chains it decoded (deleted after use).
* All model caches downloaded in phase 2 were deleted at the end.
  `bitstreams/` keeps only the 8 SmolLM2-135M demo files (282 MB, git-ignored; everything else is reproducible and was deleted);
  `data/` (363 MB, git-ignored) keeps tokenized eval sets / reference hidden states.
* Nothing secret or heavy is in git: no token, no `.wcdc` / weights (`.gitignore`), job specs contain only run names / scripts / args.
