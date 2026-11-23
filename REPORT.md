# weight-codec — REPORT

*"Compress weights like JPEG (intra), families / layers / checkpoints like video (inter)."*
Falsification-first study on real ≤1 GB models. Phase 1: 2026-11-20; phase 2 (real bitstreams, §14): 2026-11-21.
Hardware: an Apple Silicon laptop, MPS (plus x86-64 CPU Hugging Face Jobs in phase 2). Author: Aleksandr Kanunnikov (lifeart).

The experiments were run and written up by a Claude Code agent under the author's direction; "I" / "my" in this
report refers to that agent. Process deviations, reruns and bugs are kept in the text on purpose (§12, §14.9).

Legend: **FACT** = measured here, number in `results/`. **HYPOTHESIS / INTERPRETATION** = my reading, not measured.

## 0. TL;DR

**In plain terms.** The compression comes not from a clever transform on a single model but from the fact that
*related models are almost identical* — a fine-tune differs from its base by ~7.7 % of its weights, consecutive
training checkpoints by less — so you keep one model in full and entropy-code only the small (full-rank) **difference**
for the rest. A single standalone weight matrix has no structure a frequency transform can exploit (the JPEG transform
fails); there, only quantization + entropy coding help.

> **Phase 2 (2026-11-21): real ANS bitstreams, decode benchmark, Hugging Face runs → §14.** Real file bytes match the phase-1
> entropy estimates to ±0.02 %; all phase-1 winners hold on real bytes.
>
> **Phase 3 (2026-11-22): "sort / cluster the weights as vectors" (vector quantization, k-means codebooks, lattices,
> GPTVQ, VQ of the fine-tune delta, one codebook per family) → §15.** Clustering does not beat entropy-coded scalar
> quantization at equal total bits; every large VQ gain disappears against a scalar coder given the same sensitivity
> information; only the D4 lattice adds a small, repeatable ≈ 0.05–0.1 bpw.
>
> **ghcol (2026-11-22): GPTQ on a per-input-column (Hessian-scaled) step grid, confirmed on 3 models and integrated into
> the codec (opt-in) → §16.** It beats every coder the codec had (SmolLM2-135M +0.73…+0.97, SmolLM2-360M +0.28…+0.48,
> Qwen +0.09…+0.18 bpw at equal KL), but the same grid with A3's Fisher factors (§3) is better still (+0.17…+0.31);
> on the fine-tune delta the column grid saves 0.2–0.9 bpw at equal chat-KL. Real files bit-exact on all decoders.
>
> **Default coders (2026-11-22) → §17.** The Fisher-grid I-frame, a plain-rounding P-frame on its lattice (won a
> pre-registered rule on all 3 pairs) and the rounding Hessian-column Δ are now the codec's default; re-measured on real
> files: at the same pair size the fine-tune's chat-KL is **2.2× / 2.9× / 6.4× lower** (Qwen / SmolLM2-360M / -135M), the
> Δ file 19–35 % smaller at equal chat-KL; bit-exact on all 5 decoder paths; the browser demo was rebuilt with them
> ("smallest" config chat-KL 0.54 → 0.084). Earlier sections keep their (phase-2-coder) numbers unchanged.


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
| F. "sort / cluster weights as vectors" (VQ, §15) | **falsified as a compression gain** | k-means VQ needs **+0.09…0.30 bpw** vs scalar at equal KL (codebook charged, ≤ 0.06 bpw of it); one codebook per family loses to the scalar P-frame; GPTVQ / weighted VQ gains (−45 % KL) are sensitivity weighting — **scalar + per-column steps beats them** (SmolLM2 KL 0.080 vs 0.116 @ 3.7 bpw; Qwen: A3 0.027 vs GPTVQ 0.044); D4 lattice: +0.04…0.11 bpw on 2 models |

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
inter-model coding — the full-rank Δ over an fp16 base has no such problem (22.6 MB → 0.030). [Confirmed in §16.5: with a
column-step I-frame the same-size P-frame pair reaches chat-KL 0.105 instead of 0.538.]

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

## 15. Vector quantization / clustering — "sort the weights as vectors and compress the clusters" (2026-11-21/22)

**Hypothesis (owner, 2026-11-21).** Group neighbouring weights into d-dimensional vectors, cluster them (a shared
k-means codebook — the AQLM / QuIP# / GPTVQ family) and store each vector as a cluster index; scalar quantization
is the d = 1 special case. Does d ≥ 2 buy downstream quality at equal *total* size, and does clustering compose with
the family / delta coding of §5 and §14?

### 15.0 TL;DR (verdicts; numbers in §15.3–15.8)

| probe (pre-registered rule in each script docstring) | verdict | headline (FACT, SmolLM2-135M unless noted) |
|---|---|---|
| V1 plain k-means / entropy-constrained VQ, d = 2, 4 vs scalar ECSQ | **FAIL** | VQ needs **0.09–0.30 bpw *more*** than scalar for equal KL; codebook only 0.001–0.06 bpw of it |
| V1 lattice VQ (codebook-free) | D4 INCONCLUSIVE (small, replicated), E8 model-dependent | D4: −2…−21 % KL at equal rate on SmolLM2, −14…−17 % on Qwen (+0.04…+0.11 bpw), still −3…−13 % on top of the best scalar coder; E8: SmolLM2 +10…+35 % KL (despite −14 % weight-MSE), Qwen −26 % (one point) |
| V2 Hessian-aware VQ (GPTVQ, diag(H)-weighted ECVQ) vs Hessian-aware scalar (GPTQ) | PASS by its rule (+0.53…+0.65 bpw on SmolLM2; ≈ +0.09 on Qwen) | **but a sensitivity-weighted *scalar* quantizer beats it**: SmolLM2 ≈ 30 % lower KL; Qwen2.5-0.5B: the existing A3 coder (Fisher steps + GPTQ) 38 % lower KL in-domain, 30 % on C4 — the win is per-channel sensitivity, not vectors |
| V3a VQ of the fine-tune Δ (fp16 base given) vs scalar Δ | PASS vs naive scalar on SmolLM2 (+0.07…+0.24 bpw); **not replicated on Qwen** (D4 +0.06…+0.08, k-means FAIL) | **no gain vs the Hessian-aware scalar Δ coder** on either model (chat-KL wash, WikiText KL clearly worse) |
| V3b one codebook for the whole family ("cluster once, reuse") | **FAIL** | instruct-given-base costs 1.02–2.19 bpw vs 0.70–1.11 for the scalar P-frame; sharing saves ≤ 0.06 bpw |
| V4 literal "sort" | **FAIL by construction** | a row permutation costs 7.7–9.1 bpw; E[Δ \| W_base] explains 0.04 % of Δ |

**Synthesis — FACT vs HYPOTHESIS.** FACT (SmolLM2-135M for every probe, Qwen2.5-0.5B for the decisive ones; one
seed plus seed reruns; codebooks, escape streams, entropy tables and all side info charged): clustering weights into
vectors does **not** compress better than entropy-coded scalar quantization at equal total bits — not plain k-means,
not entropy-constrained k-means, not a shared family codebook; what survives is the codebook-free lattice's
space-filling gain (D4 ≈ +0.04–0.11 bpw on both models; E8 helps on Qwen but hurts on SmolLM2). Every *large* VQ win
we found (GPTVQ d = 4: −45…−50 % KL on SmolLM2; VQ of the fine-tune Δ: −20…−30 % chat-KL on SmolLM2, much less on
Qwen) **disappears against a scalar coder that is given the same side information** (per-input-channel sensitivity,
Hessian-aware error feedback) — on both models, in-domain and on C4. HYPOTHESIS: the literature's VQ gains at 2 bpw
(AQLM, QuIP#) come mostly from fixed-rate operation (no entropy coder to capture the non-uniform density), from
fine-tuning, and from Hessian-aware assignment — not from vector clustering per se; in an entropy-coded codec the
right place for "multi-dimensional" thinking is the error metric (eᵀHe), not the codebook.

### 15.1 Setup and rate accounting (FACT)

SmolLM2-135M base (V1/V2) and → -Instruct (V3); quantized set = all 210 block linears (106.2 M weights); WikiText-2
test PPL (148 windows) + KL / top-1 vs fp16 (first 16 windows); chat-KL on the instruct model's own 48 greedy answers
for V3. Vector = d consecutive input-dim weights of one row; rows normalised by their fp16 RMS (16 b/row — the same
side info as the ECSQ_row baseline). Codebooks: one per weight type shared by all 30 layers (7 codebooks), fitted on
≤ 6 M normalised vectors; k-means (`km`), entropy-constrained k-means (`ecvq`, Lagrangian assignment
‖x−c‖² + λ·(−log₂p(c)), Chou–Lookabaugh–Gray) and the same codebook with plain nearest-centroid encoding (`ecvqnn`).
Lattices D4 / E8 by Conway–Sloane nearest point (checked against brute force), coded per coordinate with the parity
bit implicit (+1 coset bit for E8). One seed (0) unless "seed 1" is stated.

**Everything is charged:** index stream (empirical order-0 entropy + a 32-bit-per-symbol frequency table, either per
tensor = `common.coded_bits` or one pooled table per codebook / weight type, whichever is smaller — *all 75 V1/V2
configs, scalar and VQ, chose the pooled table*; with per-tensor tables a d = 4 alphabet of ≤ 16 k symbols would cost
another 0.26–0.61 bpw) + escape
stream (§15.2) + **codebook 16·d·K_used bits per scope** + 16 b/row (+16 b/column where stated). `quantize_model`'s
own total is asserted equal to the toolkit's per-tensor variant for every config.

Scripts: `scripts/vq.py` (toolkit), `probe_vq.py` (V1), `probe_vq_hess.py` (V2), `vq_family.py` (V3),
`vq_summary.py`, `vq_v3_summary.py`. Envelopes are lower convex hulls of (bpw, log KL) per family (two operating
points can always be mixed across tensors); savings are read off at equal KL.

### 15.2 Pitfall: a finite codebook clips the outliers (FACT, `naive_*` in `results/vq_v1_smol135.json`)

The first V1 run gave d = 1 ECVQ at **4.42 bpw → KL 0.248** while uniform ECSQ at 4.39 bpw has **KL 0.089** — at the
*same* weight NMSE (0.0035 vs 0.0033). LLM weights reach 28 × row RMS (`mlp.down_proj`), the codebook spanned
[−8.4, 7.1]; on the top 0.01 % |w| the squared error was **2.9 vs 0.003** (≈ 900×), 6–9 % of all squared error —
invisible in weight-MSE, decisive downstream. Fix used for every VQ number below: a per-vector RD mode decision
between the codebook index and an ESCAPE symbol + d integers on the unbounded uniform lattice of step √(6λ/ln 2)
(the ECSQ step with the same RD slope), entropy-coded and charged (0.01–0.3 % of vectors at d ≤ 2; ≈ 55 % at d = 4,
where a K ≤ 16 k codebook covers only the central mass).

### 15.3 V1 — VQ vs scalar at equal total rate (FACT, `results/vq_v1_smol135.json`, `results/vq_summary_smol135.json`, `figures/vq_v1_smol135.png`)

| family (all costs charged) | bpw saved vs best scalar at equal KL | verdict |
|---|---|---|
| k-means / ECVQ d = 2 | **−0.21 … −0.09** | FAIL |
| k-means / ECVQ d = 4 (codebook 0.02–0.06 bpw) | **−0.30 … −0.13** | FAIL |
| E8 lattice | −0.18 … +0.01 | FAIL |
| D4 lattice | +0.04 … +0.11 | INCONCLUSIVE |

| step c | bpw | scalar ECSQ KL | D4 lattice KL | E8 lattice KL | E8 weight-NMSE vs scalar |
|---|---|---|---|---|---|
| 0.20 | 4.39 | 0.0886 | 0.0819 (−8 %) | 0.0865 (−2 %) | −12 % |
| 0.25 | 4.07 | 0.148 | 0.146 (−2 %) | 0.148 (0 %) | −13 % |
| 0.35 | 3.59 | 0.255 | 0.236 (−7 %) | 0.280 (+10 %) | −14 % |
| 0.50 | 3.08 | 0.523 | 0.489 (−7 %) | 0.667 (+27 %) | −14 % |
| 0.70 | 2.62 | 1.246 | 0.981 (−21 %) | 1.688 (+35 %) | −14 % |

| k-means / ECVQ config | bpw [codebook] | KL | scalar ECSQ at that bpw |
|---|---|---|---|
| ECVQ d=1, λ=0.0072, RDO assignment / nearest-centroid | 4.415 / 4.428 | 0.101 / 0.0905 | ≈ 0.088 |
| ECVQ d=2, nearest-centroid | 4.333 [0.002] | 0.108 | ≈ 0.097 |
| ECVQ d=4, λ=0.0072 | 4.173 [0.063] | 0.145 | ≈ 0.126 |
| ECVQ d=2, λ=0.0141, seed 0 / seed 1 | 3.802 / 3.798 [0.001] | 0.255 / 0.227 | ≈ 0.200 |
| ECVQ d=4, λ=0.0141 | 3.774 [0.053] | 0.236 | ≈ 0.207 |
| ECVQ d=1, nearest-centroid, λ=0.0289 | 3.397 | 0.467 | ≈ 0.335 |
| ECVQ d=4, λ=0.0566 | 2.682 [0.022] | 1.97 | ≈ 1.09 |

What the controls established (FACT):
1. *The Lagrangian (RDO) assignment hurts downstream* although it is weight-MSE-optimal: 13–50 % KL vs plain
   rounding at equal weight-NMSE. Nearest-centroid encoding removes it at 4.4 bpw, not at 3.4 bpw (0.467 vs ≈ 0.335):
   the entropy-constrained *codebook shape* (wide zero cell, conditional-mean levels) is itself downstream-worse.
2. *Centroid shrinkage is real but not the cause:* β = ⟨w,ŵ⟩/‖w‖² = 0.987 for a K = 32 codebook vs 0.9997 for
   rounding; a zero-bit per-row gain folded into the fp16 row scale changes KL by 3–7 % (noise), ECSQ by < 1 %.
3. *Codebook fits are noisy downstream:* seed 1 moves KL by −11 % at equal rate (d = 2) and +26 % at −0.08 bpw
   (d = 1); every VQ deficit has the same sign at all 4 rates.
4. *Codebook cost is not what kills k-means VQ:* 0.001–0.002 bpw (d = 2) and 0.02–0.06 bpw (d = 4) — d = 4 loses
   3–10× more than its codebook costs (§15.8).
5. *Space-filling gain is Euclidean; the downstream loss is not.* With the real input Hessians, plain rounding to
   D4 / E8 at the same step changes the layer-output error tr(E H Eᵀ) by **−5…−6 % / +2…+8 %** while the weight error
   falls 8 % / 14 % (`results/vq_v2_smol135.json` → `_diag_output_error`): E8's errors are correlated across the 8
   coordinates of a block (the half-integer coset moves all 8 by ±s/2) and land on the strongly non-isotropic H.
   (SmolLM2-specific: on Qwen2.5-0.5B E8 *lowers* KL by 26 % at 3.08 bpw, §15.9.)

### 15.4 V2 — Hessian-aware VQ, and the control that decides it (FACT, `results/vq_v2_smol135.json`, `results/vq_summary_v2_smol135.json`, `figures/vq_v2_smol135.png`)

H = E[xxᵀ] of every block-linear input (128 × 2048 WikiText-2 train windows). Implementation checks: GPTVQ at d = 1
reproduces GPTQ to 2·10⁻⁷; the joint-block update (W_R −= (e·inv(U_SS))·U_SR with U = chol(H⁻¹)) equals explicit
OBS to 10⁻¹⁶; the decoder rebuilds W bit-exactly from indices + escape integers + codebook + fp16 row scales.

| coder | bpw | WikiText KL | note |
|---|---|---|---|
| scalar ECSQ c = 0.35 / 0.5 | 3.589 / 3.082 | 0.255 / 0.523 | no Hessian |
| scalar GPTQ-ECSQ c = 0.35 / 0.5 | 3.593 / 3.089 | **0.283 / 0.610** | **worse than rounding on this model** (= phase 2's real-bytes 0.2833) |
| GPTQ on D4 / E8 lattice (Euclidean-nearest) c = 0.35 | 3.596 / 3.599 | 0.309 / 0.302 | FAIL vs scalar GPTQ |
| GPTQ + joint Hessian-metric rounding search, Z⁴ grid c = 0.35 / 0.5 | 3.593 / 3.089 | 0.304 / 0.662 | no gain: sequential GPTQ already optimal |
| GPTQ + joint Hessian-metric search, **D4** c = 0.35 / 0.5 | 3.596 / 3.095 | 0.231 / 0.439 | −19 / −28 % vs scalar GPTQ, no codebook |
| **GPTVQ d = 4** (ECVQ codebook, Mahalanobis-nearest, joint OBS), seed 0 / 1 | 3.775 | **0.116 / 0.126** | |
| GPTVQ d = 4, λ = 0.0289 | 3.355 | 0.195 | ECSQ ≈ 0.355 there |
| diag(H)-weighted ECVQ d = 4, **no error feedback** | 3.658 | 0.118 | the weighting alone does it |
| GPTVQ d = 1 / diag(H)-weighted d = 1 | 3.977 / 5.173 | 0.176 / 0.051 | d = 1 weighting cannot change the nearest centroid |
| **scalar ECSQ + per-column steps from diag(H)** (γ = 1, 16 b/col), c = 0.35 / 0.5 / 0.25 | **3.700 / 3.193 / 4.182** | **0.080 / 0.176 / 0.039** | **best coder in this study** |
| same, γ = 0.5 | 3.631 / 3.124 / 4.113 | 0.096 / 0.206 / 0.047 | |
| **D4 lattice in the same column-weighted coordinates** (γ = 1), c = 0.35 / 0.5 / 0.25 | 3.703 / 3.199 / 4.184 | **0.077 / 0.165 / 0.036** | −3 / −6 / −7 % vs the scalar line: VQ on top of the best scalar ≈ +0.03–0.05 bpw |

By its pre-registered rule V2 PASSES (Hessian-aware d = 4 VQ saves 0.53–0.65 bpw vs Hessian-aware scalar GPTQ). But
the control decides the interpretation: **a scalar quantizer with per-input-column step sizes from diag(H) — the A3
idea of §3 with H instead of Fisher — reaches KL 0.080 at 3.70 bpw, ≈ 30 % below GPTVQ / weighted VQ at equal rate
and 69 % below plain ECSQ.** The VQ "win" was the per-channel sensitivity (SmolLM2 has a few input channels with very
large E[x²]); a vector quantizer can only exploit it indirectly by steering error between the d coordinates of a
block, while the scalar coder exploits it directly for 16 bits per column. On this model the Hessian-aware lever is
*column steps*, not GPTQ's error feedback (which hurts here) and not vector clustering. What vectors add *on top of*
the best scalar coder is the D4 lattice's space-filling gain, −3…−7 % KL ≈ +0.03–0.05 bpw (consistent sign at all
three rates, below the 0.10-bpw bar). [Qwen2.5-0.5B: §15.9. GPTQ *on this column grid* ("ghcol") does pay — §16.]

### 15.5 V3a — VQ of the fine-tune delta, fp16 base given (FACT, `results/vq_v3_smol135.json`, `results/vq_v3_summary_smol135.json`, `figures/vq_v3a_smol135.png`)

Δ = W_inst − W_base, rate = Δ stream only, quality = chat-KL (+ WikiText KL) vs fp16 instruct. Δ rows normalised by
their own RMS ("nD"; for scalar Δ this beats normalising by the weight RMS, the phase-1 convention, by ≈ 0.1–0.15 bpw).

| Δ coder | Δ bpw | chat-KL | WikiText KL |
|---|---|---|---|
| scalar ECSQ, nD, c = 0.8 / 1.6 / 3.2 | 2.425 / 1.518 / 0.630 | 0.0096 / 0.0349 / 0.147 | 0.0099 / 0.0339 / 0.078 |
| D4 lattice, same steps | 2.442 / 1.575 / 0.743 | 0.0069 / 0.0252 / 0.110 | 0.0077 / 0.0252 / 0.072 |
| k-means d = 2 (nearest + escape) | 2.526 / 1.617 / 0.396 | 0.0068 / 0.0214 / 0.240 | 0.0077 / 0.0223 / 0.096 |
| k-means d = 2, c = 1.6, seed 1 | 1.608 | 0.0255 | 0.0264 |
| k-means d = 4 | 2.548 / 1.582 / 0.422 | 0.0063 / 0.0248 / 0.220 | 0.0066 / 0.0259 / 0.093 |
| **scalar + GPTQ (instruct Hessian), nD**, c = 0.8 / 1.6 | 2.441 / 1.570 | 0.0081 / 0.0248 | **0.0057 / 0.0163** |
| scalar + GPTQ, nW = phase-1/2 setting, c = 0.06 / 0.12 / 0.25 | 2.596 / 1.706 / 0.870 | 0.0101 / 0.0300 / 0.102 | 0.0069 / 0.0182 / 0.043 |
| phase 2 `T_delta` (same coder, **real bytes**) | 2.596 / 1.707 / 0.870 | 0.0101 / 0.0299 / 0.102 | 0.0069 / 0.0181 / 0.043 |
| GPTQ on the D4 lattice, nD, c = 0.8 / 1.6 | 2.456 / 1.619 | 0.0080 / 0.0260 | 0.0053 / 0.0164 |
| instruct := base (0 bits) | 0 | 0.557 | 0.151 |

* vs **naive** scalar Δ: D4 +0.07…+0.20 bpw, k-means d = 2 −0.01…+0.24 (both seeds −17…−30 % chat-KL at 1.6 bpw),
  d = 4 −0.00…+0.18 → PASS by the rule; the edge vanishes below ≈ 0.7 bpw. The harness reproduces phase 2's real-bytes
  `T_delta` to 4 digits.
* vs the **Hessian-aware** scalar Δ (the project's best Δ coder): VQ-Δ without H is a wash on chat-KL (−0.01…+0.10 bpw)
  and **loses 0.18–0.33 bpw on WikiText KL**; adding the D4 lattice to GPTQ gains nothing (−0.08…−0.02 bpw).
* Row-normalised W and Δ have the same per-pair statistics (adjacent correlation −0.0002 / +0.0015, squared-magnitude
  correlation 0.0049 / 0.0045, kurtosis 3.51 / 3.47; `results/vq_v3_delta_structure_smol135.json`), so the edge over
  naive scalar is not intra-vector memory. HYPOTHESIS: at 0.4–2.5 bits/sample (most scalar Δ symbols are 0) a 2–4-D
  cell structure keeps joint small excursions that the scalar dead zone discards — and Hessian-aware feedback recovers
  the same information.
[Qwen2.5-0.5B → -Instruct: §15.9.]

### 15.6 V3b — "cluster once, reuse the clustering": one codebook for the whole family (FACT, `results/vq_v3_smol135.json`)

No fp16 base anywhere: base I-frame + instruct coded given the decoded base. Scalar = phase-1 `pframe_rtn` (instruct
rounded on the base's grid, stream q_I − q_B). VQ: one ECVQ codebook per type fitted on the **base**, both models
assigned to it (base row scales reused), instruct stream = rank of its centroid among the 64 nearest neighbours of
the base centroid (+ escape / raw index). Lattice: same D4 / E8 lattice, stream = lattice-point difference.

| pair coder | base bpw | instruct \| base | instruct alone | **pair** | codebook | base KL | instr. KL | chat-KL |
|---|---|---|---|---|---|---|---|---|
| scalar grid c = 0.35 | 3.589 | **0.888** | 3.571 | **4.478** | 0 | 0.255 | 0.264 | 0.228 |
| D4 lattice c = 0.35 | 3.593 | 0.987 | 3.574 | 4.580 | 0 | 0.236 | 0.241 | 0.222 |
| E8 lattice c = 0.35 | 3.597 | 1.150 | 3.578 | 4.746 | 0 | 0.280 | 0.299 | 0.278 |
| k-means index P-frame d = 2 | 3.830 | 1.020 | 3.810 | 4.850 | 0.001 | 0.230 | 0.244 | 0.219 |
| k-means index P-frame d = 4 | 3.781 | **1.909** | 3.711 | **5.689** | 0.053 | 0.245 | 0.250 | 0.215 |
| scalar grid c = 0.25 / D4 / k-means d = 4 | 4.071 / 4.073 / 4.174 | 1.108 / 1.185 / 2.193 | — | 5.179 / 5.258 / 6.367 | 0 / 0 / 0.063 | 0.148 / 0.146 / 0.146 | 0.148 / 0.141 / 0.148 | 0.126 / 0.127 / 0.130 |
| scalar grid c = 0.5 / D4 / k-means d = 4 | 3.082 / 3.089 / 3.354 | 0.695 / 0.805 / 1.587 | — | 3.777 / 3.894 / 4.940 | 0 / 0 / 0.036 | 0.523 / 0.489 / 0.461 | 0.525 / 0.469 / 0.448 | 0.485 / 0.453 / 0.416 |

**FAIL.** Coding the fine-tune as indices into the base's codebook costs 1.02 (d = 2) – 2.19 (d = 4) bpw vs
0.70–1.11 for the scalar difference stream at about the same quality, although 62–69 % of instruct vectors keep the
base's index: every vector that moves costs a whole neighbour-rank symbol, while the scalar stream pays ≈ 1–2 bits
only for the individual weights that cross a cell border. Sharing the codebook saves only the codebook (0.001–0.06
bpw per extra model): "store the clustering once" amortizes the cheap part — the index streams are the cost. D4 I+P
≈ break-even (+0.08–0.12 bpw for 1–11 % lower KLs), E8 loses. **Against "store base in full + entropy-coded Δ"**
(§15.5, §14.3): that design delivers the instruct for 1.5–2.5 bpw at chat-KL 0.01–0.035 *with an exact base*; the
shared-codebook family pays ≥ 4.85 bpw for both models at chat-KL ≈ 0.22 / base KL ≈ 0.23 — not the same quality
regime, and inside the lossy regime it loses to the plain scalar I+P-frame.

### 15.7 V4 — the literal "sort" reading (FACT, `results/vq_v4_sort_smol135.json`)

Sorting a row and coding the smooth sorted sequence requires the permutation: log₂(n!)/n = **7.74 bits/weight for a
576-wide row, 9.15 for 1536** — more than twice the whole 3.5-bpw budget, and the decoder cannot recompute the sort key
without the original weights. The only free variant sorts by a key the decoder already has — for a fine-tune, the
base weights — but E[Δ | W_base] (64 quantile bins per tensor) explains **0.04 %** of the Δ energy and a per-row
linear fit α·W_base 0.23 %. A family-shared sort order therefore buys nothing either. No codec built.

### 15.8 Codebook-cost accounting (FACT / arithmetic, 106.17 M weights)

| codebook scope | d, K | codebook bits | bpw |
|---|---|---|---|
| one per weight type (7), as used | d = 2, K ≤ 4096 (K_used 289–740) | 16·d·K_used·7 | **0.0004–0.0016** (measured) |
| one per weight type (7), as used | d = 4, K ≤ 16384 | 16·4·K_used·7 | **0.022–0.063** (measured) |
| one per tensor (210) | d = 4, K = 4096 | 210·4096·4·16 | **0.52** (arithmetic) |
| one per weight type (7) | d = 8, K = 65536 | 7·65536·8·16 | **0.55** (arithmetic) |
| one global | d = 8, K = 65536 | 65536·8·16 | 0.079 (arithmetic) |
| entropy model per tensor instead of per codebook | d = 4, K ≤ 16 k | 32 b × used symbols × 210 | +0.26–0.61 (measured) |

At the 3–4.4 bpw where this model is usable, the index stream is ≈ 97–99 % of the VQ rate when codebook *and* its
entropy model are shared per weight type; per-tensor codebooks or per-tensor frequency tables would add ≈ 0.5 bpw —
more than any gain measured anywhere in §15. Unstructured k-means at d = 8 is capped at log₂K/8 = 2 bpw for K = 65 536
(and 4.4 M distance evaluations per vector of encoding work); it was not run downstream (dropped for time — the
d = 2 / 4 results and the lattice results already decide V1).

### 15.9 Second model: Qwen2.5-0.5B (FACT, `results/vq_v2_qwen05b.json`, `results/vq_v1_qwen05b.json`, `results/vq_v3_qwen05b.json`)

The decisive V2 comparison re-run on the model where scalar GPTQ *does* work (spine §1: GPTQ-ECSQ halves KL), with
an out-of-domain check: Hessians from WikiText-2 train, KL on WikiText-2 test and on C4 validation (64 × 2048,
KL on the first 16 windows; protocol of §8). Scalar baselines from `results/spine_qwen05b.json`,
`results/a3_codec_qwen05b.json`, `results/crossdomain_c4_qwen05b.json` (same quantizers, same accounting family).

| Qwen2.5-0.5B coder (all side info charged) | bpw | WikiText KL | C4 KL |
|---|---|---|---|
| scalar ECSQ c = 0.35 (spine) | 3.575 | 0.153 | 0.161 |
| scalar GPTQ-ECSQ c = 0.35 (spine) | 3.578 | 0.060 | 0.089 |
| scalar + diag(H) column steps, γ = 1, c = 0.35 / 0.5 | 3.684 / 3.176 | 0.063 / 0.135 | 0.068 / 0.142 |
| **D4 lattice in the same column-weighted coordinates**, c = 0.35 / 0.5 | 3.687 / 3.183 | **0.059 / 0.117** (−7 / −13 %) | **0.063 / 0.125** (−7 / −12 %) |
| **GPTVQ d = 4** (ECVQ codebook, Mahalanobis-nearest, joint OBS) | 3.707 | **0.044** | 0.067 |
| A3 Fisher-weighted scalar, no GPTQ, γ = 1 / 0.5 (§3) | 3.792 / 3.644 | 0.041 / 0.047 | — |
| **A3 Fisher-weighted scalar + GPTQ** (§3, the project's best intra coder) | **3.647** | **0.027** | **0.047** |
| diag(H)-weighted ECVQ d = 4 (no feedback) | 3.604 | 0.091 | 0.095 |
| scalar + diag(H) column steps, γ = 0.5, c = 0.35 (same rate) | 3.608 | **0.073** | **0.076** |

FACT (Qwen): (i) GPTVQ d = 4 beats plain scalar GPTQ by ≈ 12 % KL in-domain and ≈ 11 % on C4 at equal rate
(≈ 0.09 bpw) — far less than on SmolLM2, where scalar GPTQ was broken; (ii) against the diag(H) column-step scalar
it wins by 31 % in-domain but **only 2 % out-of-domain** (0.0665 vs 0.0678) — its in-domain edge comes with the full
WikiText-calibrated Hessian; (iii) **the project's existing sensitivity-weighted scalar coder (A3 Fisher + GPTQ)
beats GPTVQ by 38 % (WikiText) and 30 % (C4) at a lower rate**; (iv) the D4 lattice on top of the column-weighted
scalar is again a small, consistent gain (−7…−13 % KL in both domains ≈ +0.05–0.09 bpw) — on both models the only
vector effect that survives every control; (v) the diag(H)-weighted ECVQ that matched GPTVQ on SmolLM2 is **23 %
worse than the column-step scalar at the same rate** on Qwen (0.091 vs 0.073; C4 0.095 vs 0.076).

**Plain lattice VQ on Qwen (V1 confirmation, `results/vq_v1_qwen05b.json`; ECSQ re-run reproduces the spine to 4
digits):**

| step c | bpw | scalar ECSQ KL | D4 lattice KL | E8 lattice KL |
|---|---|---|---|---|
| 0.35 | 3.575 / 3.578 | 0.153 | **0.131 (−14 %)** | — |
| 0.50 | 3.068 / 3.074 / 3.082 | 0.372 | **0.311 (−17 %)** | **0.277 (−26 %)** |

D4 replicates (and is larger on Qwen: ≈ +0.09–0.11 bpw). **E8 does not replicate the SmolLM2 reversal: on Qwen it is
the best plain lattice (−26 % at 3.08 bpw, one point).** The E8 penalty of §15.3 (output error +2…+8 % despite
−14 % weight error) is therefore a property of SmolLM2's input covariances, not of E8 — the general lesson stands
(the Euclidean space-filling gain does not transfer predictably to eᵀHe), the sign is model-dependent.

**VQ of the fine-tune Δ on Qwen2.5-0.5B → -Instruct (V3a confirmation, `results/vq_v3_qwen05b.json`; "instruct :=
base" reproduces §5's 0.277 chat-KL).** Phase-1 Hessian-aware scalar Δ (`results/c2_qwen05b_instruct.json`, nW)
for reference:

| Δ coder (Qwen) | Δ bpw | chat-KL | WikiText KL |
|---|---|---|---|
| scalar ECSQ, nD, c = 1.6 / 0.8 | 1.516 / 2.418 | 0.0119 / 0.0036 | 0.0255 / 0.0055 |
| D4 lattice, nD, same steps | 1.568 / 2.434 | **0.0103 / 0.0031** | **0.0167 / 0.0044** |
| k-means d = 2, nD, λ(c = 1.6 / 0.8) | 1.619 / 2.527 | 0.0114 / 0.0029 | 0.0156 / 0.0041 |
| scalar + GPTQ Δ, nW (phase 1 `delta_gptq_c0.12 / c0.06`) | 1.687 / 2.569 | 0.0105 / 0.0031 | **0.0058 / 0.0018** |

By the pre-registered V3a rule (convex hulls, chat-KL; `results/vq_v3_summary_qwen05b.json`) the SmolLM2 PASS does
**not** replicate: D4 Δ saves **+0.06…+0.08 bpw** vs the naive scalar Δ (INCONCLUSIVE, below the 0.10 bar; it is
−13…−14 % chat-KL and −20…−35 % WikiText KL at +0.02–0.05 bpw), k-means d = 2 Δ **−0.07…+0.00 bpw (FAIL)**. Both match
the Hessian-aware scalar Δ on chat-KL (0.0029–0.0031 at 2.4–2.6 bpw) but are 2.3–3× worse on WikiText KL than
GPTQ-Δ — VQ does not displace the project's Δ coder on either model.

### 15.10 Process notes and deviations (for honesty)

* **Pre-registration held; interpretations moved.** Every probe's rule is in its script docstring, written before its
  results. The V2 rule ("Hessian-aware VQ vs Hessian-aware scalar") PASSED, but the post-hoc control (scalar +
  diag(H) column steps) shows the win is not a vector effect — reported as such. Envelopes were changed from a Pareto
  front to the convex hull after the first lattice summary inflated the D4 saving (+0.26 → +0.11 bpw); all verdicts
  quoted use the hull.
* **First V1 run was flawed** (outlier clipping, §15.2) — its two points are kept as `naive_*`, everything else re-run.
* **Plans were trimmed for time** (compute was shared with other jobs): k-means sweeps cut
  to one K per d; d = 8 k-means, per-tensor / global codebook controls and the diag(H)-weighted GPTVQ variants were
  not run downstream (codebook costs in §15.8 are arithmetic); V3a k-means Δ moved to nD normalisation only.
* **Incidents.** (1) Two MPS processes at once → Metal "command buffer exited with error status" in one of them and a
  crash with an empty codebook; from then on strictly one GPU job at a time (queue in `logs/`, not committed).
  (2) Two "hangs" (0 % CPU in `MTLCommandBuffer waitUntilCompleted`) were `index_add_` on MPS with heavy atomic
  contention once the Δ codebooks collapsed to ≈ 10 live centroids (6–8 s per call); the centroid update now runs on
  CPU (numerically equivalent), which also cut V1-style fits from 100–460 s to seconds. (3) I once killed processes
  by command-line pattern instead of PID; only my own diagnostic matched (verified) — PID-only kills afterwards.
* **Qwen Hessian precision.** Storing Qwen's 2.5 GB of Hessians in fp16 (H / max|H|) and then in bf16 made several of
  them indefinite (massive activation channels: rounding errors of the large off-diagonal entries exceed small channels'
  diagonals) → Cholesky failure, then 100 % damping, which would silently turn GPTVQ's metric Euclidean. Both runs were
  discarded; the final Qwen run stores U = chol(H⁻¹) computed in fp64 from the fp32 H (bf16 storage, no PD requirement)
  plus diag(H) in fp32 — checked to reproduce the raw-H GPTVQ result (99.8 % identical indices, rest near-ties).
* GPTVQ's decodability test first *failed* — the test decoder read the escape integers in row-major order while the
  encoder writes them block-major; with the encoder's order the reconstruction is bit-exact (the rate is unaffected:
  order-0 entropy does not depend on order).

### 15.11 Limitations

* One small model carries V1–V3 (SmolLM2-135M, fragile: ≥ 0.25 KL at 3.6 bpw for plain ECSQ, and GPTQ hurts on it);
  the second model (§15.9) covers only the decisive configs. One seed except the stated seed reruns; codebook fits
  move KL by ±10–25 %, lattice / scalar quantizers have no fitting randomness.
* Rates are the empirical-entropy proxy (validated to ±0.02 % vs real rANS bytes in phase 2), not real files; the
  pooled frequency tables are charged at 32 b/symbol.
* No VQ fine-tuning (AQLM / QuIP#-style codebook or end-to-end tuning), no trellis-coded quantization, no fixed-rate
  2-bpw regime (where the literature's VQ results live) — the question here was entropy-coded VQ at equal total bits.
* WikiText-2 calibration and evaluation (same domain) except the C4 check in §15.9.

### 15.12 Summary paragraph for README / Space

**Does "sorting the weights as vectors and compressing the clusters" (vector quantization with a shared k-means
codebook) beat plain rounding?** Not once you charge the codebook *and* compare with an entropy-coded scalar
quantizer at the same total size. On SmolLM2-135M, 2- and 4-dimensional k-means codebooks needed 0.1–0.3 more bits
per weight than scalar rounding for the same quality, even though the codebook itself cost almost nothing. Structured
lattices fare better: D4 gave a small, repeatable gain (≈ 0.05–0.1 bits per weight) on both models, while E8 cut the
raw weight error by 14 % yet made one model *worse* and the other better — what matters is the error the layer's
inputs actually see, not the raw weight error. Reusing one codebook for a whole model family lost to the
existing "store the base, code the small difference" scheme, and literally sorting weights costs more bits for the
permutation than the whole compressed model. The big gains that vector methods did show (−45 % KL) turned out to come
from knowing which input channels are sensitive — and a plain scalar quantizer given the same per-channel step sizes
did even better (KL 0.080 vs 0.116 at ≈ 3.7 bits/weight). Lesson: in an entropy-coded codec, spend the
"multi-dimensional" effort on the error metric, not on the codebook.

## 16. ghcol — GPTQ on the Hessian-column step grid: confirmation and integration (2026-11-22)

The pruning study's one positive side finding (PRUNE_FINDINGS.md, S1 dense baselines) was `ghcol`: GPTQ error
feedback (`common.gptq`) on the **Hessian-column grid** `step_ij = fp16(c·rms_row_i) × fp16((h_j / geomean h)^(−1/2))`,
h = diag(H) = E[x_j²] of the layer input (the V2 / §15.4 `hcol` grid, 16 b/row + 16 b/col side info). Found on one
model (SmolLM2-135M), CPU only. On that model GPTQ on the plain *row* grid hurts (§15.4), so the claim was: "the error
feedback pays only once the steps are Hessian-scaled per column". This section tests whether that generalizes,
applies it to the fine-tune delta, integrates it into the real codec, and re-measures the phase-2 headline pairs.
Scripts: `scripts/ghcol_probe.py` (stages `base` / `delta`), `scripts/ghcol_summary.py`, `phase2/ghcol_roundtrip.py`,
`phase2/ghcol_headline.py`; results `results/ghcol_base_*.json`, `results/ghcol_delta_*.json`, `results/ghcol_verdict.json`,
`results/ghcol_roundtrip_smol135.json`, `results/ghcol_family_*.json`, `results/ghcol_headline.json`.

### 16.0 TL;DR (FACT unless marked)

| question | answer |
|---|---|
| **A.** Does ghcol beat the codec's dense coders on other models? (pre-registered: ≥ 0.05 bpw vs the production GPTQ-row line at ≥ 2 points, never a loss > 0.02 vs GPTQ-row or hcol) | **PASS on all 3 models, never a loss.** bpw saved at equal WikiText KL vs the production I-frame coder (GPTQ-ECSQ row grid): **SmolLM2-135M +0.73…+0.97, SmolLM2-360M +0.28…+0.48, Qwen2.5-0.5B +0.09…+0.18**; vs plain `hcol` +0.16…+0.33; vs plain ECSQ +0.66…+0.84 |
| …but is it the **best** dense line? | **No.** The control I added (A3's separable **Fisher** grid, §3, in the same fp16 row × fp16 column form, same GPTQ feedback, same side info = `gfisher`) is better on **all three** models by **0.17–0.31 bpw** (it reproduces §3's Qwen numbers to 4 decimals). The mechanism generalizes — GPTQ pays on a *sensitivity-scaled separable* grid — but diag(H) is not the best source of the scales |
| **B.** ghcol on the fine-tune delta (fp16 base given) | **PASS**: Δ bpw saved at equal chat-KL vs the phase-2 delta coder: **SmolLM2-135M +0.38 / +0.83, Qwen +0.23 / +0.26**. But **plain rounding on the same column grid (`hcol-Δ`) is as good or better on chat-KL** (+0.22 / +0.86, +0.25 / +0.37): on Δ the win is the column grid; GPTQ feedback helps WikiText KL (+0.18…+0.35 bpw vs hcol-Δ) and costs 0.04–0.10 bpw of chat-KL on Qwen |
| integrated? | **Yes, opt-in, alongside phase 2.** `qrow` records can carry fp16 column factors (own or inherited by a P-frame); `quant.ghcol`, `quant.gptq_sep` + `quant.fisher_grid`; family-codec configs `B_ghcol`, `T_pframe_ghcol`, `B_gfisher`, `T_pframe_gfisher`, `T_dghcol`, `T_dhcol`. Defaults and every phase-2 file / result unchanged |
| bit-exact? | **Yes.** Real SmolLM2-135M and SmolLM2-360M pairs (210 / 224 tensors), I-frame + P-frame on the inherited column lattice + fp16-ref Δ: SHA-256 of decoded symbols and fp16 weights = encoder on **all 5 decoder paths** (constriction CPU, process-pool parallel, rans16 numba, MPS, Metal GPU). The new decoder reproduces the published phase-2 demo digests. One real bug was found on the way (int16 wrap of fine-grid lattice indices, §16.7) and fixed on encoder and decoders before the headline runs |
| headline pairs (real files, §16.5) | **Same pair size → much better fine-tune:** chat-KL Qwen 0.128 → **0.078**, SmolLM2-360M 0.138 → **0.059**, SmolLM2-135M 0.538 → **0.105** (+0.7…1.0 % bytes). **Equal quality → smaller:** −2.4 % (Qwen) / −7.4…−7.7 % (360M) pair, and vs two standard GPTQ-int4 files **−42 … −46 %** with lower KL. Fine-tune Δ file (fp16 base given): **−14 … −35 %** at equal chat-KL. The Fisher grid in the same record: a further 9–21 % lower chat-KL at equal size |

### 16.1 Harness (same as the spine / phase 2, every line re-run in it)

* **VERDICT A** = the spine harness (§1): fp16 model on MPS, Hessians 128 × 2048 WikiText-2 train windows (seed 0,
  non-sequential, percdamp 0.01), GPTQ on CPU fp32, WikiText-2 PPL over all windows + KL/top-1 vs the cached fp16
  reference on the first 16. Rate = `common.quantize_model`: per-tensor empirical order-0 entropy + 32 b per used symbol
  + side info (16 b/row; + 16 b/col for column grids, q/k/v and gate/up charged per tensor although they share H) —
  the proxy phase 2 matched to real rANS bytes within ±0.02 % (and §16.5's real files again within ±0.05 %).
* **Anchors reproduce earlier runs:** GPTQ-row c = 0.35 → Qwen 3.5784 bpw / PPL 13.8377 / KL 0.0604 (spine 0.0599:
  Hessian run-to-run noise on MPS), SmolLM2-360M 3.5827 / 0.0907 (phase 2 0.0906), SmolLM2-135M 3.5944 / 0.2833
  (= phase 2); ghcol c = 0.35 on SmolLM2-135M 0.0607 (prune lead, CPU: 0.0608); `gfisher_g0.5` on Qwen 3.6468 / 0.0270
  and 3.1425 / 0.0559 (§3 A3: 0.0271 / 0.0559).
* **Comparison rule:** line by line — bpw the other line needs for the same KL by log-linear interpolation between its
  two bracketing points, both directions (each line's points looked up on the other), never extrapolated, never a
  mixed hull (the S1 trap).
* **VERDICT B** = the phase-2 `T_delta` harness: Δ = W_inst − W_base(fp16), instruct-model Hessians, chat-KL / top-1 /
  NLL on the instruct model's 48 greedy answers + WikiText-2; rate = Δ symbols + side info.

### 16.2 VERDICT A — base weights, 3 models (FACT, `results/ghcol_base_*.json`, `results/ghcol_verdict.json`)

Lines (bpw → WikiText KL), all in-harness:

| model | ghcol (γ = 1) c = 1.0 / 0.7 / 0.5 / 0.35 / 0.25 | hcol c = 0.7 / 0.5 / 0.35 | GPTQ-row (production) c = 0.7 / 0.5 / 0.35 / 0.25 | gfisher γ = 0.5 c = 0.7 / 0.5 / 0.35 |
|---|---|---|---|---|
| SmolLM2-135M | 2.249→0.739, 2.730→0.288, 3.199→0.131, 3.706→**0.0607**, 4.190→0.0293 | 2.722→0.398, 3.195→0.176, 3.704→0.0798 | 2.624→1.376, 3.090→0.610, 3.594→0.283, 4.075→0.154 | 2.697→0.228, 3.165→0.103, 3.670→**0.0482** |
| SmolLM2-360M | 2.229→0.500, 2.709→0.196, 3.178→0.0923, 3.684→**0.0424**, 4.166→0.0214 | 2.702→0.287, 3.174→0.131, 3.682→0.0618 | 2.613→0.411, 3.079→0.203, 3.583→0.0907, 4.063→0.0495 | 2.678→0.141, 3.145→0.0662, 3.650→**0.0304** |
| Qwen2.5-0.5B | 2.235→0.410, 2.713→0.176, 3.181→0.0835, 3.686→**0.0400**, 4.167→0.0198 | 2.704→0.288, 3.176→0.137, 3.684→0.0647 | (spine) 2.610→0.249, 3.075→0.122, 3.578→0.0604, 4.059→0.0293 | 2.676→0.122, 3.142→0.0559, 3.647→**0.0270** |

bpw **saved by ghcol** at equal KL (range over all interpolable points, both directions; + = ghcol cheaper):

| model | vs GPTQ-row (production I-frame) | vs hcol (= V2's best scalar) | vs ECSQ-row (rounding) | vs gfisher (A3 Fisher grid + GPTQ) | GPTQ-int4 quality (KL of the standard int4 file) reached at |
|---|---|---|---|---|---|
| SmolLM2-135M | **+0.73 … +0.97** | +0.16 … +0.19 | +0.66 … +0.79 | **−0.17 … −0.20** | 2.74 bpw (int4 g64: 4.31 fixed-rate, 4.05 rANS) |
| SmolLM2-360M | **+0.28 … +0.48** | +0.19 … +0.24 | +0.70 … +0.84 | **−0.24 … −0.27** | 3.23 bpw (4.31 / 4.05) |
| Qwen2.5-0.5B | **+0.09 … +0.18** | +0.27 … +0.33 | +0.77 … +0.80 | **−0.27 … −0.31** | 3.26 bpw (int4 g128: 4.16 / 3.76) |

* **The finding generalizes as stated:** GPTQ feedback on the column grid beats both plain rounding on it (hcol) and
  GPTQ on the row grid, on every model and at every rate tested (2.2–4.2 bpw); it is never a net loss. On Qwen, where
  row-grid GPTQ already works (it halves KL vs rounding), the gain over the production coder is smallest (≈ 0.1–0.18).
* **γ** (column-factor exponent) 0.5 vs 1 on Qwen: −0.02…−0.03 bpw (γ = 1 kept, as in V2).
* **It is not the best dense line.** `gfisher` uses the same record and the same GPTQ, only the row/column factors come
  from the empirical Fisher (64 × 512 train tokens) with A3's global cross-matrix allocation; it wins by 0.17–0.31 bpw on
  all three models. INTERPRETATION: diag(H) sees how strongly each *input channel* is driven, but not how much the
  *loss* cares about each output row or matrix (A3's probe: per-matrix sensitivity varies ~50×, §3); Fisher supplies
  both. The pre-registered gate was against the production line (written before any Qwen number); the gfisher control
  was added after the first Qwen point lost to §3's A3 line, and is reported, not used to move the gate.

### 16.3 VERDICT B — the fine-tune delta (FACT, `results/ghcol_delta_*.json`)

Δ lines (Δ bpw → chat-KL; WikiText KL in brackets):

| model | production Δ (GPTQ, row grid c·rms(W_T)), c = 0.25 / 0.12 / 0.06 | ghcol-Δ (GPTQ, column grid, rows rms(Δ)), c = 3.2 / 1.6 / 0.8 | hcol-Δ (rounding, same grid) | ghcol-Δ with rows rms(W_T) |
|---|---|---|---|---|
| SmolLM2-135M | 0.870→0.102, 1.706→0.0299, 2.596→0.0101 | 0.815→0.0634, 1.621→**0.0121** (0.0076), 2.504→**0.0030** (0.0019) | 0.687→0.0959, 1.584→0.0121 (0.0107), 2.494→0.0031 (0.0026) | 0.896→0.0696, 1.753→0.0119, 2.654→0.0031 |
| Qwen2.5-0.5B | 0.843→0.0375, 1.686→0.0105, 2.568→0.0031 | 0.805→0.0283, 1.619→**0.0081** (0.0040), 2.495→**0.0020** (0.0011) | 0.670→0.0333, 1.576→**0.0073** (0.0067), 2.483→**0.0018** (0.0018) | 0.863→0.0303, 1.729→0.0076, 2.621→0.0021 |

Δ bpw saved at equal quality vs the production Δ coder:

| model | ghcol-Δ, chat-KL | hcol-Δ, chat-KL | ghcol-Δ, WikiText KL | hcol-Δ, WikiText KL | ghcol-Δ vs hcol-Δ, chat-KL / WikiText KL |
|---|---|---|---|---|---|
| SmolLM2-135M | **+0.38 / +0.83** | +0.22 / +0.86 | +0.53 / +0.88 | +0.60 | +0.05 / −0.04 ; +0.18 / +0.20 |
| Qwen2.5-0.5B | **+0.23 / +0.26** | +0.25 / +0.37 | +0.30 / +0.35 | +0.02 | −0.04 / −0.10 / −0.06 ; +0.31 / +0.35 |

(The ≈ 2.5-bpw points lie below the production line's range: they beat its best point outright — Qwen chat-KL
0.0020 / 0.0018 vs 0.0031 at fewer bits, SmolLM2-135M 0.0030 vs 0.0101.)

* **PASS** by the pre-registered rule (≥ 0.05 bpw vs the production Δ coder at ≥ 2 points, chat-KL).
* **What wins on Δ is the column grid itself.** GPTQ feedback on it is metric-dependent: it buys 0.18–0.35 bpw of
  WikiText KL (the calibration domain) and loses 0.04–0.10 bpw of chat-KL on Qwen (≈ neutral on SmolLM2). INTERPRETATION:
  the Hessian is WikiText-calibrated; on a small, nearly Gaussian Δ (PRUNE_FINDINGS S2) the error feedback tunes the
  reconstruction to the calibration inputs — the same in-domain/out-of-domain split §15.9 saw for GPTVQ on C4.
* Row normalisation by rms(Δ) beats rms(W_T) by 0.08–0.18 bpw (the phase-2 `T_delta` used rms(W_T)).

### 16.4 Integration (FACT: code + tests)

* **Format** (`wcodec/codecs.py`): a `qrow` record may carry `params.col = "own"` + section `col` (fp16 per input
  column) → `step_ij = step_i · col_j`, or `params.col = "ref"` (inherit the reference record's column factors: P-frame on
  that lattice, 0 side info). The product of two fp16 values is **exact in fp32**, so encoder and decoder build the
  identical lattice on any device/order. Records without `col` are byte-identical to phase 2. A P-frame that would drop
  inherited column factors, or a reference without them, raises instead of mis-decoding (several older call sites copy
  only `TState.step`). `own_div` / `div:k` (SVC) with column factors are explicitly not implemented. Not supported by
  the browser (JS) decoder of the published demo — the demo files are row-grid files and are unaffected. [Since §17 the
  JS decoder reads `col`, and the column-grid coders are the default.]
* **Quantizers** (`wcodec/quant.py`): `hcol_factor`, `hcol_grid`, `ghcol` (= `common.gptq` on the column grid, symbols
  identical — unit-tested), `gptq_sep` (GPTQ on any separable grid), `fisher_grid` / `fisher_rowcol` (A3 factors in the
  decodable form; `p2common.fisher_stats` = the a3_fisher procedure). Guards: column factors clamped to the finite
  positive fp16 range, row steps floored at the smallest fp16 (dead input columns and all-zero rows code as 0 instead of
  inf / NaN — unit-tested), and **`int16_safe_col`**: a column factor is raised only where its finest step would need
  lattice indices |W/step| > 2¹⁴ (every fast decoder holds symbols in int16; see §16.7) — after GPTQ the symbols are
  checked again (row-grid fallback for that tensor). Decoders (serial, process-pool, Metal) now **raise** on symbols
  outside int16 instead of wrapping.
* **Family codec** (`phase2/family_codec.py`, opt-in flags, defaults unchanged): `B_ghcol_c*`, `T_ghcol_c*`,
  `T_pframe_ghcol_c*` (P-frame on the decoded ghcol base lattice), `B_gfisher_c*` / `T_pframe_gfisher_c*`, `T_dghcol_c*`
  (GPTQ Δ), `T_dhcol_c*` (rounding Δ). **Fallback:** a tensor whose column grid is degenerate is coded on the row grid
  (no `col`; counted as `n_row_fallback`) and the columns raised by the int16 floor are counted (`n_cols_raised`); both
  are in `results/ghcol_family_*.json` (§16.5). `--no_baseline --store` writes new configs to
  `results/ghcol_family_<tag>.json` and never re-writes a phase-2 result or file. **Per-model applicability (FACT):**
  the column grid beat the row grid on every model and at every rate tested (§16.2, §16.5); no tensor needed the
  row-grid fallback. Per-tensor wins were not measured — the guard is for degenerate grids, not an RD selector.
* **Seam checks:** `phase2/test_wcodec.py` (CPU + MPS) adds I/P/Δ round trips on column lattices, the loud-failure cases
  and degenerate inputs; `phase2/ghcol_roundtrip.py` (§16.0 row) on the real SmolLM2-135M and -360M pairs; and every §16.5 quality
  number is measured on the model **decoded from the written file**, which reproduces the probe's in-memory numbers.
  (Found while building it: GPTQ's in-memory output holds −0.0 where symbols × lattice give +0.0 — value-identical, now
  asserted; the encoder-side digest is the lattice reconstruction, the codec's contract.)

### 16.5 Headline numbers on real files

(FACT, `results/ghcol_family_*.json`, `results/ghcol_headline.json`, `phase2/ghcol_headline.py`.) Same machinery as
§14.3: every size is the bytes of `.wcdc` files (header, tables, fp16 row + column factors, ANS words), every quality
number is measured on the model **decoded from those files**; the lossless rest files (embeddings / norms / biases,
XOR P-frame for the fine-tune) are the phase-2 ones, unchanged. Over the 45 new files: real vs proxy bpw −0.03 … +0.05 %,
row-grid fallback 0, int16 floor raised 1 column in each ghcol file of Qwen and SmolLM2-360M (0 on SmolLM2-135M, 0 for
the Fisher grid); decoded-file quality reproduces the probe (e.g. Qwen B_ghcol_c0.35 KL 0.0404 vs 0.0400 in memory —
Hessian run-to-run noise — SmolLM2-360M 0.0424 vs 0.0424, SmolLM2-135M 0.0607 vs 0.0607).

| pair (real files, every tensor) | phase 2: row-grid I + P, c = 0.35 | ghcol I + P, c = 0.35 | ghcol I + P, c = 0.5 | A3-Fisher grid I + P, c = 0.35 | A3-Fisher grid I + P, c = 0.5 | two standard GPTQ-int4 files |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B {base, Instruct} | 531.7 MB · base KL 0.060 · chat-KL **0.128** | 535.6 MB · base KL 0.040 · chat-KL **0.078** | 506.5 MB · base KL 0.084 · chat-KL **0.173** | 532.8 MB · base KL 0.027 · chat-KL **0.062** | 504.2 MB · base KL 0.057 · chat-KL **0.135** | 916.6 MB · base KL 0.075 · chat-KL **0.148** |
| SmolLM2-360M {base, Instruct} | 295.3 MB · base KL 0.091 · chat-KL **0.138** | 298.3 MB · base KL 0.042 · chat-KL **0.059** | 272.7 MB · base KL 0.092 · chat-KL **0.129** | 296.2 MB · base KL 0.030 · chat-KL **0.053** | 271.0 MB · base KL 0.066 · chat-KL **0.108** | 528.2 MB · base KL 0.086 · chat-KL **0.118** |
| SmolLM2-135M {base, Instruct} | 130.5 MB · base KL 0.283 · chat-KL **0.538** | 131.7 MB · base KL 0.061 · chat-KL **0.105** | 122.9 MB · base KL 0.131 · chat-KL **0.236** | 130.9 MB · base KL 0.048 · chat-KL **0.088** | 122.3 MB · base KL 0.102 · chat-KL **0.200** | 227.9 MB · base KL 0.281 · chat-KL **0.400** |

| fine-tune Δ file (fp16 base given) | phase-2 Δ (GPTQ, row grid) MB → chat-KL | ghcol-Δ MB → chat-KL | hcol-Δ (rounding) MB → chat-KL | Δ MB saved at equal chat-KL: ghcol-Δ / hcol-Δ |
|---|---|---|---|---|
| Qwen2.5-0.5B {base, Instruct} | 37.7 → 0.0375, 75.4 → 0.0105, 114.9 → 0.0031 | 36.0 → 0.0283, 72.4 → 0.0081, 111.6 → 0.0020 | 30.0 → 0.0333, 70.5 → 0.0073, 111.0 → 0.0018 | +10.1 MB (+22 %), +11.5 MB (+14 %), n/a / +11.3 MB (+27 %), +16.5 MB (+19 %), n/a |
| SmolLM2-360M {base, Instruct} | 30.6 → 0.0681, 63.0 → 0.0134, 97.6 → 0.0035 | 31.3 → 0.0407, 63.0 → 0.0063, 97.8 → 0.0018 | 26.5 → 0.0684, 61.6 → 0.0066, 97.4 → 0.0016 | +9.5 MB (+23 %), +19.6 MB (+24 %), n/a / n/a, +19.9 MB (+24 %), n/a |
| SmolLM2-135M {base, Instruct} | 11.5 → 0.1018, 22.6 → 0.0299, 34.5 → 0.0101 | 10.8 → 0.0634, 21.5 → 0.0121, 33.2 → 0.0030 | 9.1 → 0.0959, 21.0 → 0.0121, 33.1 → 0.0031 | +5.0 MB (+32 %), +11.0 MB (+34 %), n/a / +3.0 MB (+25 %), +11.5 MB (+35 %), n/a |

**Family pair (base I-frame + fine-tune P-frame on its lattice + lossless rest):**

* **At the same pair size the fine-tune is much better:** Qwen 535.6 vs 531.7 MB (+0.7 %): chat-KL **0.078 vs 0.128
  (1.6×)**, base KL 0.040 vs 0.060; SmolLM2-360M 298.3 vs 295.3 MB (+1.0 %): **0.059 vs 0.138 (2.4×)**; SmolLM2-135M
  131.7 vs 130.5 MB (+0.9 %): **0.105 vs 0.538 (5.1×)** — the phase-2 caveat of §14.3 (the inherited per-row step is the
  weak point on that model) is resolved.
* **At equal quality the pair gets smaller** (line-by-line interpolation where the lines overlap): Qwen −12.7 MB
  (−2.4 %) at equal chat-KL, −11.7 MB (−2.3 %) at equal base KL; SmolLM2-360M −21.9 MB (−7.4 %) at equal base KL, and its
  c = 0.5 pair (272.7 MB) is **7.7 % smaller** than the phase-2 c = 0.35 pair (295.3 MB) with better fine-tune chat-KL
  (0.129 vs 0.138) and equal base KL (0.092 vs 0.091); SmolLM2-135M's c = 0.5 pair (122.9 MB, chat-KL 0.236) beats the phase-2 c = 0.35 pair (130.5 MB,
  0.538) on both axes. Pair-level percentages are small because 38–61 % of every pair is the lossless rest (embeddings,
  norms), which none of this touches; as a share of the linears alone the same MB are 1.6–2.5× larger.
* **vs two standard GPTQ-int4 files** (the format people ship): Qwen 535.6 vs 916.6 MB (**−42 %**) with chat-KL 0.078 vs
  0.148; SmolLM2-360M 298.3 vs 528.2 MB (−44 %), 0.059 vs 0.118; SmolLM2-135M 122.9 vs 227.9 MB (−46 %), 0.236 vs 0.400.
* **The A3 Fisher grid in the same record is better again:** chat-KL 0.062 / 0.053 / 0.088 at the c = 0.35 pair sizes
  (−21 / −9 / −16 % vs ghcol), −11.4 MB (Qwen) / −7.5 MB (360M) / −2.4 MB (135M) vs the ghcol family at equal chat-KL.

**Fine-tune Δ over an fp16 base (the C2 / §14.3 "full-rank Δ" scenario):** the Δ file shrinks by **14–34 % with ghcol-Δ
and 19–35 % with hcol-Δ** at equal chat-KL (Qwen −14 / −22 % and −19 / −27 %, SmolLM2-360M −23 / −24 % and −24 %,
SmolLM2-135M −32 / −34 % and −25 / −35 %); at the ≈ 2.5-bpw end both reach chat-KL 0.0016–0.0031, below anything the
phase-2 Δ coder produced. GPTQ feedback vs plain rounding on the column grid: equal or worse on chat-KL (Qwen −2…−7 % of
the Δ file; mixed on the SmolLM2 pairs, −3 … +8 %), better on WikiText KL (Qwen +18 / +27 % of the Δ file vs hcol-Δ's
+1 %). Pair-level the Δ saving is only 1–4 % because the pair is dominated by the raw fp16 base linears (§14.3 note: a
lossless rANS of those is ≈ 4/5).

### 16.6 What did not generalize / caveats

* **"ghcol is the new best dense scalar line"** — not true on any of the three models once the A3 Fisher grid is run in
  the same harness (−0.17…−0.31 bpw). The SmolLM2-135M claim was made without that control; the memory rule "always run
  the scalar + same side-info control" applied again.
* **"GPTQ feedback is what pays on the column grid"** — true for base weights (+0.16…+0.33 bpw over hcol), **not for the
  fine-tune delta's chat quality** (hcol-Δ ≥ ghcol-Δ on Qwen chat-KL).
* **Nothing in the real files contradicted VERDICT A / B** once the decoder bug (§16.7) was fixed; the first SmolLM2-360M
  real-file run looked like "ghcol fails on the Instruct model" (chat-KL 0.388 vs 0.138) — it was that bug.
* **Margin shrinks with model strength:** +0.73…+0.97 bpw vs the production coder on the fragile SmolLM2-135M, +0.09…+0.18
  on Qwen, where row-grid GPTQ already works.

### 16.7 Process notes (for honesty)

* Pre-registered rules are in the `scripts/ghcol_probe.py` docstring (written before any Qwen / SmolLM2-360M number);
  the `gfisher` control and the `hcol-Δ` real-file configs were added after seeing data, and are labelled as such.
* **Bug found by the real-file runs (the seam rule paid off):** on SmolLM2-360M the base I-frame decoded from its
  `.wcdc` file had WikiText KL **0.313**, the identical in-memory quantization **0.0424**. Cause: column factors down to
  0.0014 gave lattice indices up to |q| = 34 798 (a single weight of `layers.3.mlp.down_proj`); the file stored it
  correctly (int32 ANS), but the decoder cast symbols to int16 and the wrapped weight wrecked the model. Phase-2 row-grid
  files never come near int16, and the SmolLM2-135M round trip had passed. Fixed on both sides (§16.4: encoder floor +
  decoders raise), re-verified bit-exact on SmolLM2-360M on all 5 decoder paths (`results/ghcol_roundtrip_smol360.json`),
  and **all §16.5 files were re-encoded with the fixed code**; the invalid pre-fix SmolLM2-360M results were discarded.
* **Incident:** the first real-model round trip ran its process-pool decoder from an unguarded script; `spawn` workers
  re-imported it and each re-ran the whole test (4 × 1.4 GB); memory and disk headroom ran out within minutes.
  The concurrently running delta probe stopped itself at its free-disk guard (no result lost — its finished configs were
  checkpointed); the round trip was killed by PID, wrapped in `main()`, and re-run alone.
* A baseline-skip flag was first applied before the LoRA rank list was parsed, so the first SmolLM2-135M real-file run
  re-created the three phase-2 LoRA files in `bitstreams/smol135/` (byte-identical sizes and identical quality —
  deterministic encoder); they were deleted again (the directory is back to its phase-2 state) and the flag fixed.
* Strictly one MPS process at a time (sequential queues in `logs/`, not committed); new `.wcdc` files are deleted after
  each family (sizes and decoded-model quality are in `results/ghcol_family_*.json`; files are reproducible).

### 16.8 Limitations / next

* One seed / one calibration set (WikiText-2) everywhere; chat-KL is the only out-of-domain metric here.
* VERDICT B's probe ran on SmolLM2-135M and Qwen (as planned); SmolLM2-360M's Δ is covered by the real files of §16.5
  only. VERDICT A / B numbers are in-memory reconstructions with the entropy-proxy rate; the decoded real files of §16.5
  reproduce them (rate within ±0.05 %, KL to the 4th digit up to Hessian run-to-run noise).
* The Fisher grid was taken over from §3 as-is (γ = 0.5, 64 × 512 tokens, global shift) — not tuned; its Δ analogue
  (instruct-model Fisher for T_dghcol) and per-tensor RD selection between grids were not tried.
* Not deployed: the published demo / Space / bitstreams are untouched; the browser decoder would need the `col` record
  parameter (and an int16 range check) before column-grid files can be served.
* NEXT (INTERPRETATION): make the Fisher grid (`gfisher`) the default I-frame/P-frame lattice and the column grid with
  plain rounding (`T_dhcol`) the default Δ coder, add `col` to the JS decoder, then rebuild the demo. [Done in §17.]

## 17. The column-grid coders become the DEFAULT — re-measured on real files, browser demo rebuilt (2026-11-22)

§16 found that the best coders in the codec are (i) the **A3 Fisher grid** (fp16 row × fp16 column steps from the
empirical Fisher, + GPTQ; `gfisher`) for base weights, and (ii) the **Hessian-column grid with plain rounding**
(`hcol-Δ`) for the fine-tune delta. Both were integrated opt-in. This section makes them the codec's **default**, re-measures
the three family pairs with the default code path on real `.wcdc` files, verifies every decoder path bit-exactly, and
rebuilds + redeploys the browser demo with them. Scripts: `phase2/family_codec.py` (`--preset`), `phase2/default_headline.py`,
`phase2/ghcol_roundtrip.py --coders default`, `phase2/test_codec_js.py`, `phase2/seam_fields.py`, `phase2/test_page_cdp.mjs`;
results `results/default_family_*.json`, `results/default_headline.json`, `results/default_roundtrip_*.json`,
`results/demo_seam_fields.json`.

### 17.0 TL;DR (FACT unless marked)

| question | answer |
|---|---|
| what is the default now? | **Base I-frame = A3 Fisher grid + GPTQ** (`B_gfisher`); **fine-tune P-frame = plain rounding on the inherited Fisher lattice** (`T_pframe_gfisher_rnd`); **fine-tune Δ over an fp16 base = Hessian-column grid, plain rounding** (`T_dhcol`). `family_codec.py --preset phase2` reproduces phase 2; every older coder stays opt-in |
| same pair size → fine-tune quality (real files, c = 0.35) | chat-KL **Qwen 0.128 → 0.059 (2.2×)**, **SmolLM2-360M 0.138 → 0.048 (2.9×)**, **SmolLM2-135M 0.538 → 0.084 (6.4×)**, at −0.3…−0.4 % bytes; base KL 2.2× / 3.0× / 5.9× lower |
| smaller at better quality | the default **c = 0.5** pair is **5.7 % / 8.9 % / 6.8 % smaller** than the phase-2 c = 0.35 pair *and* better on base KL and chat-KL (Qwen 501.2 vs 531.7 MB, 360M 269.0 vs 295.3 MB, 135M 121.6 vs 130.5 MB); vs two standard GPTQ-int4 files: −42 / −44 / −43 % bytes with 2.5–4.8× lower chat-KL |
| fine-tune Δ over the fp16 base | Δ file **−19…−35 %** at equal chat-KL (Qwen −27 / −19 %, 360M −24 %, 135M −25 / −35 %); its ≈ 2.5-bpw point beats the old coder's best point outright |
| P-frame: rounding or GPTQ? (pre-registered rule) | **rounding, on all 3 pairs** (rule needed 2): it dominates GPTQ at c = 0.35 on every pair (fewer MB *and* lower chat-KL) and saves 8–9 % of the P-frame at equal chat-KL where interpolable — but GPTQ keeps **better WikiText KL** (e.g. Qwen 0.029 vs 0.053): the metric split of §16.3 again; `--pframe gptq` remains available |
| reproducibility | all 21 files that §16.5 had already measured come out with **identical byte counts**; decoded-model KL / chat-KL equal to the 4th digit except 4 values differing by ≤ 0.0007 (MPS run-to-run noise, as in §16.5); real vs proxy bytes −0.03…+0.06 %; no row-grid fallback, no column raised by the int16 floor in any default file |
| bit-exact? | **yes** — SHA-256 of decoded symbols and fp16 weights = encoder on all 5 decoder paths (constriction CPU, process pool, rans16 numba, MPS, Metal) for I-frame, both P-frame quantizers and the Δ, on all three real model pairs (§17.5) |
| demo | JS decoder reads per-column steps (loud errors on missing / inconsistent fields); demo rebuilt with the default coders: "smallest" config **chat-KL 0.538 → 0.084** at 86.3 + 43.9 MB, near-lossless Δ **0.030 → 0.012**; seam gate clean (JS = Python SHA-256 on every file, JS greedy = PyTorch, 32 field groups zero mismatches, real page decode → verify → generate in headless Chromium) before any upload; **deployed to the Space and GitHub Pages and verified live** with the same browser flow (§17.6) |

### 17.1 What "default" means now (FACT: code)

`phase2/family_codec.py --preset default` (the default) writes, for every c in `--cs` (0.35, 0.5):

| role | config | coder |
|---|---|---|
| base I-frame | `B_gfisher_c{c}` | A3 Fisher grid (§3 factors, γ = 0.5, 64 × 512 train tokens, global shift) in the decodable fp16 row × fp16 column `qrow` record + GPTQ (`quant.fisher_rowcol` + `quant.gptq_sep`, int16-safe column floor) |
| fine-tune P-frame | `T_pframe_gfisher_rnd_c{c}` (**plain rounding, default**) / `T_pframe_gfisher_c{c}` (GPTQ, `--pframe gptq`) | closed-loop P-frame on the decoded I-frame's lattice: inherits row **and** column factors (`col = "ref"`, 0 side info), decoder `Q = Q_B + q`; the quantizer was chosen by the pre-registered rule below (§17.4) |
| fine-tune Δ over an fp16 base | `T_dhcol_c{0.8, 1.6, 3.2}` | Hessian-column grid of the Δ (rows rms(Δ), columns diag(H)^−½ of the fine-tune), **plain rounding**, own row + column factors, `base = "fp16ref"` |
| standalone control | `T_gfisher_c0.35` | the I-frame coder on the fine-tune alone (its own Fisher, `data/fisher_<tag>_instruct.pt`) |

`--preset phase2` is exactly the previous default (row-grid GPTQ-ECSQ I / P / standalone, GPTQ row-grid Δ, LoRA ranks,
zstd/gzip reference coders; `phase2/run_phase2_local.sh` now passes it), `--preset none` runs only explicit flags; every
older coder stays available opt-in (`--row_cs`, `--row_delta_cs`, `--ranks`, `--ghcol_cs`, `--ghcol_delta_cs`,
`--gfisher_cs`, `--hcol_delta_cs`, `--pframe_round_cs`). `--no_baseline` now only skips the coder-independent reference
files (fp16, GPTQ-int4, lossless rest), which are reused from `results/phase2_family_<tag>.json` exactly as in §16.5.
No phase-2 file, result or table (§14) and no §15 / §16 number was changed; the decoder accepts every older file.
The decoder also got stricter: unknown `col` modes, `col = "own"` without its section, and step / column sections whose
length does not match the record shape now raise (a 1-element section would have broadcast silently).

Pre-registered (written 20:58, after the SmolLM2-135M GPTQ P-frame but before any rounding-P-frame number): the
owner's brief asked for "plain rounding on the Hessian-column grid for the delta / P-frame". §16 measured plain
rounding only for the Δ over an fp16 base (T_dhcol); a P-frame on the inherited lattice was only measured with GPTQ.
Rule: **rounding becomes the default P-frame quantizer only if, on ≥ 2 of the 3 pairs, the rounding P-frame needs
fewer P-frame MB than the GPTQ P-frame at equal chat-KL (line by line over c = 0.35 / 0.5) or dominates it at the same
c (≤ MB and ≤ chat-KL); otherwise GPTQ stays.**

### 17.2 Headline pairs, before vs after (FACT, `results/default_headline.json`, `results/default_family_*.json`)

Same machinery as §14.3 / §16.5: sizes = bytes of the `.wcdc` files (header, tables, fp16 row + column factors, ANS
words), quality measured on the model **decoded from those files**; the lossless rest files (embeddings / norms, XOR
P-frame for the fine-tune) and the int4 / fp16 reference files are the phase-2 ones, unchanged. "before" = the phase-2
default (row-grid GPTQ-ECSQ I-frame + GPTQ P-frame on its lattice, §14.3); "after" = the new default with the rounding
P-frame.

| pair (real files, every tensor) | before: phase-2 default, row-grid I + P, c = 0.35 | **after: default, Fisher-grid I + P, c = 0.35** | after, c = 0.5 | both standalone, same new coder, c = 0.35 | two standard GPTQ-int4 files |
|---|---|---|---|---|---|
| Qwen2.5-0.5B {base, Instruct} | 531.7 MB · base KL 0.060 · chat-KL **0.128** | 529.7 MB · base KL 0.027 · chat-KL **0.059** | 501.2 MB · base KL 0.056 · chat-KL **0.122** | 684.5 MB · base KL 0.027 · chat-KL **0.065** | 916.6 MB · base KL 0.075 · chat-KL **0.148** |
| SmolLM2-360M {base, Instruct} | 295.3 MB · base KL 0.091 · chat-KL **0.138** | 294.0 MB · base KL 0.030 · chat-KL **0.048** | 269.0 MB · base KL 0.066 · chat-KL **0.108** | 411.1 MB · base KL 0.030 · chat-KL **0.053** | 528.2 MB · base KL 0.086 · chat-KL **0.118** |
| SmolLM2-135M {base, Instruct} | 130.5 MB · base KL 0.283 · chat-KL **0.538** | 130.1 MB · base KL 0.048 · chat-KL **0.084** | 121.6 MB · base KL 0.102 · chat-KL **0.179** | 172.5 MB · base KL 0.048 · chat-KL **0.095** | 227.9 MB · base KL 0.281 · chat-KL **0.400** |

| pair | same pair size (c = 0.35): MB change · fine-tune chat-KL ratio · base KL ratio | c = 0.5 | vs two int4 files (c = 0.35) |
|---|---|---|---|
| Qwen2.5-0.5B {base, Instruct} | -0.4 % · **2.2×** · 2.2× | -0.4 % · **2.1×** · 2.2× | -42 % bytes |
| SmolLM2-360M {base, Instruct} | -0.4 % · **2.9×** · 3.0× | -0.4 % · **2.8×** · 3.1× | -44 % bytes |
| SmolLM2-135M {base, Instruct} | -0.3 % · **6.4×** · 5.9× | -0.3 % · **5.7×** · 5.9× | -43 % bytes |

* At the same c the new pair is 0.3–0.4 % **smaller** (the Fisher I-frame costs +1.9…+2.4 % in bytes, the rounding
  P-frame is 9–11 % smaller than the phase-2 P-frame), so every gain above is at equal or lower bytes. The line-by-line equal-quality
  interpolation of §16.5 is undefined here because the two lines no longer overlap: the new c = 0.5 point is already
  better than the old c = 0.35 point on both base KL and chat-KL (TL;DR row 3).
* The standalone control with the same coder (both models coded independently, rest lossless) costs 684.5 / 411.1 /
  172.5 MB at chat-KL 0.065 / 0.053 / 0.095: **storing the base once and the fine-tune as a P-frame still saves 23–29 % of
  the pair** at the same quality class — the core result of the study holds with the better coder.
* The weakest phase-2 case (SmolLM2-135M, §14.3 caveat: the inherited row step was the weak point on that fragile model)
  improves most (6.4×); the strongest (Qwen, where row-grid GPTQ already worked) least (2.2×) — as in §16.6.

### 17.3 Fine-tune Δ over an fp16 base (FACT)

| fine-tune Δ file over the fp16 base | before: GPTQ row grid, MB → chat-KL | **after: column grid, rounding**, MB → chat-KL | Δ MB saved at equal chat-KL (line by line) |
|---|---|---|---|
| Qwen2.5-0.5B {base, Instruct} | 37.7 → 0.0375, 75.4 → 0.0105, 114.9 → 0.0031 | 30.0 → 0.0333, 70.5 → 0.0073, 111.0 → 0.0018 | +11.3 MB (+27 %), +16.5 MB (+19 %), n/a |
| SmolLM2-360M {base, Instruct} | 30.6 → 0.0681, 63.0 → 0.0134, 97.6 → 0.0035 | 26.5 → 0.0684, 61.6 → 0.0066, 97.4 → 0.0016 | n/a, +19.9 MB (+24 %), n/a |
| SmolLM2-135M {base, Instruct} | 11.5 → 0.1018, 22.6 → 0.0299, 34.5 → 0.0101 | 9.1 → 0.0959, 21.0 → 0.0121, 33.1 → 0.0031 | +3.0 MB (+25 %), +11.5 MB (+35 %), n/a |

(Identical to §16.5's hcol-Δ files — same coder, re-measured.) At WikiText KL the Δ saving is smaller (Qwen +1 %, 360M
+11 %, 135M +28 % of the Δ file): the rounding Δ is tuned for nothing, so it does not exploit the calibration domain
the way GPTQ does (§16.3).

### 17.4 The P-frame quantizer: pre-registered rule and result (FACT)

| P-frame on the inherited Fisher lattice | GPTQ: MB → chat-KL / WikiText KL (c = 0.35; 0.5) | plain rounding: MB → chat-KL / WikiText KL | rounding: P-frame MB saved at equal chat-KL | rule: rounding wins? |
|---|---|---|---|---|
| Qwen2.5-0.5B {base, Instruct} | 47.35 → 0.0617 / 0.0292; 41.29 → 0.1347 / 0.0622 | 44.21 → 0.0594 / 0.0528; 38.32 → 0.1217 / 0.1094 | n/a, +3.75 MB (+8.9 %) | yes (dominates at the same c) |
| SmolLM2-360M {base, Instruct} | 39.88 → 0.0533 / 0.0402; 34.53 → 0.1078 / 0.0841 | 37.67 → 0.0478 / 0.0585; 32.51 → 0.1084 / 0.1237 | n/a, n/a | yes (dominates at the same c) |
| SmolLM2-135M {base, Instruct} | 13.91 → 0.0885 / 0.0540; 12.02 → 0.1999 / 0.1126 | 13.14 → 0.0839 / 0.0727; 11.29 → 0.1789 / 0.1576 | n/a, +0.99 MB (+8.0 %) | yes (dominates at the same c) |

* **Rule outcome: rounding wins on 3 / 3 pairs** (dominance at c = 0.35 everywhere; at c = 0.5 it is smaller with lower
  chat-KL on Qwen and SmolLM2-135M and 2.0 MB smaller with +0.0006 chat-KL on SmolLM2-360M). Default: `--pframe round`.
* **Metric-dependent, as for the Δ in §16.3:** GPTQ's WikiText-calibrated error feedback buys in-domain KL (rounding is
  0.5–1.6 MB *worse* per P-frame at equal WikiText KL, or equal on SmolLM2-135M) and loses chat-KL. INTERPRETATION: a
  P-frame codes the fine-tune on the base's lattice; the fine-tune's Hessian comes from WikiText, the domain the Instruct
  model was not trained for, so feedback that minimises WikiText output error moves the Instruct weights away from what
  its chat answers need. The rounding P-frame also encodes 3.3–4.0× faster (no GPTQ pass).
* The P-frame is still the cheapest way to ship the fine-tune given the lossy base: 0.83–0.99 bpw (rounding) for the
  Instruct linears.

### 17.5 Bit-exact round trip of the default coders (FACT, `results/default_roundtrip_*.json`)

`phase2/ghcol_roundtrip.py --coders default` on the three real pairs, every block linear: Fisher-grid I-frame (c = 0.35),
GPTQ P-frame and rounding P-frame on its inherited lattice, rounding column-grid Δ (c = 1.6) over the fp16 base. For every
file, SHA-256 over all records of the decoded lattice index Q and of the decoded fp16 weights must equal the digests the
encoder computed from its own symbols and lattice reconstruction (the §16 contract), on each decoder path.

| pair | tensors | files (MB) I / P-GPTQ / P-round / Δ | max \|Q\| I / P / P-round / Δ | constriction CPU | process pool | rans16 numba | MPS | Metal GPU |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 210 | 48.7 / 14.5 / 13.5 / 21.0 | 1276 / 1269 / 1269 / 486 | ✓ | ✓ | ✓ | ✓ | ✓ |
| SmolLM2-360M | 224 | 143.5 / 42.4 / 39.3 / 61.6 | 999 / 999 / 999 / 578 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Qwen2.5-0.5B | 168 | 163.1 / 50.4 / 46.2 / 70.6 | 1612 / 1584 / 1584 / 2096 | ✓ | ✓ | ✓ | ✓ | ✓ |

**Zero mismatches** (4 files × 5 paths × 3 pairs). Lattice indices stay ≤ 2 096, far inside int16 (the Fisher grid needed
no int16 floor anywhere; §16.7's overflow was a ghcol-grid effect). Hessians here are a few CPU calibration windows (only
the round trip is tested, not quality), so these files differ slightly from the measured ones. The round trip now encodes
in two passes (one model's Hessians resident at a time) and opens one 4-worker decoder pool at a time — the §16 version
held both Hessian sets and one pool per file, which would not have fit for Qwen next to the other jobs.

### 17.6 Browser demo: JS decoder with per-column steps, rebuilt and redeployed

* **Decoder** (`docs/js/codec.js`): `qrow` records may now carry `params.col` — `"own"` reads the fp16 section `col`
  (one factor per input column), `"ref"` inherits the reference record's factors — and the lattice is
  `fround(step_r · col_j)` (a product of two fp16 values, exact in fp32 and in double), then `W = fp16(fround(Q · lattice)
  [+ W_ref])`, the PyTorch decoder's arithmetic. Every missing or inconsistent field raises a named error instead of
  decoding: `col = "own"` without a `col` section, a `col` / `step` section whose length is not the record's column / row
  count, `col = "ref"` against a reference without factors, a reference *with* factors that the record does not inherit,
  unknown `col` modes, symbols outside int16, and P-frame sums `Q_ref + q` outside int16 (the JS holds `Q` in an
  `Int16Array`; the Python decoder accumulates in int32, so this check is JS-only — the encoder already refuses such
  P-frames). Row-grid records decode exactly as before.
* **Synthetic seam test** (`phase2/test_codec_js.py` → `phase2/test_codec_js.mjs`): row-grid I + P, column-grid I +
  GPTQ P + rounding P, fp16-ref column-grid Δ, and an extremes file (int16-limit indices, fp16 overflow → inf and
  subnormal outputs) on 4 tensor shapes incl. 3 × 5 and 128 × 257: JS symbol + weight SHA-256 = Python on all 8 files;
  all 7 malformed-record cases raise in JS (6 of them also in Python, the 7th is the int16 P-frame sum above).
* **Demo files** (`phase2/demo_build.py`, SmolLM2-135M → -Instruct, rans16 twins of the measured files): the default
  coders, quality copied only from the results store that measured that exact file (path **and** byte count checked):

  | configuration | base (cached once) | Instruct delta | rebuilt Instruct (decoded-file quality) | files it replaced (§14.3) |
  |---|---|---|---|---|
  | near-lossless: lossless fp16 base + column-grid rounding Δ `T_dhcol_c1.6` | 178.1 MB | 51.8 MB | chat-KL **0.0121**, top-1 96.2 % | Δ `T_delta_c0.12`: 53.4 MB, 0.0299 |
  | smallest: Fisher-grid I-frame `B_gfisher_c0.35` + rounding P-frame `T_pframe_gfisher_rnd_c0.35` | 86.3 MB | 43.9 MB | chat-KL **0.0839**, top-1 88.2 % | 85.3 + 45.4 MB, chat-KL 0.538 |
  | control: standalone `T_gfisher_c0.35` (no base reuse) | — | 86.3 MB | chat-KL 0.0951 | `T_ecsq_c0.35`: 85.3 MB, 0.505 |

  The first rebuild used the GPTQ P-frame (`T_pframe_gfisher_c0.35`, chat-KL 0.0885) and passed the same gate; it was
  rebuilt after the P-frame rule (§17.4) chose rounding. The precomputed answers changed accordingly (e.g. the old
  "smallest" Instruct looped "An autumn rain, an autumn rain, …" on the haiku prompt; the new one writes a haiku).
* **Seam gate before any upload** (owner rule; all on the final build):
  `node phase2/test_demo.mjs` — every file of every chain: JS SHA-256 of decoded symbols **and** rebuilt fp16 weights =
  Python decoder (manifest digests); lossless parts = the original checkpoint tensors; the JS Llama forward pass gives
  PyTorch's top-1 and greedy tokens on all test prompts (32 / 32, 15 / 15, 28 / 28 … tokens agree; top-10 logits within
  2.2–5.2 · 10⁻⁵). `phase2/seam_fields.py` — names diffed between the real files / JSON the build wrote and every field
  the JS decoder, worker and page read (record kinds, params, param values, sections, manifest / results / generations
  fields, config names, worker ↔ page message types and commands): **32 groups, zero mismatches**
  (`results/demo_seam_fields.json`). `phase2/test_page_cdp.mjs local` — the real page in headless Chromium (fresh
  profile): each of the 3 configurations downloads → decodes → SHA-verifies (all ✓, the base comes from Cache Storage the
  second time), live generation equals the offline PyTorch answer for family/Instruct, family/base and delta/Instruct, no
  page exception or console error.
* **Public text.** The previous page and its exported phase-2 tables named the measurement machine and the job
  infrastructure in a few captions; these were rewritten (`demo_build.PUBLIC_REWRITE`), the offline decode-timing
  sentence was dropped (it described other files), and `demo_build.public_clean` / the deploy staging now refuse to
  finish if any public demo file or the Space card contains hardware / infrastructure / account words.
* **Deployed (2026-11-22, after the gate):** Hugging Face Space `lifeart/weight-codec` (static, public; commit
  `5476151`, previous head `c310e04`): page + the 8 new rans16 bitstreams + tokenizer + Space card
  (`phase2/deploy_static_space.py --deploy --public-ok`; the results dataset was checked to be private before and after —
  anonymous API access returns 401). GitHub `main` (`136f351`, Pages source `docs/`): `docs/` + README + REPORT + the Space
  card as two commits on top of the previously published history; the research commits behind §15–§17 (scripts,
  results, wcodec changes) are not part of that push. The phase-2 demo bitstreams were left on the Space (no longer
  referenced by the page).
* **Verified live, anonymously:** both hosts serve the new manifest (8 files, "smallest" chat-KL 0.0839) and the new
  decoder; every bitstream resolves (302 → CDN, 200) with exactly the manifest's byte count and CORS headers.
  `node phase2/test_page_cdp.mjs https://lifeart.github.io/weight-codec/` (fresh headless-browser profile, data fetched
  cross-origin from the Space): all 3 configurations download → decode → SHA-verify ✓ (symbols and weights on all 10
  file loads, lossless parts identical to the checkpoint, the base served from Cache Storage on the second chain), live
  generation identical to the offline PyTorch answer for family / Instruct, family / base and delta / Instruct, no page
  error — 37 / 37 checks. The same on the Space's own page (`https://lifeart-weight-codec.static.hf.space/`, family + delta).

### 17.7 Process notes (for honesty)

* The SmolLM2-135M / -360M default runs reproduce the §16.5 files **byte for byte** (identical sizes and decoded-model
  KL / chat-KL to the 4th digit) — the encoder is deterministic and the Hessians / Fisher are recomputed identically
  (MPS, same seeds). On Qwen2.5-0.5B the byte counts are identical too, but four decoded-model numbers moved in the 4th
  digit (B_gfisher_c0.5 KL 0.0558 vs 0.0565, T_pframe_gfisher_c0.5 0.0622 vs 0.0619, T_dhcol_c0.8 0.0018 vs 0.0017;
  SmolLM2-135M T_dhcol_c3.2 0.0437 vs 0.0436) — the MPS run-to-run noise §16.5 already reported; §16 stored sizes, not
  file hashes, so byte-identity of the content itself is not claimed.
* The P-frame rule was pre-registered after the first SmolLM2-135M GPTQ P-frame number (which §16.5 had already
  published) and before any rounding-P-frame number; the rule's metric (chat-KL) was the owner's stated reason for
  preferring rounding. The decision fell after two pairs (2 / 3 needed); Qwen was still measured with both quantizers.
* The demo was first rebuilt with the GPTQ P-frame (to exercise the new JS decoder on real column-grid files early) and
  rebuilt again after the rule; only the second build was deployed.
* The new public-text guard first flagged a CSS system-font name in `style.css` (false positive); the pattern now skips it.
* Strictly one heavy process at a time (sequential queues in `logs/`, not committed); 360M / Qwen `.wcdc` files and
  model caches were deleted after each family (sizes and decoded quality are in `results/default_family_*.json`; files
  are reproducible); the SmolLM2-135M files are kept because the demo serves their rans16 twins.
* Wording only (no number): three process anecdotes in §15.10 / §16.7 were rephrased without machine-resource details
  before the report went public.

### 17.8 Limitations

* Same harness and limits as §16 (one seed, WikiText-2 calibration, chat-KL on 48 prompts as the only out-of-domain
  metric). The P-frame quantizer choice is **metric-dependent**: rounding wins chat-KL, GPTQ wins WikiText KL (§17.4); a
  user who cares about in-domain perplexity of the fine-tune should pass `--pframe gptq`.
* The Fisher grid was not re-tuned (γ = 0.5, 64 × 512 tokens, as in §3); per-tensor selection between grids and a
  Fisher-weighted Δ grid remain untried (§16.8).
* The browser decoder is single-threaded JS; decode speed of the new files was not benchmarked natively (§14.6 measured
  the phase-2 files; the new records add one fp16 multiply per weight in reconstruction).

