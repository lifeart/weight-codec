# Pruning / sparsity probe: can we zero out weights and win on compression, with the mask charged?

Owner's hypothesis (2026-11-22): *"Can we thin out the matrix (prune weights to zero) without harming the
model, and win on compression?"*  Two probes, both on **SmolLM2-135M**:
**S1** prunes the base model's weights. **S2** prunes the fine-tune delta Δ = instruct − base, which our best
family codec entropy-codes densely.  Scripts: `scripts/prune_probe.py` (stages `selftest`, `s1`, `s2`, `ctx`)
and `scripts/prune_summary.py` (tables and figures).  Results: `results/prune_s1_smol135.json`,
`results/prune_s2_smol135.json`, `results/prune_s2_delta_dist_smol135.json`, `results/prune_ctx_smol135.json`.
Figures: `figures/prune_s1_smol135.png`, `figures/prune_s2_delta_rd.png`, `figures/prune_s2_delta_hist.png`.

## Verdict

| probe | pre-registered rule (written before any number) | result | verdict |
|---|---|---|---|
| **S1**: prune the weights (magnitude / Wanda / SparseGPT; unstructured ρ ∈ {0.7, 0.5, 0.4, 0.3} and 2:4; alone and stacked with quantization) | PASS iff some sparse scheme saves ≥ 0.10 bpw vs the best dense scalar line at equal KL at ≥ 2 points. FAIL if < 0.05 everywhere | **no sparse scheme saves any bits anywhere.** The best case, SparseGPT ρ=0.7 + quant at the coarsest step, is **−0.02 bpw** (a tie from below). Every other point loses 0.15 to 7.5 bpw | **FAIL** |
| **S2**: prune the fine-tune delta (top-ρ by \|Δ\|, by \|Δ\|·√h, SparseGPT-on-Δ, DARE; fp16 or quantized survivors) | PASS iff some sparse-Δ scheme saves ≥ 0.10 bpw vs the best dense-Δ line at equal chat-KL at ≥ 2 points. FAIL if < 0.05 everywhere | **all 26 sparse-Δ configs lose**, by 0.16–2.6 bpw on chat-KL and 0.15–2.7 bpw on WikiText KL (DARE, random mask with 1/ρ rescale: up to 5.9 / 6.4) | **FAIL** |

**FACT** (SmolLM2-135M, this harness): at equal effective bits per weight with the mask charged, pruning
never beats entropy-coded scalar quantization of the same weights, for the base weights or for the
fine-tune delta.  Three things cause this:

1. **The entropy coder already prunes, for free.**
   - The dense ECSQ/GPTQ coder maps 11–53% of weights to symbol 0 (≈15% at 3.7 bpw, ≈40% at 2.25 bpw). It pays for those zeros only through P(0) inside its entropy code.
   - For the delta, 32–86% of entries already fall within half a step of zero at the operating points of our family codec.
   - Explicit pruning adds zeros the coder would not have chosen.
2. **The mask is expensive and buys little.** It costs ≈ H(ρ): 0.88 bpw at ρ=0.7, 1.00 at ρ=0.5, 0.65 for 2:4. Meanwhile:
   - one-shot pruning alone does more damage than quantizing every weight to ~2.3–3.4 bits;
   - widening the zero bin hurts even with a *free* mask (the S2 dead-zone control below).
3. **The fine-tune delta is dense and nearly Gaussian, not sparse.**
   - Excess kurtosis 0.45 (Gaussian 0, Laplace 3).
   - The top 10% of entries carry 48.8% of the energy (Gaussian 43.9%, Laplace 59.5%).

**HYPOTHESIS** (not tested here): larger models are known to be more prunable (SparseGPT reports
near-lossless 50% only at multi-billion scale), so the S1 margin should shrink with scale. But the S1
mechanism does not depend on scale. Sparse+quant converges *onto* the dense coder as pruning shrinks toward
the coder's own natural zeros, and it loses as pruning grows. So we do not expect a crossover at 0.5B.  Not
run on Qwen2.5-0.5B: both probes fail with clear margins at 135M, and a CPU-only Qwen run would need
~2.3 GB of fp32 Hessians for little expected benefit.

## Harness (self-consistent; the dense baselines were re-run, not copied)

- **Hardware.** CPU, fp32 compute on fp16-rounded weights. Another lead held the GPU, and two MPS jobs at once corrupt results (see memory notes).
- **Distortion.** KL(ref‖model) and top-1 against our own CPU fp16-weight reference on the **first 16 WikiText-2 test windows × 2048**: the same KL windows every probe in this project uses.
  - `PPL16` is perplexity on those 16 windows only. It is *not* the 148-window PPL in REPORT.md.
  - The fp16 floor: KL 0.0000, top-1 0.998. The 0.2% top-1 gap comes from near-tie argmax flips, because the reference hidden states are stored in fp16.
  - S2 adds the chat metric: KL / top-1 / NLL on the instruct model's own greedy answers to the 48 fixed prompts (`results/chat_set_smol135_instruct.json`).
- **Harness reproduces the earlier MPS numbers to 4 decimals:**

  | config | this harness (bpw / KL) | earlier MPS result |
  |---|---|---|
  | `hcol_c0.35` | 3.700 / 0.0797 | V2 3.700 / 0.0797 |
  | `hcol_c0.5` | 3.193 / 0.1760 | V2 3.193 / 0.1760 |
  | `ecsq_row_c0.35` | 3.589 / 0.2550 | V1 3.589 / 0.2549 |
  | `zero_bits_base` (chat KL) | 0.5565 | V3 0.5567 |
  | `dscal_c0.8_nD` (bpw / chat KL) | 2.425 / 0.0096 | V3 2.425 / 0.0096 |

- **Pruned set.** All `nn.Linear` in the blocks (106,168,320 weights). Hessians: 128 × 2048 WikiText-2 train windows (seed 0), non-sequential, percdamp 0.01.
- **SparseGPT.** Our implementation (`sparsegpt()` in `scripts/prune_probe.py`) is verified **bit-exact** against `common.gptq` at ρ=1 (`--stage selftest`).
  - Unstructured masks use the reference "adaptive mask selection" per 128-column block, with saliency w²/[H⁻¹]ⱼⱼ².
  - 2:4 masks are chosen every 4 columns on the error-updated weights.
  - Pruning and quantization are joint: the OBS update absorbs both the pruning error and the quantization error.

## Mask-cost accounting (shown explicitly, charged everywhere)

Every config is charged with two decodable codes, and its bpw is the **cheaper** of the two:

- **EXPLICIT** = entropy-coded keep-mask (2:4: the entropy-coded pattern index per group of 4) + entropy-coded symbols of the *kept* weights only + side info.
- **JOINT** (unstructured only) = **one** entropy-coded symbol stream over all weights, with pruned weights as symbol 0. The mask is paid inside P(0).
  - Because S is a function of (mask, values), H(S) ≤ H(mask) + ρ·H(v | kept).
  - Joint was indeed cheaper or equal in every quantized unstructured config: 3.522 vs 3.527, 3.128 vs 3.152, and 2.709 vs 2.818 bpw at ρ=0.7 for c = 0.35 / 0.5 / 0.7.
- **Values and side info:**
  - kept fp16 values = 1 sign + 10 mantissa bits raw + entropy-coded 5-bit exponent. The raw-16-bit cost is also stored;
  - quantized values: on the S1 `hcol` grid (fp16 c·rms_row × fp16 diag(H)^(−1/2) column factor), side info is 16 b/row + 16 b/col = 0.046 bpw;
  - on the int4 g64 grid, side info is 20 b per 64-group = 0.3125 bpw.
- **Entropy tables.** All entropy-coded streams use `vq.Accounting`: min(per-tensor, pooled-per-type) frequency tables at 32 b per used symbol. This is the same rule the scalar front in REPORT §15 was charged with.

| sparsity pattern | theory | measured mask/pattern cost (bpw) |
|---|---|---|
| unstructured ρ = 0.7 / 0.5 / 0.4 / 0.3 | H(ρ) = 0.881 / 1.000 / 0.971 / 0.881 | 0.881 / 1.000 / 0.971 / 0.881 |
| unstructured ρ = 0.3 / 0.1 / 0.05 / 0.02 / 0.01 (S2, on Δ) | H(ρ) = 0.881 / 0.469 / 0.286 / 0.141 / 0.081 | 0.881 / 0.469 / 0.287 / 0.142 / 0.081 |
| 2:4 structured | ≤ log2 C(4,2) / 4 = 0.646 | 0.647 (SparseGPT), 0.646 (Wanda): the pattern distribution is ~uniform, so entropy coding saves nothing |
| DARE (random mask regenerated from a seed) | 0 | 0 (but see S2: the 1/ρ rescale destroys fidelity) |

Order-0 mask coding gave the mask no help from context.  The **CTX** section below gives the mask, *and the
dense coder*, the same context-adaptive model, to check whether a smarter mask coder could flip S1.

## S1: pruning the base weights (SmolLM2-135M)

Dense baselines (same harness):

- `hcol`: V2's best scalar coder, plain rounding on the diag(H)-column grid.
- `ghcol`: **GPTQ on the same hcol grid** (= SparseGPT machinery at ρ=1). This is the best dense line and the reference for every comparison below. *Side finding:* `ghcol` saves **≈0.18 bpw** at equal KL vs `hcol` (e.g. KL 0.0608 vs 0.0797 at 3.70 bpw). In V2, GPTQ on the plain row grid *hurt*.

`saved` = (bpw the best dense line needs for the same KL) − (bpw of the sparse config). It is log-linear
interpolation along **each dense line separately**, taking the cheapest line.  Negative means sparse loses.
The `saved` value written into the JSON *during* the run used a Pareto envelope of mixed hcol/ghcol points.
That envelope zig-zags between two lines and flattered sparse configs by up to ~0.17 bpw, so the table below
(from `prune_summary.py`) supersedes it.

**Pruning only (kept values fp16, all charged):**

| method | ρ | mask bpw | total bpw | KL | top-1 | PPL16 | the best dense coder reaches this KL at |
|---|---|---|---|---|---|---|---|
| SparseGPT | 0.7 | 0.881 | 9.82 | 0.0953 | 0.843 | 16.12 | **3.41 bpw** |
| Wanda | 0.7 | 0.882 | 9.86 | 0.1068 | 0.830 | 16.36 | 3.34 bpw |
| magnitude | 0.7 | 0.882 | 9.80 | 0.6190 | 0.613 | 27.01 | 2.33 bpw |
| SparseGPT | 0.5 | 1.000 | 7.23 | 0.6683 | 0.623 | 28.42 | **2.29 bpw** |
| Wanda | 0.5 | 1.000 | 7.28 | 0.7384 | 0.610 | 30.89 | 2.24 bpw |
| magnitude | 0.5 | 1.000 | 7.20 | 3.4823 | 0.181 | 486.3 | < 1.8 bpw |
| SparseGPT | 2:4 | 0.647 | 7.08 | 1.2471 | 0.519 | 50.72 | **2.06 bpw** |
| Wanda | 2:4 | 0.646 | 7.09 | 1.8717 | 0.421 | 95.51 | 1.91 bpw |
| magnitude | 2:4 | 0.647 | 7.07 | 4.9661 | 0.095 | 2178 | < 1.8 bpw |
| SparseGPT / Wanda / magnitude | 0.4 | 0.971 | 5.9 | 1.60 / 1.79 / 6.52 | | 73.5 / 89.0 / 10744 | ≈ 2.0 bpw or less |
| SparseGPT / Wanda / magnitude | 0.3 | 0.881 | 4.6 | 3.82 / 3.95 / 9.39 | | 684 / 792 / 189040 | < 1.8 bpw |
| *fp16 reference* | 1 | 0 | 16 | 0 | 0.998 | 14.71 | |

FACT: on a 135M model, one-shot pruning is not near-lossless at any useful density.

- **Removing 30%** of weights (SparseGPT, OBS update) costs KL 0.095. Quantizing *all* weights at 3.4 bpw costs less than that.
- **50% unstructured** costs KL 0.67 (PPL16 14.7 → 28.4), which is the damage of a ~2.3-bpw quantizer.
- **2:4** is about twice as damaging as unstructured 50% (KL 1.25 vs 0.67). Its cheaper mask (0.65 vs 1.00 bpw) cannot pay for that.
- **Magnitude pruning** is catastrophic everywhere. Activation-aware saliency (Wanda / SparseGPT) is mandatory.

**Sparse + quantization, the realistic competitor (mask charged; SparseGPT = joint OBS prune + quantize):**

| config | ρ | code | mask bpw | values bpw | side | **total bpw** | KL | top-1 | best dense KL at the same bpw | **bpw saved vs best dense line** |
|---|---|---|---|---|---|---|---|---|---|---|
| SparseGPT ρ0.7 + hcol c0.7 | 0.70 | joint | 0.881 | 1.891 | 0.046 | **2.709** | 0.3092 | 0.734 | 0.2973 | **−0.021** |
| SparseGPT ρ0.7 + hcol c0.5 | 0.70 | joint | 0.881 | 2.226 | 0.046 | **3.128** | 0.1856 | 0.788 | 0.1467 | −0.155 |
| SparseGPT ρ0.7 + hcol c0.35 | 0.70 | joint | 0.881 | 2.603 | 0.046 | **3.522** | 0.1450 | 0.810 | 0.0802 | −0.386 |
| SparseGPT ρ0.5 + hcol c0.7 | 0.50 | joint | 1.000 | 1.419 | 0.046 | **2.453** | 0.8687 | 0.573 | 0.4788 | −0.268 |
| SparseGPT ρ0.5 + hcol c0.5 | 0.50 | joint | 1.000 | 1.640 | 0.046 | **2.682** | 0.7425 | 0.603 | 0.3126 | −0.441 |
| SparseGPT ρ0.5 + hcol c0.35 | 0.50 | joint | 1.000 | 1.852 | 0.046 | **2.894** | 0.7002 | 0.615 | 0.2122 | −0.628 |
| SparseGPT ρ0.4 + hcol c0.35 | 0.40 | joint | 0.971 | 1.504 | 0.046 | **2.518** | 1.5876 | 0.463 | 0.4236 | −0.546 |
| SparseGPT 2:4 + hcol c0.7 | 0.50 | explicit | 0.647 | 1.491 | 0.046 | **2.181** | 1.4739 | 0.471 | 0.8779 | −0.183 |
| SparseGPT 2:4 + hcol c0.5 | 0.50 | explicit | 0.647 | 1.723 | 0.046 | **2.413** | 1.3848 | 0.493 | 0.5193 | −0.392 |
| SparseGPT 2:4 + hcol c0.35 | 0.50 | explicit | 0.647 | 1.974 | 0.046 | **2.662** | 1.2958 | 0.510 | 0.3244 | −0.618 |
| Wanda ρ0.7 + hcol c0.5 (no OBS update) | 0.70 | joint | 0.882 | 2.205 | 0.046 | **3.128** | 0.2292 | 0.757 | 0.1467 | −0.276 (vs its no-feedback twin `hcol`: −0.088) |
| Wanda ρ0.7 + hcol c0.35 | 0.70 | joint | 0.882 | 2.594 | 0.046 | **3.517** | 0.1692 | 0.793 | 0.0808 | −0.483 (twin −0.299) |
| Wanda ρ0.5 + hcol c0.5 / c0.35 | 0.50 | joint | 1.000 | 1.613 / 1.805 | 0.046 | **2.657 / 2.848** | 0.857 / 0.793 | | 0.327 / 0.231 | −0.467 / −0.631 |
| Wanda 2:4 + hcol c0.5 / c0.35 | 0.50 | explicit | 0.646 | 1.711 / 1.962 | 0.046 | **2.401 / 2.650** | 2.049 / 1.977 | | 0.533 / 0.332 | −0.518 / −0.755 |
| *dense GPTQ int4 g64 (control)* | 1 | – | 0 | 3.738 | 0.3125 | **4.050** (fixed-rate 4.313) | 0.2799 | 0.740 | 0.0358 | −1.308 |
| SparseGPT 2:4 + int4 g64 | 0.50 | explicit | 0.647 | 1.964 | 0.3125 | **2.923** (fixed-rate 2.959) | 1.5895 | 0.457 | 0.2016 | −0.951 |
| SparseGPT ρ0.5 + int4 g64 | 0.50 | explicit | 1.000 | 1.939 | 0.3125 | **3.251** (fixed-rate 3.313) | 0.8647 | 0.582 | 0.1216 | −1.065 |

Best dense line (`ghcol`, bpw → KL): 1.82 → 2.43, 2.25 → 0.725, 2.47 → 0.468, 2.73 → 0.287, 2.94 → 0.195,
3.20 → 0.132, 3.70 → 0.0608, 4.18 → 0.0293.  Full table: `python scripts/prune_summary.py`.

**Why sparse+quant ties at best (the mechanism, FACT).**  The dense coder is already sparse. Its symbol-0
fraction is:

| dense coder | `ghcol` c0.25 | c0.35 | c0.5 | c0.6 | c0.7 | c0.85 | c1.0 | c1.4 |
|---|---|---|---|---|---|---|---|---|
| bpw | 4.18 | 3.70 | 3.20 | 2.94 | 2.73 | 2.47 | 2.25 | 1.82 |
| weights coded as 0 | 10.7% | 14.9% | 21.0% | 25.0% | 28.9% | 34.5% | 39.8% | 52.6% |

- SparseGPT ρ=0.7 at c0.7 zeroes 32.2% of weights, against the dense coder's own 28.9%. It is nearly the same code, hence the −0.02 tie.
- At finer steps the same ρ=0.7 forces 30% zeros where the coder wanted 11–21%, and the loss grows (−0.16, −0.39).
- At ρ ≤ 0.5 it loses at every step.
- Explicit pruning is a *forced, wider zero bin plus a mask*. Error feedback partly compensates, but never enough to pay for the extra distortion.

### CTX: does a smarter (context-modelled) mask coder flip S1?  (fairness check)

A natural objection: order-0 coding charges the mask the full H(ρ), and a context-modelled mask coder
might be much cheaper (e.g. if pruning decisions are predictable per input column).  `--stage ctx`
regenerates the exact S1 symbols and offers **both** coders the same better model:

- a per-column (or per-row) **adaptive** Krichevsky–Trofimov model for the zero-flag [S = 0];
- the usual order-0 table for the nonzero symbols;
- for 2:4, an adaptive model of the pattern index per group-column.

The code lengths are exact adaptive code lengths with no side info.  This is the "smart mask coder" applied to
the sparse mask *and* to the dense coder's own zeros (`results/prune_ctx_smol135.json`).

| config | KL | bpw, order-0 | bpw, context model | mask/pattern bpw order-0 → context | **saved vs dense, both order-0** | **saved vs dense, both context-coded** |
|---|---|---|---|---|---|---|
| SparseGPT ρ0.7 + hcol c0.7 | 0.3092 | 2.710 | 2.700 | 0.881 → 0.874 | −0.021 | **−0.025** |
| SparseGPT ρ0.7 + hcol c0.5 | 0.1856 | 3.131 | 3.123 | 0.881 → 0.874 | −0.155 | −0.158 |
| SparseGPT ρ0.7 + hcol c0.35 | 0.1450 | 3.526 | 3.518 | 0.881 → 0.874 | −0.387 | −0.389 |
| SparseGPT ρ0.5 + hcol c0.7 / 0.5 / 0.35 | 0.869 / 0.743 / 0.700 | 2.454 / 2.683 / 2.896 | 2.441 / 2.669 / 2.883 | 1.000 → 0.986 | −0.269 / −0.442 / −0.630 | −0.275 / −0.448 / −0.635 |
| SparseGPT 2:4 + hcol c0.7 / 0.5 / 0.35 | 1.474 / 1.385 / 1.296 | 2.183 / 2.416 / 2.666 | 2.176 / 2.409 / 2.659 | 0.647 → 0.639 | −0.184 / −0.395 / −0.622 | −0.201 / −0.411 / −0.638 |
| *dense `ghcol`, 8 points c0.25…c1.4* | | | | *(zero-flag)* | | the context model gains the dense coder 0.004–0.027 bpw |

FACT: the SparseGPT mask is close to **i.i.d. Bernoulli(ρ)**.  Its per-column adaptive cost is only 0.7–1.4%
below H(ρ), because the OBS saliency w²/[H⁻¹]ⱼⱼ² already normalises column importance.  The same model
helps the dense coder slightly *more* at low rate.  Every saving moves by ≤ 0.016 bpw and stays negative, so
**the S1 verdict does not depend on how cleverly the mask is coded**.

A first version of this check (`ctx` v1, discarded and documented in the JSON `_meta`) used per-column
adaptive models over the *full* symbol alphabet.  It charged the dense coder ≈ +0.4 bpw of model-learning
cost (an alphabet of K ≈ 50–100 symbols with only 576–1536 samples per column) and the binary mask almost
none.  That made it asymmetric in favour of sparsity, so it was replaced by the zero-flag model above.

## S2: is the fine-tune delta sparse? (SmolLM2-135M-Instruct given the fp16 base)

### The |Δ| distribution (FACT, `results/prune_s2_delta_dist_smol135.json`, `figures/prune_s2_delta_hist.png`)

| statistic | Δ (all block linears) | Gaussian | Laplace |
|---|---|---|---|
| ‖Δ‖ / ‖W_inst‖ | **8.07%** (q 8.4%, k 7.3%, v 6.3%, o 7.2%, gate 9.4%, up 7.7%, down 7.5%) | | |
| excess kurtosis of Δ / rms_row(Δ) | **0.45** | 0 | 3 |
| \|Δ\|/rms_row quantiles q50 / q90 / q99 / q99.9 | 0.64 / 1.64 / 2.72 / 3.64 | 0.67 / 1.64 / 2.58 / 3.29 | |
| energy in the top 1% / 5% / 10% / 25% / 50% of entries (by \|Δ\|, per tensor) | 10.5% / 32.2% / **48.8%** / 76.2% / 94.3% | 8.4% / 27.9% / 43.9% / 72.4% / 92.9% | 16.2% / 42.4% / 59.5% / 83.7% / 96.7% |
| diag(H)-weighted energy in the top 1% / 10% (by \|Δ\|·√hⱼ) | 28.4% / 62.2% | | |
| entries exactly 0 in fp16 | 2.5% | | |
| **entries within ½ step of 0** (= already coded as 0 by dense ECSQ), step c·rms_row(W_inst), c = 0.06 / 0.12 / 0.25 (the phase-2 `T_delta` points) | **32.5% / 57.3% / 86.1%** | | |
| entries within 1 step of 0, same steps | 57.3% / 84.8% / 98.4% | | |
| entries within ½ step, step c·rms_row(Δ), c = 0.8 / 1.6 / 3.2 (V3 `nD`) | 33.3% / 59.8% / 89.1% | | |

FACT: the delta is **dense** (Gaussian body, tails between Gaussian and Laplace beyond ~3σ). Its energy is
only slightly more concentrated than a Gaussian's, and nowhere near "a few entries carry the update".  The
Hessian-weighted energy is more concentrated (top 10% → 62%), but that is **column sensitivity**, not
sparsity, and the dense `hcol` grid already exploits it (below).  The entropy coder's "soft sparsity" is
large: at c = 0.25 it codes 86% of Δ as zero without any mask.

### Sparse Δ vs dense entropy-coded Δ (all charged; `figures/prune_s2_delta_rd.png`)

Dense-Δ lines (same harness, bpw → chat-KL):

| dense Δ coder | points (bpw → chat-KL) |
|---|---|
| `dhcol` (ECSQ on the hcol grid, rms_row(Δ) × diag(H_inst)^(−1/2); **best**) | 0.126 → 0.412, 0.315 → 0.256, 0.687 → 0.0959, 1.181 → 0.0257, 1.584 → 0.0121, 2.065 → 0.0054, 2.494 → 0.0030 |
| `dscal_nD` (V3 row step) | 0.630 → 0.147, 1.518 → 0.0350, 2.425 → 0.0096 |
| `dgptq_nD` (phase-2 style GPTQ on Δ) | 0.793 → 0.0888, 1.570 → 0.0251, 2.441 → 0.0080 |
| `dghcol` (GPTQ on the hcol grid) | ≈ `dhcol` on chat-KL, better on WikiText KL |

`zero_bits_base` (instruct := base, 0 bits) gives chat-KL 0.5565.

| sparse-Δ config | ρ | mask bpw | **total bpw** | chat KL | WT2 KL | **saved vs best dense line (chat-KL)** | saved (WT2 KL) |
|---|---|---|---|---|---|---|---|
| top-1% \|Δ\|, fp16 values | 0.01 | 0.081 | **0.195** | 0.5104 | 0.1376 | −0.162 | −0.154 |
| top-2% \|Δ\|, fp16 | 0.02 | 0.142 | **0.367** | 0.4704 | 0.1282 | −0.305 | −0.294 |
| top-2% \|Δ\|·√h, fp16 | 0.02 | 0.142 | **0.392** | 0.4032 | 0.1087 | −0.271 | −0.242 |
| top-5% \|Δ\|·√h + hcol c1.6 | 0.05 | 0.287 | **0.451** | 0.3052 | 0.0866 | −0.226 | −0.195 |
| top-5% \|Δ\|, fp16 | 0.05 | 0.287 | **0.865** | 0.3555 | 0.1114 | −0.697 | −0.726 |
| top-10% \|Δ\|·√h + hcol c1.6 | 0.10 | 0.469 | **0.719** | 0.2133 | 0.0622 | −0.360 | −0.308 |
| SparseGPT-on-Δ 10% + hcol c1.6 | 0.10 | 0.469 | **0.742** | 0.2053 | 0.0458 | −0.369 | −0.189 |
| top-10% \|Δ\|, fp16 | 0.10 | 0.469 | **1.649** | 0.2623 | 0.0895 | −1.368 | −1.409 |
| top-30% \|Δ\|·√h + hcol c1.6 (best sparse) | 0.30 | 0.881 | **1.377** | 0.0369 | 0.0195 | **−0.333** | −0.371 |
| SparseGPT-on-Δ 30% + hcol c1.6 | 0.30 | 0.881 | **1.389** | 0.0401 | 0.0138 | −0.375 | −0.160 |
| top-30% \|Δ\| + row step c1.6 | 0.30 | 0.881 | **1.312** | 0.0792 | 0.0461 | −0.583 | −0.763 |
| top-20% \|Δ\|, fp16 | 0.20 | 0.722 | **3.089** | 0.1459 | 0.0614 | −2.588 | −2.673 |
| DARE ρ0.5 (seeded mask, 0 mask bits, Δ/ρ in fp16) | 0.50 | 0 | **6.771** | 0.0621 | 0.0645 | −5.949 | −6.378 |
| DARE ρ0.1 | 0.10 | 0 | **1.361** | **0.8042** (worse than 0 bits) | 0.7212 | < −1.8 | < −2.7 |

All 26 configs are in `prune_summary.py` output.  Every one loses, at every rate, on both metrics.

**Dead-zone control: is "drop the small Δ" a good idea even with a *free* mask?**  This is a dense hcol-Δ
coder that zeros |x| < θ·step and rounds otherwise. It is one stream with no mask: the scalar twin of
"prune small Δ, quantize the rest".

| control | c | bpw | chat KL | plain-rounding hcol KL at the same bpw | saved vs plain rounding (chat / WT2) |
|---|---|---|---|---|---|
| dead-zone θ = 0.75 | 1.1 / 1.6 / 2.2 | 1.804 / 1.169 / 0.678 | 0.0200 / 0.0699 / 0.1775 | 0.0084 / 0.0265 / 0.0981 | −0.49 / −0.36 / −0.22 (WT2 −0.21 / −0.16 / −0.14) |
| dead-zone θ = 1.0 | 1.1 / 1.6 / 2.2 | 1.462 / 0.784 / 0.363 | 0.0591 / 0.1885 / 0.3253 | 0.0152 / 0.0740 / 0.2260 | −0.59 / −0.35 / −0.14 (WT2 −0.36 / −0.25 / −0.12) |

FACT: widening the zero bin beyond ½ step **hurts by 0.12–0.59 bpw even when the mask is free**.  The plain
rounding ECSQ (dead-zone exactly ½ step) is already the right "soft sparsity" for this Gaussian-like delta.
The sensitivity-ranked sparse configs do beat this blunt dead-zone (e.g. top-30% |Δ|·√h + hcol c1.6 is +0.12 bpw vs the better of the two dead-zone lines),
because choosing *which* entries to drop by Hessian saliency helps.  They still lose 0.33 bpw to plain
rounding.  DARE's random drop with a 1/ρ rescale is the worst reconstruction code of all.  It is a
model-*merging* trick, not a compression code: at ρ=0.1 the chat-KL (0.80) is worse than not coding the delta
at all (0.56).

## Where 2:4 *is* worth it: the runtime story (separate from the storage verdict)

2:4 is the only pattern hardware accelerates: NVIDIA Ampere and newer sparse tensor cores, with up to 2× peak math throughput
on the pruned GEMMs. Speedups reported in the literature are smaller; we did not measure any here.  Its storage format (fp16 values + 2-bit index per kept value) is 9 bpw, or
2.96 bpw with int4 values (measured fixed-rate, `sgpt_24_int4_g64`).  On SmolLM2-135M the quality price is
prohibitive:

- one-shot SparseGPT 2:4 costs KL 1.25 (PPL16 14.7 → 50.7);
- 2:4 + int4 costs KL 1.59;
- for comparison, dense `ghcol` at 2.94 bpw has KL 0.195.

So 2:4 is a **compute** decision for large models, where it is far less damaging, usually with
sparse fine-tuning. It is never a storage win.  It does not change the storage verdict, and it composes badly
with our codec: once weights are entropy-coded, a 2:4 kernel needs them decoded first.

## FACT vs HYPOTHESIS synthesis

**FACT** (SmolLM2-135M; KL/top-1 on 16 WikiText-2 windows; chat-KL on 48 instruct answers; every bit of mask,
values and side info charged):

- At equal effective bpw, pruning never beats entropy-coded scalar quantization. This holds for
  - the base weights: magnitude, Wanda and SparseGPT; unstructured ρ ∈ {0.7, 0.5, 0.4, 0.3} and 2:4; alone, with ECSQ/GPTQ, or with int4. The best case ties at −0.02 bpw; typical losses are 0.15–0.6 bpw with quant and 4–7 bpw without;
  - the fine-tune delta: losses of 0.16–2.6 bpw (DARE up to 5.9).
- The mask costs H(ρ) ≈ 0.47–1.0 bpw for unstructured and 0.646 bpw for 2:4, and buys less than it costs, because:
  - the entropy coder already zeroes 11–53% of weights and 32–89% of Δ entries without a mask;
  - forcing more zeros hurts even with a free mask;
  - the delta is dense and nearly Gaussian (kurtosis 0.45, top-10% energy 48.8%).

**HYPOTHESIS** (untested):

- At multi-billion scale pruning damage shrinks. We still expect no storage crossover, because sparse+quant only approaches the dense coder from below, and only as ρ approaches the coder's own natural zero fraction.
- 2:4 remains a compute-only argument.
- The side finding "GPTQ on the diag(H)-column grid saves ≈0.18 bpw over plain column-step rounding" should be re-checked on Qwen2.5-0.5B before anyone relies on it.

## Caveats

- **One model.** SmolLM2-135M base and instruct, with one calibration set and seed. Small models are the least prunable, which is the main external-validity limit (see the HYPOTHESIS).
- **One-shot only.** No pruning-aware fine-tuning. Sparse fine-tuning helps pruning, but quantization-aware training would equally help the dense coder, so it is out of scope for a training-free codec.
- **Rates are entropy proxies.** Order-0 empirical entropy plus tables, the project's standard rule (no rANS was run). The CTX section shows that a context-adaptive zero-flag/mask coder changes the comparison by ≤ 0.016 bpw.
- **The interpolation fix.** The `saved_*` fields written into the JSON during the runs used a mixed-point Pareto envelope, which flatters sparse configs by up to ~0.17 bpw. The tables here use per-line interpolation from `scripts/prune_summary.py`; treat those numbers as authoritative.
