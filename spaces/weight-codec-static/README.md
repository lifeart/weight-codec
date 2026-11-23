---
title: weight-codec
emoji: 🗜️
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Base once + entropy-coded delta → Instruct model, in-browser
---

# weight-codec — base once + entropy-coded delta → Instruct model, in your browser

**The idea:** a fine-tune is ~92&nbsp;% identical to its base model, so you store the base **once** and ship only the
small **difference** — never a second full model. (A training run's checkpoints compress the same way, like video frames.)
Compressing a *single* standalone model, by contrast, buys little beyond ordinary rounding. This Space shows it
end-to-end, entirely in your browser.

Static Space (no server). The page downloads the compressed `.wcdc` files of
**SmolLM2-135M → SmolLM2-135M-Instruct**, entropy-decodes them in plain JavaScript (rANS, Web Worker),
rebuilds the fp16 weights, checks their **SHA-256 against the Python decoder** (and, for the lossless parts,
against the original checkpoint), and generates text **with the rebuilt weights** through a small
JavaScript Llama forward pass. The base files are kept in the browser's Cache Storage, so a second
configuration or visit downloads only the delta.

Configurations (the codec's default coders: quantization steps per output row *and* per input column):
*fp16 base (lossless, 178 MB) + full-rank Δ (52 MB)* → chat-KL **0.012** vs the original Instruct (the previous coder
needed 53 MB for 0.030); *int4-class base I-frame (86 MB) + P-frame (44 MB)* → chat-KL **0.084** (smallest; the
previous coder gave 0.54 at the same size). Storing both models this way takes 130 MB vs 228 MB for two standard
int4 files. Lossy parts are approximations, and the page says so.

The page source is `docs/` of the project repository (the same files are served by GitHub Pages); this Space
additionally hosts `bitstreams/` (rans16 re-encodes of the measured files + the official tokenizer.json),
built by `phase2/demo_build.py` and staged/uploaded by `phase2/deploy_static_space.py`.
Results and every caveat: REPORT.md in the repository (§17 = these files and the default coders, §14 = the codec).
