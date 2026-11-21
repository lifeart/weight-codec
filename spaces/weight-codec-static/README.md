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

Static Space (no server). The page downloads the compressed `.wcdc` files of
**SmolLM2-135M → SmolLM2-135M-Instruct**, entropy-decodes them in plain JavaScript (rANS, Web Worker),
rebuilds the fp16 weights, checks their **SHA-256 against the Python decoder** (and, for the lossless parts,
against the original checkpoint), and generates text **with the rebuilt weights** through a small
JavaScript Llama forward pass. The base files are kept in the browser's Cache Storage, so a second
configuration or visit downloads only the delta.

Two configurations: *fp16 base (lossless, 178 MB) + full-rank Δ (53 MB)* → chat-KL 0.030 vs the original
Instruct; *int4-class base I-frame (85 MB) + P-frame (45 MB)* → chat-KL 0.54 (smallest, visibly degraded on
this fragile 135 M model). Lossy parts are approximations, and the page says so.

The page source is `docs/` of the project repository (the same files are served by GitHub Pages); this Space
additionally hosts `bitstreams/` (rans16 re-encodes of the measured files + the official tokenizer.json),
built by `phase2/demo_build.py` and staged/uploaded by `phase2/deploy_static_space.py`.
Results and every caveat: REPORT.md §14 in the repository.
