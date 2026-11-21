---
title: weight-codec
emoji: 🗜️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
short_description: Real ANS bitstreams for LLM families + decode benchmark
---

# weight-codec — phase 2: real ANS bitstreams & decode benchmark

Phase 2 of a falsification-first study of *"compress weights like JPEG (intra), model families and
training checkpoints like video (inter)"*. Phase 1 only **estimated** rates with an order-0 entropy
proxy; here every number comes from **real files** (bytes on disk, all side info included) and every
quality number from a model **decoded from those files**.

* **Results tab** — tables and figures: formats → MB → decode ms → quality (PPL / KL / chat-KL) for
  Qwen2.5-0.5B, SmolLM2-360M and SmolLM2-135M {base, Instruct}, the Pythia-410M training trajectory
  (I/P-frames) and the progressive (SVC) stream.
* **Live decode demo** — this Space holds only our compressed WCDC bitstreams for
  SmolLM2-135M → SmolLM2-135M-Instruct. Pick a format → the Space entropy-decodes the files on its
  own CPU, rebuilds the fine-tuned model (config/tokenizer from the public Hub repo; **no fine-tune
  weights are downloaded**), reports MB and decode milliseconds, and lets the rebuilt model answer.

Codec (`wcodec/`): WCDC container (zlib JSON header + sections), static-model rANS from
[constriction](https://github.com/bamler-lab/constriction) (24-bit precision), wcodec's own `rans16`
(32-bit state, 16-bit words, 15-bit frequencies; numba CPU and Metal GPU decoders), lossless fp16
coder (hi/lo byte split, optional XOR-with-reference for lossless P-frames), GPTQ / GPTQ-ECSQ
quantizers, closed-loop P-frames on a shared lattice, nested-lattice SVC layers.

## Run locally

```bash
pip install -r requirements.txt gradio==5.31.0
python app.py
```

## Deploy (from the project repo)

```bash
python phase2/summarize_p2.py && python phase2/space_summary.py   # tables + figures
python phase2/deploy_space.py --deploy                             # stages wcodec/, assets/, bitstreams/; uploads (private)
```
