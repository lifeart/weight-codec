#!/bin/sh
# Full pipeline, strictly sequential (one heavy model process at a time: parallel model jobs
# overload RAM). Every script is resumable: finished configs are skipped.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1
PY=.venv/bin/python
$PY scripts/sanity.py
$PY scripts/test_gptq.py
$PY scripts/spine.py                                     # baselines + A2 + ECSQ
# ---- falsification probes (before any codec)
$PY scripts/probe_c1.py Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct base_instruct
$PY scripts/probe_c1.py Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-Coder-0.5B base_coder
$PY scripts/probe_c1.py Qwen/Qwen2.5-Coder-0.5B Qwen/Qwen2.5-Coder-0.5B-Instruct coder_coderinstruct
$PY scripts/probe_c_pairwise.py
$PY scripts/probe_c3.py Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-Coder-0.5B base_coder
$PY scripts/probe_c3.py Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct base_instruct
$PY scripts/c4_gop_entropy.py                            # needs the two Coder models cached
$PY scripts/probe_a1.py
$PY scripts/probe_b1.py
$PY scripts/probe_d1.py
$PY scripts/a3_fisher.py                                 # A3 probe
# ---- codecs for survivors (C, A3, D) + E
$PY scripts/c2_family_codec.py
$PY scripts/c2_embedding.py
$PY scripts/a3_fisher.py --codec
$PY scripts/svc_e.py
$PY scripts/svc_e.py --ratio 3 --layers 3 --variants rtn
$PY scripts/svc_direct_rates.py
$PY scripts/d2_pythia_video.py                           # refuses unless D1 passed
$PY scripts/crossdomain_c4.py
$PY scripts/plot_all.py
$PY scripts/summarize.py
