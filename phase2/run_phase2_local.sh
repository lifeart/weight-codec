#!/bin/bash
# Phase 2, local part, strictly sequential (one heavy model process at a time: parallel runs
# overload RAM).  Every runner is resumable: finished configs are skipped.
# Logs go to logs/p2_*.log; the run stops before a step if free disk is below $MIN_FREE_GB (default 4).
# The HF-Jobs part (Pythia trajectory, SmolLM2-360M base stage) is launched separately with phase2/hf_job.py,
# see README.md.  usage: phase2/run_phase2_local.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MIN_FREE_GB=${MIN_FREE_GB:-4}
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p logs
free_gb() { df -Pk . | awk 'NR==2{print int($4/1048576)}'; }
step() {
  local log=$1; shift
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then echo "DISK STOP before $*: $(free_gb) GB free"; exit 1; fi
  echo "$(date +%H:%M:%S) START $*"
  "$@" > "logs/$log" 2>&1; local rc=$?
  echo "$(date +%H:%M:%S) END rc=$rc $*"
  [ $rc -eq 0 ] || { echo "step failed, see logs/$log"; exit $rc; }
}
SMOL135=(--base HuggingFaceTB/SmolLM2-135M --target HuggingFaceTB/SmolLM2-135M-Instruct)
SMOL360=(--base HuggingFaceTB/SmolLM2-360M --target HuggingFaceTB/SmolLM2-360M-Instruct)

step p2_test_cpu.log     $PY phase2/test_wcodec.py cpu
step p2_test_mps.log     $PY phase2/test_wcodec.py mps          # Apple GPU; drop this line on non-Mac machines
# family codec: {base, Instruct} -> real WCDC files + quality of the decoded models
step p2_family_qwen.log    $PY phase2/family_codec.py --refcoders ""
step p2_family_smol135.log $PY phase2/family_codec.py "${SMOL135[@]}" --tag smol135 --tok_tag smollm2 \
                             --ref_base smol135 --ref_target smol135_instruct --group 64
step p2_family_smol360.log $PY phase2/family_codec.py "${SMOL360[@]}" --tag smol360 --tok_tag smollm2 \
                             --ref_base smol360 --ref_target smol360_instruct --group 64 --refcoders ""
# progressive (SVC) stream and the scene-cut control
step p2_svc.log          $PY phase2/svc_real.py
step p2_scene.log        $PY phase2/scene_cut.py --delete_coder
# decode benchmark (CPU constriction / rans16, Metal GPU) and the zstd / gzip reference
step p2_bench_qwen.log    $PY phase2/bench_decode.py --tag qwen05b --reps 2
step p2_bench_smol135.log $PY phase2/bench_decode.py --tag smol135 "${SMOL135[@]}" --reps 3
step p2_bench_smol360.log $PY phase2/bench_decode.py --tag smol360 "${SMOL360[@]}" --reps 2
step p2_zstd_qwen.log     $PY phase2/zstd_ref.py --tag qwen05b --no_gzip
step p2_zstd_smol135.log  $PY phase2/zstd_ref.py --tag smol135
step p2_zstd_smol360.log  $PY phase2/zstd_ref.py --tag smol360 --no_gzip
# tables + figures
step p2_summary.log      $PY phase2/summarize_p2.py
step p2_space_summary.log $PY phase2/space_summary.py
echo "$(date +%H:%M:%S) PHASE 2 LOCAL DONE"
