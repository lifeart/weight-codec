"""Launch a phase-2 run as a Hugging Face JOB (CPU, off-Mac) -- non-interactive.

The project code (wcodec/, phase2/, scripts/common.py, chat prompts) is shipped as a base64
tar.gz in an environment variable (the same mechanism `hf jobs uv run` uses for local scripts),
so no code repository is created.  The job installs CPU torch + deps, runs the given phase-2
script, prints the result JSON to the log and uploads results/ + selected bitstreams to the
dataset repo $WC_REPO (default: the author's PRIVATE lifeart/weight-codec-phase2 -- set WC_REPO to a
dataset repo you own) under runs/<run_name>/.
The HF token is passed with the standard job-secrets mechanism (never printed or written).

usage: python phase2/hf_job.py <run_name> <flavor> <timeout> <script.py> [script args...]
"""
import base64
import shlex
import io
import os
import sys
import tarfile

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi, get_token

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("WC_REPO", "lifeart/weight-codec-phase2")  # set WC_REPO to your own dataset repo
FILES = ["wcodec/__init__.py", "wcodec/ans.py", "wcodec/container.py", "wcodec/codecs.py",
         "wcodec/quant.py", "wcodec/parallel.py", "wcodec/rans16.py", "wcodec/transcode.py", "scripts/common.py", "scripts/chat_prompts.json",
         "phase2/p2common.py", "phase2/family_codec.py", "phase2/trajectory_codec.py",
         "phase2/bench_decode.py", "phase2/job_upload.py", "phase2/test_wcodec.py"]
PIP = ("pip install -q --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && "
       "pip install -q --no-cache-dir transformers==5.2.0 safetensors==0.8.0 datasets==4.5.0 "
       "huggingface_hub==1.4.1 numpy==2.2.6 scipy constriction==0.5.0 zstandard numba")


def bundle():
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for f in FILES:
            tf.add(os.path.join(ROOT, f), arcname=f)
    return base64.b64encode(bio.getvalue()).decode()


def main():
    run, flavor, timeout, script, *sargs = sys.argv[1:]
    code = bundle()
    assert len(code) < 120_000, f"code bundle too large for an env var: {len(code)}"
    cmd = ("set -e; mkdir -p /w && cd /w && echo \"$WC_CODE\" | base64 -d | tar xz && " + PIP +
           " && export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
           f" && python phase2/test_wcodec.py cpu"
           f" && (python {script} {' '.join(shlex.quote(a) for a in sargs)}; echo EXIT=$? > results/_exit_{run}.txt)"
           f" ; python phase2/job_upload.py {run}")
    api = HfApi()
    job = api.run_job(image="python:3.11-slim", command=["bash", "-c", cmd],
                      env={"WC_CODE": code, "WC_RUN": run, "WC_HARDWARE": f"hf-job-{flavor}",
                           "WC_REPO": REPO},
                      secrets={"HF_TOKEN": get_token()}, flavor=flavor, timeout=timeout,
                      labels={"project": "weight-codec-phase2", "run": run})
    print("job id:", job.id)
    print("url:", job.url)
    with open(os.path.join(ROOT, "results", f"phase2_job_{run}.json"), "w") as f:
        import json
        json.dump({"run": run, "job_id": job.id, "url": job.url, "flavor": flavor, "script": script,
                   "args": sargs, "timeout": timeout}, f, indent=1)


if __name__ == "__main__":
    main()
