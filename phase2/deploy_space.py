"""Assemble and deploy the PRIVATE Hugging Face Space lifeart/weight-codec (Gradio, CPU basic).

Staging = spaces/weight-codec-demo/ (+ git-ignored wcodec/, assets/, bitstreams/ copied here):
  wcodec/          the codec package (decoder runs inside the Space)
  assets/          summary.json (tables built by summarize_p2.py) + figures
  bitstreams/smol135/*.wcdc   the demo pair's real bitstreams (the Space never downloads weights)
The Space repo is created with private=True and its privacy is re-checked after the upload.
Non-interactive; uses the cached HF login; the token is never printed.
"""
import glob
import json
import os
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "spaces", "weight-codec-demo")
SPACE = os.environ.get("WC_SPACE", "lifeart/weight-codec")
DEMO_FILES = ["B_rest_ll16", "B_ecsq_c0.35", "T_rest_xor_base", "T_pframe_c0.35", "T_rest_ll16",
              "T_gptq4_packed", "T_ecsq_c0.35", "T_delta_c0.12"]
R16_FILES = ["B_rest_ll16", "B_ecsq_c0.35", "T_rest_xor_base", "T_pframe_c0.35"]


def stage():
    for d in ("wcodec", "assets", "bitstreams"):
        shutil.rmtree(os.path.join(SP, d), ignore_errors=True)
    os.makedirs(os.path.join(SP, "wcodec"))
    for f in glob.glob(os.path.join(ROOT, "wcodec", "*.py")):
        shutil.copy(f, os.path.join(SP, "wcodec"))
    os.makedirs(os.path.join(SP, "assets"))
    summ = json.load(open(os.path.join(ROOT, "results", "phase2_space_summary.json")))
    json.dump(summ, open(os.path.join(SP, "assets", "summary.json"), "w"), indent=1)
    for sec in summ["sections"]:
        for img in sec.get("images", []):
            shutil.copy(os.path.join(ROOT, "figures", img), os.path.join(SP, "assets", img))
    bd = os.path.join(SP, "bitstreams", "smol135"); os.makedirs(bd)
    for nm in DEMO_FILES:
        shutil.copy(os.path.join(ROOT, "bitstreams", "smol135", nm + ".wcdc"), bd)
    sys.path.insert(0, ROOT)
    from wcodec.transcode import transcode
    os.makedirs(os.path.join(bd, "r16"))
    for nm in R16_FILES:     # lossless re-encode with wcodec's numba rANS (decodes bit-identically)
        transcode(os.path.join(ROOT, "bitstreams", "smol135", nm + ".wcdc"), os.path.join(bd, "r16", nm + ".wcdc"))
    tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(SP, "**", "*"), recursive=True) if os.path.isfile(f))
    print(f"staged {tot / 1e6:.1f} MB in {SP}")


def deploy():
    api = HfApi()
    url = api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    info = api.space_info(SPACE)
    assert info.private, "Space is not private -- refusing to upload"
    api.upload_folder(folder_path=SP, repo_id=SPACE, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
                      commit_message="weight-codec phase 2: results + live decode demo")
    info = api.space_info(SPACE)
    print("space:", url, "private:", info.private)


if __name__ == "__main__":
    stage()
    if "--deploy" in sys.argv:
        deploy()
