"""Stage (and optionally upload) the static Hugging Face Space = the browser demo in docs/ + its bitstreams.

The page source lives in docs/ (also served by GitHub Pages).  The Space gets the same files plus
the git-ignored bitstreams/ (rans16 .wcdc files + tokenizer.json) produced by phase2/demo_build.py,
and spaces/weight-codec-static/README.md as its Space card.

    .venv/bin/python phase2/demo_build.py            # once: bitstreams, manifest, precomputed answers
    .venv/bin/python phase2/deploy_static_space.py   # stage into build/space/ (git-ignored) and check it
    .venv/bin/python phase2/deploy_static_space.py --deploy   # + upload to WC_SPACE (created PRIVATE)

Upload is non-interactive with the cached HF login and refuses to upload to a Space that is not private
(making it public is the owner's decision, done on the Hub)."""
import json
import os
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
CARD = os.path.join(ROOT, "spaces", "weight-codec-static", "README.md")
OUT = os.path.join(ROOT, "build", "space")
SPACE = os.environ.get("WC_SPACE", "lifeart/weight-codec")


def stage():
    man = json.load(open(os.path.join(DOCS, "data", "manifest.json")))
    bs = os.path.join(DOCS, "bitstreams")
    missing = [n for n, f in man["files"].items()
               if not os.path.exists(os.path.join(bs, n + ".wcdc")) or os.path.getsize(os.path.join(bs, n + ".wcdc")) != f["bytes"]]
    if missing or not os.path.exists(os.path.join(bs, "tokenizer.json")):
        sys.exit(f"docs/bitstreams is missing or stale ({missing or 'tokenizer.json'}): run phase2/demo_build.py first")
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(DOCS, OUT, ignore=shutil.ignore_patterns(".nojekyll", ".DS_Store", "__pycache__"))
    shutil.copy(CARD, os.path.join(OUT, "README.md"))
    tot = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    print(f"staged {OUT}: {tot / 1e6:.1f} MB ({len(man['files'])} bitstreams + tokenizer + page)")


def deploy():
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    assert api.space_info(SPACE).private, "Space is not private -- refusing to upload"
    api.upload_folder(folder_path=OUT, repo_id=SPACE, repo_type="space", commit_message="weight-codec browser demo",
                      ignore_patterns=["*.pyc", "__pycache__/*", ".DS_Store"])
    print("deployed https://huggingface.co/spaces/" + SPACE, "private:", api.space_info(SPACE).private)


if __name__ == "__main__":
    stage()
    if "--deploy" in sys.argv:
        deploy()
