"""Stage (and optionally upload) the static Hugging Face Space = the browser demo in docs/ + its bitstreams.

The page source lives in docs/ (also served by GitHub Pages).  The Space gets the same files plus
the git-ignored bitstreams/ (rans16 .wcdc files + tokenizer.json) produced by phase2/demo_build.py,
and spaces/weight-codec-static/README.md as its Space card.

    .venv/bin/python phase2/demo_build.py            # once: bitstreams, manifest, precomputed answers
    node phase2/test_demo.mjs                        # seam gate: JS decode == Python decode, JS forward == PyTorch
    .venv/bin/python phase2/deploy_static_space.py   # stage into build/space/ (git-ignored) and check it
    .venv/bin/python phase2/deploy_static_space.py --deploy [--public-ok]   # + upload to WC_SPACE

Upload is non-interactive with the cached HF login (the token is never printed).  A new Space is created PRIVATE;
uploading to a Space that is already public needs --public-ok (the owner made lifeart/weight-codec public).
Before any upload: every staged bitstream must match the manifest (bytes), the staged text must pass
demo_build.public_clean (no hardware / infrastructure details), and the private results dataset must still be private.
--prune additionally deletes remote bitstreams/*.wcdc that the manifest no longer lists (separate commit)."""
import json
import os
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
DOCS = os.path.join(ROOT, "docs")
CARD = os.path.join(ROOT, "spaces", "weight-codec-static", "README.md")
OUT = os.path.join(ROOT, "build", "space")
SPACE = os.environ.get("WC_SPACE", "lifeart/weight-codec")
PRIVATE_DATASET = os.environ.get("WC_REPO", "lifeart/weight-codec-phase2")


def stage():
    from demo_build import public_clean
    man = json.load(open(os.path.join(DOCS, "data", "manifest.json")))
    bs = os.path.join(DOCS, "bitstreams")
    missing = [n for n, f in man["files"].items()
               if not os.path.exists(os.path.join(bs, n + ".wcdc")) or os.path.getsize(os.path.join(bs, n + ".wcdc")) != f["bytes"]]
    if missing or not os.path.exists(os.path.join(bs, "tokenizer.json")):
        sys.exit(f"docs/bitstreams is missing or stale ({missing or 'tokenizer.json'}): run phase2/demo_build.py first")
    extra = [f for f in os.listdir(bs) if f.endswith(".wcdc") and f[:-5] not in man["files"]]
    if extra:
        sys.exit(f"docs/bitstreams holds files the manifest does not list ({extra}): re-run phase2/demo_build.py")
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(DOCS, OUT, ignore=shutil.ignore_patterns(".nojekyll", ".DS_Store", "__pycache__"))
    shutil.copy(CARD, os.path.join(OUT, "README.md"))
    txt = [os.path.join(d, f) for d, _, fs in os.walk(OUT) for f in fs if f.endswith((".html", ".js", ".json", ".css", ".md"))
           and not (d.endswith("bitstreams") and f == "tokenizer.json")]
    print("public text check:", public_clean(txt), "staged text files clean")
    tot = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    print(f"staged {OUT}: {tot / 1e6:.1f} MB ({len(man['files'])} bitstreams + tokenizer + page)")
    return man


def deploy(man, public_ok=False, prune=False):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        ds_private = api.dataset_info(PRIVATE_DATASET).private
    except Exception as e:                       # not visible -> cannot verify: refuse rather than guess
        sys.exit(f"cannot check that {PRIVATE_DATASET} is private ({type(e).__name__}); not uploading")
    assert ds_private, f"{PRIVATE_DATASET} is NOT private -- refusing to deploy anything"
    api.create_repo(SPACE, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    info = api.space_info(SPACE)
    if not info.private and not public_ok:
        sys.exit(f"{SPACE} is public: pass --public-ok to update it (owner decision)")
    if getattr(info, "sdk", None) not in (None, "static"):
        sys.exit(f"{SPACE} is not a static Space (sdk={info.sdk})")
    ci = api.upload_folder(folder_path=OUT, repo_id=SPACE, repo_type="space",
                           commit_message="Browser demo: default coders (Fisher-grid I-frame, column-grid delta), "
                                          "decoder with per-column steps",
                           ignore_patterns=["*.pyc", "__pycache__/*", ".DS_Store"])
    print("uploaded:", getattr(ci, "commit_url", ci), "| private:", api.space_info(SPACE).private)
    if prune:
        remote = [f for f in api.list_repo_files(SPACE, repo_type="space") if f.startswith("bitstreams/") and f.endswith(".wcdc")]
        stale = [f for f in remote if f[len("bitstreams/"):-5] not in man["files"]]
        if stale:
            from huggingface_hub import CommitOperationDelete
            ci2 = api.create_commit(SPACE, repo_type="space", operations=[CommitOperationDelete(f) for f in stale],
                                    commit_message="Remove bitstreams the demo no longer uses")
            print("pruned", stale, getattr(ci2, "commit_url", ci2))
        else:
            print("prune: nothing stale")
    print(f"{PRIVATE_DATASET} private:", api.dataset_info(PRIVATE_DATASET).private)


if __name__ == "__main__":
    m = stage()
    if "--deploy" in sys.argv:
        deploy(m, public_ok="--public-ok" in sys.argv, prune="--prune" in sys.argv)
