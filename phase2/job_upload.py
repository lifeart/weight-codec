"""Inside an HF Job: print results to the log and upload results/ + bitstreams to the private
dataset repo (token comes from the HF_TOKEN job secret via huggingface_hub; never printed)."""
import glob
import json
import os
import sys

from huggingface_hub import HfApi

run = sys.argv[1]
repo = os.environ.get("WC_REPO", "lifeart/weight-codec-phase2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.json")) + glob.glob(os.path.join(ROOT, "results", "_exit_*"))):
    print("=== RESULT FILE", os.path.basename(f), "===")
    with open(f) as fh:
        print(fh.read()[:200000])
    print("=== END", os.path.basename(f), "===", flush=True)
api = HfApi()
api.upload_folder(folder_path=os.path.join(ROOT, "results"), path_in_repo=f"runs/{run}/results", repo_id=repo,
                  repo_type="dataset", allow_patterns=["*.json", "*.txt"], commit_message=f"{run}: results")
bdir = os.path.join(ROOT, "bitstreams")
if os.path.isdir(bdir) and os.environ.get("WC_UPLOAD_BITS", "1") == "1":
    api.upload_folder(folder_path=bdir, path_in_repo=f"runs/{run}/bitstreams", repo_id=repo, repo_type="dataset",
                      allow_patterns=["**/IPPP_c0.35_*.wcdc", "**/IPPP_c0.7_*.wcdc", "**/allI_c0.35_*.wcdc",
                                      "**/allI_c0.7_*.wcdc", "**/T_pframe_c0.35.wcdc", "**/B_ecsq_c0.35.wcdc",
                                      "**/T_delta_c0.12.wcdc"],
                      commit_message=f"{run}: bitstreams")
print("uploaded to", repo, "runs/" + run, flush=True)
