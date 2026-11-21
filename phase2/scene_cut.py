"""Phase 2 / 'scene cut' control with REAL BYTES: is a relative a cheap P-frame?

All linears on the BASE's lattice s = c*rms(row of W_base) (RTN, c = 0.35, int4-class):
  I-frame of model X : q_X = round(W_X / s), step inherited from the base (0 side info)
  P-frame of model X : q_X - q_base  (closed loop on a shared lattice == same reconstruction)
Same reconstruction => same quality; only the bytes differ.  Qwen2.5-0.5B-Instruct (SFT/RL of the
base) vs Qwen2.5-Coder-0.5B (continued pre-training, 5.5T tokens: phase 1 measured ||Δ||/||W|| 0.88).
Rate-only (no eval): weights read lazily from the safetensors cache; the Coder snapshot is deleted
afterwards if --delete_coder.
"""
import argparse
import shutil
from p2common import *

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--others", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-Coder-0.5B")
ap.add_argument("--c", type=float, default=0.35)
ap.add_argument("--delete_coder", action="store_true")
args = ap.parse_args()
check_disk()
store = Store("phase2_scene_cut_qwen05b")
OUT = os.path.join(BITS, "qwen05b", "scene"); os.makedirs(OUT, exist_ok=True)
B = LazySD(args.base)
keys = [k for k in B.keys() if k.startswith("model.layers.") and k.endswith("proj.weight")]
ntot = sum(B[k].numel() for k in keys)
qB, sB = {}, {}
wb = Writer({"what": "base I-frame RTN on own lattice", "c": args.c})
for k in keys:
    W = B[k].to(torch.float16).float()
    sB[k] = Q.ecsq_step(W, args.c); qB[k] = Q.ecsq(W, sB[k])
    kind, p, sec = enc_qrow(qB[k], sB[k]); wb.add(k, kind, W.shape, p, sec)
nb = wb.write(os.path.join(OUT, "base_I.wcdc"))
store.put("base_I", {"bytes": nb, "bpw": 8 * nb / ntot, "n": ntot})
log(f"base I-frame {nb / 1e6:.1f} MB = {8 * nb / ntot:.4f} bpw")
for rid in args.others.split(","):
    tag = rid.split("/")[1]
    if store.has(tag):
        continue
    from huggingface_hub import snapshot_download
    snapshot_download(rid, allow_patterns=["*.safetensors", "*.json"])     # no-op if cached
    X = LazySD(rid)
    wi = Writer({"what": f"{tag} I-frame on base lattice"}); wp = Writer({"what": f"{tag} P-frame vs base"})
    num = den = 0.0
    for k in keys:
        W = X[k].to(torch.float16).float()
        q = Q.ecsq(W, sB[k])
        kind, p, sec = enc_qrow(q, step_mode="ref"); wi.add(k, kind, W.shape, p, sec)
        kind, p, sec = enc_qrow(q - qB[k], step_mode="ref", acc="add"); wp.add(k, kind, W.shape, p, sec)
        Wb = B[k].to(torch.float16).float()
        num += (W - Wb).pow(2).sum().item(); den += W.pow(2).sum().item()
    ni = wi.write(os.path.join(OUT, f"{tag}_I.wcdc")); npf = wp.write(os.path.join(OUT, f"{tag}_P.wcdc"))
    # decode check: P-frame + base == I-frame on the same lattice (bit-exact)
    base_state = {k: TState(Q=qB[k], step=sB[k]) for k in keys}
    sp, _, _ = decode_file(os.path.join(OUT, f"{tag}_P.wcdc"), ref=base_state)
    si, _, _ = decode_file(os.path.join(OUT, f"{tag}_I.wcdc"), ref={k: TState(step=sB[k]) for k in keys})
    ident = all(torch.equal(sp[k].W, si[k].W) for k in keys)
    assert ident
    r = {"rel_delta": (num / den) ** 0.5, "I_bytes": ni, "I_bpw": 8 * ni / ntot, "P_bytes": npf, "P_bpw": 8 * npf / ntot,
         "P_over_I": npf / ni, "P_equals_I_reconstruction": ident}
    store.put(tag, r)
    log(f"{tag}: ||Δ||/||W||={r['rel_delta']:.3f}  I {ni / 1e6:.1f} MB ({r['I_bpw']:.3f} bpw)  P {npf / 1e6:.1f} MB "
        f"({r['P_bpw']:.3f} bpw)  P/I={r['P_over_I']:.3f}")
    del X, sp, si; gc.collect()
    if args.delete_coder and "Coder" in rid:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == rid:
                shutil.rmtree(repo.repo_path)
                log("deleted cache of", rid)
log("done; disk free GB", round(disk_free_gb(), 2))
