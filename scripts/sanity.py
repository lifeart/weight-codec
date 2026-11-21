"""Sanity: timing + fp16-vs-fp32 numerics of the eval on MPS (first windows)."""
import sys, torch
from common import *
rid = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
tok = load_tok(rid)
ids = wt2_ids(tok, "test", "qwen25")
log("test tokens", ids.numel(), "windows", ids.numel() // SEQLEN)
out = {}
for dt in (torch.float16, torch.float32):
    m = load_model(rid, dtype=dt)
    r = eval_lm(m, ids, max_windows=8)
    log(dt, r); out[str(dt)] = r
    if dt == torch.float16:
        ref = build_ref(m, ids, 4)
        r2 = eval_lm(m, ids, max_windows=4, ref=ref); log("self-KL", r2); out["selfkl"] = r2
    del m; gc.collect(); torch.mps.empty_cache()
save_json("sanity_qwen_numerics", out)
