"""E helper: single-shot ECSQ rate (entropy, incl. fp16 per-row step) at the steps used by the SVC
layers, to measure the SVC rate overhead at identical reconstruction (nested lattices reconstruct
exactly the single-shot ECSQ point; verified by identical PPL/KL in svc_e results)."""
from common import *
sd = LazySD("Qwen/Qwen2.5-0.5B")
keys = [k for k in sd.keys() if k.startswith("model.layers.") and k.endswith("proj.weight")]
out = {}
for c in (1.4, 0.7, 0.35, 0.175, 1.4 / 3, 1.4 / 9):
    bits = n = 0
    for k in keys:
        W = sd[k].float(); s = (c * W.pow(2).mean(1, keepdim=True).sqrt()).half().float()
        bits += coded_bits(torch.round(W / s).to(torch.int32))[0] + 16 * W.shape[0]; n += W.numel()
    out[f"{c:.4f}"] = bits / n
    log(f"c={c:.4f} direct ECSQ bpw={bits / n:.4f}")
save_json("svc_direct_rates_qwen05b", out)
