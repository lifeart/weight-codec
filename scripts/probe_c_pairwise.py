"""C (GOP structure): pairwise distances between all family members (block linears + embedding):
rel = ||W_j - W_i|| / ||W_i||, cos, and optimal-alpha residual 1-cos^2.  Finds the best
reference frame for each model."""
from common import *
M = {"base": "Qwen/Qwen2.5-0.5B", "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
     "coder": "Qwen/Qwen2.5-Coder-0.5B", "coder_instruct": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}
sel = lambda k: (k.startswith("model.layers.") and k.endswith("proj.weight")) or k == "model.embed_tokens.weight"
SD = {n: load_sd(r, keys=sel) for n, r in M.items()}
out = {}
for a in M:
    for b in M:
        if a == b: continue
        num = den = dot = nb = 0.0; e = {}
        for k in SD[a]:
            x, y = SD[a][k].double(), SD[b][k].double()
            if k == "model.embed_tokens.weight":
                e = {"emb_rel": float((y - x).norm() / x.norm())}; continue
            num += float((y - x).pow(2).sum()); den += float(x.pow(2).sum()); dot += float((x * y).sum()); nb += float(y.pow(2).sum())
        cos = dot / math.sqrt(den * nb)
        out[f"{a}->{b}"] = {"rel": math.sqrt(num / den), "cos": cos, "r_alpha": 1 - cos * cos, **e}
        log(f"{a:15s}->{b:15s}", {k: round(v, 4) for k, v in out[f"{a}->{b}"].items()})
save_json("probe_c_pairwise", out)
