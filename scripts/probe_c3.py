"""C3: is a fine-tune neuron-permuted relative to its base?  (alignment base <-> ft)
MLP: Hungarian matching of ft neurons to base neurons on [gate_j, up_j, down_:,j] cosine.
  -> fraction of neurons matched to themselves; residual energy of Delta before/after.
Attn V/O: per-KV-group orthogonal Procrustes; report ||R - I||_F / sqrt(64).
usage: python scripts/probe_c3.py base_rid ft_rid tag
"""
import sys
from common import *
from scipy.optimize import linear_sum_assignment

base_rid, ft_rid, tag = sys.argv[1], sys.argv[2], sys.argv[3]
sel = lambda k: k.startswith("model.layers.") and k.endswith("proj.weight")
A = load_sd(base_rid, keys=sel); B = load_sd(ft_rid, keys=sel)
out = {}
for L in range(24):
    pre = f"model.layers.{L}.mlp."
    def feats(sd):
        F = torch.cat([sd[pre + "gate_proj.weight"], sd[pre + "up_proj.weight"], sd[pre + "down_proj.weight"].T], 1).double()
        return F / F.norm(dim=1, keepdim=True)
    S = (feats(B) @ feats(A).T).numpy()
    r, c = linear_sum_assignment(-S)
    ident = float((c == np.arange(len(c))).mean())
    rec = {"mlp_frac_identity": ident, "mlp_mean_diag_cos": float(np.diag(S).mean()),
           "mlp_mean_matched_cos": float(S[r, c].mean())}
    # residual of Delta for gate with identity vs matched alignment
    g_a, g_b = A[pre + "gate_proj.weight"].double(), B[pre + "gate_proj.weight"].double()
    rec["gate_rel_delta_identity"] = float((g_b - g_a).norm() / g_a.norm())
    rec["gate_rel_delta_matched"] = float((g_b - g_a[torch.as_tensor(c)]).norm() / g_a.norm())
    # V/O Procrustes rotation distance from identity
    vo = []
    va, vb = A[f"model.layers.{L}.self_attn.v_proj.weight"].double(), B[f"model.layers.{L}.self_attn.v_proj.weight"].double()
    oa, ob = A[f"model.layers.{L}.self_attn.o_proj.weight"].double(), B[f"model.layers.{L}.self_attn.o_proj.weight"].double()
    for g in range(2):
        Fa = torch.cat([va[g * 64:(g + 1) * 64]] + [oa[:, (g * 7 + i) * 64:(g * 7 + i + 1) * 64].T for i in range(7)], 1)
        Fb = torch.cat([vb[g * 64:(g + 1) * 64]] + [ob[:, (g * 7 + i) * 64:(g * 7 + i + 1) * 64].T for i in range(7)], 1)
        U, _, Vt = torch.linalg.svd(Fb @ Fa.T)
        vo.append(float((U @ Vt - torch.eye(64, dtype=U.dtype)).norm() / 8.0))
    rec["vo_rot_dist_from_I"] = vo
    out[L] = rec
    log(tag, L, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in rec.items()})
agg = {"mean_frac_identity": float(np.mean([v["mlp_frac_identity"] for v in out.values()])),
       "min_frac_identity": float(np.min([v["mlp_frac_identity"] for v in out.values()])),
       "mean_vo_rot_dist": float(np.mean([np.mean(v["vo_rot_dist_from_I"]) for v in out.values()]))}
log(agg)
save_json(f"probe_c3_{tag}", {"base": base_rid, "ft": ft_rid, "aggregate": agg, "layers": out})
