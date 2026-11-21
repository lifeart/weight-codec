"""C1 PROBE  (family = GOP):  Delta = W_ft - W_base per tensor.
Measures: ||D||/||W||, cos, singular spectrum of D (energy in top-r, r90/r99, effective
rank) vs a shuffled-D null (same entries, structure destroyed), and the QUANTIZED-DOMAIN
conditional cost: on a shared uniform grid (step = c*rms_row(W_base)), fraction of indices
that change and entropy of the index difference H(q_ft - q_base) vs H(q_ft).

PRE-REGISTERED criteria (written before running):
  SMALL     PASS if numel-weighted median ||D||/||W|| over block linears < 0.30
            (=> >1.7 bit/weight predicted high-rate saving)
  LOW-RANK  PASS if for >50% of block linears the top 5%-of-rank singular directions hold
            >=50% of ||D||^2 AND >=2x the shuffled null
  OVERALL C1 PASS if SMALL passes (low-rank only decides which C2 coder to emphasize)
usage: python scripts/probe_c1.py base_rid ft_rid tag
"""
import sys
from common import *

base_rid, ft_rid, tag = sys.argv[1], sys.argv[2], sys.argv[3]
torch.manual_seed(0)
A = load_sd(base_rid); B = load_sd(ft_rid)
keys = [k for k in A if k in B]
RS = [1, 4, 16, 64, 128, 256]
rows = {}
t0 = time.time()
for k in keys:
    Wa, Wb = A[k], B[k]
    D = Wb - Wa
    na, nd = Wa.norm().item(), D.norm().item()
    rec = {"shape": list(Wa.shape), "numel": Wa.numel(), "rel": nd / max(na, 1e-12),
           "cos": float((Wa.flatten() @ Wb.flatten()) / (na * Wb.norm() + 1e-12)),
           "frac_exact_equal": float((D == 0).float().mean())}
    if Wa.dim() == 2:
        sv = svals(D)
        Dn = D.flatten()[torch.randperm(D.numel())].reshape(D.shape)
        svn = svals(Dn)
        r5 = max(1, int(0.05 * min(D.shape)))
        rec.update({"topr": energy_topr(sv, RS + [r5]), "topr_null": energy_topr(svn, RS + [r5]),
                    "r5": r5, "r90": rank_for(sv, 0.90), "r99": rank_for(sv, 0.99),
                    "r90_null": rank_for(svn, 0.90), "eff_rank": eff_rank(sv),
                    "eff_rank_null": eff_rank(svn), "min_dim": min(D.shape)})
        # quantized-domain conditional cost on a shared grid
        rms = Wa.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)
        for c in (0.35, 0.7):
            st = (c * rms).half().float()
            qa = torch.round(Wa / st).to(torch.int32); qb = torch.round(Wb / st).to(torch.int32)
            rec[f"q{c}"] = {"H_ft": entropy_bits(qb)[0], "H_diff": entropy_bits(qb - qa)[0],
                            "frac_changed": float((qb != qa).float().mean())}
    rows[k] = rec
log(f"{len(rows)} tensors in {time.time() - t0:.0f}s")

# ---- aggregate over block linears
lin = {k: v for k, v in rows.items() if k.startswith("model.layers.") and k.endswith(".weight") and len(v["shape"]) == 2}
w = np.array([v["numel"] for v in lin.values()], float)
rel = np.array([v["rel"] for v in lin.values()])
o = np.argsort(rel); cw = np.cumsum(w[o]) / w.sum()
wmed_rel = float(rel[o][np.searchsorted(cw, 0.5)])
lowrank_hits = [v["topr"][v["r5"]] >= 0.5 and v["topr"][v["r5"]] >= 2 * v["topr_null"][v["r5"]] for v in lin.values()]
agg = {"wmedian_rel": wmed_rel, "wmean_rel2": float(np.sqrt((w * rel ** 2).sum() / w.sum())),
       "frac_lowrank": float(np.mean(lowrank_hits)),
       "mean_top5pct_energy": float(np.mean([v["topr"][v["r5"]] for v in lin.values()])),
       "mean_top5pct_energy_null": float(np.mean([v["topr_null"][v["r5"]] for v in lin.values()])),
       "mean_r90_over_min": float(np.mean([v["r90"] / v["min_dim"] for v in lin.values()])),
       "mean_effrank_over_min": float(np.mean([v["eff_rank"] / v["min_dim"] for v in lin.values()])),
       "predicted_saving_bits_highrate": float(-np.log2(max(wmed_rel, 1e-9)))}
for c in (0.35, 0.7):
    agg[f"q{c}_H_ft_bpw"] = float(sum(v["numel"] * v[f"q{c}"]["H_ft"] for v in lin.values()) / w.sum())
    agg[f"q{c}_H_diff_bpw"] = float(sum(v["numel"] * v[f"q{c}"]["H_diff"] for v in lin.values()) / w.sum())
    agg[f"q{c}_frac_changed"] = float(sum(v["numel"] * v[f"q{c}"]["frac_changed"] for v in lin.values()) / w.sum())
for k in rows:
    if "embed" in k or "lm_head" in k:
        agg[f"rel[{k}]"] = rows[k]["rel"]
agg["SMALL"] = "PASS" if wmed_rel < 0.30 else "FAIL"
agg["LOW_RANK"] = "PASS" if agg["frac_lowrank"] > 0.5 else "FAIL"
agg["C1"] = agg["SMALL"]
log(json.dumps(agg, indent=1))
save_json(f"probe_c1_{tag}", {"base": base_rid, "ft": ft_rid, "aggregate": agg, "tensors": rows})
