"""Summarise Probe V1/V2: lower envelopes of log-KL vs total bpw per VQ dimension, rate saving at
equal KL vs the best d=1 (scalar) envelope, pre-registered verdict, RD figure.

usage: python scripts/vq_summary.py [--tag smol135] [--file vq_v1_smol135] [--extra vq_v2_smol135]
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="smol135")
ap.add_argument("--file", default=None)
ap.add_argument("--extra", default="")
ap.add_argument("--lo", type=float, default=2.5)
ap.add_argument("--hi", type=float, default=4.5)
args = ap.parse_args()
D = json.load(open(os.path.join(RES, (args.file or f"vq_v1_{args.tag}") + ".json")))
for e in [x for x in args.extra.split(",") if x]:
    p = os.path.join(RES, e + ".json")
    if os.path.exists(p):
        D.update({f"{k}": v for k, v in json.load(open(p)).items() if not k.startswith("_")})


def dim_of(name, rec):
    cfg = rec.get("cfg", {})
    if cfg.get("kind") in ("ecsq", "rtn", "gptq_ecsq") or name.startswith(("ecsq", "rtn", "gptq_ecsq")):
        return 1
    if cfg.get("kind") in ("lattice", "gptq_lattice", "gptq_search", "hcol_lattice"):
        return 100 + cfg.get("d")          # separate family: 104 = D4 lattice, 108 = E8 lattice
    return cfg.get("d")


def envelope(pts):
    """Lower convex hull of (bpw, log KL), decreasing part only.  Convex, because any two operating
    points can be mixed (time-sharing across tensors); a plain Pareto front with linear interpolation
    is NOT achievable-consistent (adding a mediocre Pareto point raises the interpolated front)."""
    P = sorted((b, np.log(k), n) for b, k, n in pts)
    hull = []
    for p in P:
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    # keep the decreasing part (more bits must buy lower KL)
    out = [hull[0]]
    for h in hull[1:]:
        if h[1] < out[-1][1]:
            out.append(h)
    return [(b, float(np.exp(y)), n) for b, y, n in out]


def bpw_at(front, kl):
    """bpw needed to reach `kl` on a Pareto front (piecewise linear in log KL); None if outside."""
    for (b0, k0, _), (b1, k1, _) in zip(front, front[1:]):
        if k1 <= kl <= k0:
            t = (np.log(kl) - np.log(k0)) / (np.log(k1) - np.log(k0))
            return b0 + t * (b1 - b0)
    return None


def main():
    groups = {}
    rows = []
    out = {}
    for n, r in D.items():
        if n.startswith(("_", "naive_")) or n == "fp16" or "kl" not in r:
            continue
        if n.endswith("_s1"):          # seed reruns: noise estimate only, not part of the envelopes
            base = n[:-3]
            if base in D:
                out.setdefault("seed_noise", {})[base] = {"bpw_s0": D[base]["bpw"], "kl_s0": D[base]["kl"],
                                                          "bpw_s1": r["bpw"], "kl_s1": r["kl"],
                                                          "kl_rel_diff": r["kl"] / D[base]["kl"] - 1}
            continue
        d = dim_of(n, r)
        hess = n.startswith(("gptq", "gptvq", "hw", "hcol"))
        key = (d, hess)
        b = r["bpw"]
        groups.setdefault(key, []).append((b, r["kl"], n))
        rows.append((d, hess, n, b, r.get("codebook_bpw", 0.0), r.get("idx_bpw"), r.get("scale_bpw"),
                     r["kl"], r["ppl"], r["top1"], r.get("weight_nmse_diag")))
    out.setdefault("fronts", {}); out.setdefault("savings", {})
    for key, pts in sorted(groups.items()):
        out["fronts"][f"d{key[0]}{'_hess' if key[1] else ''}"] = envelope(pts)
    for hess in (False, True):
        base = out["fronts"].get(f"d1{'_hess' if hess else ''}")
        if not base:
            continue
        for key in sorted(groups):
            if key[1] != hess or key[0] == 1:
                continue
            fk = f"d{key[0]}{'_hess' if hess else ''}"
            fr = out["fronts"][fk]
            sav = []
            for kl in np.geomspace(0.02, 5.0, 40):
                b1, bd = bpw_at(base, kl), bpw_at(fr, kl)
                if b1 is None or bd is None:
                    continue
                if not (args.lo <= b1 <= args.hi or args.lo <= bd <= args.hi):
                    continue
                sav.append({"kl": float(kl), "bpw_d1": b1, "bpw_d": bd, "saving_bpw": b1 - bd})
            out["savings"][fk] = sav
    verdict = {}
    for fk, sav in out["savings"].items():
        s = [x["saving_bpw"] for x in sav]
        if not s:
            verdict[fk] = "NO-OVERLAP"
        elif sum(x >= 0.10 for x in s) >= 2:
            verdict[fk] = "PASS"
        elif all(x < 0.05 for x in s):
            verdict[fk] = "FAIL"
        else:
            verdict[fk] = "INCONCLUSIVE"
        verdict[fk + "_saving_range"] = [min(s), max(s)] if s else None
    out["verdict_V1"] = verdict
    path = os.path.join(RES, f"vq_summary_{'v2_' if args.file and 'v2' in args.file else ''}{args.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{'name':30s} {'d':>2s} {'bpw':>6s} {'cb':>6s} {'idx':>6s} {'sc':>5s} {'KL':>7s} {'PPL':>8s} {'top1':>6s} {'nmse':>7s}")
    for d, hess, n, b, cb, ib, sb, kl, ppl, t1, nm in sorted(rows, key=lambda x: (x[0], x[1], x[3])):
        f = lambda v, w, p: f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"
        print(f"{n:30s} {d:2d} {b:6.3f} {f(cb, 6, 4)} {f(ib, 6, 3)} {f(sb, 5, 3)} {kl:7.4f} {ppl:8.2f} {t1:6.3f} {f(nm, 7, 4)}")
    for fk, sav in out["savings"].items():
        print(fk, " ".join(f"KL{x['kl']:.3f}:{x['saving_bpw']:+.3f}" for x in sav[::3]))
    print("VERDICT", json.dumps(verdict))
    for k, v in out.get("seed_noise", {}).items():
        print(f"seed noise {k}: s0 {v['bpw_s0']:.3f}b KL {v['kl_s0']:.4f} | s1 {v['bpw_s1']:.3f}b KL {v['kl_s1']:.4f} | rel {v['kl_rel_diff']:+.1%}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        colors = {1: "#3b6fb6", 2: "#d9822b", 4: "#2a9d6f", 8: "#b23a48", 104: "#8e5ea2", 108: "#111111"}
        for key, pts in sorted(groups.items()):
            d, hess = key
            pts = sorted(pts)
            lab = ({104: "D4 lattice VQ", 108: "E8 lattice VQ"}.get(d) or
                   f"d={d}{' (scalar)' if d == 1 else ' k-means/ECVQ'}{' +Hessian' if hess else ''}")
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=colors.get(d, "k"),
                       marker="s" if hess else "o", alpha=0.55)
            fr = envelope(pts)
            ax.plot([p[0] for p in fr], [p[1] for p in fr], color=colors.get(d, "k"),
                    ls="--" if hess else "-", lw=1.6, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("total bits / weight (indices + codebook + scales + tables)")
        ax.set_ylabel("KL(fp16 || quantized), WikiText-2"); ax.grid(alpha=0.3)
        ax.set_title(f"{'V2: Hessian-aware' if args.file and 'v2' in args.file else 'V1:'} vector vs scalar quantization at equal total rate ({args.tag})", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = f"vq_{'v2' if args.file and 'v2' in args.file else 'v1'}_{args.tag}.png"
        fig.savefig(os.path.join(FIG, fname), dpi=130)
        print("figure", os.path.join(FIG, fname))
    except ImportError as e:
        print("no matplotlib:", e)


main()
