"""Summarise Probe V3 (VQ x delta / family coding): chat-KL RD envelopes of the Delta coders (V3a),
pair-rate table of the shared-codebook family (V3b), pre-registered verdicts, figure.

usage: python scripts/vq_v3_summary.py [--tag smol135]
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="smol135")
args = ap.parse_args()
D = json.load(open(os.path.join(RES, f"vq_v3_{args.tag}.json")))


def front(pts):
    """Lower convex hull of (bpw, log KL), decreasing part (see vq_summary.envelope)."""
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
    out = [hull[0]]
    for h in hull[1:]:
        if h[1] < out[-1][1]:
            out.append(h)
    return [(b, float(np.exp(y)), n) for b, y, n in out]


def bpw_at(fr, y):
    for (b0, k0, _), (b1, k1, _) in zip(fr, fr[1:]):
        if k1 <= y <= k0:
            t = (np.log(y) - np.log(k0)) / (np.log(k1) - np.log(k0))
            return b0 + t * (b1 - b0)
    return None


def group(n, r):
    if r.get("kind") == "dscal":
        return "scalar"
    if r.get("kind") == "dgptq":
        return "scalar_gptq"     # Hessian-aware scalar Delta (phase-1/2 best Delta coder)
    if r.get("kind") == "dgptq_lat":
        return "lat_d4_gptq"
    if r.get("kind") == "dlat":
        return f"lat_{r['lat']}"
    if r.get("kind") == "decvq":
        return f"vq_d{r['d']}" + ("" if r.get("nn", False) else "_rdo")
    return None


out = {"v3a": {}, "v3b": []}
G = {}
for n, r in D.items():
    if n.endswith("_s1"):                     # seed reruns: noise estimate, not in the envelopes
        continue
    g = group(n, r) if isinstance(r, dict) else None
    if g and "chat_kl" in r:
        G.setdefault(g, []).append((r["bpw"], r["chat_kl"], n))
fr = {g: front(p) for g, p in G.items()}
out["v3a"]["fronts"] = fr
sav = {}
for g in fr:
    if g == "scalar" or "scalar" not in fr:
        continue
    ref = "scalar_gptq" if g.endswith("_gptq") else "scalar"
    if ref not in fr or g == ref:
        continue
    s = []
    for y in np.geomspace(0.005, 0.5, 30):
        b1, bd = bpw_at(fr[ref], y), bpw_at(fr[g], y)
        if b1 is not None and bd is not None:
            s.append({"chat_kl": float(y), "bpw_scalar": b1, "bpw_vq": bd, "saving_bpw": b1 - bd})
    sav[g] = s
out["v3a"]["savings"] = sav
ver = {}
for g, s in sav.items():
    x = [e["saving_bpw"] for e in s]
    ver[g] = ("NO-OVERLAP" if not x else "PASS" if sum(v >= 0.10 for v in x) >= 2 else
              "FAIL" if all(v < 0.05 for v in x) else "INCONCLUSIVE")
    ver[g + "_range"] = [min(x), max(x)] if x else None
out["v3a"]["verdict"] = ver
# cross-check vs the project's best Delta coder (scalar + GPTQ), on BOTH metrics
cross = {}
for metric in ("chat_kl", "kl"):
    Gm = {}
    for n, r in D.items():
        if n.endswith("_s1") or not isinstance(r, dict) or metric not in r:
            continue
        g = group(n, r)
        if g:
            Gm.setdefault(g, []).append((r["bpw"], r[metric], n))
    frm = {g: front(p) for g, p in Gm.items()}
    if "scalar_gptq" not in frm:
        continue
    for g in ("vq_d2", "vq_d4", "lat_d4", "lat_e8", "scalar"):
        if g not in frm:
            continue
        s = []
        for y in np.geomspace(0.003, 0.5, 40):
            b1, bd = bpw_at(frm["scalar_gptq"], y), bpw_at(frm[g], y)
            if b1 is not None and bd is not None:
                s.append(b1 - bd)
        cross[f"{g}_vs_scalar_gptq_{metric}"] = [min(s), max(s)] if s else None
out["v3a"]["vs_scalar_gptq"] = cross
print("vs scalar+GPTQ Delta (bpw saved, range):", json.dumps({k: [round(x, 3) for x in v] if v else None for k, v in cross.items()}))
print(f"{'V3a config':26s} {'bpw':>6s} {'cb':>6s} {'esc':>6s} {'KL':>7s} {'chatKL':>7s} {'chat_top1':>9s}")
for g, p in sorted(G.items()):
    for b, y, n in sorted(p):
        r = D[n]
        print(f"{n:26s} {b:6.3f} {r.get('codebook_bpw', 0):6.4f} {r.get('n_escaped_vectors', 0):6d} {r['kl']:7.4f} {y:7.4f} {r['chat_top1']:9.3f}")
for g, s in sav.items():
    print(g, " ".join(f"cKL{e['chat_kl']:.3f}:{e['saving_bpw']:+.3f}" for e in s[::3]))
print("V3a VERDICT", json.dumps(ver))
for n, r in D.items():
    if isinstance(r, dict) and r.get("kind") in ("pscal", "pvq", "plat"):
        out["v3b"].append({"name": n, **{k: r.get(k) for k in ("kind", "d", "lam", "c", "base_bpw", "inst_cond_bpw",
                                                               "inst_indep_bpw", "pair_bpw", "codebook_bpw", "p_same",
                                                               "p_miss", "p_inst_escape", "base_kl", "kl", "chat_kl",
                                                               "chat_top1")}})
if out["v3b"]:
    print(f"\n{'V3b config':22s} {'base':>6s} {'inst|base':>9s} {'inst-alone':>10s} {'pair':>6s} {'cb':>6s} {'p_same':>6s} {'baseKL':>7s} {'instKL':>7s} {'chatKL':>7s}")
    for e in sorted(out["v3b"], key=lambda e: e["pair_bpw"]):
        f = lambda v, w=6, p=3: f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"
        print(f"{e['name']:22s} {f(e['base_bpw'])} {f(e['inst_cond_bpw'], 9)} {f(e['inst_indep_bpw'], 10)} {f(e['pair_bpw'])} "
              f"{f(e['codebook_bpw'], 6, 4)} {f(e['p_same'])} {f(e['base_kl'], 7, 4)} {f(e['kl'], 7, 4)} {f(e['chat_kl'], 7, 4)}")
json.dump(out, open(os.path.join(RES, f"vq_v3_summary_{args.tag}.json"), "w"), indent=1)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    col = {"scalar": "#3b6fb6", "scalar_gptq": "#6fa8dc", "lat_d4_gptq": "#c27ba0", "vq_d1": "#7f7f7f", "vq_d2": "#d9822b", "vq_d4": "#2a9d6f",
           "lat_e8": "#111111", "lat_d4": "#8e5ea2", "vq_d1_rdo": "#bbbbbb"}
    for g, p in sorted(G.items()):
        p = sorted(p)
        ax.scatter([x[0] for x in p], [x[1] for x in p], s=14, color=col.get(g, "k"), alpha=0.5)
        f_ = front(p)
        ax.plot([x[0] for x in f_], [x[1] for x in f_], color=col.get(g, "k"), lw=1.6,
                label={"scalar": "scalar Δ (ECSQ)", "scalar_gptq": "scalar Δ + GPTQ", "lat_d4_gptq": "D4 lattice Δ + GPTQ", "lat_e8": "E8 lattice Δ", "lat_d4": "D4 lattice Δ",
                       "vq_d1_rdo": "ECVQ Δ d=1 (RDO)"}.get(g, f"ECVQ Δ (nearest), d={g[-1]}"))
    z = D.get("zero_bits_base")
    if z:
        ax.axhline(z["chat_kl"], color="0.6", ls=":", lw=1, label="instruct := base (0 bits)")
    ax.set_yscale("log"); ax.set_xlabel("Δ rate: bits / weight (all side info + codebooks charged)")
    ax.set_ylabel("chat-KL(fp16 instruct || decoded)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"V3a: vector vs scalar quantization of the fine-tune delta ({args.tag}, fp16 base given)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"vq_v3a_{args.tag}.png"), dpi=130)
    print("figure", os.path.join(FIG, f"vq_v3a_{args.tag}.png"))
except ImportError as e:
    print("no matplotlib", e)
