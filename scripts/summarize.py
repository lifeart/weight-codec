"""Matched-quality summaries: for every RD curve, the bpw needed to reach a target
quality (linear interpolation of bpw vs log(metric) between bracketing points).
Writes results/summary.json and prints markdown tables."""
import json, os, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
J = lambda n: json.load(open(os.path.join(RES, n + ".json"))) if os.path.exists(os.path.join(RES, n + ".json")) else None


def bits_at(points, target):
    """points: [(bpw, metric)], metric decreasing in bpw. bpw at which metric == target."""
    pts = sorted(points)
    for (b0, m0), (b1, m1) in zip(pts, pts[1:]):
        l0, l1, lt = math.log(m0), math.log(m1), math.log(target)
        if (l0 - lt) * (l1 - lt) <= 0 and l0 != l1:
            return b0 + (b1 - b0) * (lt - l0) / (l1 - l0)
    return None


out = {}
sp = J("spine_qwen05b")
if sp:
    curves = {
        "RTN g128 fixed-rate": [(sp[k]["fixed_bpw"], sp[k]["ppl"], sp[k]["kl"]) for k in ("rtn_int4_g128", "rtn_int3_g128", "rtn_int2_g128") if k in sp],
        "RTN g128 + entropy (A2)": [(sp[k]["entropy_bpw"], sp[k]["ppl"], sp[k]["kl"]) for k in ("rtn_int4_g128", "rtn_int3_g128", "rtn_int2_g128") if k in sp],
        "GPTQ g128 fixed-rate": [(sp[k]["fixed_bpw"], sp[k]["ppl"], sp[k]["kl"]) for k in ("gptq_int4_g128", "gptq_int3_g128", "gptq_int2_g128") if k in sp],
        "GPTQ g128 + entropy (A2)": [(sp[k]["entropy_bpw"], sp[k]["ppl"], sp[k]["kl"]) for k in ("gptq_int4_g128", "gptq_int3_g128", "gptq_int2_g128") if k in sp],
        "ECSQ row-step": [(sp[k]["entropy_bpw"], sp[k]["ppl"], sp[k]["kl"]) for k in sp if k.startswith("ecsq_row")],
        "GPTQ-ECSQ row-step": [(sp[k]["entropy_bpw"], sp[k]["ppl"], sp[k]["kl"]) for k in sp if k.startswith("gptq_ecsq_row")],
    }
    a3 = J("a3_codec_qwen05b")
    if a3:
        for g in ("1.0", "0.5"):
            curves[f"A3 Fisher-step ECSQ (gamma={g})"] = [(v["entropy_bpw"], v["ppl"], v["kl"]) for k, v in a3.items() if k.startswith(f"fisher_g{g}")]
            curves[f"A3 Fisher-step GPTQ-ECSQ (gamma={g})"] = [(v["entropy_bpw"], v["ppl"], v["kl"]) for k, v in a3.items() if k.startswith(f"gptq_fisher_g{g}")]
    sv = J("svc_e_qwen05b")
    if sv:
        for var in ("rtn", "gptq"):
            curves[f"E SVC ({var} layers), one stream"] = [(v["cum_bpw"], v["ppl"], v["kl"]) for k, v in sv.items() if k.startswith(var + "_L")]
    targets = {"PPL=GPTQ-int4 (%.2f)" % sp["gptq_int4_g128"]["ppl"]: ("ppl", sp["gptq_int4_g128"]["ppl"]),
               "PPL=RTN-int4 (%.2f)" % sp["rtn_int4_g128"]["ppl"]: ("ppl", sp["rtn_int4_g128"]["ppl"]),
               "PPL=GPTQ-int3 (%.2f)" % sp["gptq_int3_g128"]["ppl"]: ("ppl", sp["gptq_int3_g128"]["ppl"]),
               "PPL=14.5": ("ppl", 14.5), "KL=0.05": ("kl", 0.05)}
    tab = {}
    for cname, pts in curves.items():
        row = {}
        for tname, (m, t) in targets.items():
            b = bits_at([(p[0], p[1] if m == "ppl" else p[2]) for p in pts], t)
            row[tname] = None if b is None else round(b, 3)
        tab[cname] = row
    out["spine_matched"] = tab
    print("| curve | " + " | ".join(targets) + " |")
    print("|---|" + "---|" * len(targets))
    for c, r in tab.items():
        print(f"| {c} | " + " | ".join("—" if v is None else f"{v:.2f}" for v in r.values()) + " |")

c2 = J("c2_qwen05b_instruct")
if c2:
    fam = {"independent RTN g128+H": "indep_rtn", "independent GPTQ g128+H": "indep_gptq_int",
           "independent GPTQ-ECSQ": "indep_gptq_ecsq", "P-frame shared grid (RTN)": "pframe_rtn",
           "P-frame RTN base + GPTQ residual": "pframe_gptq_c", "P-frame GPTQ base + GPTQ residual": "pframe_gptq2_c", "fp16 base + ECSQ(delta)": "delta_ecsq",
           "fp16 base + GPTQ(delta)": "delta_gptq", "fp16 base + low-rank delta": "delta_lowrank"}
    ref = c2.get("indep_gptq_int4_g128")
    targets = {}
    if ref:
        targets["chatKL=indepGPTQ4 (%.4f)" % ref["chat_kl"]] = ("chat_kl", ref["chat_kl"])
        targets["KL=indepGPTQ4 (%.4f)" % ref["kl"]] = ("kl", ref["kl"])
    targets["chatKL=0.01"] = ("chat_kl", 0.01)
    targets["chatKL=0.05"] = ("chat_kl", 0.05)
    tab = {}
    for name, pre in fam.items():
        pts = [v for k, v in c2.items() if k.startswith(pre)]
        if not pts: continue
        tab[name] = {t: (lambda b: None if b is None else round(b, 3))(bits_at([(p["bpw"], max(p[m], 1e-6)) for p in pts], val))
                     for t, (m, val) in targets.items()}
    out["c2_matched"] = tab
    print()
    print("| instruct coder | " + " | ".join(targets) + " |")
    print("|---|" + "---|" * len(targets))
    for c, r in tab.items():
        print(f"| {c} | " + " | ".join("—" if v is None else f"{v:.2f}" for v in r.values()) + " |")
json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
