"""All figures from results/*.json (skips whatever is missing).  Static PNGs (matplotlib).
Palette: reference categorical slots (fixed order); baselines in recessive greys."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
G1, G2, INK, INK2 = "#8c8b86", "#52514e", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def J(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def line(ax, xs, ys, color, label, ls="-", marker="o", lw=2, ms=6, z=3, labels=None):
    o = np.argsort(xs)
    xs, ys = np.array(xs)[o], np.array(ys)[o]
    ax.plot(xs, ys, ls, color=color, lw=lw, marker=marker, ms=ms, label=label, zorder=z,
            markeredgecolor="white", markeredgewidth=1.2)


def save(fig, name):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for ax in fig.axes:
        if ax.get_yscale() == "log":
            from matplotlib.ticker import NullFormatter
            lo, hi = ax.get_ylim()
            ax.yaxis.set_major_formatter(fmt)
            if hi / max(lo, 1e-12) < 30:          # short range: label minor ticks too
                ax.yaxis.set_minor_formatter(fmt); ax.tick_params(axis="y", which="minor", labelsize=7)
            else:
                ax.yaxis.set_minor_formatter(NullFormatter())
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------ spine RD
sp = J("spine_qwen05b")
if sp:
    fp = sp["fp16"]
    for metric, ylab, logy, zoom in (("ppl", "WikiText-2 PPL (lower = better)", True, False),
                                     ("ppl", "WikiText-2 PPL (lower = better)", True, True),
                                     ("kl", "KL(fp16 || q) nats/token", True, False)):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        def pts(prefix, key, must=""):
            ks = [k for k in sp if k.startswith(prefix) and must in k]
            return [sp[k][key] for k in ks], [max(sp[k][metric], 1e-4) for k in ks], ks
        x, y, _ = pts("rtn_int", "fixed_bpw", "g128"); line(ax, x, y, G1, "RTN g128 fixed-rate", ls="--", marker="s")
        x, y, _ = pts("rtn_int", "entropy_bpw", "g128"); line(ax, x, y, S[1], "RTN g128 + entropy (A2)", marker="s")
        x, y, _ = pts("gptq_int", "fixed_bpw"); line(ax, x, y, G2, "GPTQ g128 fixed-rate", ls="--")
        x, y, _ = pts("gptq_int", "entropy_bpw"); line(ax, x, y, S[0], "GPTQ g128 + entropy (A2)")
        x, y, _ = pts("ecsq_row", "entropy_bpw"); line(ax, x, y, S[2], "ECSQ (uniform step, entropy-coded)", marker="D")
        x, y, _ = pts("gptq_ecsq_row", "entropy_bpw"); line(ax, x, y, S[6], "GPTQ-ECSQ", marker="D")
        if not zoom:
            x, y, _ = pts("rtn_int", "fixed_bpw", "_pc")
            ax.scatter(x, y, marker="P", s=55, color=G1, zorder=4, label="RTN per-channel (int8 / int4), fixed-rate")
        a3 = J("a3_codec_qwen05b")
        if a3:
            ks = [k for k in a3 if k.startswith("fisher_g1.0")]
            if ks: line(ax, [a3[k]["entropy_bpw"] for k in ks], [max(a3[k][metric], 1e-4) for k in ks], S[4], "Fisher-step ECSQ (A3)", marker="^")
            ks = [k for k in a3 if k.startswith("gptq_fisher_g1.0")]
            if ks: line(ax, [a3[k]["entropy_bpw"] for k in ks], [max(a3[k][metric], 1e-4) for k in ks], S[3], "Fisher-step GPTQ-ECSQ (A3)", marker="^")
            ks = [k for k in a3 if k.startswith("gptq_fisher_g0.5")]
            if ks: line(ax, [a3[k]["entropy_bpw"] for k in ks], [max(a3[k][metric], 1e-4) for k in ks], S[3], "Fisher^0.5-step GPTQ-ECSQ (A3, gamma=0.5)", marker="^", ls="--")
        sv = None   # SVC has its own figure (rd_svc_e.png)
        if sv:
            for var, col, lab in (("rtn", S[7], "SVC one stream, RTN layers (E)"), ("gptq", S[5], "SVC one stream, GPTQ layers (E)")):
                ks = sorted([k for k in sv if k.startswith(var + "_L")])
                if ks: line(ax, [sv[k]["cum_bpw"] for k in ks], [max(sv[k][metric], 1e-4) for k in ks], col, lab, ls=":", marker="v")
            ks = sorted([k for k in sv if k.startswith("r3_rtn_L")])
            if ks: line(ax, [sv[k]["cum_bpw_ctx"] for k in ks], [max(sv[k][metric], 1e-4) for k in ks], S[7], "SVC one stream, nested x3 lattice + ctx (E)", ls="-.", marker="P")
        if metric == "ppl":
            ax.axhline(fp["ppl"], color=G1, lw=1, ls=":")
            ax.text(4.35 if zoom else 8.1, fp["ppl"] * 1.005, f"fp16 {fp['ppl']:.2f}", color=INK2, fontsize=9, va="bottom", ha="right" if zoom else "left")
        if logy: ax.set_yscale("log")
        if zoom:
            ax.set_xlim(2.3, 4.4); ax.set_ylim(fp["ppl"] * 0.98, 24)
        else:
            ax.set_xlim(1.5, 8.6)
            if metric == "ppl": ax.set_ylim(fp["ppl"] * 0.97, 120)
        ax.set_xlabel("effective bits / weight (block linears, incl. side info)"); ax.set_ylabel(ylab)
        ax.set_title(f"Qwen2.5-0.5B: rate-distortion, {'PPL' if metric == 'ppl' else 'KL'} vs bpw" + (" (zoom)" if zoom else ""), color=INK, loc="left")
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        save(fig, f"rd_spine_{metric}{'_zoom' if zoom else ''}.png")

    sv = J("svc_e_qwen05b")
    if sv:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        x = [sp[k]["entropy_bpw"] for k in sp if k.startswith("ecsq_row")]; y = [sp[k]["ppl"] for k in sp if k.startswith("ecsq_row")]
        line(ax, x, y, G1, "single-shot ECSQ (one file per point)", ls="--", marker="D")
        x = [sp[k]["entropy_bpw"] for k in sp if k.startswith("gptq_ecsq")]; y = [sp[k]["ppl"] for k in sp if k.startswith("gptq_ecsq")]
        line(ax, x, y, G2, "single-shot GPTQ-ECSQ (one file per point)", ls="--", marker="D")
        for pre, key, col, lab, ls, mk in (("rtn_L", "cum_bpw", S[1], "one stream, x2 refinement, RTN", "-", "v"),
                                           ("gptq_L", "cum_bpw", S[0], "one stream, x2 refinement, GPTQ", "-", "v"),
                                           ("r3_rtn_L", "cum_bpw_ctx", S[2], "one stream, nested x3 lattice + context", "-", "P")):
            ks = sorted([k for k in sv if k.startswith(pre)])
            if ks: line(ax, [sv[k][key] for k in ks], [sv[k]["ppl"] for k in ks], col, lab, ls=ls, marker=mk)
        ax.axhline(fp["ppl"], color=G1, lw=1, ls=":")
        ax.set_yscale("log"); ax.set_xlim(1.4, 7.2); ax.set_ylim(fp["ppl"] * 0.97, 60)
        ax.set_xlabel("cumulative bits / weight decoded"); ax.set_ylabel("WikiText-2 PPL")
        ax.set_title("E: progressive (SVC) stream vs separate files, Qwen2.5-0.5B", color=INK, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        save(fig, "rd_svc_e.png")

# ------------------------------------------------------------------ C2 instruct RD
c2 = J("c2_qwen05b_instruct")
if c2:
    panels = [("base stored losslessly (fp16)", [("delta_ecsq", S[2], "fp16 base + ECSQ(Δ)", "D"),
                                                 ("delta_gptq", S[5], "fp16 base + GPTQ(Δ)", "D"),
                                                 ("delta_lowrank", S[7], "fp16 base + low-rank Δ (LoRA-style)", "^")]),
              ("base stored quantized (same grid as instruct)", [("pframe_rtn", S[1], "P: instruct on base's RTN-ECSQ grid", "s"),
                                                                   ("pframe_gptq_c", S[0], "P: RTN-ECSQ base + GPTQ residual", "o"),
                                                                   ("pframe_gptq2_c", S[3], "P: GPTQ-ECSQ base + GPTQ residual", "o")])]
    indep = [("indep_rtn", G1, "independent RTN g128 + entropy", "s"),
             ("indep_gptq_int", G2, "independent GPTQ g128 + entropy", "o"),
             ("indep_gptq_ecsq", S[6], "independent GPTQ-ECSQ", "D")]
    for metric, ylab in (("chat_kl", "chat KL(instruct fp16 || reconstruction)"), ("kl", "WikiText KL(instruct fp16 || reconstruction)"),
                         ("ppl", "WikiText-2 PPL")):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
        for ax, (ptitle, groups) in zip(axes, panels):
            for pre, col, lab, mk in indep:
                ks = [k for k in c2 if k.startswith(pre)]
                if ks: line(ax, [c2[k]["bpw"] for k in ks], [max(c2[k][metric], 1e-4) for k in ks], col, lab, marker=mk, ls="--")
            for pre, col, lab, mk in groups:
                ks = [k for k in c2 if k.startswith(pre)]
                if ks: line(ax, [c2[k]["bpw"] for k in ks], [max(c2[k][metric], 1e-4) for k in ks], col, lab, marker=mk)
            if "zero_bits_base" in c2:
                z = c2["zero_bits_base"]; ax.scatter([0.0], [max(z[metric], 1e-4)], color=INK, marker="x", s=60, zorder=5)
                ax.annotate("instruct := base (0 bits)", (0.0, max(z[metric], 1e-4)), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
            if metric == "ppl" and "fp16" in c2:
                ax.axhline(c2["fp16"]["ppl"], color=G1, lw=1, ls=":")
            ax.set_yscale("log"); ax.set_xlim(-0.15, 4.0)
            ax.set_title(ptitle, color=INK, loc="left", fontsize=10)
            ax.set_xlabel("bits / weight of the INSTRUCT stream")
            ax.legend(fontsize=7.5, loc="upper right")
        if metric == "ppl":
            axes[0].set_ylim(c2["fp16"]["ppl"] * 0.97, 60)
        axes[0].set_ylabel(ylab)
        fig.suptitle("C2: Qwen2.5-0.5B-Instruct coded given Qwen2.5-0.5B (block linears; base cost excluded)", x=0.01, ha="left", color=INK)
        save(fig, f"rd_c2_instruct_{metric}.png")

# ------------------------------------------------------------------ probes
a1 = J("probe_a1")
if a1:
    vs = ["raw", "norm", "spectral", "tsp", "hadamard"]
    real = [k for k in a1 if k.startswith("L")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    w = 0.2
    for i, (lab, keys, col) in enumerate((("Qwen weight matrices (12)", real, S[0]),
                                          ("iid Gaussian null", [k for k in a1 if k.startswith("NULL")], G1),
                                          ("smooth AR(1) field, rows/cols shuffled (positive control)", ["POS_ar1_shuffled_896x896"], S[1]))):
        vals = [np.median([a1[k][v]["ecsq"]["delta_bits"] for k in keys]) for v in vs]
        ax.bar(np.arange(len(vs)) + (i - 1) * w, vals, w * 0.9, color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.axhline(-0.10, color=S[7], lw=1, ls=":")
    ax.text(4.45, -0.13, "pass line -0.10", fontsize=8, color=INK2, ha="right")
    ax.set_xticks(range(len(vs))); ax.set_xticklabels(["raw order", "norm sort", "spectral", "greedy TSP", "Hadamard"])
    ax.set_ylabel("bits/weight: 8x8 DCT minus no transform\n(ECSQ, equal step; < 0 = DCT wins)")
    ax.set_title("A1: block-DCT energy compaction on weights (median over matrices)", color=INK, loc="left")
    ax.set_yscale("symlog", linthresh=0.05); ax.legend(fontsize=8, loc="lower left")
    save(fig, "probe_a1_dct.png")

b1 = J("probe_b1")
if b1:
    ks = [k for k in b1 if "->" in k]
    x = [int(k.split("->")[0]) for k in ks]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for p, col, lab in (("mlp.gate_proj", S[0], "gate_proj"), ("mlp.down_proj", S[1], "down_proj"), ("self_attn.v_proj", S[2], "v_proj")):
        line(ax, x, [b1[k][p]["raw"]["r_alpha"] for k in ks], col, f"{lab} raw", ls="--", marker="o", ms=4)
        line(ax, x, [b1[k][p]["aligned"]["r_alpha"] for k in ks], col, f"{lab} aligned (Hungarian / Procrustes)", ms=4)
        line(ax, x, [b1[k][p]["aligned_null"]["r_alpha"] for k in ks], G1, None, ls=":", marker=None)
    ax.plot([], [], ":", color=G1, label="same alignment vs Gaussian null")
    ax.axhline(0.70, color=S[7], lw=1, ls=":"); ax.text(0, 0.71, "pass line 0.70", fontsize=8, color=INK2)
    ax.set_ylim(0.6, 1.02); ax.set_xlabel("block k (predict k+1 from k)")
    ax.set_ylabel("residual energy ratio\n||W[k+1] - a*P(W[k])||² / ||W[k+1]||²")
    ax.set_title("B1/B3: inter-layer prediction with motion compensation (Qwen2.5-0.5B)", color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    save(fig, "probe_b1_layers.png")

c1 = J("probe_c1_base_instruct")
if c1:
    T = c1["tensors"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, t in enumerate(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
        ks = sorted([k for k in T if k.endswith(t + ".weight")], key=lambda k: int(k.split(".")[2]))
        line(ax, [int(k.split(".")[2]) for k in ks], [T[k]["rel"] for k in ks], S[i], t, ms=4)
    ax.set_xlabel("block"); ax.set_ylabel("||W_instruct - W_base|| / ||W_base||")
    ax.set_title("C1: size of the instruct delta per matrix", color=INK, loc="left"); ax.legend(fontsize=8, ncol=4)
    save(fig, "probe_c1_rel_delta.png")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (tag, lab) in enumerate((("probe_c1_base_instruct", "base -> instruct"), ("probe_c1_base_coder", "base -> coder"),
                                    ("probe_c1_coder_coderinstruct", "coder -> coder-instruct"))):
        d = J(tag)
        if not d: continue
        Tt = d["tensors"]
        ks = [k for k in Tt if k.startswith("model.layers.") and "topr" in Tt[k]]
        rs = [1, 4, 16, 64, 128, 256]
        m = [np.mean([Tt[k]["topr"][str(r)] for k in ks]) for r in rs]
        mn = [np.mean([Tt[k]["topr_null"][str(r)] for k in ks]) for r in rs]
        line(ax, rs, m, S[i], lab)
        if i == 0: line(ax, rs, mn, G1, "shuffled-delta null (no structure)", ls=":", marker=None)
    ax.set_xscale("log", base=2); ax.set_xlabel("rank r"); ax.set_ylabel("fraction of ||Delta||^2 in top-r singular directions")
    ax.set_title("C1: is the fine-tune delta low-rank? (mean over 168 block linears)", color=INK, loc="left"); ax.legend(fontsize=8)
    save(fig, "probe_c1_spectrum.png")

d1 = J("probe_d1")
if d1:
    ks = [k for k in d1 if k.startswith("step")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    g1000 = [k for k in ks if d1[k]["aggregate"]["gap"] == 1000]
    other = [k for k in ks if d1[k]["aggregate"]["gap"] != 1000]
    xs = lambda kk: [int(k.split("->")[1][4:]) for k in kk]
    line(ax, xs(other), [d1[k]["aggregate"]["q0.35_H_diff"] / d1[k]["aggregate"]["q0.35_H_t"] for k in other], S[1], "log-grid neighbours (gap varies)")
    line(ax, xs(g1000), [d1[k]["aggregate"]["q0.35_H_diff"] / d1[k]["aggregate"]["q0.35_H_t"] for k in g1000], S[0], "1000-step neighbours")
    ax.axhline(0.5, color=S[7], lw=1, ls=":"); ax.text(2, 0.51, "pass line 0.5", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_xlabel("training step of the target checkpoint"); ax.set_ylim(0, 1.1)
    ax.set_ylabel("P-frame / I-frame bits (int4-class grid)")
    ax.set_title("D1: Pythia-410M checkpoint deltas - P-frame cost relative to I-frame", color=INK, loc="left"); ax.legend(fontsize=8)
    save(fig, "probe_d1_ckpt.png")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    line(ax, xs(g1000), [d1[k]["aggregate"]["rel_rms"] for k in g1000], S[0], "1000-step delta, block linears")
    line(ax, xs(other), [d1[k]["aggregate"]["rel_rms"] for k in other], S[1], "log-grid neighbours")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training step of the target checkpoint")
    ax.set_ylabel("||W_t - W_s|| / ||W_t||"); ax.set_title("D1: relative size of checkpoint deltas", color=INK, loc="left"); ax.legend(fontsize=8)
    save(fig, "probe_d1_rel.png")

d2 = J("d2_pythia")
if d2:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, w in zip(axes, ("late", "mid")):
        r = d2.get(f"{w}_summary")
        if not r: continue
        st = [x // 1000 for x in r["steps"]]
        L = r["lossy"]["0.35"]
        ax.bar(np.arange(len(st)) - 0.27, L["I"], 0.26, color=G1, label="all-I (independent)")
        ax.bar(np.arange(len(st)), L["K4"], 0.26, color=S[1], label="I every 4 + closed-loop P")
        ax.bar(np.arange(len(st)) + 0.27, L["IPPP"], 0.26, color=S[0], label="I + closed-loop P...P")
        ax.set_xticks(range(len(st))); ax.set_xticklabels([f"{x}k" for x in st], fontsize=8)
        ax.set_title(f"{w} training, 1000-step frames (int4-class grid c=0.35)", color=INK, loc="left", fontsize=10)
        ax.set_xlabel("checkpoint")
    axes[0].set_ylabel("bits / weight of the frame (block linears)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle("D2: Pythia-410M training as video", x=0.01, ha="left", color=INK)
    save(fig, "d2_pythia_frames.png")
