"""Tables + figures for the sparsity probe (scripts/prune_probe.py).  Reads results/prune_s1_smol135.json,
results/prune_s2_smol135.json, results/prune_s2_delta_dist_smol135.json; writes figures/prune_*.png and
prints markdown tables (pasted into PRUNE_FINDINGS.md)."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def style(ax, title, xl, yl):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def env(pts):
    pts = sorted(pts)
    return [p for p in pts if not any(q[0] <= p[0] and q[1] < p[1] for q in pts if q != p)]


def interp_bpw(front, k):
    """bpw of the dense front at distortion k (log-linear), None outside range."""
    pts = sorted(front)
    for (b0, k0), (b1, k1) in zip(pts, pts[1:]):
        l0, l1, lk = math.log(k0), math.log(k1), math.log(k)
        if min(l0, l1) <= lk <= max(l0, l1) and l0 != l1:
            return b0 + (lk - l0) / (l1 - l0) * (b1 - b0)
    return None


def saved_vs(line, b, k):
    """bpw the dense LINE needs for distortion k, minus b (log-linear; extrapolates on the end segment)."""
    pts = sorted(line)
    if len(pts) < 2:
        return None, None
    lk = math.log(k)
    for (b0, k0), (b1, k1) in zip(pts, pts[1:]):
        l0, l1 = math.log(k0), math.log(k1)
        if min(l0, l1) <= lk <= max(l0, l1) and l0 != l1:
            return b0 + (lk - l0) / (l1 - l0) * (b1 - b0) - b, True
    (b0, k0), (b1, k1) = (pts[0], pts[1]) if lk > math.log(pts[0][1]) else (pts[-2], pts[-1])
    l0, l1 = math.log(k0), math.log(k1)
    return b0 + (lk - l0) / (l1 - l0) * (b1 - b0) - b, False


def best_saved(lines, b, k):
    """bpw saved vs the CHEAPEST dense line at equal distortion: min over lines of (line_bpw(k) - b), each line
    interpolated on its own points (a Pareto envelope of mixed points zig-zags and flatters the sparse scheme).
    Prefers interpolated values; flags extrapolation."""
    res = [saved_vs(L, b, k) for L in lines if len(L) >= 2]
    ins = [r for r in res if r[1]]
    if ins:
        return min(ins)[0], True
    return (min(res)[0], False) if res else (None, None)


def line_kl(lines, b):
    """lowest interpolated dense KL at rate b over lines (None if b outside every line)."""
    ks = [interp_kl(L, b) for L in lines]
    ks = [x for x in ks if x is not None]
    return min(ks) if ks else None


def interp_kl(front, b):
    pts = sorted(front)
    for (b0, k0), (b1, k1) in zip(pts, pts[1:]):
        if b0 <= b <= b1 and b1 != b0:
            return math.exp(math.log(k0) + (b - b0) / (b1 - b0) * (math.log(k1) - math.log(k0)))
    return None


def s1():
    d = load("prune_s1_smol135.json")
    if not d:
        return
    R = {k: v for k, v in d.items() if not k.startswith("_") and isinstance(v, dict)}
    dense = {k: v for k, v in R.items() if k.startswith(("hcol_", "ghcol_"))}
    front = env([(v["bpw"], v["kl"]) for v in dense.values()])
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    hc = sorted((v["bpw"], v["kl"]) for k, v in R.items() if k.startswith("hcol_"))
    gh = sorted((v["bpw"], v["kl"]) for k, v in R.items() if k.startswith("ghcol_"))
    if gh:
        ax.plot(*zip(*gh), "-s", color=BLUE, lw=2, ms=6, label="dense GPTQ on hcol grid (best dense line)")
    ax.plot(*zip(*hc), "--o", color=BLUE, lw=1.3, ms=5, alpha=0.6, label="dense hcol, plain rounding (V2 best scalar)")
    for pref, col, mk, lab in (("sgpt_", ORANGE, "D", "SparseGPT + hcol quant (mask charged)"),
                               ("wanda_", AQUA, "^", "Wanda + hcol quant (mask charged)")):
        pts = [(v["bpw"], v["kl"], k) for k, v in R.items() if k.startswith(pref) and "_hcol_" in k]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, s=46, zorder=3,
                       edgecolor="white", lw=1, label=lab)
            for b, k_, nm in pts:
                tag = nm.split("_")[1]
                ax.annotate(tag.replace("r", "ρ").replace("24", "2:4"), (b, k_), textcoords="offset points",
                            xytext=(5, -10) if pref == "sgpt_" else (5, 4), fontsize=7, color=INK2)
    i4 = [(v.get("bpw_fixed_rate", v["bpw"]), v["kl"], k) for k, v in R.items() if "int4" in k]
    if i4:
        ax.scatter([p[0] for p in i4], [p[1] for p in i4], color=GRAY, marker="x", s=44, zorder=3,
                   label="int4 g64 fixed-rate (dense / 2:4 / 50%)")
    ax.set_yscale("log")
    style(ax, "S1  SmolLM2-135M: sparse + quant vs dense scalar, mask charged", "effective bits / weight (all charged)",
          "KL(fp16 || model), WikiText-2, 16 windows")
    ax.set_xlim(1.75, 4.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "prune_s1_smol135.png"))
    plt.close(fig)

    # tables
    print("\n### S1 table\n")
    hcl = [(v["bpw"], v["kl"]) for k, v in R.items() if k.startswith("hcol_")]
    ghl = [(v["bpw"], v["kl"]) for k, v in R.items() if k.startswith("ghcol_")]
    print("| config | rho | code | mask bpw | values bpw | side bpw | **total bpw** | H(rho) | KL | top-1 | PPL16 | front KL @ same bpw | **bpw saved vs best dense front** | bpw saved vs matched twin |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {"fp16": 0, "hcol": 1, "gptq_hcol": 2, "ecsq_row": 3, "prune_fp16": 4, "sgpt_hcol": 5, "wanda_hcol": 6, "int4": 7}
    for k, v in sorted(R.items(), key=lambda kv: (order.get(kv[1]["cfg"].get("kind", ""), 9), -kv[1].get("rho", 1), kv[1].get("bpw", 0))):
        if k == "fp16":
            continue
        fk = line_kl([hcl, ghl], v["bpw"])
        dense = k.startswith(("hcol_", "ghcol_"))
        sv, ip = (None, None) if dense else best_saved([hcl, ghl], v["bpw"], v["kl"])
        twin = ghl if k.startswith("sgpt_") and "_hcol_" in k else hcl if k.startswith("wanda_") and "_hcol_" in k else None
        tv, tip = saved_vs(twin, v["bpw"], v["kl"]) if twin else (None, None)
        f = lambda x, i: "" if x is None else f"{x:+.3f}" + ("" if i else " (extrap.)")
        print(f"| {k} | {v.get('rho', 1):.2f} | {v.get('code', '-')} | {v.get('mask_bpw', 0):.3f} | {v.get('values_bpw', 0):.3f} | "
              f"{v.get('side_bpw', 0):.3f} | **{v['bpw']:.3f}**{' (fixed ' + format(v['bpw_fixed_rate'], '.3f') + ')' if 'bpw_fixed_rate' in v else ''} | "
              f"{v.get('H_rho', 0):.3f} | {v['kl']:.4f} | {v['top1']:.3f} | {v['ppl']:.2f} | "
              f"{'' if fk is None else f'{fk:.4f}'} | {f(sv, ip)} | {f(tv, tip)} |")
    print("\nfront:", front)


def s2():
    d = load("prune_s2_smol135.json")
    dist = load("prune_s2_delta_dist_smol135.json").get("global")
    if dist:
        e = dist["hist_row_normalised"]
        cen = [(a + b) / 2 for a, b in zip(e["edges"], e["edges"][1:])]
        tot = sum(e["counts"])
        w = e["edges"][1] - e["edges"][0]
        dens = [c / tot / w for c in e["counts"]]
        fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
        ax.semilogy(cen, [max(x, 1e-9) for x in dens], color=BLUE, lw=2, label="Delta / rms_row(Delta), SmolLM2-135M")
        g = [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) for x in cen]
        lp = [math.exp(-abs(x) * math.sqrt(2)) / math.sqrt(2) for x in cen]
        ax.semilogy(cen, g, "--", color=GRAY, lw=1.4, label="Gaussian, same variance")
        ax.semilogy(cen, lp, ":", color=ORANGE, lw=1.6, label="Laplace, same variance")
        ax.set_ylim(1e-6, 2)
        style(ax, "S2  fine-tune delta distribution (row-normalised)", "Delta / rms_row(Delta)", "density (log)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "prune_s2_delta_hist.png"))
        plt.close(fig)
        print("\n### S2 delta distribution\n")
        print(json.dumps({k: v for k, v in dist.items() if k != "hist_row_normalised"}, indent=1))
    if not d:
        return
    R = {k: v for k, v in d.items() if not k.startswith("_") and isinstance(v, dict)}
    DENSE = ("dscal_", "dhcol_", "dgptq_", "dghcol_", "zero_bits_base")
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    for pref, mk, ls, lab, col, al in (("dhcol_", "s", "-", "dense Delta hcol (best dense line)", BLUE, 1.0),
                                       ("dghcol_", "D", "--", "dense Delta GPTQ on hcol grid", BLUE, 0.6),
                                       ("dscal_", "o", ":", "dense Delta ECSQ, row step (V3 nD)", BLUE, 0.45),
                                       ("dhcoldz1.0_", "s", "--", "dense Delta hcol + dead-zone 1.0 step (no mask)", GRAY, 0.9)):
        pts = sorted((v["bpw"], v["chat_kl"]) for k, v in R.items() if k.startswith(pref))
        if pts:
            ax.plot(*zip(*pts), ls, marker=mk, color=col, lw=2 if ls == "-" else 1.2, ms=5 if ls != "-" else 6,
                    alpha=al, label=lab)
    for pref, col, mk, lab in (("dmag_", ORANGE, "D", "sparse Delta by |Delta| (mask charged)"),
                               ("dwanda_", AQUA, "^", "sparse Delta by |Delta| sqrt(h) (mask charged)"),
                               ("dsgpt_", ORANGE, "P", "SparseGPT on Delta + hcol quant"),
                               ("ddare_", GRAY, "x", "DARE random mask (seeded, 0 mask bits)")):
        pts = [(v["bpw"], v["chat_kl"], k) for k, v in R.items() if k.startswith(pref)]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, s=44, zorder=3,
                       edgecolor="white" if mk not in "xP" else None, lw=1, label=lab)
    zb = R.get("zero_bits_base")
    if zb:
        ax.axhline(zb["chat_kl"], color=GRAY, lw=1, ls=":")
        ax.annotate("instruct := base (0 bits)", (0.05, zb["chat_kl"]), textcoords="offset points", xytext=(0, 4),
                    fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xlim(0, 3.2)
    style(ax, "S2  SmolLM2-135M-Instruct given fp16 base: sparse vs dense Delta", "Delta bits / weight (all charged)",
          "chat KL(instruct fp16 || model)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "prune_s2_delta_rd.png"))
    plt.close(fig)
    front = env([(v["bpw"], v["chat_kl"]) for k, v in R.items() if k.startswith(DENSE)])
    zb = R.get("zero_bits_base")
    def line(pref, key):
        pts = [(v["bpw"], v[key]) for k, v in R.items() if k.startswith(pref)]
        if zb and pts:
            pts.append((0.0, zb[key]))          # every Delta coder has the 0-bit 'instruct := base' end point
        return sorted(pts)
    print("\n### S2 comparison (per-line interpolation, chat-KL and WikiText KL)\n")
    print("| sparse config | rho | total bpw | mask bpw | chat KL | WT2 KL | saved vs best plain dense line (chat) | saved vs best plain dense line (WT2) | saved vs dead-zone line (chat) |")
    print("|---|---|---|---|---|---|---|---|---|")
    plain = ("dscal_", "dhcol_", "dgptq_", "dghcol_")
    dz = sorted({k.split("_")[0] for k in R if k.startswith("dhcoldz")})
    for k, v in sorted(R.items(), key=lambda kv: kv[1].get("bpw", 0)):
        if k.startswith(plain + ("dhcoldz", "zero_bits", "fp16")):
            continue
        f = lambda x: "" if x[0] is None else f"{x[0]:+.3f}" + ("" if x[1] else " (extrap.)")
        sc = best_saved([line(p, "chat_kl") for p in plain], v["bpw"], v["chat_kl"])
        sw = best_saved([line(p, "kl") for p in plain], v["bpw"], v["kl"])
        sz = best_saved([line(p + "_", "chat_kl") for p in dz], v["bpw"], v["chat_kl"]) if dz else (None, None)
        print(f"| {k} | {v.get('rho', 1):.2f} | **{v['bpw']:.3f}** | {v.get('mask_bpw', 0):.3f} | {v['chat_kl']:.4f} | {v['kl']:.4f} | "
              f"{f(sc)} | {f(sw)} | {f(sz)} |")
    print("\n### S2 table\n")
    print("| config | rho | code | mask bpw | values bpw | side bpw | **total bpw** | chat KL | chat top-1 | WT2 KL | dense chat-KL @ same bpw | bpw saved @ equal chat-KL |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(R.items(), key=lambda kv: (kv[1]["cfg"].get("kind", ""), kv[1].get("bpw", 0))):
        fk = interp_kl(front, v["bpw"]) if 0 < v.get("bpw", 16) < 16 else None
        sv = v.get("saved_bpw_vs_dense_chat_kl")
        ip = v.get("saved_interp_chat_kl")
        print(f"| {k} | {v.get('rho', 1):.3f} | {v.get('code', '-')} | {v.get('mask_bpw', 0):.3f} | {v.get('values_bpw', 0):.3f} | "
              f"{v.get('side_bpw', 0):.3f} | **{v['bpw']:.3f}** | {v['chat_kl']:.4f} | {v['chat_top1']:.3f} | {v['kl']:.4f} | "
              f"{'' if fk is None else f'{fk:.4f}'} | {'' if sv is None else f'{sv:+.3f}' + ('' if ip else ' (extrap.)')} |")
    print("\nfront:", front)


if __name__ == "__main__":
    s1()
    s2()
