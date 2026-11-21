"""Phase-2 tables + figures from the real-bitstream results.

Outputs: results/phase2_tables.md (markdown for REPORT.md), results/phase2_formats.csv,
results/phase2_space_summary.json (Space tables), figures/p2_*.png.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
J = lambda n: json.load(open(os.path.join(RES, n))) if os.path.exists(os.path.join(RES, n)) else {}
HOST = os.environ.get("WC_HOST", "Darwin-arm64-8cpu")

FAMS = {"qwen05b": {"title": "Qwen2.5-0.5B {base, Instruct}", "group": 128, "base": "Qwen/Qwen2.5-0.5B",
                    "target": "Qwen/Qwen2.5-0.5B-Instruct", "eval": "MPS fp16, WikiText-2 PPL on all 146 windows"},
        "smol360": {"title": "SmolLM2-360M {base, Instruct}", "group": 64, "base": "HuggingFaceTB/SmolLM2-360M",
                    "target": "HuggingFaceTB/SmolLM2-360M-Instruct", "eval": "local MPS fp16, PPL on all windows (HF jobs were cancelled externally)"},
        "smol135": {"title": "SmolLM2-135M {base, Instruct} (Space demo pair)", "group": 64,
                    "base": "HuggingFaceTB/SmolLM2-135M", "target": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "eval": "MPS fp16, PPL on all windows"}}
LABEL = {"gptq4_packed": "GPTQ-int4 g{g} fixed-rate (packed nibbles)", "gptq4_ans": "GPTQ-int4 g{g} + rANS",
         "ecsq_c0.35": "GPTQ-ECSQ c=0.35 + rANS", "ecsq_c0.5": "GPTQ-ECSQ c=0.5 + rANS",
         "pframe_c0.35": "P-frame on base I-frame lattice (c=0.35)", "pframe_c0.5": "P-frame on base lattice (c=0.5)",
         "delta_c0.06": "full-rank Δ vs fp16 base, GPTQ c=0.06 + rANS", "delta_c0.12": "full-rank Δ vs fp16 base, GPTQ c=0.12 + rANS",
         "delta_c0.25": "full-rank Δ vs fp16 base, GPTQ c=0.25 + rANS", "lowrank_r32": "LoRA-style Δ rank 32 (fp16 factors)",
         "lowrank_r64": "LoRA-style Δ rank 64 (fp16 factors)", "lowrank_r128": "LoRA-style Δ rank 128 (fp16 factors)",
         "fp16": "fp16 (raw)"}


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def ms(rec, key):
    if not rec or key not in rec:
        return None
    return 1e3 * rec[key]["total"] if "total" in rec[key] else 1e3 * rec[key].get("entropy", np.nan)


def family_rows(tag):
    fam = J(f"phase2_family_{tag}.json")
    if not fam:
        return None
    dec = J(f"phase2_decode_{tag}_{HOST}.json")
    zr = J(f"phase2_zstd_ref_{tag}.json")           # per-tensor narrowest dtype (fair to zstd)
    g = FAMS[tag]["group"]
    rows = []
    for nm, r in fam.items():
        if "rest" in nm or not nm.startswith(("B_", "T_")) or "MB" not in r:
            continue
        role = "base" if nm.startswith("B_") else "fine-tune"
        key = nm[2:]
        dk = nm + "+base" if nm.startswith("T_pframe") else nm
        d0, d16, dz = dec.get(dk), dec.get(dk + "@rans16"), dec.get(nm + "@zstd19")
        dg = dec.get(dk + "@rans16_4k")
        donly = dec.get(nm + "_only@rans16")
        rows.append({
            "tag": tag, "config": nm, "role": role, "format": LABEL.get(key, key).format(g=g),
            "MB": r["MB"], "real_bpw": r["real_bpw"],
            "proxy_bpw": r.get("proxy_bpw") if "packed" not in nm else 4 + (16 + 4) / g,
            "zstd19_bpw": (zr.get(nm) or {}).get("zstd19_bpw") or (r.get("refcoder_bpw") or {}).get("zstd19") or
                          ((dz["MB"] * 8e6 / r["n_weights"]) if dz and r.get("n_weights") else None),
            "zstd19_vs_ans_pct": (zr.get(nm) or {}).get("zstd19_vs_ans_pct"),
            "zstd19_decode_ms": 1e3 * zr[nm]["zstd19_decode_1t_s"] if nm in zr else None,
            "gzip9_bpw": (r.get("refcoder_bpw") or {}).get("gzip9"),
            "ppl": r.get("ppl"), "kl": r.get("kl"), "top1": r.get("top1"), "chat_kl": r.get("chat_kl"),
            "chat_top1": r.get("chat_top1"),
            "dec_cpu_constriction_1t_ms": ms(d0, "serial_cpu"), "dec_cpu_constriction_pool_ms": ms(d0, "parallel_cpu"),
            "dec_cpu_rans16_1t_ms": ms(d16, "serial_cpu"), "dec_cpu_rans16_par_ms": ms(d16, "parallel_cpu"),
            "dec_mps_rans16_par_ms": ms(d16, "parallel_mps"), "dec_mps_constriction_pool_ms": ms(d0, "parallel_mps"),
            "dec_chain_MB": (d0 or d16 or {}).get("MB"),
            "dec_gpu_ms": ms(dg, "gpu_mps"), "dec_gpu_MB": (dg or {}).get("MB"),
            "dec_mps_constriction_1t_ms": ms(d0, "serial_mps"), "dec_mps_rans16_1t_ms": ms(d16, "serial_mps"),
            "dec_pframe_only_rans16_par_ms": ms(donly, "parallel_cpu"),
            "zstd19_sym_decode_ms": ms(dz, "serial_cpu"), "zstd19_MB": (dz or {}).get("MB"),
        })
    return rows


def rest(fam, nm):
    return fam.get(nm, {}).get("MB")


def pair_rows(tag):
    fam = J(f"phase2_family_{tag}.json")
    if not fam or "T_pframe_c0.35" not in fam:
        return None
    S = lambda *ns: sum(fam[n]["MB"] for n in ns) if all(n in fam for n in ns) else None
    Tq = lambda n: (fam.get(n, {}).get("chat_kl"), fam.get(n, {}).get("kl"), fam.get(n, {}).get("ppl"))
    out = []
    def add(name, files, tq, bq, note=""):
        mb = S(*files)
        if mb is None:
            return
        out.append({"tag": tag, "scenario": name, "files": files, "pair_MB": mb, "ft_chat_kl": Tq(tq)[0],
                    "ft_wiki_kl": Tq(tq)[1], "base_ppl": fam.get(bq, {}).get("ppl"), "base_kl": fam.get(bq, {}).get("kl"),
                    "note": note})
    add("both fp16 (raw)", ["B_fp16", "B_rest_raw16", "T_fp16", "T_rest_raw16"], "T_fp16", "B_fp16")
    add("both GPTQ-int4 fixed-rate (standard int4 files), rest fp16", ["B_gptq4_packed", "B_rest_raw16", "T_gptq4_packed", "T_rest_raw16"], "T_gptq4_packed", "B_gptq4_packed")
    add("both GPTQ-int4 + rANS, rest lossless", ["B_gptq4_ans", "B_rest_ll16", "T_gptq4_ans", "T_rest_ll16"], "T_gptq4_ans", "B_gptq4_ans")
    add("both independent GPTQ-ECSQ c=0.35, rest lossless", ["B_ecsq_c0.35", "B_rest_ll16", "T_ecsq_c0.35", "T_rest_ll16"], "T_ecsq_c0.35", "B_ecsq_c0.35")
    add("FAMILY: base I-frame + fine-tune P-frame (c=0.35), rest lossless (+XOR P)", ["B_ecsq_c0.35", "B_rest_ll16", "T_pframe_c0.35", "T_rest_xor_base"], "T_pframe_c0.35", "B_ecsq_c0.35")
    add("FAMILY: base I-frame + P-frame (c=0.5)", ["B_ecsq_c0.5", "B_rest_ll16", "T_pframe_c0.5", "T_rest_xor_base"], "T_pframe_c0.5", "B_ecsq_c0.5")
    for c in ("0.06", "0.12", "0.25"):
        add(f"lossless fp16 base + full-rank Δ c={c}", ["B_fp16", "B_rest_ll16", f"T_delta_c{c}", "T_rest_xor_base"], f"T_delta_c{c}", "B_fp16",
            "base linears raw fp16 (lossless); a lossless rANS of them would be ~4/5 of that")
    for r in ("32", "64", "128"):
        add(f"lossless fp16 base + LoRA-style Δ r={r}", ["B_fp16", "B_rest_ll16", f"T_lowrank_r{r}", "T_rest_xor_base"], f"T_lowrank_r{r}", "B_fp16")
    return out


def md_table(rows, cols, heads):
    s = "| " + " | ".join(heads) + " |\n|" + "---|" * len(heads) + "\n"
    for r in rows:
        s += "| " + " | ".join(c(r) if callable(c) else f(r.get(c)) for c in cols) + " |\n"
    return s


def plots(allrows, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    for tag in FAMS:
        rows = [r for r in allrows if r["tag"] == tag and r["role"] == "fine-tune" and r.get("chat_kl") is not None and r["config"] != "T_fp16"]
        if not rows:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        groups = {"independent": [r for r in rows if r["config"].startswith(("T_gptq4", "T_ecsq"))],
                  "P-frame (quantized base)": [r for r in rows if r["config"].startswith("T_pframe")],
                  "full-rank Δ (fp16 base)": [r for r in rows if r["config"].startswith("T_delta")],
                  "LoRA-style low-rank Δ": [r for r in rows if r["config"].startswith("T_lowrank")]}
        for (gname, rs), col, mk in zip(groups.items(), ["#4C72B0", "#DD8452", "#55A868", "#C44E52"], "osD^"):
            rs = sorted(rs, key=lambda r: r["MB"])
            if not rs:
                continue
            ax[0].plot([r["MB"] for r in rs], [max(r["chat_kl"], 1e-4) for r in rs], marker=mk, color=col, label=gname,
                       ls="none" if gname == "independent" else "-")
            if gname == "independent":
                for r in rs:
                    ax[0].annotate(r["config"][2:].replace("gptq4_", "int4 ").replace("ecsq_", "ECSQ "), (r["MB"], r["chat_kl"]),
                                   fontsize=7, xytext=(3, 3), textcoords="offset points")
            dm = [(r["dec_cpu_rans16_par_ms"], r["chat_kl"]) for r in rs if r.get("dec_cpu_rans16_par_ms")]
            if dm:
                ax[1].plot([d[0] for d in dm], [max(d[1], 1e-4) for d in dm], marker=mk, ls="none", color=col, label=gname)
        for a in ax:
            a.set_yscale("log"); a.grid(alpha=0.3); a.set_ylabel("chat-KL to fp16 fine-tune (lower = better)")
        ax[0].set_xlabel("fine-tune linears: REAL MB on disk (P-frame: excl. base)")
        ax[1].set_xlabel("decode ms, CPU, rans16 parallel (P-frame: base+P chain)")
        ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
        fig.suptitle(f"{FAMS[tag]['title']}: real bytes vs quality, decode time vs quality")
        fig.tight_layout()
        p = f"p2_{tag}_mb_dec_quality.png"; fig.savefig(os.path.join(FIG, p), dpi=110); plt.close(fig)
        out.append(p)
    # pair MB plot
    for tag in FAMS:
        ps = [p for p in (pairs or []) if p["tag"] == tag and p["ft_chat_kl"] is not None]
        if not ps:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for p in ps:
            c = "#DD8452" if p["scenario"].startswith("FAMILY") else ("#55A868" if "full-rank" in p["scenario"] else
                 ("#C44E52" if "LoRA" in p["scenario"] else "#4C72B0"))
            ax.scatter(p["pair_MB"], max(p["ft_chat_kl"], 1e-4), color=c, s=40)
            ax.annotate(p["scenario"].split(",")[0][:38], (p["pair_MB"], max(p["ft_chat_kl"], 1e-4)), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.set_xlabel("MB on disk for the PAIR {base, fine-tune} (all tensors)"); ax.set_ylabel("fine-tune chat-KL")
        ax.set_title(f"{FAMS[tag]['title']}: pair storage vs fine-tune quality")
        fig.tight_layout(); p = f"p2_{tag}_pair_mb.png"; fig.savefig(os.path.join(FIG, p), dpi=110); plt.close(fig)
        out.append(p)
    # trajectory: per-frame real bpw
    T = J("phase2_trajectory_pythia410m.json")
    ws = [w for w in ("late", "mid") if f"{w}_summary" in T]
    if ws:
        fig, axs = plt.subplots(1, len(ws), figsize=(6.2 * len(ws), 4.2), squeeze=False)
        for ax, w in zip(axs[0], ws):
            s = T[f"{w}_summary"]; fr = s["frames"]; x = [f["step"] / 1000 for f in fr]
            for c, col in (("0.35", "#4C72B0"), ("0.7", "#DD8452")):
                ax.plot(x, [f[f"allI_c{c}_bpw"] for f in fr], "o--", color=col, alpha=.6, label=f"all-I c={c}")
                ax.plot(x, [f[f"IPPP_c{c}_bpw"] for f in fr], "s-", color=col, label=f"I+P (IPPP) c={c}")
            ax.plot(x, [f["ll_lin_bpp"] for f in fr], "^-", color="#55A868", label="lossless (XOR-to-prev)")
            ax.set_yscale("log"); ax.grid(alpha=.3); ax.set_xlabel("training step (k)")
            ax.set_ylabel("REAL bits/weight of the frame file (linears)"); ax.set_title(f"Pythia-410M, {w} window")
            ax.legend(fontsize=7)
        fig.tight_layout(); p = "p2_pythia_frames_real.png"; fig.savefig(os.path.join(FIG, p), dpi=110); plt.close(fig)
        out.append(p)
    V = J("phase2_svc_qwen05b.json")
    if V:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for key, col in (("r3", "#55A868"), ("r2", "#C44E52")):
            s = V.get(f"{key}_summary")
            if not s:
                continue
            L = s["layers"]
            ax.plot([l["single_bpw"] for l in L], [l["cum_bpw"] for l in L], "o-", color=col, label=f"SVC ratio {s['ratio']} (cumulative real bpw)")
        lim = [1.5, 6.5]; ax.plot(lim, lim, "k:", label="no tax (= single-shot)")
        ax.set_xlabel("single-shot ECSQ file at the same lattice: real bpw"); ax.set_ylabel("progressive stream up to that layer: real bpw")
        ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("Qwen2.5-0.5B: SVC tax in real bytes")
        fig.tight_layout(); p = "p2_svc_tax_real.png"; fig.savefig(os.path.join(FIG, p), dpi=110); plt.close(fig)
        out.append(p)
    return out


PH1 = {  # phase-2 config -> (phase-1 results file, key, phase-1 rate field)
    "B_gptq4_ans": ("spine_qwen05b.json", "gptq_int4_g128", "entropy_bpw"),
    "B_ecsq_c0.35": ("spine_qwen05b.json", "gptq_ecsq_row_c0.35", "entropy_bpw"),
    "B_ecsq_c0.5": ("spine_qwen05b.json", "gptq_ecsq_row_c0.5", "entropy_bpw"),
    "T_gptq4_ans": ("c2_qwen05b_instruct.json", "indep_gptq_int4_g128", "bpw"),
    "T_ecsq_c0.35": ("c2_qwen05b_instruct.json", "indep_gptq_ecsq_c0.35", "bpw"),
    "T_ecsq_c0.5": ("c2_qwen05b_instruct.json", "indep_gptq_ecsq_c0.5", "bpw"),
    "T_pframe_c0.35": ("c2_qwen05b_instruct.json", "pframe_gptq2_c0.35", "bpw"),
    "T_pframe_c0.5": ("c2_qwen05b_instruct.json", "pframe_gptq2_c0.5", "bpw"),
    "T_delta_c0.06": ("c2_qwen05b_instruct.json", "delta_gptq_c0.06", "bpw"),
    "T_delta_c0.12": ("c2_qwen05b_instruct.json", "delta_gptq_c0.12", "bpw"),
    "T_delta_c0.25": ("c2_qwen05b_instruct.json", "delta_gptq_c0.25", "bpw"),
    "T_lowrank_r32": ("c2_qwen05b_instruct.json", "delta_lowrank_r32", "bpw"),
    "T_lowrank_r128": ("c2_qwen05b_instruct.json", "delta_lowrank_r128", "bpw"),
}


def phase1_compare():
    fam = J("phase2_family_qwen05b.json")
    rows = []
    for nm, (fn, key, rf) in PH1.items():
        if nm not in fam:
            continue
        p1 = J(fn).get(key, {})
        r = fam[nm]
        q2 = r.get("chat_kl", r.get("kl")); q1 = p1.get("chat_kl", p1.get("kl"))
        rows.append({"config": nm, "phase1": key, "p1_bpw": p1.get(rf), "p2_real_bpw": r["real_bpw"], "MB": r["MB"],
                     "delta_pct": 100 * (r["real_bpw"] / p1[rf] - 1) if p1.get(rf) else None,
                     "p1_q": q1, "p2_q": q2, "p1_ppl": p1.get("ppl"), "p2_ppl": r.get("ppl"),
                     "metric": "chat-KL" if "chat_kl" in r else "wiki-KL"})
    md = md_table(rows, ["config", "phase1", lambda r: f(r["p1_bpw"], 4), lambda r: f(r["p2_real_bpw"], 4),
                         lambda r: f(r["delta_pct"], 2) + " %", lambda r: f(r["MB"], 2), "metric",
                         lambda r: f(r["p1_q"], 4), lambda r: f(r["p2_q"], 4), lambda r: f(r["p1_ppl"], 3), lambda r: f(r["p2_ppl"], 3)],
                  ["phase-2 file", "phase-1 config", "phase-1 estimate bpw", "phase-2 REAL bpw", "real vs estimate", "MB",
                   "quality metric", "phase-1", "phase-2 (decoded file)", "phase-1 PPL", "phase-2 PPL"])
    return rows, md


def main():
    allrows, pairs, md = [], [], []
    cmp_rows, cmp_md = phase1_compare()
    if cmp_rows:
        md.append("#### Qwen2.5-0.5B: real file bytes vs the phase-1 entropy estimate (same algorithms, same seeds)\n")
        md.append(cmp_md)
    for tag in FAMS:
        rows = family_rows(tag)
        if not rows:
            continue
        allrows += rows
        pr = pair_rows(tag) or []
        pairs += pr
        md.append(f"#### {FAMS[tag]['title']} — linears (block nn.Linear), real files ({FAMS[tag]['eval']})\n")
        md.append(md_table(sorted(rows, key=lambda r: (r["role"] != "base", -r["MB"])),
                           ["role", "format", lambda r: f(r["MB"], 1), lambda r: f(r["real_bpw"], 3),
                            lambda r: f(r["proxy_bpw"], 3),
                            lambda r: f(100 * (r["real_bpw"] / r["proxy_bpw"] - 1), 2) + " %" if r.get("proxy_bpw") else "—",
                            lambda r: f(r["zstd19_bpw"], 3), lambda r: f(r["ppl"], 2), lambda r: f(r["kl"], 4),
                            lambda r: f(r["chat_kl"], 4),
                            lambda r: f(r["dec_cpu_constriction_1t_ms"], 0), lambda r: f(r["dec_cpu_rans16_par_ms"], 0),
                            lambda r: f(r["dec_mps_rans16_par_ms"], 0), lambda r: f(r.get("dec_gpu_ms"), 0)],
                           ["model", "format", "MB", "real bpw", "phase-1 proxy bpw", "real vs proxy", "zstd-19 bpw",
                            "PPL", "wiki-KL", "chat-KL", "decode ms CPU constriction 1-thr", "decode ms CPU rans16 parallel",
                            "decode ms MPS (rans16 CPU-par + GPU recon)", "decode ms GPU (Metal rANS + GPU recon)"]))
        if pr:
            md.append(f"\n**Pair {{base, fine-tune}} — every tensor, real MB** ({FAMS[tag]['title']})\n")
            md.append(md_table(pr, ["scenario", lambda r: f(r["pair_MB"], 1), lambda r: f(r["base_ppl"], 2),
                                    lambda r: f(r["ft_chat_kl"], 4), lambda r: f(r["ft_wiki_kl"], 4)],
                               ["scenario", "pair MB", "base PPL", "fine-tune chat-KL", "fine-tune wiki-KL"]))
        rest = J(f"phase2_family_{tag}.json")
        md.append("\nRest tensors (embeddings + norms + biases, lossless): " + ", ".join(
            f"{k}: {v['MB']:.1f} MB ({v['bpp']:.2f} bit/param)" for k, v in rest.items() if "rest" in k) + "\n")
    with open(os.path.join(RES, "phase2_formats.csv"), "w", newline="") as fh:
        if allrows:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    json.dump({"formats": allrows, "pairs": pairs}, open(os.path.join(RES, "phase2_summary.json"), "w"), indent=1)
    imgs = plots(allrows, pairs)
    open(os.path.join(RES, "phase2_tables.md"), "w").write("\n".join(md))
    print("\n".join(md)); print("figures:", imgs)


if __name__ == "__main__":
    main()
