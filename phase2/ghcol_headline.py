"""Step-4 headline tables for ghcol (REPORT §16): the phase-2 family-codec pair table re-built with the ghcol
files (real .wcdc bytes, quality measured on the decoded models; results/phase2_family_<tag>.json), side by side
with the phase-2 rows, plus equal-quality savings computed LINE BY LINE (log-linear interpolation of MB against
log(quality) on each line separately; never a mixed hull, never extrapolated).

Writes results/ghcol_headline.json and prints markdown.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ghcol_summary import interp_bpw  # noqa: E402  (same interpolation as VERDICT A / B)

RES = os.path.join(ROOT, "results")
TITLE = {"qwen05b": "Qwen2.5-0.5B {base, Instruct}", "smol360": "SmolLM2-360M {base, Instruct}",
         "smol135": "SmolLM2-135M {base, Instruct}"}


def J(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else {}


def pairs(tag):
    fam = dict(J(f"phase2_family_{tag}.json"))       # phase-2 baselines + lossless rest files (unchanged)
    if not fam:
        return []
    fam.update({k: v for k, v in J(f"ghcol_family_{tag}.json").items() if k.startswith(("B_", "T_"))})  # new files
    S = lambda fs: sum(fam[n]["MB"] for n in fs) if all(n in fam for n in fs) else None
    out = []

    def add(kind, name, files, tq, bq):
        mb = S(files)
        if mb is None:
            return
        t, b = fam[tq], fam[bq]
        out.append({"kind": kind, "scenario": name, "files": files, "pair_MB": mb,
                    "linears_MB": sum(fam[n]["MB"] for n in files if "rest" not in n),
                    "ft_MB": t["MB"], "base_MB": b["MB"],
                    "ft_chat_kl": t.get("chat_kl"), "ft_wiki_kl": t.get("kl"), "ft_bpw": t.get("real_bpw"),
                    "base_kl": b.get("kl"), "base_ppl": b.get("ppl"), "base_bpw": b.get("real_bpw")})

    cs = sorted({k.split("_c")[-1] for k in fam if k.startswith(("B_ecsq_c", "B_ghcol_c", "B_gfisher_c"))}, key=float)
    for c in cs:
        add("family_row", f"FAMILY (phase 2): row-grid I-frame + P-frame c={c}",
            [f"B_ecsq_c{c}", "B_rest_ll16", f"T_pframe_c{c}", "T_rest_xor_base"], f"T_pframe_c{c}", f"B_ecsq_c{c}")
        add("family_ghcol", f"FAMILY ghcol: column-grid I-frame + P-frame c={c}",
            [f"B_ghcol_c{c}", "B_rest_ll16", f"T_pframe_ghcol_c{c}", "T_rest_xor_base"], f"T_pframe_ghcol_c{c}", f"B_ghcol_c{c}")
        add("family_gfisher", f"FAMILY A3-Fisher grid: column-grid I-frame + P-frame c={c}",
            [f"B_gfisher_c{c}", "B_rest_ll16", f"T_pframe_gfisher_c{c}", "T_rest_xor_base"], f"T_pframe_gfisher_c{c}", f"B_gfisher_c{c}")
        add("indep_row", f"both independent GPTQ-ECSQ c={c}",
            [f"B_ecsq_c{c}", "B_rest_ll16", f"T_ecsq_c{c}", "T_rest_ll16"], f"T_ecsq_c{c}", f"B_ecsq_c{c}")
        add("indep_ghcol", f"both independent ghcol c={c}",
            [f"B_ghcol_c{c}", "B_rest_ll16", f"T_ghcol_c{c}", "T_rest_ll16"], f"T_ghcol_c{c}", f"B_ghcol_c{c}")
    for k in sorted(fam):
        if k.startswith("T_delta_c"):
            add("delta_row", f"fp16 base + Δ (phase 2, GPTQ row grid) c={k.split('_c')[-1]}",
                ["B_fp16", "B_rest_ll16", k, "T_rest_xor_base"], k, "B_fp16")
        if k.startswith("T_dghcol"):
            add("delta_ghcol", f"fp16 base + ghcol-Δ (GPTQ, column grid) {k[2:]}", ["B_fp16", "B_rest_ll16", k, "T_rest_xor_base"], k, "B_fp16")
        if k.startswith("T_dhcol_c"):
            add("delta_hcol", f"fp16 base + hcol-Δ (rounding, column grid) {k[2:]}", ["B_fp16", "B_rest_ll16", k, "T_rest_xor_base"], k, "B_fp16")
    for k in ("T_gptq4_packed",):
        add("int4", "both GPTQ-int4 fixed-rate (standard int4 files), rest fp16",
            ["B_gptq4_packed", "B_rest_raw16", "T_gptq4_packed", "T_rest_raw16"], k, "B_gptq4_packed")
    return out


def saved(rows, new, old, q, mb="pair_MB"):
    """for every row of kind `new`: MB the `old` line needs for the same quality q, minus the new MB."""
    L = [(r[mb], r[q]) for r in rows if r["kind"] == old and r[q]]
    out = []
    for r in rows:
        if r["kind"] != new or not r[q]:
            continue
        o = interp_bpw(L, r[q])
        out.append({"scenario": r["scenario"], mb: r[mb], q: r[q], f"{old}_{mb}_at_equal_{q}": o,
                    "saved_MB": None if o is None else o - r[mb], "saved_pct": None if o is None else 100 * (o - r[mb]) / o})
    return out


def dominance(rows, new, q="ft_chat_kl"):
    """For rows of kind `new` whose quality lies outside the old lines' range: the cheapest OLD scenario (phase-2
    kinds) that is at least as good on q, and the old row-grid family point at the closest pair MB."""
    old = [r for r in rows if r["kind"] in ("family_row", "indep_row", "delta_row", "int4") and r[q] is not None]
    out = []
    for r in rows:
        if r["kind"] != new or r[q] is None:
            continue
        better = [o for o in old if o[q] <= r[q]]
        cheapest = min(better, key=lambda o: o["pair_MB"]) if better else None
        fam = [o for o in rows if o["kind"] == "family_row"]
        near = min(fam, key=lambda o: abs(o["pair_MB"] - r["pair_MB"])) if fam else None
        out.append({"scenario": r["scenario"], "pair_MB": r["pair_MB"], q: r[q],
                    "cheapest_old_at_least_as_good": None if cheapest is None else
                    {"scenario": cheapest["scenario"], "pair_MB": cheapest["pair_MB"], q: cheapest[q],
                     "saved_MB": cheapest["pair_MB"] - r["pair_MB"],
                     "saved_pct": 100 * (cheapest["pair_MB"] - r["pair_MB"]) / cheapest["pair_MB"]},
                    "row_family_at_nearest_MB": None if near is None else
                    {"scenario": near["scenario"], "pair_MB": near["pair_MB"], q: near[q],
                     "quality_ratio": near[q] / r[q] if r[q] else None}})
    return out


def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


if __name__ == "__main__" and "--report" not in sys.argv:
    allr, md = {}, []
    for tag in ("qwen05b", "smol360", "smol135"):
        rows = pairs(tag)
        if not rows:
            continue
        cmp_ = {"family_ghcol_vs_family_row_chatKL": saved(rows, "family_ghcol", "family_row", "ft_chat_kl"),
                "family_ghcol_vs_family_row_baseKL": saved(rows, "family_ghcol", "family_row", "base_kl"),
                "family_gfisher_vs_family_row_chatKL": saved(rows, "family_gfisher", "family_row", "ft_chat_kl"),
                "family_gfisher_vs_family_ghcol_chatKL": saved(rows, "family_gfisher", "family_ghcol", "ft_chat_kl"),
                "indep_ghcol_vs_indep_row_chatKL": saved(rows, "indep_ghcol", "indep_row", "ft_chat_kl"),
                "delta_ghcol_vs_delta_row_chatKL": saved(rows, "delta_ghcol", "delta_row", "ft_chat_kl"),
                "delta_ghcol_vs_delta_row_chatKL_linearsMB": saved(rows, "delta_ghcol", "delta_row", "ft_chat_kl", "linears_MB"),
                "delta_hcol_vs_delta_row_chatKL": saved(rows, "delta_hcol", "delta_row", "ft_chat_kl"),
                "delta_hcol_vs_delta_row_chatKL_linearsMB": saved(rows, "delta_hcol", "delta_row", "ft_chat_kl", "linears_MB"),
                "delta_ghcol_vs_delta_hcol_chatKL_linearsMB": saved(rows, "delta_ghcol", "delta_hcol", "ft_chat_kl", "linears_MB"),
                "delta_ghcol_vs_delta_row_wikiKL_linearsMB": saved(rows, "delta_ghcol", "delta_row", "ft_wiki_kl", "linears_MB"),
                "delta_hcol_vs_delta_row_wikiKL_linearsMB": saved(rows, "delta_hcol", "delta_row", "ft_wiki_kl", "linears_MB"),
                "delta_ghcol_vs_delta_row_chatKL_deltaMB": saved(rows, "delta_ghcol", "delta_row", "ft_chat_kl", "ft_MB"),
                "delta_hcol_vs_delta_row_chatKL_deltaMB": saved(rows, "delta_hcol", "delta_row", "ft_chat_kl", "ft_MB"),
                "delta_ghcol_vs_delta_hcol_chatKL_deltaMB": saved(rows, "delta_ghcol", "delta_hcol", "ft_chat_kl", "ft_MB"),
                "delta_ghcol_vs_delta_row_wikiKL_deltaMB": saved(rows, "delta_ghcol", "delta_row", "ft_wiki_kl", "ft_MB"),
                "delta_hcol_vs_delta_row_wikiKL_deltaMB": saved(rows, "delta_hcol", "delta_row", "ft_wiki_kl", "ft_MB")}
        dom = {k: dominance(rows, k) for k in ("family_ghcol", "family_gfisher", "indep_ghcol")}
        allr[tag] = {"pairs": rows, "saved": cmp_, "dominance": dom}
        md.append(f"#### {TITLE[tag]} — pair storage, real files (every tensor; quality of the DECODED models)\n")
        md.append("| scenario | pair MB | linears MB | base KL | base PPL | fine-tune chat-KL | fine-tune wiki-KL |")
        md.append("|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['scenario']} | {r['pair_MB']:.1f} | {r['linears_MB']:.1f} | {f(r['base_kl'], 4)} | "
                      f"{f(r['base_ppl'], 2)} | {f(r['ft_chat_kl'], 4)} | {f(r['ft_wiki_kl'], 4)} |")
        md.append("")
        def sv(x):
            if x["saved_MB"] is None:
                return "n/a"
            return "%+.1f MB (%+.1f %%)" % (x["saved_MB"], x["saved_pct"])
        for k, v in cmp_.items():
            if v:
                md.append(f"- {k}: " + "; ".join(f"{x['scenario'].split(': ')[-1]} → {sv(x)}" for x in v))
        md.append("")
        for k, v in dom.items():
            for x in v:
                c, n = x["cheapest_old_at_least_as_good"], x["row_family_at_nearest_MB"]
                md.append(f"- {x['scenario']} ({x['pair_MB']:.1f} MB, chat-KL {x['ft_chat_kl']:.4f}): cheapest phase-2 scenario "
                          f"at least as good = " + ("none" if c is None else f"{c['scenario']} {c['pair_MB']:.1f} MB "
                          f"(chat-KL {c['ft_chat_kl']:.4f}; {c['saved_MB']:+.1f} MB, {c['saved_pct']:+.1f} %)") +
                          ("" if n is None else f"; phase-2 family at nearest size: {n['pair_MB']:.1f} MB, chat-KL "
                           f"{n['ft_chat_kl']:.4f} ({n['quality_ratio']:.1f}x)"))
        md.append("")
    with open(os.path.join(RES, "ghcol_headline.json"), "w") as fh:
        json.dump(allr, fh, indent=1)
    print("\n".join(md))


def report_tables(allr):
    """Compact §16.5 tables (markdown) from the headline rows."""
    out = []
    out.append("| pair (real files, every tensor) | phase 2: row-grid I + P, c = 0.35 | ghcol I + P, c = 0.35 | ghcol I + P, c = 0.5 | "
               "A3-Fisher grid I + P, c = 0.35 | A3-Fisher grid I + P, c = 0.5 | two standard GPTQ-int4 files |")
    out.append("|---|---|---|---|---|---|---|")
    for tag, R in allr.items():
        rows = {r["scenario"]: r for r in R["pairs"]}
        def cell(name):
            r = rows.get(name)
            return "—" if r is None else f"{r['pair_MB']:.1f} MB · base KL {r['base_kl']:.3f} · chat-KL **{r['ft_chat_kl']:.3f}**"
        out.append(f"| {TITLE[tag]} | " + " | ".join(cell(n) for n in (
            "FAMILY (phase 2): row-grid I-frame + P-frame c=0.35", "FAMILY ghcol: column-grid I-frame + P-frame c=0.35",
            "FAMILY ghcol: column-grid I-frame + P-frame c=0.5", "FAMILY A3-Fisher grid: column-grid I-frame + P-frame c=0.35",
            "FAMILY A3-Fisher grid: column-grid I-frame + P-frame c=0.5",
            "both GPTQ-int4 fixed-rate (standard int4 files), rest fp16")) + " |")
    out.append("")
    out.append("| fine-tune Δ file (fp16 base given) | phase-2 Δ (GPTQ, row grid) MB → chat-KL | ghcol-Δ MB → chat-KL | hcol-Δ (rounding) MB → chat-KL | Δ MB saved at equal chat-KL: ghcol-Δ / hcol-Δ |")
    out.append("|---|---|---|---|---|")
    for tag, R in allr.items():
        L = lambda kind: ", ".join(f"{r['ft_MB']:.1f} → {r['ft_chat_kl']:.4f}" for r in sorted(
            [r for r in R["pairs"] if r["kind"] == kind], key=lambda r: r["ft_MB"]))
        sv = lambda key: ", ".join("n/a" if x["saved_MB"] is None else f"{x['saved_MB']:+.1f} MB ({x['saved_pct']:+.0f} %)"
                                   for x in sorted(R["saved"][key], key=lambda x: x["ft_MB"]))
        out.append(f"| {TITLE[tag]} | {L('delta_row')} | {L('delta_ghcol')} | {L('delta_hcol')} | "
                   f"{sv('delta_ghcol_vs_delta_row_chatKL_deltaMB')} / {sv('delta_hcol_vs_delta_row_chatKL_deltaMB')} |")
    return "\n".join(out)


if __name__ == "__main__" and "--report" in sys.argv:
    print(report_tables(json.load(open(os.path.join(RES, "ghcol_headline.json")))))
