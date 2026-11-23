"""Headline numbers of the family codec's DEFAULT coders (REPORT §17), re-measured on real .wcdc files:
before (phase-2 defaults, REPORT §14.3: results/phase2_family_<tag>.json) vs after (--preset default:
results/default_family_<tag>.json), for the three {base, Instruct} pairs.

  pair   = base I-frame + B_rest_ll16 + Instruct P-frame + T_rest_xor_base   (every tensor of both models)
  delta  = Instruct Δ file over the fp16 base (the C2 / §14.3 "hub keeps the fp16 base" scenario)
  P-frame control: GPTQ vs plain rounding on the inherited Fisher lattice (same c, same base file)
Equal-quality savings are computed LINE BY LINE (log-linear interpolation of MB against log(quality) on each line
separately, never a mixed hull, never extrapolated) -- the §16 rule.  Also checks that the default run reproduces the
§16.5 files of the same configs (results/ghcol_family_<tag>.json).

Writes results/default_headline.json and prints markdown.  usage: python phase2/default_headline.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ghcol_summary import interp_bpw  # noqa: E402

RES = os.path.join(ROOT, "results")
TAGS = {"qwen05b": "Qwen2.5-0.5B {base, Instruct}", "smol360": "SmolLM2-360M {base, Instruct}",
        "smol135": "SmolLM2-135M {base, Instruct}"}
CS = ("0.35", "0.5")
# the default P-frame quantizer on the inherited Fisher lattice, decided by the pre-registered rule of REPORT §17
# (decide_pframe below; "auto" = apply the rule to the measured pairs)
PFRAME_DEFAULT = os.environ.get("WC_PFRAME", "auto")
PNAME = {"gptq": "T_pframe_gfisher_c{c}", "round": "T_pframe_gfisher_rnd_c{c}"}


def J(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else {}


def row(F, files, ft, base):
    if not all(n in F for n in files):
        return None
    t, b = F[ft], F[base]
    return {"files": files, "pair_MB": sum(F[n]["MB"] for n in files),
            "linears_MB": sum(F[n]["MB"] for n in files if "rest" not in n), "ft_MB": t["MB"], "base_MB": b["MB"],
            "ft_bpw": t.get("real_bpw"), "base_bpw": b.get("real_bpw"),
            "ft_chat_kl": t.get("chat_kl"), "ft_chat_top1": t.get("chat_top1"), "ft_wiki_kl": t.get("kl"),
            "ft_ppl": t.get("ppl"), "base_kl": b.get("kl"), "base_ppl": b.get("ppl")}


def line_saving(new, old, q, mb):
    """for every point of `new`: MB the `old` line needs for the same quality q (interpolated), minus the new MB."""
    L = [(r[mb], r[q]) for r in old if r and r.get(q)]
    out = []
    for r in new:
        if not r or not r.get(q):
            continue
        o = interp_bpw(L, r[q])
        out.append({mb: r[mb], q: r[q], "old_" + mb: o, "saved_MB": None if o is None else o - r[mb],
                    "saved_pct": None if o is None else 100 * (o - r[mb]) / o})
    return out


def pframe_wins(tag):
    """Pre-registered rule, per pair: does the ROUNDING P-frame win -- fewer P-frame MB than the GPTQ P-frame at equal
    chat-KL (line by line, any interpolable point) or dominance at the same c (<= MB and <= chat-KL)?"""
    D = J(f"default_family_{tag}.json")
    G = [(D[f"T_pframe_gfisher_c{c}"]["MB"], D[f"T_pframe_gfisher_c{c}"]["chat_kl"]) for c in CS if f"T_pframe_gfisher_c{c}" in D]
    R = [(D[f"T_pframe_gfisher_rnd_c{c}"]["MB"], D[f"T_pframe_gfisher_rnd_c{c}"]["chat_kl"]) for c in CS if f"T_pframe_gfisher_rnd_c{c}" in D]
    if len(G) < 2 or len(R) < 2:
        return None
    dom = any(r[0] <= g[0] and r[1] <= g[1] for g, r in zip(G, R))
    sav = [interp_bpw(G, k) - mb for mb, k in R if interp_bpw(G, k) is not None]
    return {"round_dominates_at_same_c": dom, "round_saves_MB_at_equal_chat_kl": sav,
            "round_wins": bool(dom or (sav and all(x > 0 for x in sav)))}


def decide_pframe():
    w = {t: pframe_wins(t) for t in TAGS}
    n = sum(1 for v in w.values() if v and v["round_wins"])
    decided = "round" if n >= 2 else "gptq"
    return (decided if PFRAME_DEFAULT == "auto" else PFRAME_DEFAULT), {"per_pair": w, "pairs_where_round_wins": n,
                                                                         "rule_decision": decided}


def headline(tag, pq):
    P2 = J(f"phase2_family_{tag}.json")
    D = J(f"default_family_{tag}.json")
    G16 = J(f"ghcol_family_{tag}.json")
    if not P2 or not D:
        return None
    F = dict(P2)
    F.update({k: v for k, v in D.items() if k.startswith(("B_", "T_"))})
    fam = lambda I, P: [I, "B_rest_ll16", P, "T_rest_xor_base"]
    before = {c: row(F, fam(f"B_ecsq_c{c}", f"T_pframe_c{c}"), f"T_pframe_c{c}", f"B_ecsq_c{c}") for c in CS}
    gpt = {c: row(F, fam(f"B_gfisher_c{c}", f"T_pframe_gfisher_c{c}"), f"T_pframe_gfisher_c{c}", f"B_gfisher_c{c}") for c in CS}
    rnd = {c: row(F, fam(f"B_gfisher_c{c}", f"T_pframe_gfisher_rnd_c{c}"), f"T_pframe_gfisher_rnd_c{c}", f"B_gfisher_c{c}") for c in CS}
    after = rnd if pq == "round" else gpt
    int4 = row(F, ["B_gptq4_packed", "B_rest_raw16", "T_gptq4_packed", "T_rest_raw16"], "T_gptq4_packed", "B_gptq4_packed")
    indep_before = row(F, ["B_ecsq_c0.35", "B_rest_ll16", "T_ecsq_c0.35", "T_rest_ll16"], "T_ecsq_c0.35", "B_ecsq_c0.35")
    indep_after = row(F, ["B_gfisher_c0.35", "B_rest_ll16", "T_gfisher_c0.35", "T_rest_ll16"], "T_gfisher_c0.35", "B_gfisher_c0.35")
    dl = lambda pfx: sorted([row(F, ["B_fp16", "B_rest_ll16", k, "T_rest_xor_base"], k, "B_fp16")
                             for k in F if k.startswith(pfx)], key=lambda r: r["ft_MB"])
    d_before, d_after = dl("T_delta_c"), dl("T_dhcol_c")
    ba = []
    for c in CS:
        b, a = before[c], after[c]
        if b and a:
            ba.append({"c": float(c), "before": b, "after": a,
                       "pair_MB_change_pct": 100 * (a["pair_MB"] / b["pair_MB"] - 1),
                       "chat_kl_ratio_before_over_after": b["ft_chat_kl"] / a["ft_chat_kl"],
                       "base_kl_ratio_before_over_after": b["base_kl"] / a["base_kl"],
                       "vs_int4_pair_MB_pct": None if not int4 else 100 * (a["pair_MB"] / int4["pair_MB"] - 1)})
    B, A, Gp, R = [before[c] for c in CS], [after[c] for c in CS], [gpt[c] for c in CS], [rnd[c] for c in CS]
    eq = {"pair_at_equal_chat_kl": line_saving(A, B, "ft_chat_kl", "pair_MB"),
          "pair_at_equal_base_kl": line_saving(A, B, "base_kl", "pair_MB"),
          "delta_file_at_equal_chat_kl": line_saving(d_after, d_before, "ft_chat_kl", "ft_MB"),
          "delta_file_at_equal_wiki_kl": line_saving(d_after, d_before, "ft_wiki_kl", "ft_MB")}
    # dominance where the lines do not overlap: cheapest phase-2 family point that is at least as good on chat-KL
    dom = []
    for a in A:
        if not a:
            continue
        good = [b for b in B if b and b["ft_chat_kl"] <= a["ft_chat_kl"]]
        dom.append({"after_pair_MB": a["pair_MB"], "after_chat_kl": a["ft_chat_kl"],
                    "cheapest_before_at_least_as_good": None if not good else min(g["pair_MB"] for g in good)})
    pc = []
    for c in CS:
        g, r = gpt[c], rnd[c]
        if g and r:
            pc.append({"c": float(c), "gptq": {k: g[k] for k in ("ft_MB", "ft_bpw", "ft_chat_kl", "ft_chat_top1", "ft_wiki_kl")},
                       "round": {k: r[k] for k in ("ft_MB", "ft_bpw", "ft_chat_kl", "ft_chat_top1", "ft_wiki_kl")}})
    pfile = lambda d: [{"ft_MB": x["ft_MB"], "ft_chat_kl": x["ft_chat_kl"], "ft_wiki_kl": x["ft_wiki_kl"]} for x in d]
    pframe_ctrl = {"same_c": pc,
                   "round_vs_gptq_pframe_MB_at_equal_chat_kl": line_saving(pfile(R), pfile(Gp), "ft_chat_kl", "ft_MB"),
                   "round_vs_gptq_pframe_MB_at_equal_wiki_kl": line_saving(pfile(R), pfile(Gp), "ft_wiki_kl", "ft_MB")}
    repro = {}
    for k in sorted(D):
        if k in G16 and isinstance(D[k], dict) and "MB" in D[k]:
            repro[k] = {f: {"now": D[k].get(f), "s16": G16[k].get(f)} for f in ("bytes", "kl", "chat_kl", "ppl")}
            repro[k]["bytes_identical"] = D[k].get("bytes") == G16[k].get("bytes")
    return {"title": TAGS[tag], "after_pframe": pq, "after_pframe_file": PNAME[pq], "before_after_same_c": ba, "equal_quality": eq, "dominance_chat_kl": dom,
            "pframe_control": pframe_ctrl, "standalone": {"before": indep_before, "after": indep_after},
            "delta_lines": {"before_T_delta": d_before, "after_T_dhcol": d_after}, "int4_pair": int4,
            "after_pframe_gptq": {c: gpt[c] for c in CS}, "after_pframe_round": {c: rnd[c] for c in CS},
            "fallback_and_int16_floor": {k: {f: D[k].get(f) for f in ("n_row_fallback", "n_cols_raised")}
                                         for k in sorted(D) if isinstance(D[k], dict) and "n_cols_raised" in D[k]},
            "real_vs_proxy_pct": {k: 100 * (D[k]["real_bpw"] / D[k]["proxy_bpw"] - 1) for k in sorted(D)
                                  if isinstance(D[k], dict) and D[k].get("proxy_bpw")},
            "reproduces_s16_5": repro}


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    pq, rule = decide_pframe()
    out, md = {"pframe_rule": dict(rule, used=pq)}, [f"P-frame rule: round wins on {rule['pairs_where_round_wins']} pair(s) "
                                                     f"-> rule says {rule['rule_decision']}; tables use {pq}\n"]
    for tag in TAGS:
        h = headline(tag, pq)
        if h is None:
            continue
        out[tag] = h
        md.append(f"#### {h['title']} (after = Fisher-grid I-frame + {pq} P-frame)\n")
        md.append("| c | pair MB before → after | base KL before → after | fine-tune chat-KL before → after (ratio) | wiki-KL before → after |")
        md.append("|---|---|---|---|---|")
        for x in h["before_after_same_c"]:
            b, a = x["before"], x["after"]
            md.append(f"| {x['c']} | {b['pair_MB']:.1f} → **{a['pair_MB']:.1f}** ({x['pair_MB_change_pct']:+.1f} %) | "
                      f"{b['base_kl']:.4f} → **{a['base_kl']:.4f}** | {b['ft_chat_kl']:.4f} → **{a['ft_chat_kl']:.4f}** "
                      f"({x['chat_kl_ratio_before_over_after']:.1f}×) | {b['ft_wiki_kl']:.4f} → {a['ft_wiki_kl']:.4f} |")
        md.append("")
        for k, v in h["equal_quality"].items():
            md.append(f"- {k}: " + "; ".join("n/a" if s["saved_MB"] is None else f"{s['saved_MB']:+.1f} MB ({s['saved_pct']:+.1f} %)" for s in v))
        for x in h["dominance_chat_kl"]:
            md.append(f"- after {x['after_pair_MB']:.1f} MB (chat-KL {x['after_chat_kl']:.4f}): cheapest phase-2 family at least as good = "
                      f"{x['cheapest_before_at_least_as_good']}")
        md.append("- P-frame control (GPTQ vs rounding on the inherited lattice): " + "; ".join(
            f"c={p['c']}: GPTQ {p['gptq']['ft_MB']:.2f} MB → chat-KL {p['gptq']['ft_chat_kl']:.4f} / wiki {p['gptq']['ft_wiki_kl']:.4f}, "
            f"round {p['round']['ft_MB']:.2f} MB → {p['round']['ft_chat_kl']:.4f} / {p['round']['ft_wiki_kl']:.4f}"
            for p in h["pframe_control"]["same_c"]))
        for k in ("round_vs_gptq_pframe_MB_at_equal_chat_kl", "round_vs_gptq_pframe_MB_at_equal_wiki_kl"):
            md.append(f"  - {k}: " + "; ".join("n/a" if s["saved_MB"] is None else f"{s['saved_MB']:+.2f} MB ({s['saved_pct']:+.1f} %)"
                                                for s in h["pframe_control"][k]))
        sb, sa = h["standalone"]["before"], h["standalone"]["after"]
        if sb and sa:
            md.append(f"- standalone pair (no base reuse) c=0.35: {sb['pair_MB']:.1f} MB chat-KL {sb['ft_chat_kl']:.4f} → "
                      f"{sa['pair_MB']:.1f} MB chat-KL {sa['ft_chat_kl']:.4f}")
        md.append("- Δ file (fp16 base) before: " + ", ".join(f"{r['ft_MB']:.1f} MB → {r['ft_chat_kl']:.4f}" for r in h["delta_lines"]["before_T_delta"]) +
                  " | after: " + ", ".join(f"{r['ft_MB']:.1f} MB → {r['ft_chat_kl']:.4f}" for r in h["delta_lines"]["after_T_dhcol"]))
        same = [k for k, v in h["reproduces_s16_5"].items() if v["bytes_identical"]]
        md.append(f"- reproduces §16.5: {len(same)}/{len(h['reproduces_s16_5'])} files byte-identical in size; "
                  + ", ".join(f"{k} KL {v['kl']['now']:.4f} vs {v['kl']['s16']:.4f}" for k, v in h["reproduces_s16_5"].items()
                              if not v["bytes_identical"]))
        md.append("")
    with open(os.path.join(RES, "default_headline.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__" and "--report" not in sys.argv:
    main()


def report_tables(H):
    """Markdown tables for REPORT §17 from results/default_headline.json."""
    o = []
    cell = lambda r: "—" if not r else f"{r['pair_MB']:.1f} MB · base KL {r['base_kl']:.3f} · chat-KL **{r['ft_chat_kl']:.3f}**"
    o.append("| pair (real files, every tensor) | before: phase-2 default, row-grid I + P, c = 0.35 | **after: default, Fisher-grid I + P, "
             "c = 0.35** | after, c = 0.5 | both standalone, same new coder, c = 0.35 | two standard GPTQ-int4 files |")
    o.append("|---|---|---|---|---|---|")
    for tag, h in H.items():
        if tag == "pframe_rule":
            continue
        ba = {x["c"]: x for x in h["before_after_same_c"]}
        o.append(f"| {h['title']} | {cell(ba[0.35]['before'])} | {cell(ba[0.35]['after'])} | {cell(ba[0.5]['after'])} | "
                 f"{cell(h['standalone']['after'])} | {cell(h['int4_pair'])} |")
    o.append("")
    o.append("| pair | same pair size (c = 0.35): MB change · fine-tune chat-KL ratio · base KL ratio | c = 0.5 | vs two int4 files (c = 0.35) |")
    o.append("|---|---|---|---|")
    for tag, h in H.items():
        if tag == "pframe_rule":
            continue
        ba = {x["c"]: x for x in h["before_after_same_c"]}
        f_ = lambda x: f"{x['pair_MB_change_pct']:+.1f} % · **{x['chat_kl_ratio_before_over_after']:.1f}×** · {x['base_kl_ratio_before_over_after']:.1f}×"
        o.append(f"| {h['title']} | {f_(ba[0.35])} | {f_(ba[0.5])} | {ba[0.35]['vs_int4_pair_MB_pct']:+.0f} % bytes |")
    o.append("")
    o.append("| fine-tune Δ file over the fp16 base | before: GPTQ row grid, MB → chat-KL | **after: column grid, rounding**, MB → chat-KL | "
             "Δ MB saved at equal chat-KL (line by line) |")
    o.append("|---|---|---|---|")
    for tag, h in H.items():
        if tag == "pframe_rule":
            continue
        L = lambda rs: ", ".join(f"{r['ft_MB']:.1f} → {r['ft_chat_kl']:.4f}" for r in rs)
        sv = ", ".join("n/a" if s["saved_MB"] is None else f"{s['saved_MB']:+.1f} MB ({s['saved_pct']:+.0f} %)"
                       for s in h["equal_quality"]["delta_file_at_equal_chat_kl"])
        o.append(f"| {h['title']} | {L(h['delta_lines']['before_T_delta'])} | {L(h['delta_lines']['after_T_dhcol'])} | {sv} |")
    o.append("")
    o.append("| P-frame on the inherited Fisher lattice | GPTQ: MB → chat-KL / WikiText KL (c = 0.35; 0.5) | plain rounding: MB → chat-KL / WikiText KL | "
             "rounding: P-frame MB saved at equal chat-KL | rule: rounding wins? |")
    o.append("|---|---|---|---|---|")
    rule = H.get("pframe_rule", {}).get("per_pair", {})
    for tag, h in H.items():
        if tag == "pframe_rule":
            continue
        pc = h["pframe_control"]
        g = "; ".join(f"{p['gptq']['ft_MB']:.2f} → {p['gptq']['ft_chat_kl']:.4f} / {p['gptq']['ft_wiki_kl']:.4f}" for p in pc["same_c"])
        r = "; ".join(f"{p['round']['ft_MB']:.2f} → {p['round']['ft_chat_kl']:.4f} / {p['round']['ft_wiki_kl']:.4f}" for p in pc["same_c"])
        sv = ", ".join("n/a" if s["saved_MB"] is None else f"{s['saved_MB']:+.2f} MB ({s['saved_pct']:+.1f} %)"
                       for s in pc["round_vs_gptq_pframe_MB_at_equal_chat_kl"])
        w = rule.get(tag) or {}
        o.append(f"| {h['title']} | {g} | {r} | {sv} | {'yes' if w.get('round_wins') else 'no'}"
                 f"{' (dominates at the same c)' if w.get('round_dominates_at_same_c') else ''} |")
    return "\n".join(o)


if __name__ == "__main__" and "--report" in sys.argv:
    print(report_tables(json.load(open(os.path.join(RES, "default_headline.json")))))
