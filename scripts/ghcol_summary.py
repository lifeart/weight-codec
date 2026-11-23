"""VERDICT A / B tables for the ghcol confirmation (scripts/ghcol_probe.py results).

Every comparison is LINE vs LINE: for a point (b, k) of line X, the bpw line Y needs for the same distortion k
is found by log-linear interpolation of bpw against log(k) between Y's two bracketing points (never a hull of
several lines; never extrapolated -- points outside Y's distortion range are reported as n/a).
saved = bpw_Y(k) - b  (positive = X is cheaper at equal quality).

usage: python scripts/ghcol_summary.py   -> results/ghcol_verdict.json + markdown on stdout
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def interp_bpw(line, k):
    """line: [(bpw, dist)] -> bpw at distortion k (log-linear), or None if k is outside the line's range."""
    pts = sorted((d, b) for b, d in line if d and d > 0)
    if len(pts) < 2 or not k or k <= 0:
        return None
    lk = math.log(k)
    for (d0, b0), (d1, b1) in zip(pts, pts[1:]):
        l0, l1 = math.log(d0), math.log(d1)
        if l0 <= lk <= l1 and l1 > l0:
            t = (lk - l0) / (l1 - l0)
            return b0 + t * (b1 - b0)
    return None


def compare(X, Y):
    """[(bpw_x, k, saved_vs_Y or None)] for every point of X."""
    return [(b, k, None if interp_bpw(Y, k) is None else interp_bpw(Y, k) - b) for b, k in sorted(X)]


def line(store, prefix, key="kl", bkey="entropy_bpw"):
    out = []
    for n, v in store.items():
        if not isinstance(v, dict) or not n.startswith(prefix):
            continue
        rest = n[len(prefix):]
        if not rest.replace(".", "").isdigit():      # exact family only (ghcol_c* must not match ghcol_g0.5_c*)
            continue
        out.append((v[bkey], v[key]))
    return sorted(out)


def fmt_line(L):
    return ", ".join(f"{b:.3f}→{k:.4f}" for b, k in L)


def verdict_base(tag, extra):
    g = J(f"ghcol_base_{tag}.json")
    if not g:
        return None
    lines = {"ghcol": line(g, "ghcol_c"), "hcol": line(g, "hcol_c"), "ghcol_g0.5": line(g, "ghcol_g0.5_c"),
             "gptq_row": line(g, "gptq_row_c"), "ecsq_row": line(g, "ecsq_row_c"),
             "gfisher_g0.5": line(g, "gfisher_g0.5_c"), "gfisher_g1.0": line(g, "gfisher_g1.0_c"),
             "fisher_g0.5": line(g, "fisher_g0.5_c")}
    for k, v in extra.items():
        lines[k] = sorted(set(lines.get(k, [])) | set(v))
    out = {"lines": lines, "vs": {}}
    for base in [k for k in lines if k != "ghcol" and lines[k]]:
        out["vs"][base] = compare(lines["ghcol"], lines[base])
        out["vs"][base + "_reverse"] = [(b, k, None if s is None else -s) for b, k, s in
                                        ((b, k, interp_bpw(lines["ghcol"], k) and interp_bpw(lines["ghcol"], k) - b)
                                         for b, k in lines[base])]
    return out


def spine_lines(tag):
    """Existing in-harness lines from the spine / A3 / phase-2 runs (same eval, same rate proxy)."""
    ex = {}
    if tag == "qwen05b":
        s = J("spine_qwen05b.json")
        ex["gptq_row"] = [(v["entropy_bpw"], v["kl"]) for k, v in s.items() if k.startswith("gptq_ecsq_row_c")]
        ex["ecsq_row"] = [(v["entropy_bpw"], v["kl"]) for k, v in s.items() if k.startswith("ecsq_row_c")]
        a = J("a3_codec_qwen05b.json")
        ex["A3_gptq_fisher_g0.5"] = [(v["entropy_bpw"], v["kl"]) for k, v in a.items() if k.startswith("gptq_fisher_g0.5")]
        ex["A3_gptq_fisher_g1.0"] = [(v["entropy_bpw"], v["kl"]) for k, v in a.items() if k.startswith("gptq_fisher_g1.0")]
    else:
        p = J(f"phase2_family_{tag}.json")
        ex["gptq_row"] = [(v["proxy_bpw"], v["kl"]) for k, v in p.items() if k.startswith("B_ecsq_c")]
    return ex


def md_base(tag, r):
    L = r["lines"]
    s = [f"### {tag}: lines (bpw → WikiText KL)"]
    for k, v in L.items():
        if v:
            s.append(f"- `{k}`: {fmt_line(v)}")
    s.append("")
    s.append(f"saved bpw of `ghcol` at equal KL (positive = ghcol cheaper; n/a = outside the other line's range)")
    s.append("")
    keys = [k for k in r["vs"] if not k.endswith("_reverse")]
    s.append("| ghcol bpw | KL | " + " | ".join(f"vs `{k}`" for k in keys) + " |")
    s.append("|---|---|" + "---|" * len(keys))
    for i, (b, k, _) in enumerate(r["vs"][keys[0]]):
        row = [f"{b:.3f}", f"{k:.4f}"]
        for kk in keys:
            sv = r["vs"][kk][i][2]
            row.append("n/a" if sv is None else f"{sv:+.3f}")
        s.append("| " + " | ".join(row) + " |")
    return "\n".join(s)


def verdict_delta(tag, extra):
    g = J(f"ghcol_delta_{tag}.json")
    if not g:
        return None
    out = {}
    for key in ("chat_kl", "kl"):
        lines = {"dghcol": line(g, "dghcol_c", key), "dhcol": line(g, "dhcol_c", key), "dghcolW": line(g, "dghcolW_c", key),
                 "tdelta": line(g, "tdelta_c", key)}
        for k, v in extra.get(key, {}).items():
            lines[k] = sorted(set(lines.get(k, [])) | set(v))
        vs = {}
        for X in ("dghcol", "dghcolW", "dhcol"):
            if not lines[X]:
                continue
            for Y in lines:
                if Y != X and lines[Y]:
                    vs[f"{X}_vs_{Y}"] = compare(lines[X], lines[Y])
        out[key] = {"lines": lines, "vs": vs}
    return out


def delta_extra(tag):
    p = J(f"phase2_family_{tag}.json")
    ex = {"chat_kl": {"tdelta": [(v["proxy_bpw"], v["chat_kl"]) for k, v in p.items() if k.startswith("T_delta_c")]},
          "kl": {"tdelta": [(v["proxy_bpw"], v["kl"]) for k, v in p.items() if k.startswith("T_delta_c")]}}
    return ex


def md_delta(tag, r):
    s = []
    for key, rr in r.items():
        s.append(f"### {tag} Δ ({key}): lines (Δ bpw → {key})")
        for k, v in rr["lines"].items():
            if v:
                s.append(f"- `{k}`: {fmt_line(v)}")
        s.append("")
        for name, rows in rr["vs"].items():
            s.append(f"- {name}: " + "; ".join(f"{b:.3f} bpw ({k:.4f}): {'n/a' if sv is None else f'{sv:+.3f}'}" for b, k, sv in rows))
        s.append("")
    return "\n".join(s)


if __name__ == "__main__":
    out = {"base": {}, "delta": {}}
    md = []
    for tag in ("smol135", "smol360", "qwen05b"):
        r = verdict_base(tag, spine_lines(tag))
        if r:
            out["base"][tag] = r
            md.append(md_base(tag, r))
        d = verdict_delta(tag, delta_extra(tag))
        if d:
            out["delta"][tag] = d
            md.append(md_delta(tag, d))
    with open(os.path.join(RES, "ghcol_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n\n".join(md))
