"""Seam test: the browser decoder (docs/js/codec.js) vs the Python decoder (wcodec.codecs) on synthetic files that
cover every qrow lattice the codec writes -- row grid (phase 2) and the separable row x column grid of REPORT §16/§17
(own column factors, inherited column factors in a P-frame, fp16-ref delta with own factors) -- plus extreme values
(fp16 subnormal / overflow outputs, int16-limit lattice indices) and malformed records that must FAIL LOUDLY in both.

Digests are the demo's (phase2/demo_build.py): per record SHA-256 of the lattice index Q (int16 LE) or the fp16 bit
pattern, per file SHA-256 of the concatenated record digests; weights = SHA-256 of the rebuilt fp16 bits.
usage: .venv/bin/python phase2/test_codec_js.py      (needs node >= 18 on PATH); exits non-zero on any mismatch
"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from wcodec import Writer, Reader, decode, enc_qrow, enc_ll16, TState
from wcodec import quant as Q
from wcodec.transcode import transcode


def H(b):
    return hashlib.sha256(b).digest()


def bits16(t):
    return t.detach().cpu().contiguous().to(torch.float16).view(torch.int16).numpy().view(np.uint16).tobytes()


def digests(rd, st):
    sym, w = [], []
    for r in rd.records:
        s = st[r["name"]]
        if r["kind"] == "qrow":
            q = s.Q.cpu().numpy()
            assert q.min() >= -32768 and q.max() <= 32767, f"{r['name']}: Q outside int16 (digest would wrap)"
            sym.append(H(q.astype("<i2").tobytes()))
        else:
            sym.append(H(bits16(s.W)))
        w.append(H(bits16(s.W)))
    return {"sha256": H(b"".join(sym)).hex(), "w_sha256": H(b"".join(w)).hex()}


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    tmp = tempfile.mkdtemp(prefix="wc_jsseam_")
    shapes = {"a": (64, 96), "b": (96, 64), "odd": (3, 5), "big": (128, 257)}
    W = {k: torch.randn(*s, generator=g) * 0.02 for k, s in shapes.items()}
    W["a"][3, 5] = 0.9
    WT = {k: v + torch.randn(*v.shape, generator=g) * 0.003 for k, v in W.items()}
    Hs = {k: (lambda X: X.T @ X / X.shape[0])(torch.randn(2048, s[1], generator=g) * torch.linspace(0.05, 4, s[1]))
          for k, s in shapes.items()}
    files = {n: Writer({"test": n}) for n in ("B", "rI", "rP", "cI", "cPg", "cPr", "cD", "xI",
                                              "bad_mode", "bad_nosec", "bad_short", "bad_int16", "bad_Pnocol",
                                              "bad_Pdrop", "bad_Pwrap", "wrapI")}
    for k in shapes:
        Wk, WTk = W[k], WT[k]
        kind, p, s = enc_ll16(Wk.half()); files["B"].add(k, kind, Wk.shape, p, s)
        # row grid (phase 2): I-frame + closed-loop P-frame
        st = Q.ecsq_step(Wk, 0.35); q = Q.ecsq(Wk, st)
        kind, p, s = enc_qrow(q, st); files["rI"].add(k, kind, Wk.shape, p, s)
        kind, p, s = enc_qrow(Q.ecsq(WTk, st) - q, step_mode="ref", acc="add"); files["rP"].add(k, kind, Wk.shape, p, s)
        # column grid (Fisher-style factors) + GPTQ / rounding P-frames on the inherited lattice
        stats = {k: {"row": torch.rand(Wk.shape[0], generator=g) + 0.1,
                     "col": torch.rand(Wk.shape[1], generator=g) * torch.linspace(0.01, 3, Wk.shape[1]) + 1e-3, "mean": 0.5}}
        FG = Q.fisher_grid({k: Q.rms_row(Wk)[:, 0]}, stats, gamma=0.5)
        S, Qd, row, col = Q.gptq_sep(Wk, Hs[k], *Q.fisher_rowcol(FG[k], 0.35))
        lat = row * col[None, :]
        kind, p, s = enc_qrow(S, row, col=col); files["cI"].add(k, kind, Wk.shape, p, s)
        qg, _, _, _ = Q.gptq(WTk - Qd, Hs[k], "grid", step=lat)
        kind, p, s = enc_qrow(qg, step_mode="ref", acc="add", col_mode="ref"); files["cPg"].add(k, kind, Wk.shape, p, s)
        qr = Q.ecsq(WTk - S.float() * lat, lat)
        kind, p, s = enc_qrow(qr, step_mode="ref", acc="add", col_mode="ref"); files["cPr"].add(k, kind, Wk.shape, p, s)
        # plain-rounding column-grid delta against the fp16 base
        D = WTk - Wk.half().float()
        rowd, cold0 = Q.hcol_grid(D, torch.diag(Hs[k]), 1.6)
        cold, _ = Q.int16_safe_col(D, rowd, cold0)
        kind, p, s = enc_qrow(Q.ecsq(D, rowd * cold[None, :]), rowd, base="fp16ref", col=cold)
        files["cD"].add(k, kind, Wk.shape, p, s)
        # extremes: int16-limit indices, fp16 overflow (-> inf) and subnormal outputs, on an own column lattice
        O, I = Wk.shape
        qx = torch.randint(-32767, 32768, (O, I), generator=g, dtype=torch.int32)
        qx[0, 0], qx[-1, -1] = 32767, -32768
        rowx = torch.tensor([2.0 ** e for e in np.linspace(-24, 2, O)], dtype=torch.float32).half().float()[:, None]
        colx = (torch.rand(I, generator=g) * 4 + 2.0 ** -14).half().float()
        kind, p, s = enc_qrow(qx, rowx, col=colx); files["xI"].add(k, kind, Wk.shape, p, s)
        # malformed records (each must raise in the Python AND the JS decoder)
        kind, p, s = enc_qrow(S, row, col=col)
        files["bad_mode"].add(k, kind, Wk.shape, dict(p, col="bogus"), s)
        files["bad_nosec"].add(k, kind, Wk.shape, p, {a: b for a, b in s.items() if a != "col"})
        files["bad_short"].add(k, kind, Wk.shape, p, dict(s, col=s["col"][:2]))
        qb = S.clone(); qb[0, 0] = 40000
        kind, p, s = enc_qrow(qb, row, col=col); files["bad_int16"].add(k, kind, Wk.shape, p, s)
        kind, p, s = enc_qrow(qg, step_mode="ref", acc="add", col_mode="ref"); files["bad_Pnocol"].add(k, kind, Wk.shape, p, s)
        kind, p, s = enc_qrow(qg, step_mode="ref", acc="add"); files["bad_Pdrop"].add(k, kind, Wk.shape, p, s)
        qw = torch.full((O, I), 32000, dtype=torch.int32)
        kind, p, s = enc_qrow(qw, row, col=col); files["wrapI"].add(k, kind, Wk.shape, p, s)
        kind, p, s = enc_qrow(torch.full((O, I), 1000, dtype=torch.int32), step_mode="ref", acc="add", col_mode="ref")
        files["bad_Pwrap"].add(k, kind, Wk.shape, p, s)
    for n, w in files.items():
        w.write(os.path.join(tmp, n + ".orig.wcdc"))
        transcode(os.path.join(tmp, n + ".orig.wcdc"), os.path.join(tmp, n + ".wcdc"))      # the JS reads rans16
    chains = [["rI", "rP"], ["cI", "cPg"], ["cI", "cPr"], ["B", "cD"], ["xI"]]
    py, fails = {}, []
    for ch in chains:
        prev = None
        for n in ch:
            out = {}
            for suf in (".orig", ""):               # constriction original and its rans16 twin must agree in Python
                rd = Reader(os.path.join(tmp, n + suf + ".wcdc"))
                st = decode(rd, ref=prev, device="cpu", keep_state=True)
                out[suf] = digests(rd, st)
            if out[".orig"] != out[""]:
                fails.append(f"{n}: constriction vs rans16 differ in Python")
            py[n] = out[""]
            prev = st
    bad = [["bad_mode", None], ["bad_nosec", None], ["bad_short", None], ["bad_int16", None], ["bad_Pnocol", "rI"],
           ["bad_Pdrop", "cI"], ["bad_Pwrap", "wrapI"]]
    py_err = {}
    for n, refn in bad:
        rd = Reader(os.path.join(tmp, n + ".wcdc"))
        ref = None if refn is None else decode(Reader(os.path.join(tmp, refn + ".wcdc")), device="cpu", keep_state=True)
        for r in rd.records:
            r2 = copy.copy(rd); r2.records = [r]
            try:
                decode(r2, ref=ref, device="cpu"); py_err[f"{n}/{r['name']}"] = None
            except (ValueError, NotImplementedError) as e:
                py_err[f"{n}/{r['name']}"] = str(e)
    json.dump({"chains": chains, "bad": bad}, open(os.path.join(tmp, "fixture.json"), "w"))
    res = subprocess.run(["node", os.path.join(ROOT, "phase2", "test_codec_js.mjs"), tmp], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr); sys.exit("node failed")
    js = json.loads(res.stdout)
    for n, d in py.items():
        ok = js["files"].get(n) == d
        print(f"  {'ok  ' if ok else 'FAIL'} {n:6s} symbols + rebuilt fp16 weights: JS == Python  ({d['w_sha256'][:16]})")
        if not ok:
            fails.append(f"{n}: JS {js['files'].get(n)} vs Python {d}")
    for key, msg in js["errors"].items():
        pmsg = py_err.get(key)
        # the Python decoder accumulates P-frame indices in int32 (exact), the JS one in int16 -> only JS must refuse
        py_must = not key.startswith("bad_Pwrap/")
        ok = msg is not None and (pmsg is not None or not py_must)
        print(f"  {'ok  ' if ok else 'FAIL'} {key:16s} JS raises: {msg!s:.70}" + ("" if not py_must else f" | Python raises: {pmsg!s:.50}"))
        if not ok:
            fails.append(f"{key}: JS {msg!r}, Python {pmsg!r}")
    shutil.rmtree(tmp)
    print("ALL OK" if not fails else f"{len(fails)} FAILED:\n" + "\n".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
