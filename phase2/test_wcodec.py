"""Unit tests for the real codec: every record kind must round-trip bit-exactly and the ANS
payload must land within a hair of the order-0 entropy (the phase-1 rate proxy)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numpy as np, torch
from wcodec import *
from wcodec import quant as Q
import common

torch.manual_seed(0); rng = np.random.default_rng(0)
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"

# 1. ANS round trip + rate vs entropy, incl. multi-chunk, constant and wide-range streams
for name, x in [("laplace", np.round(rng.laplace(0, 1.5, 3_000_001)).astype(np.int32)),
                ("const", np.full(1000, 7, np.int32)), ("wide", rng.integers(-3000, 3000, 50_000).astype(np.int32)),
                ("empty", np.zeros(0, np.int32)), ("single", np.array([-5], np.int32))]:
    b = ans_encode(x)
    y = ans_decode(b)
    assert np.array_equal(x, y), name
    H = order0_bits(x)
    print(f"ans {name:8s} n={x.size:8d} bytes={len(b):8d} bits/sym={8*len(b)/max(1,x.size):.4f} "
          f"order0+table={H/max(1,x.size):.4f}")

# 2. record kinds round trip through a real file
W = torch.randn(256, 512) * 0.02
W[3, 5] = 0.9  # outlier
Wref = W + torch.randn_like(W) * 0.002
Hs = (lambda X: X.T @ X / X.shape[0])(torch.randn(4096, 512))
wr = Writer({"test": 1})
# ll16 (I and XOR-P), raw16
wr.add("ll", *[enc_ll16(W)[0]], shape=W.shape, params=enc_ll16(W)[1], sections=enc_ll16(W)[2])
k, p, s = enc_ll16(Wref, ref=W); wr.add("llx", k, Wref.shape, p, s)
k, p, s = enc_raw16(W); wr.add("raw", k, W.shape, p, s)
# qrow ECSQ own step; P-frame add on ref lattice; SVC nest with ctx
step = Q.ecsq_step(W, 0.35); q = Q.ecsq(W, step)
k, p, s = enc_qrow(q, step); wr.add("q", k, W.shape, p, s)
qd = Q.ecsq(Wref, step) - q
k, p, s = enc_qrow(qd, step_mode="ref", acc="add"); wr.add("qp", k, W.shape, p, s)
q3 = Q.ecsq(W, step / 3) - 3 * q
k, p, s = enc_qrow(q3, step_mode="div:3", acc="nest:3", ctx_prev=q.numpy()); wr.add("qn", k, W.shape, p, s)
# fp16ref delta
sd = Q.ecsq_step(Wref, 0.1); qdelta = torch.round((Wref - W.half().float()) / sd).to(torch.int32)
k, p, s = enc_qrow(qdelta, sd, base="fp16ref"); wr.add("qd", k, W.shape, p, s)
# g128 GPTQ int4 packed + ans, compare with phase-1 gptq symbols
S, Qw, sc, z = Q.gptq(W, Hs, "rtn", 4, 128)
S1, Q1, _ = common.gptq(W, Hs, "rtn", 4, 128)
assert torch.equal(S, S1) and torch.equal(Qw, Q1), "gptq mismatch vs phase-1 implementation"
k, p, s = enc_g128(S, sc, z, 4, 128, packed=True); wr.add("gp", k, W.shape, p, s)
k, p, s = enc_g128(S, sc, z, 4, 128, packed=False); wr.add("ga", k, W.shape, p, s)
A, B = Q.lowrank(Wref - W.half().float(), 16)
k, p, s = enc_lowrank(A, B); wr.add("lr", k, W.shape, p, s)
path = os.path.join(tempfile.mkdtemp(), "t.wcdc"); nb = wr.write(path)
rd = Reader(path); assert rd.nbytes == nb == os.path.getsize(path)
# decode records individually with the right reference
import copy
def dec_one(name, R):
    r2 = copy.copy(rd); r2.records = [rd.by_name[name]]
    return decode(r2, ref={name: R} if R is not None else None, device=dev)[name]
e = lambda a, b: torch.equal(a.cpu(), b.cpu())
assert e(dec_one("ll", None).W, W.half()), "ll16"
assert e(dec_one("llx", TState(W=W.half().to(dev))).W, Wref.half()), "ll16 xor"
assert e(dec_one("raw", None).W, W.half()), "raw16"
tq = dec_one("q", None); assert e(tq.W, (q.float() * step).half()), "qrow"
tp = dec_one("qp", tq); assert e(tp.W, (Q.ecsq(Wref, step).float() * step).half()), "qrow add"
tn = dec_one("qn", tq); assert e(tn.W, (Q.ecsq(W, step / 3).float() * (step / 3)).half()), "qrow nest ctx"
td = dec_one("qd", TState(W=W.half().to(dev)))
assert e(td.W, (W.half().float() + qdelta.float() * sd).half()), "qrow fp16ref"
assert e(dec_one("gp", None).W, Qw.half()), "g128 packed"
assert e(dec_one("ga", None).W, Qw.half()), "g128 ans"
assert e(dec_one("lr", TState(W=W.half().to(dev))).W, (W.half().float() + A @ B).half()), "lowrank"
# 3. separable row x column lattice (quant.ghcol, REPORT §16): I-frame, P-frame inheriting the lattice, fp16ref delta
Hc = (lambda X: X.T @ X / X.shape[0])(torch.randn(4096, 512) * torch.linspace(0.05, 4, 512))  # spread diag(H)
Sg, Qg, rowg, colg = Q.ghcol(W, Hc, 0.35)
S1g, Q1g, _ = common.gptq(W, Hc, "grid", step=rowg * colg[None, :])
assert torch.equal(Sg, S1g) and torch.equal(Qg, Q1g), "ghcol != common.gptq on the same grid"
assert not torch.equal(Sg, Q.gptq(W, Hc, "grid", step=Q.ecsq_step(W, 0.35))[0]), "column grid had no effect"
wr2 = Writer({"test": 2})
k, p_, s_ = enc_qrow(Sg, rowg, col=colg); wr2.add("gI", k, W.shape, p_, s_)
qP, _, _, _ = Q.gptq(Wref - Qg, Hc, "grid", step=rowg * colg[None, :])
k, p_, s_ = enc_qrow(qP, step_mode="ref", acc="add", col_mode="ref"); wr2.add("gP", k, W.shape, p_, s_)
k, p_, s_ = enc_qrow(qP, step_mode="ref", acc="add"); wr2.add("gPbad", k, W.shape, p_, s_)   # forgot col=ref
Dg = Wref - W.half().float()
SgD, QgD, rowD, colD = Q.ghcol(Dg, Hc, 1.6)
k, p_, s_ = enc_qrow(SgD, rowD, base="fp16ref", col=colD); wr2.add("gD", k, W.shape, p_, s_)
path2 = os.path.join(tempfile.mkdtemp(), "t2.wcdc"); wr2.write(path2)
rd2 = Reader(path2)
def dec2(name, R):
    r2 = copy.copy(rd2); r2.records = [rd2.by_name[name]]
    return decode(r2, ref={name: R} if R is not None else None, device=dev)[name]
tgI = dec2("gI", None)
assert e(tgI.W, Qg.half()), "ghcol I-frame != encoder reconstruction"
assert e(tgI.W, (Sg.float() * (rowg * colg[None, :])).half()), "ghcol I-frame lattice"
tgP = dec2("gP", tgI)
assert e(tgP.W, ((Sg + qP).float() * (rowg * colg[None, :])).half()), "P-frame on the column lattice"
try:
    dec2("gPbad", tgI); raise AssertionError("P-frame that drops the column factors must not decode")
except ValueError:
    pass
try:
    dec2("gP", TState(Q=tgI.Q, step=tgI.step)); raise AssertionError("reference without column factors must fail")
except ValueError:
    pass
tgD = dec2("gD", TState(W=W.half().to(dev)))
assert e(tgD.W, (W.half().float() + SgD.float() * (rowD * colD[None, :])).half()), "ghcol fp16ref delta"
# degenerate inputs: a dead input column (h_j = 0) and an all-zero row must not produce inf / NaN / step 0
Hd = Hc.clone(); Hd[7, :] = 0; Hd[:, 7] = 0
Wz = W.clone(); Wz[11] = 0
Sz, Qz, rz, cz = Q.ghcol(Wz, Hd, 0.35)
assert torch.isfinite(Qz).all() and torch.isfinite(cz).all() and bool((rz > 0).all()) and bool((cz > 0).all())
assert bool((Sz[11] == 0).all()) and bool((Sz[:, 7] == 0).all()), "zero row / dead column must code as 0"
# int16 symbol contract (SmolLM2-360M incident: a 0.0014 column factor gave |q| = 34798, wrapped by the int16 decoders)
He = Hc.clone(); He[5, :] *= 3e4; He[:, 5] *= 3e4                         # one hyper-sensitive input channel
row_e, col_e = Q.hcol_grid(W, torch.diag(He), 0.25)
assert int((W / (row_e * col_e[None, :])).round().abs().max()) > 32767, "test channel not extreme enough"
Se, Qe, row_e2, col_e2 = Q.ghcol(W, He, 0.25)
assert Q.sym_fits_int16(Se) and int(Se.abs().max()) <= 2 ** 14 + 64, "ghcol must keep symbols int16-safe"
wr3 = Writer({"test": 3})
k, p_, s_ = enc_qrow(Se, row_e2, col=col_e2); wr3.add("ok", k, W.shape, p_, s_)
bad = torch.zeros(4, 8, dtype=torch.int32); bad[1, 2] = 40000
k, p_, s_ = enc_qrow(bad, torch.ones(4, 1)); wr3.add("bad", k, (4, 8), p_, s_)
path3 = os.path.join(tempfile.mkdtemp(), "t3.wcdc"); wr3.write(path3); rd3 = Reader(path3)
r3 = copy.copy(rd3); r3.records = [rd3.by_name["ok"]]
assert e(decode(r3, device=dev)["ok"].W, Qe.half()), "int16-safe ghcol record"
r3.records = [rd3.by_name["bad"]]
try:
    decode(r3, device=dev); raise AssertionError("out-of-int16 symbols must not decode silently")
except ValueError:
    pass
# 4. the DEFAULT coders (REPORT §17): Fisher-grid I-frame (gptq_sep on fisher_rowcol), P-frame on its inherited lattice
#    (GPTQ and plain rounding), plain-rounding Hessian-column delta (family_codec T_dhcol)
fst = {"w": {"row": torch.rand(256) + 0.1, "col": torch.rand(512) * torch.linspace(0.01, 3, 512) + 1e-3, "mean": 0.5}}
FGt = Q.fisher_grid({"w": Q.rms_row(W)[:, 0]}, fst, gamma=0.5)
rowf, colf0 = Q.fisher_rowcol(FGt["w"], 0.35)
Sf, Qf, rowf, colf = Q.gptq_sep(W, Hc, rowf, colf0)
latf = rowf * colf[None, :]
assert torch.equal((Sf.float() * latf).half(), Qf.half()), "gptq_sep output != symbols x lattice"
qPg, _, _, _ = Q.gptq(Wref - Qf, Hc, "grid", step=latf)
qPr = Q.ecsq(Wref - Sf.float() * latf, latf)
rowh, colh0 = Q.hcol_grid(Dg, torch.diag(Hc), 1.6)
colh, _ = Q.int16_safe_col(Dg, rowh, colh0)
Sh = Q.ecsq(Dg, rowh * colh[None, :])
wr4 = Writer({"test": 4})
for nm_, (q_, kw) in {"fI": (Sf, dict(step=rowf, col=colf)), "fPg": (qPg, dict(step_mode="ref", acc="add", col_mode="ref")),
                      "fPr": (qPr, dict(step_mode="ref", acc="add", col_mode="ref")),
                      "hD": (Sh, dict(step=rowh, base="fp16ref", col=colh))}.items():
    k, p_, s_ = enc_qrow(q_, **kw); wr4.add(nm_, k, W.shape, p_, s_)
# malformed records must fail loudly, never decode: unknown col mode, col="own" without its section, short col section
k, p_, s_ = enc_qrow(Sf, rowf, col=colf); p_ = dict(p_, col="bogus"); wr4.add("bad_mode", k, W.shape, p_, s_)
k, p_, s_ = enc_qrow(Sf, rowf, col=colf); s_ = {a: b for a, b in s_.items() if a != "col"}; wr4.add("bad_nosec", k, W.shape, p_, s_)
k, p_, s_ = enc_qrow(Sf, rowf, col=colf); s_ = dict(s_, col=s_["col"][:2]); wr4.add("bad_short", k, W.shape, p_, s_)
k, p_, s_ = enc_qrow(Sf, rowf, col=colf); s_ = dict(s_, step=s_["step"][:2]); wr4.add("bad_step", k, W.shape, p_, s_)
path4 = os.path.join(tempfile.mkdtemp(), "t4.wcdc"); wr4.write(path4); rd4 = Reader(path4)
def dec4(name, R):
    r2 = copy.copy(rd4); r2.records = [rd4.by_name[name]]
    return decode(r2, ref={name: R} if R is not None else None, device=dev)[name]
tfI = dec4("fI", None)
assert e(tfI.W, Qf.half()), "Fisher-grid I-frame != encoder reconstruction"
assert e(dec4("fPg", tfI).W, ((Sf + qPg).float() * latf).half()), "GPTQ P-frame on the Fisher lattice"
assert e(dec4("fPr", tfI).W, ((Sf + qPr).float() * latf).half()), "rounding P-frame on the Fisher lattice"
assert e(dec4("hD", TState(W=W.half().to(dev))).W, (W.half().float() + Sh.float() * (rowh * colh[None, :])).half()), \
    "rounding column-grid delta"
for bad_ in ("bad_mode", "bad_nosec", "bad_short", "bad_step"):
    try:
        dec4(bad_, None); raise AssertionError(f"{bad_}: malformed column-grid record must not decode")
    except ValueError:
        pass
print("record sizes:", {r["name"]: sum(n for _, n in r["sec"].values()) for r in rd.records})
print("column-lattice records:", {r["name"]: sum(n for _, n in r["sec"].values()) for r in rd2.records})
gp = sum(n for _, n in rd.by_name["gp"]["sec"].values()) * 8 / W.numel()
print(f"g128 packed bpw = {gp:.4f} (theory 4 + 20/128 = {4 + 20/128:.4f})")
print("ALL TESTS PASSED on", dev)
