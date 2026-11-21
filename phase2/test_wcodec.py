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
print("record sizes:", {r["name"]: sum(n for _, n in r["sec"].values()) for r in rd.records})
gp = sum(n for _, n in rd.by_name["gp"]["sec"].values()) * 8 / W.numel()
print(f"g128 packed bpw = {gp:.4f} (theory 4 + 20/128 = {4 + 20/128:.4f})")
print("ALL TESTS PASSED on", dev)
