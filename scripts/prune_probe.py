"""Probe S (sparsity / pruning): can we zero out weights and WIN on compression - with the MASK CHARGED?

Harness (self-consistent, CPU because another lead holds the GPU): fp16-rounded weights, fp32 compute on
CPU, our own CPU fp16-weight reference.  Distortion = KL(ref||model) + top-1 + PPL on the first 16
WikiText-2 test windows (2048 tok; the KL windows of every other probe; PPL here is over those 16
windows only, NOT the 148-window PPL of REPORT.md).  S2 adds the chat metric (KL / top-1 / NLL on the
fp16 instruct model's own greedy answers to the 48 fixed prompts, results/chat_set_smol135_instruct.json).
Every dense baseline we compare against is RE-RUN in this harness (no cross-harness comparisons).

Quantized/pruned set = all nn.Linear in the blocks (common.block_linears), rate = bits / #weights of
that set.  Hessians H = E[x x^T] of each linear's input, 128 x 2048 WikiText-2 train windows (seed 0),
non-sequential, as in V2 / GPTQ (common.collect_hessians).

RATE ACCOUNTING (everything charged; scripts/vq.Accounting = min(per-tensor, pooled-per-type tables),
32 b per used symbol per table - exactly the rule the scalar front was charged with):
  EXPLICIT code  = mask stream + kept-value stream + side info
      unstructured mask : entropy-coded 0/1 keep-mask  (~ H(rho) b/weight)
      n:m pattern       : entropy-coded pattern index per group of m (2:4: <= log2 C(4,2)/4 = 0.646 b/w)
      values (quant)    : entropy-coded integer symbols of the KEPT weights only
      values (fp16)     : kept fp16 values = 1 sign + 10 mantissa bits raw + entropy-coded 5-bit exponent
                          (the 16-bit raw cost is also reported)
  JOINT code (unstructured/dense, quantized values only) = ONE entropy-coded symbol stream over ALL
      weights, pruned weights = symbol 0.  The mask is paid inside P(symbol 0).  H(S) <= H(mask)+rho*H(v|kept)
      (S is a function of (mask, v)), so this is never worse; both are decodable, bpw = min(EXPLICIT, JOINT).
  side info: hcol grid 16 b/row (fp16 c*rms_row) + 16 b/column (fp16 diag(H)^-1/2 column factor);
             int4 g64: 16 b scale + 4 b zero per 64-group per row.

S1 (SmolLM2-135M base)
  hcol_c*            dense scalar: step_ij = fp16(c*rms_i) * fp16((h_j/geomean h)^-1/2), round  [best scalar, V2]
  ghcol_c*           dense GPTQ on the same hcol grid (= SparseGPT machinery at rho=1)  [error-feedback control]
  ecsq_row_c*        dense plain ECSQ, step c*rms_row (V1 baseline, harness continuity)
  {mag,wanda,sgpt}_r{rho}|_24          prune only, kept values fp16 (mag/wanda: per-output-row top-rho;
                     sgpt = SparseGPT OBS prune + update, 128-col adaptive mask blocks)
  sgpt_r{rho}|_24_hcol_c*              SparseGPT JOINT prune + quantize on the hcol grid (OBS feedback)
  wanda_r{rho}|_24_hcol_c*             Wanda mask, then round kept weights on the hcol grid (no feedback)
  gptq_int4_g64 / sgpt_*_int4_g64      int4 asym g64 (GPTQ / SparseGPT joint): fixed-rate and entropy-coded
S2 (SmolLM2-135M-Instruct given the fp16 base: Delta = W_inst - W_base)
  dist               |Delta| distribution: fraction within 1/2 and 1 quant step of 0, energy concentration
  dscal_c*_nD        dense ECSQ of Delta, step c*rms_row(Delta)  (V3 best scalar normalisation)
  dhcol_c*           dense ECSQ of Delta on the hcol grid (rms_row(Delta) x diag(H_inst)^-1/2)
  dgptq_c*_nD        dense GPTQ on Delta (instruct Hessian), phase-2 T_delta style
  dmag_r* / dwanda_r*            sparse Delta, top-rho per tensor by |Delta| or |Delta|*sqrt(h_j), fp16 values
  dmag_r*_c* / dwanda_r*_c*      sparse Delta + quantized kept values (nD grid), mask charged
  dsgpt_r*_c*                    SparseGPT on Delta (instruct H, OBS update) + nD grid

PRE-REGISTERED (written before any number): S1 PASS iff some sparse scheme saves >= 0.10 bpw vs the best
dense scalar line of this harness (hcol / ghcol) at equal KL at >= 2 operating points; FAIL if < 0.05
everywhere.  S2 PASS iff some sparse-Delta scheme saves >= 0.10 bpw vs the best dense-Delta line at equal
chat-KL at >= 2 points; FAIL if < 0.05 everywhere.

usage: python scripts/prune_probe.py --stage selftest|s1|s2 [--only a,b] [--groups ...]
"""
import argparse
import common
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="s1", choices=["selftest", "s1", "s2", "ctx"])
ap.add_argument("--dev", default="cpu", help="cpu (default: another lead holds the GPU) | mps")
ap.add_argument("--groups", default="")
ap.add_argument("--only", default="")
ap.add_argument("--windows", type=int, default=16)
args = ap.parse_args()
common.DEV = args.dev                       # common's eval/hessian functions read this global
from vq import Accounting, wtype, hdiag    # noqa: E402  (vq imports DEV at import time -> after override)

BASE = "HuggingFaceTB/SmolLM2-135M"
INST = "HuggingFaceTB/SmolLM2-135M-Instruct"
STATUS = os.path.join(ROOT, "logs", "prune_status.txt")
os.makedirs(os.path.dirname(STATUS), exist_ok=True)
seed_all(0)


def hb(p):
    return 0.0 if p <= 0 or p >= 1 else float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def load_m(rid):
    m = load_model(rid, dtype=torch.float16, device="cpu")
    if args.dev == "cpu":
        m = m.float()                       # fp16-rounded values, fp32 compute
    return m.to(args.dev)


def get_H(model, train_ids, path):
    if os.path.exists(path):
        return torch.load(path)
    H = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
    check_disk()
    torch.save(H, path)                     # shared tensors are saved once (~0.4 GB); cache, deleted at the end
    return H


# ----------------------------------------------------------------------------------------------------
# masks / grids
# ----------------------------------------------------------------------------------------------------
def row_topk_mask(score, rho):
    O, I = score.shape
    k = int(round(rho * I))
    M = torch.zeros(O, I, dtype=torch.bool)
    M.scatter_(1, score.topk(k, dim=1).indices, True)
    return M


def tensor_topk_mask(score, rho):
    k = int(round(rho * score.numel()))
    M = torch.zeros(score.numel(), dtype=torch.bool)
    if k > 0:
        M[score.reshape(-1).topk(k).indices] = True
    return M.reshape(score.shape)


def nm_mask(score, n=2, m=4):
    O, I = score.shape
    s = score.reshape(O, I // m, m)
    M = torch.zeros_like(s, dtype=torch.bool)
    M.scatter_(-1, s.topk(n, dim=-1).indices, True)
    return M.reshape(O, I)


def pattern_symbols(M, m=4):
    O, I = M.shape
    w = (2 ** torch.arange(m)).to(torch.int32)
    return (M.reshape(O, I // m, m).to(torch.int32) * w).sum(-1)


def rms_row(W):
    return W.float().pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)


def hcol_factor(H, gamma=1.0):
    hd = hdiag(H).clamp(min=1e-12)
    hd = hd / torch.exp(torch.log(hd).mean())
    return hd.pow(-gamma / 2).half().float()


def hcol_step(W, H, c, gamma=1.0, rowref=None):
    """V2 run_hcol grid: fp16(c*rms_row) x fp16 column factor; side 16 b/row + 16 b/col."""
    rr = rms_row(W if rowref is None else rowref)
    return (c * rr).half().float() * hcol_factor(H, gamma)[None, :], 16 * (W.shape[0] + W.shape[1])


def row_step(W, c, rowref=None):
    rr = rms_row(W if rowref is None else rowref)
    return (c * rr).half().float(), 16 * W.shape[0]


# ----------------------------------------------------------------------------------------------------
# SparseGPT (Frantar & Alistarh 2023) with optional joint quantization of the kept weights
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def sparsegpt(W, H, rho=1.0, nm=None, quant=None, blocksize=128, percdamp=0.01):
    """OBS prune + update.  rho: keep density (unstructured; mask chosen per 128-column block over the whole
    O x 128 block by w^2 / [H^-1]_jj^2 on the error-updated weights = reference 'adaptive mask selection');
    nm=(n,m): keep n of every m consecutive inputs (mask chosen when column i%m==0, reference behaviour).
    quant: None (kept values fp16) | ("grid", step (O,I)|(O,1)) | ("rtn", bits, group) (asym min/max params
    from the current block values incl. to-be-pruned ones, as the reference SparseGPT+GPTQ).
    With rho=1, nm=None, quant=grid this is exactly common.gptq(mode='grid') (selftest).
    Returns keep-mask M (bool), symbols S (int32, 0 at pruned for grid), deq Q, side bits (rtn only)."""
    W = W.float().clone()
    H = H.float().clone()
    O, I = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(I)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    S = torch.zeros(O, I, dtype=torch.int32)
    M = torch.ones(O, I, dtype=torch.bool)
    side = 0
    kind = None if quant is None else quant[0]
    if kind == "grid":
        stp = quant[1].float().expand(O, I)
    elif kind == "rtn":
        bits, g = quant[1], quant[2]
        maxq = 2 ** bits - 1
        assert blocksize % g == 0
    scale = zero = None
    for i1 in range(0, I, blocksize):
        i2 = min(i1 + blocksize, I)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        dg2 = torch.diag(Hinv1).pow(2)
        P1 = torch.zeros(O, cnt, dtype=torch.bool)          # True = pruned
        if nm is None and rho < 1:
            tmp = W1.pow(2) / dg2[None, :]
            k = int(round(tmp.numel() * (1 - rho)))
            if k > 0:
                P1.view(-1)[tmp.reshape(-1).topk(k, largest=False).indices] = True
        for i in range(cnt):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]
            if nm is not None and i % nm[1] == 0:
                n_, m_ = nm
                tmp = W1[:, i:i + m_].pow(2) / dg2[i:i + m_][None, :]
                P1.scatter_(1, i + tmp.topk(m_ - n_, dim=1, largest=False).indices, True)
            if kind == "rtn" and col % g == 0:
                blk = W1[:, i:i + g]
                xmin = blk.amin(1).clamp(max=0)
                xmax = blk.amax(1).clamp(min=0)
                xmax = torch.where(xmax - xmin == 0, xmax + 1e-8, xmax)
                scale = ((xmax - xmin) / maxq).half().float()
                zero = torch.round(-xmin / scale).clamp(0, maxq)
                side += O * (16 + bits)
            pr = P1[:, i]
            if kind is None:
                q = torch.where(pr, torch.zeros_like(w), w.half().float())
            elif kind == "grid":
                s = stp[:, col]
                qi = torch.round(torch.where(pr, torch.zeros_like(w), w) / s)
                q = qi * s
                S[:, col] = qi.to(torch.int32)
            else:
                qi = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
                q = torch.where(pr, torch.zeros_like(w), scale * (qi - zero))
                S[:, col] = qi.to(torch.int32)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        M[:, i1:i2] = ~P1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return M, S, Q, side


# ----------------------------------------------------------------------------------------------------
# accounting
# ----------------------------------------------------------------------------------------------------
def fp16_exponent(x):
    b = x.half().view(torch.int16).to(torch.int32) & 0xFFFF
    return (b >> 10) & 31


class Rate:
    """EXPLICIT and JOINT codes (see module doc) + an explicit per-stream breakdown (per-tensor tables)."""

    def __init__(self):
        self.ex, self.jo = Accounting(), Accounting()
        self.joint_ok = True
        self.b = {"mask": 0.0, "values": 0.0, "raw": 0.0, "side": 0.0, "fixed_values": 0.0}
        self.n = self.nnz = self.zero_kept = 0
        self.fixed_ok = False

    def add(self, name, rec):
        """rec: dict(mask=None|bool, nm=bool, vals='grid'|'fp16'|'int', sym=int32|None, deq, side, fixed_bits)"""
        n = rec["deq"].numel()
        wt = wtype(name)
        M = rec.get("mask")
        side = rec.get("side", 0)
        self.n += n
        self.b["side"] += side
        if M is None:
            nnz = n
            kept = slice(None)
        else:
            nnz = int(M.sum())
            if rec.get("mask_free"):              # seeded random mask: regenerated by the decoder, 0 bits
                ms = None
            elif rec.get("nm"):
                ms = pattern_symbols(M)
                self.ex.add_stream(wt + "#pat", ms.reshape(-1))
            else:
                ms = M.to(torch.int32)
                self.ex.add_stream(wt + "#mask", ms.reshape(-1))
            if ms is not None:
                self.b["mask"] += coded_bits(ms)[0]
        self.nnz += nnz
        if rec["vals"] == "fp16":
            v = rec["deq"][M] if M is not None else rec["deq"].reshape(-1)
            e = fp16_exponent(v)
            self.ex.add(wt + "#v", e, n, side + 11 * nnz)
            self.b["values"] += coded_bits(e)[0]
            self.b["raw"] += 11 * nnz
            self.joint_ok = False
        else:
            S = rec["sym"]
            sv = S[M] if M is not None else S.reshape(-1)
            self.ex.add(wt + "#v", sv, n, side)
            self.b["values"] += coded_bits(sv)[0]
            self.zero_kept += int((sv == 0).sum()) if rec["vals"] == "grid" else 0
            if rec["vals"] == "grid" and not rec.get("nm"):
                self.jo.add(wt, S.reshape(-1), n, side)
            else:
                self.joint_ok = False
            if rec.get("fixed_bits"):
                self.fixed_ok = True
                self.b["fixed_values"] += rec["fixed_bits"] * nnz

    def total(self):
        ex = self.ex.total()
        out = {"bpw_explicit": ex["bpw"], "table_choice_explicit": ex["table_choice"]}
        bpw = ex["bpw"]
        code = "explicit"
        if self.joint_ok:
            jo = self.jo.total()
            out["bpw_joint"] = jo["bpw"]
            if jo["bpw"] < bpw:
                bpw, code = jo["bpw"], "joint"
        rho = self.nnz / self.n
        out.update({"bpw": bpw, "code": code, "rho": rho, "H_rho": hb(rho),
                    "mask_bpw": self.b["mask"] / self.n, "values_bpw": (self.b["values"] + self.b["raw"]) / self.n,
                    "side_bpw": self.b["side"] / self.n, "n_weights": self.n,
                    "kept_rounded_to_zero_frac": self.zero_kept / max(self.nnz, 1)})
        if self.b["raw"]:
            out["bpw_fp16_raw16"] = (self.b["mask"] + 16 * self.nnz + self.b["side"]) / self.n
        if self.fixed_ok:
            out["bpw_fixed_rate"] = (self.b["mask"] + self.b["fixed_values"] + self.b["side"]) / self.n
        return out


# ----------------------------------------------------------------------------------------------------
# front comparison (in-harness dense line): bpw saved at equal distortion (log-linear interpolation)
# ----------------------------------------------------------------------------------------------------
def front_saved(front, b, k):
    """front: [(bpw, kl)] dense points. Returns (bpw_front_at_equal_kl - b, interpolated?) or (None, None)."""
    pts = sorted((x, y) for x, y in front if y and y > 0)
    if len(pts) < 2 or k is None or k <= 0:
        return None, None
    lk = math.log(k)
    for (b0, k0), (b1, k1) in zip(pts, pts[1:]):
        l0, l1 = math.log(k0), math.log(k1)
        if min(l0, l1) <= lk <= max(l0, l1) and l0 != l1:
            t = (lk - l0) / (l1 - l0)
            return (b0 + t * (b1 - b0)) - b, True
    # extrapolate with the nearest segment
    (b0, k0), (b1, k1) = (pts[0], pts[1]) if lk > math.log(pts[0][1]) else (pts[-2], pts[-1])
    l0, l1 = math.log(k0), math.log(k1)
    if l0 == l1:
        return None, None
    t = (lk - l0) / (l1 - l0)
    return (b0 + t * (b1 - b0)) - b, False


def status(lines):
    txt = "\n".join(lines)
    with open(STATUS, "w") as f:
        f.write(txt + "\n")
    print("=" * 100 + "\n" + txt + "\n" + "=" * 100, flush=True)


# ====================================================================================================
if args.stage == "selftest":
    tok = load_tok(BASE)
    W = torch.randn(96, 256)
    X = torch.randn(4096, 256) * torch.linspace(0.2, 3, 256)
    H = X.T @ X / X.shape[0]
    s = (0.4 * rms_row(W)).half().float()
    S0, Q0, _ = gptq(W, H, "grid", step=s)
    M1, S1_, Q1_, _ = sparsegpt(W, H, 1.0, None, ("grid", s))
    print("rho=1 grid == common.gptq:", torch.equal(S0, S1_), float((Q0 - Q1_).abs().max()))
    S0, Q0, sd0 = gptq(W, H, "rtn", 4, 64)
    M1, S1_, Q1_, sd1 = sparsegpt(W, H, 1.0, None, ("rtn", 4, 64))
    print("rho=1 rtn4 g64 == common.gptq:", torch.equal(S0, S1_), float((Q0 - Q1_).abs().max()), sd0 == sd1)
    for rho, nm in ((0.5, None), (1.0, (2, 4))):
        M, S, Q, _ = sparsegpt(W, H, rho, nm, ("grid", s))
        dens = float(M.float().mean())
        ok24 = bool((M.reshape(96, -1, 4).sum(-1) == 2).all()) if nm else None
        print(f"rho={rho} nm={nm}: density={dens:.4f} 2:4-exact={ok24} zeros-at-pruned={bool((Q[~M] == 0).all())}",
              f"sym0-at-pruned={bool((S[~M] == 0).all())}")
        r = Rate()
        r.add("layers.0.mlp.up_proj", {"mask": M, "nm": nm is not None, "vals": "grid", "sym": S, "deq": Q, "side": 0})
        print("   rate:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.total().items()})
    # mask cost sanity: Bernoulli(rho) mask ~ H(rho)
    for rho in (0.7, 0.5, 0.3, 0.1):
        Mr = torch.rand(1000, 1000) < rho
        print(f"Bernoulli mask rho={rho}: coded {coded_bits(Mr.to(torch.int32))[0] / 1e6:.4f} b/w vs H(rho)={hb(rho):.4f}")
    Mr = nm_mask(torch.rand(1000, 1000))
    print(f"random 2:4 pattern: coded {coded_bits(pattern_symbols(Mr))[0] / 1e6:.4f} b/w vs log2(6)/4={math.log2(6) / 4:.4f}")
    sys.exit(0)


# ====================================================================================================
if args.stage == "s1":
    log(f"S1 sparsity vs scalar; dev={args.dev}; disk free {check_disk():.2f} GB")
    tok = load_tok(BASE)
    test_ids = wt2_ids(tok, "test", "smollm2")
    train_ids = wt2_ids(tok, "train", "smollm2")
    model = load_m(BASE)
    snap = snapshot_weights(model)
    ref = get_ref(model, test_ids, os.path.join(DATA, f"prune_ref_smol135_{args.dev}.pt"))
    H = get_H(model, train_ids, os.path.join(DATA, "prune_H_smol135.pt"))
    restore_weights(model, snap)
    store = Store("prune_s1_smol135")
    store.put("_meta", {"model": BASE, "dev": args.dev, "compute": "fp32 on fp16-rounded weights" if args.dev == "cpu" else "fp16",
                        "eval": f"first {args.windows} WikiText-2 test windows x 2048: KL/top1 vs own fp16-weight ref; PPL over the same windows",
                        "hessian": "128 x 2048 WikiText-2 train windows, seed 0, non-sequential, percdamp 0.01",
                        "rate": "min(EXPLICIT mask+values+side, JOINT single stream pruned=0); vq.Accounting tables",
                        "n_quantized_set": sum(v.numel() for v in snap.values())})

    def apply(fn):
        r = Rate()
        t0 = time.time()
        for n, m in block_linears(model):
            rec = fn(n, snap[n].float())
            m.weight.data.copy_(rec["deq"].half().to(m.weight.device, m.weight.dtype))
            r.add(n, rec)
        return r.total(), round(time.time() - t0, 1)

    # -------- config builders --------
    def c_hcol(c):
        def fn(n, W):
            st, side = hcol_step(W, H[n], c)
            q = torch.round(W / st)
            return {"vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def c_ecsq_row(c):
        def fn(n, W):
            st, side = row_step(W, c)
            q = torch.round(W / st)
            return {"vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def c_sgpt(rho, nm, c=None, int4=False):
        def fn(n, W):
            if int4:
                M, S, Q, side = sparsegpt(W, H[n], rho, nm, ("rtn", 4, 64))
                return {"mask": None if (rho >= 1 and nm is None) else M, "nm": nm is not None, "vals": "int",
                        "sym": S, "deq": Q, "side": side, "fixed_bits": 4}
            if c is None:
                M, S, Q, _ = sparsegpt(W, H[n], rho, nm, None)
                return {"mask": M, "nm": nm is not None, "vals": "fp16", "deq": Q, "side": 0}
            st, side = hcol_step(W, H[n], c)
            M, S, Q, _ = sparsegpt(W, H[n], rho, nm, ("grid", st))
            return {"mask": None if (rho >= 1 and nm is None) else M, "nm": nm is not None, "vals": "grid",
                    "sym": S, "deq": Q, "side": side}
        return fn

    def c_mask(method, rho, nm, c=None):
        def fn(n, W):
            score = W.abs() if method == "mag" else W.abs() * hdiag(H[n]).clamp(min=0).sqrt()[None, :]
            M = nm_mask(score) if nm else row_topk_mask(score, rho)
            if c is None:
                return {"mask": M, "nm": bool(nm), "vals": "fp16", "deq": (W * M).half().float(), "side": 0}
            st, side = hcol_step(W, H[n], c)
            q = torch.round(W / st) * M
            return {"mask": M, "nm": bool(nm), "vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def pat(rho, nm):
        return "24" if nm else f"r{rho}"

    PLAN = [("fp16", None, {"kind": "fp16"})]
    for c in (0.35, 0.5, 0.7, 0.25, 1.0):
        PLAN.append((f"hcol_c{c}", c_hcol(c), {"kind": "hcol", "c": c, "gamma": 1.0}))
    for c in (0.35, 0.5, 0.7, 0.25, 1.0, 0.6, 0.85, 1.4):
        PLAN.append((f"ghcol_c{c}", c_sgpt(1.0, None, c), {"kind": "gptq_hcol", "c": c}))
    PLAN.append(("ecsq_row_c0.35", c_ecsq_row(0.35), {"kind": "ecsq_row", "c": 0.35}))
    # prune only (fp16 values)
    for rho, nm in ((0.5, None), (None, (2, 4)), (0.7, None), (0.4, None), (0.3, None)):
        for meth in ("sgpt", "wanda", "mag"):
            nmn = f"{meth}_{pat(rho, nm)}"
            cfg = {"kind": "prune_fp16", "method": meth, "rho": rho, "nm": nm}
            f = c_sgpt(rho if rho else 1.0, nm) if meth == "sgpt" else c_mask(meth, rho, nm)
            PLAN.append((nmn, f, cfg))
    # sparse + quant on the hcol grid
    for c in (0.35, 0.5, 0.7):
        for rho, nm in ((0.7, None), (0.5, None), (None, (2, 4))):
            PLAN.append((f"sgpt_{pat(rho, nm)}_hcol_c{c}", c_sgpt(rho if rho else 1.0, nm, c),
                         {"kind": "sgpt_hcol", "rho": rho, "nm": nm, "c": c}))
    for c in (0.35, 0.5):
        for rho, nm in ((0.7, None), (0.5, None), (None, (2, 4))):
            PLAN.append((f"wanda_{pat(rho, nm)}_hcol_c{c}", c_mask("wanda", rho, nm, c),
                         {"kind": "wanda_hcol", "rho": rho, "nm": nm, "c": c}))
    PLAN.append(("sgpt_r0.4_hcol_c0.35", c_sgpt(0.4, None, 0.35), {"kind": "sgpt_hcol", "rho": 0.4, "nm": None, "c": 0.35}))
    # int4 stacking (deployment format)
    PLAN.append(("gptq_int4_g64", c_sgpt(1.0, None, int4=True), {"kind": "int4", "rho": 1.0}))
    PLAN.append(("sgpt_24_int4_g64", c_sgpt(1.0, (2, 4), int4=True), {"kind": "int4", "nm": (2, 4)}))
    PLAN.append(("sgpt_r0.5_int4_g64", c_sgpt(0.5, None, int4=True), {"kind": "int4", "rho": 0.5}))

    def front():
        pts = []
        for k, v in store.d.items():
            if k.startswith("hcol_c") or k.startswith("ghcol_c"):
                pts.append((v["bpw"], v["kl"]))
        # best dense scalar envelope: drop dominated points
        pts.sort()
        env = [p for p in pts if not any(q[0] <= p[0] and q[1] < p[1] for q in pts if q != p)]
        return env

    only = set(args.only.split(",")) if args.only else None
    last = "none yet"
    for i, (name, fn, cfg) in enumerate(PLAN):
        if (only and name not in only) or store.has(name):
            continue
        check_disk()
        restore_weights(model, snap)
        if fn is None:
            acc, qs = {"bpw": 16.0}, 0.0
        else:
            acc, qs = apply(fn)
        ev = eval_lm(model, test_ids, max_windows=args.windows, ref=ref)
        rec = {"cfg": cfg, **acc, "quant_s": qs, **ev}
        sv, interp = front_saved(front(), rec["bpw"], ev["kl"]) if not name.startswith(("hcol", "ghcol")) else (None, None)
        rec["saved_bpw_vs_dense_front"] = sv
        rec["saved_interpolated"] = interp
        store.put(name, rec)
        extra = f" rho={acc.get('rho', 1):.3f} mask={acc.get('mask_bpw', 0):.3f} val={acc.get('values_bpw', 0):.3f} side={acc.get('side_bpw', 0):.3f} code={acc.get('code', '-')}"
        log(f"{name:24s} bpw={rec['bpw']:.3f}{extra} kl={ev['kl']:.4f} ppl16={ev['ppl']:.3f} top1={ev['top1']:.3f} "
            f"saved={'n/a' if sv is None else f'{sv:+.3f}'}{'' if interp in (None, True) else '(extrap)'} q={qs}s")
        last = f"{name}: bpw={rec['bpw']:.3f} kl={ev['kl']:.4f} saved_vs_front={'n/a' if sv is None else f'{sv:+.3f}'}"
        fr = front()
        status([f"[prune S1 | SmolLM2-135M | dev={args.dev}] {time.strftime('%H:%M:%S')}  progress {i + 1}/{len(PLAN)}",
                f"last: {last}",
                "dense front (bpw,KL): " + " ".join(f"({b:.2f},{k:.3f})" for b, k in fr),
                f"best sparse saved so far: " + str(max(((v.get('saved_bpw_vs_dense_front') or -9, k) for k, v in store.d.items()
                                                       if isinstance(v, dict) and v.get('saved_bpw_vs_dense_front') is not None), default=None)),
                f"disk free {disk_free_gb():.2f} GB"])
    restore_weights(model, snap)
    log("S1 done; disk free GB", round(disk_free_gb(), 2))


# ====================================================================================================
# S2: is the fine-tune DELTA sparse?  instruct = base(fp16) + code(Delta)
# ====================================================================================================
def gauss_energy_top(p):
    """fraction of E[x^2] carried by the top-p fraction of |x|, x ~ N(0,1)."""
    from statistics import NormalDist
    nd = NormalDist()
    t = nd.inv_cdf(1 - p / 2)
    return 2 * (t * nd.pdf(t) + (1 - nd.cdf(t)))


def laplace_energy_top(p):
    t = -math.log(p)
    return p * (t * t + 2 * t + 2) / 2


if args.stage == "s2":
    log(f"S2 delta sparsity; dev={args.dev}; disk free {check_disk():.2f} GB")
    tok = load_tok(INST)
    test_ids = wt2_ids(tok, "test", "smollm2")
    train_ids = wt2_ids(tok, "train", "smollm2")
    model = load_m(INST)
    snap = snapshot_weights(model)
    Bsd = LazySD(BASE)
    WB = {n: Bsd[f"model.{n}.weight"].to(torch.float16).float() for n in snap}
    ref = get_ref(model, test_ids, os.path.join(DATA, f"prune_ref_smol135_instruct_{args.dev}.pt"))
    chat = json.load(open(os.path.join(RES, "chat_set_smol135_instruct.json")))
    cpath = os.path.join(DATA, f"prune_chatref_smol135_instruct_{args.dev}.pt")
    if os.path.exists(cpath):
        chat_ref = torch.load(cpath)
    else:
        chat_ref = chat_hidden(model, chat)
        torch.save(chat_ref, cpath)
    H = get_H(model, train_ids, os.path.join(DATA, "prune_H_smol135_instruct.pt"))
    restore_weights(model, snap)

    # ---------------- |Delta| distribution (FACT: how sparse is it?) ----------------
    dstore = Store("prune_s2_delta_dist_smol135")
    if not dstore.has("global"):
        acc = {"n": 0, "zero_exact": 0, "dd": 0.0, "ww": 0.0}
        steps = {f"nW_c{c}": ("W", c) for c in (0.06, 0.12, 0.25)}
        steps.update({f"nD_c{c}": ("D", c) for c in (0.8, 1.6, 3.2)})
        within = {k: [0, 0] for k in steps}            # [#|D|<s/2, #|D|<s]
        KS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50)
        e_top = {k: 0.0 for k in KS}                     # raw |Delta|, per-tensor top-k
        e_top_rn = {k: 0.0 for k in KS}                  # row-normalised Delta/rms_row(Delta), pooled
        e_top_h = {k: 0.0 for k in KS}                   # diag(H)-weighted energy Delta^2 h_j, top-k by |Delta| sqrt(h_j)
        e_h_tot = 0.0
        hist_edges = torch.linspace(-8, 8, 161)
        hist = torch.zeros(160, dtype=torch.float64)
        m2 = m4 = 0.0
        rn_all = []
        per_type = {}
        for n in snap:
            WT = snap[n].float()
            D = WT - WB[n]
            N = D.numel()
            acc["n"] += N
            acc["zero_exact"] += int((D == 0).sum())
            acc["dd"] += float(D.pow(2).sum())
            acc["ww"] += float(WT.pow(2).sum())
            pt = per_type.setdefault(wtype(n), [0.0, 0.0])
            pt[0] += float(D.pow(2).sum()); pt[1] += float(WT.pow(2).sum())
            for k, (nrm, c) in steps.items():
                s = c * rms_row(WT if nrm == "W" else D)
                within[k][0] += int((D.abs() < s / 2).sum())
                within[k][1] += int((D.abs() < s).sum())
            a2 = D.pow(2).reshape(-1)
            srt = a2.sort(descending=True).values
            cs = srt.cumsum(0)
            for k in KS:
                e_top[k] += float(cs[max(int(round(k * N)) - 1, 0)])
            rn = (D / rms_row(D)).reshape(-1)
            rn_all.append(rn.half())
            hist += torch.histc(rn.clamp(-7.999, 7.999), 160, -8, 8).double()
            m2 += float(rn.pow(2).sum()); m4 += float(rn.pow(4).sum())
            hj = hdiag(H[n]).clamp(min=0)
            eh = (D.pow(2) * hj[None, :]).reshape(-1)
            e_h_tot += float(eh.sum())
            order = eh.sort(descending=True).values.cumsum(0)
            for k in KS:
                e_top_h[k] += float(order[max(int(round(k * N)) - 1, 0)])
        rn_all = torch.cat(rn_all).float()
        srt = rn_all.pow(2).sort(descending=True).values.cumsum(0)
        for k in KS:
            e_top_rn[k] = float(srt[max(int(round(k * rn_all.numel())) - 1, 0)] / srt[-1])
        Ntot = acc["n"]
        q = torch.quantile(rn_all.abs()[torch.randperm(rn_all.numel())[:2_000_000]], torch.tensor([0.5, 0.9, 0.99, 0.999]))
        out = {
            "rel_norm_delta": math.sqrt(acc["dd"] / acc["ww"]),
            "rel_norm_by_type": {k: math.sqrt(a / b) for k, (a, b) in per_type.items()},
            "frac_exactly_zero_fp16": acc["zero_exact"] / Ntot,
            "frac_within_half_step": {k: v[0] / Ntot for k, v in within.items()},
            "frac_within_one_step": {k: v[1] / Ntot for k, v in within.items()},
            "energy_frac_top_k_raw_per_tensor": {str(k): e_top[k] / acc["dd"] for k in KS},
            "energy_frac_top_k_row_normalised": {str(k): v for k, v in e_top_rn.items()},
            "hdiag_energy_frac_top_k": {str(k): e_top_h[k] / e_h_tot for k in KS},
            "gaussian_energy_frac_top_k": {str(k): gauss_energy_top(k) for k in KS},
            "laplace_energy_frac_top_k": {str(k): laplace_energy_top(k) for k in KS},
            "excess_kurtosis_row_normalised": (m4 / Ntot) / (m2 / Ntot) ** 2 - 3,
            "abs_quantiles_row_normalised": {"q50": float(q[0]), "q90": float(q[1]), "q99": float(q[2]), "q999": float(q[3])},
            "gaussian_abs_quantiles": {"q50": 0.6745, "q90": 1.6449, "q99": 2.5758, "q999": 3.2905},
            "hist_row_normalised": {"edges": hist_edges.tolist(), "counts": hist.tolist()},
            "n": Ntot}
        dstore.put("global", out)
        del rn_all
        gc.collect()
    dd = dstore.get("global")
    log(f"Delta: rel norm {dd['rel_norm_delta']:.4f}; exact zeros {dd['frac_exactly_zero_fp16']:.4f}; "
        f"within half-step {dd['frac_within_half_step']}; energy top10% {dd['energy_frac_top_k_raw_per_tensor']['0.1']:.3f} "
        f"(gauss {dd['gaussian_energy_frac_top_k']['0.1']:.3f}, laplace {dd['laplace_energy_frac_top_k']['0.1']:.3f}); "
        f"kurtosis {dd['excess_kurtosis_row_normalised']:.2f}")

    store = Store("prune_s2_smol135")
    store.put("_meta", {"base": BASE, "target": INST, "dev": args.dev,
                        "eval": f"WikiText-2 first {args.windows} windows KL/top1/PPL vs own fp16-weight instruct ref + chat KL/top1/NLL "
                                "on the instruct model's 48 greedy answers (results/chat_set_smol135_instruct.json)",
                        "hessian": "instruct model, 128 x 2048 WikiText-2 train windows",
                        "rate": "bits of the Delta stream (base fp16 is given) over the block linears; min(EXPLICIT, JOINT)"})

    def apply2(fn):
        r = Rate()
        t0 = time.time()
        for n, m in block_linears(model):
            WT = snap[n].float()
            rec = fn(n, WT - WB[n])
            rec["deq"] = rec["deq"]                      # = Delta_hat
            m.weight.data.copy_((WB[n] + rec["deq"]).half().to(m.weight.device, m.weight.dtype))
            r.add(n, rec)
        return r.total(), round(time.time() - t0, 1)

    def g_row(c):
        def fn(n, D):
            st, side = row_step(D, c)
            q = torch.round(D / st)
            return {"vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def g_hcol(c):
        def fn(n, D):
            st, side = hcol_step(D, H[n], c)
            q = torch.round(D / st)
            return {"vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def g_hcol_dz(c, theta):
        """dense DEAD-ZONE scalar coder on the hcol grid: 0 if |x/step| < theta, else round(x/step). No mask:
        one entropy-coded symbol stream. The scalar twin of 'prune small Delta + quantize the rest'."""
        def fn(n, D):
            st, side = hcol_step(D, H[n], c)
            x = D / st
            q = torch.where(x.abs() < theta, torch.zeros_like(x), torch.round(x))
            return {"vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def g_gptq(c, grid="row", rho=1.0):
        def fn(n, D):
            st, side = row_step(D, c) if grid == "row" else hcol_step(D, H[n], c)
            M, S, Q, _ = sparsegpt(D, H[n], rho, None, ("grid", st))
            return {"mask": None if rho >= 1 else M, "vals": "grid", "sym": S, "deq": Q, "side": side}
        return fn

    def g_sparse(method, rho, c=None, grid="row"):
        def fn(n, D):
            hj = hdiag(H[n]).clamp(min=0).sqrt()[None, :]
            score = D.abs() if method == "mag" else D.abs() * hj
            M = tensor_topk_mask(score, rho)
            if c is None:
                return {"mask": M, "vals": "fp16", "deq": (D * M).half().float(), "side": 0}
            st, side = row_step(D, c) if grid == "row" else hcol_step(D, H[n], c)
            q = torch.round(D / st) * M
            return {"mask": M, "vals": "grid", "sym": q.to(torch.int32), "deq": q * st, "side": side}
        return fn

    def g_dare(rho):
        """DARE-style: seeded random keep-mask (mask costs 0 bits: regenerated from the seed), rescale 1/rho."""
        def fn(n, D):
            import zlib
            g = torch.Generator().manual_seed(zlib.crc32(n.encode()))
            M = torch.rand(D.shape, generator=g) < rho
            v = (D / rho).half().float() * M
            return {"mask": M, "mask_free": True, "vals": "fp16", "deq": v, "side": 0}
        return fn

    PLAN = [("fp16", None, {"kind": "fp16"}), ("zero_bits_base", "zero", {"kind": "base"})]
    for c in (0.8, 1.6, 3.2, 6.4):
        PLAN.append((f"dscal_c{c}_nD", g_row(c), {"kind": "dense_ecsq_nD", "c": c}))
    for c in (0.8, 1.6, 3.2, 6.4):
        PLAN.append((f"dhcol_c{c}", g_hcol(c), {"kind": "dense_hcol", "c": c}))
    for c in (0.8, 1.6, 3.2):
        PLAN.append((f"dgptq_c{c}_nD", g_gptq(c), {"kind": "dense_gptq_nD", "c": c}))
    for rho in (0.1, 0.05, 0.02, 0.2, 0.01):
        PLAN.append((f"dmag_r{rho}", g_sparse("mag", rho), {"kind": "sparse_fp16", "method": "mag", "rho": rho}))
    for rho in (0.1, 0.05, 0.02):
        PLAN.append((f"dwanda_r{rho}", g_sparse("wanda", rho), {"kind": "sparse_fp16", "method": "wanda", "rho": rho}))
    for rho in (0.3, 0.1, 0.05):
        for c in (0.8, 1.6):
            PLAN.append((f"dmag_r{rho}_c{c}", g_sparse("mag", rho, c), {"kind": "sparse_q_nD", "method": "mag", "rho": rho, "c": c}))
            PLAN.append((f"dwanda_r{rho}_hc{c}", g_sparse("wanda", rho, c, "hcol"),
                         {"kind": "sparse_q_hcol", "method": "wanda", "rho": rho, "c": c}))
    for rho in (0.3, 0.1):
        for c in (0.8, 1.6):
            PLAN.append((f"dsgpt_r{rho}_hc{c}", g_gptq(c, "hcol", rho), {"kind": "sparsegpt_hcol", "rho": rho, "c": c}))
    for rho in (0.5, 0.1):
        PLAN.append((f"ddare_r{rho}", g_dare(rho), {"kind": "dare_fp16_free_mask", "rho": rho}))
    for c in (1.6, 3.2):
        PLAN.append((f"dghcol_c{c}", g_gptq(c, "hcol"), {"kind": "dense_gptq_hcol", "c": c}))
    # follow-up (added after the first sparse results): densify the dense lines + dead-zone scalar controls
    for c in (1.1, 2.2, 4.5):
        PLAN.append((f"dhcol_c{c}", g_hcol(c), {"kind": "dense_hcol", "c": c}))
    PLAN.append(("dghcol_c0.8", g_gptq(0.8, "hcol"), {"kind": "dense_gptq_hcol", "c": 0.8}))
    for c in (1.1, 1.6, 2.2):
        for th in (0.75, 1.0):
            PLAN.append((f"dhcoldz{th}_c{c}", g_hcol_dz(c, th), {"kind": "dense_hcol_deadzone", "c": c, "theta": th}))

    DENSE = ("dscal_", "dhcol_", "dgptq_", "dghcol_", "dhcoldz", "zero_bits_base")

    def front2(key):
        pts = [(v["bpw"], v[key]) for k, v in store.d.items() if k.startswith(DENSE) and isinstance(v, dict)]
        pts.sort()
        return [p for p in pts if not any(q[0] <= p[0] and q[1] < p[1] for q in pts if q != p)]

    only = set(args.only.split(",")) if args.only else None
    for i, (name, fn, cfg) in enumerate(PLAN):
        if (only and name not in only) or store.has(name):
            continue
        check_disk()
        restore_weights(model, snap)
        if fn is None:
            acc, qs = {"bpw": 16.0}, 0.0
        elif fn == "zero":
            for n, m in block_linears(model):
                m.weight.data.copy_(WB[n].half().to(m.weight.device, m.weight.dtype))
            acc, qs = {"bpw": 0.0}, 0.0
        else:
            acc, qs = apply2(fn)
            if cfg["kind"].startswith("dare"):
                acc["note"] = "mask regenerated from a seed (0 bits); values fp16 of Delta/rho"
        ev = eval_lm(model, test_ids, max_windows=args.windows, ref=ref)
        ev.update(eval_chat(model, chat, chat_ref, ref["head"]))
        rec = {"cfg": cfg, **acc, "quant_s": qs, **ev}
        if not name.startswith(DENSE) and name != "fp16":
            for key in ("chat_kl", "kl"):
                sv, ip = front_saved(front2(key), rec["bpw"], ev[key])
                rec[f"saved_bpw_vs_dense_{key}"] = sv
                rec[f"saved_interp_{key}"] = ip
        store.put(name, rec)
        sv = rec.get("saved_bpw_vs_dense_chat_kl")
        log(f"{name:22s} bpw={rec['bpw']:.3f} rho={acc.get('rho', 1):.3f} mask={acc.get('mask_bpw', 0):.3f} "
            f"val={acc.get('values_bpw', 0):.3f} code={acc.get('code', '-')} kl={ev['kl']:.4f} chat_kl={ev['chat_kl']:.4f} "
            f"chat_top1={ev['chat_top1']:.3f} saved(chat)={'n/a' if sv is None else f'{sv:+.3f}'} q={qs}s")
        best = max(((v.get("saved_bpw_vs_dense_chat_kl") or -9, k) for k, v in store.d.items()
                    if isinstance(v, dict) and v.get("saved_bpw_vs_dense_chat_kl") is not None), default=None)
        status([f"[prune S2 delta | SmolLM2-135M-Instruct | dev={args.dev}] {time.strftime('%H:%M:%S')}  progress {i + 1}/{len(PLAN)}",
                f"last: {name}: bpw={rec['bpw']:.3f} chat_kl={ev['chat_kl']:.4f} kl={ev['kl']:.4f} saved_vs_dense(chat)={sv}",
                "dense-Delta front (bpw,chatKL): " + " ".join(f"({b:.2f},{k:.3f})" for b, k in front2("chat_kl")),
                f"best sparse saved so far (chat-KL): {best}",
                f"Delta rel norm {dd['rel_norm_delta']:.4f}; frac |D|<s/2 {dd['frac_within_half_step']}",
                f"disk free {disk_free_gb():.2f} GB"])
    restore_weights(model, snap)
    log("S2 done; disk free GB", round(disk_free_gb(), 2))


# ====================================================================================================
# CTX: fairness check of the mask charge - give BOTH the dense coder and the sparse coder the same better
# (context-adaptive) entropy model and see whether the S1 verdict moves.  Rate only (distortion unchanged:
# the symbols are regenerated deterministically and the KL is taken from results/prune_s1_smol135.json).
# Code length of an adaptive Krichevsky-Trofimov (KT) coder on a context (column / row) with alphabet K:
#   L = log2 G(n + K/2) - log2 G(K/2) - sum_s [log2 G(c_s + 1/2) - log2 G(1/2)]   (exact, sequential, no side info)
# plus the per-tensor alphabet table (32 b per used symbol), as in the order-0 accounting.
# ====================================================================================================
def kt_bits_ctx(S, axis):
    """Total KT code length of integer matrix S coded with one adaptive model per column (axis=0) / row (axis=1)."""
    S = S.to(torch.int64)
    u, inv = torch.unique(S, return_inverse=True)
    K = u.numel()
    X = inv if axis == 0 else inv.T                      # contexts along dim 1
    n, C = X.shape                                       # n symbols per context, C contexts
    cnt = torch.zeros(C, K, dtype=torch.float64)
    cnt.scatter_add_(1, X.T.contiguous(), torch.ones(C, n, dtype=torch.float64))
    lg = torch.special.gammaln
    k2 = torch.tensor(K / 2, dtype=torch.float64)
    half = torch.tensor(0.5, dtype=torch.float64)
    L = (lg(torch.tensor(n, dtype=torch.float64) + k2) - lg(k2)) * C - (lg(cnt + 0.5) - lg(half)).sum()
    return float(L / math.log(2)) + 32 * K


if args.stage == "ctx":
    # SYMMETRIC context model (v2).  v1 (per-column KT over the full symbol alphabet) was unfair to the dense
    # coder: a K~50-100 symbol alphabet pays ~(K-1)/2*log2(n) bits of learning cost per 576-1536-sample column
    # (+0.4 bpw), a binary mask almost none.  v2 gives BOTH coders the same 'smart mask coder':
    #   zero-flag Z = [S == 0] coded with a per-column (or per-row) adaptive binary KT model
    #   + the nonzero symbols S[S != 0] with the usual order-0 per-tensor table.
    # For the dense coder Z is its own natural zeros; for the sparse joint stream Z = pruned u kept-rounded-to-0.
    # 2:4 (explicit): pattern index with a per-group-column adaptive KT (K<=16) + kept symbols order-0.
    log(f"CTX v2: symmetric context-adaptive zero-flag model; disk free {check_disk():.2f} GB")
    Bsd = LazySD(BASE)
    H = torch.load(os.path.join(DATA, "prune_H_smol135.pt"))
    names = list(H.keys())
    snap = {n: Bsd[f"model.{n}.weight"].to(torch.float16).float() for n in names}
    s1 = json.load(open(os.path.join(RES, "prune_s1_smol135.json")))
    store = Store("prune_ctx_smol135")
    store.d = {}
    store.put("_meta", {"what": "rate of the SAME S1 symbols under (a) order-0 per-tensor tables [o0] and (b) a "
                                "context-adaptive zero-flag model: per-column (zcol) / per-row (zrow) adaptive binary KT "
                                "for [S==0] + order-0 table for the nonzero symbols; 2:4: per-group-column adaptive KT "
                                "pattern + order-0 kept symbols. Same model offered to dense and sparse. KL copied from "
                                "prune_s1_smol135.json (symbols regenerated deterministically).",
                        "v1_discarded": "per-column KT over the full alphabet: +0.4 bpw learning cost for dense, ~0 for a "
                                        "binary mask -> asymmetric; replaced"})
    CF = [("ghcol_c%s" % c, 1.0, None, c) for c in (0.25, 0.35, 0.5, 0.6, 0.7, 0.85, 1.0, 1.4)]
    CF += [("sgpt_r0.7_hcol_c%s" % c, 0.7, None, c) for c in (0.35, 0.5, 0.7)]
    CF += [("sgpt_r0.5_hcol_c%s" % c, 0.5, None, c) for c in (0.35, 0.5, 0.7)]
    CF += [("sgpt_24_hcol_c%s" % c, 1.0, (2, 4), c) for c in (0.35, 0.5, 0.7)]
    for name, rho, nm, c in CF:
        t0 = time.time()
        b = {"o0": 0.0, "zcol": 0.0, "zrow": 0.0, "z_o0": 0.0, "z_col": 0.0, "z_row": 0.0, "nz_o0": 0.0,
             "mask_o0": 0.0, "mask_col": 0.0, "side": 0.0, "n": 0, "zeros": 0}
        for n in names:
            W = snap[n]
            st, side = hcol_step(W, H[n], c)
            M, S, Q, _ = sparsegpt(W, H[n], rho, nm, ("grid", st))
            b["side"] += side
            b["n"] += W.numel()
            if nm is None:
                Z = (S == 0).to(torch.int32)
                nzb = coded_bits(S[S != 0])[0]
                zc, zr = kt_bits_ctx(Z, 0), kt_bits_ctx(Z, 1)
                b["o0"] += coded_bits(S)[0]
                b["zcol"] += zc + nzb
                b["zrow"] += zr + nzb
                b["z_o0"] += coded_bits(Z)[0]
                b["z_col"] += zc
                b["z_row"] += zr
                b["nz_o0"] += nzb
                b["zeros"] += int(Z.sum())
                if rho < 1:
                    b["mask_o0"] += coded_bits(M.to(torch.int32))[0]
                    b["mask_col"] += kt_bits_ctx(M.to(torch.int32), 0)
            else:
                P = pattern_symbols(M)
                kb = coded_bits(S[M])[0]
                b["o0"] += coded_bits(P)[0] + kb
                b["zcol"] += kt_bits_ctx(P, 0) + kb
                b["zrow"] += kt_bits_ctx(P, 1) + kb
                b["mask_o0"] += coded_bits(P)[0]
                b["mask_col"] += kt_bits_ctx(P, 0)
        N = b["n"]
        out = {k: (v / N if k not in ("n",) else v) for k, v in b.items()}
        out["zeros"] = b["zeros"] / N
        for k in ("o0", "zcol", "zrow"):
            out[f"bpw_{k}"] = (b[k] + b["side"]) / N
        out["bpw_best_ctx"] = min(out["bpw_zcol"], out["bpw_zrow"], out["bpw_o0"])
        out["bpw_s1"] = s1[name]["bpw"]
        out["kl"] = s1[name]["kl"]
        store.put(name, out)
        log(f"{name:22s} s1={out['bpw_s1']:.3f} o0={out['bpw_o0']:.3f} zcol={out['bpw_zcol']:.3f} zrow={out['bpw_zrow']:.3f} "
            f"| zero-flag o0={out['z_o0']:.3f} col={out['z_col']:.3f} row={out['z_row']:.3f} nz={out['nz_o0']:.3f} "
            f"mask o0={out['mask_o0']:.3f} col={out['mask_col']:.3f} zeros={out['zeros']:.3f} kl={out['kl']:.4f} ({time.time() - t0:.0f}s)")
    log("CTX done")
