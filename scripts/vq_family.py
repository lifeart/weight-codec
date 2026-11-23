"""Probe V3: does vector quantization compose with delta / family coding?

V3a  HUB scenario (fp16 base available, code only the fine-tune's Delta = W_inst - W_base):
     dscal_c*_n{W,D}        entropy-coded scalar Delta (ECSQ), step c * rms_row(W_inst | Delta)
     dgptq_c*               scalar Delta with GPTQ error feedback (instruct Hessian) = phase-2 T_delta
     decvqnn_d*_l*_n{W,D}   ECVQ codebook on row-normalised Delta vectors (one per weight type), nearest-
                            centroid encoding + lattice escape (decvq_*: Lagrangian/RDO assignment)
     Rate = bits of the Delta stream only (indices/symbols + codebooks + 16 b/row scale).
V3b  FAMILY-FROM-SCRATCH scenario (no fp16 base; store base AND instruct lossy):
     pscal_c*   base on the uniform grid s = c*rms_row(W_base) [I-frame q_B]; instruct on the SAME
                grid, stream = q_I - q_B (closed loop; = phase-1 c2 pframe_rtn)
     pvq_d*_l*  ONE ECVQ codebook per weight type fitted on the BASE ("sort/cluster once"); base
                indices i_B (nearest-centroid + lattice escape, like the scalar grid's rounding); instruct normalised by the base row scales and assigned to the SAME
                codebook (i_I); stream = rank of c[i_I] among the T nearest neighbours of c[i_B]
                (decoder recomputes the neighbour lists from the codebook; escape + raw index
                for rank >= T).  Codebook + scales stored ONCE for the pair.
     Reported: pair bits (base + instruct streams + shared side info), base KL vs fp16 base,
     instruct WikiText KL + chat KL vs fp16 instruct, instruct-alone conditional rate, and the
     rate the instruct would cost coded independently on the shared codebook.
PRE-REGISTERED (before results):
  V3a PASS iff some d>=2 decvq line needs >= 0.10 bpw less Delta rate than the best scalar Delta
  line (dscal / dgptq) at equal chat-KL, at >= 2 chat-KL levels.  FAIL if < 0.05 everywhere.
  V3b PASS iff pvq (d>=2) reaches the same (base KL, instruct chat-KL) as pscal at >= 0.10 bpw
  lower PAIR rate per weight.  Prior: the index P-frame is worse than the scalar difference
  stream because a VQ cell is larger than a scalar cell (a fine-tune move crosses fewer cell
  borders per dimension, but each crossing costs a whole neighbour-rank symbol).

usage: python scripts/vq_family.py [--groups a,b] [--only names]
"""
import argparse
from common import *
from vq import *

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--target", default="HuggingFaceTB/SmolLM2-135M-Instruct")
ap.add_argument("--tag", default="smol135")
ap.add_argument("--tok_tag", default="smollm2")
ap.add_argument("--groups", default="a,b")
ap.add_argument("--only", default="")
ap.add_argument("--T", type=int, default=64, help="neighbour list length for the index P-frame")
ap.add_argument("--gain", type=int, default=0, help="1: zero-bit per-row gain fix of centroid shrinkage (V1 gain test)")
args = ap.parse_args()
seed_all(0)
log(f"disk free {check_disk():.2f} GB")
tok = load_tok(args.target)
test_ids = wt2_ids(tok, "test", args.tok_tag)
train_ids = wt2_ids(tok, "train", args.tok_tag)
model = load_model(args.target)
snapI = snapshot_weights(model)
ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}_instruct.pt"))
chat = json.load(open(os.path.join(RES, f"chat_set_{args.tag}_instruct.json")))
chat_ref = torch.load(os.path.join(DATA, f"chatref_{args.tag}_instruct.pt"))
SB = LazySD(args.base)
snapB = {n: SB[f"model.{n}.weight"].to(torch.float16) for n in snapI}
class _LazyDelta(dict):
    """W_inst - W_base computed per access (keeps RAM flat for 0.5 B models)."""
    def __getitem__(self, n):
        return snapI[n].float() - snapB[n].float()


DELTA = _LazyDelta({n: None for n in snapI})
N = sum(w.numel() for w in snapI.values())
store = Store(f"vq_v3_{args.tag}")
store.put("_meta", {"base": args.base, "target": args.target, "n_quantized_set": N, "seed": 0, "T": args.T,
                    "distortion": "instruct: WikiText-2 PPL, KL/top1 (16 windows) + chat NLL/KL/top1 on 48 own "
                                  "greedy answers vs fp16 instruct; base (V3b): WikiText-2 PPL + KL vs fp16 base"})
_mB = {}


def base_model():
    if "m" not in _mB:
        m = load_model(args.base)
        _mB["m"] = m
        _mB["snap"] = snapshot_weights(m)
        _mB["ref"] = get_ref(m, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
    return _mB["m"], _mB["snap"], _mB["ref"]


_H = {}


def hessI():
    if "H" not in _H:
        restore_weights(model, snapI)
        _H["H"] = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
    return _H["H"]


rWI = {n: row_scale(snapI[n]) for n in snapI}
rWB = {n: row_scale(snapB[n]) for n in snapI}
rD = {n: row_scale(DELTA[n]) for n in snapI}


def eval_inst(acc_bpw_override=None):
    ev = eval_lm(model, test_ids, ref=ref)
    ev.update(eval_chat(model, chat, chat_ref, ref["head"]))
    return ev


def put(name, rec, ev):
    if args.gain and rec.get("kind") in ("decvq", "pvq"):
        name = name + "_gain"
    rec.update(ev)
    store.put(name, rec)
    msg = " ".join(f"{k}={rec[k]:.4f}" for k in ("bpw", "pair_bpw", "codebook_bpw", "kl", "chat_kl", "chat_top1",
                                                 "base_kl", "p_same") if k in rec and rec[k] is not None)
    log(f"{name:28s} {msg} | disk {disk_free_gb():.1f}GB")


# ------------------------------------------------------------------ V3a: hub, code Delta
def run_dscal(name, c, norm):
    acct = Accounting()

    def fn(n, W):
        r = rWI[n] if norm == "W" else rD[n]
        s = (c * r)
        q = torch.round(DELTA[n] / s)
        dq = snapB[n].float() + q * s
        sym = q.to(torch.int32)
        acct.add(wtype(n), sym, W.numel(), 16 * W.shape[0], W, dq)
        return sym, dq, 16 * W.shape[0]
    acc = quantize_model(model, snapI, fn)
    put(name, {"kind": "dscal", "c": c, "norm": norm, **acct.total()}, eval_inst())


def run_dgptq(name, c, norm="W"):
    acct = Accounting()

    def fn(n, W):
        s = (c * (rWI[n] if norm == "W" else rD[n]))
        q, D, _ = gptq(DELTA[n], hessI()[n], "grid", step=s)
        dq = snapB[n].float() + D
        acct.add(wtype(n), q, W.numel(), 16 * W.shape[0], W, dq)
        return q, dq, 16 * W.shape[0]
    quantize_model(model, snapI, fn)
    put(name, {"kind": "dgptq", "c": c, "norm": norm, **acct.total()}, eval_inst())


def run_dgptq_lat(name, kind, c):
    """GPTQ (instruct Hessian) on the D4/E8 lattice of the row-normalised (nD) Delta."""
    acct = Accounting()
    s = c / 2 ** 0.25 if kind == "d4" else c

    def fn(n, W):
        streams, D = gptq_lattice(DELTA[n], hessI()[n], kind, s)
        dq = snapB[n].float() + D
        acct.add(wtype(n), streams[0], W.numel(), 16 * W.shape[0], W, dq)
        side = 16 * W.shape[0]
        for j, st in enumerate(streams[1:]):
            side += acct.add_stream(f"{wtype(n)}#s{j}", st)
        return streams[0], dq, side
    quantize_model(model, snapI, fn)
    put(name, {"kind": "dgptq_lat", "lat": kind, "c": c, "norm": "D", **acct.total()}, eval_inst())


_init = {}


def nn_entries(cbs):
    """nearest-centroid encoding (penalty only in the escape decision); V1: RDO assignment hurts KL."""
    return {k: {**v, "pen": None, "dec": v["pen"]} for k, v in cbs.items()}


def run_decvq(name, d, K, lam, norm, nn=True, seed=0):
    acct = Accounting()
    rf = (lambda n: rWI[n]) if norm == "W" else (lambda n: rD[n])
    cbs, fit = fit_codebooks(DELTA, d, K, "type", method="ecvq", lam=lam, init_cache=_init, rfn=rf,
                             cache_tag=f"delta_{norm}", seed=seed)
    if nn:
        cbs = nn_entries(cbs)
    fn = make_vq_fn(cbs, d, "type", acct, rfn=rf, base=lambda n: snapB[n].float(), gain=bool(args.gain))
    quantize_model(model, snapI, fn)
    put(name, {"kind": "decvq", "d": d, "K": K, "lam": lam, "norm": norm, "nn": nn, "seed": seed, "gain": bool(args.gain), "fit_s": fit["fit_s"],
               **acct.total()}, eval_inst())


def run_dlat(name, kind, c, norm):
    """Lattice VQ (D4 / E8, codebook-free, unbiased rounding) of the row-normalised Delta."""
    acct = Accounting()
    d = 4 if kind == "d4" else 8
    s = c / 2 ** 0.25 if kind == "d4" else c          # same point density as the scalar grid of step c

    def fn(n, W):
        r = rWI[n] if norm == "W" else rD[n]
        V = (DELTA[n] / r).reshape(-1, d).to(DEV)
        streams, rec = lattice_encode(V, kind, s)
        dq = snapB[n].float() + rec.reshape(W.shape).cpu() * r
        head = streams[0].cpu()
        acct.add(wtype(n), head, W.numel(), 16 * W.shape[0], W, dq)
        side = 16 * W.shape[0]
        for j, st in enumerate(streams[1:]):
            side += acct.add_stream(f"{wtype(n)}#s{j}", st.cpu())
        return head, dq, side
    quantize_model(model, snapI, fn)
    put(name, {"kind": "dlat", "lat": kind, "d": d, "c": c, "norm": norm, **acct.total()}, eval_inst())


# ------------------------------------------------------------------ V3b: lossy family, shared codebook
def eval_base(fnB):
    """Quantize the base with fnB (returns its accounting) and evaluate it vs fp16 base."""
    mB, sB, refB = base_model()
    quantize_model(mB, sB, fnB)
    ev = eval_lm(mB, test_ids, ref=refB)
    restore_weights(mB, sB)
    return {"base_kl": ev["kl"], "base_ppl": ev["ppl"], "base_top1": ev["top1"]}


def run_pscal(name, c):
    aB, aI, aInd = Accounting(), Accounting(), Accounting()
    qB = {}

    def fB(n, W):
        s = c * rWB[n]
        q = torch.round(snapB[n].float() / s)
        qB[n] = q
        dq = q * s
        aB.add(wtype(n), q.to(torch.int32), W.numel(), 16 * W.shape[0])
        return q.to(torch.int32), dq, 16 * W.shape[0]

    evB = eval_base(fB)

    def fI(n, W):
        s = c * rWB[n]
        q = torch.round(W / s)
        diff = (q - qB[n]).to(torch.int32)
        aI.add(wtype(n), diff, W.numel(), 0, W, q * s)
        aInd.add(wtype(n), q.to(torch.int32), W.numel(), 0)
        return diff, q * s, 0
    quantize_model(model, snapI, fI)
    tB, tI, tInd = aB.total(), aI.total(), aInd.total()
    rec = {"kind": "pscal", "c": c, "base_bpw": tB["bpw"], "inst_cond_bpw": tI["bpw"],
           "inst_indep_bpw": tInd["bpw"], "pair_bpw": tB["bpw"] + tI["bpw"],
           "p_same": None, "base_scale_bpw": tB["scale_bpw"], "codebook_bpw": 0.0, **evB}
    put(name, rec, eval_inst())


def run_plat(name, kind, c):
    """Lattice P-frame: base on the E8/D4 lattice of step s (row scale = base RMS), instruct on the SAME
    lattice, instruct stream = lattice-point difference (itself a lattice point -> same symbol coding)."""
    d = 4 if kind == "d4" else 8
    s = c / 2 ** 0.25 if kind == "d4" else c
    aB, aI, aInd = Accounting(), Accounting(), Accounting()
    yB = {}

    def add_streams(acc, n, streams, W, dq, scale_bits):
        acc.add(wtype(n), streams[0].cpu(), W.numel(), scale_bits, W, dq)
        side = scale_bits
        for j, st in enumerate(streams[1:]):
            side += acc.add_stream(f"{wtype(n)}#s{j}", st.cpu())
        return side

    def fB(n, W):
        V = (snapB[n].float() / rWB[n]).reshape(-1, d).to(DEV)
        y = lattice_nearest(V / s, kind)
        yB[n] = y
        dq = (y * s).reshape(W.shape).cpu() * rWB[n]
        st = lattice_symbols(y, kind)
        side = add_streams(aB, n, st, W, dq, 16 * W.shape[0])
        return st[0].cpu(), dq, side

    evB = eval_base(fB)
    same = [0, 0]

    def fI(n, W):
        V = (W / rWB[n]).reshape(-1, d).to(DEV)
        y = lattice_nearest(V / s, kind)
        dq = (y * s).reshape(W.shape).cpu() * rWB[n]
        st = lattice_symbols(y - yB[n], kind)
        same[0] += int((y == yB[n]).all(1).sum()); same[1] += y.shape[0]
        side = add_streams(aI, n, st, W, dq, 0)
        add_streams(aInd, n, lattice_symbols(y, kind), W, dq, 0)
        return st[0].cpu(), dq, side
    quantize_model(model, snapI, fI)
    tB, tI, tInd = aB.total(), aI.total(), aInd.total()
    rec = {"kind": "plat", "lat": kind, "d": d, "c": c, "base_bpw": tB["bpw"], "inst_cond_bpw": tI["bpw"],
           "inst_indep_bpw": tInd["bpw"], "pair_bpw": tB["bpw"] + tI["bpw"], "codebook_bpw": 0.0,
           "base_scale_bpw": tB["scale_bpw"], "p_same": same[0] / same[1], **evB}
    put(name, rec, eval_inst())


@torch.no_grad()
def neighbour_table(C, T):
    """[K, T] indices of the T nearest centroids of every centroid (itself first), ties by index."""
    K = C.shape[0]
    T = min(T, K)
    out = torch.empty(K, T, dtype=torch.long, device=C.device)
    cc = (C * C).sum(1)
    for i in range(0, K, 1024):
        dist = cc[None, :] - 2 * C[i:i + 1024] @ C.T + cc[i:i + 1024, None]
        dist[torch.arange(dist.shape[0]), torch.arange(i, i + dist.shape[0])] = -1.0
        out[i:i + 1024] = dist.topk(T, dim=1, largest=False).indices
    return out


def run_pvq(name, d, K, lam):
    """Index P-frame on a codebook shared by base and instruct (see docstring).  Instruct symbol per
    vector: neighbour rank r < T of its centroid around the base centroid; T = 'raw index follows'
    (miss, or base vector escaped); T+1 = instruct vector escaped -> lattice integers, coded as
    the difference to the base integers when the base vector escaped too, else raw."""
    rfB = lambda n: rWB[n]
    cbs, fit = fit_codebooks({n: snapB[n] for n in snapB}, d, K, "type", method="ecvq", lam=lam,
                             init_cache=_init, rfn=rfB, cache_tag="base")
    cbs = nn_entries(cbs)        # closed-loop analog of the scalar grid's rounding
    T = args.T
    nbr = {sk: neighbour_table(cb["C"], T) for sk, cb in cbs.items()}
    aB, aI, aInd = Accounting(), Accounting(), Accounting()
    encB, rS = {}, {}
    charged = set()

    def fB(n, W):
        sk = wtype(n)
        cb = cbs[sk]
        V = (snapB[n].float() / rWB[n]).reshape(-1, d).to(DEV)
        idx, rec, esc = encode(V, cb)
        qfull = torch.round(V / cb["step"])
        encB[n] = (idx, qfull)
        dq = rec.reshape(W.shape).cpu() * rWB[n]
        rS[n] = rWB[n]
        if args.gain:              # shared row scale r' = fp16(r * g_base): same 16 b/row, reused by instruct
            rS[n] = (rWB[n] * gain_fix(snapB[n].float(), dq)).half().float()
            dq = rec.reshape(W.shape).cpu() * rS[n]
        side = 16 * W.shape[0]
        sb = side
        if sk not in charged:
            charged.add(sk); cbb = 16 * cb["C"].shape[0] * d; aB.cb_bits += cbb; side += cbb
        sym = idx.to(torch.int32).cpu()
        aB.add(sk, sym, W.numel(), sb)
        if esc is not None and esc.numel():
            side += aB.add_stream(sk + "#esc", esc.to(torch.int32).cpu())
        return sym, dq, side

    evB = eval_base(fB)
    st = {"same": 0, "miss": 0, "iesc": 0, "n": 0}

    def fI(n, W):
        sk = wtype(n)
        cb = cbs[sk]
        Kc = cb["C"].shape[0]
        V = (W / rWB[n]).reshape(-1, d).to(DEV)
        idx, rec, esc = encode(V, cb)
        dq = rec.reshape(W.shape).cpu() * rS[n]
        bidx, bq = encB[n]
        bok, iok = bidx < Kc, idx < Kc
        nb = nbr[sk][bidx.clamp(max=Kc - 1)]
        hit = (nb == idx[:, None]) & bok[:, None] & iok[:, None]
        found = hit.any(1)
        sym = torch.where(found, hit.float().argmax(1), torch.full_like(idx, T))
        sym = torch.where(~iok, torch.full_like(idx, T + 1), sym)
        raw = idx[(sym == T)]
        iq = torch.round(V / cb["step"])
        both = (~iok) & (~bok)
        escd = torch.cat([(iq[both] - bq[both]), iq[(~iok) & bok]]) if (~iok).any() else None
        st["same"] += int((idx == bidx).sum()); st["miss"] += int((sym == T).sum())
        st["iesc"] += int((~iok).sum()); st["n"] += idx.numel()
        s32 = sym.to(torch.int32).cpu()
        aI.add(sk, s32, W.numel(), 0, W, dq)
        side = 0
        if raw.numel():
            side += aI.add_stream(sk + "#raw", raw.to(torch.int32).cpu())
        if escd is not None and escd.numel():
            side += aI.add_stream(sk + "#esc", escd.to(torch.int32).cpu())
        aInd.add(sk, idx.to(torch.int32).cpu(), W.numel(), 0)
        if esc is not None and esc.numel():
            aInd.add_stream(sk + "#esc", esc.to(torch.int32).cpu())
        return s32, dq, side
    quantize_model(model, snapI, fI)
    tB, tI, tInd = aB.total(), aI.total(), aInd.total()
    rec = {"kind": "pvq", "d": d, "K": K, "lam": lam, "T": T, "gain": bool(args.gain), "base_bpw": tB["bpw"],
           "inst_cond_bpw": tI["bpw"], "inst_indep_bpw": tInd["bpw"], "pair_bpw": tB["bpw"] + tI["bpw"],
           "codebook_bpw": tB["codebook_bpw"], "base_scale_bpw": tB["scale_bpw"],
           "p_same": st["same"] / st["n"], "p_miss": st["miss"] / st["n"], "p_inst_escape": st["iesc"] / st["n"],
           "fit_s": fit["fit_s"], **evB}
    put(name, rec, eval_inst())


# ------------------------------------------------------------------ plan
G = set(args.groups.split(","))
KMAX = {1: 256, 2: 4096, 4: 16384}
lamW = lambda c: 2 * math.log(2) * c * c / 12          # ECVQ lambda matched to an ECSQ step c
PLAN = []
if "a" in G:
    for c in (0.06, 0.12, 0.25, 0.5):
        PLAN.append((f"dscal_c{c}_nW", lambda c=c: run_dscal(f"dscal_c{c}_nW", c, "W")))
    for c in (0.8, 1.6, 3.2):
        PLAN.append((f"dscal_c{c}_nD", lambda c=c: run_dscal(f"dscal_c{c}_nD", c, "D")))
    for c in (0.8, 1.6, 3.2):                          # lattice VQ of Delta (nD, the better scalar normalisation)
        for kind in ("e8", "d4"):
            PLAN.append((f"dlat_{kind}_c{c}_nD", lambda c=c, kind=kind: run_dlat(f"dlat_{kind}_c{c}_nD", kind, c, "D")))
    # k-means/ECVQ Delta, nearest-centroid + escape, nD normalisation (better for scalar and lattice)
    for c in (1.6, 0.8, 3.2):
        for d in ((1, 2, 4) if c == 1.6 else (2, 4)):
            nm = f"decvqnn_d{d}_l{lamW(c):.3g}_nD"
            PLAN.append((nm, lambda nm=nm, d=d, c=c: run_decvq(nm, d, KMAX[d], lamW(c), "D")))
if "b" in G:                                           # seed-noise check of the k-means Delta result (runs first)
    nm = f"decvqnn_d2_l{lamW(1.6):.3g}_nD_s1"
    PLAN.append((nm, lambda nm=nm: run_decvq(nm, 2, KMAX[2], lamW(1.6), "D", seed=1)))
if "lat" in G:                                         # second-family confirmation of the VQ-Delta results (trimmed)
    for c in (1.6, 0.8):
        PLAN.append((f"dscal_c{c}_nD", lambda c=c: run_dscal(f"dscal_c{c}_nD", c, "D")))
        PLAN.append((f"dlat_d4_c{c}_nD", lambda c=c: run_dlat(f"dlat_d4_c{c}_nD", "d4", c, "D")))
    for c in (1.6, 0.8):
        nm = f"decvqnn_d2_l{lamW(c):.3g}_nD"
        PLAN.append((nm, lambda nm=nm, c=c: run_decvq(nm, 2, KMAX[2], lamW(c), "D")))
if "agptq" in G:     # Hessian-aware Delta: phase-1/2 setting (nW), nD, and GPTQ on the D4 lattice
    for c in (0.12, 0.06, 0.25):
        PLAN.append((f"dgptq_c{c}", lambda c=c: run_dgptq(f"dgptq_c{c}", c)))
    for c in (1.6, 0.8):
        PLAN.append((f"dgptq_c{c}_nD", lambda c=c: run_dgptq(f"dgptq_c{c}_nD", c, "D")))
        PLAN.append((f"dgptq_d4_c{c}_nD", lambda c=c: run_dgptq_lat(f"dgptq_d4_c{c}_nD", "d4", c)))
if "b" in G:
    for c in (0.35, 0.25, 0.5):
        PLAN.append((f"pscal_c{c}", lambda c=c: run_pscal(f"pscal_c{c}", c)))
        PLAN.append((f"plat_d4_c{c}", lambda c=c: run_plat(f"plat_d4_c{c}", "d4", c)))
        nm = f"pvq_d4_l{lamW(c):.3g}"
        PLAN.append((nm, lambda nm=nm, c=c: run_pvq(nm, 4, KMAX[4], lamW(c))))
    PLAN.append(("plat_e8_c0.35", lambda: run_plat("plat_e8_c0.35", "e8", 0.35)))
    nm = f"pvq_d2_l{lamW(0.35):.3g}"
    PLAN.append((nm, lambda nm=nm: run_pvq(nm, 2, KMAX[2], lamW(0.35))))
only = set(args.only.split(",")) if args.only else None
if not store.has("fp16"):
    restore_weights(model, snapI)
    put("fp16", {"bpw": 16.0}, eval_inst())
if not store.has("zero_bits_base"):
    quantize_model(model, snapI, lambda n, W: (None, snapB[n].float(), 0))
    put("zero_bits_base", {"bpw": 0.0}, eval_inst())
for name, f in PLAN:
    target = name + ("_gain" if args.gain and name.startswith(("decvq", "pvq")) else "")
    if (only and name not in only) or store.has(target):
        continue
    check_disk()
    f()
restore_weights(model, snapI)
log("done; disk free GB", round(disk_free_gb(), 2))
