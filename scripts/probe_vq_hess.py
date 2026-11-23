"""Probe V2: Hessian-aware VQ (GPTVQ / AQLM-style sensitivity weighting) vs Hessian-aware scalar.

H = E[x x^T] of each block linear's input (common.collect_hessians, 128 x 2048 WikiText-2 train
windows, non-sequential).  All rate accounting as in V1 (scripts/vq.py, everything charged).

  gptq_ecsq_row_c*     scalar GPTQ on the uniform grid c*rms_row (spine's best intra coder; d=1)
  hw_ecvq_d*_l*        ECVQ in the diag(H)-weighted metric (weighted centroids; weighted NEAREST
                       assignment), NO error feedback  -> "does the weighted metric alone help?"
                       (compare with V1 ecvqnn_d*_l0.0141, the unweighted nearest-centroid twin)
  gptvq_d*_l*          ECVQ codebook (plain, per type) + GPTQ error feedback with the exact joint
                       d-column OBS update and Mahalanobis NEAREST-centroid assignment ||e inv(U_SS)||^2
                       (lam only in the escape decision; V1: the RDO assignment hurts downstream)
                       -> "does VQ still beat scalar once BOTH use the Hessian?"
PRE-REGISTERED: (i) weighting helps iff hw_ecvq beats plain ecvq (V1, same d) by >= 0.10 bpw at
equal KL;  (ii) V2 PASS iff the best d>=2 gptvq line saves >= 0.10 bpw vs the best d=1
Hessian line (gptq_ecsq_row, gptvq_d1) at equal KL at >= 2 levels; FAIL if < 0.05 everywhere.

usage: python scripts/probe_vq_hess.py [--groups gptq,hw,gptvq] [--only ...]
"""
import argparse
from common import *
from vq import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--tag", default="smol135")
ap.add_argument("--tok_tag", default="smollm2")
ap.add_argument("--groups", default="gptq,gptvq,hw")
ap.add_argument("--only", default="")
ap.add_argument("--no_diag", action="store_true", help="skip the output-error diagnostic")
ap.add_argument("--h_half", action="store_true", help="keep Hessians in fp16 on CPU (0.5 B models: 2.5 -> 1.25 GB)")
ap.add_argument("--c4", action="store_true", help="also evaluate out-of-domain on C4 val (data/c4_val_qwen25.pt, Qwen only)")
args = ap.parse_args()
seed_all(0)
log(f"disk free {check_disk():.2f} GB")
tok = load_tok(args.model)
test_ids = wt2_ids(tok, "test", args.tok_tag)
train_ids = wt2_ids(tok, "train", args.tok_tag)
model = load_model(args.model)
snap = snapshot_weights(model)
ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
H = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
if args.h_half:
    _u = {}
    # store U = chol(H^-1) (fp64 from the fp32 H, kept in bf16) + diag(H) instead of H: fp16 H/max|H| and
    # bf16 H both made Qwen's H indefinite (massive channels) -> Cholesky failure / heavy damping (06:16, 06:31)
    _ids = {}
    for n in list(H):
        k = id(H[n])
        if k not in _ids:
            _ids[k] = HU(H[n])
        H[n] = _ids[k]
    gc.collect()
    gc.collect()
restore_weights(model, snap)
ref_c4, c4 = None, None
if args.c4:     # calibration = WikiText-2 train; out-of-domain check as in scripts/crossdomain_c4.py
    c4 = torch.load(os.path.join(DATA, "c4_val_qwen25.pt"))
    ref_c4 = attach_head(build_ref(model, c4, 16), model)
store = Store(f"vq_v2_{args.tag}")
store.put("_meta", {"model": args.model, "hessian": "128 x 2048 WikiText-2 train windows, seed 0, fp model",
                    "seed": 0, "percdamp": 0.01})
KMAX = {1: 256, 2: 4096, 4: 16384}
_init = {}

# ---- diagnostic for the V1 lattice result: Euclidean weight error vs OUTPUT error tr(E H E^T) -------
# (hypothesis: E8's errors are correlated across the 8 coordinates of a block and LLM input covariances
#  H have strong off-diagonal / shared components, so its lower ||E||^2 does not become lower output error)
if not args.no_diag and not store.has("_diag_output_error"):
    diag = {}
    for c in (0.35, 0.5, 0.7):
        acc = {"scalar": [0.0, 0.0], "d4": [0.0, 0.0], "e8": [0.0, 0.0]}
        for n in snap:
            W = snap[n].float()
            r = row_scale(W)
            V = W / r
            Hn = H[n].float()
            recs = {"scalar": torch.round(V / c) * c}
            for kind, d in (("d4", 4), ("e8", 8)):
                s = c / 2 ** 0.25 if kind == "d4" else c
                recs[kind] = (lattice_nearest(V.reshape(-1, d) / s, kind) * s).reshape(V.shape)
            for k, R in recs.items():
                E = (V - R) * r
                acc[k][0] += float(E.pow(2).sum())
                acc[k][1] += float(((E @ Hn) * E).sum())
        diag[f"c{c}"] = {k: {"weight_se": v[0], "output_err": v[1]} for k, v in acc.items()}
        rel = {k: (v[0] / acc["scalar"][0], v[1] / acc["scalar"][1]) for k, v in acc.items()}
        log(f"diag c={c}: (weight-SE ratio, output-error ratio) vs scalar: " +
            " ".join(f"{k}=({a:.3f},{b:.3f})" for k, (a, b) in rel.items()))
    store.put("_diag_output_error", diag)


def finish(name, cfg, acct, fit_s=None):
    ev = eval_lm(model, test_ids, ref=ref)
    if ref_c4 is not None:
        e4 = eval_lm(model, c4, ref=ref_c4)
        ev.update({"c4_kl": e4["kl"], "c4_ppl": e4["ppl"], "c4_top1": e4["top1"]})
    a = acct.total()
    store.put(name, {"cfg": cfg, **a, "fit_s": fit_s, **ev})
    log(f"{name:24s} bpw={a['bpw']:.3f} (idx {a['idx_bpw']:.3f} cb {a['codebook_bpw']:.4f} sc {a['scale_bpw']:.3f}) "
        f"ppl={ev['ppl']:.3f} kl={ev['kl']:.4f} top1={ev['top1']:.3f}" + (f" c4_kl={ev['c4_kl']:.4f}" if 'c4_kl' in ev else "") +
        f" | disk {disk_free_gb():.1f}GB")


def run_gptq(name, c):
    acct = Accounting()

    def fn(n, W):
        s = (c * W.pow(2).mean(1, keepdim=True).sqrt()).half().float()
        q, D, _ = gptq(W, H[n], "grid", step=s)
        acct.add(wtype(n), q, W.numel(), 16 * W.shape[0], W, D)
        return q, D, 16 * W.shape[0]
    quantize_model(model, snap, fn)
    finish(name, {"kind": "gptq_ecsq", "c": c}, acct)


def run_gptvq(name, d, lam, seed=0):
    t0 = time.time()
    cbs, fit = fit_codebooks(snap, d, KMAX[d], "type", method="ecvq", lam=lam, init_cache=_init, seed=seed)
    # nearest-centroid encoding (penalty only in the escape decision): V1 showed the Lagrangian
    # (RDO) assignment costs downstream KL vs rounding, and GPTQ-ECSQ itself rounds
    cbs = {k: {**v, "pen": None, "dec": v["pen"]} for k, v in cbs.items()}
    acct = Accounting()
    charged = set()

    def fn(n, W):
        sk = wtype(n)
        cb = cbs[sk]
        idx, D, esc = gptvq(W, H[n], cb, d)
        side = 16 * W.shape[0]
        sb = side
        if sk not in charged:
            charged.add(sk); cbb = 16 * cb["C"].shape[0] * d; acct.cb_bits += cbb; side += cbb
        acct.add(sk, idx, W.numel(), sb, W, D)
        if esc is not None and esc.numel():
            side += acct.add_stream(sk + "#esc", esc)
        return idx, D, side
    quantize_model(model, snap, fn)
    finish(name, {"kind": "gptvq", "d": d, "lam": lam, "K": KMAX[d], "seed": seed}, acct, round(time.time() - t0, 1))


def run_hw(name, d, lam):
    """diag(H)-weighted ECVQ, codebook per type; train on a subsample with its weights."""
    t0 = time.time()
    groups = {}
    for n in snap:
        groups.setdefault(wtype(n), []).append(n)
    K = KMAX[d]
    g = torch.Generator().manual_seed(0)
    cbs = {}
    for sk, names in groups.items():
        tot = sum(snap[n].numel() // d for n in names)
        frac = min(1.0, max(1_000_000, 64 * K) / tot)
        Xs, Ws = [], []
        for n in names:
            V, _ = to_vecs(snap[n], d)
            w = hdiag_weights(H[n], d, snap[n].shape[0])
            if frac < 1:
                sel = torch.randperm(V.shape[0], generator=g)[:max(1, int(round(V.shape[0] * frac)))]
                V, w = V[sel], w[sel]
            Xs.append(V); Ws.append(w)
        X, Wt = torch.cat(Xs).to(DEV), torch.cat(Ws).to(DEV)
        ck = (d, K, sk)
        if ck not in _init:
            _init[ck], _ = kmeans(X, K, iters=25)
        cb = ecvq(X, _init[ck], lam, w=Wt)
        cb["C"] = cb["C"].half().float()
        cbs[sk] = {**cb, "pen": None, "dec": cb["pen"]}      # weighted NEAREST-centroid encoding
        del X, Wt
    acct = Accounting()
    charged = set()

    def fn(n, W):
        sk = wtype(n)
        cb = cbs[sk]
        V, r = to_vecs(W, d)
        idx, rec, esc = encode(V.to(DEV), cb, w=hdiag_weights(H[n], d, W.shape[0]).to(DEV))
        D = (rec.reshape(W.shape) * r.to(DEV)).cpu()
        side = 16 * W.shape[0]
        sb = side
        if sk not in charged:
            charged.add(sk); cbb = 16 * cb["C"].shape[0] * d; acct.cb_bits += cbb; side += cbb
        sym = idx.to(torch.int32).cpu()
        acct.add(sk, sym, W.numel(), sb, W, D)
        if esc is not None and esc.numel():
            side += acct.add_stream(sk + "#esc", esc.to(torch.int32).cpu())
        return sym, D, side
    quantize_model(model, snap, fn)
    finish(name, {"kind": "hw_ecvq", "d": d, "lam": lam, "K": K}, acct, round(time.time() - t0, 1))


def run_gptq_lat(name, kind, c):
    s = c / 2 ** 0.25 if kind == "d4" else c       # same point density as the scalar grid of step c
    acct = Accounting()

    def fn(n, W):
        streams, D = gptq_lattice(W, H[n], kind, s)
        acct.add(wtype(n), streams[0], W.numel(), 16 * W.shape[0], W, D)
        side = 16 * W.shape[0]
        for j, st in enumerate(streams[1:]):
            side += acct.add_stream(f"{wtype(n)}#s{j}", st)
        return streams[0], D, side
    quantize_model(model, snap, fn)
    finish(name, {"kind": "gptq_lattice", "lat": kind, "c": c, "d": 4 if kind == "d4" else 8}, acct)


G = set(args.groups.split(","))
if "gptq" in G:
    G.add("gptlat")
PLAN = []
if "gptq" in G:
    for c in (0.25, 0.35, 0.5, 0.7):
        PLAN.append((f"gptq_ecsq_row_c{c}", lambda c=c: run_gptq(f"gptq_ecsq_row_c{c}", c)))
if "gptq1" in G:                  # Qwen: one in-harness scalar-GPTQ anchor (spine has the rest)
    PLAN.append(("gptq_ecsq_row_c0.35", lambda: run_gptq("gptq_ecsq_row_c0.35", 0.35)))
if "gptlat" in G:                 # V1 survivor: does the D4 edge survive GPTQ error feedback?
    for c in (0.35, 0.5, 0.7, 0.25):
        PLAN.append((f"gptq_d4_c{c}", lambda c=c: run_gptq_lat(f"gptq_d4_c{c}", "d4", c)))
    for c in (0.35, 0.5):
        PLAN.append((f"gptq_e8_c{c}", lambda c=c: run_gptq_lat(f"gptq_e8_c{c}", "e8", c)))
if "gptvq" in G:                  # trimmed: k-means/ECVQ already lost in V1; d=1 = GPTQ-ECVQ control
    for lam in (0.0141, 0.0289):
        for d in (1, 4):
            PLAN.append((f"gptvq_d{d}_l{lam}", lambda d=d, lam=lam: run_gptvq(f"gptvq_d{d}_l{lam}", d, lam)))
def run_gsearch(name, kind, c):
    s = c / 2 ** 0.25 if kind == "d4" else c
    acct = Accounting()

    def fn(n, W):
        streams, D = gptq_joint_search(W, H[n], kind, s)
        acct.add(wtype(n), streams[0], W.numel(), 16 * W.shape[0], W, D)
        side = 16 * W.shape[0]
        for j, st in enumerate(streams[1:]):
            side += acct.add_stream(f"{wtype(n)}#s{j}", st)
        return streams[0], D, side
    quantize_model(model, snap, fn)
    finish(name, {"kind": "gptq_search", "lat": kind, "c": c, "d": 4}, acct)


def run_hcol(name, c, gamma):
    """Sensitivity-weighted SCALAR control (A3-style, diag(H) instead of Fisher): step s_ij = fp16(c*rms_i) *
    fp16(col_j), col_j = (h_j / geomean(h))^(-gamma/2), h = diag(H); side info 16 b/row + 16 b/column.
    Same per-coordinate sensitivity information the weighted VQ uses, but no vector decisions."""
    acct = Accounting()

    def fn(n, W):
        hd = hdiag(H[n]).clamp(min=1e-12)
        hd = hd / torch.exp(torch.log(hd).mean())
        col = hd.pow(-gamma / 2).half().float()
        rs = (c * W.pow(2).mean(1, keepdim=True).sqrt()).half().float()
        st = rs * col[None, :]
        q = torch.round(W / st)
        dq = q * st
        side = 16 * (W.shape[0] + W.shape[1])
        acct.add(wtype(n), q.to(torch.int32), W.numel(), side, W, dq)
        return q.to(torch.int32), dq, side
    quantize_model(model, snap, fn)
    finish(name, {"kind": "hcol", "c": c, "gamma": gamma, "d": 1}, acct)


def run_hcol_lat(name, c, gamma, kind="d4"):
    """Does VQ add anything ON TOP of the best scalar coder?  Same diag(H) column scaling as run_hcol, then a
    D4 lattice (same point density as the scalar grid of step c) in the column-weighted, row-normalised
    coordinates.  Side info identical to hcol (16 b/row + 16 b/column)."""
    acct = Accounting()
    d = 4 if kind == "d4" else 8
    s = c / 2 ** 0.25 if kind == "d4" else c

    def fn(n, W):
        hd = hdiag(H[n]).clamp(min=1e-12)
        hd = hd / torch.exp(torch.log(hd).mean())
        col = hd.pow(-gamma / 2).half().float()
        rs = W.pow(2).mean(1, keepdim=True).sqrt().half().float()
        sc = rs * col[None, :]
        V = (W / sc).reshape(-1, d).to(DEV)
        streams, rec = lattice_encode(V, kind, s)
        dq = rec.reshape(W.shape).cpu() * sc
        side = 16 * (W.shape[0] + W.shape[1])
        acct.add(wtype(n), streams[0].cpu(), W.numel(), side, W, dq)
        for j, st in enumerate(streams[1:]):
            side += acct.add_stream(f"{wtype(n)}#s{j}", st.cpu())
        return streams[0].cpu(), dq, side
    quantize_model(model, snap, fn)
    finish(name, {"kind": "hcol_lattice", "lat": kind, "c": c, "gamma": gamma, "d": d}, acct)


if "hcollat" in G:
    for c in (0.35, 0.5, 0.25):
        PLAN.append((f"hcol_d4_g1.0_c{c}", lambda c=c: run_hcol_lat(f"hcol_d4_g1.0_c{c}", c, 1.0)))
if "hcol" in G:                   # scalar + diag(H) column steps: the control for weighted VQ (hw_ecvq / gptvq)
    for gamma in (0.5, 1.0):
        for c in (0.35, 0.5, 0.25):
            PLAN.append((f"hcol_g{gamma}_c{c}", lambda c=c, gamma=gamma: run_hcol(f"hcol_g{gamma}_c{c}", c, gamma)))
if "gseed" in G:
    G.add("gsearch")
if "gsearch" in G:                # mechanism ablation: joint metric-aware rounding on a plain grid, no codebook
    for c in (0.35, 0.5):
        PLAN.append((f"gptq_z4search_c{c}", lambda c=c: run_gsearch(f"gptq_z4search_c{c}", "z4", c)))
        PLAN.append((f"gptq_d4search_c{c}", lambda c=c: run_gsearch(f"gptq_d4search_c{c}", "d4", c)))
if "gseed" in G:                  # seed check of the GPTVQ d=4 result
    PLAN.append(("gptvq_d4_l0.0141_s1", lambda: run_gptvq("gptvq_d4_l0.0141_s1", 4, 0.0141, seed=1)))
if "hw" in G:
    for d in (1, 4):
        PLAN.append((f"hw_ecvq_d{d}_l0.0141", lambda d=d: run_hw(f"hw_ecvq_d{d}_l0.0141", d, 0.0141)))
only = set(args.only.split(",")) if args.only else None
for name, f in PLAN:
    if (only and name not in only) or store.has(name):
        continue
    check_disk()
    f()
restore_weights(model, snap)
log("done; disk free GB", round(disk_free_gb(), 2))
