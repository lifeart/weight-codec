"""Probe V1: plain vector quantization (codebook clustering of weight sub-vectors) vs scalar
quantization at EQUAL TOTAL bpw (codebook, scales and entropy tables all charged).

Hypothesis (user, 2026-11-21): "sort the weights in a multi-dimensional space (as vectors) and
compress the clusters" = k-means VQ over d-dim sub-vectors with a shared codebook.  Scalar
quantization is VQ at d=1.

Methods (all on row-RMS-normalised weights, 16 bits/row fp16 scale, scripts/vq.py):
  ecsq_row_c*        uniform scalar step c*rms_row, entropy coded (the spine's ECSQ line; d=1)
  rtn_int{3,4}_g64   fixed-rate asymmetric RTN (SmolLM2 dims are not /128), reference only
  km_d{d}_K{K}_{scope}       plain Lloyd k-means codebook, nearest-centroid, entropy-coded indices
  ecvq_d{d}_l{lam}_{scope}   entropy-constrained k-means (Lagrangian assignment) - the VQ that is
                             actually comparable with entropy-coded scalar quantization
  scope: type (one codebook per weight type shared across all layers; default), tensor
         (one per matrix: the amortization control), global (one for everything)

PRE-REGISTERED VERDICT (written before any result):
  Let E1(bpw) = lower envelope of log KL over every d=1 method (uniform ECSQ, d=1 k-means,
  d=1 ECVQ) and Ed(bpw) = same over d>=2 VQ.  Rate saving S(KL) = bpw_1(KL) - bpw_d(KL) at equal
  WikiText-2 KL (piecewise-linear in log KL), evaluated at KL levels inside both curves' range.
  V1 PASS   iff S >= 0.10 bpw at >= 2 KL levels in the 2.5-4.5 bpw window;
  V1 FAIL   iff S <  0.05 bpw at every comparable level;  otherwise INCONCLUSIVE.
  Prior (high-rate theory, entropy-coded): VQ's only edge over ECSQ on i.i.d.-like data is the
  space-filling gain: A2 0.028, D4 0.061, E8 0.108, d->inf 0.254 bits/weight; k-means at finite
  K reaches less.  Expect S ~ 0.03-0.1 -> likely FAIL/INCONCLUSIVE.  One seed (0) unless noted.

usage: python scripts/probe_vq.py [--model HuggingFaceTB/SmolLM2-135M --tag smol135 --tok_tag smollm2]
       [--only name1,name2] [--groups ecsq,ecvq,km,amort,seed,d8]
"""
import argparse
from common import *
from vq import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--tag", default="smol135")
ap.add_argument("--tok_tag", default="smollm2")
ap.add_argument("--only", default="")
ap.add_argument("--groups", default="ecsq,ecvq,km,d8,amort,seed")
ap.add_argument("--max_windows", type=int, default=0)
args = ap.parse_args()
seed_all(0)
log(f"disk free {check_disk():.2f} GB")
tok = load_tok(args.model)
test_ids = wt2_ids(tok, "test", args.tok_tag)
model = load_model(args.model)
snap = snapshot_weights(model)
ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
store = Store(f"vq_v1_{args.tag}")
N = sum(w.numel() for w in snap.values())
store.put("_meta", {"model": args.model, "n_quantized_set": N, "device": DEV, "seed": 0,
                    "eval": "WikiText-2 test full (non-overlapping 2048) PPL; KL/top1 vs fp16 on first 16 windows",
                    "vector": "d consecutive input-dim weights of a row, row-RMS normalised (fp16, 16 b/row)",
                    "rate": "bpw = min(per-tensor, pooled-per-scope) entropy-coded indices incl. 32 b/symbol "
                            "tables + 16*d*K_used codebook bits + 16 b/row scales, over all block linears"})

LAMS = (0.0072, 0.0141, 0.0289, 0.0566)       # ~ 2 ln2 * (c^2/12) for c = 0.25, 0.35, 0.5, 0.7
KMAX = {1: 256, 2: 4096, 4: 16384, 8: 65536}
G = set(args.groups.split(","))
CONFIGS = []
if "ecsq" in G:
    for c in (0.2, 0.25, 0.35, 0.5, 0.7, 1.0):
        CONFIGS.append((f"ecsq_row_c{c}", {"kind": "ecsq", "c": c}))
    CONFIGS += [("rtn_int4_g64", {"kind": "rtn", "bits": 4, "group": 64}),
                ("rtn_int3_g64", {"kind": "rtn", "bits": 3, "group": 64})]
if "ecvq" in G:
    for lam in LAMS:
        for d in (1, 2, 4):
            CONFIGS.append((f"ecvq_d{d}_l{lam}_type", {"kind": "ecvq", "d": d, "K": KMAX[d], "lam": lam, "scope": "type"}))
# plan trimmed 2026-11-21 23:58 (machine under external memory pressure, d=4 K=16384 configs take
# ~20 min): km = one K per d; d=8 = one k-means + one ECVQ point; amortization = 3 controls;
# seed check on d=1 and d=2 (cheap) instead of d=4.
if "nn" in G:
    # same ECVQ codebook, but NEAREST-centroid encoding (penalty only in the escape decision):
    # separates "vector dimension" from "Lagrangian (RDO) assignment" - at d=1 RDO already costs
    # ~13 % KL vs plain rounding at equal weight-MSE
    for lam in (0.0072, 0.0141, 0.0289):
        for d in (1, 2):
            CONFIGS.append((f"ecvqnn_d{d}_l{lam}_type", {"kind": "ecvq", "nn": True, "d": d, "K": KMAX[d], "lam": lam, "scope": "type"}))
    CONFIGS.append(("ecvqnn_d4_l0.0141_type", {"kind": "ecvq", "nn": True, "d": 4, "K": KMAX[4], "lam": 0.0141, "scope": "type"}))
    G.add("seed")          # seed-noise check moved forward: KL scatter decides how to read 10 % gaps
if "lattice" in G:
    # structured, codebook-free VQ (D4 / E8 nearest point, per-coordinate entropy coding): isolates the
    # space-filling gain (weight domain: D4 +0.055, E8 +0.10 bit at equal MSE) from codebook-shape effects.
    # D4 step c/2^(1/4) and E8 step c have the same point density as the ECSQ grid of step c.
    for c in (0.2, 0.25, 0.35, 0.5, 0.7):
        CONFIGS.append((f"lat_e8_c{c}", {"kind": "lattice", "lat": "e8", "d": 8, "c": c, "s": c}))
        CONFIGS.append((f"lat_d4_c{c}", {"kind": "lattice", "lat": "d4", "d": 4, "c": c, "s": c / 2 ** 0.25}))
if "confirm" in G:
    # second-model confirmation of the only V1 survivor (D4 lattice) + the E8 reversal; scalar ECSQ rerun
    # with this script's accounting so all points share it
    for c in (0.35, 0.5):          # trimmed 05:52 (time): 2 rates, E8 at one
        CONFIGS.append((f"ecsq_row_c{c}", {"kind": "ecsq", "c": c}))
        CONFIGS.append((f"lat_d4_c{c}", {"kind": "lattice", "lat": "d4", "d": 4, "c": c, "s": c / 2 ** 0.25}))
    CONFIGS.append(("lat_e8_c0.5", {"kind": "lattice", "lat": "e8", "d": 8, "c": 0.5, "s": 0.5}))
if "gain" in G:
    # attenuation test: conditional-mean centroids shrink every row by ~(1-NMSE); a per-row gain
    # folded into the fp16 row scale (0 extra bits) undoes it.  Same correction on scalar ECSQ.
    for c in (0.25, 0.35, 0.5):
        CONFIGS.append((f"ecsq_row_c{c}_gain", {"kind": "ecsq", "c": c, "gain": True}))
    for lam in (0.0141, 0.0289):
        for d in (1, 2, 4):
            CONFIGS.append((f"ecvqnn_d{d}_l{lam}_type_gain", {"kind": "ecvq", "nn": True, "gain": True, "d": d,
                                                               "K": KMAX[d], "lam": lam, "scope": "type"}))
    for d in (1, 4):
        CONFIGS.append((f"ecvq_d{d}_l0.0289_type_gain", {"kind": "ecvq", "gain": True, "d": d, "K": KMAX[d],
                                                         "lam": 0.0289, "scope": "type"}))
if "seed" in G:
    for d in (1, 2):
        CONFIGS.append((f"ecvq_d{d}_l0.0141_type_s1", {"kind": "ecvq", "d": d, "K": KMAX[d], "lam": 0.0141,
                                                       "scope": "type", "seed": 1}))

if "km" in G:
    for d, K in ((1, 16), (2, 256), (4, 4096)):
        CONFIGS.append((f"km_d{d}_K{K}_type", {"kind": "km", "d": d, "K": K, "scope": "type"}))
if "d8" in G:
    # per-type K=65536 d=8 codebooks would cost 7*65536*8*16 b = 0.55 bpw -> one GLOBAL codebook
    # (0.079 bpw); reduced fit budget (k-means on 2M vectors, 12 Lloyd + 10 ECVQ iterations)
    D8 = {"n_train": 2_000_000, "km_iters": 12, "ecvq_iters": 10}
    CONFIGS.append(("km_d8_K4096_type", {"kind": "km", "d": 8, "K": 4096, "scope": "type"}))
    CONFIGS.append(("ecvq_d8_l0.0566_global", {"kind": "ecvq", "d": 8, "K": 65536, "lam": 0.0566, "scope": "global", **D8}))
if "amort" in G:
    CONFIGS.append(("km_d4_K4096_tensor", {"kind": "km", "d": 4, "K": 4096, "scope": "tensor"}))
    CONFIGS.append(("km_d4_K4096_global", {"kind": "km", "d": 4, "K": 4096, "scope": "global"}))
    CONFIGS.append(("ecvq_d4_l0.0141_global", {"kind": "ecvq", "d": 4, "K": 16384, "lam": 0.0141, "scope": "global"}))
_init = {}   # k-means init cache shared by the ECVQ lambdas: (d, K, scope_key, seed) -> C


def run(name, cfg):
    acct = Accounting()
    t0 = time.time()
    fit = {}
    if cfg["kind"] == "ecsq":
        def fn(n, W):
            st = (cfg["c"] * W.pow(2).mean(1, keepdim=True).sqrt()).half().float()
            s, dq = ecsq(W, st)
            if cfg.get("gain"):        # reconstruction step s' = fp16(s * g): same 16 b/row
                dq = s.float() * (st * gain_fix(W, dq)).half().float()
            acct.add(wtype(n), s, W.numel(), 16 * W.shape[0], W, dq)
            return s, dq, 16 * W.shape[0]
    elif cfg["kind"] == "lattice":
        def fn(n, W):
            V, r = to_vecs(W, cfg["d"])
            streams, rec = lattice_encode(V.to(DEV), cfg["lat"], cfg["s"])
            dq = (rec.reshape(W.shape) * r.to(DEV)).cpu()
            head = streams[0].cpu()
            acct.add(wtype(n), head, W.numel(), 16 * W.shape[0], W, dq)
            side = 16 * W.shape[0]
            for j, st in enumerate(streams[1:]):
                side += acct.add_stream(f"{wtype(n)}#s{j}", st.cpu())
            return head, dq, side
    elif cfg["kind"] == "rtn":
        def fn(n, W):
            s, dq, side = rtn_asym(W, cfg["bits"], cfg["group"])
            acct.add(wtype(n), s, W.numel(), side, W, dq)
            return s, dq, side
    else:
        cbs, fit = fit_codebooks(snap, cfg["d"], cfg["K"], cfg["scope"], method=cfg["kind"],
                                 lam=cfg.get("lam"), seed=cfg.get("seed", 0), init_cache=_init,
                                 n_train=cfg.get("n_train"), km_iters=cfg.get("km_iters"),
                                 ecvq_iters=cfg.get("ecvq_iters", 20))
        if cfg.get("nn"):
            cbs = {k: {**v, "pen": None, "dec": v["pen"]} for k, v in cbs.items()}
        fn = make_vq_fn(cbs, cfg["d"], cfg["scope"], acct, gain=cfg.get("gain", False))
    acc = quantize_model(model, snap, fn)
    tq = time.time() - t0
    ev = eval_lm(model, test_ids, max_windows=args.max_windows or None, ref=ref)
    a = acct.total()
    assert abs(a["bpw_harness_per_tensor"] - acc["entropy_bpw"]) < 1e-6, (a, acc["entropy_bpw"])
    rec = {"cfg": cfg, **a, "fit_s": fit.get("fit_s"), "quant_s": round(tq, 1),
           "K_used_total": sum(v["K_used"] for v in fit.get("scopes", {}).values()) or None, **ev}
    if cfg["kind"] == "rtn":
        b, g = cfg["bits"], cfg["group"]
        rec["fixed_bpw"] = b + (16 + b) / g
    store.put(name, rec)
    log(f"{name:26s} bpw={a['bpw']:.3f} (idx {a['idx_bpw']:.3f}[{a['table_choice'][:4]}] cb {a['codebook_bpw']:.4f} "
        f"sc {a['scale_bpw']:.3f}) nmse={a['weight_nmse_diag']:.4f} ppl={ev['ppl']:.3f} kl={ev['kl']:.4f} "
        f"top1={ev['top1']:.3f} | fit {fit.get('fit_s')}s q {tq:.0f}s ev {ev['eval_s']}s | disk {disk_free_gb():.1f}GB")


only = set(args.only.split(",")) if args.only else None
if not store.has("fp16"):
    restore_weights(model, snap)
    store.put("fp16", {"bpw": 16.0, **eval_lm(model, test_ids, max_windows=args.max_windows or None, ref=ref)})
    log("fp16", store.get("fp16"))
for name, cfg in CONFIGS:
    if (only and name not in only) or store.has(name):
        continue
    check_disk()
    run(name, cfg)
restore_weights(model, snap)
log("done; disk free GB", round(disk_free_gb(), 2))
