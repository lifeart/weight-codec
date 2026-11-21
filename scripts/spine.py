"""SPINE: baselines (fp16 / int8 / int4 / int3 / int2, RTN and GPTQ) + A2 (entropy of the
quantized indices) + ECSQ (entropy-coded uniform quantizer, the 'JPEG without transform'
reference line) on one model.  Results -> results/spine_<tag>.json (one entry per config,
re-runs skip finished configs).

usage: python scripts/spine.py [--model Qwen/Qwen2.5-0.5B] [--tag qwen05b] [--only a,b]
"""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--only", default="")
ap.add_argument("--max_windows", type=int, default=0)
ap.add_argument("--kl_windows", type=int, default=16)
ap.add_argument("--ncalib", type=int, default=128)
args = ap.parse_args()

seed_all(0)
check_disk()
tok = load_tok(args.model)
toktag = "qwen25" if "Qwen" in args.model else args.tag
test_ids = wt2_ids(tok, "test", toktag)
train_ids = wt2_ids(tok, "train", toktag)
model = load_model(args.model)
snap = snapshot_weights(model)
store = Store(f"spine_{args.tag}")
MW = args.max_windows or None

n_all = sum(p.numel() for p in model.parameters())
n_q = sum(w.numel() for w in snap.values())
store.put("_meta", {"model": args.model, "n_params": n_all, "n_quantized_set": n_q,
                    "n_other_fp16": n_all - n_q, "seqlen": SEQLEN, "device": DEV,
                    "eval": "WikiText-2 test, non-overlapping 2048 windows; KL/top1 vs fp16 on first %d windows" % args.kl_windows,
                    "calib": f"{args.ncalib} x 2048 random windows of WikiText-2 train, seed 0"})

ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"), args.kl_windows)

_H = {}


def hessians():
    if "H" not in _H:
        restore_weights(model, snap)
        calib = calib_windows(train_ids, args.ncalib)
        share = QWEN_SHARE if "Qwen" in args.model else {}
        _H["H"] = collect_hessians(model, calib, share)
    return _H["H"]


def row_rms(W):
    return W.pow(2).mean(1, keepdim=True).sqrt()


def make_fn(cfg):
    k = cfg["kind"]
    if k == "rtn":
        return lambda n, W: rtn_asym(W, cfg["bits"], cfg["group"])
    if k == "gptq":
        H = hessians()
        return lambda n, W: gptq(W, H[n], "rtn", bits=cfg["bits"], group=cfg["group"])
    if k == "ecsq_row":       # step = c * rms(row), stored fp16 per row
        def f(n, W):
            st = (cfg["c"] * row_rms(W)).half().float()
            s, d = ecsq(W, st)
            return s, d, W.shape[0] * 16
        return f
    if k == "gptq_ecsq_row":
        H = hessians()
        def f(n, W):
            st = (cfg["c"] * row_rms(W)).half().float()
            s, d, _ = gptq(W, H[n], "grid", step=st)
            return s, d, W.shape[0] * 16
        return f
    raise ValueError(k)


def fixed_bpw(cfg, I_list):
    if cfg["kind"] in ("rtn", "gptq"):
        b, g = cfg["bits"], cfg["group"]
        # weighted over tensors (per-channel overhead depends on in_features)
        num = sum(O * I * b + O * (I // (I if g <= 0 else g)) * (16 + b) for O, I in I_list)
        return num / sum(O * I for O, I in I_list)
    return None


CONFIGS = [("fp16", {"kind": "fp16"}),
           ("rtn_int8_pc", {"kind": "rtn", "bits": 8, "group": -1}),
           ("rtn_int4_g128", {"kind": "rtn", "bits": 4, "group": 128}),
           ("rtn_int3_g128", {"kind": "rtn", "bits": 3, "group": 128}),
           ("rtn_int4_pc", {"kind": "rtn", "bits": 4, "group": -1}),
           ("gptq_int4_g128", {"kind": "gptq", "bits": 4, "group": 128}),
           ("gptq_int3_g128", {"kind": "gptq", "bits": 3, "group": 128}),
           ("rtn_int2_g128", {"kind": "rtn", "bits": 2, "group": 128}),
           ("gptq_int2_g128", {"kind": "gptq", "bits": 2, "group": 128}),
           ]
for c in (0.25, 0.35, 0.5, 0.7, 1.0):
    CONFIGS.append((f"ecsq_row_c{c}", {"kind": "ecsq_row", "c": c}))
for c in (0.35, 0.5, 0.7, 1.0, 1.4):
    CONFIGS.append((f"gptq_ecsq_row_c{c}", {"kind": "gptq_ecsq_row", "c": c}))

only = set(args.only.split(",")) if args.only else None
shapes = [tuple(w.shape) for w in snap.values()]
for name, cfg in CONFIGS:
    if only and name not in only:
        continue
    if store.has(name):
        log("skip", name, store.get(name)["ppl"]); continue
    check_disk()
    t0 = time.time()
    if cfg["kind"] == "fp16":
        restore_weights(model, snap)
        acc = {"entropy_bpw": 16.0, "side_bpw": 0.0}
        fb = 16.0
    else:
        acc = quantize_model(model, snap, make_fn(cfg))
        fb = fixed_bpw(cfg, shapes)
    tq = time.time() - t0
    ev = eval_lm(model, test_ids, max_windows=MW, ref=ref)
    rec = {"cfg": cfg, "fixed_bpw": fb, "entropy_bpw": acc["entropy_bpw"],
           "side_bpw": acc["side_bpw"], "quant_s": round(tq, 1), **ev}
    if "per_tensor" in acc:
        rec["per_tensor_H"] = {k: v["H"] for k, v in acc["per_tensor"].items()}
    store.put(name, rec)
    log(f"{name}: fixed={fb} ent={acc['entropy_bpw']:.3f} ppl={ev['ppl']:.3f} kl={ev['kl']:.4f} top1={ev['top1']:.3f} (q {tq:.0f}s, ev {ev['eval_s']}s)")
restore_weights(model, snap)
log("done; disk free GB", round(disk_free_gb(), 2))
