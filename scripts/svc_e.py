"""E: SVC / progressive bitstream.  ONE representation, several quality points.

Layer 0 : ECSQ with step s0 = c0 * rms_row(W)            (c0 = 1.4, ~int2-class)
Layer k : residual refinement r_k = W - W^_{k-1}, q_k = round(r_k / s_k), s_k = s0 / 2^k
          W^_k = W^_{k-1} + s_k q_k
Variants: 'rtn'  (plain residual rounding)
          'gptq' (every layer is GPTQ on the residual with the layer Hessian: closed-loop,
                  Hessian-aware refinement)
Rate of layer k: per-tensor order-0 entropy H(q_k)  AND context-conditioned entropy
H(q_k | ctx), ctx = clip(q_{k-1}, -3, 3) of the same weight (7 contexts, table cost counted).
Side info: one fp16 step per row (shared by all layers: steps are fixed ratios).
PPL / KL measured at every cumulative layer -> compare with single-shot ECSQ / GPTQ-ECSQ
at the same final step and with fixed-rate int2/int3/int4 (spine).
"""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--rid", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--c0", type=float, default=1.4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--ratio", type=int, default=2, help="step ratio between layers; 3 = exactly nested midtread cells")
ap.add_argument("--variants", default="rtn,gptq")
args = ap.parse_args()
seed_all(0); check_disk()
tok = load_tok(args.rid)
test_ids = wt2_ids(tok, "test", "qwen25"); train_ids = wt2_ids(tok, "train", "qwen25")
model = load_model(args.rid)
snap = snapshot_weights(model)
ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
store = Store(f"svc_e_{args.tag}")
lin = block_linears(model)
ntot = sum(snap[n].numel() for n, _ in lin)


def cond_bits(q, ctx):
    tot = 0.0
    for v in torch.unique(ctx):
        m = ctx == v
        tot += coded_bits(q[m])[0]
    return tot


_H = {}
PFX = "" if args.ratio == 2 else f"r{args.ratio}_"
for variant in args.variants.split(","):
    if all(store.has(f"{PFX}{variant}_L{k}") for k in range(args.layers)):
        continue
    if variant == "gptq" and "H" not in _H:
        restore_weights(model, snap)
        _H["H"] = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
    rec_w = {n: torch.zeros_like(snap[n], dtype=torch.float32) for n, _ in lin}
    prev_q = {}
    cum_bits = cum_bits_ctx = 0.0
    side = sum(16 * snap[n].shape[0] for n, _ in lin)
    for k in range(args.layers):
        bits = bits_ctx = 0.0
        for n, m in lin:
            W = snap[n].float()
            s = ((args.c0 * W.pow(2).mean(1, keepdim=True).sqrt()).half().float()) / (args.ratio ** k)
            R = W - rec_w[n]
            if variant == "rtn":
                q = torch.round(R / s).to(torch.int32)
                rec_w[n] = rec_w[n] + q.float() * s
            else:
                q, D, _ = gptq(R, _H["H"][n], "grid", step=s)
                rec_w[n] = rec_w[n] + D
            b = coded_bits(q)[0]
            bits += b
            bits_ctx += cond_bits(q, prev_q[n].clamp(-3, 3)) if n in prev_q else b
            prev_q[n] = q
            m.weight.data.copy_(rec_w[n].to(m.weight.device, m.weight.dtype))
        cum_bits += bits; cum_bits_ctx += bits_ctx
        name = f"{PFX}{variant}_L{k}"
        if store.has(name):
            continue
        check_disk()
        ev = eval_lm(model, test_ids, ref=ref)
        r = {"layer": k, "ratio": args.ratio, "c_equiv": args.c0 / args.ratio ** k, "layer_bpw": bits / ntot, "layer_bpw_ctx": bits_ctx / ntot,
             "cum_bpw": (cum_bits + side) / ntot, "cum_bpw_ctx": (cum_bits_ctx + side) / ntot, **ev}
        store.put(name, r)
        log(f"{name}: layer={r['layer_bpw']:.3f} (ctx {r['layer_bpw_ctx']:.3f}) cum={r['cum_bpw']:.3f} "
            f"(ctx {r['cum_bpw_ctx']:.3f}) ppl={ev['ppl']:.3f} kl={ev['kl']:.4f}")
restore_weights(model, snap)
