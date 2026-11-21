"""D2: 'video of training' codec for Pythia-410M checkpoints (runs only if D1 PASSED, or --force).

GOP windows of consecutive 1000-step checkpoints (streamed one at a time, deleted after load):
  late: step136000..143000 (8 frames)      mid: step32000..37000 (6 frames)
LOSSY (block linears, ECSQ on a uniform grid, step s = c*rms_row(W_I-frame), fixed per GOP):
  all-I     : every frame independently           bits = H(round(W_t/s))
  IPPP      : closed loop, P_t = round((W_t - W^_{t-1})/s), W^_t = W^_{t-1} + s*P_t
  K=4       : I-frame every 4 frames, P in between (closed loop)
  -> equal step == equal weight-MSE (reported); PPL of reconstructed last frame (P-chain vs I)
LOSSLESS (bit-exact fp16 archive of the checkpoints):
  I: byte-split entropy H(hi byte)+H(lo byte|hi-byte-class) ; P: same on (bits_t XOR bits_{t-1})
  (256-entry tables, cost counted) plus the 16-bit-symbol entropy (no table cost, lower bound).
"""
import argparse, shutil
from common import *
from probe_d1 import fetch, is_lin, REPO, fp16_bits

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
ap.add_argument("--cs", default="0.35,0.7")
args = ap.parse_args()
v = Store("probe_d1").get("_verdict", {})
if v.get("D1") != "PASS" and not args.force:
    raise SystemExit(f"D1 verdict is {v.get('D1')} -> D2 not built (falsification-first). Use --force to override.")
CS = [float(c) for c in args.cs.split(",")]
WINDOWS = {"late": list(range(136000, 143001, 1000)), "mid": list(range(32000, 37001, 1000))}
store = Store("d2_pythia")


def byte_split_bits(b16):
    hi = (b16 >> 8) & 0xFF; lo = b16 & 0xFF
    bits = coded_bits(hi)[0]
    # low byte conditioned on 16 classes of the high byte (sign+exponent bucket)
    cls = hi >> 4
    for c in torch.unique(cls):
        bits += coded_bits(lo[cls == c])[0]
    return bits


def build_model(sd):
    from transformers import AutoConfig, GPTNeoXForCausalLM
    cfg = AutoConfig.from_pretrained(REPO, revision="step143000")
    m = GPTNeoXForCausalLM(cfg).to(torch.float16)
    missing, unexpected = m.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()}, strict=False)
    real_missing = [k for k in missing if not k.endswith(("rotary_emb.inv_freq", "attention.bias", "attention.masked_bias"))]
    assert not real_missing, real_missing[:5]
    m.eval(); m.config.use_cache = False
    return m.to(DEV)


tok = load_tok(REPO)
test_ids = wt2_ids(tok, "test", "pythia")


def ppl_of(sd):
    m = build_model(sd)
    r = eval_lm(m, test_ids, max_windows=40)
    del m; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return r


for wname, steps in WINDOWS.items():
    if store.has(f"{wname}_summary"):
        continue
    ntot = None
    acc = {c: {"I": [], "IPPP": [], "K4": [], "mse_I": [], "mse_IPPP": [], "mse_K4": []} for c in CS}
    ll = {"I_bytes": [], "P_bytes": [], "I_H16": [], "P_H16": []}
    rec = {c: {"IPPP": None, "K4": None} for c in CS}
    step_I = {}
    prev = None
    first_sd = None
    for fi, st in enumerate(steps):
        t0 = time.time()
        sd = fetch(st)
        keys = [k for k in sd if is_lin(k) and sd[k].dim() == 2]   # excl. 1-D post_attention_layernorm
        ntot = sum(sd[k].numel() for k in keys)
        # lossless
        bI = bP = hI = hP = 0.0
        for k in keys:
            b = fp16_bits(sd[k])
            bI += byte_split_bits(b); hI += entropy_bits(b)[0] * b.numel()
            if prev is not None:
                x = b ^ fp16_bits(prev[k])
                bP += byte_split_bits(x); hP += entropy_bits(x)[0] * x.numel()
        ll["I_bytes"].append(bI / ntot); ll["I_H16"].append(hI / ntot)
        if prev is not None:
            ll["P_bytes"].append(bP / ntot); ll["P_H16"].append(hP / ntot)
        # lossy
        for c in CS:
            if fi == 0:
                step_I[c] = {k: (c * sd[k].float().pow(2).mean(1, keepdim=True).sqrt()).half().float() for k in keys}
            bitsI = bitsP = bitsK = 0.0
            mI = mP = mK = 0.0
            newP, newK = {}, {}
            for k in keys:
                W = sd[k].float(); s = step_I[c][k]
                qI = torch.round(W / s); recI = (qI * s).half().float()   # decoder state kept in fp16
                bitsI += coded_bits(qI.to(torch.int32))[0]; mI += float((recI - W).pow(2).sum() / W.pow(2).sum()) * W.numel()
                if fi == 0:
                    newP[k] = recI.half(); bitsP += coded_bits(qI.to(torch.int32))[0]; mP += float((recI - W).pow(2).sum() / W.pow(2).sum()) * W.numel()
                else:
                    pr = rec[c]["IPPP"][k].float()
                    q = torch.round((W - pr) / s); newP[k] = (pr + q * s).half()
                    bitsP += coded_bits(q.to(torch.int32))[0]; mP += float((newP[k].float() - W).pow(2).sum() / W.pow(2).sum()) * W.numel()
                if fi % 4 == 0:
                    newK[k] = recI.half(); bitsK += coded_bits(qI.to(torch.int32))[0]; mK += float((recI - W).pow(2).sum() / W.pow(2).sum()) * W.numel()
                else:
                    pr = rec[c]["K4"][k].float()
                    q = torch.round((W - pr) / s); newK[k] = (pr + q * s).half()
                    bitsK += coded_bits(q.to(torch.int32))[0]; mK += float((newK[k].float() - W).pow(2).sum() / W.pow(2).sum()) * W.numel()
            side = sum(16 * sd[k].shape[0] for k in keys)
            a = acc[c]
            a["I"].append((bitsI + side) / ntot); a["IPPP"].append((bitsP + (side if fi == 0 else 0)) / ntot)
            a["K4"].append((bitsK + (side if fi % 4 == 0 else 0)) / ntot)
            a["mse_I"].append(mI / ntot); a["mse_IPPP"].append(mP / ntot); a["mse_K4"].append(mK / ntot)
            rec[c]["IPPP"], rec[c]["K4"] = newP, newK
            del newP, newK
        log(f"{wname} step{st}: lossless I={ll['I_bytes'][-1]:.2f} P={ll['P_bytes'][-1] if ll['P_bytes'] and fi else float('nan'):.2f} | " +
            " ".join(f"c{c}: I={acc[c]['I'][-1]:.3f} P={acc[c]['IPPP'][-1]:.3f}" for c in CS) + f" ({time.time() - t0:.0f}s)")
        if fi == 0:
            first_sd = None
        prev = {k: sd[k].clone() for k in keys}
        last_sd = sd
        gc.collect()
    # PPL check on the LAST frame: original, all-I reconstruction, end of closed-loop P chain
    ev = {"orig": ppl_of(last_sd)}
    for c in CS:
        sdI = dict(last_sd); sdP = dict(last_sd)
        for k in step_I[c]:
            sdI[k] = torch.round(last_sd[k].float() / step_I[c][k]) * step_I[c][k]
            sdP[k] = rec[c]["IPPP"][k].float()
        ev[f"I_c{c}"] = ppl_of(sdI); ev[f"IPPP_end_c{c}"] = ppl_of(sdP)
        log(wname, c, {k: round(v["ppl"], 3) for k, v in ev.items()})
    summ = {"steps": steps, "lossless": ll, "lossy": {str(c): acc[c] for c in CS}, "ppl_last_frame": ev,
            "mean": {"lossless_I": float(np.mean(ll["I_bytes"])), "lossless_P": float(np.mean(ll["P_bytes"])),
                     **{f"c{c}_{k}": float(np.mean(acc[c][k][1:] if k != "I" else acc[c][k])) for c in CS for k in ("I", "IPPP", "K4")},
                     **{f"c{c}_gop_total_{k}": float(np.sum(acc[c][k])) for c in CS for k in ("I", "IPPP", "K4")}}}
    store.put(f"{wname}_summary", summ)
    log(wname, json.dumps(summ["mean"], indent=1))
    del prev, rec, last_sd; gc.collect()
