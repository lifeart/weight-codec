"""C2 (family = GOP codec): code the INSTRUCT model given the BASE model.

Quantized set = block linears (embeddings/norms of the target stay fp16 in every config;
embedding-delta cost is reported separately by c2_embedding.py).  Distortion is measured
against the fp16 INSTRUCT model: KL & top-1 on WikiText-2 (first 16 windows), WikiText-2
PPL, and chat-behaviour metrics (NLL / KL / top-1 on the instruct model's own greedy answers
to 48 fixed prompts).  Rate = bits per weight of the INSTRUCT stream (conditional on base).

Families of configs:
  zero_bits_base       instruct := base (0 bits) - the 'do nothing' anchor
  indep_*              instruct quantized alone (RTN / GPTQ / ECSQ / GPTQ-ECSQ), I-frame
  delta_ecsq_c*        fp16 base available: W = W_base + s*round(Delta/s), s = c*rms_row(W_inst)
  delta_gptq_c*        same grid but GPTQ (instruct Hessian) on Delta: Hessian-aware P-frame
  delta_lowrank_r*     W = W_base + SVD_r(Delta), fp16 factors (the LoRA-style hypothesis)
  pframe_rtn_c*        QUANTIZED base on grid s = c*rms_row(W_base); instruct on the same
                       grid; instruct stream = q_inst - q_base (closed loop == indep. RTN
                       reconstruction on that grid, bits = H(diff))
  pframe_gptq_c*       closed loop: R = W_inst - W^_base(RTN grid), GPTQ(R, H_inst) on grid
"""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--tag", default="qwen05b_instruct")
ap.add_argument("--only", default="")
ap.add_argument("--max_windows", type=int, default=0)
args = ap.parse_args()
seed_all(0); check_disk()

tok = load_tok(args.target)
test_ids = wt2_ids(tok, "test", "qwen25"); train_ids = wt2_ids(tok, "train", "qwen25")
model = load_model(args.target)
snap = snapshot_weights(model)
B = LazySD(args.base)
WB16 = {n: B[f"model.{n}.weight"].to(torch.float16) for n in snap}   # fp16 in RAM
class _WB(dict):
    def __getitem__(self, n):
        return WB16[n].float()
WB = _WB()
ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))

chat_path = os.path.join(RES, f"chat_set_{args.tag}.json")
if os.path.exists(chat_path):
    chat = json.load(open(chat_path))
else:
    prompts = json.load(open(os.path.join(ROOT, "scripts", "chat_prompts.json")))
    chat = build_chat_set(model, tok, prompts)
    json.dump(chat, open(chat_path, "w"))
chref_path = os.path.join(DATA, f"chatref_{args.tag}.pt")
if os.path.exists(chref_path):
    chat_ref = torch.load(chref_path)
else:
    chat_ref = chat_hidden(model, chat); torch.save(chat_ref, chref_path)

store = Store(f"c2_{args.tag}")
_H = {}
def hess():
    if "H" not in _H:
        restore_weights(model, snap)
        _H["H"] = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
    return _H["H"]

rms = lambda W: W.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)


def cfgs():
    C = [("fp16", None), ("zero_bits_base", lambda n, W: (None, WB[n], 0))]
    C += [("indep_rtn_int4_g128", lambda n, W: rtn_asym(W, 4, 128)),
          ("indep_gptq_int4_g128", lambda n, W: gptq(W, hess()[n], "rtn", 4, 128)),
          ("indep_gptq_int3_g128", lambda n, W: gptq(W, hess()[n], "rtn", 3, 128))]
    for c in (0.35, 0.7, 1.4):
        C.append((f"pframe_rtn_c{c}", (lambda c: lambda n, W: _pframe_rtn(n, W, c))(c)))
    # NOTE: Delta ~ 0.077*rms(W) -> steps relative to rms(W) must be << 0.35 to code anything
    # (first sweep 0.7..5.6 rounded ~all of Delta to 0: kept only c0.7 as the ~0-bit end point)
    for c in (0.03, 0.06, 0.12, 0.25):
        C.append((f"delta_ecsq_c{c}", (lambda c: lambda n, W: _delta_ecsq(n, W, c))(c)))
    for c in (0.06, 0.12, 0.25, 0.5):
        C.append((f"delta_gptq_c{c}", (lambda c: lambda n, W: _delta_gptq(n, W, c))(c)))
    for r in (32, 128):
        C.append((f"delta_lowrank_r{r}", (lambda r: lambda n, W: _delta_lowrank(n, W, r))(r)))
    for c in (0.35, 0.5, 0.7):
        C.append((f"pframe_gptq_c{c}", (lambda c: lambda n, W: _pframe_gptq(n, W, c))(c)))
    for c in (0.35, 0.5, 0.7, 1.0):
        C.append((f"indep_gptq_ecsq_c{c}", (lambda c: lambda n, W: _gptq_ecsq(n, W, c))(c)))
    C += [("indep_rtn_int3_g128", lambda n, W: rtn_asym(W, 3, 128))]
    # I-frame = GPTQ-ECSQ base (best intra coder of the spine; base Hessian approximated by the
    # instruct Hessian - only affects how good the *base* reconstruction is), P = GPTQ residual
    for c in (0.35, 0.5):
        C.append((f"pframe_gptq2_c{c}", (lambda c: lambda n, W: _pframe_gptq2(n, W, c))(c)))
    return C


def _pframe_gptq2(n, W, c):
    s = (c * rms(WB[n])).half().float()
    _, WBq, _ = gptq(WB[n], hess()[n], "grid", step=s)
    q, R, _ = gptq(W - WBq, hess()[n], "grid", step=s)
    return q, WBq + R, W.shape[0] * 16


def _pframe_rtn(n, W, c):
    s = (c * rms(WB[n])).half().float()
    qb = torch.round(WB[n] / s); qi = torch.round(W / s)
    return (qi - qb).to(torch.int32), qi * s, W.shape[0] * 16


def _delta_ecsq(n, W, c):
    s = (c * rms(W)).half().float()
    q = torch.round((W - WB[n]) / s)
    return q.to(torch.int32), WB[n] + q * s, W.shape[0] * 16


def _delta_gptq(n, W, c):
    s = (c * rms(W)).half().float()
    q, D, _ = gptq(W - WB[n], hess()[n], "grid", step=s)
    return q, WB[n] + D, W.shape[0] * 16


def _delta_lowrank(n, W, r):
    U, S, Vh = torch.linalg.svd(W - WB[n], full_matrices=False)
    A = (U[:, :r] * S[:r]).half().float(); Bm = Vh[:r].half().float()
    return None, WB[n] + A @ Bm, 16 * r * (W.shape[0] + W.shape[1])


def _pframe_gptq(n, W, c):
    s = (c * rms(WB[n])).half().float()
    WBq = torch.round(WB[n] / s) * s
    q, R, _ = gptq(W - WBq, hess()[n], "grid", step=s)
    return q, WBq + R, W.shape[0] * 16


def _gptq_ecsq(n, W, c):
    s = (c * rms(W)).half().float()
    q, D, _ = gptq(W, hess()[n], "grid", step=s)
    return q, D, W.shape[0] * 16


only = set(args.only.split(",")) if args.only else None
for name, fn in cfgs():
    if (only and name not in only) or store.has(name):
        continue
    check_disk()
    t0 = time.time()
    if fn is None:
        restore_weights(model, snap); acc = {"entropy_bpw": 16.0, "side_bpw": 0.0}
    else:
        acc = quantize_model(model, snap, fn)
    tq = time.time() - t0
    ev = eval_lm(model, test_ids, max_windows=args.max_windows or None, ref=ref)
    ev.update(eval_chat(model, chat, chat_ref, ref["head"]))
    rec = {"bpw": acc["entropy_bpw"], "side_bpw": acc["side_bpw"], "quant_s": round(tq, 1), **ev}
    store.put(name, rec)
    log(f"{name:24s} bpw={rec['bpw']:.3f} ppl={ev['ppl']:.3f} kl={ev['kl']:.4f} top1={ev['top1']:.3f} "
        f"chat_kl={ev['chat_kl']:.4f} chat_nll={ev['chat_nll']:.3f} chat_top1={ev['chat_top1']:.3f}")
restore_weights(model, snap)
log("done; disk free GB", round(disk_free_gb(), 2))
