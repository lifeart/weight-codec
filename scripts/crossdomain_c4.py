"""Honesty check: all calibration (GPTQ Hessians, Fisher) used WikiText-2 *train*; the spine evaluates
on WikiText-2 *test* (same domain).  Re-evaluate the key intra-model coders on an OUT-OF-DOMAIN set:
C4 validation (allenai/c4, en, shard 0), first 64 x 2048-token windows of the concatenated documents,
plus KL/top-1 vs fp16 on the first 16 windows.  Quantizers are re-run exactly as in spine.py /
a3_fisher.py (same seeds, same calibration)."""
import gzip
from common import *
from huggingface_hub import hf_hub_download

RID, TAG = "Qwen/Qwen2.5-0.5B", "qwen05b"
seed_all(0); check_disk()
tok = load_tok(RID)
c4_path = os.path.join(DATA, "c4_val_qwen25.pt")
if os.path.exists(c4_path):
    c4 = torch.load(c4_path)
else:
    f = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    texts = []
    with gzip.open(f, "rt") as fh:
        for i, line in enumerate(fh):
            texts.append(json.loads(line)["text"])
            if i >= 3000:
                break
    c4 = tok("\n\n".join(texts), return_tensors="pt").input_ids[0][:64 * SEQLEN + 1].to(torch.int32)
    torch.save(c4, c4_path)
train_ids = wt2_ids(tok, "train", "qwen25")
model = load_model(RID)
snap = snapshot_weights(model)
ref = attach_head(build_ref(model, c4, 16), model)
H = None
stats = torch.load(os.path.join(DATA, f"fisher_{TAG}.pt"))
rms = {n: snap[n].float().pow(2).mean(1, keepdim=True).sqrt() for n in snap}
ntot = sum(w.numel() for w in snap.values())
L0 = sum(float(torch.log(rms[n]).sum()) * snap[n].shape[1] for n in snap) / ntot
logu, acc = {}, 0.0
for n in snap:
    O, I = snap[n].shape
    ur = -0.5 * (torch.log(stats[n]["row"].double().clamp(min=1e-30)) - math.log(max(stats[n]["mean"], 1e-30)))
    uc = -0.5 * torch.log(stats[n]["col"].double().clamp(min=1e-30))
    logu[n] = (ur, uc); acc += float(ur.sum()) * I + float(uc.sum()) * O
shift = L0 - acc / ntot


def fisher_step(n, lam, gamma):
    ur, uc = logu[n]
    L = math.log(lam) + (1 - gamma) * torch.log(rms[n][:, 0].double())[:, None] + gamma * (ur[:, None] + uc[None, :] + shift)
    return torch.exp(L).float().half().float().clamp(min=1e-7)


def hess():
    global H
    if H is None:
        restore_weights(model, snap)
        H = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
    return H


CFG = {
    "fp16": None,
    "rtn_int4_g128": lambda n, W: rtn_asym(W, 4, 128),
    "gptq_int4_g128": lambda n, W: gptq(W, hess()[n], "rtn", 4, 128),
    "ecsq_row_c0.35": lambda n, W: (*ecsq(W, (0.35 * rms[n]).half().float()), W.shape[0] * 16),
    "gptq_ecsq_row_c0.35": lambda n, W: (lambda q, D, _: (q, D, W.shape[0] * 16))(*gptq(W, hess()[n], "grid", step=(0.35 * rms[n]).half().float())),
    "gptq_ecsq_row_c0.5": lambda n, W: (lambda q, D, _: (q, D, W.shape[0] * 16))(*gptq(W, hess()[n], "grid", step=(0.5 * rms[n]).half().float())),
    "fisher_g0.5_lam0.5": lambda n, W: (*ecsq(W, fisher_step(n, 0.5, 0.5)), 16 * sum(W.shape)),
    "gptq_fisher_g0.5_lam0.5": lambda n, W: (lambda q, D, _: (q, D, 16 * sum(W.shape)))(*gptq(W, hess()[n], "grid", step=fisher_step(n, 0.5, 0.5))),
    "gptq_fisher_g0.5_lam0.35": lambda n, W: (lambda q, D, _: (q, D, 16 * sum(W.shape)))(*gptq(W, hess()[n], "grid", step=fisher_step(n, 0.35, 0.5))),
}
store = Store(f"crossdomain_c4_{TAG}")
for name, fn in CFG.items():
    if store.has(name):
        continue
    check_disk()
    if fn is None:
        restore_weights(model, snap); bpw = 16.0
    else:
        bpw = quantize_model(model, snap, fn)["entropy_bpw"]
    ev = eval_lm(model, c4, ref=ref)
    store.put(name, {"entropy_bpw": bpw, **ev})
    log(f"{name}: bpw={bpw:.3f} c4_ppl={ev['ppl']:.3f} kl={ev['kl']:.4f} top1={ev['top1']:.3f}")
restore_weights(model, snap)
