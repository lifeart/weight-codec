"""Unit check: GPTQ must reduce layer-output error tr((W-Q) H (W-Q)^T) vs RTN on the same grid."""
from common import *
rid = "Qwen/Qwen2.5-0.5B"
tok = load_tok(rid); tr = wt2_ids(tok, "train", "qwen25")
m = load_model(rid)
H = collect_hessians(m, calib_windows(tr, 8), QWEN_SHARE)
snap = snapshot_weights(m)
out = {}
for n in ["layers.3.self_attn.q_proj", "layers.10.mlp.down_proj", "layers.20.mlp.gate_proj"]:
    W = snap[n].float(); Hn = H[n]
    err = lambda Q: float(torch.einsum("oi,ij,oj->", W - Q, Hn, W - Q) / torch.einsum("oi,ij,oj->", W, Hn, W))
    t = time.time()
    _, Qr, _ = rtn_asym(W, 4, 128); _, Qg, _ = gptq(W, Hn, "rtn", 4, 128)
    st = (0.5 * W.pow(2).mean(1, keepdim=True).sqrt()).half().float()
    _, Qe = ecsq(W, st); Se, Qge, _ = gptq(W, Hn, "grid", step=st)
    Sr, _ = ecsq(W, st)
    out[n] = dict(rtn4=err(Qr), gptq4=err(Qg), ecsq=err(Qe), gptq_ecsq=err(Qge),
                  H_ecsq=entropy_bits(Sr)[0], H_gptq_ecsq=entropy_bits(Se)[0], s=time.time() - t)
    log(n, out[n])
save_json("unit_gptq", out)
