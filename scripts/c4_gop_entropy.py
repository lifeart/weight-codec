"""C2/C4 (rate side, CPU): closed-loop P-frame coding between family members on a uniform
grid.  For target t with reference r (decoded reference R^ = ECSQ(W_r, step_r)):
  I-frame   : symbols round(W_t / s)                                 -> H_I
  P (a=1)   : round((W_t - R^)/s)                                     -> H_P1
  P (a_row) : round((W_t - a_i R^_i)/s), a_i least-squares per row (fp16 side) -> H_Pa
All variants have the SAME quantization error statistics (uniform rounding of a residual
with step s), so equal step == equal weight-MSE (measured & reported); rate is compared at
equal step.  s = c * rms_row(W_t), c in {0.35 (~int4-class), 0.7 (~int3-class)}.
Chains: star (everything from base) vs chain base->coder->coder_instruct, plus OPEN-LOOP
drift (residual against the ORIGINAL reference, decoded against the reconstruction).
"""
from common import *
M = {"base": "Qwen/Qwen2.5-0.5B", "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
     "coder": "Qwen/Qwen2.5-Coder-0.5B", "coder_instruct": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}
SD = {n: LazySD(r) for n, r in M.items()}
keys = [k for k in SD["base"].keys() if k.startswith("model.layers.") and k.endswith("proj.weight")]
EDGES = [("base", "instruct"), ("base", "coder"), ("coder", "coder_instruct"), ("base", "coder_instruct"),
         ("instruct", "coder_instruct")]
CS = (0.35, 0.7)
acc = {}
def add(key, n, bits):
    a = acc.setdefault(key, [0.0, 0]); a[0] += bits; a[1] += n
t0 = time.time()
for k in keys:
    W = {n: SD[n][k].float() for n in M}
    for c in CS:
        st = {n: (c * W[n].pow(2).mean(1, keepdim=True).sqrt()).half().float() for n in M}
        dec = {n: torch.round(W[n] / st[n]) * st[n] for n in M}
        for n in M:
            add(f"I/{n}/c{c}", W[n].numel(), coded_bits(torch.round(W[n] / st[n]).to(torch.int32))[0] + 16 * W[n].shape[0])
            add(f"mse/I/{n}/c{c}", W[n].numel(), float((dec[n] - W[n]).pow(2).sum() / W[n].pow(2).sum()) * W[n].numel())
        for r, t in EDGES:
            s = st[t]
            q1 = torch.round((W[t] - dec[r]) / s)
            add(f"P1/{r}->{t}/c{c}", W[t].numel(), coded_bits(q1.to(torch.int32))[0] + 16 * W[t].shape[0])
            a = ((W[t] * dec[r]).sum(1, keepdim=True) / dec[r].pow(2).sum(1, keepdim=True).clamp(min=1e-20)).half().float()
            qa = torch.round((W[t] - a * dec[r]) / s)
            add(f"Pa/{r}->{t}/c{c}", W[t].numel(), coded_bits(qa.to(torch.int32))[0] + 32 * W[t].shape[0])
            rec = a * dec[r] + qa * s
            add(f"mse/Pa/{r}->{t}/c{c}", W[t].numel(), float((rec - W[t]).pow(2).sum() / W[t].pow(2).sum()) * W[t].numel())
        # chain base->coder->coder_instruct, closed loop vs open loop (alpha per row)
        s_c, s_ci = st["coder"], st["coder_instruct"]
        a1 = ((W["coder"] * dec["base"]).sum(1, keepdim=True) / dec["base"].pow(2).sum(1, keepdim=True)).half().float()
        rec_c = a1 * dec["base"] + torch.round((W["coder"] - a1 * dec["base"]) / s_c) * s_c          # closed loop
        a2 = ((W["coder_instruct"] * rec_c).sum(1, keepdim=True) / rec_c.pow(2).sum(1, keepdim=True)).half().float()
        q2 = torch.round((W["coder_instruct"] - a2 * rec_c) / s_ci)
        rec_ci = a2 * rec_c + q2 * s_ci
        add(f"chainCL/coder_instruct/c{c}", q2.numel(), coded_bits(q2.to(torch.int32))[0] + 32 * q2.shape[0])
        add(f"mse/chainCL/coder_instruct/c{c}", q2.numel(), float((rec_ci - W["coder_instruct"]).pow(2).sum() / W["coder_instruct"].pow(2).sum()) * q2.numel())
        # open loop: residuals vs ORIGINAL refs, decoder only has reconstructions -> drift
        a1o = ((W["coder"] * W["base"]).sum(1, keepdim=True) / W["base"].pow(2).sum(1, keepdim=True)).half().float()
        q1o = torch.round((W["coder"] - a1o * W["base"]) / s_c)
        rec_c_o = a1o * dec["base"] + q1o * s_c
        a2o = ((W["coder_instruct"] * W["coder"]).sum(1, keepdim=True) / W["coder"].pow(2).sum(1, keepdim=True)).half().float()
        q2o = torch.round((W["coder_instruct"] - a2o * W["coder"]) / s_ci)
        rec_ci_o = a2o * rec_c_o + q2o * s_ci
        add(f"chainOL/coder_instruct/c{c}", q2o.numel(), coded_bits(q2o.to(torch.int32))[0] + 32 * q2o.shape[0])
        add(f"mse/chainOL/coder/c{c}", q2o.numel(), float((rec_c_o - W["coder"]).pow(2).sum() / W["coder"].pow(2).sum()) * q2o.numel())
        add(f"mse/chainOL/coder_instruct/c{c}", q2o.numel(), float((rec_ci_o - W["coder_instruct"]).pow(2).sum() / W["coder_instruct"].pow(2).sum()) * q2o.numel())
out = {k: v[0] / v[1] for k, v in acc.items()}
log(f"{len(keys)} tensors {time.time() - t0:.0f}s")
for k in sorted(out):
    print(f"{k:45s} {out[k]:.5f}")
save_json("c4_gop_entropy", {"note": "bpw over block linears incl. per-row fp16 side info; mse/* = relative weight MSE ||W^-W||^2/||W||^2",
                             "bpw_or_relmse": out})
