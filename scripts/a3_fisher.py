"""A3: sensitivity-weighted quantization ("JPEG quant table by sensitivity").

Step 1 (probe): empirical Fisher diagonal F = sum_s (dL_s/dW)^2 over calibration sequences
(64 x 512 tokens WikiText-2 train, fp32 on MPS).  High-rate theory for ECSQ with per-weight
distortion weight F_i:  optimal steps s_i ~ F_i^-1/2 and the gain over a baseline step
allocation s_i^0 (here c*rms_row) at equal rate is
        delta_bits = 0.5 * log2( AM(x) / GM(x) ),   x_i = F_i * (s_i^0)^2
evaluated at several side-info granularities (per matrix / per row / per column /
separable row x col / per row-x-128-col block).  PRE-REGISTERED: A3 PASS iff the predicted
gain at a granularity whose side info is <= 0.05 bpw exceeds 0.10 bpw.

Step 2 (codec, only if PASS): ECSQ / GPTQ-ECSQ with separable Fisher steps
  s_ij = lam * rms_row_i * (x_row_i * x_col_j / mean)^-1/2 normalised (geometric) so that
  lam plays the role of c; fp16 row + col factors as side info. RD vs spine lines.
usage: python scripts/a3_fisher.py [--codec]
"""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--rid", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--nseq", type=int, default=64)
ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--codec", action="store_true")
args = ap.parse_args()
seed_all(0); check_disk()
tok = load_tok(args.rid)
train_ids = wt2_ids(tok, "train", "qwen25"); test_ids = wt2_ids(tok, "test", "qwen25")
fpath = os.path.join(DATA, f"fisher_{args.tag}.pt")

if not os.path.exists(fpath):
    model = load_model(args.rid, dtype=torch.float32)
    lin = block_linears(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for n, m in lin:
        m.weight.requires_grad_(True)
    F = {n: torch.zeros_like(m.weight) for n, m in lin}
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for i in range(args.nseq):
        s = int(torch.randint(0, train_ids.numel() - args.seqlen - 1, (1,), generator=g))
        x = train_ids[s:s + args.seqlen].long().unsqueeze(0).to(DEV)
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        for n, m in lin:
            F[n].add_(m.weight.grad.pow(2)); m.weight.grad = None
    log(f"fisher {args.nseq}x{args.seqlen} in {time.time() - t0:.0f}s")
    stats = {}
    for n, m in lin:
        f = (F[n] / args.nseq).cpu().double()
        O, I = f.shape
        stats[n] = {"row": f.mean(1).float(), "col": f.mean(0).float(), "mean": float(f.mean()),
                    "blk128": f.reshape(O, I // 128, 128).mean(-1).float(),
                    "logmean": float(torch.log(f.clamp(min=1e-30)).mean())}
    torch.save(stats, fpath)
    del model, F; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
stats = torch.load(fpath)

# ---------------- probe ----------------
sd = LazySD(args.rid)
W = {n: sd[f"model.{n}.weight"].float() for n in stats}
rms = {n: W[n].pow(2).mean(1, keepdim=True).sqrt() for n in W}


def am_gm_bits(xs, ws):
    xs, ws = torch.cat(xs).double(), torch.cat(ws).double()
    ws = ws / ws.sum()
    am = (ws * xs).sum(); gm = torch.exp((ws * torch.log(xs.clamp(min=1e-30))).sum())
    return float(0.5 * torch.log2(am / gm))


probe = {}
# baseline step^2 per weight ~ rms_row^2 ; x = F * s0^2 aggregated at each granularity
gran = {}
xs, ws = [], []
for n in W:
    O, I = W[n].shape
    xs.append(torch.tensor([stats[n]["mean"] * float(rms[n].pow(2).mean())])); ws.append(torch.tensor([float(O * I)]))
gran["per_matrix"] = (am_gm_bits(xs, ws), 16 * len(W) / sum(w.numel() for w in W.values()))
xs, ws = [], []
for n in W:
    O, I = W[n].shape
    xs.append(stats[n]["row"].double() * rms[n][:, 0].double() ** 2); ws.append(torch.full((O,), float(I)))
gran["per_row"] = (am_gm_bits(xs, ws), 0.0)    # baseline already stores a per-row step
xs, ws = [], []
for n in W:
    O, I = W[n].shape
    xs.append((stats[n]["blk128"].double() * rms[n].double() ** 2).flatten()); ws.append(torch.full((O * (I // 128),), 128.0))
gran["per_row_x_128col_block"] = (am_gm_bits(xs, ws), 16 / 128)
xs, ws = [], []
for n in W:
    O, I = W[n].shape
    sep = torch.outer(stats[n]["row"].double(), stats[n]["col"].double()) / max(stats[n]["mean"], 1e-30)
    # separable model of F; evaluate the TRUE x on blocks under separable allocation:
    # gain(separable) = 0.5 log2(AM(x)/GM(x)) - 0.5 log2(AM(x/sep)/GM(x/sep)) approximated on blk128
    xs.append((stats[n]["blk128"].double() * rms[n].double() ** 2).flatten())
    ws.append(torch.full((O * (I // 128),), 128.0))
probe = {k: {"pred_gain_bits": v[0], "extra_side_bpw": v[1], "net_bits": v[0] - v[1]} for k, v in gran.items()}
# separable allocation: residual dispersion after removing row & column factors
num, den, logs, wsum = 0.0, 0.0, 0.0, 0.0
xs_res, ws_res = [], []
for n in W:
    O, I = W[n].shape
    blk = stats[n]["blk128"].double() * rms[n].double() ** 2
    colg = stats[n]["col"].double().reshape(I // 128, 128).mean(-1)
    sep = torch.outer(stats[n]["row"].double() * rms[n][:, 0].double() ** 2, colg) / max(stats[n]["mean"], 1e-30)
    xs_res.append((blk / sep.clamp(min=1e-30)).flatten()); ws_res.append(torch.full((O * (I // 128),), 128.0))
resid_bits = am_gm_bits(xs_res, ws_res)
side_sep = 16 * sum(W[n].shape[0] + W[n].shape[1] for n in W) / sum(w.numel() for w in W.values())
probe["separable_row_x_col"] = {"pred_gain_bits": gran["per_row_x_128col_block"][0] - resid_bits,
                                "extra_side_bpw": side_sep - 16 * sum(W[n].shape[0] for n in W) / sum(w.numel() for w in W.values()),
                                }
probe["separable_row_x_col"]["net_bits"] = probe["separable_row_x_col"]["pred_gain_bits"] - probe["separable_row_x_col"]["extra_side_bpw"]
ok = [v["net_bits"] > 0.10 for k, v in probe.items() if v["extra_side_bpw"] <= 0.05]
probe["A3"] = "PASS" if any(ok) else "FAIL"
per_layer = {}
for n in W:
    L = int(n.split(".")[1]); per_layer.setdefault(L, []).append(stats[n]["mean"] * float(rms[n].pow(2).mean()))
probe["per_layer_mean_x"] = {L: float(np.mean(v)) for L, v in sorted(per_layer.items())}
probe["per_type_mean_x"] = {}
for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
    probe["per_type_mean_x"][t] = float(np.mean([stats[n]["mean"] * float(rms[n].pow(2).mean()) for n in W if n.endswith(t)]))
save_json(f"probe_a3_{args.tag}", probe)
log(json.dumps({k: v for k, v in probe.items() if not k.startswith("per_")}, indent=1))

# ---------------- codec ----------------
if args.codec:
    model = load_model(args.rid)
    snap = snapshot_weights(model)
    ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
    store = Store(f"a3_codec_{args.tag}")
    _H = {}

    # separable Fisher model F_ij ~ Frow_i * Fcol_j / Fbar_n ; optimal ECSQ step s ~ F^-1/2.
    # log s_ij = logK - 0.5*(log Frow_i + log Fcol_j - log Fbar_n); K is GLOBAL (cross-layer
    # allocation) and set so that the weight-averaged log-step equals that of ECSQ_row at
    # c = lam (=> roughly equal rate at equal lam; RD curves are compared anyway).
    ntot = sum(w.numel() for w in W.values())
    L0 = sum(float(torch.log(rms[n]).sum()) * W[n].shape[1] for n in W) / ntot
    shapes = {n: tuple(W[n].shape) for n in W}
    logu = {}                       # separable: u_ij = ur_i + uc_j  (vectors only, keeps RAM flat)
    acc_log = 0.0
    for n in W:
        O, I = shapes[n]
        fr = stats[n]["row"].double().clamp(min=1e-30); fc = stats[n]["col"].double().clamp(min=1e-30)
        ur = -0.5 * (torch.log(fr) - math.log(max(stats[n]["mean"], 1e-30)))
        uc = -0.5 * torch.log(fc)
        logu[n] = (ur, uc)
        acc_log += float(ur.sum()) * I + float(uc.sum()) * O
    shift = L0 - acc_log / ntot
    del W
    gc.collect()

    logrms = {n: torch.log(rms[n][:, 0].double()) for n in rms}

    def make_fn(lam, use_gptq, gamma):
        """log s_ij = log lam + (1-gamma)*log rms_i + gamma*(u_ij + shift); gamma=0 == ECSQ_row,
        gamma=1 == pure separable-Fisher allocation; the mean log-step is identical for all gamma."""
        def f(n, Wt):
            ur, uc = logu[n]
            L = math.log(lam) + (1 - gamma) * logrms[n][:, None] + gamma * (ur[:, None] + uc[None, :] + shift)
            S = torch.exp(L).float().half().float().clamp(min=1e-7)
            side = 16 * (Wt.shape[0] + Wt.shape[1])        # fp16 row + col factors
            if use_gptq:
                if "H" not in _H:
                    restore_weights(model, snap)
                    _H["H"] = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
                q, D, _ = gptq(Wt, _H["H"][n], "grid", step=S)
                return q, D, side
            q, D = ecsq(Wt, S)
            return q, D, side
        return f

    plan = [(False, 1.0, l) for l in (0.35, 0.5, 0.7)] + [(False, 0.5, l) for l in (0.35, 0.5, 0.7)] + \
           [(True, 1.0, l) for l in (0.5, 0.7)] + [(True, 0.5, l) for l in (0.35, 0.5, 0.7)]
    for use_gptq, gamma, lam in plan:
        name = f"{'gptq_' if use_gptq else ''}fisher_g{gamma}_lam{lam}"
        if store.has(name):
            continue
        check_disk()
        acc = quantize_model(model, snap, make_fn(lam, use_gptq, gamma))
        ev = eval_lm(model, test_ids, ref=ref)
        store.put(name, {"entropy_bpw": acc["entropy_bpw"], "side_bpw": acc["side_bpw"], "gamma": gamma, "lam": lam, **ev})
        log(f"{name}: ent={acc['entropy_bpw']:.3f} ppl={ev['ppl']:.3f} kl={ev['kl']:.4f}")
    restore_weights(model, snap)
    restore_weights(model, snap)
