"""ghcol confirmation probe: GPTQ error feedback on the Hessian-COLUMN step grid, on models other than
SmolLM2-135M (where the prune lead found it: PRUNE_FINDINGS.md, S1 dense baselines).

  step_ij = fp16(c * rms_row_i) * fp16((h_j / geomean(h))^(-gamma/2)),   h = diag(H)  ("hcol" grid, V2 / S1)
  hcol_c*      plain rounding on that grid                                 (V2's best scalar coder)
  ghcol_c*     common.gptq(mode='grid') on that grid  (= prune_probe.sparsegpt at rho=1, verified bit-exact there)
  gptq_row_c*  common.gptq on the plain row grid c*rms_row                 (= phase-2 production I-frame B_ecsq_c*)
  ecsq_row_c*  plain rounding on the row grid                              (A2+ / spine)
  ghcol_g0.5_c*  gamma = 0.5 variant (A3 found gamma 0.5 best for its Fisher steps)
  gfisher_g*_c*  CONTROL (added after the first Qwen ghcol point, 3.686 bpw -> KL 0.0400, was seen to lose to the
                 REPORT §3 A3 line 3.647 -> 0.0271): GPTQ on the A3 separable FISHER grid, in the decodable
                 fp16 row x fp16 column form (same 16 b/row + 16 b/col side info as hcol); fisher_g*_c* = rounding.
                 Fisher = a3_fisher.py procedure (64 x 512 train tokens), data/fisher_<tag>.pt.

Stage `base` (VERDICT A) uses the SPINE harness exactly (REPORT §1): fp16 model on DEV (MPS), Hessians
128 x 2048 WikiText-2 train windows (seed 0, non-sequential, percdamp 0.01) from the fp16 model, GPTQ on CPU
fp32, rate = common.quantize_model (per-tensor empirical order-0 entropy + 32 b/used symbol + side info:
16 b/row, + 16 b/column for the hcol grid) -- the proxy phase 2 matched to real ANS bytes within 0.02 %.
Distortion = WikiText-2 test PPL over all 2048-token windows + KL/top-1 vs the cached fp16 reference on the
first 16 windows (data/ref_<tag>.pt, the same file the spine / phase-2 runs used).

Stage `delta` (VERDICT B) = phase-2 family-codec harness for the fp16-base scenario (T_delta_c*): Delta =
W_inst - W_base(fp16), Hessians of the INSTRUCT model, rate = Delta symbols + side info over the block
linears, distortion = chat KL/top-1/NLL on the instruct model's 48 greedy answers + WikiText-2 KL/PPL.
  tdelta_c*    phase-2 production delta coder: GPTQ on Delta, step c*rms_row(W_inst)          (= T_delta_c*)
  dhcol_c*     plain rounding of Delta on the hcol grid, rows rms_row(Delta) x diag(H_inst)^-1/2  (prune S2)
  dghcol_c*    GPTQ on the same grid (ghcol-Delta)
  dghcolW_c*   GPTQ on the hcol grid with rows rms_row(W_inst) (the T_delta row normalisation + columns)

PRE-REGISTERED (written before any Qwen / SmolLM2-360M number): VERDICT A PASS on a model iff the ghcol line
saves >= 0.05 bpw at equal WikiText KL vs the production line gptq_row at >= 2 interpolated points AND is
never a net loss (> 0.02 bpw) vs gptq_row or hcol at any interpolated point. Compared line by line (never a
mixed hull).  Against the A3 Fisher research coder (Qwen only, REPORT §3) the margin is reported, not gated.
VERDICT B PASS iff dghcol saves >= 0.05 bpw at equal chat-KL vs tdelta (the production delta coder) at >= 2
points; ghcol-Delta vs hcol-Delta (does the GPTQ feedback add anything on Delta?) is reported separately.

usage: python scripts/ghcol_probe.py --stage base --model Qwen/Qwen2.5-0.5B --tag qwen05b --tok_tag qwen25
       python scripts/ghcol_probe.py --stage delta --model <base> --target <instruct> --tag ... --ref_target ...
"""
import argparse
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="base", choices=["base", "delta"])
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--target", default="", help="instruct model (stage delta)")
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--tok_tag", default="qwen25")
ap.add_argument("--ref_target", default="", help="ref/chat cache tag of the instruct model (stage delta)")
ap.add_argument("--plan", default="", help="comma list of config names (default: stage plan)")
ap.add_argument("--max_windows", type=int, default=0, help="0 = all windows (spine protocol)")
ap.add_argument("--gptq_dev", default="cpu")
args = ap.parse_args()
seed_all(0)
log(f"ghcol probe stage={args.stage} model={args.model} DEV={DEV}; disk free {check_disk():.2f} GB")


# ---------------------------------------------------------------------------------------------------
# grids (identical arithmetic to scripts/prune_probe.py hcol_factor / hcol_step / row_step)
# ---------------------------------------------------------------------------------------------------
def rms_row(W):
    return W.float().pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)


def hcol_factor(H, gamma=1.0):
    hd = torch.diag(H).float().clamp(min=1e-12)
    hd = hd / torch.exp(torch.log(hd).mean())
    f = hd.pow(-gamma / 2).half().float()
    if not torch.isfinite(f).all():
        raise RuntimeError(f"hcol column factor overflows fp16 ({int((~torch.isfinite(f)).sum())} columns)")
    return f


def hcol_step(W, H, c, gamma=1.0, rowref=None):
    rr = rms_row(W if rowref is None else rowref)
    return (c * rr).half().float() * hcol_factor(H, gamma)[None, :], 16 * (W.shape[0] + W.shape[1])


def row_step(W, c, rowref=None):
    rr = rms_row(W if rowref is None else rowref)
    return (c * rr).half().float(), 16 * W.shape[0]


def parse(name):
    """config name -> (kind, gamma, c)."""
    kind, rest = name.rsplit("_c", 1)
    gamma = 1.0
    if "_g" in kind:
        kind, g = kind.split("_g")
        gamma = float(g)
    return kind, gamma, float(rest)


def fisher_stats(model_id, tag, train_ids, nseq=64, seqlen=512):
    """Empirical Fisher diagonal, row/col means -- the exact procedure and file format of scripts/a3_fisher.py
    (64 x 512 WikiText-2 train tokens, generator seed 1, fp32 model on DEV).  Cached as data/fisher_<tag>.pt."""
    fpath = os.path.join(DATA, f"fisher_{tag}.pt")
    if os.path.exists(fpath):
        return torch.load(fpath)
    m = load_model(model_id, dtype=torch.float32)
    lin = block_linears(m)
    for p in m.parameters():
        p.requires_grad_(False)
    for n, mm in lin:
        mm.weight.requires_grad_(True)
    F = {n: torch.zeros_like(mm.weight) for n, mm in lin}
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for i in range(nseq):
        s = int(torch.randint(0, train_ids.numel() - seqlen - 1, (1,), generator=g))
        x = train_ids[s:s + seqlen].long().unsqueeze(0).to(DEV)
        m(input_ids=x, labels=x).loss.backward()
        for n, mm in lin:
            F[n].add_(mm.weight.grad.pow(2)); mm.weight.grad = None
    log(f"fisher {nseq}x{seqlen} in {time.time() - t0:.0f}s")
    stats = {}
    for n, mm in lin:
        f = (F[n] / nseq).cpu().double()
        stats[n] = {"row": f.mean(1).float(), "col": f.mean(0).float(), "mean": float(f.mean()),
                    "logmean": float(torch.log(f.clamp(min=1e-30)).mean())}
    torch.save(stats, fpath)
    del m, F
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return stats


def fisher_grid(snap, stats):
    """A3 separable Fisher allocation (REPORT §3, scripts/a3_fisher.py make_fn) in a DECODABLE form:
    log s_ij = log lam + (1-g) log rms_i + g (ur_i + uc_j + shift) is split into an fp16 row factor
    exp(log lam + (1-g) log rms_i + g (ur_i + shift)) and an fp16 column factor exp(g uc_j); step = row * col
    (a3_fisher.py rounded exp(L) to fp16 per element instead -- same allocation, not decodable from 16 b/row+col).
    shift is GLOBAL (cross-matrix allocation), set so the weight-averaged log step equals that of ECSQ_row."""
    rms = {n: rms_row(snap[n].float())[:, 0].double() for n in snap}
    ntot = sum(snap[n].numel() for n in snap)
    L0 = sum(float(torch.log(rms[n]).sum()) * snap[n].shape[1] for n in snap) / ntot
    logu, acc_log = {}, 0.0
    for n in snap:
        O, I = snap[n].shape
        fr = stats[n]["row"].double().clamp(min=1e-30); fc = stats[n]["col"].double().clamp(min=1e-30)
        ur = -0.5 * (torch.log(fr) - math.log(max(stats[n]["mean"], 1e-30)))
        uc = -0.5 * torch.log(fc)
        logu[n] = (ur, uc)
        acc_log += float(ur.sum()) * I + float(uc.sum()) * O
    shift = L0 - acc_log / ntot
    return {"rms": rms, "logu": logu, "shift": shift}


def fisher_step(n, c, gamma, FG):
    ur, uc = FG["logu"][n]
    lr = math.log(c) + (1 - gamma) * torch.log(FG["rms"][n]) + gamma * (ur + FG["shift"])
    row = torch.exp(lr).float().half().float().clamp(min=2.0 ** -24)[:, None]
    col = torch.exp(gamma * uc).float().clamp(max=65504.0).half().float().clamp(min=2.0 ** -24)
    return row * col[None, :], 16 * (row.shape[0] + col.shape[0])


def make_fn(name, H, WB=None, colstats=None, FG=None):
    kind, gamma, c = parse(name)

    def fn(n, W):
        X = W if WB is None else W - WB[n].float()
        if kind in ("fisher", "gfisher"):
            st, side = fisher_step(n, c, gamma, FG)
        elif kind in ("ecsq_row", "gptq_row", "dscal", "tdelta"):
            st, side = row_step(X, c, rowref=W if kind == "tdelta" else None)
        else:
            st, side = hcol_step(X, H[n], c, gamma, rowref=W if kind == "dghcolW" else None)
            if colstats is not None:
                f = hcol_factor(H[n], gamma)
                colstats["min"] = min(colstats.get("min", 9e9), float(f.min()))
                colstats["max"] = max(colstats.get("max", 0.0), float(f.max()))
        if kind in ("ecsq_row", "hcol", "dhcol", "dscal", "fisher"):
            q = torch.round(X / st)
            S, D = q.to(torch.int32), q * st
        else:
            S, D, _ = gptq(X, H[n], "grid", step=st, dev=args.gptq_dev)
        return S, (D if WB is None else WB[n].float() + D), side
    return fn


# ---------------------------------------------------------------------------------------------------
tok = load_tok(args.target or args.model)
test_ids = wt2_ids(tok, "test", args.tok_tag)
train_ids = wt2_ids(tok, "train", args.tok_tag)

if args.stage == "base":
    PLAN = args.plan.split(",") if args.plan else (
        ["gptq_row_c0.35"] + [f"ghcol_c{c}" for c in (0.35, 0.5, 0.7, 0.25, 1.0)] +
        [f"hcol_c{c}" for c in (0.35, 0.5, 0.7)] + [f"ghcol_g0.5_c{c}" for c in (0.35, 0.5)])
    model = load_model(args.model)
    snap = snapshot_weights(model)
    ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
    store = Store(f"ghcol_base_{args.tag}")
    store.put("_meta", {"model": args.model, "dev_eval": DEV, "gptq_dev": args.gptq_dev, "harness": "spine (REPORT §1)",
                        "eval": "WikiText-2 test PPL over all 2048 windows; KL/top-1 vs data/ref_%s.pt on the first 16" % args.tag,
                        "hessian": "128 x 2048 WikiText-2 train windows, seed 0, fp16 model, non-sequential, percdamp 0.01",
                        "rate": "common.quantize_model: per-tensor order-0 entropy + 32 b/used symbol + side (16 b/row [+16 b/col])",
                        "n_quantized_set": sum(v.numel() for v in snap.values())})
    todo = [p for p in PLAN if not store.has(p)]
    FG = None
    if any(parse(p)[0] in ("fisher", "gfisher") for p in todo):
        FG = fisher_grid(snap, fisher_stats(args.model, args.tag, train_ids))
    if todo:
        H = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
        restore_weights(model, snap)
        for name in todo:
            check_disk()
            cs = {}
            t0 = time.time()
            acc = quantize_model(model, snap, make_fn(name, H, colstats=cs, FG=FG))
            qs = round(time.time() - t0, 1)
            ev = eval_lm(model, test_ids, max_windows=args.max_windows or None, ref=ref)
            kind, gamma, c = parse(name)
            rec = {"cfg": {"kind": kind, "gamma": gamma, "c": c}, "entropy_bpw": acc["entropy_bpw"],
                   "side_bpw": acc["side_bpw"], "sym_entropy_bpw": acc["sym_entropy_bpw"], "quant_s": qs,
                   "colfactor_range": [cs.get("min"), cs.get("max")], **ev}
            store.put(name, rec)
            log(f"{name:18s} bpw={acc['entropy_bpw']:.4f} (side {acc['side_bpw']:.4f}) ppl={ev['ppl']:.4f} "
                f"kl={ev['kl']:.4f} top1={ev['top1']:.4f} q={qs}s | disk {disk_free_gb():.1f} GB")
            restore_weights(model, snap)
        del H
    log("base stage done")

else:
    assert args.target and args.ref_target
    PLAN = args.plan.split(",") if args.plan else (
        [f"tdelta_c{c}" for c in (0.12,)] + [f"dghcol_c{c}" for c in (0.8, 1.6, 3.2)] +
        [f"dhcol_c{c}" for c in (0.8, 1.6, 3.2)])
    model = load_model(args.target)
    snap = snapshot_weights(model)
    from huggingface_hub import snapshot_download
    snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json"])
    Bsd = LazySD(args.model)
    pfx = model.base_model_prefix
    WB = {n: Bsd[f"{pfx}.{n}.weight"].to(torch.float16) for n in snap}
    del Bsd
    ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.ref_target}.pt"))
    chat = json.load(open(os.path.join(RES, f"chat_set_{args.ref_target}.json")))
    chat_ref = torch.load(os.path.join(DATA, f"chatref_{args.ref_target}.pt"))
    store = Store(f"ghcol_delta_{args.tag}")
    store.put("_meta", {"base": args.model, "target": args.target, "dev_eval": DEV, "gptq_dev": args.gptq_dev,
                        "harness": "phase-2 family codec, fp16-base scenario (T_delta)",
                        "eval": "WikiText-2 PPL all windows + KL first 16 vs data/ref_%s.pt; chat KL/top-1/NLL on "
                                "results/chat_set_%s.json" % (args.ref_target, args.ref_target),
                        "hessian": "INSTRUCT model, 128 x 2048 WikiText-2 train windows, seed 0",
                        "rate": "Delta symbols: per-tensor order-0 entropy + 32 b/used symbol + side (16 b/row [+16 b/col])"})
    todo = [p for p in PLAN if not store.has(p)]
    if todo:
        H = collect_hessians(model, calib_windows(train_ids, 128), QWEN_SHARE)
        restore_weights(model, snap)
        for name in todo:
            check_disk()
            t0 = time.time()
            acc = quantize_model(model, snap, make_fn(name, H, WB=WB))
            qs = round(time.time() - t0, 1)
            ev = eval_lm(model, test_ids, max_windows=args.max_windows or None, ref=ref)
            ev.update(eval_chat(model, chat, chat_ref, ref["head"]))
            kind, gamma, c = parse(name)
            rec = {"cfg": {"kind": kind, "gamma": gamma, "c": c}, "entropy_bpw": acc["entropy_bpw"],
                   "side_bpw": acc["side_bpw"], "quant_s": qs, **ev}
            store.put(name, rec)
            log(f"{name:18s} bpw={acc['entropy_bpw']:.4f} (side {acc['side_bpw']:.4f}) chat_kl={ev['chat_kl']:.4f} "
                f"chat_top1={ev['chat_top1']:.4f} kl={ev['kl']:.4f} ppl={ev['ppl']:.4f} q={qs}s | disk {disk_free_gb():.1f} GB")
            restore_weights(model, snap)
        del H
    log("delta stage done")
