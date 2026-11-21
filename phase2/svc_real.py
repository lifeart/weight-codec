"""Phase 2 / SVC with REAL BYTES: one progressive stream, one WCDC file per quality layer.

Layer 0 : ECSQ, step s0 = c0*rms(row) (fp16 side info, the only side info of the stream)
Layer k : refinement on a lattice nested by `ratio`: s_k = s_{k-1}/ratio,
          q_k = round(W/s_k) - ratio*Q_{k-1},  decoder Q_k = ratio*Q_{k-1} + q_k,  W^_k = Q_k*s_k
          symbols coded in 7 context streams, ctx = clip(q_{k-1}, -3, 3)  (phase-1 context)
Single-shot reference at every layer's lattice: ECSQ q = round(W/s_k), same fp16 s0 stored once
(step mode own_div), coded as one file.  "SVC tax" = sum of layer files - single-shot file.
Because Q_k == round(W/s_k) by construction, decoded cumulative layers must be BIT-IDENTICAL to
the single-shot decode (asserted), so quality is shared; PPL/KL is measured on the decoded layers.
"""
import argparse
from p2common import *

ap = argparse.ArgumentParser()
ap.add_argument("--rid", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--tok_tag", default="qwen25")
ap.add_argument("--c0", type=float, default=1.4)
ap.add_argument("--ratios", default="3,2")
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--eval_ratios", default="3")
args = ap.parse_args()
seed_all(0); check_disk()
tok = load_tok(args.rid)
test_ids = wt2_ids(tok, "test", args.tok_tag)
model = load_model(args.rid)
snap = snapshot_weights(model)
ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.tag}.pt"))
lin = [n for n, _ in block_linears(model)]
ntot = sum(snap[n].numel() for n in lin)
OUT = os.path.join(BITS, args.tag, "svc"); os.makedirs(OUT, exist_ok=True)
store = Store(f"phase2_svc_{args.tag}")

for ratio in [int(r) for r in args.ratios.split(",")]:
    key = f"r{ratio}"
    if store.has(f"{key}_summary"):
        continue
    s0 = {n: Q.ecsq_step(snap[n], args.c0) for n in lin}
    Qc = {}; qlast = {}; step = {}
    layers = []
    for k in range(args.layers):
        wr = Writer({"svc": True, "ratio": ratio, "layer": k, "c0": args.c0})
        ws = Writer({"svc_single_shot": True, "ratio": ratio, "layer": k, "c0": args.c0})
        proxy = proxy_ss = 0.0
        for n in lin:
            W = snap[n].float()
            if k == 0:
                step[n] = s0[n]
                q = Q.ecsq(W, step[n])
                kind, p, sec = enc_qrow(q, s0[n])
                Qc[n] = q.to(torch.int16)
                proxy += order0_bits(q) + 16 * W.shape[0]
            else:
                step[n] = step[n] / float(ratio)
                Qn = Q.ecsq(W, step[n])
                q = Qn - ratio * Qc[n]
                kind, p, sec = enc_qrow(q, step_mode=f"div:{ratio}", acc=f"nest:{ratio}",
                                        ctx_prev=qlast[n].numpy())
                ctx = np.clip(qlast[n].numpy().reshape(-1), -3, 3)
                proxy += sum(order0_bits(q.numpy().reshape(-1)[ctx == v]) for v in np.unique(ctx))
                Qc[n] = Qn.to(torch.int16)
            qlast[n] = q.to(torch.int8) if k > 0 else q.clamp(-128, 127).to(torch.int8)
            wr.add(n, kind, W.shape, p, sec)
            # single-shot on exactly the same lattice
            mode = "own" if k == 0 else "own_div:" + ",".join([str(ratio)] * k)
            kind2, p2, sec2 = enc_qrow(Qc[n].to(torch.int32), s0[n], step_mode=mode)
            ws.add(n, kind2, W.shape, p2, sec2)
            proxy_ss += order0_bits(Qc[n]) + 16 * W.shape[0]
        f = os.path.join(OUT, f"{key}_L{k}.wcdc"); fs = os.path.join(OUT, f"{key}_single_L{k}.wcdc")
        nb = wr.write(f); nbs = ws.write(fs)
        layers.append({"layer": k, "file": os.path.relpath(f, ROOT), "bytes": nb, "layer_bpw": 8 * nb / ntot,
                       "layer_proxy_bpw": proxy / ntot, "single_file": os.path.relpath(fs, ROOT),
                       "single_bytes": nbs, "single_bpw": 8 * nbs / ntot, "single_proxy_bpw": proxy_ss / ntot})
        log(f"{key} L{k}: layer {8 * nb / ntot:.4f} bpw (proxy {proxy / ntot:.4f}); single-shot {8 * nbs / ntot:.4f}")
    del Qc, qlast; gc.collect()
    # decode the progressive stream layer by layer from the files; compare with single-shot
    st = None
    cum = 0
    for k, L in enumerate(layers):
        tim = {}
        st, tim, _ = decode_file(os.path.join(ROOT, L["file"]), ref=st, device="cpu")
        ss, tim2, _ = decode_file(os.path.join(ROOT, L["single_file"]), device="cpu")
        ident = all(torch.equal(st[n].W.cpu(), ss[n].W.cpu()) for n in lin)
        assert ident, f"SVC layer {k} differs from the single-shot quantizer"
        cum += L["bytes"]
        L.update({"cum_bytes": cum, "cum_bpw": 8 * cum / ntot, "cum_MB": cum / 1e6,
                  "tax_bpw": 8 * (cum - L["single_bytes"]) / ntot,
                  "tax_pct": 100 * (cum / L["single_bytes"] - 1), "bit_identical_to_single": ident,
                  "decode_layer_s": {k2: round(v, 3) for k2, v in tim.items()},
                  "decode_single_s": {k2: round(v, 3) for k2, v in tim2.items()}})
        if str(ratio) in args.eval_ratios.split(","):
            load_into(model, st)
            ev = eval_lm(model, test_ids, ref=ref)
            L.update(ev)
            restore_weights(model, snap)
        log(f"{key} cum L0..{k}: {L['cum_MB']:.2f} MB = {L['cum_bpw']:.4f} bpw vs single {L['single_bpw']:.4f} "
            f"-> tax {L['tax_bpw']:+.4f} bpw ({L['tax_pct']:+.2f}%)" + (f" ppl={L['ppl']:.3f} kl={L['kl']:.4f}" if 'ppl' in L else ""))
        st = {n: TState(Q=s.Q.to(torch.int16), q_last=s.q_last.clamp(-128, 127).to(torch.int8), step=s.step)
              for n, s in st.items()}  # drop W, narrow ints (RAM)
        del ss; gc.collect()
    store.put(f"{key}_summary", {"ratio": ratio, "c0": args.c0, "n_weights": ntot, "layers": layers})
restore_weights(model, snap)
log("done")
