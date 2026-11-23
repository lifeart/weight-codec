"""Seam check for the column-grid coders (REPORT §16 / §17): encoder -> real .wcdc container -> every decoder path ->
bit-identical weights.  Uses a real model pair (default SmolLM2-135M {base, Instruct}), all block linears.

--coders ghcol (REPORT §16):
  G_I   base I-frame            Q.ghcol(W_B, H_B, c)                   qrow + "col" section
  G_P   fine-tune P-frame       GPTQ(W_T - Q_B*s, H_T) on the I-frame's row x column lattice, col="ref"
  G_D   fine-tune delta         Q.ghcol(W_T - W_B16, H_T, c_delta), base="fp16ref" (+ own column factors)
--coders default (REPORT §17 = the family codec's default coders):
  G_I   base I-frame            A3 Fisher grid + GPTQ: Q.gptq_sep(W_B, H_B, *Q.fisher_rowcol(fisher_grid, c))
  G_P   fine-tune P-frame       GPTQ(W_T - Q_B*s, H_T) on the inherited Fisher lattice     (T_pframe_gfisher)
  G_Pr  fine-tune P-frame       plain rounding on the same inherited lattice              (T_pframe_gfisher_rnd)
  G_D   fine-tune delta         Hessian-column grid of the delta, PLAIN ROUNDING (family_codec T_dhcol)

For every file, SHA-256 over all records (file order) of (a) the decoded lattice index Q (int32 LE) and (b) the
decoded fp16 weights (uint16 LE) must equal the digests computed on the ENCODER side from its own symbols and its
own reconstruction (Q * row * col, the GPTQ output).  Decoder paths: constriction rANS on CPU and on MPS
(codecs.decode), the persistent process-pool parallel decoder (wcodec.parallel), the rans16 transcode decoded by
the numba decoder, and the Metal GPU rans16 decoder (chunk 4096).  Hessians: a few calibration windows on CPU
(only the round trip is tested here, not quality).  Writes results/ghcol_roundtrip_<tag>.json; exits non-zero
on any mismatch.

usage: python phase2/ghcol_roundtrip.py [--base ... --target ... --tag smol135 --tok_tag smollm2]
"""
import argparse
import copy
import hashlib
import shutil
import tempfile

from p2common import *



def main():
    # everything runs inside main(): wcodec.parallel uses 'spawn' workers, which re-import this module --
    # unguarded top-level code would re-run the whole test in every worker (happened once: 4 x 1.4 GB)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--target", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--tag", default="smol135")
    ap.add_argument("--tok_tag", default="smollm2")
    ap.add_argument("--c", type=float, default=0.35)
    ap.add_argument("--c_delta", type=float, default=1.6)
    ap.add_argument("--calib", type=int, default=8)
    ap.add_argument("--no_gpu", action="store_true")
    ap.add_argument("--coders", default="ghcol", choices=["ghcol", "default"])
    args = ap.parse_args()
    seed_all(0); check_disk()
    import common
    common.DEV = "cpu"                                   # Hessians on CPU (a GPU job may be running)
    tok = load_tok(args.target)
    train_ids = wt2_ids(tok, "train", args.tok_tag)
    tmp = tempfile.mkdtemp(prefix="ghcol_rt_")
    H = lambda b: hashlib.sha256(b).digest()


    def hessians(rid):
        """fp16 snapshot of the block linears (exact: the model is loaded as fp16) + CPU Hessians."""
        m = load_model(rid, dtype=torch.float16, device="cpu").float()
        snap = {k: v.half() for k, v in snapshot_weights(m).items()}
        Hs = collect_hessians(m, calib_windows(train_ids, args.calib), QWEN_SHARE)
        del m; gc.collect()
        return snap, Hs


    def bits16(t):
        return t.detach().cpu().contiguous().to(torch.float16).view(torch.int16).numpy().tobytes()


    def q32(t):
        return t.detach().cpu().to(torch.int32).numpy().astype("<i4").tobytes()


    t0 = time.time()
    # two passes, one Hessian set resident at a time (Qwen2.5-0.5B: ~2.4 GB of Hessians per model)
    snapB, HB = hessians(args.base)
    names = list(snapB)
    if args.coders == "default":             # Fisher factors of the base (cached data/fisher_<tag>.pt, = family codec)
        FG = Q.fisher_grid({n: Q.rms_row(snapB[n].float())[:, 0] for n in names},
                           fisher_stats(args.base, args.tag, train_ids), gamma=0.5)
    FN = ["G_I", "G_P"] + (["G_Pr"] if args.coders == "default" else []) + ["G_D"]
    maxabs = {f: 0 for f in FN}
    enc = {f: ([], []) for f in FN}
    W_ = {f: Writer({"t": f"{args.coders}_rt", "f": f}) for f in FN}
    negzero = [0]
    LAT = {}                                             # name -> (I-frame symbols int16, row [O,1], col [I])
    for n in names:
        WB = snapB[n].float()
        if args.coders == "ghcol":
            S, Qd, row, col = Q.ghcol(WB, HB[n], args.c)
        else:
            S, Qd, row, col = Q.gptq_sep(WB, HB[n], *Q.fisher_rowcol(FG[n], args.c))
        k, p, s = enc_qrow(S, row, col=col); W_["G_I"].add(n, k, WB.shape, p, s)
        sfull = row * col[None, :]
        # encoder-side reference = the lattice reconstruction from the encoder's own symbols (the codec contract).
        # GPTQ's in-memory output round(w/s)*s is VALUE-identical but holds -0.0 where the decoder (int symbols)
        # gives +0.0 -- checked here, counted, and not hashed (a raw-bit hash would flag only signed zeros)
        recI = S.float() * sfull
        assert torch.equal(recI.half(), Qd.half()), f"{n}: GPTQ output != symbols x lattice"
        negzero[0] += int(((Qd == 0) & torch.signbit(Qd)).sum())
        enc["G_I"][0].append(H(q32(S))); enc["G_I"][1].append(H(bits16(recI)))
        assert Q.sym_fits_int16(S), f"{n} G_I: lattice index beyond int16"
        maxabs["G_I"] = max(maxabs["G_I"], int(S.abs().max()))
        LAT[n] = (S.to(torch.int16), row, col)
    del HB; gc.collect()
    snapT, HTt = hessians(args.target)
    for n in names:
        WT = snapT[n].float()
        S16, row, col = LAT[n]
        S = S16.to(torch.int32)
        sfull = row * col[None, :]
        recI = S.float() * sfull                         # value-identical to the GPTQ output (asserted above)
        q, Qp, _, _ = Q.gptq(WT - recI, HTt[n], "grid", step=sfull)
        k, p, s = enc_qrow(q, step_mode="ref", acc="add", col_mode="ref"); W_["G_P"].add(n, k, WT.shape, p, s)
        enc["G_P"][0].append(H(q32(S + q))); enc["G_P"][1].append(H(bits16((S + q).float() * sfull)))
        syms = [("G_P", S + q)]
        if "G_Pr" in FN:                                 # rounding P-frame on the same inherited lattice
            qr = Q.ecsq(WT - recI, sfull)
            k, p, s = enc_qrow(qr, step_mode="ref", acc="add", col_mode="ref"); W_["G_Pr"].add(n, k, WT.shape, p, s)
            enc["G_Pr"][0].append(H(q32(S + qr))); enc["G_Pr"][1].append(H(bits16((S + qr).float() * sfull)))
            syms.append(("G_Pr", S + qr))
        D = WT - snapB[n].float()                        # snapshot weights are the fp16 base values
        if args.coders == "ghcol":
            Sd, Qdd, rowd, cold = Q.ghcol(D, HTt[n], args.c_delta)
        else:                                            # = family_codec T_dhcol (rounding on the delta's column grid)
            rowd, cold0 = Q.hcol_grid(D, torch.diag(HTt[n]), args.c_delta)
            cold, _ = Q.int16_safe_col(D, rowd, cold0)
            Sd = Q.ecsq(D, rowd * cold[None, :])
        for f_, sym in syms + [("G_D", Sd)]:
            maxabs[f_] = max(maxabs[f_], int(sym.abs().max()))
            assert Q.sym_fits_int16(sym), f"{n} {f_}: lattice index beyond int16"
        k, p, s = enc_qrow(Sd, rowd, base="fp16ref", col=cold); W_["G_D"].add(n, k, WT.shape, p, s)
        enc["G_D"][0].append(H(q32(Sd))); enc["G_D"][1].append(H(bits16(snapB[n].float() + Sd.float() * (rowd * cold[None, :]))))
    del HTt, snapT, LAT; gc.collect()
    files = {}
    for nm, w in W_.items():
        files[nm] = os.path.join(tmp, nm + ".wcdc")
        w.write(files[nm])
    log(f"encoded {len(files)} files ({sum(os.path.getsize(f) for f in files.values()) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    ENC = {nm: {"sym_sha256": H(b"".join(a)).hex(), "w_sha256": H(b"".join(b)).hex()} for nm, (a, b) in enc.items()}
    fp16ref = {n: TState(W=snapB[n].half()) for n in names}


    def digests(rd, st):
        return {"sym_sha256": H(b"".join(H(q32(st[r["name"]].Q)) for r in rd.records)).hex(),
                "w_sha256": H(b"".join(H(bits16(st[r["name"]].W)) for r in rd.records)).hex()}


    def to_dev(states, dev):
        return {k: TState(W=None if s.W is None else s.W.to(dev), Q=None if s.Q is None else s.Q.to(dev),
                          step=None if s.step is None else s.step.to(dev), col=None if s.col is None else s.col.to(dev))
                for k, s in states.items()}


    def chain(decode_fn, Fm, dev):
        """decode I, then every P-frame with the decoded I state as reference, then D with the fp16 base.
        Fm: {file key: path}; decode_fn(reader, ref, dev, path)."""
        out = {}
        rdI = Reader(Fm["G_I"]); stI = decode_fn(rdI, None, dev, Fm["G_I"]); out["G_I"] = digests(rdI, stI)
        for st_ in stI.values():                         # the P-frames only need Q, step, col of the reference
            st_.W = None
        for f_ in [f for f in FN if f.startswith("G_P")]:
            rdP = Reader(Fm[f_]); stP = decode_fn(rdP, stI, dev, Fm[f_]); out[f_] = digests(rdP, stP)
            del stP
        del stI
        rdD = Reader(Fm["G_D"]); stD = decode_fn(rdD, to_dev(fp16ref, dev), dev, Fm["G_D"]); out["G_D"] = digests(rdD, stD)
        return out


    res = {"coders": args.coders, "files": FN, "max_abs_lattice_index": maxabs,
           "encoder": ENC, "paths": {}, "n_tensors": len(names), "c": args.c, "c_delta": args.c_delta,
           "gptq_output_value_equal_to_symbols_x_lattice": True, "gptq_output_negative_zeros": negzero[0],
           "bytes": {k: os.path.getsize(f) for k, f in files.items()}}
    # keep_state="Q": Q, step, col (enough to be a P-frame reference and for the digests) -- no q_last copy
    plain = lambda rd, ref, dev, path: decode(rd, ref=ref, device=dev, keep_state="Q")
    res["paths"]["constriction_cpu"] = chain(plain, files, "cpu")
    from wcodec.parallel import ParallelDecoder, decode_parallel

    def par(rd, ref, dev, path):                        # one 4-worker pool at a time (each worker holds the file)
        pool = ParallelDecoder(path, 4)
        try:
            return decode_parallel(rd, pool, ref=ref, device=dev, keep_state="Q")
        finally:
            pool.close()
    res["paths"]["constriction_parallel_cpu"] = chain(par, files, "cpu")
    from wcodec.transcode import transcode
    R16 = {k: f.replace(".wcdc", ".r16.wcdc") for k, f in files.items()}
    for k in files:
        transcode(files[k], R16[k])
    res["paths"]["rans16_numba_cpu"] = chain(plain, R16, "cpu")
    if not args.no_gpu and torch.backends.mps.is_available():
        res["paths"]["constriction_mps"] = chain(plain, files, "mps")
        from wcodec.metal import decode_gpu
        G4 = {k: f.replace(".wcdc", ".r16_4k.wcdc") for k, f in files.items()}
        for k in files:
            transcode(files[k], G4[k], chunk=4096)
        res["paths"]["rans16_metal_gpu"] = chain(lambda rd, ref, dev, path: decode_gpu(rd, ref=ref, keep_state="Q"), G4, "mps")
    ok = all(res["paths"][p][f] == ENC[f] for p in res["paths"] for f in ENC)
    res["bit_exact_all_paths"] = ok
    for p in res["paths"]:
        log(f"{p:28s} " + " ".join(f"{f}:{'OK' if res['paths'][p][f] == ENC[f] else 'MISMATCH'}" for f in ENC))
    save_json(f"{'ghcol' if args.coders == 'ghcol' else 'default'}_roundtrip_{args.tag}", res)
    shutil.rmtree(tmp)
    log("BIT-EXACT on every path" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
