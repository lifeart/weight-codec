"""Phase 2 / training-trajectory codec with REAL BYTES (Pythia checkpoints as video frames).

Checkpoints are streamed from the Hub straight into RAM (never written to disk), one at a time.
Per frame t, real WCDC files are written for:
  allI     every frame coded independently: ECSQ, own step c*rms(row of W_t)
  IPPP     frame 0 = I-frame (own step s), frames 1.. = closed-loop P-frames on the FIXED lattice s:
           q_t = round(W_t/s) - Q_{t-1};  decoder Q_t = Q_{t-1} + q_t  (integer state -> no drift)
  K4       I-frame (own step) every 4 frames, P-frames on the latest I-frame lattice in between
  lossless bit-exact archive of the linears: frame 0 = ll16, frame t = ll16(bits_t XOR bits_{t-1})
  rest     embeddings / norms / biases, lossless: frame 0 ll16, then XOR-to-previous ll16
After the window: the IPPP chain is DECODED FROM THE FILES (I + all P) and the last frame's PPL is
compared with the original and with the last frame's own I-frame; decode time of the chain is
measured; sha256 of the decoded chain is stored for a cross-machine decode check.
"""
import argparse
import hashlib
import io
from p2common import *

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="EleutherAI/pythia-410m")
ap.add_argument("--tag", default="pythia410m")
ap.add_argument("--windows", default="late")
ap.add_argument("--cs", default="0.35,0.7")
ap.add_argument("--ppl_windows", type=int, default=40)
ap.add_argument("--dtype", default="fp16")
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--no_ppl", action="store_true", help="skip PPL evals (done on the decoding machine instead)")
args = ap.parse_args()
seed_all(0)
DT = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
CS = [float(c) for c in args.cs.split(",")]
WINDOWS = {"late": list(range(136000, 143001, 1000)), "mid": list(range(32000, 37001, 1000)),
           "tiny": [142000, 143000]}
store = Store(f"phase2_trajectory_{args.tag}")


def fetch(step):
    from huggingface_hub import hf_hub_url, get_session
    check_disk(3.0)
    url = hf_hub_url(args.repo, "pytorch_model.bin", revision=f"step{step}")
    for attempt in range(4):
        try:
            r = get_session().get(url, timeout=300, follow_redirects=True)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            break
        except Exception as e:  # network hiccup: log, retry, then re-raise
            log(f"fetch step{step} attempt {attempt} failed: {e!r}")
            if attempt == 3:
                raise
            time.sleep(15)
    del r
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    del buf
    return {k: v.to(torch.float16) for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def is_lin(k, v):
    return k.startswith("gpt_neox.layers.") and k.endswith(".weight") and v.dim() == 2


def build_model(sd):
    from transformers import AutoConfig, GPTNeoXForCausalLM
    cfg = AutoConfig.from_pretrained(args.repo, revision="step143000")
    m = GPTNeoXForCausalLM(cfg).to(DT)
    missing, _ = m.load_state_dict({k: v.to(DT) for k, v in sd.items()}, strict=False)
    real_missing = [k for k in missing if not k.endswith(("rotary_emb.inv_freq", "attention.bias", "attention.masked_bias"))]
    assert not real_missing, real_missing[:5]
    m.eval(); m.config.use_cache = False
    return m.to(DEV)


tok = load_tok(args.repo)
test_ids = wt2_ids(tok, "test", "pythia")


def ppl_of(sd):
    if args.no_ppl:
        return {"ppl": None, "skipped": True}
    m = build_model(sd)
    r = eval_lm(m, test_ids, max_windows=args.ppl_windows)
    del m; gc.collect()
    return r


def sha_states(states, keys):
    h = hashlib.sha256()
    for k in keys:
        h.update(states[k].W.detach().cpu().contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


for wname in args.windows.split(","):
    steps = WINDOWS[wname]
    if args.max_frames:
        steps = steps[:args.max_frames]
    if store.has(f"{wname}_summary"):
        continue
    OUT = os.path.join(BITS, args.tag, wname); os.makedirs(OUT, exist_ok=True)
    F = lambda nm: os.path.join(OUT, nm + ".wcdc")
    frames = []
    stateP = {c: {} for c in CS}       # Q_{t-1} (int32) for IPPP
    stateK = {c: {} for c in CS}       # (Q, step) for K4
    stepI = {c: {} for c in CS}
    prev = None
    keys = rest_keys = None
    for fi, st in enumerate(steps):
        t0 = time.time()
        sd = fetch(st)
        if keys is None:
            keys = [k for k, v in sd.items() if is_lin(k, v)]
            rest_keys = [k for k in sd if k not in set(keys)]
        ntot = sum(sd[k].numel() for k in keys)
        nrest = sum(sd[k].numel() for k in rest_keys)
        fr = {"step": st, "n_linear": ntot, "n_rest": nrest}
        # ---- lossless (linears and rest) ----
        for part, ks in (("ll_lin", keys), ("ll_rest", rest_keys)):
            wr = Writer({"frame": st, "what": part, "xor_prev": fi > 0})
            for k in ks:
                kind, p, s = enc_ll16(sd[k], None if prev is None else prev[k])
                wr.add(k, kind, sd[k].shape, p, s)
            nb = wr.write(F(f"{part}_f{fi}"))
            fr[f"{part}_bytes"] = nb
            fr[f"{part}_bpp"] = 8 * nb / (ntot if part == "ll_lin" else nrest)
            if fi == 1:   # verify the lossless XOR P-frame round trip once per window
                dst, _, _ = decode_file(F(f"{part}_f{fi}"), ref={k: TState(W=prev[k]) for k in ks})
                assert all(torch.equal(dst[k].W, sd[k]) for k in ks), f"lossless {part} P-frame not exact"
                fr[f"{part}_roundtrip_verified"] = True
                del dst
            os.remove(F(f"{part}_f{fi}"))   # sizes recorded; lossless archive not kept on disk
        # ---- lossy ----
        for c in CS:
            wI = Writer({"frame": st, "scheme": "allI", "c": c})
            wP = Writer({"frame": st, "scheme": "IPPP", "c": c, "ref_frame": None if fi == 0 else steps[fi - 1]})
            wK = Writer({"frame": st, "scheme": "K4", "c": c})
            prox = {"allI": 0.0, "IPPP": 0.0, "K4": 0.0}
            for k in keys:
                W = sd[k].float()
                s_own = Q.ecsq_step(W, c)
                qI = Q.ecsq(W, s_own)
                kind, p, sec = enc_qrow(qI, s_own); wI.add(k, kind, W.shape, p, sec)
                prox["allI"] += order0_bits(qI) + 16 * W.shape[0]
                if fi == 0:
                    stepI[c][k] = s_own
                    wP.add(k, kind, W.shape, p, sec); prox["IPPP"] += order0_bits(qI) + 16 * W.shape[0]
                    stateP[c][k] = qI
                else:
                    Qt = Q.ecsq(W, stepI[c][k])
                    q = Qt - stateP[c][k]
                    kind2, p2, sec2 = enc_qrow(q, step_mode="ref", acc="add")
                    wP.add(k, kind2, W.shape, p2, sec2); prox["IPPP"] += order0_bits(q)
                    stateP[c][k] = Qt
                if fi % 4 == 0:
                    wK.add(k, kind, W.shape, p, sec); prox["K4"] += order0_bits(qI) + 16 * W.shape[0]
                    stateK[c][k] = (qI, s_own)
                else:
                    Qp, sK = stateK[c][k]
                    Qt = Q.ecsq(W, sK)
                    q = Qt - Qp
                    kind3, p3, sec3 = enc_qrow(q, step_mode="ref", acc="add")
                    wK.add(k, kind3, W.shape, p3, sec3); prox["K4"] += order0_bits(q)
                    stateK[c][k] = (Qt, sK)
            for sch, w in (("allI", wI), ("IPPP", wP), ("K4", wK)):
                path = F(f"{sch}_c{c}_f{fi}")
                nb = w.write(path)
                fr[f"{sch}_c{c}_bytes"] = nb
                fr[f"{sch}_c{c}_bpw"] = 8 * nb / ntot
                fr[f"{sch}_c{c}_proxy_bpw"] = prox[sch] / ntot
                keep = (sch == "IPPP") or (sch == "allI" and fi == len(steps) - 1)
                if not keep:
                    os.remove(path)
        frames.append(fr)
        log(f"{wname} step{st} ({time.time() - t0:.0f}s): lossless lin {fr['ll_lin_bpp']:.3f} bpp | " +
            " ".join(f"c{c}: I={fr[f'allI_c{c}_bpw']:.3f} P={fr[f'IPPP_c{c}_bpw']:.3f} K4={fr[f'K4_c{c}_bpw']:.3f}" for c in CS))
        prev = {k: v for k, v in sd.items()}
        last_sd = sd
        store.put(f"{wname}_frames_partial", frames)
        job_sync(store.path)
        gc.collect()
    del stateP, stateK; gc.collect()
    # ---- decode the IPPP chain from the files, time it, check drift ----
    ev = {"orig": ppl_of(last_sd)}
    chain = {}
    for c in CS:
        stt = None
        tim_tot = {"entropy": 0.0, "reconstruct": 0.0}
        t0 = time.perf_counter()
        for fi in range(len(steps)):
            stt, tim, _ = decode_file(F(f"IPPP_c{c}_f{fi}"), ref=stt, device="cpu")
            for k2 in ("entropy", "reconstruct"):
                tim_tot[k2] += tim[k2]
        t_chain = time.perf_counter() - t0
        sha = sha_states(stt, keys)
        # last frame as its own I-frame (decoded from file)
        sI, timI, _ = decode_file(F(f"allI_c{c}_f{len(steps) - 1}"), device="cpu")
        sdP = dict(last_sd); sdI = dict(last_sd)
        for k in keys:
            sdP[k] = stt[k].W.cpu(); sdI[k] = sI[k].W.cpu()
        # integer state => P-chain end must equal ECSQ of the last frame on frame-0's lattice
        exact = all(torch.equal(stt[k].W.cpu(), (Q.ecsq(last_sd[k].float(), stepI[c][k]).float() * stepI[c][k]).half())
                    for k in keys)
        # same chain re-encoded with wcodec's own numba rANS (rans16): bytes + parallel decode time
        from wcodec.transcode import transcode
        r16 = [F(f"IPPP_c{c}_f{fi}") + ".r16" for fi in range(len(steps))]
        r16_bytes = [transcode(F(f"IPPP_c{c}_f{fi}"), r16[fi]) for fi in range(len(steps))]
        s16 = None
        decode_file(r16[0], device="cpu")                    # numba JIT warm-up
        t0 = time.perf_counter()
        for f in r16:
            s16, _, _ = decode_file(f, ref=s16, device="cpu")
        t_r16 = time.perf_counter() - t0
        assert sha_states(s16, keys) == sha, "rans16 chain decodes to different weights"
        for f in r16:
            os.remove(f)
        del s16
        ev[f"I_c{c}"] = ppl_of(sdI); ev[f"IPPP_end_c{c}"] = ppl_of(sdP)
        chain[str(c)] = {"decode_chain_s": round(t_chain, 3), "decode_breakdown_s": {k: round(v, 3) for k, v in tim_tot.items()},
                         "decode_lastI_s": {k: round(v, 3) for k, v in timI.items()}, "sha256_chain_end": sha,
                         "chain_end_equals_lattice_quantizer": exact,
                         "rans16_chain_bytes": r16_bytes, "rans16_decode_chain_s": round(t_r16, 3),
                         "constriction_chain_bytes": [os.path.getsize(F(f"IPPP_c{c}_f{fi}")) for fi in range(len(steps))],
                         "chain_files": [os.path.relpath(F(f"IPPP_c{c}_f{fi}"), ROOT) for fi in range(len(steps))]}
        log(wname, c, {k: (None if v["ppl"] is None else round(v["ppl"], 3)) for k, v in ev.items()},
            chain[str(c)]["decode_chain_s"], "s chain decode")
        del stt, sI, sdP, sdI; gc.collect()
    summ = {"steps": steps, "frames": frames, "ppl_last_frame": ev, "chain": chain, "device": DEV,
            "hardware": os.environ.get("WC_HARDWARE", "local")}
    for c in CS:
        for sch in ("allI", "IPPP", "K4"):
            summ[f"gop_total_bpw_{sch}_c{c}"] = sum(f[f"{sch}_c{c}_bpw"] for f in frames)
            summ[f"gop_total_MB_{sch}_c{c}"] = sum(f[f"{sch}_c{c}_bytes"] for f in frames) / 1e6
        summ[f"mean_P_bpw_c{c}"] = float(np.mean([f[f"IPPP_c{c}_bpw"] for f in frames[1:]])) if len(frames) > 1 else None
        summ[f"mean_I_bpw_c{c}"] = float(np.mean([f[f"allI_c{c}_bpw"] for f in frames]))
    summ["lossless_lin_I_bpp"] = frames[0]["ll_lin_bpp"]
    summ["lossless_lin_P_bpp_mean"] = float(np.mean([f["ll_lin_bpp"] for f in frames[1:]])) if len(frames) > 1 else None
    summ["lossless_rest_P_bpp_mean"] = float(np.mean([f["ll_rest_bpp"] for f in frames[1:]])) if len(frames) > 1 else None
    store.put(f"{wname}_summary", summ)
    job_sync(store.path)
    log(wname, json.dumps({k: v for k, v in summ.items() if k.startswith(("gop", "mean", "lossless"))}, indent=1))
    del prev, last_sd; gc.collect()
log("done")
