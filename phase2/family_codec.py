"""Phase 2 / family codec with REAL BYTES: base I-frame + fine-tune P-frame / delta, vs independent.

Every config writes real .wcdc files (bitstreams/<tag>/...), and every quality number is measured
on a model DECODED FROM THAT FILE (never from the encoder's in-memory reconstruction).

Linears (all nn.Linear in the blocks = phase-1 quantized set) and the rest (embeddings, norms,
biases) are separate files, so sizes compose: model bytes = linears file + rest file.

BASE stage (model = base):
  B_fp16              raw fp16 linears (sanity: = 2 bytes / weight + header)
  B_gptq4_packed      GPTQ int4 g128, fixed-rate nibbles + fp16 scale + 4-bit zero  (standard int4 file)
  B_gptq4_ans         same symbols, rANS-coded (A2)
  B_ecsq_c{c}         GPTQ-ECSQ, per-row step c*rms(row), rANS  (the family I-frame)
  rest_raw16 / rest_ll16   rest tensors raw / lossless rANS
TARGET stage (model = fine-tune):
  T_fp16, T_gptq4_packed, T_gptq4_ans, T_ecsq_c{c}       independent coding of the fine-tune
  T_pframe_c{c}       closed-loop P-frame on the DECODED base I-frame lattice (B_ecsq_c{c}):
                      q = GPTQ(W_T - Q_B*s, H_T) on step s (inherited, 0 side info); decoder Q = Q_B + q
  T_delta_c{c}        fp16 base available ("hub keeps fp16 base"): W = W_B16 + s*q, q = GPTQ(W_T - W_B16)
                      on s = c*rms(row of W_T)  (full-rank, entropy-coded delta)
  T_lowrank_r{r}      W = W_B16 + A@B, fp16 factors (the LoRA-style delta)
  rest_raw16 / rest_ll16 / rest_xor_base   (lossless; xor_base = lossless P-frame vs base rest)
Distortion for the fine-tune = KL / top-1 / NLL on the fp16 fine-tune's own greedy chat answers
(48 prompts) + WikiText-2 KL (16 windows) + PPL; for the base = WikiText-2 PPL + KL vs fp16 base.
"""
import argparse
from p2common import *

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--tok_tag", default="qwen25", help="cache tag of tokenized WikiText")
ap.add_argument("--ref_base", default="qwen05b")
ap.add_argument("--ref_target", default="qwen05b_instruct")
ap.add_argument("--cs", default="0.35,0.5", help="I-frame / P-frame / independent ECSQ steps")
ap.add_argument("--delta_cs", default="0.06,0.12,0.25")
ap.add_argument("--ranks", default="32,64,128")
ap.add_argument("--stages", default="base,target")
ap.add_argument("--max_windows", type=int, default=0)
ap.add_argument("--calib", type=int, default=128)
ap.add_argument("--refcoders", default="B_gptq4_ans,B_ecsq_c0.35,T_gptq4_ans,T_ecsq_c0.35,T_pframe_c0.35,T_delta_c0.12")
ap.add_argument("--dtype", default="fp16")
ap.add_argument("--gptq_dev", default="cpu")
ap.add_argument("--skip", default="", help="config names not to run (split work across jobs)")
ap.add_argument("--group", type=int, default=128, help="GPTQ-int4 group size (SmolLM2: 64, dims not /128)")
args = ap.parse_args()
seed_all(0); check_disk()
DT = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
CS = [float(c) for c in args.cs.split(",") if c]
DCS = [float(c) for c in args.delta_cs.split(",") if c]
RANKS = [int(r) for r in args.ranks.split(",") if r]
REFC = set(args.refcoders.split(","))
OUT = os.path.join(BITS, args.tag); os.makedirs(OUT, exist_ok=True)
store = Store(f"phase2_family_{args.tag}")
fpath = lambda name: os.path.join(OUT, name + ".wcdc")
tok = load_tok(args.target)
test_ids = wt2_ids(tok, "test", args.tok_tag); train_ids = wt2_ids(tok, "train", args.tok_tag)
GD = args.gptq_dev


def load(rid):
    m = load_model(rid, dtype=DT)
    if DT != torch.float16:          # CPU runs compute in fp32 but on exactly the fp16 values the files store
        with torch.no_grad():
            for p in m.parameters():
                p.data = p.data.half().to(DT)
    return m


def evaluate(model, name, ref, chat=None, chat_ref=None, states=None):
    if states is not None:
        load_into(model, states)
    ev = eval_lm(model, test_ids, max_windows=args.max_windows or None, ref=ref)
    if chat is not None:
        ev.update(eval_chat(model, chat, chat_ref, ref["head"]))
    msg = " ".join(f"{k}={ev[k]:.4f}" for k in ("ppl", "kl", "top1", "chat_kl", "chat_top1") if k in ev)
    log(f"  eval {name}: {msg}")
    return ev


def run_linears(name, model, snap, build, ref, chat=None, chat_ref=None, R_of=None, also=None):
    """build(n, W) -> (kind, params, sections, sym, side_bits); writes the file, decodes it, evals.
    also: optional second writer spec (name2, build2) sharing the same quantization pass."""
    if store.has(name) and (also is None or store.has(also[0])):
        return
    if name in set(args.skip.split(",")):
        log("skip", name); return
    check_disk()
    t0 = time.time()
    meta = {"tag": args.tag, "config": name, "group": args.group}
    wr = Writer(meta); acc = Acc(refcoders=name in REFC)
    wr2 = acc2 = None
    if also:
        wr2 = Writer({"tag": args.tag, "config": also[0]}); acc2 = Acc(refcoders=also[0] in REFC)
    for n, _ in block_linears(model):
        W = snap[n].float()
        outs = build(n, W)
        kind, p, s, sym, side = outs[0]
        wr.add(n, kind, W.shape, p, s); acc.add(sym, side, W.numel())
        if also:
            kind, p, s, sym, side = outs[1]
            wr2.add(n, kind, W.shape, p, s); acc2.add(sym, side, W.numel())
    t_enc = time.time() - t0
    recs = [(name, wr, acc)] + ([(also[0], wr2, acc2)] if also else [])
    del outs
    decoded = []
    for i, (nm, w, a) in enumerate(recs):
        r = finish_file(w, fpath(nm), a, {"encode_s": round(t_enc, 1)})
        # only the weights are needed for evaluation: no Q / q_last state (saves ~3 GB on Qwen)
        st, tim, _ = decode_file(fpath(nm), ref=R_of, device=DEV, keep_state=False)
        r["decode_s_first"] = {k: round(v, 3) for k, v in tim.items()}
        if i == 0:
            load_into(model, st)        # decoded weights go straight into the live model
        else:                           # identical reconstructions (packed vs ans): cross-check, eval once
            lin = dict(block_linears(model))
            same = all(torch.equal(lin[n].weight.data, st[n].W.to(lin[n].weight.device, lin[n].weight.dtype)) for n in st)
            assert same, "packed and ANS files decode to different weights!"
            r["identical_to"] = recs[0][0]
        del st, w; gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
        decoded.append((nm, r))
    del recs, wr, wr2
    ev = evaluate(model, name, ref, chat, chat_ref, None)
    for nm, r in decoded:
        r.update(ev)
        if name.endswith("_fp16"):          # raw fp16 size is deterministic: keep the number, free the disk
            os.remove(fpath(nm)); r["file_deleted"] = True
        store.put(nm, r)
        job_sync(store.path)
        log(f"{nm:20s} {r['MB']:8.2f} MB real_bpw={r['real_bpw']:.4f} proxy={r['proxy_bpw']:.4f} "
            f"(real-proxy {100 * (r['real_bpw'] / max(r['proxy_bpw'], 1e-9) - 1):+.3f}%) enc {t_enc:.0f}s")
    restore_weights(model, snap)


def side_row(W):
    return 16 * W.shape[0]


def b_gptq4(H):
    def f(n, W):
        G = args.group
        S, Qw, sc, z = Q.gptq(W, H[n], "rtn", 4, G, blocksize=min(128, G), dev=GD)
        side = sc.numel() * 16 + z.numel() * 4
        k1, p1, s1 = enc_g128(S, sc, z, 4, G, packed=True)
        k2, p2, s2 = enc_g128(S, sc, z, 4, G, packed=False)
        # packed file: "proxy" = the exact fixed-rate formula (4 bits/weight + side info)
        return [(k1, p1, s1, None, side + 4 * W.numel()), (k2, p2, s2, S, side)]
    return f


def b_ecsq(H, c):
    def f(n, W):
        s = Q.ecsq_step(W, c)
        S, _, _, _ = Q.gptq(W, H[n], "grid", step=s, dev=GD)
        k, p, sec = enc_qrow(S, s)
        return [(k, p, sec, S, side_row(W))]
    return f


def b_fp16(n, W):
    k, p, s = enc_raw16(W)
    return [(k, p, s, None, 16 * W.numel())]


# ======================================================================================
if "base" in args.stages:
    log("== BASE stage:", args.base)
    model = load(args.base)
    snap = snapshot_weights(model)
    rest = rest_state(model)
    for mode in ("raw16", "ll16"):
        nm = f"B_rest_{mode}"
        if not store.has(nm):
            r = write_rest(rest, fpath(nm), mode)
            st, tim, _ = decode_file(fpath(nm))
            assert all(torch.equal(st[k].W, rest[k]) for k in rest), "rest not lossless"
            r["decode_s_first"] = {k: round(v, 3) for k, v in tim.items()}; r["lossless_verified"] = True
            if mode == "raw16":
                os.remove(fpath(nm)); r["file_deleted"] = True
            store.put(nm, r); log(f"{nm}: {r['MB']:.2f} MB ({r['bpp']:.3f} bit/param)")
    ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.ref_base}.pt"))
    if not store.has("B_fp16"):
        run_linears("B_fp16", model, snap, b_fp16, ref)
    need = [c for c in CS if not store.has(f"B_ecsq_c{c}")] or (not store.has("B_gptq4_ans"))
    if need:
        restore_weights(model, snap)
        HB = collect_hessians(model, calib_windows(train_ids, args.calib), QWEN_SHARE)
        run_linears("B_gptq4_packed", model, snap, b_gptq4(HB), ref, also=("B_gptq4_ans", None))
        for c in CS:
            run_linears(f"B_ecsq_c{c}", model, snap, b_ecsq(HB, c), ref)
        del HB
    del model, snap; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()

# ======================================================================================
if "target" in args.stages:
    log("== TARGET stage:", args.target, "given base", args.base)
    model = load(args.target)
    snap = snapshot_weights(model)
    rest = rest_state(model)
    from huggingface_hub import snapshot_download
    snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])   # no-op if cached
    Bsd = LazySD(args.base)
    pfx = model.base_model_prefix
    rest_base = {k: Bsd[k].to(torch.float16) for k in rest}
    for mode, refd in (("raw16", None), ("ll16", None), ("xor_base", rest_base)):
        nm = f"T_rest_{mode}"
        if not store.has(nm):
            r = write_rest(rest, fpath(nm), "ll16" if mode == "xor_base" else mode, ref=refd)
            st, tim, _ = decode_file(fpath(nm), ref=None if refd is None else {k: TState(W=v) for k, v in refd.items()})
            assert all(torch.equal(st[k].W, rest[k]) for k in rest), "rest not lossless"
            r["decode_s_first"] = {k: round(v, 3) for k, v in tim.items()}; r["lossless_verified"] = True
            if mode == "raw16":
                os.remove(fpath(nm)); r["file_deleted"] = True
            store.put(nm, r); log(f"{nm}: {r['MB']:.2f} MB ({r['bpp']:.3f} bit/param)")
    WB16 = {n: Bsd[f"{pfx}.{n}.weight"].to(torch.float16) for n in snap}
    del rest_base
    if not store.has("probe_c1_lite"):   # falsification check: is the fine-tune a SMALL delta of the base?
        num = den = 0.0; rels = []
        for n in snap:
            d = (snap[n].float() - WB16[n].float()).pow(2).sum().item(); w = snap[n].float().pow(2).sum().item()
            num += d; den += w; rels.append((d / w) ** 0.5)
        store.put("probe_c1_lite", {"rel_delta_global": (num / den) ** 0.5, "rel_delta_median": float(np.median(rels)),
                                    "pass_small_lt_0.30": (num / den) ** 0.5 < 0.30})
        log("probe_c1_lite", store.get("probe_c1_lite"))
    ref = get_ref(model, test_ids, os.path.join(DATA, f"ref_{args.ref_target}.pt"))
    chat_path = os.path.join(RES, f"chat_set_{args.ref_target}.json")
    if os.path.exists(chat_path):
        chat = json.load(open(chat_path))
    else:
        prompts = json.load(open(os.path.join(ROOT, "scripts", "chat_prompts.json")))
        chat = build_chat_set(model, tok, prompts); json.dump(chat, open(chat_path, "w"))
    chref_path = os.path.join(DATA, f"chatref_{args.ref_target}.pt")
    if os.path.exists(chref_path):
        chat_ref = torch.load(chref_path)
    else:
        chat_ref = chat_hidden(model, chat); torch.save(chat_ref, chref_path)
    E = dict(ref=ref, chat=chat, chat_ref=chat_ref)
    fp16ref = {n: TState(W=WB16[n]) for n in snap}

    if not store.has("T_fp16"):
        run_linears("T_fp16", model, snap, b_fp16, **E)
    _H = {}
    def HT():
        if "H" not in _H:
            restore_weights(model, snap)
            _H["H"] = collect_hessians(model, calib_windows(train_ids, args.calib), QWEN_SHARE)
        return _H["H"]

    if not store.has("T_gptq4_ans"):
        run_linears("T_gptq4_packed", model, snap, b_gptq4(HT()), also=("T_gptq4_ans", None), **E)
    for c in CS:
        if not store.has(f"T_ecsq_c{c}"):
            run_linears(f"T_ecsq_c{c}", model, snap, b_ecsq(HT(), c), **E)
    # ---- P-frames on the decoded base I-frame lattice (closed loop) ----
    for c in CS:
        nm = f"T_pframe_c{c}"
        if store.has(nm):
            continue
        rdB = Reader(fpath(f"B_ecsq_c{c}"))
        Bstate = {}
        def pf(n, W, c=c):
            sB = decode_record(rdB, n, None, "cpu")        # decoded base: Q_B, step
            Bstate[n] = TState(Q=sB.Q.to(torch.int16), step=sB.step)
            WBq = sB.Q.float() * sB.step
            S, _, _, _ = Q.gptq(W - WBq, HT()[n], "grid", step=sB.step, dev=GD)
            k, p, sec = enc_qrow(S, step_mode="ref", acc="add")
            return [(k, p, sec, S, 0)]
        class _R(dict):  # decoder reference = decoded base state (Q_B, step)
            def get(self, n, d=None):
                s = Bstate.get(n)
                return None if s is None else TState(Q=s.Q.to(torch.int32), step=s.step)
        run_linears(nm, model, snap, pf, R_of=_R(), **E)
        store.d[nm]["base_file"] = os.path.relpath(fpath(f"B_ecsq_c{c}"), ROOT); store.put(nm, store.d[nm])
        del Bstate, rdB; gc.collect()
    # ---- scenario 1: fp16 base available, entropy-coded full-rank delta ----
    for c in DCS:
        nm = f"T_delta_c{c}"
        if store.has(nm):
            continue
        def df(n, W, c=c):
            s = Q.ecsq_step(W, c)
            D = W - WB16[n].float()
            S, _, _, _ = Q.gptq(D, HT()[n], "grid", step=s, dev=GD)
            k, p, sec = enc_qrow(S, s, base="fp16ref")
            return [(k, p, sec, S, side_row(W))]
        run_linears(nm, model, snap, df, R_of=fp16ref, **E)
    # ---- LoRA-style low-rank delta (fp16 factors) ----
    for r in RANKS:
        nm = f"T_lowrank_r{r}"
        if store.has(nm):
            continue
        def lf(n, W, r=r):
            A, B = Q.lowrank(W - WB16[n].float(), r)
            k, p, sec = enc_lowrank(A, B)
            return [(k, p, sec, None, 16 * (A.numel() + B.numel()))]
        run_linears(nm, model, snap, lf, R_of=fp16ref, **E)
    restore_weights(model, snap)
log("done; disk free GB", round(disk_free_gb(), 2))
