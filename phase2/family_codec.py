"""Phase 2 / family codec with REAL BYTES: base I-frame + fine-tune P-frame / delta, vs independent.

Every config writes real .wcdc files (bitstreams/<tag>/...), and every quality number is measured
on a model DECODED FROM THAT FILE (never from the encoder's in-memory reconstruction).

DEFAULT CODERS (--preset default, REPORT §17; chosen from the §16 measurements):
  I-frame (base)        B_gfisher_c{c}          A3 Fisher grid (fp16 row x fp16 column steps) + GPTQ      c in --cs
  P-frame (fine-tune)   T_pframe_gfisher_rnd_c{c}  closed-loop P-frame on the decoded I-frame's lattice (inherits
                                                row AND column factors, 0 side info), PLAIN ROUNDING of the residual
                                                (--pframe gptq -> T_pframe_gfisher_c{c}, GPTQ on it; §17 control)
  delta over fp16 base  T_dhcol_c{c}            Hessian-column grid of the delta, PLAIN ROUNDING (no GPTQ)  c in --delta_cs
  independent control   T_gfisher_c{c}          the I-frame coder on the fine-tune alone (its own Fisher)   c in --indep_cs
--preset phase2 = the phase-2 coders of REPORT §14 (row-grid GPTQ-ECSQ B_ecsq / T_ecsq / T_pframe at --cs, GPTQ
row-grid T_delta at --delta_cs, LoRA T_lowrank at --ranks, zstd/gzip reference coders) -- the pre-§17 default
command line `family_codec.py ...` is now `family_codec.py --preset phase2 ...`.  --preset none = only the explicit
opt-in flags below (the §16 queue was `--preset none --no_baseline --ghcol_cs ... --gfisher_cs ...`).
Every old coder stays available as an opt-in flag: --row_cs, --row_delta_cs, --ranks, --ghcol_cs, --ghcol_delta_cs,
--gfisher_cs, --hcol_delta_cs, --pframe_round_cs.

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
ghcol configs (REPORT §16; opt-in via --ghcol_cs / --ghcol_delta_cs, the configs above are unchanged):
  B_ghcol_c{c}        base I-frame: GPTQ on the Hessian-COLUMN grid step_ij = fp16(c*rms_i) * fp16(diag(H)_j^-1/2
                      normalised), qrow record with a "col" section (16 b/row + 16 b/col side info)
  T_ghcol_c{c}        independent fine-tune, same coder (fine-tune's own Hessian)
  T_pframe_ghcol_c{c} closed-loop P-frame on the DECODED B_ghcol_c{c} lattice (inherits row AND column factors,
                      0 side info): q = GPTQ(W_T - Q_B*s, H_T) on s; decoder Q = Q_B + q
  T_dghcol_c{c}       fp16 base available: W = W_B16 + s*q, q = GPTQ(W_T - W_B16, H_T) on the hcol grid of the
                      delta (rows rms_row(Delta) [--ghcol_delta_rows D] or rms_row(W_T) [W])
  T_dhcol_c{c}        same delta grid, plain rounding (no GPTQ feedback)                           (--hcol_delta_cs)
  B_gfisher_c{c} / T_pframe_gfisher_c{c}   the same record type with the A3 FISHER factors instead of diag(H)
                      (quant.fisher_grid, gamma 0.5, global cross-matrix shift; base Fisher = 64 x 512 train
                      tokens, data/fisher_<ref_base>.pt) + GPTQ; the P-frame inherits that lattice  (--gfisher_cs)
  Fallback: a tensor whose column grid is degenerate (non-finite / zero factors) is coded on the row grid
  (record without "col"; counted in the result as n_row_fallback).
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
ap.add_argument("--preset", default="default", choices=["default", "phase2", "none"],
                help="default: Fisher-grid I-frame + P-frame on its lattice, plain-rounding Hessian-column delta "
                     "(REPORT §17); phase2: the row-grid coders of REPORT §14; none: only the explicit opt-in flags")
ap.add_argument("--cs", default="0.35,0.5", help="I-frame / P-frame steps c of the preset's family coder")
ap.add_argument("--delta_cs", default=None, help="steps c of the preset's delta coder "
                "(default preset: 0.8,1.6,3.2 on the column grid; phase2: 0.06,0.12,0.25 on the row grid)")
ap.add_argument("--indep_cs", default=None, help="steps c of the preset's independent fine-tune control "
                "(default preset: 0.35; phase2: = --cs)")
ap.add_argument("--pframe", default="round", choices=["gptq", "round"],
                help="P-frame quantizer of the default preset on the inherited Fisher lattice: plain rounding "
                     "(default; won the pre-registered REPORT §17 rule on chat-KL) or GPTQ (better WikiText KL)")
ap.add_argument("--ranks", default=None, help="LoRA ranks (phase2 preset: 32,64,128; default preset: none)")
ap.add_argument("--stages", default="base,target")
ap.add_argument("--max_windows", type=int, default=0)
ap.add_argument("--calib", type=int, default=128)
ap.add_argument("--refcoders", default=None, help="configs that also get zstd/gzip reference coding "
                "(phase2 preset: the phase-2 list; default preset: none)")
ap.add_argument("--dtype", default="fp16")
ap.add_argument("--gptq_dev", default="cpu")
ap.add_argument("--skip", default="", help="config names not to run (split work across jobs)")
ap.add_argument("--group", type=int, default=128, help="GPTQ-int4 group size (SmolLM2: 64, dims not /128)")
# ---- opt-in coders (added to whatever the preset runs) ----
ap.add_argument("--row_cs", default="", help="steps c of the phase-2 row-grid I-frame / independent / P-frame configs")
ap.add_argument("--row_delta_cs", default="", help="steps c of the phase-2 GPTQ row-grid delta (T_delta)")
ap.add_argument("--ghcol_cs", default="", help="steps c of the ghcol I-frame / independent / P-frame configs")
ap.add_argument("--ghcol_delta_cs", default="", help="steps c of the ghcol delta (fp16 base) configs")
ap.add_argument("--hcol_delta_cs", default="", help="steps c of the plain-rounding column-grid delta (no GPTQ)")
ap.add_argument("--ghcol_delta_rows", default="D", choices=["D", "W"], help="row RMS of the delta grid: Delta or W_T")
ap.add_argument("--gfisher_cs", default="", help="steps c of the Fisher-grid (A3) I-frame + P-frame configs")
ap.add_argument("--gfisher_indep_cs", default="", help="steps c of the independent Fisher-grid fine-tune (T_gfisher)")
ap.add_argument("--pframe_round_cs", default="", help="steps c of a PLAIN-ROUNDING P-frame on the Fisher lattice "
                "(T_pframe_gfisher_rnd_c*; needs B_gfisher_c* of the same c)")
ap.add_argument("--no_baseline", action="store_true", help="skip the reference files (fp16, GPTQ-int4, lossless rest, "
                "probe_c1_lite): they are coder-independent and already in results/phase2_family_<tag>.json")
ap.add_argument("--store", default="", help="results store name (default phase2_family_<tag>)")
args = ap.parse_args()
seed_all(0); check_disk()
DT = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
L = lambda x: [float(c) for c in (x or "").split(",") if c]
CS = L(args.cs)
if args.preset == "default":
    DCS = L(args.delta_cs if args.delta_cs is not None else "0.8,1.6,3.2")
    ICS = L(args.indep_cs if args.indep_cs is not None else "0.35")
    RANKS_S = args.ranks if args.ranks is not None else ""
    REFC_S = args.refcoders if args.refcoders is not None else ""
    RCS, RICS, RDCS = [], [], []                      # phase-2 row grid: opt-in only
    GFCS, GFICS, HDCS = list(CS), list(ICS), list(DCS)
elif args.preset == "phase2":
    DCS = L(args.delta_cs if args.delta_cs is not None else "0.06,0.12,0.25")
    ICS = L(args.indep_cs) if args.indep_cs is not None else list(CS)
    RANKS_S = args.ranks if args.ranks is not None else "32,64,128"
    REFC_S = args.refcoders if args.refcoders is not None else \
        "B_gptq4_ans,B_ecsq_c0.35,T_gptq4_ans,T_ecsq_c0.35,T_pframe_c0.35,T_delta_c0.12"
    RCS, RICS, RDCS = list(CS), list(ICS), list(DCS)
    GFCS, GFICS, HDCS = [], [], []
else:
    RANKS_S, REFC_S = args.ranks or "", args.refcoders or ""
    RCS, RICS, RDCS, GFCS, GFICS, HDCS = [], [], [], [], [], []
uniq = lambda xs: sorted(set(xs))
RCS = uniq(RCS + L(args.row_cs)); RICS = uniq(RICS + L(args.row_cs)); RDCS = uniq(RDCS + L(args.row_delta_cs))
GCS = L(args.ghcol_cs)
GDCS = L(args.ghcol_delta_cs)
GFCS = uniq(GFCS + L(args.gfisher_cs)); GFICS = uniq(GFICS + L(args.gfisher_indep_cs))
HDCS = uniq(HDCS + L(args.hcol_delta_cs))
PRCS = L(args.pframe_round_cs)
# P-frames on the Fisher lattice: (c, quantizer) -- the preset's own (args.pframe) + the rounding control
GF_PFRAMES = [(c, args.pframe if c in CS and args.preset == "default" else "gptq") for c in GFCS] + \
             [(c, "round") for c in PRCS if not (args.preset == "default" and args.pframe == "round" and c in CS)]
RANKS = [int(r) for r in RANKS_S.split(",") if r]
log(f"preset={args.preset}: gfisher I/P {GFCS} P-frames {GF_PFRAMES} indep {GFICS} | hcol-delta {HDCS} | "
    f"row I/P {RCS} indep {RICS} delta {RDCS} lora {RANKS} | ghcol {GCS} ghcol-delta {GDCS}")
REFC = set(REFC_S.split(","))
OUT = os.path.join(BITS, args.tag); os.makedirs(OUT, exist_ok=True)
store = Store(args.store or f"phase2_family_{args.tag}")
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


FALLBACK = {}
RAISED = {}      # config -> number of columns whose factor was raised by quant.int16_safe_col


def count_raised(tag, col_used, col_raw):
    RAISED[tag] = RAISED.get(tag, 0) + int((col_used != col_raw).sum())


def ghcol_rec(W, H, c, base="none", rowref=None, tag=""):
    """ghcol on the column grid -> qrow record with a 'col' section; row-grid GPTQ fallback if the column grid of
    this tensor is degenerate.  Returns (kind, params, sections, symbols, side_bits, Qdeq, row, col|None)."""
    S, Qd, row, col = Q.ghcol(W, H, c, rowref=rowref, dev=GD)
    count_raised(tag, col, Q.hcol_grid(W, torch.diag(H), c, rowref=rowref)[1])
    if not (torch.isfinite(Qd).all() and torch.isfinite(col).all() and bool((col > 0).all()) and Q.sym_fits_int16(S)):
        FALLBACK[tag] = FALLBACK.get(tag, 0) + 1
        row = Q.ecsq_step(W if rowref is None else rowref, c)
        S, Qd, _, _ = Q.gptq(W, H, "grid", step=row, dev=GD)
        k, p, sec = enc_qrow(S, row, base=base)
        return k, p, sec, S, 16 * W.shape[0], Qd, row, None
    k, p, sec = enc_qrow(S, row, base=base, col=col)
    return k, p, sec, S, 16 * (W.shape[0] + W.shape[1]), Qd, row, col


def b_ghcol(H, c, tag):
    def f(n, W):
        k, p, sec, S, side, _, _, _ = ghcol_rec(W, H[n], c, tag=tag)
        return [(k, p, sec, S, side)]
    return f


def b_gfisher(H, FG, c, tag=""):
    def f(n, W):
        row, col0 = Q.fisher_rowcol(FG[n], c)
        S, _, row, col = Q.gptq_sep(W, H[n], row, col0, dev=GD)       # col = the int16-safe factors actually used
        count_raised(tag, col, col0)
        if not Q.sym_fits_int16(S):                                  # row-grid fallback for this tensor
            FALLBACK[tag] = FALLBACK.get(tag, 0) + 1
            row = Q.ecsq_step(W, c)
            S, _, _, _ = Q.gptq(W, H[n], "grid", step=row, dev=GD)
            k, p, sec = enc_qrow(S, row)
            return [(k, p, sec, S, 16 * W.shape[0])]
        k, p, sec = enc_qrow(S, row, col=col)
        return [(k, p, sec, S, 16 * (W.shape[0] + W.shape[1]))]
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
    for mode in (() if args.no_baseline else ("raw16", "ll16")):
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
    if not store.has("B_fp16") and not args.no_baseline:
        run_linears("B_fp16", model, snap, b_fp16, ref)
    need_int4 = not store.has("B_gptq4_ans") and not args.no_baseline
    if [c for c in RCS if not store.has(f"B_ecsq_c{c}")] or need_int4:
        restore_weights(model, snap)
        HB = collect_hessians(model, calib_windows(train_ids, args.calib), QWEN_SHARE)
        if need_int4:
            run_linears("B_gptq4_packed", model, snap, b_gptq4(HB), ref, also=("B_gptq4_ans", None))
        for c in RCS:
            run_linears(f"B_ecsq_c{c}", model, snap, b_ecsq(HB, c), ref)
        del HB
    BGF = uniq(GFCS + PRCS)          # Fisher-grid I-frames (also the lattice of the rounding P-frame control)
    if any(not store.has(f"B_ghcol_c{c}") for c in GCS) or any(not store.has(f"B_gfisher_c{c}") for c in BGF):
        FG = None
        if BGF:
            FG = Q.fisher_grid({n: Q.rms_row(snap[n].float())[:, 0] for n in snap},
                               fisher_stats(args.base, args.ref_base, train_ids), gamma=0.5)
        restore_weights(model, snap)
        HB = collect_hessians(model, calib_windows(train_ids, args.calib), QWEN_SHARE)
        for c in GCS:
            nm = f"B_ghcol_c{c}"
            run_linears(nm, model, snap, b_ghcol(HB, c, nm), ref)
            if store.has(nm) and "n_row_fallback" not in store.d[nm]:
                store.d[nm]["n_row_fallback"] = FALLBACK.get(nm, 0); store.d[nm]["n_cols_raised"] = RAISED.get(nm, 0)
                store.put(nm, store.d[nm])
        for c in BGF:
            nm = f"B_gfisher_c{c}"
            run_linears(nm, model, snap, b_gfisher(HB, FG, c, nm), ref)
            if store.has(nm) and "n_row_fallback" not in store.d[nm]:
                store.d[nm]["n_row_fallback"] = FALLBACK.get(nm, 0); store.d[nm]["n_cols_raised"] = RAISED.get(nm, 0)
                store.put(nm, store.d[nm])
        del HB
    del model, snap; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()

# ======================================================================================
if "target" in args.stages:
    log("== TARGET stage:", args.target, "given base", args.base)
    # fine-tune's own Fisher for the independent Fisher-grid control: computed (or loaded from data/) BEFORE the
    # target model, Hessians and fp16 base are resident (fisher_stats loads its own fp32 copy)
    FST = fisher_stats(args.target, args.ref_target, train_ids) if any(
        not store.has(f"T_gfisher_c{c}") for c in GFICS) else None
    model = load(args.target)
    snap = snapshot_weights(model)
    rest = rest_state(model)
    from huggingface_hub import snapshot_download
    snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])   # no-op if cached
    Bsd = LazySD(args.base)
    pfx = model.base_model_prefix
    rest_base = {k: Bsd[k].to(torch.float16) for k in rest}
    for mode, refd in (() if args.no_baseline else (("raw16", None), ("ll16", None), ("xor_base", rest_base))):
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
    if not store.has("probe_c1_lite") and not args.no_baseline:   # falsification check: is the fine-tune a SMALL delta?
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

    if not store.has("T_fp16") and not args.no_baseline:
        run_linears("T_fp16", model, snap, b_fp16, **E)
    _H = {}
    def HT():
        if "H" not in _H:
            restore_weights(model, snap)
            _H["H"] = collect_hessians(model, calib_windows(train_ids, args.calib), QWEN_SHARE)
        return _H["H"]

    if not store.has("T_gptq4_ans") and not args.no_baseline:
        run_linears("T_gptq4_packed", model, snap, b_gptq4(HT()), also=("T_gptq4_ans", None), **E)
    for c in RICS:
        if not store.has(f"T_ecsq_c{c}"):
            run_linears(f"T_ecsq_c{c}", model, snap, b_ecsq(HT(), c), **E)
    # ---- P-frames on the decoded base I-frame lattice (closed loop) ----
    for c in RCS:
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
    for c in RDCS:
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
    # ---- ghcol (REPORT §16): independent, P-frame on the decoded ghcol I-frame lattice, delta vs fp16 base ----
    def note_fallback(nm):
        if store.has(nm) and "n_row_fallback" not in store.d[nm]:
            store.d[nm]["n_row_fallback"] = FALLBACK.get(nm, 0); store.d[nm]["n_cols_raised"] = RAISED.get(nm, 0)
            store.put(nm, store.d[nm])
    for c in GCS:
        nm = f"T_ghcol_c{c}"
        if not store.has(nm):
            run_linears(nm, model, snap, b_ghcol(HT(), c, nm), **E); note_fallback(nm)
    if FST is not None:              # independent control with the default I-frame coder (fine-tune's own Fisher)
        FGT = Q.fisher_grid({n: Q.rms_row(snap[n].float())[:, 0] for n in snap}, FST, gamma=0.5)
        for c in GFICS:
            nm = f"T_gfisher_c{c}"
            if not store.has(nm):
                run_linears(nm, model, snap, b_gfisher(HT(), FGT, c, nm), **E); note_fallback(nm)
        del FGT
    for lat, c, pq in [("ghcol", c, "gptq") for c in GCS] + [("gfisher", c, m) for c, m in GF_PFRAMES]:
        nm = f"T_pframe_{lat}_c{c}" if pq == "gptq" else f"T_pframe_{lat}_rnd_c{c}"
        if store.has(nm):
            continue
        rdB = Reader(fpath(f"B_{lat}_c{c}"))
        Bstate = {}
        def pfg(n, W, c=c, pq=pq, nm=nm):
            sB = decode_record(rdB, n, None, "cpu")        # decoded base: Q_B, row step, column factors
            Bstate[n] = TState(Q=sB.Q.to(torch.int16), step=sB.step, col=sB.col)
            s_full = sB.lattice_step()
            if pq == "gptq":       # GPTQ on the residual (target Hessian): = GPTQ of W_T on the base lattice
                S, _, _, _ = Q.gptq(W - sB.Q.float() * s_full, HT()[n], "grid", step=s_full, dev=GD)
            else:                  # plain rounding of the residual on the same lattice (REPORT §17 control)
                S = Q.ecsq(W - sB.Q.float() * s_full, s_full)
            if not Q.sym_fits_int16(S) or not Q.sym_fits_int16(sB.Q.to(torch.int32) + S):
                raise ValueError(f"{nm} {n}: P-frame lattice index beyond int16 (decoders hold int16 symbols)")
            k, p, sec = enc_qrow(S, step_mode="ref", acc="add", col_mode="ref" if sB.col is not None else None)
            return [(k, p, sec, S, 0)]
        class _RG(dict):  # decoder reference = decoded ghcol base state (Q_B, row step, column factors)
            def get(self, n, d=None):
                s = Bstate.get(n)
                return None if s is None else TState(Q=s.Q.to(torch.int32), step=s.step, col=s.col)
        run_linears(nm, model, snap, pfg, R_of=_RG(), **E)
        store.d[nm]["base_file"] = os.path.relpath(fpath(f"B_{lat}_c{c}"), ROOT); store.put(nm, store.d[nm])
        del Bstate, rdB; gc.collect()
    for c in GDCS:
        nm = f"T_dghcol{'W' if args.ghcol_delta_rows == 'W' else ''}_c{c}"
        if store.has(nm):
            continue
        def dgf(n, W, c=c, nm=nm):
            D = W - WB16[n].float()
            k, p, sec, S, side, _, _, _ = ghcol_rec(D, HT()[n], c, base="fp16ref",
                                                    rowref=W if args.ghcol_delta_rows == "W" else None, tag=nm)
            return [(k, p, sec, S, side)]
        run_linears(nm, model, snap, dgf, R_of=fp16ref, **E); note_fallback(nm)
    for c in HDCS:
        nm = f"T_dhcol_c{c}"
        if store.has(nm):
            continue
        def dhf(n, W, c=c):
            D = W - WB16[n].float()
            row, col0 = Q.hcol_grid(D, torch.diag(HT()[n]), c)
            col, _ = Q.int16_safe_col(D, row, col0)
            count_raised(f"T_dhcol_c{c}", col, col0)
            S = Q.ecsq(D, row * col[None, :])
            assert Q.sym_fits_int16(S), f"{n}: rounding symbols beyond int16 after the column floor"
            k, p, sec = enc_qrow(S, row, base="fp16ref", col=col)
            return [(k, p, sec, S, 16 * (W.shape[0] + W.shape[1]))]
        run_linears(nm, model, snap, dhf, R_of=fp16ref, **E); note_fallback(nm)
    restore_weights(model, snap)
log("done; disk free GB", round(disk_free_gb(), 2))
