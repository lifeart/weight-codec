"""D1 PROBE (training checkpoints = frames), Pythia-410M, a SAMPLE of revisions.
Streaming, disk-safe: each revision's pytorch_model.bin (fp16, ~0.9GB) is downloaded into
its own empty dir models/pythia/<rev>/, loaded with torch.load(weights_only=True), and
deleted immediately; only the previous checkpoint is kept (in RAM).

Per consecutive pair (s -> t) in the sample, over block linears (and embeddings separately):
  ||W_t - W_s|| / ||W_t||, singular spectrum of the delta (top-5%-rank energy vs shuffled
  null, r90), quantized-domain open-loop P-frame cost on a shared uniform grid
  step = c*rms_row(W_s): H(q_t - q_s) vs H(q_t), fraction of changed indices,
  and lossless fp16 bit-pattern coding: H(bits_t) vs H(bits_t XOR bits_s) (16-bit symbols).

PRE-REGISTERED: D1 PASS iff for the late-training 1000-step pairs (s >= 64000)
  H(q_t - q_s) <= 0.5 * H(q_t) at c = 0.35 (int4-like grid), i.e. P-frames cost <= half an
  I-frame.  Secondary (reported, not gating): 1000-step rel-delta shrinks with training.
"""
from common import *
from huggingface_hub import hf_hub_download

REPO = "EleutherAI/pythia-410m"
STEPS = [1, 128, 512, 1000, 2000, 8000, 32000, 33000, 64000, 65000, 128000, 142000, 143000]
MDIR = os.path.join(ROOT, "models", "pythia")


def fetch(step):
    """Download pytorch_model.bin of a revision straight into RAM (no disk write at all),
    load with torch.load(weights_only=True) (restricted unpickler: no code execution)."""
    import io
    from huggingface_hub import hf_hub_url, get_session
    check_disk(3.0)
    url = hf_hub_url(REPO, "pytorch_model.bin", revision=f"step{step}")
    for attempt in range(3):
        try:
            r = get_session().get(url, timeout=120, follow_redirects=True)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            break
        except Exception as e:                     # network hiccup: retry, then re-raise
            log(f"fetch step{step} attempt {attempt} failed: {e!r}")
            if attempt == 2:
                raise
            time.sleep(10)
    del r
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    del buf
    return {k: v for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def is_lin(k):
    return k.startswith("gpt_neox.layers.") and k.endswith(".weight") and ("attention" in k or "mlp" in k)


def fp16_bits(t):
    return t.to(torch.float16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def pair_stats(S, T):
    rows = {}
    for k in T:
        if k not in S or T[k].dim() != 2:
            continue
        a, b = S[k].float(), T[k].float()
        D = b - a
        rec = {"numel": b.numel(), "rel": float(D.norm() / (b.norm() + 1e-12)),
               "frac_bit_identical": float((S[k] == T[k]).float().mean())}
        if is_lin(k):
            sv = svals(D)
            Dn = D.flatten()[torch.randperm(D.numel())].reshape(D.shape)
            svn = svals(Dn)
            r5 = max(1, int(0.05 * min(D.shape)))
            rec.update({"top5pct": energy_topr(sv, [r5])[r5], "top5pct_null": energy_topr(svn, [r5])[r5],
                        "r90_over_min": rank_for(sv, 0.9) / min(D.shape)})
            rms = a.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)
            for c in (0.35, 0.7):
                st = (c * rms).half().float()
                qa = torch.round(a / st).to(torch.int32); qb = torch.round(b / st).to(torch.int32)
                rec[f"q{c}"] = {"H_t": entropy_bits(qb)[0], "H_diff": entropy_bits(qb - qa)[0],
                                "frac_changed": float((qb != qa).float().mean())}
            bt, bs = fp16_bits(T[k]), fp16_bits(S[k])
            rec["lossless"] = {"H_fp16": entropy_bits(bt)[0], "H_xor": entropy_bits(bt ^ bs)[0]}
        rows[k] = rec
    lin = {k: v for k, v in rows.items() if is_lin(k)}
    w = np.array([v["numel"] for v in lin.values()], float)
    wa = lambda f: float(sum(v["numel"] * f(v) for v in lin.values()) / w.sum())
    agg = {"rel_rms": float(np.sqrt(wa(lambda v: v["rel"] ** 2))), "top5pct": wa(lambda v: v["top5pct"]),
           "top5pct_null": wa(lambda v: v["top5pct_null"]), "r90_over_min": wa(lambda v: v["r90_over_min"]),
           "frac_bit_identical": wa(lambda v: v["frac_bit_identical"]),
           "lossless_H_fp16": wa(lambda v: v["lossless"]["H_fp16"]), "lossless_H_xor": wa(lambda v: v["lossless"]["H_xor"])}
    for c in (0.35, 0.7):
        agg[f"q{c}_H_t"] = wa(lambda v: v[f"q{c}"]["H_t"])
        agg[f"q{c}_H_diff"] = wa(lambda v: v[f"q{c}"]["H_diff"])
        agg[f"q{c}_frac_changed"] = wa(lambda v: v[f"q{c}"]["frac_changed"])
    for k in rows:
        if "embed" in k:
            agg[f"rel[{k}]"] = rows[k]["rel"]
    return agg, rows


if __name__ == "__main__":
    store = Store("probe_d1")
    torch.manual_seed(0)
    prev, prev_step = None, None
    for st in STEPS:
        key_next = None
        if prev is not None:
            key = f"step{prev_step}->step{st}"
            if store.has(key):
                log("have", key)
        need = prev is None or not store.has(f"step{prev_step}->step{st}")
        # we always need the tensors of st if the next pair is missing too
        idx = STEPS.index(st)
        nxt_missing = idx + 1 < len(STEPS) and not store.has(f"step{st}->step{STEPS[idx + 1]}")
        if not need and not nxt_missing:
            prev, prev_step = None, st
            continue
        t0 = time.time()
        cur = fetch(st)
        log(f"fetched step{st} in {time.time() - t0:.0f}s; disk free {disk_free_gb():.2f} GB")
        if prev is not None and need:
            agg, rows = pair_stats(prev, cur)
            agg["gap"] = st - prev_step
            store.put(f"step{prev_step}->step{st}", {"aggregate": agg, "tensors": rows})
            log(f"step{prev_step}->step{st}", json.dumps({k: round(v, 4) for k, v in agg.items()}))
        prev, prev_step = cur, st
        gc.collect()
    # verdict
    late = [v["aggregate"] for k, v in store.d.items() if k.startswith("step") and v["aggregate"]["gap"] == 1000
            and int(k.split("->")[0][4:]) >= 64000]
    ratio = float(np.mean([a["q0.35_H_diff"] / a["q0.35_H_t"] for a in late])) if late else float("nan")
    gap1000 = sorted([(int(k.split("->")[0][4:]), v["aggregate"]["rel_rms"]) for k, v in store.d.items()
                      if k.startswith("step") and v["aggregate"]["gap"] == 1000])
    verdict = {"late_1000step_Hdiff_over_Ht": ratio, "rel_rms_1000step_by_stage": gap1000,
               "D1": "PASS" if ratio <= 0.5 else "FAIL"}
    store.put("_verdict", verdict)
    log(json.dumps(verdict, indent=1))
