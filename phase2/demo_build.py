"""Build the data of the browser demo (docs/): SmolLM2-135M -> SmolLM2-135M-Instruct as
"base cached once + entropy-coded delta", decoded and verified in the visitor's browser.

What it writes
  bitstreams/smol135/B_lin_ll16.wcdc   lossless fp16 base linears (new; needed by the fp16-base + delta chain)
  <stage>/*.wcdc                      rans16 twins (lossless re-encode, +0.05 %) of every demo file  [git-ignored]
  <stage>/tokenizer.json              official tokenizer of SmolLM2-135M-Instruct (Apache-2.0)     [git-ignored]
  docs/data/manifest.json             chains, sizes and the SHA-256 digests the browser must reproduce
  docs/data/generations.json          precomputed greedy answers (PyTorch, fp32, CPU) for the fallback table
  docs/data/results.json, summary.json, docs/figures/   offline numbers / phase-2 tables shown on the page
  phase2/demo_testvec.json            tokenizer + forward-pass test vectors for phase2/test_demo.mjs

Digests (all hex SHA-256 of the concatenated per-record SHA-256, in the file's record order):
  sha256      decoded SYMBOLS: lattice index Q (qrow, int16 LE) or fp16 bit pattern (ll16)  -- codec bit-exactness
  w_sha256    RECONSTRUCTED fp16 weights (uint16 LE) exactly as the Python decoder rebuilds them
  orig_sha256 lossless files only: the ORIGINAL tensors (official bf16 checkpoint cast to fp16), computed from
              the checkpoint, not from the decode.  w_sha256 == orig_sha256  <=>  bit-identical to the original.
Lossy files (I-frame / P-frame / delta linears) have no orig_sha256: they are approximations; their measured
quality (chat-KL etc.) is copied from the results store that measured THAT file (results/default_family_smol135.json
for the default coders of REPORT §17, results/phase2_family_smol135.json for the phase-2 ones): models decoded from
the very same constriction files these rans16 twins are a lossless transcode of -- checked by path AND byte count.

Demo files = the family codec's DEFAULT coders (REPORT §17), written by
  phase2/family_codec.py --base HuggingFaceTB/SmolLM2-135M --target HuggingFaceTB/SmolLM2-135M-Instruct --tag smol135 ...
  I-frame  B_gfisher_c0.35   Fisher-grid steps (fp16 row x fp16 column) + GPTQ     ("smallest" chain, with the P-frame)
  P-frame  T_pframe_gfisher_rnd_c0.35  closed-loop P-frame on that lattice (inherits row + column factors), plain rounding
  delta    T_dhcol_c1.6      Hessian-column grid of the delta, plain rounding, over the lossless fp16 base
  control  T_gfisher_c0.35   the I-frame coder on the Instruct model alone (no base reuse)
Public text (docs/data/*.json) passes public_clean(): no hardware / infrastructure details.

Usage:  .venv/bin/python phase2/demo_build.py [--stage docs/bitstreams] [--no-gen] [--results-only]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from wcodec import Reader, Writer, decode, enc_ll16, TState
from wcodec.transcode import transcode

BASE_ID, INST_ID = "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"
SRC = os.path.join(ROOT, "bitstreams", "smol135")
DOCS = os.path.join(ROOT, "docs")
RES = os.path.join(ROOT, "results")

# the family codec's default coders (REPORT §17) -- names as written by phase2/family_codec.py --preset default
IFRAME, PFRAME, DELTA, INDEP = "B_gfisher_c0.35", "T_pframe_gfisher_rnd_c0.35", "T_dhcol_c1.6", "T_gfisher_c0.35"
QUALITY_STORES = ("default_family_smol135", "phase2_family_smol135")      # results/<store>.json, searched in order
PREV = {"iframe": "B_ecsq_c0.35", "pframe": "T_pframe_c0.35", "delta": "T_delta_c0.12", "indep": "T_ecsq_c0.35"}  # phase 2
FILES = {   # name -> (role, model, lossless?, description)
    "B_rest_ll16": ("base", "base", True, "base embeddings + norms, lossless fp16 (ll16)"),
    IFRAME: ("base", "base", False, "base linears, I-frame: Fisher-grid steps (row x column) + GPTQ, c=0.35 (int4-class)"),
    "B_lin_ll16": ("base", "base", True, "base linears, lossless fp16 (ll16)"),
    "T_rest_xor_base": ("delta", "instruct", True, "Instruct embeddings + norms, lossless, XOR with the base"),
    PFRAME: ("delta", "instruct", False, "Instruct linears, P-frame on the base I-frame's lattice, plain rounding (c=0.35)"),
    DELTA: ("delta", "instruct", False, "Instruct linears, full-rank delta over the fp16 base: Hessian-column grid, "
                                         "plain rounding (c=1.6)"),
    "T_rest_ll16": ("standalone", "instruct", True, "Instruct embeddings + norms, lossless, no base"),
    INDEP: ("standalone", "instruct", False, "Instruct linears, standalone: the same Fisher-grid coder, c=0.35 (no base)"),
}
CHAINS = {
    "delta": {"title": "fp16 base + full-rank Δ (column grid, c=1.6)", "short": "near-lossless",
              "groups": [["B_rest_ll16", "T_rest_xor_base"], ["B_lin_ll16", DELTA]],
              "quality_file": DELTA, "base_lossless": True, "prev_quality_file": PREV["delta"]},
    "family": {"title": "int4-class family: base I-frame + Instruct P-frame (Fisher grid, c=0.35)", "short": "smallest",
               "groups": [["B_rest_ll16", "T_rest_xor_base"], [IFRAME, PFRAME]],
               "quality_file": PFRAME, "base_lossless": False, "prev_quality_file": PREV["pframe"]},
    "indep": {"title": "control: standalone Instruct, same coder c=0.35 (no base reuse)", "short": "control",
              "groups": [["T_rest_ll16"], [INDEP]], "quality_file": INDEP, "base_lossless": None,
              "prev_quality_file": PREV["indep"]},
}
# public text: no hardware / infrastructure details (applied to every string exported into docs/data)
PUBLIC_REWRITE = [(r"own numba/Metal rANS", "own numba / GPU rANS"),
                  (r"\(Apple Silicon laptop, rans16 on 8 cores\)", "(rans16 on 8 CPU threads)"),
                  (r" on an Apple Silicon laptop\.", ", measured offline."),
                  (r"GPU \(Metal rANS\)", "GPU (rANS kernel)"),
                  (r"encoded on hf-job", "encoded on a separate x86 machine"),
                  (r"\(MPS fp16,", "(GPU fp16,"),
                  (r"decodes bit-exactly on the Mac", "decodes bit-exactly on a different (ARM) machine")]
PUBLIC_FORBIDDEN = (r"(?i)((?<!-)apple(?!-system)|silicon|\bmacs?\b|macbook|\bm[1-4]\b|laptop|\bmetal\b|\bmps\b|swap|free disk|"
                    r"\$\s?\d|\busd\b|pay-as-you-go|\bPRO\b|hf[- _]?jobs?\b|cpu-xl|cpu-upgrade|hf_[A-Za-z0-9]{20,}|"
                    r"/users/|lifeart92|gmail)")
GEN_PROMPTS = [1, 2, 6, 10, 37]          # indices into results/chat_set_smol135_instruct.json
TEST_PROMPTS = [1, 19]
MAX_NEW = 64


def H(b):
    return hashlib.sha256(b).digest()


def bits16(t):
    return t.detach().cpu().contiguous().to(torch.float16).view(torch.int16).numpy().view(np.uint16).tobytes()


def lin_names(n_layers=30):
    return [f"layers.{i}.{m}" for i in range(n_layers) for m in
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]


def load_checkpoint(rid):
    """Official checkpoint (bf16 safetensors) -> {key: bf16 tensor}."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(rid, allow_patterns=["model.safetensors", "*.json"])
    sd = {}
    with safe_open(os.path.join(d, "model.safetensors"), "pt") as fh:
        for k in fh.keys():
            sd[k] = fh.get_tensor(k)
    return sd, d


def fp16_view(sd):
    """The codec's 'original' = the checkpoint cast to fp16 (phase-2 loads models with dtype=float16).
    Also reports whether that cast is exact (bf16 -> fp16 loses values < 2^-14 or > 65504)."""
    out, inexact, tot = {}, 0, 0
    for k, v in sd.items():
        w = v.to(torch.float16)
        inexact += int((w.to(v.dtype) != v).sum()); tot += v.numel()
        out[k] = w
    return out, {"n_params": tot, "n_inexact_bf16_to_fp16": inexact}


def make_lossless_base_linears(B16):
    dst = os.path.join(SRC, "B_lin_ll16.wcdc")
    if os.path.exists(dst):
        return dst
    wr = Writer({"what": "linears", "mode": "ll16", "xor_ref": False, "tag": "smol135"})
    for n in lin_names():
        k, p, sec = enc_ll16(B16[f"model.{n}.weight"])
        wr.add(n, k, list(B16[f"model.{n}.weight"].shape), p, sec)
    nb = wr.write(dst)
    st = decode(Reader(dst), device="cpu")
    assert all(torch.equal(st[n].W, B16[f"model.{n}.weight"]) for n in lin_names()), "B_lin_ll16 not lossless"
    print(f"wrote {dst}: {nb / 1e6:.2f} MB ({8 * nb / sum(B16[f'model.{n}.weight'].numel() for n in lin_names()):.3f} bit/weight)")
    return dst


def rec_sym_digest(rec, st):
    if rec["kind"] == "qrow":
        q = st.Q.cpu().numpy()
        if q.size and (q.min() < -32768 or q.max() > 32767):        # the int16 digest (and the JS decoder) would wrap
            raise ValueError(f"{rec['name']}: lattice index outside int16 [{q.min()}, {q.max()}]")
        return H(q.astype("<i2").tobytes())
    return H(bits16(st.W))


def public_rewrite(x):
    """Apply PUBLIC_REWRITE to every string of a JSON-like object."""
    import re
    if isinstance(x, dict):
        return {k: public_rewrite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [public_rewrite(v) for v in x]
    if isinstance(x, str):
        for a, b in PUBLIC_REWRITE:
            x = re.sub(a, b, x)
    return x


def public_clean(paths):
    """Raise if any public demo file contains hardware / infrastructure / account details (PUBLIC_FORBIDDEN)."""
    import re
    pat, bad = re.compile(PUBLIC_FORBIDDEN), []
    for pth in paths:
        txt = open(pth, encoding="utf-8").read()
        for m in pat.finditer(txt):
            bad.append(f"{os.path.relpath(pth, ROOT)}: {m.group(0)!r} in ...{txt[max(0, m.start() - 50):m.end() + 30]!r}")
    if bad:
        raise SystemExit("public text check FAILED:\n" + "\n".join(bad[:40]))
    return len(paths)


def public_files():
    out = [os.path.join(DOCS, f) for f in ("index.html", "app.js", "config.js", "style.css")]
    out += [os.path.join(DOCS, "js", f) for f in sorted(os.listdir(os.path.join(DOCS, "js")))]
    out += [os.path.join(DOCS, "data", f) for f in sorted(os.listdir(os.path.join(DOCS, "data")))]
    out.append(os.path.join(ROOT, "spaces", "weight-codec-static", "README.md"))
    return out


def build_stage(stage):
    os.makedirs(stage, exist_ok=True)
    for f in sorted(os.listdir(stage)):          # the stage (git-ignored) holds exactly the manifest's files
        if f.endswith(".wcdc") and f[:-5] not in FILES:
            os.remove(os.path.join(stage, f)); print("removed stale stage file", f)
    for nm in FILES:
        dst = os.path.join(stage, nm + ".wcdc")
        src = os.path.join(SRC, nm + ".wcdc")
        # re-transcode when the source is newer (a re-encoded file must never be served from a stale twin)
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            transcode(src, dst)


def decode_chains(stage, B16, T16):
    files, models = {}, {}
    orig = {"base": B16, "instruct": T16}
    for cname, ch in CHAINS.items():
        models[cname] = {}
        for grp in ch["groups"]:
            ref = None
            for nm in grp:
                rd = Reader(os.path.join(stage, nm + ".wcdc"))
                tim = {}
                st = decode(rd, ref=ref, device="cpu", timing=tim, keep_state=True)
                if nm not in files:
                    role, mdl, lossless, desc = FILES[nm]
                    sd, wd = [], []
                    od = [] if lossless else None
                    for r in rd.records:
                        sd.append(rec_sym_digest(r, st[r["name"]]))
                        wd.append(H(bits16(st[r["name"]].W)))
                        if lossless:
                            key = r["name"] if r["name"].startswith("model.") else f"model.{r['name']}.weight"
                            od.append(H(bits16(orig[mdl][key])))
                    files[nm] = {"bytes": rd.nbytes, "role": role, "model": mdl, "lossless": lossless, "desc": desc,
                                 "n_records": len(rd.records),
                                 "n_weights": int(sum(int(np.prod(r["shape"])) for r in rd.records)),
                                 "sha256": H(b"".join(sd)).hex(), "w_sha256": H(b"".join(wd)).hex(),
                                 "src_bytes_constriction": os.path.getsize(os.path.join(SRC, nm + ".wcdc")),
                                 "py_decode_s_1thread_constriction_free": round(sum(tim.values()), 2)}
                    if lossless:
                        files[nm]["orig_sha256"] = H(b"".join(od)).hex()
                        files[nm]["identical_to_original"] = files[nm]["orig_sha256"] == files[nm]["w_sha256"]
                        assert files[nm]["identical_to_original"], nm
                for k, s in st.items():
                    models[cname].setdefault(FILES[nm][1], {})[k] = s.W
                ref = st
    return files, models


def store_entry(nm):
    """(store, entry) of the results store that measured bitstreams/smol135/<nm>.wcdc -- the file on disk must be
    the measured one (same path and byte count), else the quality shown on the page would belong to another file."""
    src = os.path.join(SRC, nm + ".wcdc")
    for st in QUALITY_STORES:
        p = os.path.join(RES, st + ".json")
        r = json.load(open(p)).get(nm) if os.path.exists(p) else None
        if r is None:
            continue
        if r.get("file") != os.path.relpath(src, ROOT) or r.get("bytes") != os.path.getsize(src):
            raise SystemExit(f"{nm}: {src} ({os.path.getsize(src)} B) is not the file measured in {st} "
                             f"({r.get('file')}, {r.get('bytes')} B) -- re-run the family codec or fix the store")
        return st, r
    raise SystemExit(f"{nm}: no results store measured it ({QUALITY_STORES})")


def quality(nm):
    st, r = store_entry(nm)
    return dict({k: r.get(k) for k in ("chat_kl", "chat_top1", "kl", "top1", "ppl", "real_bpw")}, store=f"results/{st}.json")


def state_dict_for(Wmap):
    """{record name -> fp16 W} (linears 'layers.*' + rest 'model.*') -> HF LlamaForCausalLM state dict, fp32."""
    sd = {}
    for k, w in Wmap.items():
        key = k if k.startswith("model.") else f"model.{k}.weight"
        sd[key] = w.float()
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd


def hf_model(cfg_dir, sd):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(cfg_dir)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)
    m.tie_weights()
    return m.eval()


@torch.no_grad()
def greedy(m, ids, n=MAX_NEW, eos=2):
    out = m.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                     max_new_tokens=n, do_sample=False, eos_token_id=eos, pad_token_id=eos)
    return out[0, len(ids):].tolist()


@torch.no_grad()
def first_logits(m, ids, k=10):
    lg = m(torch.tensor([ids])).logits[0, -1].float()
    v, i = torch.topk(lg, k)
    return {"top_ids": i.tolist(), "top_vals": [round(x, 5) for x in v.tolist()]}


def export_results():
    """Small JSON + figures the page shows (all measured offline): docs/data/results.json = the demo's default-coder
    files next to the phase-2 ones they replace (REPORT §17 / §14.3), docs/data/summary.json = the phase-2 tables
    (REPORT §14, unchanged numbers; public_rewrite'd), docs/figures/*.png."""
    F = {}
    for st in reversed(QUALITY_STORES):                   # phase 2 first; the default store adds the new configs
        for k, v in json.load(open(os.path.join(RES, st + ".json"))).items():
            if k.startswith(("B_", "T_")) and isinstance(v, dict) and "MB" in v:
                F[k] = {f: v.get(f) for f in ("MB", "real_bpw", "chat_kl", "chat_top1", "kl", "ppl") if f in v}
    for nm in FILES:                                      # every lossy demo file: quality from the store that measured it
        if not FILES[nm][2]:
            store_entry(nm)
    F["B_lin_ll16"] = {"MB": os.path.getsize(os.path.join(SRC, "B_lin_ll16.wcdc")) / 1e6, "real_bpw": None,
                       "kl": 0.0, "ppl": F["B_fp16"]["ppl"]}

    def pair(scen, kind, files, ft, base):
        return {"scenario": scen, "kind": kind, "files": files, "pair_MB": sum(F[n]["MB"] for n in files),
                "base_ppl": F[base].get("ppl"), "base_kl": F[base].get("kl"), "ft_chat_kl": F[ft].get("chat_kl"),
                "ft_chat_top1": F[ft].get("chat_top1")}
    pairs = [
        pair("both fp16 (raw)", "ref", ["B_fp16", "B_rest_raw16", "T_fp16", "T_rest_raw16"], "T_fp16", "B_fp16"),
        pair("two standard GPTQ-int4 g64 files (rest fp16)", "ref",
             ["B_gptq4_packed", "B_rest_raw16", "T_gptq4_packed", "T_rest_raw16"], "T_gptq4_packed", "B_gptq4_packed"),
        pair("both standalone, Fisher-grid coder c=0.35 (no base reuse)", "control",
             [IFRAME, "B_rest_ll16", INDEP, "T_rest_ll16"], INDEP, IFRAME),
        pair("previous coder: family, row-grid I-frame + P-frame c=0.35", "prev",
             [PREV["iframe"], "B_rest_ll16", PREV["pframe"], "T_rest_xor_base"], PREV["pframe"], PREV["iframe"]),
        pair("family: Fisher-grid I-frame + P-frame c=0.35", "default",
             [IFRAME, "B_rest_ll16", PFRAME, "T_rest_xor_base"], PFRAME, IFRAME),
        pair("previous coder: lossless fp16 base + Δ (GPTQ, row grid) c=0.12", "prev",
             ["B_lin_ll16", "B_rest_ll16", PREV["delta"], "T_rest_xor_base"], PREV["delta"], "B_lin_ll16"),
        pair("lossless fp16 base + Δ (column grid, rounding) c=1.6", "default",
             ["B_lin_ll16", "B_rest_ll16", DELTA, "T_rest_xor_base"], DELTA, "B_lin_ll16"),
    ]
    out = {"source": "results/default_family_smol135.json (default coders, REPORT §17) and results/phase2_family_smol135.json "
                     "(phase-2 coders and lossless parts, REPORT §14.3); every number measured on a model decoded from the file",
           "names": {"iframe": IFRAME, "pframe": PFRAME, "delta": DELTA, "indep": INDEP, "prev": PREV},
           "pairs": pairs, "files": F,
           "chat_set": "48 fixed prompts; chat-KL = KL(original fp16 Instruct || candidate) on the original's greedy answers",
           "delta_vs_lora": [["T_dhcol_c3.2", "T_lowrank_r32"], ["T_dhcol_c1.6", "T_lowrank_r64"],
                             ["T_dhcol_c0.8", "T_lowrank_r128"]]}
    os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(DOCS, "data", "results.json"), "w"), indent=1)
    SS = json.load(open(os.path.join(RES, "phase2_space_summary.json")))
    SS.pop("demo", None)
    json.dump(public_rewrite(SS), open(os.path.join(DOCS, "data", "summary.json"), "w"), indent=1, ensure_ascii=False)
    os.makedirs(os.path.join(DOCS, "figures"), exist_ok=True)
    for sec in SS["sections"]:
        for img in sec.get("images") or []:
            shutil.copy(os.path.join(ROOT, "figures", img), os.path.join(DOCS, "figures", img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=os.path.join(DOCS, "bitstreams"))
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--results-only", action="store_true", help="only rewrite docs/data/{results,summary}.json + figures")
    a = ap.parse_args()
    export_results()
    if a.results_only:
        print("public text check:", public_clean(public_files()), "files clean")
        return
    t0 = time.time()
    Bsd, bdir = load_checkpoint(BASE_ID)
    Tsd, tdir = load_checkpoint(INST_ID)
    B16, bcast = fp16_view(Bsd)
    T16, tcast = fp16_view(Tsd)
    del Bsd, Tsd
    print("bf16->fp16 cast:", bcast, tcast)
    make_lossless_base_linears(B16)
    build_stage(a.stage)
    shutil.copy(os.path.join(tdir, "tokenizer.json"), os.path.join(a.stage, "tokenizer.json"))
    files, models = decode_chains(a.stage, B16, T16)
    chains = {}
    for c, ch in CHAINS.items():
        names = [nm for g in ch["groups"] for nm in g]
        chains[c] = dict(ch, base_bytes=sum(files[n]["bytes"] for n in names if files[n]["role"] == "base"),
                         delta_bytes=sum(files[n]["bytes"] for n in names if files[n]["role"] == "delta"),
                         standalone_bytes=sum(files[n]["bytes"] for n in names if files[n]["role"] == "standalone"),
                         quality=quality(ch["quality_file"]), prev_quality=quality(ch["prev_quality_file"]))
    man = {"model_pair": [BASE_ID, INST_ID], "chains": chains, "files": files,
           "original": {"what": "official bf16 checkpoints cast to fp16 (the codec's reference)",
                        "base_cast": bcast, "instruct_cast": tcast,
                        "base_bf16_safetensors_bytes": os.path.getsize(os.path.join(bdir, "model.safetensors")),
                        "instruct_bf16_safetensors_bytes": os.path.getsize(os.path.join(tdir, "model.safetensors"))},
           "note": "rans16 = lossless re-encode of the constriction files measured in REPORT §14 (+0.05 % bytes)"}
    os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
    json.dump(man, open(os.path.join(DOCS, "data", "manifest.json"), "w"), indent=1)
    print(json.dumps({n: {k: f[k] for k in ("bytes", "lossless")} for n, f in files.items()}, indent=0))
    if a.no_gen:
        return
    # ---------------- precomputed answers + JS test vectors (PyTorch fp32 CPU, greedy) ----------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(INST_ID)
    chat = json.load(open(os.path.join(RES, "chat_set_smol135_instruct.json")))
    origW = {"base": {k: v for k, v in B16.items() if k != "lm_head.weight"},
             "instruct": {k: v for k, v in T16.items() if k != "lm_head.weight"}}
    variants = {   # label -> weights
        "base_original": origW["base"], "instruct_original": origW["instruct"],
        "instruct_delta": models["delta"]["instruct"], "instruct_family": models["family"]["instruct"],
        "base_family_iframe": models["family"]["base"],
    }
    gens = {"prompts": [chat[i]["prompt"] for i in GEN_PROMPTS], "max_new_tokens": MAX_NEW,
            "how": "PyTorch fp32 on CPU, greedy (no sampling), chat template of SmolLM2-135M-Instruct for every model "
                   "(the base model gets the same token ids).", "outputs": {}}
    tv = {"prompts": [], "chat_ids": [], "forward": {}}
    for lbl, W in variants.items():
        m = hf_model(tdir, state_dict_for(W))
        outs = []
        for i in GEN_PROMPTS:
            ids = chat[i]["ids"][:chat[i]["start"]]
            outs.append(tok.decode(greedy(m, ids), skip_special_tokens=True))
        gens["outputs"][lbl] = outs
        if lbl in ("instruct_delta", "instruct_family", "base_family_iframe"):
            tv["forward"][lbl] = []
            for i in TEST_PROMPTS:
                ids = chat[i]["ids"][:chat[i]["start"]]
                tv["forward"][lbl].append({"prompt_ids": ids, "greedy": greedy(m, ids, 32), **first_logits(m, ids)})
        print(lbl, "done", round(time.time() - t0), "s")
        del m
    json.dump(gens, open(os.path.join(DOCS, "data", "generations.json"), "w"), indent=1, ensure_ascii=False)
    texts = ["Hello world", "  leading spaces and\ttabs\n\nnew lines", "Numbers: 3x + 7 = 22, 1234567 and 3.14159",
             "Unicode: café, naïve, Москва, 東京, emoji 🙂👍", "def f(x):\n    return x**2  # comment",
             "What's the capital? It's 'Canberra' — isn't it?", "don't I'll we've they're SHE'S"]
    tv["tokenizer"] = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in texts]
    for i in TEST_PROMPTS + GEN_PROMPTS:
        tv["prompts"].append(chat[i]["prompt"])
        tv["chat_ids"].append(chat[i]["ids"][:chat[i]["start"]])       # = apply_chat_template(add_generation_prompt)
    json.dump(tv, open(os.path.join(ROOT, "phase2", "demo_testvec.json"), "w"))
    print("public text check:", public_clean(public_files()), "files clean")
    print("all done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
