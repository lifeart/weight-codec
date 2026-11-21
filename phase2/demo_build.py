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
quality (chat-KL etc.) is copied from results/phase2_family_smol135.json (measured on models decoded from the
very same constriction files these rans16 twins are a lossless transcode of).

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

FILES = {   # name -> (role, model, lossless?, description)
    "B_rest_ll16": ("base", "base", True, "base embeddings + norms, lossless fp16 (ll16)"),
    "B_ecsq_c0.35": ("base", "base", False, "base linears, I-frame: GPTQ-ECSQ c=0.35 (int4-class)"),
    "B_lin_ll16": ("base", "base", True, "base linears, lossless fp16 (ll16)"),
    "T_rest_xor_base": ("delta", "instruct", True, "Instruct embeddings + norms, lossless, XOR with the base"),
    "T_pframe_c0.35": ("delta", "instruct", False, "Instruct linears, P-frame on the base I-frame lattice (c=0.35)"),
    "T_delta_c0.12": ("delta", "instruct", False, "Instruct linears, full-rank delta over the fp16 base (c=0.12)"),
    "T_rest_ll16": ("standalone", "instruct", True, "Instruct embeddings + norms, lossless, no base"),
    "T_ecsq_c0.35": ("standalone", "instruct", False, "Instruct linears, standalone GPTQ-ECSQ c=0.35 (no base)"),
}
CHAINS = {
    "delta": {"title": "fp16 base + full-rank Δ (c=0.12)", "short": "near-lossless",
              "groups": [["B_rest_ll16", "T_rest_xor_base"], ["B_lin_ll16", "T_delta_c0.12"]],
              "quality_file": "T_delta_c0.12", "base_lossless": True},
    "family": {"title": "int4-class family: base I-frame + Instruct P-frame (c=0.35)", "short": "smallest",
               "groups": [["B_rest_ll16", "T_rest_xor_base"], ["B_ecsq_c0.35", "T_pframe_c0.35"]],
               "quality_file": "T_pframe_c0.35", "base_lossless": False},
    "indep": {"title": "control: standalone Instruct GPTQ-ECSQ c=0.35 (no base reuse)", "short": "control",
              "groups": [["T_rest_ll16"], ["T_ecsq_c0.35"]], "quality_file": "T_ecsq_c0.35", "base_lossless": None},
}
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
        return H(st.Q.cpu().numpy().astype("<i2").tobytes())
    return H(bits16(st.W))


def build_stage(stage):
    os.makedirs(stage, exist_ok=True)
    for nm in FILES:
        dst = os.path.join(stage, nm + ".wcdc")
        if not os.path.exists(dst):
            transcode(os.path.join(SRC, nm + ".wcdc"), dst)


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


def quality(nm):
    F = json.load(open(os.path.join(RES, "phase2_family_smol135.json")))
    r = F.get(nm, {})
    return {k: r.get(k) for k in ("chat_kl", "chat_top1", "kl", "top1", "ppl", "real_bpw")}


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
    """Small JSON + figures the page shows (all measured offline, REPORT §14): docs/data/results.json,
    docs/data/summary.json (= the phase-2 tables of the former static Space), docs/figures/*.png."""
    S = json.load(open(os.path.join(RES, "phase2_summary.json")))
    F = json.load(open(os.path.join(RES, "phase2_family_smol135.json")))
    pairs = [{k: p.get(k) for k in ("scenario", "files", "pair_MB", "base_ppl", "ft_chat_kl", "ft_wiki_kl", "note")}
             for p in S["pairs"] if p["tag"] == "smol135"]
    fmts = []
    for r in S["formats"]:
        if r["tag"] == "smol135" and r["role"] == "fine-tune":
            fmts.append({k: r.get(k) for k in ("format", "MB", "real_bpw", "chat_kl", "kl", "ppl", "dec_cpu_constriction_1t_ms",
                                                "dec_cpu_rans16_par_ms", "dec_gpu_ms")})
    files = {k: {f: F[k].get(f) for f in ("MB", "real_bpw", "chat_kl", "chat_top1", "kl", "ppl") if f in F[k]}
             for k in F if k.startswith(("B_", "T_"))}
    out = {"source": "results/phase2_summary.json, results/phase2_family_smol135.json (REPORT §14.3)",
           "pairs": pairs, "finetune_formats": fmts, "files": files,
           "chat_set": "48 fixed prompts; chat-KL = KL(original fp16 Instruct || candidate) on the original's greedy answers",
           "delta_vs_lora": [["T_delta_c0.25", "T_lowrank_r32"], ["T_delta_c0.06", "T_lowrank_r64"], ["T_delta_c0.06", "T_lowrank_r128"]]}
    os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(DOCS, "data", "results.json"), "w"), indent=1)
    SS = json.load(open(os.path.join(RES, "phase2_space_summary.json")))
    SS.pop("demo", None)
    json.dump(SS, open(os.path.join(DOCS, "data", "summary.json"), "w"), indent=1, ensure_ascii=False)
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
                         quality=quality(ch["quality_file"]))
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
    print("all done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
