"""weight-codec phase 2 -- results + live decode demo (Gradio Space, CPU).

Results tab: precomputed tables / figures (assets/summary.json, assets/*.png) from the real
bitstream runs (bytes on disk, decode ms, PPL / KL).
Live demo: the Space never downloads the model weights.  It only has our compressed WCDC files
(bitstreams/<pair>/*.wcdc).  Pressing "decode" entropy-decodes them on this Space's CPU, rebuilds
the fine-tuned model (config + tokenizer are fetched from the public Hub repo), reports MB and
decode milliseconds, and lets the rebuilt model answer a prompt.
"""
import json
import os
import time

import gradio as gr
import numpy as np
import torch

from wcodec import Reader, decode

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
SUMMARY = json.load(open(os.path.join(ASSETS, "summary.json")))
DEMO = SUMMARY["demo"]
torch.set_num_threads(max(1, os.cpu_count() or 1))
_MODELS = {}


def _cfg_tok(rid):
    from transformers import AutoConfig, AutoTokenizer
    return AutoConfig.from_pretrained(rid), AutoTokenizer.from_pretrained(rid)


def _hub_base_state():
    """Public fp16 base from the Hub (no token): linears keyed like the bitstreams, rest by state-dict key."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from wcodec import TState
    t0 = time.perf_counter()
    p = hf_hub_download(DEMO["base"], "model.safetensors")
    t_dl = time.perf_counter() - t0
    sd = load_file(p)
    st = {}
    for k, v in sd.items():
        v = v.to(torch.float16)
        if k.startswith("model.layers.") and k.endswith("proj.weight"):
            st[k[len("model."):-len(".weight")]] = TState(W=v)
        else:
            st[k] = TState(W=v)
    return st, t_dl, time.perf_counter() - t0 - t_dl, os.path.getsize(p) / 1e6


def decode_format(fmt_name, progress=gr.Progress()):
    fmt = DEMO["formats"][fmt_name]
    pdir = os.path.join(HERE, DEMO["dir"])
    states, rows = {}, []
    if fmt.get("coder") == "rans16":
        from wcodec import rans16  # noqa: F401  (numba JIT happens on first use; reported below)
    t_all = time.perf_counter()
    t_hub = 0.0
    for i, (fname, ref) in enumerate(fmt["chain"]):
        progress(i / len(fmt["chain"]), desc=f"decoding {fname}")
        if fname == "@hub_base":
            states[fname], t_dl, t_ld, mb = _hub_base_state()
            t_hub = t_dl + t_ld
            rows.append([f"{DEMO['base']} (public fp16, from Hub cache/CDN)", "-", round(mb, 1), round(1e3 * t_dl, 1),
                         "—", round(1e3 * t_ld, 1), round(1e3 * (t_dl + t_ld), 1)])
            continue
        path = os.path.join(pdir, fmt.get("subdir", ""), fname + ".wcdc")
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            rd = Reader(f.read())
        t_io = time.perf_counter() - t0
        tim = {}
        t1 = time.perf_counter()
        states[fname] = decode(rd, ref=states.get(ref) if ref else None, device="cpu", timing=tim,
                               keep_state="Q")
        rows.append([fname, ref or "-", round(rd.nbytes / 1e6, 2), round(1e3 * t_io, 1),
                     round(1e3 * tim["entropy"], 1), round(1e3 * tim["reconstruct"], 1),
                     round(1e3 * (time.perf_counter() - t1), 1)])
    t_total = time.perf_counter() - t_all - t_hub
    # assemble the fine-tuned model from the decoded tensors (no fine-tune weights are downloaded)
    from transformers import AutoModelForCausalLM
    cfg, tok = _cfg_tok(DEMO["target"])
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)   # random init, overwritten below
    sd = {}
    for fname in fmt["model_files"]:
        for name, st in states[fname].items():
            key = name if not name.startswith("layers.") else f"model.{name}.weight"
            sd[key] = st.W.float()
    if getattr(cfg, "tie_word_embeddings", False):
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
    model.eval()
    _MODELS["m"], _MODELS["tok"], _MODELS["fmt"] = model, tok, fmt_name
    mb = sum(r[2] for r in rows if not str(r[0]).startswith(DEMO["base"]))
    coder = fmt.get("coder", "constriction")
    info = (f"**{fmt_name}** — **{mb:.1f} MB** of our bitstreams ({fmt.get('what', '')}); "
            f"fp16 safetensors of the fine-tune alone: {DEMO['fp16_MB']:.0f} MB. "
            f"Decoded on this Space's CPU ({os.cpu_count()} vCPU) in **{1e3 * t_total:.0f} ms** "
            f"({'constriction rANS, 1 thread' if coder == 'constriction' else 'wcodec rans16 (numba, all cores; first call includes JIT)'})"
            + (f" + {t_hub:.1f} s to fetch/load the public fp16 base" if t_hub else "") + ". "
            f"Precomputed quality of this exact file: {fmt.get('quality', '')}. "
            + (f"⚠ missing tensors: {missing[:3]}" if missing else "All tensors present."))
    return info, rows


@torch.no_grad()
def chat(prompt):
    if "m" not in _MODELS:
        return "Decode a format first."
    m, tok = _MODELS["m"], _MODELS["tok"]
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    t0 = time.perf_counter()
    out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return f"[{_MODELS['fmt']}] ({time.perf_counter() - t0:.1f}s)\n\n{txt}"


with gr.Blocks(title="weight-codec: real ANS bitstreams for LLM families") as demo:
    gr.Markdown(SUMMARY["intro_md"])
    with gr.Tab("Results"):
        for sec in SUMMARY["sections"]:
            gr.Markdown(f"### {sec['title']}\n{sec.get('md', '')}")
            if sec.get("table"):
                gr.Dataframe(value=sec["table"]["rows"], headers=sec["table"]["headers"], interactive=False,
                             wrap=True)
            for img in sec.get("images", []):
                gr.Image(os.path.join(ASSETS, img), show_label=False, interactive=False)
    with gr.Tab("Live decode demo"):
        gr.Markdown(DEMO["md"])
        fmt = gr.Dropdown(list(DEMO["formats"]), value=list(DEMO["formats"])[0], label="format")
        btn = gr.Button("Decode bitstream → rebuild fine-tune", variant="primary")
        info = gr.Markdown()
        tbl = gr.Dataframe(headers=["file", "reference", "MB", "read ms", "entropy-decode ms",
                                    "reconstruct ms", "total ms"], interactive=False)
        btn.click(decode_format, fmt, [info, tbl])
        prompt = gr.Textbox("Give me three tips for writing clear code.", label="prompt for the rebuilt model")
        go = gr.Button("Generate (greedy, 64 tokens)")
        ans = gr.Textbox(label="answer", lines=8)
        go.click(chat, prompt, ans)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
