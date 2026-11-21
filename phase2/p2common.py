"""Phase-2 helpers: write linears / rest of a model into WCDC files, account REAL bytes, decode a
file back into a live model.  Re-uses the phase-1 measurement spine (scripts/common.py)."""
import os
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from common import *  # noqa: F401,F403  (phase-1 spine: eval_lm, eval_chat, block_linears, ...)
from wcodec import (Writer, Reader, TState, decode, decode_record, enc_raw16, enc_ll16, enc_qrow, enc_g128,
                    enc_lowrank, order0_bits, ref_coders, small_int_bytes)
from wcodec import quant as Q

BITS = os.path.join(ROOT, "bitstreams")
os.makedirs(BITS, exist_ok=True)


def lin_key(n):
    """phase-1 linear name 'layers.3.mlp.up_proj' -> state-dict key."""
    return n


def rest_state(model):
    """Every parameter / buffer of the state dict that is NOT a block-linear weight (embeddings,
    norms, biases, untied head).  Tied lm_head is skipped (stored once)."""
    lin = {f"{model.base_model_prefix}.{n}.weight" for n, _ in block_linears(model)}
    sd = model.state_dict()
    tied = getattr(model.config, "tie_word_embeddings", False)
    out = {}
    for k, v in sd.items():
        if k in lin:
            continue
        if tied and k == "lm_head.weight":
            continue
        if not torch.is_floating_point(v):
            continue
        out[k] = v.detach().to("cpu", torch.float16)
    return out


class Acc:
    """Collects the per-file accounting: real bytes, phase-1 entropy proxy, reference coders."""

    def __init__(self, refcoders=False):
        self.proxy_bits = 0.0
        self.side_bits = 0.0
        self.n = 0
        self.ref = {}
        self.refcoders = refcoders
        self.sym_bytes = []

    def add(self, sym, side_bits, n):
        self.n += n
        self.side_bits += side_bits
        if sym is not None:
            self.proxy_bits += order0_bits(sym)
            if self.refcoders:
                self.sym_bytes.append(small_int_bytes(sym))

    def finish(self):
        out = {"n_weights": self.n, "proxy_bpw": (self.proxy_bits + self.side_bits) / max(1, self.n),
               "side_bpw": self.side_bits / max(1, self.n)}
        if self.refcoders and self.sym_bytes:
            raw = b"".join(self.sym_bytes)
            t0 = time.time()
            rc = ref_coders(raw, ("zstd3", "zstd19", "gzip9"))
            out["refcoder_s"] = round(time.time() - t0, 1)
            # reference coder bpw: compressed symbols + the SAME side info as raw bytes
            out["refcoder_bpw"] = {k: (8 * v + self.side_bits) / self.n for k, v in rc.items()}
            out["raw_sym_bytes"] = len(raw)
        return out


def finish_file(wr, path, acc, extra=None):
    nb = wr.write(path)
    wr.blobs = []          # free the in-memory sections as soon as they are on disk
    rec = {"file": os.path.relpath(path, ROOT), "bytes": nb, "MB": nb / 1e6,
           "real_bpw": 8 * nb / max(1, acc.n), **acc.finish(), **(extra or {})}
    rd = Reader(path)
    rec["bytes_header"] = rd.header_bytes
    rec["bytes_by_section"] = rd.section_bytes()
    return rec


def write_rest(model_rest, path, mode="ll16", ref=None):
    """Rest tensors (lossless).  mode raw16 | ll16 ; ref: {key: fp16 tensor} for an XOR P-frame."""
    wr = Writer({"what": "rest", "mode": mode, "xor_ref": ref is not None})
    n = 0
    for k, v in model_rest.items():
        if mode == "raw16":
            kind, p, s = enc_raw16(v)
        else:
            kind, p, s = enc_ll16(v, None if ref is None else ref[k])
        wr.add(k, kind, v.shape, p, s)
        n += v.numel()
    nb = wr.write(path)
    return {"file": os.path.relpath(path, ROOT), "bytes": nb, "MB": nb / 1e6, "n_params": n,
            "bpp": 8 * nb / n}


def decode_file(path, ref=None, device="cpu", keep_state=True):
    rd = Reader(path)
    tim = {}
    t0 = time.perf_counter()
    st = decode(rd, ref=ref, device=device, timing=tim, keep_state=keep_state)
    tim["total"] = time.perf_counter() - t0
    return st, tim, rd


def load_into(model, states):
    """Write decoded linears into the live model (names = phase-1 'layers.i.xxx')."""
    lin = dict(block_linears(model))
    for n, st in states.items():
        m = lin[n]
        m.weight.data.copy_(st.W.to(m.weight.device, m.weight.dtype))


def states_to_cpu(states):
    out = {}
    for k, s in states.items():
        out[k] = TState(W=s.W.cpu() if s.W is not None else None, Q=None if s.Q is None else s.Q.cpu(),
                        q_last=None if s.q_last is None else s.q_last.cpu(),
                        step=None if s.step is None else s.step.cpu())
    return out


def job_sync(path):
    """Inside an HF Job: push a results file to the private dataset repo right away, so a timeout
    never loses finished configs.  No-op outside jobs.  Failures are logged, not hidden."""
    run = os.environ.get("WC_RUN")
    if not run:
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"runs/{run}/results/{os.path.basename(path)}",
                            repo_id=os.environ.get("WC_REPO", "lifeart/weight-codec-phase2"), repo_type="dataset",
                            commit_message=f"{run}: partial {os.path.basename(path)}")
    except Exception as e:
        log(f"job_sync failed for {path}: {type(e).__name__}: {str(e)[:200]}")
