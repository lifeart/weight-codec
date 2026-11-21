"""Tensor-level encoders / decoders on top of the WCDC container.

Record kinds
------------
raw16    fp16 bytes as-is.
ll16     LOSSLESS fp16: 16-bit pattern (optionally XOR the reference tensor's pattern = lossless
         P-frame) split into hi byte (one ANS table) and lo byte (ANS, 16 tables conditioned on
         hi>>4 = sign + top exponent bits).
qrow     symbols on a per-row uniform grid (ECSQ / GPTQ-ECSQ / P-frames / SVC layers):
           params.step : "own"   -> section "step" (fp16 per output row)
                         "own_div:a,b" -> stored step divided by a then b (same lattice as SVC)
                         "ref"   -> step of the reference record (same lattice, 0 side info)
                         "div:k" -> reference step / k (SVC nested layer)
           params.acc  : "none"  -> Q = q
                         "add"   -> Q = Q_ref + q        (closed-loop P-frame on a shared lattice)
                         "nest:k"-> Q = k*Q_ref + q      (SVC refinement, nested lattice)
           params.base : "fp16ref" -> W = W_ref + q*step (delta against an fp16 reference)
           params.ctx  : "clip3" -> symbols coded in 7 context streams, ctx = clip(q_ref_last, -3, 3)
           W = Q * step  (fp32 math, then fp16)
g128     GPTQ/RTN asymmetric min-max grid, groups of `group` along the input dim:
           sections "scale" (fp16 [O, I/g]), "zero" (packed 4-bit, or bytes if bits > 4),
           "sym" (ANS) or "packed" (fixed-rate nibbles: the standard int4 file layout)
lowrank  W = W_ref + A @ B with fp16 factors (LoRA-style delta).

Decoder state per tensor (TState): W (fp16 torch), Q (int32 cumulative lattice index),
q_last (last layer's symbols), step (fp32 per row).
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .ans import ans_encode, AnsSection, open_section
from .container import Writer, Reader


@dataclass
class TState:
    W: Optional[torch.Tensor] = None
    Q: Optional[torch.Tensor] = None
    q_last: Optional[torch.Tensor] = None
    step: Optional[torch.Tensor] = None


def _np16(t):
    return t.detach().to("cpu", torch.float16).contiguous().numpy()


def _bits16(t):
    return _np16(t).view(np.uint16)


# ------------------------------------------------------------------------------------------
# encoders  (each returns kind, params, sections)
# ------------------------------------------------------------------------------------------
def enc_raw16(W):
    return "raw16", {}, {"d": _np16(W).tobytes()}


def enc_ll16(W, ref=None):
    b = _bits16(W).reshape(-1)
    if ref is not None:
        b = b ^ _bits16(ref).reshape(-1)
    hi = (b >> 8).astype(np.int32)
    lo = (b & 0xFF).astype(np.int32)
    cls = hi >> 4
    sec = {"hi": ans_encode(hi)}
    present = [int(c) for c in np.unique(cls)]
    for c in present:
        sec[f"lo{c}"] = ans_encode(lo[cls == c])
    return "ll16", {"xor": ref is not None, "cls": present}, sec


def _sym_sections(q, ctx=None):
    q = np.asarray(q).reshape(-1)
    if ctx is None:
        return {"sym": ans_encode(q)}, None
    ctx = np.clip(np.asarray(ctx).reshape(-1), -3, 3)
    present = [int(v) for v in np.unique(ctx)]
    return {f"sym@{v}": ans_encode(q[ctx == v]) for v in present}, present


def enc_qrow(q, step=None, step_mode="own", acc="none", base="none", ctx_prev=None):
    """q: int symbols [O, I]; step: fp32/fp16 [O] or [O,1] (stored as fp16) when step_mode=='own'."""
    params = {"step": step_mode, "acc": acc, "base": base}
    sec, present = _sym_sections(q, ctx_prev)
    if present is not None:
        params["ctx"] = "clip3"
        params["ctx_vals"] = present
    if step_mode.startswith("own"):
        sec["step"] = _np16(torch.as_tensor(step).reshape(-1)).tobytes()
    return "qrow", params, sec


def _pack_nibbles(x):
    x = np.asarray(x, dtype=np.uint8).reshape(-1)
    if x.size % 2:
        x = np.concatenate([x, np.zeros(1, np.uint8)])
    return (x[0::2] | (x[1::2] << 4)).tobytes()


def _unpack_nibbles(buf, n):
    b = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 0x0F
    out[1::2] = b >> 4
    return out[:n]


def enc_g128(q, scale, zero, bits=4, group=128, packed=False):
    q = np.asarray(q)
    sec = {"scale": _np16(torch.as_tensor(scale)).tobytes()}
    z = np.asarray(zero).astype(np.uint8)
    sec["zero"] = _pack_nibbles(z) if bits <= 4 else z.tobytes()
    if packed:
        assert bits == 4
        sec["packed"] = _pack_nibbles(q.astype(np.uint8))
    else:
        sec["sym"] = ans_encode(q)
    return "g128", {"bits": bits, "group": group, "zshape": list(z.shape),
                    "coding": "packed" if packed else "ans"}, sec


def enc_lowrank(A, B):
    return "lowrank", {"r": int(A.shape[1])}, {"A": _np16(A).tobytes(), "B": _np16(B).tobytes()}


# ------------------------------------------------------------------------------------------
# decoder
# ------------------------------------------------------------------------------------------
def _sync(dev):
    if str(dev).startswith("mps"):
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def _syms(rd, rec, n, ctx=None):
    p = rec["params"]
    if p.get("ctx") == "clip3":
        out = np.empty(n, dtype=np.int32)
        c = np.clip(ctx.reshape(-1), -3, 3)
        for v in p["ctx_vals"]:
            out[c == v] = open_section(rd.buf, _sec_off(rd, rec, f"sym@{v}")).decode()
        return out
    return open_section(rd.buf, _sec_off(rd, rec, "sym")).decode()


def _sec_off(rd, rec, key):
    return rd.base + rec["sec"][key][0]


def decode_record(rd: Reader, rec, R=None, device="cpu", tim=None, keep_state=True, pre_sym=None):
    """Decode one record (dict from rd.records, or its name) given its reference TState R.
    pre_sym: symbols already entropy-decoded elsewhere (parallel decoder) for the 'sym' section."""
    if isinstance(rec, str):
        rec = rd.by_name[rec]
    tim = tim if tim is not None else {}
    name, kind, shape, p = rec["name"], rec["kind"], rec["shape"], rec["params"]
    n = int(np.prod(shape))
    st = TState()
    t0 = time.perf_counter()
    if kind == "raw16":
        arr = np.frombuffer(rd.section(rec, "d"), dtype=np.float16).reshape(shape)
        t1 = time.perf_counter()
        st.W = torch.from_numpy(arr.copy()).to(device)
    elif kind == "ll16":
        hi = open_section(rd.buf, _sec_off(rd, rec, "hi")).decode()
        parts = {c: open_section(rd.buf, _sec_off(rd, rec, f"lo{c}")).decode() for c in p["cls"]}
        try:
            from .rans16 import merge_ll16            # one-pass numba scatter
            b = merge_ll16(hi, parts)
        except ImportError:                           # no numba: numpy masks (slower, same result)
            cls = hi >> 4
            lo = np.empty(n, dtype=np.int32)
            for c in p["cls"]:
                lo[cls == c] = parts[c]
            b = ((hi << 8) | lo).astype(np.uint16)
        if p["xor"]:
            b ^= R.W.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        t1 = time.perf_counter()
        st.W = torch.from_numpy(b.view(np.float16).reshape(shape)).to(device)
    elif kind == "qrow":
        if torch.is_tensor(pre_sym):                     # already decoded on the GPU
            t1 = time.perf_counter()
            qt = pre_sym.reshape(shape).to(device).to(torch.int32)
        else:
            if pre_sym is not None:
                q = pre_sym
            else:
                q = _syms(rd, rec, n, None if R is None or R.q_last is None else R.q_last.cpu().numpy())
            q = q.astype(np.int8 if (q.min() >= -128 and q.max() <= 127) else np.int16, copy=False).reshape(shape)
            t1 = time.perf_counter()
            qt = torch.from_numpy(q).to(device).to(torch.int32)
        if p["step"].startswith("own"):
            step = torch.from_numpy(np.frombuffer(rd.section(rec, "step"), dtype=np.float16).copy()
                                    ).to(device).float().reshape(-1, 1)
            if p["step"].startswith("own_div:"):          # e.g. own_div:3,3 -> (s/3)/3 (sequential)
                for d in p["step"].split(":")[1].split(","):
                    step = step / float(d)
        elif p["step"] == "ref":
            step = R.step.to(device)
        else:
            step = R.step.to(device) / float(p["step"].split(":")[1])
        if p["acc"] == "none":
            Qi = qt
        elif p["acc"] == "add":
            Qi = R.Q.to(device).to(torch.int32) + qt
        else:
            Qi = int(p["acc"].split(":")[1]) * R.Q.to(device).to(torch.int32) + qt
        Wf = Qi.float() * step
        if p["base"] == "fp16ref":
            Wf = R.W.to(device).float() + Wf
        st.W = Wf.half()
        if keep_state is True:
            st.Q, st.q_last, st.step = Qi, qt, step
        elif keep_state == "Q":                # enough to be a P-frame reference (no context)
            st.Q, st.step = Qi, step
    elif kind == "g128":
        O, I = shape
        g = p["group"]
        if p["coding"] == "packed":
            q = _unpack_nibbles(rd.section(rec, "packed"), n)
        elif torch.is_tensor(pre_sym):
            q = None
        elif pre_sym is not None:
            q = pre_sym.astype(np.uint8)
        else:
            q = open_section(rd.buf, _sec_off(rd, rec, "sym")).decode().astype(np.uint8)
        zn = int(np.prod(p["zshape"]))
        z = _unpack_nibbles(rd.section(rec, "zero"), zn) if p["bits"] <= 4 else \
            np.frombuffer(rd.section(rec, "zero"), dtype=np.uint8)
        sc = np.frombuffer(rd.section(rec, "scale"), dtype=np.float16)
        t1 = time.perf_counter()
        qt = (pre_sym.reshape(O, I // g, g).to(device).float() if q is None else
              torch.from_numpy(q.reshape(O, I // g, g)).to(device).float())
        zt = torch.from_numpy(z.reshape(O, I // g, 1).copy()).to(device).float()
        st_ = torch.from_numpy(sc.reshape(O, I // g, 1).copy()).to(device).float()
        st.W = (st_ * (qt - zt)).reshape(O, I).half()
    elif kind == "lowrank":
        O, I = shape
        r = p["r"]
        A = np.frombuffer(rd.section(rec, "A"), dtype=np.float16).reshape(O, r)
        B = np.frombuffer(rd.section(rec, "B"), dtype=np.float16).reshape(r, I)
        t1 = time.perf_counter()
        At = torch.from_numpy(A.copy()).to(device).float()
        Bt = torch.from_numpy(B.copy()).to(device).float()
        st.W = (R.W.to(device).float() + At @ Bt).half()
    else:
        raise ValueError(kind)
    _sync(device)
    t2 = time.perf_counter()
    tim["entropy"] = tim.get("entropy", 0.0) + t1 - t0
    tim["reconstruct"] = tim.get("reconstruct", 0.0) + t2 - t1
    return st


def decode(rd: Reader, ref: Optional[dict] = None, device="cpu", timing=None, keep_state=True):
    """Decode every record of a container.  ref: {name: TState} of the reference stream
    (I-frame / previous frame / previous SVC layer / fp16 base).  Returns {name: TState}.
    timing (dict) accumulates seconds: 'entropy' (CPU ANS / unpack / parse) and 'reconstruct'
    (dequantization + reference add on `device`, synchronized)."""
    tim = timing if timing is not None else {}
    out = {}
    for rec in rd.records:
        R = ref.get(rec["name"]) if ref is not None else None
        out[rec["name"]] = decode_record(rd, rec, R, device, tim, keep_state)
    return out


def fp16_state(sd, device="cpu"):
    """Wrap an fp16 state dict {name: tensor} as reference TStates (for 'fp16ref' / lowrank)."""
    return {k: TState(W=v.to(device, torch.float16)) for k, v in sd.items()}
