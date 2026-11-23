"""GPU (Apple Metal via torch.mps.compile_shader) decoder for rans16 sections.

One GPU thread per independent rans16 chunk (2^16 symbols): every chunk of every tensor of a file
is decoded in ONE kernel dispatch straight into an int16 symbol buffer on the GPU, then the usual
reconstruction (Q [+ Q_ref]) * step -> fp16 runs on the GPU as well.  Host work = parsing the
section headers, building the per-table slot->symbol lookup tables and handing the compressed words
to the GPU (unified memory copy); all of it is inside the measured time.
"""
import numpy as np
import torch

from .ans import open_section
from .rans16 import Rans16Section, PB

_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void rans16_decode(device const ushort* words [[buffer(0)]],
                          device const uint*   cinfo [[buffer(1)]],
                          device const ushort* lut   [[buffer(2)]],
                          device const uint*   fc    [[buffer(3)]],
                          device const uint*   toff  [[buffer(4)]],
                          device const int*    smin  [[buffer(5)]],
                          device short*        out   [[buffer(6)]],
                          uint c [[thread_position_in_grid]]) {
    uint b = c * 5;
    uint p = cinfo[b + 0];
    uint st = cinfo[b + 1];
    uint o = cinfo[b + 2];
    uint n = cinfo[b + 3];
    uint t = cinfo[b + 4];
    device const ushort* L = lut + t * 32768u;
    uint fo = toff[t];
    int sm = smin[t];
    for (uint i = 0; i < n; i++) {
        uint slot = st & 32767u;
        uint s = L[slot];
        uint v = fc[fo + s];
        st = (v & 0xFFFFu) * (st >> 15) + slot - (v >> 16);
        if (st < 65536u) { st = (st << 16) | words[p]; p++; }
        out[o + i] = (short)((int)s + sm);
    }
}
"""
_LIB = {}


def _lib():
    if "lib" not in _LIB:
        _LIB["lib"] = torch.mps.compile_shader(_SRC)
    return _LIB["lib"]


def gpu_entropy_decode(rd):
    """Decode every rans16 'sym' section of a container on the GPU.
    Returns {record name: int16 MPS tensor (flat)}."""
    secs = []
    for r in rd.records:
        if "sym" in r["sec"] and not r["params"].get("ctx"):
            s = open_section(rd.section(r, "sym"))
            if s.n and (s.smin < -32768 or s.smin + s.nalpha - 1 > 32767):
                raise ValueError(f"{r['name']}: symbols outside int16 (GPU output is int16) -- invalid file")
            if isinstance(s, Rans16Section):
                secs.append((r["name"], s))
    if not secs:
        return {}
    words, cinfo, luts, fcs, toff, smins = [], [], [], [], [], []
    wbase = obase = fbase = 0
    consts = {}
    out_off = {}
    for t, (name, s) in enumerate(secs):
        out_off[name] = (obase, s.n)
        if s.nalpha == 1:
            consts[name] = s.smin
        freq = s.freq.astype(np.uint32)
        cum = np.concatenate([[0], np.cumsum(freq)[:-1]]).astype(np.uint32)
        lut = np.zeros(1 << PB, dtype=np.uint16)
        if s.nalpha > 1:
            lut = np.repeat(np.arange(s.nalpha, dtype=np.uint16), freq.astype(np.int64))
        luts.append(lut)
        fcs.append((freq & 0xFFFF) | (cum << 16))
        toff.append(fbase); fbase += s.nalpha
        smins.append(s.smin)
        if s.nalpha > 1:
            woff = np.concatenate([[0], np.cumsum(s.nws.astype(np.int64))[:-1]]) + wbase
            for c in range(s.nch):
                cinfo.append((woff[c], s.states[c], obase + c * s.chunk, min(s.chunk, s.n - c * s.chunk), t))
            words.append(s.words)
            wbase += s.words.size
        obase += s.n
    dev = "mps"
    W = torch.from_numpy(np.concatenate(words).view(np.int16)).to(dev)
    CI = torch.from_numpy(np.asarray(cinfo, dtype=np.uint64).astype(np.uint32).view(np.int32).reshape(-1)).to(dev)
    LUT = torch.from_numpy(np.concatenate(luts).view(np.int16)).to(dev)
    FC = torch.from_numpy(np.concatenate(fcs).astype(np.uint32).view(np.int32)).to(dev)
    TO = torch.from_numpy(np.asarray(toff, dtype=np.int32)).to(dev)
    SM = torch.from_numpy(np.asarray(smins, dtype=np.int32)).to(dev)
    OUT = torch.empty(obase, dtype=torch.int16, device=dev)
    nthreads = len(cinfo)
    if nthreads:
        _lib().rans16_decode(W, CI, LUT, FC, TO, SM, OUT, threads=nthreads)
    for name, v in consts.items():
        a, n = out_off[name]
        OUT[a:a + n] = v
    return {name: OUT[a:a + n] for name, (a, n) in out_off.items()}


def decode_gpu(rd, ref=None, timing=None, keep_state=True):
    """Full decode on the GPU (entropy decode + reconstruction); same output as codecs.decode."""
    import time
    from .codecs import decode_record
    tim = timing if timing is not None else {}
    t0 = time.perf_counter()
    syms = gpu_entropy_decode(rd)
    torch.mps.synchronize()
    tim["entropy_gpu"] = tim.get("entropy_gpu", 0.0) + time.perf_counter() - t0
    out = {}
    for rec in rd.records:
        R = ref.get(rec["name"]) if ref is not None else None
        out[rec["name"]] = decode_record(rd, rec, R, "mps", tim, keep_state, pre_sym=syms.get(rec["name"]))
    return out
