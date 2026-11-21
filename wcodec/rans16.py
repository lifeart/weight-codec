"""wcodec's own rANS ("rans16"): byte-oriented-style rANS with a 32-bit state, 16-bit word I/O and
15-bit quantized frequencies, compiled with numba.  Chunks (default 2^16 symbols) are independent
streams, so decoding parallelises over chunks across all cores (numba prange) -- the layout a
WASM / Metal / CUDA decoder would use.  Real bytes: everything below is written to the file.

Section layout (little endian):
    u64 n | (1 << 56)  (coder id 1) | i32 sym_min | u32 n_alphabet | u32 chunk | u32 n_chunks
    varint[n_alphabet] quantized frequencies (sum = 2^15)
    u32[n_chunks] final encoder states | u32[n_chunks] words per chunk | u16[...] words
"""
import struct

import numpy as np
import numba as nb

from .ans import varint_encode, varint_decode

PB = 15
M = 1 << PB
L = 1 << 16
CODER_ID = 1
CHUNK = 1 << 16
_HDR = struct.Struct("<QiIII")


def quantize_freqs(counts):
    """Counts -> integer frequencies summing to 2^15, every present symbol >= 1 (largest-remainder
    with a fix-up that takes from the largest bins)."""
    c = counts.astype(np.float64)
    tot = c.sum()
    f = np.floor(c * M / tot).astype(np.int64)
    f[(counts > 0) & (f == 0)] = 1
    diff = M - f.sum()
    order = np.argsort(-(c * M / tot - np.floor(c * M / tot)))
    i = 0
    while diff > 0:
        k = order[i % len(order)]
        if counts[k] > 0:
            f[k] += 1
            diff -= 1
        i += 1
    while diff < 0:
        k = int(np.argmax(f))
        f[k] -= 1
        diff += 1
    assert f.sum() == M and (f[counts > 0] > 0).all()
    return f.astype(np.uint32)


@nb.njit(cache=True)
def _enc_chunk(x, freq, cum, out):
    """Encode symbols x (indices) in reverse; writes words into out (reverse order).
    Returns (n_words, final_state)."""
    st = np.uint32(L)
    nw = 0
    for i in range(x.size - 1, -1, -1):
        s = x[i]
        f = freq[s]
        xmax = np.uint32(((L >> PB) << 16) * f)
        while st >= xmax:
            out[nw] = np.uint16(st & 0xFFFF)
            nw += 1
            st = np.uint32(st >> 16)
        st = np.uint32(((st // f) << PB) + (st % f) + cum[s])
    return nw, st


@nb.njit(parallel=True, cache=True)
def _enc_all(x, freq, cum, chunk, nch, buf, nws, states):
    for c in nb.prange(nch):
        a = c * chunk
        b = min(a + chunk, x.size)
        nw, st = _enc_chunk(x[a:b], freq, cum, buf[c])
        nws[c] = nw
        states[c] = st


@nb.njit(parallel=True, cache=True)
def _dec_all(words, woff, states, lut, freq, cum, chunk, n, out, smin):
    nch = states.size
    for c in nb.prange(nch):
        st = np.uint32(states[c])
        p = woff[c]
        a = c * chunk
        b = min(a + chunk, n)
        for i in range(a, b):
            slot = st & np.uint32(M - 1)
            s = lut[slot]
            st = np.uint32(freq[s] * (st >> PB) + slot - cum[s])
            if st < L:
                st = np.uint32((st << 16) | words[p])
                p += 1
            out[i] = s + smin


def encode(sym, chunk=CHUNK):
    x = np.ascontiguousarray(np.asarray(sym).reshape(-1)).astype(np.int64)
    n = x.size
    if n == 0:
        return _HDR.pack(CODER_ID << 56, 0, 0, chunk, 0)
    smin = int(x.min())
    xs = (x - smin).astype(np.int32)
    counts = np.bincount(xs).astype(np.uint64)
    nalpha = counts.size
    freq = quantize_freqs(counts)
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]]).astype(np.uint32)
    head = varint_encode(freq)
    if nalpha == 1:
        return _HDR.pack(n | (CODER_ID << 56), smin, nalpha, chunk, 0) + head
    nch = (n + chunk - 1) // chunk
    buf = np.zeros((nch, chunk + 4), dtype=np.uint16)   # <= 1 word per symbol + flush slack
    nws = np.zeros(nch, dtype=np.int64)
    states = np.zeros(nch, dtype=np.uint32)
    _enc_all(xs, freq, cum, chunk, nch, buf, nws, states)
    words = b"".join(buf[c, :nws[c]][::-1].tobytes() for c in range(nch))   # forward read order
    return (_HDR.pack(n | (CODER_ID << 56), smin, nalpha, chunk, nch) + head +
            states.astype("<u4").tobytes() + nws.astype("<u4").tobytes() + words)


class Rans16Section:
    def __init__(self, buf, off=0):
        n, smin, nalpha, chunk, nch = _HDR.unpack_from(buf, off)
        self.n = n & ((1 << 56) - 1)
        self.smin, self.nalpha, self.chunk, self.nch = smin, nalpha, chunk, nch
        pos = off + _HDR.size
        self.freq, pos = varint_decode(buf, nalpha, pos)
        self.freq = self.freq.astype(np.uint32)
        self.states = np.frombuffer(buf, dtype="<u4", count=nch, offset=pos); pos += 4 * nch
        self.nws = np.frombuffer(buf, dtype="<u4", count=nch, offset=pos); pos += 4 * nch
        tot = int(self.nws.sum()) if nch else 0
        self.words = np.frombuffer(buf, dtype="<u2", count=tot, offset=pos)

    def decode(self, out_dtype=np.int32):
        if self.n == 0:
            return np.zeros(0, dtype=out_dtype)
        if self.nalpha == 1:
            return np.full(self.n, self.smin, dtype=out_dtype)
        cum = np.concatenate([[0], np.cumsum(self.freq)[:-1]]).astype(np.uint32)
        lut = np.repeat(np.arange(self.nalpha, dtype=np.int32), self.freq.astype(np.int64))
        woff = np.concatenate([[0], np.cumsum(self.nws.astype(np.int64))[:-1]])
        out = np.empty(self.n, dtype=out_dtype)
        _dec_all(self.words, woff, self.states.astype(np.uint32), lut, self.freq, cum, self.chunk, self.n, out,
                 np.int32(self.smin))
        return out


@nb.njit(cache=True)
def _merge_ll16(hi, flat, off, out):
    """lo bytes were coded per class c = hi >> 4 in raster order: scatter them back in one pass and
    assemble the 16-bit pattern (hi << 8) | lo."""
    k = off.copy()
    for i in range(hi.size):
        c = hi[i] >> 4
        out[i] = np.uint16((hi[i] << 8) | flat[k[c]])
        k[c] += 1


def merge_ll16(hi, parts):
    """hi: int array; parts: {class: lo array}. Returns uint16 bit patterns."""
    flat = np.concatenate([parts[c] if c in parts else np.zeros(0, np.int32) for c in range(16)]).astype(np.int32)
    sizes = np.array([parts[c].size if c in parts else 0 for c in range(16)], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    out = np.empty(hi.size, dtype=np.uint16)
    _merge_ll16(hi.astype(np.int32), flat, off, out)
    return out
