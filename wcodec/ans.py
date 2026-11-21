"""Entropy-coding back ends for integer symbol streams.

Every function here produces / consumes REAL BYTES.  Two coders:

* ``constriction`` rANS (``constriction.stream.stack.AnsCoder``: 64-bit state, 32-bit words,
  24-bit probability precision) with a static per-stream ``Categorical`` model.  The model is
  rebuilt by the decoder from the stored symbol counts, so the counts ARE the side info and
  are paid for in the file.  The stream is cut into independent chunks (default 2^20 symbols)
  so that chunks can be decoded in parallel (constriction itself holds the GIL).
* reference general-purpose coders (zstd / gzip / lzma) applied to the same symbols stored as
  int8/int16 bytes -- what you get "for free" without a model.

Serialized ANS section layout (little endian):
    u64 n_symbols | i32 sym_min | u32 n_alphabet | u32 chunk | u32 n_chunks
    varint[n_alphabet] counts            (dense over [sym_min, sym_min + n_alphabet))
    u32[n_chunks] words_per_chunk
    u32[...] words
"""
import struct
import zlib
import lzma

import numpy as np
import constriction

CHUNK = 1 << 20
_HDR = struct.Struct("<QiIII")


# ----------------------------------------------------------------------------------------------
# varints (LEB128, unsigned)
# ----------------------------------------------------------------------------------------------
def varint_encode(vals):
    out = bytearray()
    for v in np.asarray(vals, dtype=np.uint64).tolist():
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def varint_decode(buf, n, pos=0):
    vals = np.empty(n, dtype=np.uint64)
    for i in range(n):
        shift = 0
        v = 0
        while True:
            b = buf[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        vals[i] = v
    return vals, pos


# ----------------------------------------------------------------------------------------------
# constriction rANS
# ----------------------------------------------------------------------------------------------
def _model(counts):
    p = counts.astype(np.float64)
    return constriction.stream.model.Categorical(p / p.sum(), perfect=False)


def ans_encode(sym, chunk=CHUNK):
    """sym: integer array (any shape) -> bytes of one self-contained ANS section."""
    x = np.ascontiguousarray(np.asarray(sym).reshape(-1)).astype(np.int64)
    n = x.size
    if n == 0:
        return _HDR.pack(0, 0, 0, chunk, 0)
    smin = int(x.min())
    xs = (x - smin).astype(np.int32)
    counts = np.bincount(xs).astype(np.uint64)
    nalpha = counts.size
    parts = [_HDR.pack(n, smin, nalpha, chunk, 0), varint_encode(counts)]
    if nalpha == 1:  # constant stream: the table says it all
        return parts[0] + parts[1]
    model = _model(counts)
    words, lens = [], []
    for a in range(0, n, chunk):
        c = constriction.stream.stack.AnsCoder()
        c.encode_reverse(xs[a:a + chunk], model)
        w = c.get_compressed()
        words.append(w.astype("<u4", copy=False))
        lens.append(w.size)
    parts[0] = _HDR.pack(n, smin, nalpha, chunk, len(lens))
    parts.append(np.asarray(lens, dtype="<u4").tobytes())
    parts.extend(w.tobytes() for w in words)
    return b"".join(parts)


class AnsSection:
    """Parsed ANS section (zero-copy views into the container buffer)."""

    def __init__(self, buf, off=0):
        n, smin, nalpha, chunk, nch = _HDR.unpack_from(buf, off)
        self.n, self.smin, self.nalpha, self.chunk, self.nch = n, smin, nalpha, chunk, nch
        pos = off + _HDR.size
        self.counts, pos = varint_decode(buf, nalpha, pos)
        self.lens = np.frombuffer(buf, dtype="<u4", count=nch, offset=pos)
        pos += 4 * nch
        self.words_off = pos
        self.buf = buf
        self.starts = np.concatenate([[0], np.cumsum(self.lens, dtype=np.int64)])

    def model(self):
        return _model(self.counts)

    def chunk_words(self, i):
        a = self.words_off + 4 * int(self.starts[i])
        return np.frombuffer(self.buf, dtype="<u4", count=int(self.lens[i]), offset=a)

    def chunk_len(self, i):
        return min(self.chunk, self.n - i * self.chunk)

    def decode_chunk(self, i, model=None):
        model = model or self.model()
        c = constriction.stream.stack.AnsCoder(self.chunk_words(i).astype(np.uint32))
        return c.decode(model, self.chunk_len(i))

    def decode(self, out_dtype=np.int32):
        if self.n == 0:
            return np.zeros(0, dtype=out_dtype)
        if self.nalpha == 1:
            return np.full(self.n, self.smin, dtype=out_dtype)
        m = self.model()
        out = np.empty(self.n, dtype=out_dtype)
        for i in range(self.nch):
            a = i * self.chunk
            y = self.decode_chunk(i, m)
            out[a:a + y.size] = y + self.smin if self.smin else y
        return out


def ans_decode(buf, off=0, out_dtype=np.int32):
    return AnsSection(buf, off).decode(out_dtype)


# ----------------------------------------------------------------------------------------------
# bit accounting helpers (for comparison with the phase-1 entropy proxy)
# ----------------------------------------------------------------------------------------------
def order0_bits(sym):
    """Phase-1 proxy: n*H + 32 bits per used symbol (frequency table)."""
    x = np.asarray(sym).reshape(-1).astype(np.int64)
    if x.size == 0:
        return 0.0
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum() + 32 * c.size)


def small_int_bytes(sym):
    """Symbols as the narrowest signed integer bytes (int8 if the range fits)."""
    x = np.asarray(sym).reshape(-1)
    lo, hi = int(x.min()), int(x.max())
    dt = np.int8 if (lo >= -128 and hi <= 127) else (np.int16 if (lo >= -32768 and hi <= 32767) else np.int32)
    return x.astype(dt).tobytes()


def ref_coders(raw: bytes, which=("zstd19", "zstd3", "gzip9", "lzma6")):
    """Compressed sizes (bytes) of raw bytes with general-purpose coders."""
    out = {}
    for w in which:
        if w.startswith("zstd"):
            import zstandard
            # multi-threaded zstd (all cores); level-19 single-thread is ~1-2 MB/s on 100s of MB
            out[w] = len(zstandard.ZstdCompressor(level=int(w[4:]), threads=-1).compress(raw))
        elif w == "gzip9":
            out[w] = len(zlib.compress(raw, 9))
        elif w == "lzma6":
            out[w] = len(lzma.compress(raw, preset=6))
    return out


def open_section(buf, off=0):
    """Dispatch on the coder id stored in the top byte of the first u64 (0 = constriction, 1 = rans16)."""
    n = struct.unpack_from("<Q", buf, off)[0]
    if n >> 56 == 1:
        from .rans16 import Rans16Section
        return Rans16Section(buf, off)
    return AnsSection(buf, off)


def is_entropy_section(key):
    return key == "sym" or key.startswith("sym@") or key == "hi" or (key.startswith("lo") and key[2:].isdigit())
