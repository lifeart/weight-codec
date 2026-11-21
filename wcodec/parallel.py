"""Parallel entropy decoding of a WCDC file with a persistent process pool.

constriction holds the GIL, so threads do not help; chunks of every ANS 'sym' section are farmed
out to worker processes that write the decoded symbols straight into one shared-memory int8/int16
buffer (no pickling of arrays).  The pool is created once (think: a page's Web Workers) and its
start-up time is reported separately from the decode time.  Reconstruction (dequant / reference
add) then runs in the main process on CPU or MPS exactly as in codecs.decode_record.
"""
import os
import time
from multiprocessing import get_context, shared_memory

import numpy as np

from .ans import AnsSection, open_section
from .container import Reader

_W = {}


def _init(path):
    with open(path, "rb") as f:
        _W["rd"] = Reader(f.read())


def _work(task):
    off, i, shm_name, out_off, n, dt = task
    rd = _W["rd"]
    sec = AnsSection(rd.buf, rd.base + off)
    y = sec.decode_chunk(i)
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        out = np.ndarray((n,), dtype=dt, buffer=shm.buf, offset=out_off)
        if sec.smin:
            y = y + sec.smin
        out[:] = y
    finally:
        del out
        shm.close()
    return i


class ParallelDecoder:
    def __init__(self, path, workers=None):
        self.path = path
        self.workers = workers or os.cpu_count()
        t0 = time.perf_counter()
        self.pool = get_context("spawn").Pool(self.workers, initializer=_init, initargs=(path,))
        self.pool.map(_noop, range(self.workers))  # make sure every worker is up and has the file
        self.startup_s = time.perf_counter() - t0

    def close(self):
        self.pool.close()
        self.pool.join()

    def entropy_decode(self, rd):
        """Decode every plain 'sym' ANS section in parallel. Returns ({record name: int array},
        seconds).  Records without a plain 'sym' section (ll16, ctx-coded, packed, raw) are
        left to the serial decoder."""
        plan = []
        total = 0
        for r in rd.records:
            if "sym" not in r["sec"] or r["params"].get("ctx"):
                continue
            off = r["sec"]["sym"][0]
            sec = open_section(rd.buf, rd.base + off)
            if not isinstance(sec, AnsSection):     # rans16 sections parallelise internally (numba)
                continue
            plan.append((r["name"], off, sec, total))
            total += sec.n
        t0 = time.perf_counter()
        dt = np.int16
        shm = shared_memory.SharedMemory(create=True, size=max(1, total * 2))
        try:
            tasks = []
            for name, off, sec, base in plan:
                if sec.nalpha == 1 or sec.n == 0:
                    continue
                for i in range(sec.nch):
                    tasks.append((off, i, shm.name, 2 * (base + i * sec.chunk), sec.chunk_len(i), dt))
            # big chunks first for load balance
            tasks.sort(key=lambda t: -t[4])
            self.pool.map(_work, tasks, chunksize=1)
        except BaseException:
            shm.close(); shm.unlink()
            raise
        buf = np.ndarray((total,), dtype=dt, buffer=shm.buf)
        out = {}
        for name, off, sec, base in plan:
            if sec.nalpha == 1:
                out[name] = np.full(sec.n, sec.smin, dtype=np.int16)
            else:
                out[name] = buf[base:base + sec.n]     # zero-copy view into shared memory
        self._live = (shm, buf)
        return out, time.perf_counter() - t0

    def release(self):
        """Free the shared symbol buffer (call after reconstruction)."""
        shm, buf = self._live
        del buf
        self._live = None
        shm.close(); shm.unlink()


def decode_parallel(rd, pdec, ref=None, device="cpu", timing=None, keep_state=True):
    """Full decode of a container with parallel entropy decoding; same output as codecs.decode."""
    from .codecs import decode_record
    tim = timing if timing is not None else {}
    syms, t_ent = pdec.entropy_decode(rd)
    tim["entropy_parallel"] = tim.get("entropy_parallel", 0.0) + t_ent
    out = {}
    try:
        for rec in rd.records:
            R = ref.get(rec["name"]) if ref is not None else None
            out[rec["name"]] = decode_record(rd, rec, R, device, tim, keep_state, pre_sym=syms.get(rec["name"]))
    finally:
        del syms
        pdec.release()
    return out


def _noop(i):
    return i
