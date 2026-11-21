"""Generic-coder reference for every entropy-coded linears file: the SAME symbols, each tensor stored
as its narrowest integer type (int8 unless that tensor has |index| > 127), concatenated, zstd-19
(+ gzip-9).  Reports compressed symbol bytes vs the file's ANS symbol bytes and the single-thread
decompression time (the reconstruction step afterwards is identical to the ANS file)."""
import argparse
import zlib
import zstandard
from p2common import *
from wcodec.ans import open_section

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="qwen05b")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--no_gzip", action="store_true", help="gzip-9 is very slow on low-entropy streams")
args = ap.parse_args()
fam = Store(f"phase2_family_{args.tag}").d
store = Store(f"phase2_zstd_ref_{args.tag}")
for nm, r in sorted(fam.items()):
    f = os.path.join(ROOT, r.get("file", "")) if r.get("file") else None
    if not f or not os.path.exists(f) or "rest" in nm or store.has(nm):
        continue
    rd = Reader(f)
    parts, ans_bytes, n = [], 0, 0
    for rec in rd.records:
        if "sym" not in rec["sec"]:
            continue
        x = open_section(rd.section(rec, "sym")).decode()
        parts.append(small_int_bytes(x)); ans_bytes += rec["sec"]["sym"][1]; n += x.size
    if not parts:
        continue
    raw = b"".join(parts)
    t0 = time.time(); comp = zstandard.ZstdCompressor(level=19, threads=4).compress(raw); tc = time.time() - t0
    gz = None if args.no_gzip else len(zlib.compress(raw, 9))
    dz = zstandard.ZstdDecompressor()
    ts = []
    for i in range(args.reps + 1):
        t0 = time.perf_counter(); y = dz.decompress(comp); dt = time.perf_counter() - t0
        if i:
            ts.append(dt)
    assert y == raw
    side = rd.nbytes - ans_bytes
    out = {"n_symbols": n, "raw_sym_MB": len(raw) / 1e6, "ans_sym_MB": ans_bytes / 1e6, "zstd19_sym_MB": len(comp) / 1e6,
           "gzip9_sym_MB": gz and gz / 1e6, "zstd19_vs_ans_pct": 100 * (len(comp) / ans_bytes - 1),
           "gzip9_vs_ans_pct": gz and 100 * (gz / ans_bytes - 1),
           "file_MB_if_zstd19": (side + len(comp)) / 1e6, "zstd19_bpw": 8 * (side + len(comp)) / n,
           "zstd19_decode_1t_s": float(np.median(ts)), "zstd19_compress_4t_s": round(tc, 1)}
    store.put(nm, out)
    log(f"{nm:18s} ANS {ans_bytes / 1e6:8.2f} MB | zstd-19 {len(comp) / 1e6:8.2f} MB ({out['zstd19_vs_ans_pct']:+.1f}%) "
        + (f"gzip-9 {gz / 1e6:8.2f} MB ({out['gzip9_vs_ans_pct']:+.1f}%) " if gz else "") +
        f"| zstd decode {out['zstd19_decode_1t_s'] * 1e3:.0f} ms")
log("done")
