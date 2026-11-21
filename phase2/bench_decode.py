"""Phase 2 / DECODE-SPEED benchmark: time to rebuild a model's weights from its bitstream.

For each format of a family (results/phase2_family_<tag>.json) the full decode chain is timed:
  independent formats : linears file
  P-frame             : base I-frame file -> P-frame file   (also reported: P-frame alone, base in RAM)
  delta / low-rank    : fp16 base already in RAM -> delta file
  + rest file (lossless embeddings/norms) and the fp16 safetensors load as the reference.
Devices: 'cpu' (entropy decode + reconstruction on CPU) and 'mps' (entropy decode on CPU --
constriction is CPU-only -- then reconstruction on the GPU).  Entropy decode: serial (1 core)
and parallel (persistent process pool, shared memory).  Files are in the OS page cache (warm):
download time is modelled separately from the measured MB.  Median of --reps runs.
"""
import argparse
import platform
import statistics
from p2common import *
from wcodec.parallel import ParallelDecoder, decode_parallel
from wcodec.codecs import decode as wdecode

def main():
    # (everything runs inside main(): the decode pool uses 'spawn', which re-imports this module)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen05b")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--host", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    host = args.host or f"{platform.system()}-{platform.machine()}-{os.cpu_count()}cpu"
    fam = Store(f"phase2_family_{args.tag}").d
    store = Store(f"phase2_decode_{args.tag}_{host}")
    devs = [d for d in args.devices.split(",") if d != "mps" or torch.backends.mps.is_available()]
    W = args.workers or os.cpu_count()
    P = lambda nm: os.path.join(ROOT, fam[nm]["file"])


    def med(xs):
        return round(statistics.median(xs), 4)


    def read(path):
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            b = f.read()
        return Reader(b), time.perf_counter() - t0


    def run_chain(files, dev, par, ref0=None):
        """Decode a chain of files; each file's state is the next one's reference. Returns timings."""
        t = {"io": 0.0, "entropy": 0.0, "entropy_parallel": 0.0, "reconstruct": 0.0}
        st = ref0
        t0 = time.perf_counter()
        for i, f in enumerate(files):
            rd, tio = read(f)
            t["io"] += tio
            keep = "Q" if i < len(files) - 1 else False
            if par is not None:
                st = decode_parallel(rd, par[f], ref=st, device=dev, timing=t, keep_state=keep)
            else:
                st = wdecode(rd, ref=st, device=dev, timing=t, keep_state=keep)
        if str(dev) == "mps":
            torch.mps.synchronize()
        t["total"] = time.perf_counter() - t0
        del st
        return t


    def r16_file(f, chunk=None):
        """rans16 twin of a constriction file (lossless transcode, cached next to it)."""
        from wcodec.transcode import transcode
        d = os.path.join(os.path.dirname(f), "r16" if chunk is None else f"r16_{chunk}"); os.makedirs(d, exist_ok=True)
        g = os.path.join(d, os.path.basename(f))
        if not os.path.exists(g) or os.path.getmtime(g) < os.path.getmtime(f):
            transcode(f, g, chunk=chunk)
        return g


    def bench(label, files, ref_fn=None, extra=None, coders=("constriction", "rans16", "rans16_4k")):
        for coder in coders:
            if coder == "rans16_4k" and "mps" not in devs:
                continue
            lab = label if coder == "constriction" else f"{label}@{coder}"
            if (args.only and label not in args.only.split(",")) or store.has(lab):
                continue
            fl = files if coder == "constriction" else [r16_file(f, 4096 if coder == "rans16_4k" else None) for f in files]
            _bench(lab, fl, ref_fn, dict(extra or {}, coder=coder), coder)
            if coder != "constriction" and os.environ.get("WC_KEEP_R16") != "1":
                for g in fl:            # twins are reproducible (lossless transcode): free the disk
                    os.remove(g)


    def _bench(label, files, ref_fn, extra, coder):
        import numba
        rec = {"files": [os.path.relpath(f, ROOT) for f in files], "MB": sum(os.path.getsize(f) for f in files) / 1e6,
               "host": host, "workers": W, **(extra or {})}
        pools = None
        if coder == "rans16_4k":                      # whole decode on the GPU (Metal rANS kernel)
            from wcodec.metal import decode_gpu
            runs = []
            for r in range(args.reps + 1):
                ref0 = ref_fn("mps") if ref_fn else None
                t = {"io": 0.0, "entropy_gpu": 0.0, "reconstruct": 0.0}
                st = ref0
                t0 = time.perf_counter()
                for i, fl in enumerate(files):
                    rd, tio = read(fl); t["io"] += tio
                    st = decode_gpu(rd, ref=st, timing=t, keep_state="Q" if i < len(files) - 1 else False)
                torch.mps.synchronize()
                t["total"] = time.perf_counter() - t0
                del st, ref0; gc.collect()
                if r:
                    runs.append(t)
            rec["gpu_mps"] = {k: med([x[k] for x in runs]) for k in runs[0]}
            log(f"{label:24s} GPU-rANS mps: total {rec['gpu_mps']['total']:.3f}s (entropy {rec['gpu_mps']['entropy_gpu']:.3f}, "
                f"recon {rec['gpu_mps']['reconstruct']:.3f}) {rec['MB']:.1f} MB")
            store.put(label, rec)
            return
        for mode in ("serial", "parallel"):
            if coder == "rans16":
                numba.set_num_threads(1 if mode == "serial" else numba.config.NUMBA_NUM_THREADS)
                rec["numba_threads_parallel"] = numba.config.NUMBA_NUM_THREADS
            elif mode == "parallel":
                t0 = time.perf_counter()
                pools = {f: ParallelDecoder(f, W) for f in files}
                rec["pool_startup_s"] = round(time.perf_counter() - t0, 3)
            for dev in devs:
                runs = []
                for r in range(args.reps + 1):              # first run = warm-up (page cache, kernels, JIT)
                    ref0 = ref_fn(dev) if ref_fn else None
                    tt = run_chain(files, dev, pools if (mode == "parallel" and coder != "rans16") else None, ref0)
                    del ref0; gc.collect()
                    if r:
                        runs.append(tt)
                rec[f"{mode}_{dev}"] = {k: med([x[k] for x in runs]) for k in runs[0]}
                log(f"{label:24s} {mode:8s} {dev}: total {rec[f'{mode}_{dev}']['total']:.3f}s "
                    f"(entropy {rec[f'{mode}_{dev}']['entropy'] + rec[f'{mode}_{dev}']['entropy_parallel']:.3f}, "
                    f"recon {rec[f'{mode}_{dev}']['reconstruct']:.3f}, io {rec[f'{mode}_{dev}']['io']:.3f}) {rec['MB']:.1f} MB")
        for p in (pools or {}).values():
            p.close()
        if coder == "rans16":
            numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        store.put(label, rec)


    # ---- reference: fp16 safetensors load (whole model file from the HF cache) ----
    def st_files(rid):
        import glob
        return sorted(glob.glob(os.path.join(snapshot_dir(rid), "*.safetensors")))


    if not store.has("safetensors_fp16_target") and (not args.only or "safetensors_fp16_target" in args.only):
        from safetensors.torch import load_file
        rec = {"MB": sum(os.path.getsize(f) for f in st_files(args.target)) / 1e6, "host": host}
        for dev in devs:
            runs = []
            for r in range(args.reps + 1):
                t0 = time.perf_counter()
                sd = {}
                for f in st_files(args.target):
                    sd.update(load_file(f, device="cpu"))
                sd = {k: v.to(dev, torch.float16) for k, v in sd.items()}   # bf16 hub file -> fp16 on device
                if dev == "mps":
                    torch.mps.synchronize()
                if r:
                    runs.append(time.perf_counter() - t0)
                del sd; gc.collect()
            rec[f"serial_{dev}"] = {"total": med(runs)}
            log(f"safetensors fp16 {dev}: {med(runs):.3f}s ({rec['MB']:.1f} MB)")
        store.put("safetensors_fp16_target", rec)

    # fp16 base in RAM (reference for delta / low-rank; its own load time is the safetensors line above)
    _B = {}
    def fp16_base(dev):
        if "sd" not in _B:
            Bsd = LazySD(args.base)
            lin = [r["name"] for r in Reader(P("T_ecsq_c0.35")).records]
            pfx = "model"
            _B["sd"] = {n: Bsd[f"{pfx}.{n}.weight"].to(torch.float16) for n in lin}
        return {n: TState(W=v.to(dev)) for n, v in _B["sd"].items()}


    def zstd_reference(label, f):
        """Same symbols as int8/int16 bytes + zstd-19 (generic coder): size and decode time."""
        import zstandard
        lab = label + "@zstd19"
        if store.has(lab) or (args.only and label not in args.only.split(",")):
            return
        rd = Reader(f)
        from wcodec.ans import open_section
        syms, ans_bytes = [], 0
        for r in rd.records:
            if "sym" in r["sec"]:
                syms.append(open_section(rd.section(r, "sym")).decode())
                ans_bytes += r["sec"]["sym"][1]
        if not syms:
            return
        x = np.concatenate(syms)
        raw = x.astype(np.int8 if (x.min() >= -128 and x.max() <= 127) else np.int16).tobytes()
        comp = zstandard.ZstdCompressor(level=19, threads=4).compress(raw)
        dz = zstandard.ZstdDecompressor()
        runs = []
        for r in range(args.reps + 1):
            t0 = time.perf_counter()
            y = np.frombuffer(dz.decompress(comp), dtype=np.int8 if len(raw) == x.size else np.int16)
            if r:
                runs.append(time.perf_counter() - t0)
        assert np.array_equal(y, x.astype(y.dtype))
        rec = {"MB": (rd.nbytes - ans_bytes + len(comp)) / 1e6, "sym_MB_zstd19": len(comp) / 1e6,
               "sym_MB_ans": ans_bytes / 1e6, "serial_cpu": {"entropy": med(runs)}, "host": host,
               "note": "symbol decode only (single-thread zstd); reconstruction cost is the same as the ANS file"}
        store.put(lab, rec)
        log(f"{lab:30s} sym zstd19 {len(comp)/1e6:.2f} MB vs ANS {ans_bytes/1e6:.2f} MB; zstd decode {med(runs):.3f}s")


    cfgs = sorted(fam)
    for nm in cfgs:
        if fam[nm].get("file") and nm.startswith(("T_", "B_")) and "rest" not in nm and not nm.endswith("_fp16") \
                and "packed" not in nm and "lowrank" not in nm:
            zstd_reference(nm, P(nm))
    for nm in cfgs:
        if not nm.startswith(("T_", "B_")) or "rest" in nm or nm.endswith("_fp16"):
            continue
        if nm.startswith("T_pframe"):
            c = nm.split("_c")[1]
            bench(f"{nm}+base", [P(f"B_ecsq_c{c}"), P(nm)], extra={"what": "base I-frame + P-frame"})
            # P-frame alone, base state already decoded in RAM
            rdB = Reader(P(f"B_ecsq_c{c}"))
            _cache = {}
            def baseQ(dev, rdB=rdB, c=c):
                if c not in _cache:
                    _cache[c] = {k: TState(Q=v.Q.cpu(), step=v.step.cpu()) for k, v in wdecode(rdB, device="cpu", keep_state="Q").items()}
                return {k: TState(Q=v.Q.to(dev), step=v.step.to(dev)) for k, v in _cache[c].items()}
            bench(f"{nm}_only", [P(nm)], ref_fn=baseQ, extra={"what": "P-frame given decoded base in RAM"})
            _cache.clear(); gc.collect()
        elif nm.startswith(("T_delta", "T_lowrank")):
            bench(nm, [P(nm)], ref_fn=fp16_base, extra={"what": "delta given fp16 base in RAM"})
        else:
            bench(nm, [P(nm)], extra={"what": "independent"})
    for nm in cfgs:
        if "rest" in nm and not nm.endswith("raw16"):
            rdref = None
            if nm == "T_rest_xor_base":
                Bsd = LazySD(args.base)
                keys = [r["name"] for r in Reader(P(nm)).records]
                base_rest = {k: Bsd[k].to(torch.float16) for k in keys}
                rdref = lambda dev: {k: TState(W=v) for k, v in base_rest.items()}
            bench(nm, [P(nm)], ref_fn=rdref, extra={"what": "rest (lossless)"})
    log("done", host)


if __name__ == "__main__":
    main()
