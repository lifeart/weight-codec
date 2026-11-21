"""Cross-machine check + drift evaluation of the trajectory bitstreams.

Files ENCODED on the HF Job (x86 Linux CPU) are downloaded from the private dataset repo and DECODED
here (Apple Silicon).  Per window and step size c:
  * the decoded I+P chain must hash to the job's sha256 (bit-exact across machines / OSes),
  * decode time on this machine: constriction 1 thread, rans16 numba parallel, rans16-4k Metal GPU,
  * drift check (phase-1 protocol: MPS fp16, 40 WikiText-2 windows): PPL of the original last
    checkpoint (streamed from the Hub into RAM), of its own I-frame (decoded allI file) and of the
    end of the closed-loop P chain (decoded IPPP files).
Downloaded files live in a fresh data/xcheck_* dir and are deleted afterwards."""
import argparse
import hashlib
import io
import shutil
from p2common import *
from wcodec.transcode import transcode
from wcodec.codecs import decode as wdecode

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="pythia410m_traj_v2")
ap.add_argument("--windows", default="late,mid")
ap.add_argument("--cs", default="0.35,0.7")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ppl_windows", type=int, default=40)
args = ap.parse_args()
from huggingface_hub import hf_hub_download, hf_hub_url, get_session
REPO = os.environ.get("WC_REPO", "lifeart/weight-codec-phase2")  # author's private dataset; set WC_REPO to yours
PY = "EleutherAI/pythia-410m"
store = Store("phase2_xcheck_trajectory")
res_dir = os.path.join(DATA, "hfjob_" + args.run)
T = json.load(open(hf_hub_download(REPO, f"runs/{args.run}/results/phase2_trajectory_pythia410m.json",
                                   repo_type="dataset", local_dir=res_dir)))
tok = load_tok(PY)
test_ids = wt2_ids(tok, "test", "pythia")


def fetch(step):
    check_disk(3.0)
    r = get_session().get(hf_hub_url(PY, "pytorch_model.bin", revision=f"step{step}"), timeout=300, follow_redirects=True)
    r.raise_for_status()
    sd = torch.load(io.BytesIO(r.content), map_location="cpu", weights_only=True)
    return {k: v.to(torch.float16) for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def ppl_of(sd):
    from transformers import AutoConfig, GPTNeoXForCausalLM
    cfg = AutoConfig.from_pretrained(PY, revision="step143000")
    m = GPTNeoXForCausalLM(cfg).to(torch.float16)
    m.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()}, strict=False)
    m.eval(); m.config.use_cache = False; m = m.to(DEV)
    r = eval_lm(m, test_ids, max_windows=args.ppl_windows)
    del m; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return r


def sha(st, keys):
    h = hashlib.sha256()
    for k in keys:
        h.update(st[k].W.detach().cpu().contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def chain(fl, mode):
    st = None
    t0 = time.perf_counter()
    for i, f in enumerate(fl):
        with open(f, "rb") as fh:
            rd = Reader(fh.read())
        keep = "Q" if i < len(fl) - 1 else False
        if mode == "gpu":
            from wcodec.metal import decode_gpu
            st = decode_gpu(rd, ref=st, keep_state=keep)
        else:
            st = wdecode(rd, ref=st, device="cpu", keep_state=keep)
    if mode == "gpu":
        torch.mps.synchronize()
    return st, time.perf_counter() - t0


for w in args.windows.split(","):
    summ = T.get(f"{w}_summary")
    if summ is None or store.has(f"{w}_done"):
        continue
    tmp = os.path.join(DATA, f"xcheck_{w}"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    last = summ["steps"][-1]
    orig = fetch(last)
    ev = {"orig": ppl_of(orig)}
    log(w, "orig ppl", round(ev["orig"]["ppl"], 3))
    for c in args.cs.split(","):
        ch = summ["chain"][c]
        files = [hf_hub_download(REPO, f"runs/{args.run}/{rel}", repo_type="dataset", local_dir=tmp) for rel in ch["chain_files"]]
        fI = hf_hub_download(REPO, f"runs/{args.run}/bitstreams/pythia410m/{w}/allI_c{c}_f{len(summ['steps']) - 1}.wcdc",
                             repo_type="dataset", local_dir=tmp)
        keys = [r["name"] for r in Reader(files[0]).records]
        st, t1 = chain(files, "constriction")
        rec = {"MB_chain": sum(os.path.getsize(f) for f in files) / 1e6, "job_sha256": ch["sha256_chain_end"],
               "mac_sha256": sha(st, keys), "job_decode_chain_s_constriction": ch["decode_chain_s"],
               "job_decode_chain_s_rans16": ch.get("rans16_decode_chain_s"), "job_hardware": summ.get("hardware")}
        rec["sha_match"] = rec["mac_sha256"] == rec["job_sha256"]
        sdP = dict(orig); sdP.update({k: st[k].W for k in keys})
        del st; gc.collect()
        sI = wdecode(Reader(fI), device="cpu", keep_state=False)
        sdI = dict(orig); sdI.update({k: sI[k].W for k in keys}); del sI
        ev[f"I_c{c}"] = ppl_of(sdI); ev[f"IPPP_end_c{c}"] = ppl_of(sdP)
        del sdI, sdP; gc.collect()
        if c == args.cs.split(",")[0]:
            rec["mac_constriction_1t_s"] = float(np.median([chain(files, "constriction")[1] for _ in range(args.reps)]))
            r16 = [f + ".r16" for f in files]
            for f, g in zip(files, r16):
                transcode(f, g)
            chain(r16, "cpu")
            rec["mac_rans16_par_s"] = float(np.median([chain(r16, "cpu")[1] for _ in range(args.reps)]))
            r4 = [f + ".r4k" for f in files]
            for f, g in zip(files, r4):
                transcode(f, g, chunk=4096)
            st, _ = chain(r4, "gpu"); rec["gpu_sha_match"] = sha(st, keys) == rec["job_sha256"]; del st
            rec["mac_gpu_s"] = float(np.median([chain(r4, "gpu")[1] for _ in range(args.reps)]))
            rec["MB_chain_rans16"] = sum(os.path.getsize(f) for f in r16) / 1e6
            rec["MB_chain_rans16_4k"] = sum(os.path.getsize(f) for f in r4) / 1e6
            for g in r16 + r4:
                os.remove(g)
        store.put(f"{w}_c{c}", rec)
        log(w, c, {k: v for k, v in rec.items() if "sha256" not in k},
            {k: round(v["ppl"], 3) for k, v in ev.items()})
    store.put(f"{w}_ppl", ev)
    store.put(f"{w}_done", True)
    del orig; gc.collect()
    shutil.rmtree(tmp)
log("done")
