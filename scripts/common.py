"""Shared measurement spine for weight-codec experiments.

Distortion  = downstream: WikiText-2 test perplexity (GPTQ-style non-overlapping
              2048-token windows) + KL(ref || model) and top-1 agreement vs an
              fp16 reference on the first K windows.
Size        = effective bits per weight (bpw) over the *quantized set*
              (all nn.Linear inside transformer blocks). Entropy-coded sizes use
              the EMPIRICAL ORDER-0 ENTROPY of the symbol stream per tensor
              (-sum p log2 p) plus the cost of the frequency table and all side
              info (scales/zeros) at their stored precision. This is a PROXY for
              what a static-model rANS/arithmetic coder achieves (within ~0.1%);
              no actual entropy coder is run.
"""
import os, sys, json, math, time, gc, random

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
for _d in (RES, FIG, DATA):
    os.makedirs(_d, exist_ok=True)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEQLEN = 2048


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def disk_free_gb(path=ROOT):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def check_disk(min_gb=3.0):
    f = disk_free_gb()
    if f < min_gb:
        raise SystemExit(f"DISK STOP: only {f:.2f} GB free (< {min_gb})")
    return f


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ----------------------------------------------------------------------------
# result store (checkpoint discipline: every finished config is persisted)
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, name):
        self.path = os.path.join(RES, name + ".json")
        self.d = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.d = json.load(f)

    def has(self, k):
        return k in self.d

    def get(self, k, default=None):
        return self.d.get(k, default)

    def put(self, k, v):
        self.d[k] = v
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=1, default=_json_default)
        os.replace(tmp, self.path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(type(o))


def save_json(name, obj):
    p = os.path.join(RES, name if name.endswith(".json") else name + ".json")
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(p + ".tmp", p)
    return p


# ----------------------------------------------------------------------------
# models / data
# ----------------------------------------------------------------------------
def load_model(rid, revision=None, dtype=torch.float16, device=DEV):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(rid, revision=revision, dtype=dtype)
    m.eval()
    m.config.use_cache = False
    return m.to(device)


def load_tok(rid):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(rid)


def decoder_layers(model):
    base = getattr(model, model.base_model_prefix)
    return base.layers


def block_linears(model):
    """[(name, module)] for every nn.Linear inside transformer blocks (the quantized set)."""
    out = []
    for li, layer in enumerate(decoder_layers(model)):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                out.append((f"layers.{li}.{n}", m))
    return out


def snapshot_weights(model):
    """CPU fp16 copy of the quantized set (to restore between configs)."""
    return {n: m.weight.detach().to("cpu", copy=True) for n, m in block_linears(model)}


def restore_weights(model, snap):
    for n, m in block_linears(model):
        m.weight.data.copy_(snap[n].to(m.weight.device, m.weight.dtype))


def wt2_ids(tok, split="test", tag="tok"):
    path = os.path.join(DATA, f"wt2_{split}_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0].to(torch.int32)
    torch.save(ids, path)
    return ids


def calib_windows(train_ids, n=128, seqlen=SEQLEN, seed=0):
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, train_ids.numel() - seqlen - 1, (n,), generator=g)
    return [train_ids[s:s + seqlen].long() for s in starts.tolist()]


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def _base_and_head(model):
    return getattr(model, model.base_model_prefix), model.get_output_embeddings()


@torch.no_grad()
def build_ref(model, ids, k_windows=16, seqlen=SEQLEN):
    """Cache final (post-norm) hidden states + LM head of the reference model on the first
    k windows -> exact KL(ref||q) later without keeping the reference model in memory."""
    base, head = _base_and_head(model)
    hid = []
    for i in range(k_windows):
        x = ids[i * seqlen:(i + 1) * seqlen].long().unsqueeze(0).to(DEV)
        hid.append(base(input_ids=x).last_hidden_state[0].to("cpu", torch.float16))
    return {"hidden": hid}


def attach_head(ref, model):
    """Attach the (unmodified!) LM head of the reference model. Not stored on disk to save
    space: call right after loading the fp16 reference model, before any quantization."""
    head = model.get_output_embeddings()
    ref["head"] = head.weight.detach().to("cpu", torch.float16, copy=True)
    ref["head_bias"] = None if head.bias is None else head.bias.detach().cpu().clone()
    return ref


def get_ref(model, ids, path, k_windows=16):
    """Load/build cached reference hidden states (disk) and attach the live head (RAM)."""
    if os.path.exists(path):
        ref = torch.load(path)
        ref = {"hidden": ref["hidden"]}
    else:
        ref = build_ref(model, ids, k_windows)
        torch.save(ref, path)
    return attach_head(ref, model)


@torch.no_grad()
def eval_lm(model, ids, seqlen=SEQLEN, max_windows=None, ref=None, chunk=512):
    """Returns dict(ppl, nll, n_tokens, kl, top1) ; kl/top1 only if ref given (first K windows)."""
    model.eval()
    base, head = _base_and_head(model)
    n = ids.numel() // seqlen
    if max_windows:
        n = min(n, max_windows)
    nll, ntok, kl, agree, kltok = 0.0, 0, 0.0, 0, 0
    rhead = ref["head"].to(DEV) if ref is not None else None
    rbias = ref["head_bias"].to(DEV) if (ref is not None and ref["head_bias"] is not None) else None
    t0 = time.time()
    for i in range(n):
        x = ids[i * seqlen:(i + 1) * seqlen].long().unsqueeze(0).to(DEV)
        h = base(input_ids=x).last_hidden_state[0]
        tgt = x[0, 1:]
        use_ref = ref is not None and i < len(ref["hidden"])
        if use_ref:
            hr_all = ref["hidden"][i].to(DEV)
        for c0 in range(0, seqlen - 1, chunk):
            c1 = min(c0 + chunk, seqlen - 1)
            lp = torch.log_softmax(head(h[c0:c1]).float(), -1)
            nll += -lp.gather(1, tgt[c0:c1, None].long()).sum().item()
            if use_ref:
                lr = hr_all[c0:c1] @ rhead.T
                if rbias is not None:
                    lr = lr + rbias
                lpr = torch.log_softmax(lr.float(), -1)
                kl += (lpr.exp() * (lpr - lp)).sum().item()
                agree += (lpr.argmax(-1) == lp.argmax(-1)).sum().item()
                kltok += c1 - c0
        ntok += seqlen - 1
    out = {"ppl": math.exp(nll / ntok), "nll": nll / ntok, "n_tokens": ntok, "n_windows": n,
           "eval_s": round(time.time() - t0, 1)}
    if ref is not None:
        out["kl"] = kl / kltok
        out["top1"] = agree / kltok
        out["kl_tokens"] = kltok
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


# ----------------------------------------------------------------------------
# entropy / bit accounting
# ----------------------------------------------------------------------------
def entropy_bits(sym):
    """Empirical order-0 entropy (bits/symbol) and #distinct symbols of an integer tensor."""
    s = sym.reshape(-1)
    if s.device.type != "cpu":
        s = s.cpu()
    _, counts = torch.unique(s, return_counts=True)
    p = counts.double() / counts.sum()
    H = float(-(p * torch.log2(p)).sum())
    return H, int(counts.numel())


def coded_bits(sym):
    """Total bits to entropy-code one tensor's symbols with a static per-tensor model:
    n*H + frequency table (each used symbol: 32-bit value+count)."""
    H, k = entropy_bits(sym)
    return sym.numel() * H + 32 * k, H


# ----------------------------------------------------------------------------
# quantizers  (all return integer symbols, dequantized weights, side-info bits)
# ----------------------------------------------------------------------------
def _fp16(t):
    return t.half().float()


def rtn_asym(W, bits, group=128):
    """GPTQ-style asymmetric min/max RTN (range includes 0). group<=0 -> per output row.
    Side info: fp16 scale + `bits`-bit zero point per group."""
    W = W.float()
    O, I = W.shape
    g = I if group <= 0 else group
    assert I % g == 0
    maxq = 2 ** bits - 1
    Wg = W.reshape(O, I // g, g)
    xmin = Wg.amin(-1, keepdim=True).clamp(max=0)
    xmax = Wg.amax(-1, keepdim=True).clamp(min=0)
    same = (xmax - xmin) == 0
    xmax = torch.where(same, xmax + 1e-8, xmax)
    scale = _fp16((xmax - xmin) / maxq)
    zero = torch.round(-xmin / scale).clamp(0, maxq)
    q = torch.clamp(torch.round(Wg / scale) + zero, 0, maxq)
    deq = (scale * (q - zero)).reshape(O, I)
    side = (O * (I // g)) * (16 + bits)
    return q.reshape(O, I).to(torch.int16), deq, side


def ecsq(W, step):
    """Entropy-coded uniform scalar quantizer (no clipping, unbounded levels).
    `step` broadcastable to W (already fp16-representable)."""
    q = torch.round(W.float() / step)
    return q.to(torch.int32), q * step


# ----------------------------------------------------------------------------
# GPTQ (Frantar et al. 2022), generic quantizer callback, CPU fp32
# ----------------------------------------------------------------------------
@torch.no_grad()
def gptq(W, H, mode, bits=4, group=128, step=None, blocksize=128, percdamp=0.01, dev="cpu"):
    """mode='rtn': asym min/max grid with per-group params computed lazily (standard GPTQ g128)
       mode='grid': fixed uniform grid, step tensor (O,I) or (O,1) (ECSQ-GPTQ, no clipping).
    Returns (symbols int, dequantized W, side_bits)."""
    W = W.to(dev).float().clone()
    H = H.to(dev).float().clone()
    O, I = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(I, device=dev)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    S = torch.zeros(O, I, dtype=torch.int32, device=dev)
    maxq = 2 ** bits - 1
    g = I if group <= 0 else group
    scale = zero = None
    side = 0
    if mode == "grid":
        stp = step.to(dev).float().expand(O, I) if step.dim() == 2 else step.to(dev).float()
    for i1 in range(0, I, blocksize):
        i2 = min(i1 + blocksize, I)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]
            if mode == "rtn":
                if col % g == 0:
                    # reference-GPTQ behaviour: params from the outer W (errors of previous
                    # blocks already propagated); with blocksize==group this equals W1 here.
                    blk = W[:, col:col + g]
                    xmin = blk.amin(1).clamp(max=0)
                    xmax = blk.amax(1).clamp(min=0)
                    xmax = torch.where(xmax - xmin == 0, xmax + 1e-8, xmax)
                    scale = _fp16((xmax - xmin) / maxq)
                    zero = torch.round(-xmin / scale).clamp(0, maxq)
                    side += O * (16 + bits)
                qi = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
                q = scale * (qi - zero)
            else:
                s = stp[:, col]
                qi = torch.round(w / s)
                q = qi * s
            Q1[:, i] = q
            S[:, col] = qi.to(torch.int32)
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return S.cpu(), Q.cpu(), side


@torch.no_grad()
def collect_hessians(model, calib, share=None):
    """H = sum_x x x^T / N for the input of every block linear (from the fp model, i.e.
    non-sequential GPTQ). `share` maps module-suffix -> suffix whose H is identical
    (e.g. k_proj->q_proj). Returns {name: H (cpu fp32)}, plus diag-only dict."""
    share = share or {}
    lin = block_linears(model)
    H, cnt, hooks = {}, {}, []

    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            if name not in H:
                H[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                cnt[name] = 0
            H[name].addmm_(x.T, x)
            cnt[name] += x.shape[0]
        return hook

    for n, m in lin:
        suf = n.split(".", 2)[2]
        if suf in share:
            continue
        hooks.append(m.register_forward_hook(mk(n)))
    base = getattr(model, model.base_model_prefix)
    t0 = time.time()
    for j, x in enumerate(calib):
        base(input_ids=x.unsqueeze(0).to(DEV))
    for h in hooks:
        h.remove()
    out = {}
    for n, m in lin:
        li, suf = n.split(".", 2)[1], n.split(".", 2)[2]
        src = f"layers.{li}.{share[suf]}" if suf in share else n
        out[n] = src
    Hc = {k: (v / cnt[k]).cpu() for k, v in H.items()}
    log(f"hessians: {len(Hc)} unique in {time.time() - t0:.0f}s")
    del H
    if DEV == "mps":
        torch.mps.empty_cache()
    return {n: Hc[src] for n, src in out.items()}


QWEN_SHARE = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj",
              "mlp.up_proj": "mlp.gate_proj"}


# ----------------------------------------------------------------------------
# applying a quantizer to the whole quantized set
# ----------------------------------------------------------------------------
@torch.no_grad()
def quantize_model(model, snap, fn, entropy=True):
    """fn(name, W_fp32_cpu) -> (symbols, deq, side_bits). Writes deq into model.
    Returns accounting dict: fixed-rate bpw is computed by the caller; here we return
    entropy-coded total bits and per-tensor entropies."""
    tot_w, tot_side, tot_coded = 0, 0, 0.0
    per = {}
    for n, m in block_linears(model):
        W = snap[n].float()
        sym, deq, side = fn(n, W)
        m.weight.data.copy_(deq.to(m.weight.device, m.weight.dtype))
        cb, H = coded_bits(sym) if (entropy and sym is not None) else (0.0, float("nan"))
        tot_w += W.numel(); tot_side += side; tot_coded += cb
        per[n] = {"H": H, "side_bpw": side / W.numel()}
    return {"n_weights": tot_w, "side_bpw": tot_side / tot_w,
            "entropy_bpw": (tot_coded + tot_side) / tot_w,
            "sym_entropy_bpw": tot_coded / tot_w, "per_tensor": per}


# ----------------------------------------------------------------------------
# raw state-dict access (CPU, no model build) for probes
# ----------------------------------------------------------------------------
def snapshot_dir(rid, revision=None, patterns=("*.safetensors", "*.json")):
    from huggingface_hub import snapshot_download
    return snapshot_download(rid, revision=revision, allow_patterns=list(patterns), local_files_only=True)


def load_sd(rid, revision=None, dtype=torch.float32, keys=None):
    """Load a safetensors state dict on CPU. keys: optional predicate on key name."""
    import glob
    from safetensors import safe_open
    d = snapshot_dir(rid, revision)
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if keys is None or keys(k):
                    sd[k] = fh.get_tensor(k).to(dtype)
    return sd


def energy_topr(sv, rs):
    e = sv.double() ** 2
    c = torch.cumsum(e, 0) / e.sum()
    return {int(r): float(c[min(r, len(c)) - 1]) for r in rs}


def rank_for(sv, frac):
    e = sv.double() ** 2
    c = torch.cumsum(e, 0) / e.sum()
    return int(torch.searchsorted(c, torch.tensor(frac, dtype=c.dtype)).item()) + 1


def eff_rank(sv):
    """exp(spectral entropy) of the normalized energy spectrum (Roy & Vetterli)."""
    p = sv.double() ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def svals(M):
    M = M.float()
    if min(M.shape) > 2048 and max(M.shape) > 8 * min(M.shape):
        # tall matrices (embeddings): eig of the small Gram in float64
        G = (M.T @ M).double() if M.shape[0] > M.shape[1] else (M @ M.T).double()
        ev = torch.linalg.eigvalsh(G).clamp(min=0).flip(0)
        return ev.sqrt().float()
    return torch.linalg.svdvals(M)


class LazySD:
    """Read tensors on demand from a cached safetensors snapshot (keeps RAM flat)."""
    def __init__(self, rid, revision=None):
        import glob
        from safetensors import safe_open
        d = snapshot_dir(rid, revision)
        self.h = [safe_open(f, "pt") for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))]
        self.where = {k: h for h in self.h for k in h.keys()}

    def keys(self):
        return list(self.where)

    def __getitem__(self, k):
        return self.where[k].get_tensor(k)


# ----------------------------------------------------------------------------
# chat-behaviour metric for instruct models: the fp16 instruct model's own greedy answers to
# a fixed prompt set; candidate scored by teacher-forced NLL of those answers and
# KL(instruct_fp16 || candidate) on the answer tokens.
# ----------------------------------------------------------------------------
def _chat_ids(tok, prompt):
    r = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                return_tensors="pt")
    if not torch.is_tensor(r):
        r = r["input_ids"]
    return r


@torch.no_grad()
def build_chat_set(model, tok, prompts, max_new=128):
    seqs = []
    for p in prompts:
        ids = _chat_ids(tok, p).to(DEV)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                             do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
        seqs.append({"prompt": p, "ids": out[0].tolist(), "start": int(ids.shape[1])})
    return seqs


@torch.no_grad()
def chat_hidden(model, seqs):
    base, _ = _base_and_head(model)
    hs = []
    for s in seqs:
        x = torch.tensor(s["ids"], device=DEV).unsqueeze(0)
        h = base(input_ids=x).last_hidden_state[0]
        hs.append(h[s["start"] - 1:len(s["ids"]) - 1].to("cpu", torch.float16))
    return hs


@torch.no_grad()
def eval_chat(model, seqs, ref_hidden=None, ref_head=None):
    base, head = _base_and_head(model)
    nll, n, kl, agree = 0.0, 0, 0.0, 0
    rh = ref_head.to(DEV) if ref_head is not None else None
    for i, s in enumerate(seqs):
        x = torch.tensor(s["ids"], device=DEV).unsqueeze(0)
        h = base(input_ids=x).last_hidden_state[0][s["start"] - 1:len(s["ids"]) - 1]
        tgt = x[0, s["start"]:]
        lp = torch.log_softmax(head(h).float(), -1)
        nll += -lp.gather(1, tgt[:, None]).sum().item(); n += tgt.numel()
        if ref_hidden is not None:
            lpr = torch.log_softmax((ref_hidden[i].to(DEV) @ rh.T).float(), -1)
            kl += (lpr.exp() * (lpr - lp)).sum().item()
            agree += (lpr.argmax(-1) == lp.argmax(-1)).sum().item()
    out = {"chat_nll": nll / n, "chat_tokens": n}
    if ref_hidden is not None:
        out.update({"chat_kl": kl / n, "chat_top1": agree / n})
    return out
