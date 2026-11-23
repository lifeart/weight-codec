"""Vector-quantization (codebook clustering) toolkit for the VQ probes (V1-V3).

A weight matrix W (O x I) is cut into sub-vectors of `d` CONSECUTIVE input-dim weights of one row
(W.reshape(-1, d)); each row is first normalised by its fp16 RMS r_o (16 bits/row side info, the
same side info as the scalar ECSQ_row baseline).  A codebook C (K x d, stored fp16 = 16*K*d bits)
is fitted by k-means (or entropy-constrained k-means, ECVQ, Chou-Lookabaugh-Gray 1989) on the
normalised vectors of a *scope* (one tensor / one weight type across all layers / everything).
Reconstruction: W_hat[o, j*d:(j+1)*d] = r_o * C[idx[o, j]].

Rate accounting (bits charged per SCALAR weight, everything included):
  index stream  : empirical order-0 entropy of the int32 indices + frequency table, EITHER per
                  tensor (= scripts/common.coded_bits, 32 bits per used symbol per tensor) OR one
                  pooled table per codebook scope (N_scope * H_pooled + 32 * K_used) - whichever is
                  smaller for the config (a 1-bit flag; both variants are reported)
  codebook      : 16 * d * K_used per scope (fp16 centroids, only centroids that are used)
  scales        : 16 bits per output row (fp16 RMS)
"""
import math
import time

import numpy as np
import torch

from common import DEV, log, coded_bits


def wtype(name):
    """'layers.3.mlp.up_proj' -> 'mlp.up_proj'."""
    return name.split(".", 2)[2]


def scope_key(name, scope):
    if scope == "tensor":
        return name
    if scope == "type":
        return wtype(name)
    if scope == "global":
        return "all"
    raise ValueError(scope)


def row_scale(W):
    """fp16 per-row RMS (the stored side info), as fp32."""
    return W.float().pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8).half().float()


def to_vecs(W, d, r=None):
    r = row_scale(W) if r is None else r
    return (W.float() / r).reshape(-1, d), r


def _chunk(K):
    return max(256, min(1 << 16, (1 << 26) // max(K, 1)))     # ~256 MB fp32 distance block


@torch.no_grad()
def assign(X, C, pen=None, M=None, return_min=False):
    """argmin_k ||(x - c_k) M||^2 + pen_k  for every row x of X (on DEV).  M: optional d x d metric
    transform (Mahalanobis / Hessian-weighted).  Returns int64 indices (and the minimum cost)."""
    if M is not None:
        X = X @ M
        C = C @ M
    cc = (C * C).sum(1)
    if pen is not None:
        cc = cc + pen
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    mins = torch.empty(X.shape[0], device=X.device) if return_min else None
    ch = _chunk(C.shape[0])
    Ct = C.T.contiguous()
    for i in range(0, X.shape[0], ch):
        x = X[i:i + ch]
        dist = torch.addmm(cc[None, :], x, Ct, beta=1.0, alpha=-2.0)
        if return_min:
            m, a = dist.min(1)
            out[i:i + ch] = a
            mins[i:i + ch] = m + (x * x).sum(1)
        else:
            out[i:i + ch] = dist.argmin(1)
    return (out, mins) if return_min else out


# ---- escape to an unbounded uniform lattice (outlier vectors) -------------------------------
# A finite codebook clips the heavy tails of LLM weights (|w| up to ~28 rms): the outermost
# centroid of a quantile-initialised codebook sits near the 99.8 % quantile and the ECVQ rate
# penalty pushes rare tail vectors onto cheap inner centroids.  Weight-MSE hardly notices
# (top-0.01 % weights = 6-9 % of the SE) but downstream KL triples (first V1 run, kept as
# naive_* in results).  Fix = RD mode decision per vector: codebook index  vs  ESCAPE symbol +
# d integers on the uniform lattice of step s = sqrt(6 lam / ln2) (the ECSQ step whose high-rate
# slope equals lam), entropy coded as a separate stream.  Cost model for the decision:
# ||x - q||^2 + lam * (len_esc + sum_k len_s(q_k)), len_s = Elias-gamma-like 1 + 2 floor(log2(1+|q|));
# the bits actually CHARGED are the empirical entropy of the escape streams.
def esc_step(lam):
    return math.sqrt(6.0 * lam / math.log(2))


def _len_s(q):
    return 1.0 + 2.0 * torch.floor(torch.log2(1.0 + q.abs()))


def _esc_cost(X, step, lam, M=None, w=None):
    q = torch.round(X / step)
    e = X - q * step
    if M is not None:
        e = e @ M
    se = (e * e * w).sum(1) if w is not None else (e * e).sum(1)
    return q, se + lam * _len_s(q).sum(1)


@torch.no_grad()
def encode(X, cb, M=None, w=None):
    """Encode rows of X with codebook entry cb (dict C, pen, dec, lam, step, lesc).
    M: metric (already normalised), w: per-row diagonal weights (hw).  Returns
    (idx int64 with K == ESCAPE, reconstruction (n,d), escape integers (n_esc, d) or None)."""
    C = cb["C"]
    K = C.shape[0]
    if w is not None:
        idx, J = assign_w(X, w, C, cb["pen"], return_min=True)
    else:
        idx, J = assign(X, C, cb["pen"], M, return_min=True)
    if cb["pen"] is None and cb.get("dec") is not None:
        J = J + cb["dec"][idx]
    if cb.get("step") is None:
        return idx, C[idx], None
    q, Je = _esc_cost(X, cb["step"], cb["lam"], M, w)
    use = (Je + cb["lam"] * cb["lesc"]) < J
    rec = torch.where(use[:, None], q * cb["step"], C[idx])
    return torch.where(use, torch.full_like(idx, K), idx), rec, q[use]


@torch.no_grad()
def _update(X, idx, C, w=None):
    """Centroid re-estimation.  Runs on CPU: index_add_ on MPS with few live centroids (heavy atomic
    contention) took 6-8 s per call and stalled for > 10 min inside the full job (2026-11-22)."""
    dev = C.device
    X, idx, C = X.cpu(), idx.cpu(), C.cpu()
    w = None if w is None else w.cpu()
    newC, cnt = _update_cpu(X, idx, C, w)
    return newC.to(dev), cnt.to(dev)


def _update_cpu(X, idx, C, w=None):
    K, d = C.shape
    if w is None:
        cnt = torch.zeros(K, device=X.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        sums = torch.zeros(K, d, device=X.device).index_add_(0, idx, X)
        newC = torch.where(cnt[:, None] > 0, sums / cnt.clamp(min=1)[:, None], C)
    else:   # per-coordinate weights w (n x d): weighted centroid per coordinate
        cnt = torch.zeros(K, device=X.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        ws = torch.zeros(K, d, device=X.device).index_add_(0, idx, w)
        sums = torch.zeros(K, d, device=X.device).index_add_(0, idx, X * w)
        newC = torch.where(ws > 0, sums / ws.clamp(min=1e-30), C)
    return newC, cnt


@torch.no_grad()
def kmeans(X, K, iters=25, seed=0, verbose=False):
    """Plain Lloyd k-means on DEV (fp32).  d=1: quantile init; d>1: random-sample init.
    Empty clusters are re-seeded with random data points."""
    n, d = X.shape
    g = torch.Generator().manual_seed(seed)
    if d == 1:
        qs = (torch.arange(K, dtype=torch.float64) + 0.5) / K
        xs = X[:, 0].cpu().double()
        if xs.numel() > 4_000_000:
            xs = xs[torch.randperm(xs.numel(), generator=g)[:4_000_000]]
        C = torch.quantile(xs, qs).float().unsqueeze(1).to(X.device)
    else:
        C = X[torch.randperm(n, generator=g)[:K].to(X.device)].clone()
    prev = None
    Xc = X.cpu()
    for it in range(iters):
        idx = assign(X, C)
        C, cnt = _update(Xc, idx, C)
        empty = (cnt == 0).nonzero().flatten()
        if empty.numel():
            C[empty] = X[torch.randint(0, n, (empty.numel(),), generator=g).to(X.device)]
        if verbose or it == iters - 1:
            mse = float((X - C[idx]).pow(2).sum() / X.numel())
            if verbose:
                log(f"   km K={K} d={d} it{it} mse={mse:.5f} empty={empty.numel()}")
            prev = mse
    return C, prev


@torch.no_grad()
def km_entry(X, C, escape=True):
    """Codebook entry for PLAIN k-means: nearest-centroid assignment; the escape decision uses
    lam = 2 ln2 * D (high-rate RD slope at the codebook's own per-dim distortion D) and the
    training-set code lengths."""
    n = X.shape[0]
    idx, J = assign(X, C, return_min=True)
    D = float(J.mean()) / X.shape[1]
    cnt = torch.bincount(idx, minlength=C.shape[0]).float()
    lam = 2 * math.log(2) * D
    Lk = -torch.log2(cnt.clamp(min=1) / n)
    cb = {"C": C, "pen": None, "dec": lam * Lk, "lam": lam, "step": None, "lesc": None}
    if escape:
        cb["step"] = esc_step(lam)
        _, Je = _esc_cost(X, cb["step"], lam)
        nesc = int(((Je + lam * 16.0) < (J + cb["dec"][idx])).sum())
        cb["lesc"] = -math.log2(max(nesc, 1) / n)
    return cb


@torch.no_grad()
def ecvq(X, C0, lam, iters=20, verbose=False, escape=True, w=None):
    """Entropy-constrained VQ (Chou-Lookabaugh-Gray): assignment argmin ||x-c||^2 + lam*len(c),
    len = -log2 p(c), alternated with centroid / probability re-estimation; dead centroids are
    dropped.  With escape: vectors whose lattice-escape cost is lower are excluded from the
    centroid update and get the ESCAPE symbol (its probability is re-estimated too).
    w: optional per-vector diagonal metric weights (Hessian-weighted ECVQ)."""
    n = X.shape[0]
    C = C0.clone()
    K = C.shape[0]
    L = torch.full((K,), math.log2(K), device=X.device)
    step = esc_step(lam)
    lesc = 16.0
    if escape:
        _, Je = _esc_cost(X, step, lam, w=w)
    Xc = X.cpu()
    wc = None if w is None else w.cpu()
    for it in range(iters):
        if w is None:
            idx, J = assign(X, C, pen=lam * L, return_min=True)
        else:
            idx, J = assign_w(X, w, C, pen=lam * L, return_min=True)
        if escape:
            keep = ((Je + lam * lesc) >= J).cpu()
            ic = idx.cpu()
            Xk, ik, wk = Xc[keep], ic[keep], (None if wc is None else wc[keep])
        else:
            keep, Xk, ik, wk = None, Xc, idx.cpu(), wc
        C, cnt = _update(Xk, ik, C, w=wk)
        alive = cnt > 0
        L = torch.where(alive, -torch.log2(cnt.clamp(min=1) / n), torch.full_like(cnt, 1e9))
        nesc = 0 if keep is None else int((~keep).sum())
        lesc = -math.log2(max(nesc, 1) / n)
        if verbose:
            p = cnt[alive] / n
            H = float(-(p * torch.log2(p)).sum())
            log(f"   ecvq lam={lam} it{it} K_alive={int(alive.sum())} H/dim={H / X.shape[1]:.3f} esc={nesc / n:.2e}")
    alive = L < 1e8
    pen = (lam * L[alive]).contiguous()
    return {"C": C[alive].contiguous(), "pen": pen, "dec": None, "lam": lam,
            "step": step if escape else None, "lesc": lesc if escape else None}


class Hist:
    """Pooled symbol histogram of a scope (for the shared-frequency-table accounting)."""

    def __init__(self):
        self.c = {}

    def add(self, sym):
        u, c = torch.unique(sym.reshape(-1).cpu(), return_counts=True)
        for a, b in zip(u.tolist(), c.tolist()):
            self.c[a] = self.c.get(a, 0) + b

    def bits(self):
        c = np.array(list(self.c.values()), dtype=np.float64)
        p = c / c.sum()
        return float(c.sum() * -(p * np.log2(p)).sum() + 32 * len(c)), len(c)


class Accounting:
    """Collects, per scope, the pooled histogram, per-tensor coded bits and #weights; plus codebook
    and scale bits.  total() returns the fully-charged bpw for both entropy-table variants."""

    def __init__(self):
        self.h, self.pt_bits, self.nw = {}, {}, {}
        self.cb_bits, self.scale_bits, self.n = 0.0, 0.0, 0
        self.sse, self.ss = 0.0, 0.0

    def add(self, skey, sym, nweights, scale_bits, W=None, deq=None):
        self.h.setdefault(skey, Hist()).add(sym)
        cb, _ = coded_bits(sym)
        self.pt_bits[skey] = self.pt_bits.get(skey, 0.0) + cb
        self.nw[skey] = self.nw.get(skey, 0) + nweights
        self.scale_bits += scale_bits
        self.n += nweights
        if W is not None:
            self.sse += float((W.float() - deq.float()).pow(2).sum())
            self.ss += float(W.float().pow(2).sum())

    def add_stream(self, skey, sym):
        """An extra entropy-coded stream (e.g. escape integers): charged, no weights added."""
        if sym is None or sym.numel() == 0:
            return 0.0
        self.h.setdefault(skey, Hist()).add(sym)
        cb, _ = coded_bits(sym)
        self.pt_bits[skey] = self.pt_bits.get(skey, 0.0) + cb
        self.esc_n = getattr(self, "esc_n", 0) + sym.shape[0]
        return cb

    def total(self):
        pooled = sum(h.bits()[0] for h in self.h.values())
        per_t = sum(self.pt_bits.values())
        idx_bits = min(pooled, per_t)
        tot = idx_bits + self.cb_bits + self.scale_bits
        return {"bpw": tot / self.n,
                "idx_bpw": idx_bits / self.n,
                "idx_bpw_per_tensor_tables": per_t / self.n,
                "idx_bpw_pooled_table": pooled / self.n,
                "table_choice": "pooled" if pooled < per_t else "per_tensor",
                "codebook_bpw": self.cb_bits / self.n,
                "scale_bpw": self.scale_bits / self.n,
                "bpw_harness_per_tensor": (per_t + self.cb_bits + self.scale_bits) / self.n,
                "weight_nmse_diag": self.sse / max(self.ss, 1e-30),
                "n_escaped_vectors": getattr(self, "esc_n", 0),
                "n_weights": self.n}


def gather_train(snap, names, d, n_max, seed=0, dev=DEV, rfn=None):
    """Normalised d-vectors of the given tensors, uniformly subsampled to <= n_max rows, on dev.
    rfn(name) -> optional (O,1) row scale to normalise by (default: the tensor's own row RMS)."""
    g = torch.Generator().manual_seed(seed)
    tot = sum(snap[n].numel() // d for n in names)
    frac = min(1.0, n_max / tot)
    parts = []
    for n in names:
        V, _ = to_vecs(snap[n], d, None if rfn is None else rfn(n))
        if frac < 1.0:
            k = max(1, int(round(V.shape[0] * frac)))
            V = V[torch.randperm(V.shape[0], generator=g)[:k]]
        parts.append(V)
    return torch.cat(parts).to(dev).contiguous(), tot


def fit_codebooks(snap, d, K, scope, method="km", lam=None, iters=25, seed=0, n_train=None,
                  verbose=False, init_cache=None, rfn=None, km_iters=None, ecvq_iters=20, cache_tag="",
                  escape=True):
    """Returns {scope_key: codebook entry dict (C fp16-rounded on DEV, pen, escape params)} + stats."""
    groups = {}
    for n in snap:
        groups.setdefault(scope_key(n, scope), []).append(n)
    n_train = n_train or max(1_000_000, 64 * K)
    n_train = min(n_train, 6_000_000)
    out, stats = {}, {}
    t0 = time.time()
    for sk, names in groups.items():
        X, tot = gather_train(snap, names, d, n_train, seed, rfn=rfn)
        ck = (cache_tag, d, K, sk, seed)
        if init_cache is not None and ck in init_cache:
            C = init_cache[ck]
            mse = None
        else:
            C, mse = kmeans(X, K, iters=km_iters or iters, seed=seed)
            if init_cache is not None:
                init_cache[ck] = C
        if method == "ecvq":
            cb = ecvq(X, C, lam, iters=ecvq_iters, escape=escape)
            cb["C"] = cb["C"].half().float()
        else:
            cb = km_entry(X, C.half().float(), escape=escape)
        out[sk] = cb
        stats[sk] = {"n_train": int(X.shape[0]), "n_vec_scope": int(tot), "K_used": int(cb["C"].shape[0]),
                     "km_train_mse": mse, "lam": cb["lam"], "esc_step": cb["step"], "esc_len_bits": cb["lesc"]}
        del X
    if DEV == "mps":
        torch.mps.empty_cache()
    return out, {"fit_s": round(time.time() - t0, 1), "scopes": stats}


def gain_fix(T, deq_rel):
    """Per-row gain g = |t|^2 / <t, t_hat> that removes the attenuation of conditional-mean
    (centroid) reconstructions: for a centroid codebook <t, t_hat> = |t_hat|^2 < |t|^2, i.e. every
    row is shrunk by ~(1 - NMSE).  Folded into the stored fp16 row scale -> 0 extra bits.
    Returns g (O,1), clamped to [0.5, 2]."""
    num = T.float().pow(2).sum(1, keepdim=True)
    den = (T.float() * deq_rel.float()).sum(1, keepdim=True)
    g = torch.where(den > 0, num / den.clamp(min=1e-30), torch.ones_like(num))
    return g.clamp(0.5, 2.0)


def make_vq_fn(codebooks, d, scope, acct, rfn=None, base=None, gain=False):
    """quantize_model callback: (symbols int32, deq, side_bits).  The codebook bits of a scope are
    charged on its first tensor (also tallied in acct.cb_bits); escape integers are charged as a
    per-scope extra stream (included in side_bits so the harness total is identical).
    rfn(name): optional row scale; base(name): optional additive base (delta coding)."""
    charged = set()

    def fn(n, W):
        sk = scope_key(n, scope)
        cb = codebooks[sk]
        T = W if base is None else (W - base(n))
        V, r = to_vecs(T, d, None if rfn is None else rfn(n))
        idx, rec, esc = encode(V.to(DEV), cb)
        deq = (rec.reshape(W.shape) * r.to(DEV)).cpu()
        if gain:                       # store r' = fp16(r * g) instead of r: same 16 b/row
            r2 = (r * gain_fix(T, deq)).half().float()
            deq = (rec.reshape(W.shape) * r2.to(DEV)).cpu()
        if base is not None:
            deq = deq + base(n)
        sym = idx.to(torch.int32).cpu()
        side = 16 * W.shape[0]
        sb = side
        if sk not in charged:
            charged.add(sk)
            cbb = 16 * cb["C"].shape[0] * d
            acct.cb_bits += cbb
            side += cbb
        acct.add(sk, sym, W.numel(), sb, W, deq)
        if esc is not None and esc.numel():
            side += acct.add_stream(sk + "#esc", esc.to(torch.int32).cpu())
        return sym, deq, side
    return fn


# ----------------------------------------------------------------------------------------------
# V2: Hessian-aware VQ
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def assign_w(X, Wt, C, pen=None, return_min=False):
    """Diagonal-weighted assignment argmin_k sum_j w_j (x_j - c_kj)^2 + pen_k (w per row of X)."""
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    mins = torch.empty(X.shape[0], device=X.device) if return_min else None
    ch = _chunk(C.shape[0])
    Ct, C2t = C.T.contiguous(), (C * C).T.contiguous()
    for i in range(0, X.shape[0], ch):
        x, w = X[i:i + ch], Wt[i:i + ch]
        dist = w @ C2t - 2.0 * ((w * x) @ Ct)
        if pen is not None:
            dist += pen[None, :]
        if return_min:
            m, a = dist.min(1)
            out[i:i + ch] = a
            mins[i:i + ch] = m + (w * x * x).sum(1)
        else:
            out[i:i + ch] = dist.argmin(1)
    return (out, mins) if return_min else out


class HU:
    """Reduced-memory Hessian: the GPTQ factor U = chol(H^-1, upper) computed in fp64 from the full-precision H
    and stored in bf16 (no positive-definiteness requirement on U), plus diag(H) in fp32."""

    def __init__(self, H, percdamp=0.01):
        U, dead = hinv_chol(H, percdamp)
        self.U, self.dead, self.diag = U.to(torch.bfloat16), dead, torch.diag(H).float().clone()


def hdiag(H):
    return H.diag if isinstance(H, HU) else torch.diag(H).float()


def hdiag_weights(H, d, O):
    """Per-vector diagonal weights (O*I/d, d): H_jj / mean(diag H), same for every row."""
    h = hdiag(H).clamp(min=0)
    h = h / h.mean().clamp(min=1e-30)
    I = h.numel()
    return h.reshape(1, I // d, d).expand(O, I // d, d).reshape(-1, d)


@torch.no_grad()
def hinv_chol(H, percdamp=0.01):
    """Upper Cholesky factor U of H^-1 (GPTQ convention; dead inputs get H_jj = 1).  If H + damp is not
    positive definite (reduced-precision storage), the damping is raised x10 up to 3 times."""
    if isinstance(H, HU):
        return H.U.float(), H.dead
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    md = torch.mean(torch.diag(H))
    for k in range(4):
        try:
            Hd = H + percdamp * (10 ** k) * md * torch.eye(H.shape[0], dtype=H.dtype)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))
            return torch.linalg.cholesky(Hinv, upper=True).float(), dead
        except torch._C._LinAlgError:
            if k == 3:
                raise
            log(f"   hinv_chol: not PD at damp {percdamp * 10 ** k:g} -> x10")


@torch.no_grad()
def gptvq(W, H, cb, d, blocksize=128, dev=DEV):
    """GPTQ with joint d-column vector quantization (GPTVQ-style, no codebook fine-tuning).
    For a block S of d columns with U = chol(H^-1, upper): loss of error e (row vector) =
    ||e inv(U_SS)||^2 (exact OBS for joint removal of S); compensation of the not-yet-quantized
    columns R: W_R -= (e inv(U_SS)) U_SR.  Works in row-RMS-normalised coordinates (row scaling
    commutes with the column-space update).  The assignment metric is normalised per tensor so
    that its mean diagonal weight is 1 (keeps lam in the units of the unweighted ECVQ).
    Codebook entry cb as in encode() (escape to the lattice included).
    d=1 with a uniform codebook and no penalty reproduces GPTQ-ECSQ (checked to 2e-7).
    Returns (symbols int32 [O*I/d], deq W, escape integers (n_esc, d) int32)."""
    U, dead = hinv_chol(H)
    r = row_scale(W)
    V = (W.float() / r)
    V[:, dead] = 0
    O, I = V.shape
    V, U = V.to(dev), U.to(dev)
    sq = math.sqrt(float((1.0 / torch.diag(U).pow(2)).mean()))
    Q = torch.zeros_like(V)
    idx = torch.zeros(O, I // d, dtype=torch.long, device=dev)
    escs = []
    for i1 in range(0, I, blocksize):
        i2 = min(i1 + blocksize, I)
        V1 = V[:, i1:i2].clone()
        E1 = torch.zeros_like(V1)
        U1 = U[i1:i2, i1:i2]
        for j in range(0, i2 - i1, d):
            Uss = U1[j:j + d, j:j + d]
            M = torch.linalg.inv(Uss.cpu()).to(dev) if d > 1 else (1.0 / Uss)
            x = V1[:, j:j + d]
            k, q, esc = encode(x, cb, M=M / sq)
            e = (x - q) @ M
            Q[:, i1 + j:i1 + j + d] = q
            idx[:, (i1 + j) // d] = k
            if esc is not None and esc.numel():
                escs.append(esc)
            if j + d < i2 - i1:
                V1[:, j + d:] -= e @ U1[j:j + d, j + d:]
            E1[:, j:j + d] = e
        if i2 < I:
            V[:, i2:] -= E1 @ U[i1:i2, i2:]
    esc = torch.cat(escs).to(torch.int32).cpu() if escs else None
    return idx.reshape(-1).to(torch.int32).cpu(), (Q.cpu() * r), esc


# ----------------------------------------------------------------------------------------------
# Lattice VQ (structured, codebook-free, unbounded, unbiased rounding): D4 and E8 (QuIP#-style
# E8 but entropy coded).  Nearest points: Conway & Sloane, "Fast quantizing and decoding
# algorithms for lattice quantizers" (1982).  Entropy coding: per-coordinate symbol streams;
# the parity constraint makes the last coordinate's low bit implicit (k = (y_d - parity)/2),
# E8 adds one coset bit per vector.  Normalised second moments: Z 0.08333, D4 0.07660 (-0.37 dB,
# 0.061 bit/dim), E8 0.07168 (-0.65 dB, 0.108 bit/dim)  -> the ONLY high-rate edge of VQ over ECSQ.
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def dn_nearest(x):
    """Nearest point of D_n = {y in Z^n : sum(y) even} for every row of x."""
    f = torch.round(x)
    odd = (f.sum(1).remainder(2) != 0)
    if odd.any():
        diff = x - f
        j = diff.abs().argmax(1)
        rows = odd.nonzero().flatten()
        dj = diff[rows, j[rows]]
        f[rows, j[rows]] += torch.where(dj >= 0, 1.0, -1.0)
    return f


@torch.no_grad()
def e8_nearest(x):
    """Nearest point of E8 = D8 u (D8 + 1/2)."""
    y0 = dn_nearest(x)
    y1 = dn_nearest(x - 0.5) + 0.5
    d0 = (x - y0).pow(2).sum(1)
    d1 = (x - y1).pow(2).sum(1)
    return torch.where((d1 < d0)[:, None], y1, y0), (d1 < d0)


@torch.no_grad()
def lattice_nearest(x, kind):
    """Nearest lattice point (lattice units)."""
    if kind == "d4":
        return dn_nearest(x)
    if kind == "e8":
        return e8_nearest(x)[0]
    raise ValueError(kind)


@torch.no_grad()
def lattice_symbols(y, kind):
    """Integer symbol streams of lattice points y (lattice units): head coords, last coord with the
    parity bit removed, and (E8) the coset bit."""
    coset = None
    if kind == "e8":
        coset = (y[:, 0] - torch.floor(y[:, 0])).abs() > 0.25     # half-integer coset
        z = y - 0.5 * coset[:, None].float()
    else:
        z = y
    d = z.shape[1]
    head = z[:, :d - 1]
    par = head.sum(1).remainder(2)
    k = (z[:, d - 1] - par) / 2
    streams = [head.reshape(-1).to(torch.int32), k.to(torch.int32)]
    if coset is not None:
        streams.append(coset.to(torch.int32))
    return streams


@torch.no_grad()
def lattice_encode(V, kind, s):
    """V: (n, d) normalised vectors.  Returns (list of int32 symbol streams, reconstruction)."""
    y = lattice_nearest(V / s, kind)
    return lattice_symbols(y, kind), y * s


@torch.no_grad()
def gptq_lattice(W, H, kind, s, blocksize=128, dev=DEV):
    """GPTQ error feedback with joint d-column LATTICE quantization (d=4 D4 / d=8 E8, step s in
    row-normalised units).  The block is rounded to the Euclidean-nearest lattice point (not the
    metric-nearest one - suboptimal assignment, exact OBS compensation W_R -= (e inv(U_SS)) U_SR).
    Returns (list of symbol streams, deq W)."""
    d = 4 if kind == "d4" else 8
    U, dead = hinv_chol(H)
    r = row_scale(W)
    V = (W.float() / r)
    V[:, dead] = 0
    O, I = V.shape
    V, U = V.to(dev), U.to(dev)
    Y = torch.zeros(O, I, device=dev)
    for i1 in range(0, I, blocksize):
        i2 = min(i1 + blocksize, I)
        V1 = V[:, i1:i2].clone()
        E1 = torch.zeros_like(V1)
        U1 = U[i1:i2, i1:i2]
        for j in range(0, i2 - i1, d):
            M = torch.linalg.inv(U1[j:j + d, j:j + d].cpu()).to(dev)
            x = V1[:, j:j + d]
            y = lattice_nearest(x / s, kind)
            e = (x - y * s) @ M
            Y[:, i1 + j:i1 + j + d] = y
            if j + d < i2 - i1:
                V1[:, j + d:] -= e @ U1[j:j + d, j + d:]
            E1[:, j:j + d] = e
        if i2 < I:
            V[:, i2:] -= E1 @ U[i1:i2, i2:]
    streams = [t.cpu() for t in lattice_symbols(Y.reshape(-1, d), kind)]
    return streams, (Y.cpu() * s * r)


@torch.no_grad()
def gptq_joint_search(W, H, kind, s, blocksize=128, dev=DEV):
    """Ablation for GPTVQ: NO codebook, plain grid (kind 'z4' = scalar grid Z^4, 'd4' = D4 lattice) but the
    4-column block is rounded JOINTLY under the block's Hessian metric: candidates = the 3^4 grid points
    around the Euclidean rounding (D4: those with even sum), pick argmin ||(x - s*y) inv(U_SS)||^2, then the
    same exact joint OBS update as gptvq().  Isolates 'metric-aware vector decisions' from 'learned codebook'.
    Symbols: Z^4 -> per-coordinate integers (like scalar GPTQ); D4 -> lattice_symbols."""
    import itertools
    d = 4
    offs = torch.tensor(list(itertools.product([-1.0, 0.0, 1.0], repeat=d)), device=dev)      # (81, 4)
    U, dead = hinv_chol(H)
    r = row_scale(W)
    V = (W.float() / r)
    V[:, dead] = 0
    O, I = V.shape
    V, U = V.to(dev), U.to(dev)
    Y = torch.zeros(O, I, device=dev)
    for i1 in range(0, I, blocksize):
        i2 = min(i1 + blocksize, I)
        V1 = V[:, i1:i2].clone()
        E1 = torch.zeros_like(V1)
        U1 = U[i1:i2, i1:i2]
        for j in range(0, i2 - i1, d):
            M = torch.linalg.inv(U1[j:j + d, j:j + d].cpu()).to(dev)
            x = V1[:, j:j + d]
            base = torch.round(x / s)
            cand = base[:, None, :] + offs[None]                                   # (O, 81, 4)
            err = (x[:, None, :] - s * cand) @ M                                   # (O, 81, 4)
            cost = err.pow(2).sum(-1)
            if kind == "d4":
                cost = cost.masked_fill(cand.sum(-1).remainder(2) != 0, float("inf"))
            y = cand[torch.arange(O, device=dev), cost.argmin(1)]
            e = (x - s * y) @ M
            Y[:, i1 + j:i1 + j + d] = y
            if j + d < i2 - i1:
                V1[:, j + d:] -= e @ U1[j:j + d, j + d:]
            E1[:, j:j + d] = e
        if i2 < I:
            V[:, i2:] -= E1 @ U[i1:i2, i2:]
    if kind == "d4":
        streams = [t.cpu() for t in lattice_symbols(Y.reshape(-1, d), "d4")]
    else:
        streams = [Y.reshape(-1).to(torch.int32).cpu()]
    return streams, (Y.cpu() * s * r)
