"""A1 PROBE (JPEG-intra transform): does an 8x8 block-DCT compact weight energy, raw / after
row+col REORDER (norm sort, spectral/Fiedler, greedy TSP on |corr| with sign flips) /
after random-sign block-HADAMARD rotation?

Metrics per (matrix, variant):
  * topk_energy: mean fraction of block energy in the k largest coefficients (k=1,4,8,16)
    for DCT vs identity (raw values, same blocks).
  * coding_gain_dB: AM/GM of the 64 per-position coefficient variances (classic transform
    coding gain; identity-transform value reported as the reference).
  * ECSQ test (decisive): uniform step D (same in both domains; orthonormal -> same MSE),
    per-position (64 histograms, JPEG-style) entropy of the DCT symbols vs entropy of the
    raw symbols; measured MSE reported. delta_bits = bits(DCT) - bits(raw)  (<0 = DCT wins)
Controls: iid Gaussian null (spurious gain from reordering), smooth AR(1) field rho=0.95
(positive control; DCT must win big) and the same field with rows/cols shuffled (TSP must
recover it).  All matrices are row-RMS-normalised first (the baseline coder already has a
per-row step).

PRE-REGISTERED: A1 PASS iff some variant gives delta_bits <= -0.10 bit/weight on the
majority of real matrices AND beats the same variant on the Gaussian null by >= 0.05.
"""
from common import *
import scipy.fft
import scipy.linalg

torch.manual_seed(0); np.random.seed(0)
RID = "Qwen/Qwen2.5-0.5B"


def dct_mat(n=8):
    return scipy.fft.dct(np.eye(n), norm="ortho", axis=0)


C8 = dct_mat(8)


def blocks(M):
    r, c = (M.shape[0] // 8) * 8, (M.shape[1] // 8) * 8
    M = M[:r, :c]
    return M.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)


def dct_blocks(B):
    return np.einsum("ij,njk,lk->nil", C8, B, C8).reshape(len(B), 64)


def topk_frac(X, ks=(1, 4, 8, 16)):
    e = np.sort(X ** 2, axis=1)[:, ::-1]
    tot = e.sum(1) + 1e-30
    c = np.cumsum(e, 1)
    return {k: float((c[:, k - 1] / tot).mean()) for k in ks}


def coding_gain_db(X):
    v = (X ** 2).mean(0) + 1e-30
    return float(10 * np.log10(v.mean() / np.exp(np.log(v).mean())))


def H_sym(q):
    _, c = np.unique(q, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def ecsq_test(Xraw, Xdct, step):
    qr = np.round(Xraw / step); qd = np.round(Xdct / step)
    b_raw = H_sym(qr)                                       # one histogram (raw is ~stationary)
    b_raw_pos = float(np.mean([H_sym(qr[:, k]) for k in range(64)]))
    b_dct = float(np.mean([H_sym(qd[:, k]) for k in range(64)]))   # JPEG-style per position
    mse_r = float(((qr * step - Xraw) ** 2).mean()); mse_d = float(((qd * step - Xdct) ** 2).mean())
    return {"bits_raw": b_raw, "bits_raw_perpos": b_raw_pos, "bits_dct": b_dct,
            "delta_bits": b_dct - min(b_raw, b_raw_pos), "mse_raw": mse_r, "mse_dct": mse_d}


def corr_rows(M):
    Z = M - M.mean(1, keepdims=True)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z @ Z.T


def order_norm(M, axis):
    n = np.linalg.norm(M, axis=1 - axis)
    return np.argsort(n)


def order_spectral(M, knn=8):
    """Fiedler ordering of the kNN-sparsified |corr| graph (dense |corr| graphs are dominated
    by the sea of ~1/sqrt(d) noise correlations and fail even on the positive control)."""
    A = np.abs(corr_rows(M)); np.fill_diagonal(A, 0)
    idx = np.argsort(-A, axis=1)[:, :knn]
    S = np.zeros_like(A)
    np.put_along_axis(S, idx, np.take_along_axis(A, idx, 1), 1)
    A = np.maximum(S, S.T)
    L = np.diag(A.sum(1)) - A
    w, v = scipy.linalg.eigh(L, subset_by_index=[1, 1])
    return np.argsort(v[:, 0])


def order_tsp(M):
    """greedy nearest-neighbour tour on |corr|; returns order and sign flips."""
    C = corr_rows(M)
    n = len(C)
    vis = np.zeros(n, bool); order = [0]; sign = np.ones(n); vis[0] = True
    cur = 0
    A = np.abs(C)
    for _ in range(n - 1):
        a = A[cur].copy(); a[vis] = -1
        nxt = int(a.argmax())
        sign[nxt] = sign[cur] * (1 if C[cur, nxt] >= 0 else -1)
        vis[nxt] = True; order.append(nxt); cur = nxt
    return np.array(order), sign


def reorder(M, how):
    if how == "raw":
        return M
    if how == "norm":
        return M[order_norm(M, 0)][:, order_norm(M, 1)]
    if how == "spectral":
        M = M[order_spectral(M)]
        return M[:, order_spectral(M.T)]
    if how == "tsp":
        o, s = order_tsp(M); M = M[o] * s[o][:, None]
        o, s = order_tsp(M.T); M = M[:, o] * s[o][None, :]
        return M
    raise ValueError(how)


def block_hadamard(n, bs=128):
    H = scipy.linalg.hadamard(bs) / np.sqrt(bs)
    k = n // bs
    D = np.random.choice([-1.0, 1.0], n)
    return np.kron(np.eye(k), H) * D[None, :]


def hadamard_rotate(M):
    o, i = M.shape
    return block_hadamard(o) @ M @ block_hadamard(i).T


def analyse(M, tag, step_rel=0.5):
    M = M / (np.sqrt((M ** 2).mean(1, keepdims=True)) + 1e-12)      # row-RMS normalise
    out = {}
    variants = [("raw", lambda X: X), ("norm", lambda X: reorder(X, "norm")),
                ("spectral", lambda X: reorder(X, "spectral")), ("tsp", lambda X: reorder(X, "tsp")),
                ("hadamard", hadamard_rotate)]
    for vname, f in variants:
        t = time.time()
        X = f(M)
        B = blocks(X)
        Xr = B.reshape(len(B), 64); Xd = dct_blocks(B)
        rec = {"topk_id": topk_frac(Xr), "topk_dct": topk_frac(Xd),
               "cg_id_dB": coding_gain_db(Xr), "cg_dct_dB": coding_gain_db(Xd),
               "ecsq": ecsq_test(Xr, Xd, step_rel), "s": round(time.time() - t, 1)}
        out[vname] = rec
        log(f"{tag:28s} {vname:9s} top8 id={rec['topk_id'][8]:.3f} dct={rec['topk_dct'][8]:.3f} "
            f"CG id={rec['cg_id_dB']:.2f}dB dct={rec['cg_dct_dB']:.2f}dB  dbits={rec['ecsq']['delta_bits']:+.3f}")
    return out


def ar1_field(n, m, rho=0.95):
    x = np.random.randn(n, m)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * x[i]
    for j in range(1, m):
        x[:, j] = rho * x[:, j - 1] + np.sqrt(1 - rho ** 2) * x[:, j]
    return x


if __name__ == "__main__":
    store = Store("probe_a1")
    sd = load_sd(RID, keys=lambda k: k.startswith("model.layers.") and k.endswith("proj.weight"))
    mats = {}
    for L in (2, 12, 22):
        for p in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
            mats[f"L{L}.{p}"] = sd[f"model.layers.{L}.{p}.weight"].numpy().astype(np.float64)
    del sd
    ctrl = {"NULL_gauss_896x896": np.random.randn(896, 896),
            "NULL_gauss_4864x896": np.random.randn(4864, 896),
            "POS_ar1_896x896": ar1_field(896, 896)}
    sh = ar1_field(896, 896)
    ctrl["POS_ar1_shuffled_896x896"] = sh[np.random.permutation(896)][:, np.random.permutation(896)]
    for name, M in list(ctrl.items()) + list(mats.items()):
        if store.has(name):
            continue
        store.put(name, {"shape": list(M.shape), **analyse(M, name)})

    # ---- verdict
    real = [k for k in store.d if k.startswith("L")]
    verdict = {}
    for v in ("raw", "norm", "spectral", "tsp", "hadamard"):
        db = np.array([store.d[k][v]["ecsq"]["delta_bits"] for k in real])
        nulls = [store.d[k][v]["ecsq"]["delta_bits"] for k in store.d if k.startswith("NULL")]
        verdict[v] = {"median_delta_bits_real": float(np.median(db)), "frac_real_le_-0.10": float((db <= -0.10).mean()),
                      "null_delta_bits": float(np.mean(nulls)),
                      "pass": bool((db <= -0.10).mean() > 0.5 and np.median(db) <= np.mean(nulls) - 0.05)}
    verdict["A1"] = "PASS" if any(v["pass"] for v in verdict.values() if isinstance(v, dict)) else "FAIL"
    verdict["pos_control_delta_bits_raw"] = store.d["POS_ar1_896x896"]["raw"]["ecsq"]["delta_bits"]
    verdict["pos_control_shuffled_tsp_delta_bits"] = store.d["POS_ar1_shuffled_896x896"]["tsp"]["ecsq"]["delta_bits"]
    store.put("_verdict", verdict)
    log(json.dumps(verdict, indent=1))
