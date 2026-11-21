"""B1 PROBE (layers = frames) + B3-lite (motion compensation = neuron alignment).

For every adjacent pair of blocks (k -> k+1) and every same-named matrix:
  cos(W_k, W_k+1), ||W_k+1 - W_k|| / ||W_k+1||, and the optimal-alpha residual energy
  r = ||W_k+1 - a W_k||^2 / ||W_k+1||^2 = 1 - cos^2.
Motion compensation (exact network symmetries only):
  MLP: permutation of the 4864 intermediate neurons (rows of gate/up, cols of down) by
       Hungarian matching on concatenated [gate_j, up_j, down_:,j] cosine.
  Attn V/O: per-KV-group orthogonal Procrustes R (64x64) on [V_g ; O_i^T for its 7 q-heads]
       (+ best of the 2 KV-group permutations). q/k: no alignment (RoPE-constrained), raw only.
Null: the same alignment between layer k+1 and a Gaussian matrix with identical row norms
(how much 'gain' the matching manufactures out of nothing).
High-rate predicted saving of predictive coding = -0.5*log2(r) bit/weight.

PRE-REGISTERED: B PASS iff over middle blocks (6..17) the numel-weighted aligned residual
r <= 0.70 (>= 0.26 bit/weight) AND it is >= 0.10 below the aligned-null residual.
"""
from common import *
from scipy.optimize import linear_sum_assignment

RID = "Qwen/Qwen2.5-0.5B"
torch.manual_seed(0); np.random.seed(0)
sd = load_sd(RID, keys=lambda k: k.startswith("model.layers.") and k.endswith("proj.weight"))
NL = 24
P = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
W = {(l, p): sd[f"model.layers.{l}.{p}.weight"].double() for l in range(NL) for p in P}
del sd


def resid(a, b):
    """optimal-alpha residual energy ratio of predicting b from a, plus cos and plain rel diff."""
    a, b = a.flatten(), b.flatten()
    c = float(a @ b / (a.norm() * b.norm()))
    return {"cos": c, "r_alpha": 1 - c * c, "rel_diff": float((b - a).norm() / b.norm())}


def gauss_like(M):
    G = torch.randn_like(M)
    return G * (M.norm(dim=1, keepdim=True) / G.norm(dim=1, keepdim=True))


def mlp_align(gk, uk, dk, g1, u1, d1):
    def feats(g, u, d):
        F = torch.cat([g, u, d.T], 1)
        return F / F.norm(dim=1, keepdim=True)
    S = (feats(g1, u1, d1) @ feats(gk, uk, dk).T).numpy()
    r, c = linear_sum_assignment(-S)
    perm = torch.as_tensor(c)           # neuron i of k+1  <- neuron perm[i] of k
    return gk[perm], uk[perm], dk[:, perm], float(S[r, c].mean())


def vo_align(vk, ok, v1, o1, nkv=2, hd=64, nq=14):
    grp = nq // nkv
    best = None
    for gperm in ([0, 1], [1, 0]):
        vk_new, ok_new = torch.zeros_like(vk), torch.zeros_like(ok)
        for g in range(nkv):
            src = gperm[g]
            Fk = [vk[src * hd:(src + 1) * hd]] + [ok[:, (src * grp + i) * hd:(src * grp + i + 1) * hd].T for i in range(grp)]
            F1 = [v1[g * hd:(g + 1) * hd]] + [o1[:, (g * grp + i) * hd:(g * grp + i + 1) * hd].T for i in range(grp)]
            Fk, F1 = torch.cat(Fk, 1), torch.cat(F1, 1)
            U, _, Vt = torch.linalg.svd(F1 @ Fk.T)
            R = U @ Vt
            vk_new[g * hd:(g + 1) * hd] = R @ vk[src * hd:(src + 1) * hd]
            for i in range(grp):
                ok_new[:, (g * grp + i) * hd:(g * grp + i + 1) * hd] = ok[:, (src * grp + i) * hd:(src * grp + i + 1) * hd] @ R.T
        score = resid(vk_new, v1)["r_alpha"] + resid(ok_new, o1)["r_alpha"]
        if best is None or score < best[0]:
            best = (score, vk_new, ok_new)
    return best[1], best[2]


store = Store("probe_b1")
for k in range(NL - 1):
    key = f"{k}->{k+1}"
    if store.has(key):
        continue
    t = time.time()
    rec = {p: {"raw": resid(W[(k, p)], W[(k + 1, p)])} for p in P}
    # cross-type sanity: is block k+1 better predicted by block k than by a random far block?
    far = (k + 12) % NL
    rec["mlp.gate_proj"]["raw_far_layer"] = resid(W[(far, "mlp.gate_proj")], W[(k + 1, "mlp.gate_proj")])
    # MLP neuron alignment, real and null
    g, u, d, _ = mlp_align(W[(k, "mlp.gate_proj")], W[(k, "mlp.up_proj")], W[(k, "mlp.down_proj")],
                        W[(k + 1, "mlp.gate_proj")], W[(k + 1, "mlp.up_proj")], W[(k + 1, "mlp.down_proj")])
    gn, un, dn = (gauss_like(W[(k, "mlp.gate_proj")]), gauss_like(W[(k, "mlp.up_proj")]),
                  gauss_like(W[(k, "mlp.down_proj")].T).T)   # null keeps per-neuron norms
    g0, u0, d0, _ = mlp_align(gn, un, dn, W[(k + 1, "mlp.gate_proj")], W[(k + 1, "mlp.up_proj")], W[(k + 1, "mlp.down_proj")])
    for p, a, a0 in (("mlp.gate_proj", g, g0), ("mlp.up_proj", u, u0), ("mlp.down_proj", d, d0)):
        rec[p]["aligned"] = resid(a, W[(k + 1, p)])
        rec[p]["aligned_null"] = resid(a0, W[(k + 1, p)])
    v, o = vo_align(W[(k, "self_attn.v_proj")], W[(k, "self_attn.o_proj")], W[(k + 1, "self_attn.v_proj")], W[(k + 1, "self_attn.o_proj")])
    v0, o0 = vo_align(gauss_like(W[(k, "self_attn.v_proj")]), gauss_like(W[(k, "self_attn.o_proj")]),
                      W[(k + 1, "self_attn.v_proj")], W[(k + 1, "self_attn.o_proj")])
    for p, a, a0 in (("self_attn.v_proj", v, v0), ("self_attn.o_proj", o, o0)):
        rec[p]["aligned"] = resid(a, W[(k + 1, p)])
        rec[p]["aligned_null"] = resid(a0, W[(k + 1, p)])
    store.put(key, rec)
    log(key, " ".join(f"{p.split('.')[1][:4]}:raw={rec[p]['raw']['r_alpha']:.3f}" +
                      (f"/al={rec[p]['aligned']['r_alpha']:.3f}/null={rec[p]['aligned_null']['r_alpha']:.3f}" if "aligned" in rec[p] else "")
                      for p in P), f"{time.time() - t:.0f}s")

# ---- verdict over middle blocks
num = {p: W[(0, p)].numel() for p in P}
def wavg(field, ks):
    tot, acc = 0, 0
    for k in ks:
        for p in P:
            r = store.d[f"{k}->{k+1}"][p]
            v = r.get(field, r["raw"])["r_alpha"]
            acc += num[p] * v; tot += num[p]
    return acc / tot
mid = range(6, 18)
best_r = wavg("aligned", mid); null_r = wavg("aligned_null", mid); raw_r = wavg("raw", mid)
verdict = {"mid_raw_r": raw_r, "mid_aligned_r": best_r, "mid_aligned_null_r": null_r,
           "pred_saving_bits_aligned": -0.5 * math.log2(best_r),
           "pred_saving_bits_vs_null": -0.5 * math.log2(best_r) + 0.5 * math.log2(null_r),
           "max_raw_cos_any": max(abs(store.d[f"{k}->{k+1}"][p]["raw"]["cos"]) for k in range(NL - 1) for p in P),
           "B1": "PASS" if (best_r <= 0.70 and best_r <= null_r - 0.10) else "FAIL"}
store.put("_verdict", verdict)
log(json.dumps(verdict, indent=1))
