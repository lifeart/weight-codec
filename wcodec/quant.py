"""Quantizers that expose everything the bitstream must store (symbols, steps, scales, zeros).

Same algorithms as phase 1 (scripts/common.py: rtn_asym / ecsq / gptq) -- only extended to
return the per-group scale/zero tensors so that they can be written to the file.
"""
import torch


def fp16r(t):
    return t.half().float()


def rms_row(W):
    return W.float().pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)


def ecsq_step(W, c):
    """Per-row ECSQ step s = c * rms(row), fp16-representable, shape [O, 1]."""
    return fp16r(c * rms_row(W))


def ecsq(W, step):
    return torch.round(W.float() / step).to(torch.int32)


def rtn_g128(W, bits=4, group=128):
    W = W.float()
    O, I = W.shape
    g = I if group <= 0 else group
    maxq = 2 ** bits - 1
    Wg = W.reshape(O, I // g, g)
    xmin = Wg.amin(-1, keepdim=True).clamp(max=0)
    xmax = Wg.amax(-1, keepdim=True).clamp(min=0)
    xmax = torch.where((xmax - xmin) == 0, xmax + 1e-8, xmax)
    scale = fp16r((xmax - xmin) / maxq)
    zero = torch.round(-xmin / scale).clamp(0, maxq)
    q = torch.clamp(torch.round(Wg / scale) + zero, 0, maxq)
    return q.reshape(O, I).to(torch.int32), scale.reshape(O, I // g), zero.reshape(O, I // g).to(torch.int32)


@torch.no_grad()
def gptq(W, H, mode, bits=4, group=128, step=None, blocksize=128, percdamp=0.01, dev="cpu"):
    """GPTQ (non-sequential, Hessian from the fp model).  mode 'rtn': asym min/max g128 grid
    (params computed lazily per group from the error-updated weights, as in reference GPTQ);
    mode 'grid': fixed uniform grid `step` [O,1] per row (GPTQ-ECSQ / GPTQ on a residual) or [O,I] (ghcol).
    Returns symbols int32 [O,I], dequantized fp32 [O,I], scale [O,I/g] | None, zero | None."""
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
    scales, zeros = [], []
    scale = zero = None
    if mode == "grid":
        stp = step.to(dev).float().expand(O, I)
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
                    blk = W[:, col:col + g]
                    xmin = blk.amin(1).clamp(max=0)
                    xmax = blk.amax(1).clamp(min=0)
                    xmax = torch.where(xmax - xmin == 0, xmax + 1e-8, xmax)
                    scale = fp16r((xmax - xmin) / maxq)
                    zero = torch.round(-xmin / scale).clamp(0, maxq)
                    scales.append(scale)
                    zeros.append(zero)
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
    if mode == "rtn":
        return S.cpu(), Q.cpu(), torch.stack(scales, 1).cpu(), torch.stack(zeros, 1).to(torch.int32).cpu()
    return S.cpu(), Q.cpu(), None, None


FP16_TINY = 2.0 ** -24          # smallest positive fp16 (subnormal)
FP16_MAX = 65504.0


def hcol_factor(hdiag, gamma=1.0):
    """Column factors of the Hessian-column ("hcol") grid: fp16((h_j / geomean(h))^(-gamma/2)), h = diag(H) =
    E[x_j^2] of the layer input.  Sensitive input channels (large h_j) get finer steps.  Normalised to geometric
    mean 1 per matrix, so c keeps the meaning of the row-ECSQ step c*rms_row.  Clamped to the positive finite
    fp16 range (only bites on degenerate columns, e.g. h_j = 0 -> GPTQ zeroes those weights anyway)."""
    hd = hdiag.float().clamp(min=1e-12)
    hd = hd / torch.exp(torch.log(hd).mean())
    return fp16r(hd.pow(-gamma / 2).clamp(max=FP16_MAX)).clamp(min=FP16_TINY)


def hcol_grid(W, hdiag, c, gamma=1.0, rowref=None):
    """Separable step grid step_ij = row_i * col_j: row = fp16(c * rms_row(rowref or W)) [O,1] (floored at the
    smallest fp16 so an all-zero row cannot give step 0), col = hcol_factor [I].  Both are stored as fp16 side
    info (16 b/row + 16 b/col); their product is exact in fp32, so a decoder rebuilds the identical lattice."""
    row = fp16r(c * rms_row(W if rowref is None else rowref)).clamp(min=FP16_TINY)
    return row, hcol_factor(hdiag, gamma)


SYM_MAX = 32767                 # every qrow decoder path (rans16 / Metal / parallel / JS) holds symbols in int16
SYM_SAFE = 2 ** 14              # grid guarantee before GPTQ: |W / step| <= 16384 (2x headroom for error feedback)


def int16_safe_col(W, row, col, limit=SYM_SAFE):
    """Raise the column factors where a column's finest step would need lattice indices beyond `limit`:
    col_j >= max_i |W_ij| / (row_i * limit).  Only bites on hyper-sensitive columns (SmolLM2-360M: col factors down
    to 0.0014 gave |q| = 34798 in one down_proj, which an int16 decoder would wrap); a step below max|W|/16384 of its
    column buys nothing measurable.  Returns fp16-valued col and the number of raised columns."""
    need = (W.float().abs() / row).amax(0) / limit
    raised = need > col
    if bool(raised.any()):
        col = torch.where(raised, fp16r(need * (1 + 2.0 ** -9)), col)     # round-up margin for the fp16 cast
    return col, int(raised.sum())


def sym_fits_int16(S):
    return int(S.abs().max()) <= SYM_MAX if S.numel() else True


@torch.no_grad()
def ghcol(W, H, c, gamma=1.0, rowref=None, blocksize=128, percdamp=0.01, dev="cpu"):
    """"ghcol" = GPTQ error feedback on the Hessian-COLUMN step grid (hcol_grid) -- scripts/ghcol_probe.py,
    REPORT §16.  The error feedback pays only on this grid: on the plain row grid GPTQ hurt SmolLM2 (REPORT §15.4).
    W may be a weight matrix or a fine-tune delta (rowref = the matrix whose row RMS sets the row step).
    Returns symbols int32 [O,I], dequantized fp32 [O,I] (= S * row * col exactly as the decoder computes it),
    row [O,1], col [I]."""
    row, col = hcol_grid(W, torch.diag(H), c, gamma, rowref)
    col, _ = int16_safe_col(W, row, col)
    S, Qw, _, _ = gptq(W, H, "grid", step=row * col[None, :], blocksize=blocksize, percdamp=percdamp, dev=dev)
    return S, Qw, row, col


def fisher_grid(rms, stats, gamma=0.5):
    """A3 separable FISHER allocation (REPORT §3, scripts/a3_fisher.py) in the decodable fp16 row x fp16 column form
    of hcol_grid: log s_ij = log c + (1-g) log rms_i + g (ur_i + uc_j + shift), ur = -1/2 log(F_row/mean F),
    uc = -1/2 log F_col; shift is GLOBAL (cross-matrix allocation: the weight-averaged log step equals that of
    ECSQ_row at the same c).  rms: {name: rms_row [O] of W}; stats: {name: {"row", "col", "mean"}} empirical Fisher
    row/column means (p2common.fisher_stats).  Returns {name: (row log-terms [O] fp64 x2, col [I] fp16-valued)};
    the per-c grid is fisher_rowcol(entry, c)."""
    import math
    ntot = sum(rms[n].numel() * stats[n]["col"].numel() for n in rms)
    L0 = sum(float(torch.log(rms[n].double()).sum()) * stats[n]["col"].numel() for n in rms) / ntot
    logu, acc = {}, 0.0
    for n in rms:
        fr = stats[n]["row"].double().clamp(min=1e-30)
        fc = stats[n]["col"].double().clamp(min=1e-30)
        ur = -0.5 * (torch.log(fr) - math.log(max(stats[n]["mean"], 1e-30)))
        uc = -0.5 * torch.log(fc)
        logu[n] = (ur, uc)
        acc += float(ur.sum()) * fc.numel() + float(uc.sum()) * fr.numel()
    shift = L0 - acc / ntot
    out = {}
    for n in rms:
        ur, uc = logu[n]
        a = (1 - gamma) * torch.log(rms[n].double().reshape(-1))
        b = gamma * (ur + shift)
        col = fp16r(torch.exp(gamma * uc).float().clamp(max=FP16_MAX)).clamp(min=FP16_TINY)
        out[n] = (a, b, col)
    return out


def fisher_rowcol(entry, c):
    """(row [O,1], col [I]) of the Fisher grid at step scale c (same evaluation order as scripts/ghcol_probe.py)."""
    import math
    a, b, col = entry
    row = fp16r(torch.exp(math.log(c) + a + b).float()).clamp(min=FP16_TINY)[:, None]
    return row, col


@torch.no_grad()
def gptq_sep(W, H, row, col, blocksize=128, percdamp=0.01, dev="cpu"):
    """GPTQ on any separable row x column grid (step = row [O,1] * col [I], both fp16-valued): the machinery of
    ghcol with externally supplied factors (e.g. fisher_grid); column factors made int16-safe first.  Returns
    symbols, dequantized fp32, row, col (the col actually used -- store THIS one)."""
    col, _ = int16_safe_col(W, row, col)
    S, Qw, _, _ = gptq(W, H, "grid", step=row * col[None, :], blocksize=blocksize, percdamp=percdamp, dev=dev)
    return S, Qw, row, col


def lowrank(D, r):
    U, S, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    return fp16r(U[:, :r] * S[:r]), fp16r(Vh[:r])
