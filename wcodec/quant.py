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
    mode 'grid': fixed uniform per-row grid `step` [O,1] (GPTQ-ECSQ / GPTQ on a residual).
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


def lowrank(D, r):
    U, S, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    return fp16r(U[:, :r] * S[:r]), fp16r(Vh[:r])
