"""C2 side-table: cost of the (tied) EMBEDDING matrix of the instruct model given the base,
on the same uniform grid as the block linears (s = c*rms_row(W_base)); rate only.
Rows whose base and instruct vectors are bit-identical are counted separately."""
from common import *
B = LazySD("Qwen/Qwen2.5-0.5B"); I = LazySD("Qwen/Qwen2.5-0.5B-Instruct")
k = "model.embed_tokens.weight"
Wb, Wi = B[k].float(), I[k].float()
out = {"shape": list(Wb.shape), "rel_delta": float((Wi - Wb).norm() / Wb.norm()),
       "rows_identical": int((Wi == Wb).all(1).sum())}
for c in (0.35, 0.7):
    s = (c * Wb.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)).half().float()
    qb, qi = torch.round(Wb / s).to(torch.int32), torch.round(Wi / s).to(torch.int32)
    out[f"c{c}"] = {"I_bpw": (coded_bits(qi)[0] + 16 * Wb.shape[0]) / Wb.numel(),
                    "P_bpw": (coded_bits(qi - qb)[0] + 16 * Wb.shape[0]) / Wb.numel(),
                    "frac_changed": float((qi != qb).float().mean())}
log(out)
save_json("c2_embedding_qwen05b_instruct", out)
