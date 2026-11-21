"""Fill the last data-driven placeholders of REPORT.md §14 (SmolLM2-360M table, LoRA comparison, scene-cut table)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda n: json.load(open(os.path.join(ROOT, "results", n)))
rep_path = os.path.join(ROOT, "REPORT.md")
rep = open(rep_path).read()

F = J("phase2_family_smol360.json")
D = J("phase2_decode_smol360_Darwin-arm64-8cpu.json") if os.path.exists(os.path.join(ROOT, "results", "phase2_decode_smol360_Darwin-arm64-8cpu.json")) else {}
f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
def dec(label, mode):
    v = D.get(label, {}).get(mode, {})
    return f"{v['total']:.2f} s" if "total" in v else "—"
rows = [("GPTQ-int4 g64 fixed-rate packed", "T_gptq4_packed", "T_gptq4_packed"), ("GPTQ-int4 g64 + rANS", "T_gptq4_ans", "T_gptq4_ans"),
        ("GPTQ-ECSQ c=0.35", "T_ecsq_c0.35", "T_ecsq_c0.35"), ("GPTQ-ECSQ c=0.5", "T_ecsq_c0.5", None),
        ("P-frame c=0.35 (+ base I-frame)", "T_pframe_c0.35", "T_pframe_c0.35+base"), ("P-frame c=0.5", "T_pframe_c0.5", None),
        ("full-rank Δ c=0.06 (fp16 base)", "T_delta_c0.06", None), ("full-rank Δ c=0.12", "T_delta_c0.12", "T_delta_c0.12"),
        ("full-rank Δ c=0.25", "T_delta_c0.25", None), ("LoRA-style r=32", "T_lowrank_r32", "T_lowrank_r32"),
        ("LoRA-style r=64", "T_lowrank_r64", None), ("LoRA-style r=128", "T_lowrank_r128", None)]
t = ["| fine-tune format | MB | real bpw | chat-KL | wiki-KL | PPL | decode CPU constriction | CPU rans16 8 thr | GPU |",
     "|---|---|---|---|---|---|---|---|---|"]
for name, k, dk in rows:
    if k not in F:
        continue
    r = F[k]
    t.append(f"| {name} | {r['MB']:.1f} | {r['real_bpw']:.3f} | {f(r.get('chat_kl'), 4)} | {f(r.get('kl'), 4)} | {f(r.get('ppl'), 2)} | "
             + (f"{dec(dk, 'serial_cpu')} | {dec(dk + '@rans16', 'parallel_cpu')} | {dec(dk + '@rans16_4k', 'gpu_mps')} |" if dk else "— | — | — |"))
base = [f"base: fp16 PPL {F['B_fp16']['ppl']:.2f}; GPTQ-int4 g64 {F['B_gptq4_packed']['MB']:.1f} MB packed / {F['B_gptq4_ans']['MB']:.1f} MB rANS "
        f"(PPL {F['B_gptq4_ans']['ppl']:.2f}); I-frame GPTQ-ECSQ c=0.35 {F['B_ecsq_c0.35']['MB']:.1f} MB (PPL {F['B_ecsq_c0.35']['ppl']:.2f})."]
S = lambda *ns: sum(F[n]["MB"] for n in ns)
pair = (f"Pair, every tensor: two GPTQ-int4 files (rest fp16) {S('B_gptq4_packed', 'B_rest_raw16', 'T_gptq4_packed', 'T_rest_raw16'):.0f} MB; "
        f"independent GPTQ-ECSQ + lossless rest {S('B_ecsq_c0.35', 'B_rest_ll16', 'T_ecsq_c0.35', 'T_rest_ll16'):.0f} MB; "
        f"**family I+P {S('B_ecsq_c0.35', 'B_rest_ll16', 'T_pframe_c0.35', 'T_rest_xor_base'):.0f} MB** "
        f"(Instruct chat-KL {F['T_gptq4_packed']['chat_kl']:.3f} / {F['T_ecsq_c0.35']['chat_kl']:.3f} / {F['T_pframe_c0.35']['chat_kl']:.3f}).")
# LoRA comparison at matched / fewer bytes
def ratio(d, l):
    return F[l]["chat_kl"] / F[d]["chat_kl"], F[d]["MB"] / F[l]["MB"]
cmp = []
for d, l in (("T_delta_c0.25", "T_lowrank_r32"), ("T_delta_c0.12", "T_lowrank_r64"), ("T_delta_c0.06", "T_lowrank_r128")):
    if d in F and l in F:
        x, b = ratio(d, l)
        cmp.append((d, l, x, b))
lora_md = "Full Δ vs LoRA, SmolLM2-360M: " + "; ".join(
    f"{d[2:].replace('delta_c', 'Δ c=')} ({F[d]['MB']:.1f} MB) vs {l[2:].replace('lowrank_r', 'r=')} ({F[l]['MB']:.1f} MB): **{x:.1f}×** lower chat-KL at {100 * b:.0f} % of the bytes"
    for d, l, x, b in cmp) + "."
s360 = "\n".join(t) + "\n\n" + base[0] + " " + pair + "\n\n" + lora_md + "\n"
rep = rep.replace("⟨S360_TABLE⟩", s360)
rep = rep.replace("⟨S360_LORA⟩", " / ".join(f"**{x:.1f}×** at {100 * b:.0f} %" for _, _, x, b in cmp))

SC = J("phase2_scene_cut_qwen05b.json")
st = ["| relative of Qwen2.5-0.5B (base lattice, c=0.35) | ‖Δ‖/‖W‖ | I-frame MB (bpw) | P-frame MB (bpw) | P / I |", "|---|---|---|---|---|"]
for tag, r in SC.items():
    if tag == "base_I":
        continue
    st.append(f"| {tag} | {r['rel_delta']:.3f} | {r['I_bytes'] / 1e6:.1f} ({r['I_bpw']:.3f}) | {r['P_bytes'] / 1e6:.1f} ({r['P_bpw']:.3f}) | **{r['P_over_I']:.3f}** |")
scene = "\n".join(st) + (f"\n\nBase I-frame on its own lattice: {SC['base_I']['bytes'] / 1e6:.1f} MB ({SC['base_I']['bpw']:.4f} bpw). "
                         "Phase-1 estimate (C4, α=1): Instruct P 0.911 vs I 3.575 (0.25), Coder P 3.452 vs I 3.575 (0.97).\n")
rep = rep.replace("⟨SCENE_TABLE⟩", scene)
open(rep_path, "w").write(rep)
print("placeholders left:", rep.count("⟨"))
