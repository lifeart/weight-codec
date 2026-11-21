"""Print the C2 results as a markdown table (used for REPORT.md)."""
import json, os
d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "c2_qwen05b_instruct.json")))
order = ["fp16", "zero_bits_base"] + sorted([k for k in d if k.startswith("indep")]) + \
        sorted([k for k in d if k.startswith("delta_ecsq")], key=lambda k: -d[k]["bpw"]) + \
        sorted([k for k in d if k.startswith("delta_gptq")], key=lambda k: -d[k]["bpw"]) + \
        sorted([k for k in d if k.startswith("delta_lowrank")]) + \
        sorted([k for k in d if k.startswith("pframe")])
print("| config | instruct bpw | chat KL | chat top-1 | chat NLL | wiki KL | wiki top-1 | wiki PPL |")
print("|---|---|---|---|---|---|---|---|")
for k in order:
    if k not in d: continue
    v = d[k]
    print(f"| {k} | {v['bpw']:.3f} | {v['chat_kl']:.4f} | {v['chat_top1']:.3f} | {v['chat_nll']:.3f} | {v['kl']:.4f} | {v['top1']:.3f} | {v['ppl']:.2f} |")
