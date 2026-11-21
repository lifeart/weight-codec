"""Build results/phase2_space_summary.json (what the Space shows) from the phase-2 results."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
J = lambda n: json.load(open(os.path.join(RES, n))) if os.path.exists(os.path.join(RES, n)) else {}
r3 = lambda x: None if x is None else (round(x, 4) if abs(x) < 10 else round(x, 1))

S = J("phase2_summary.json")
fmt_rows = S.get("formats", [])
pairs = S.get("pairs", [])
sections = []

# --- family tables ---
TITLES = {"qwen05b": "Qwen2.5-0.5B → Instruct", "smol360": "SmolLM2-360M → Instruct", "smol135": "SmolLM2-135M → Instruct"}
for tag, title in TITLES.items():
    rows = [r for r in fmt_rows if r["tag"] == tag and r["role"] == "fine-tune"]
    if not rows:
        continue
    rows.sort(key=lambda r: -r["MB"])
    sections.append({
        "title": f"{title}: fine-tune linears — real MB, decode ms, quality",
        "md": "Real file bytes of the block linears of the fine-tune (P-frame/Δ rows: the fine-tune's own stream; "
              "the base is stored once). Quality measured on the model decoded from that file. "
              "Decode ms: full chain (P-frame = base I-frame + P-frame) on an Apple Silicon laptop.",
        "table": {"headers": ["format", "MB", "real bpw", "phase-1 entropy estimate bpw", "chat-KL", "wiki-KL", "PPL",
                              "decode ms CPU (constriction, 1 thread)", "decode ms CPU (rans16, parallel)",
                              "decode ms GPU (Metal rANS)"],
                  "rows": [[r["format"], r3(r["MB"]), r3(r["real_bpw"]), r3(r.get("proxy_bpw")), r3(r.get("chat_kl")),
                            r3(r.get("kl")), r3(r.get("ppl")), r3(r.get("dec_cpu_constriction_1t_ms")),
                            r3(r.get("dec_cpu_rans16_par_ms")), r3(r.get("dec_gpu_ms"))] for r in rows]},
        "images": [i for i in (f"p2_{tag}_mb_dec_quality.png",) if os.path.exists(os.path.join(ROOT, "figures", i))]})
    pr = [p for p in pairs if p["tag"] == tag]
    if pr:
        sections.append({
            "title": f"{title}: storing the PAIR {{base, fine-tune}} — every tensor",
            "md": "Total MB on disk for both models (linears + embeddings/norms; rest coded losslessly).",
            "table": {"headers": ["scenario", "pair MB", "base PPL", "fine-tune chat-KL", "fine-tune wiki-KL"],
                      "rows": [[p["scenario"], r3(p["pair_MB"]), r3(p.get("base_ppl")), r3(p.get("ft_chat_kl")),
                                r3(p.get("ft_wiki_kl"))] for p in pr]},
            "images": [i for i in (f"p2_{tag}_pair_mb.png",) if os.path.exists(os.path.join(ROOT, "figures", i))]})

# --- trajectory (encoded on an HF Job; PPL measured on the Mac from the job's files) ---
T = J("phase2_trajectory_pythia410m.json")
X = J("phase2_xcheck_trajectory.json")
for w in ("late", "mid"):
    s = T.get(f"{w}_summary")
    if not s:
        continue
    P = X.get(f"{w}_ppl", {})
    pp = lambda k: r3(P.get(k, {}).get("ppl")) if P.get(k) else None
    rows = []
    for c in ("0.35", "0.7"):
        rows.append([f"c={c} all-I", r3(s[f"gop_total_MB_allI_c{c}"]), r3(s[f"mean_I_bpw_c{c}"]), pp(f"I_c{c}")])
        rows.append([f"c={c} I + P-frames (IPPP)", r3(s[f"gop_total_MB_IPPP_c{c}"]), r3(s[f"mean_P_bpw_c{c}"]), pp(f"IPPP_end_c{c}")])
    xc = X.get(f"{w}_c0.35", {})
    sections.append({"title": f"Pythia-410M training trajectory ({w} window, {len(s['steps'])} checkpoints) — real bytes",
                     "md": f"steps {s['steps'][0]}…{s['steps'][-1]}; encoded on {s.get('hardware')}; original last-frame PPL "
                           f"{pp('orig')} (MPS fp16, 40 windows); x86-encoded chain decodes bit-exactly on the Mac: "
                           f"{xc.get('sha_match')}.",
                     "table": {"headers": ["scheme", "GOP MB (linears)", "mean bpw (I or P)", "last-frame PPL (decoded)"], "rows": rows},
                     "images": [i for i in ("p2_pythia_frames_real.png",) if os.path.exists(os.path.join(ROOT, "figures", i))]})

# --- SVC ---
V = J("phase2_svc_qwen05b.json")
for key in ("r3", "r2"):
    s = V.get(f"{key}_summary")
    if not s:
        continue
    sections.append({"title": f"Progressive (SVC) stream, Qwen2.5-0.5B, step ratio {s['ratio']} — real bytes per layer",
                     "md": "Cumulative layers decode bit-identically to the single-shot quantizer at the same step.",
                     "table": {"headers": ["layers", "cum MB", "cum bpw", "single-shot bpw", "tax bpw", "PPL"],
                               "rows": [[f"L0..{L['layer']}", r3(L["cum_MB"]), r3(L["cum_bpw"]), r3(L["single_bpw"]),
                                         r3(L["tax_bpw"]), r3(L.get("ppl"))] for L in s["layers"]]},
                     "images": [i for i in ("p2_svc_tax_real.png",) if os.path.exists(os.path.join(ROOT, "figures", i))] if key == "r3" else []})

# --- delivery model for Qwen Instruct (MB measured, decode measured, download modelled) ---
FQ, DQ = J("phase2_family_qwen05b.json"), J("phase2_decode_qwen05b_Darwin-arm64-8cpu.json")
def dms(label, mode="parallel_cpu"):
    v = DQ.get(label, {}).get(mode, {})
    return v.get("total")
if FQ and DQ:
    rows = []
    def add(name, files, dec, q):
        if None in dec or any(f not in FQ for f in files):
            return
        mb = sum(FQ[f]["MB"] for f in files); d = sum(dec)
        rows.append([name, round(mb, 1), round(d, 2)] + [round(mb * 8 / bw + d, 1) for bw in (50, 200, 1000)] + [q])
    add("fp16 safetensors", [], [DQ.get("safetensors_fp16_target", {}).get("serial_cpu", {}).get("total")], 0.0)
    rows[-1][1] = round(DQ["safetensors_fp16_target"]["MB"], 1) if rows else None
    if rows:
        mb = rows[-1][1]; d = rows[-1][2]; rows[-1][3:6] = [round(mb * 8 / bw + d, 1) for bw in (50, 200, 1000)]
    add("standard GPTQ-int4 g128 file + fp16 rest", ["T_gptq4_packed", "T_rest_raw16"], [dms("T_gptq4_packed"), 0.0], round(FQ["T_gptq4_packed"]["chat_kl"], 4))
    add("GPTQ-ECSQ c=0.35 + rANS + lossless rest", ["T_ecsq_c0.35", "T_rest_ll16"], [dms("T_ecsq_c0.35@rans16"), dms("T_rest_ll16@rans16")], round(FQ["T_ecsq_c0.35"]["chat_kl"], 4))
    add("P-frame + XOR rest (base cached)", ["T_pframe_c0.35", "T_rest_xor_base"], [dms("T_pframe_c0.35_only@rans16"), dms("T_rest_xor_base@rans16")], round(FQ["T_pframe_c0.35"]["chat_kl"], 4))
    add("full-rank Δ c=0.12 + XOR rest (fp16 base cached)", ["T_delta_c0.12", "T_rest_xor_base"], [dms("T_delta_c0.12@rans16"), dms("T_rest_xor_base@rans16")], round(FQ["T_delta_c0.12"]["chat_kl"], 4))
    sections.insert(0, {"title": "Delivering Qwen2.5-0.5B-Instruct: bytes on the wire + decode",
                        "md": "MB = real files; decode = measured (Apple Silicon laptop, rans16 on 8 cores); download = MB / bandwidth (modelled).",
                        "table": {"headers": ["what the client fetches", "MB", "decode s", "total s @50 Mbit/s", "@200 Mbit/s", "@1 Gbit/s", "Instruct chat-KL"],
                                  "rows": rows}})
for tag in ("smol135", "qwen05b"):
    Z = J(f"phase2_zstd_ref_{tag}.json")
    if Z:
        sections.append({"title": f"Generic coder reference ({tag}): zstd-19 / gzip-9 on the same symbols vs rANS",
                         "md": "Each tensor's symbols as int8 (int16 only if needed), one stream; decode = single-thread zstd.",
                         "table": {"headers": ["file", "rANS sym MB", "zstd-19 MB", "zstd vs rANS %", "gzip-9 vs rANS %", "zstd decode ms"],
                                   "rows": [[k, r3(v["ans_sym_MB"]), r3(v["zstd19_sym_MB"]), r3(v["zstd19_vs_ans_pct"]),
                                             r3(v.get("gzip9_vs_ans_pct")), r3(1e3 * v["zstd19_decode_1t_s"])] for k, v in sorted(Z.items())]}})

fam = J("phase2_family_smol135.json")
mb = lambda n: fam.get(n, {}).get("MB", float("nan"))
q = lambda n: (f"chat-KL {fam[n]['chat_kl']:.3f}, wiki-KL {fam[n]['kl']:.3f}" if n in fam and "chat_kl" in fam[n] else "")
demo = {
    "dir": "bitstreams/smol135", "target": "HuggingFaceTB/SmolLM2-135M-Instruct", "base": "HuggingFaceTB/SmolLM2-135M",
    "fp16_MB": 269.1,
    "md": ("Pair **SmolLM2-135M → SmolLM2-135M-Instruct** (‖Δ‖/‖W‖ = 0.08). Only our WCDC bitstreams live in this "
           "Space; model config + tokenizer come from the public Hub repo and **no fine-tune weights are downloaded** "
           "(the first option also fetches the *public base* in fp16, the 'hub keeps the base, ships deltas' case). "
           "Honest note: this 135M model is fragile — every int4-class format (≈3.6–4.3 bpw) degrades it badly "
           "(chat-KL 0.4–0.54, see table); the full-rank Δ over an fp16 base keeps it close to the original."),
    "formats": {
        "Hub fp16 base + instruct full-rank Δ (GPTQ c=0.12, rANS) + lossless XOR rest": {
            "chain": [["@hub_base", None], ["T_rest_xor_base", "@hub_base"], ["T_delta_c0.12", "@hub_base"]],
            "model_files": ["T_rest_xor_base", "T_delta_c0.12"], "quality": q("T_delta_c0.12"),
            "what": "instruct stream only, given the public base"},
        "FAMILY int4-class: base I-frame + instruct P-frame (constriction rANS)": {
            "chain": [["B_rest_ll16", None], ["B_ecsq_c0.35", None], ["T_rest_xor_base", "B_rest_ll16"],
                      ["T_pframe_c0.35", "B_ecsq_c0.35"]],
            "model_files": ["T_rest_xor_base", "T_pframe_c0.35"], "quality": q("T_pframe_c0.35"),
            "what": "BOTH models: base + instruct"},
        "FAMILY int4-class, same streams re-encoded with wcodec rans16 (fast decoder)": {
            "subdir": "r16", "coder": "rans16",
            "chain": [["B_rest_ll16", None], ["B_ecsq_c0.35", None], ["T_rest_xor_base", "B_rest_ll16"],
                      ["T_pframe_c0.35", "B_ecsq_c0.35"]],
            "model_files": ["T_rest_xor_base", "T_pframe_c0.35"], "quality": q("T_pframe_c0.35") + " (bit-identical decode)",
            "what": "BOTH models: base + instruct"},
        "Independent instruct: GPTQ-int4 g64 fixed-rate + lossless rest": {
            "chain": [["T_rest_ll16", None], ["T_gptq4_packed", None]],
            "model_files": ["T_rest_ll16", "T_gptq4_packed"], "quality": q("T_gptq4_packed"), "what": "instruct only"},
        "Independent instruct: GPTQ-ECSQ c=0.35 + rANS + lossless rest": {
            "chain": [["T_rest_ll16", None], ["T_ecsq_c0.35", None]],
            "model_files": ["T_rest_ll16", "T_ecsq_c0.35"], "quality": q("T_ecsq_c0.35"), "what": "instruct only"},
    }}
intro = ("# weight-codec — real ANS bitstreams for LLM families\n"
         "Phase 2 of a falsification-first study (*compress weights like JPEG, families/checkpoints like video*). "
         "Everything here is **measured on real files** (bytes on disk incl. all side info) and on models "
         "**decoded from those files**. Code: `wcodec` (WCDC container, constriction rANS, own numba/Metal rANS).")
json.dump({"intro_md": intro, "sections": sections, "demo": demo},
          open(os.path.join(RES, "phase2_space_summary.json"), "w"), indent=1)
print("sections:", [s["title"] for s in sections])
