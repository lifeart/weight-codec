"""Seam check (owner rule: diff caller names against handler names, zero mismatches) for the browser demo:

  1. record params / section names: what the REAL demo files carry (headers of docs/bitstreams/*.wcdc) vs what the
     JS decoder (docs/js/codec.js) reads, and vs what the Python decoder (wcodec/codecs.py decode_record) reads;
  2. JSON fields: what docs/js/worker.js + docs/app.js read from manifest.json / results.json / generations.json vs
     what phase2/demo_build.py wrote into them (and every config name the page looks up exists);
  3. worker <-> page messages: message types / fields posted by worker.js vs handled in app.js (and commands).

Prints the diff, writes results/demo_seam_fields.json, exits non-zero on any mismatch.
usage: .venv/bin/python phase2/seam_fields.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wcodec import Reader  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
rd = lambda p: open(os.path.join(ROOT, p), encoding="utf-8").read()
codec_js, worker_js, app_js, codecs_py = rd("docs/js/codec.js"), rd("docs/js/worker.js"), rd("docs/app.js"), rd("wcodec/codecs.py")
fails, out = [], {}


def check(name, missing, extra_ok=True):
    out[name] = sorted(missing)
    print(f"  {'ok  ' if not missing else 'FAIL'} {name}: " + ("zero mismatches" if not missing else f"MISSING {sorted(missing)}"))
    if missing:
        fails.append(name)


# ---------------------------------------------------------------- 1. container records
files = sorted(glob.glob(os.path.join(DOCS, "bitstreams", "*.wcdc")))
kinds, params, secs = set(), {}, {}
for f in files:
    r = Reader(f)
    for rec in r.records:
        kinds.add(rec["kind"])
        params.setdefault(rec["kind"], set()).update(rec["params"].keys())
        secs.setdefault(rec["kind"], set()).update(re.sub(r"^lo\d+$", "lo*", k) for k in rec["sec"])
        for k, v in rec["params"].items():                   # every VALUE the JS must understand
            params.setdefault(f"{rec['kind']}.{k}=", set()).add(json.dumps(v) if not isinstance(v, str) else v)
print(f"demo files: {len(files)}; record kinds {sorted(kinds)}")
for k in sorted(params):
    if not k.endswith("="):
        print(f"  {k}: params {sorted(params[k])}, sections {sorted(secs[k])}")
    else:
        print(f"    {k} {sorted(params[k])[:8]}")
dec_body = codec_js[codec_js.index("function decodeRecord"):]
js_kinds = set(re.findall(r'rec\.kind === "(\w+)"', dec_body))
js_params = set(re.findall(r"\bp\.(\w+)", dec_body))
js_secs = set(re.findall(r'secOff\(f, rec, "(\w+)"', dec_body)) | set(re.findall(r'f16Section\(f, rec, "(\w+)"', dec_body))
if re.search(r'secOff\(f, rec, "lo" \+ c\)', dec_body):
    js_secs.add("lo*")
py_body = codecs_py[codecs_py.index("def decode_record"):codecs_py.index("def decode(")]
py_params = set(re.findall(r'p\["(\w+)"\]', py_body)) | set(re.findall(r'p\.get\("(\w+)"\)', py_body))
py_secs = set(re.findall(r'(?:section\(rec|_sec_off\(rd, rec), "(\w+)"', py_body)) | {"lo*", "sym"}
print(f"JS decoder: kinds {sorted(js_kinds)}, params {sorted(js_params)}, sections {sorted(js_secs)}")
print(f"Python decoder: params {sorted(py_params)}")
check("record kinds in the demo files not handled by the JS decoder", kinds - js_kinds)
for k in sorted(kinds):
    check(f"{k}: params in the files not read by the JS decoder", params[k] - js_params)
    check(f"{k}: params in the files not read by the Python decoder", params[k] - py_params)
    check(f"{k}: sections in the files not read by the JS decoder", secs[k] - js_secs)
    check(f"{k}: sections the JS decoder reads that the Python decoder does not", (js_secs & secs[k]) - py_secs)
# values: every param value present in the files must be one the JS decoder accepts explicitly
JS_VALUES = {"qrow.step=": {"own", "ref"}, "qrow.acc=": {"none", "add"}, "qrow.base=": {"none", "fp16ref"},
             "qrow.col=": {"own", "ref"}, "ll16.xor=": {"true", "false"}}
for key, vals in JS_VALUES.items():
    for v in vals:                                         # the accepted values must literally appear in the decoder
        lit = {"true": "p.xor", "false": "p.xor"}.get(v, f'"{v}"')
        if lit not in dec_body:
            fails.append(f"{key}{v} not in codec.js"); print(f"  FAIL {key}{v}: literal {lit} not found in codec.js")
    if key in params:
        check(f"{key} values in the files the JS decoder does not accept", params[key] - vals)

# ---------------------------------------------------------------- 2. JSON fields
MAN = json.load(open(os.path.join(DOCS, "data", "manifest.json")))
RES = json.load(open(os.path.join(DOCS, "data", "results.json")))
GEN = json.load(open(os.path.join(DOCS, "data", "generations.json")))
js = worker_js + "\n" + app_js
file_keys = set().union(*(set(v) for v in MAN["files"].values()))
used_file = set(re.findall(r"MAN\.files\[[^\]]+\]\.(\w+)", js)) | set(re.findall(r"\bf\.(desc|role|bytes|lossless)\b", app_js))
mf = worker_js[worker_js.index("const m = MAN.files[nm]"):]
used_file |= set(re.findall(r"\bm\.(\w+)", mf[:mf.index("results.push")]))
check("manifest.files fields read by JS but not written", used_file - file_keys)
chain_keys = set().union(*(set(v) for v in MAN["chains"].values()))
used_chain = set(re.findall(r"MAN\.chains\[\w+\]\.(\w+)", js)) | set(re.findall(r"\bch\.(\w+)", js))
check("manifest.chains fields read by JS but not written", used_chain - chain_keys)
used_orig = set(re.findall(r"MAN\.original\.(\w+(?:\.\w+)?)", js))
have_orig = {f"{a}.{b}" for a, v in MAN["original"].items() if isinstance(v, dict) for b in v} | set(MAN["original"])
check("manifest.original fields read by JS but not written", used_orig - have_orig)
qkeys = set(MAN["chains"]["family"]["quality"]) & set(MAN["chains"]["family"]["prev_quality"])
used_q = set(re.findall(r"\b[QP]\.(chat_kl|chat_top1|kl|top1|ppl|real_bpw)\b", app_js))
check("chain quality fields read by JS but not written", used_q - qkeys)
used_res = set(re.findall(r"\bRES\.(\w+)", app_js))
check("results.json top-level fields read by JS but not written", used_res - set(RES))
used_names = set(re.findall(r"\bN\.(\w+)", app_js)) - {"prev"} | {f"prev.{k}" for k in re.findall(r"\bN\.prev\.(\w+)", app_js)}
have_names = set(RES["names"]) - {"prev"} | {f"prev.{k}" for k in RES["names"]["prev"]}
check("results.names read by JS but not written", used_names - have_names)
cfgs = set(re.findall(r'\bF\.(\w+)\b', app_js)) | set(re.findall(r'RES\.files\.(\w+)', app_js)) | set(re.findall(r'F\["([\w.]+)"\]', app_js))
cfgs |= {RES["names"][k] for k in ("iframe", "pframe", "delta", "indep")} | set(RES["names"]["prev"].values())
cfgs |= {n for pr in RES["delta_vs_lora"] for n in pr}
check("config names the page looks up that results.files lacks", cfgs - set(RES["files"]))
fkeys = set().union(*(set(v) for v in RES["files"].values()))
used_f = set(re.findall(r'F\[[^\]]+\]\.(\w+)', app_js)) | set(re.findall(r'\bF\.\w+\.(\w+)', app_js)) | set(re.findall(r'RES\.files\.\w+\.(\w+)', app_js))
check("results.files fields read by JS but not written", used_f - fkeys)
pkeys = set().union(*(set(p) for p in RES["pairs"]))
body = lambda a, b: app_js[app_js.index(a):app_js.index(b)]
nums, tiles = body("function renderNumbers", "function renderAllResults"), body("function renderTiles", "function renderSession")
used_p = set(re.findall(r"\bp\.(\w+)", nums)) | set(re.findall(r"\b(?:fam|famPrev|int4|indep)\.(\w+)", nums)) | \
    set(re.findall(r"\b(?:pFam|pInt4)\.(\w+)", tiles))        # variables that hold results.pairs rows there
check("results.pairs fields read by JS but not written", used_p - pkeys)
need_pairs = {"default-family": any(p["kind"] == "default" and p["scenario"].startswith("family") for p in RES["pairs"]),
              "prev-family": any(p["kind"] == "prev" and "family" in p["scenario"] for p in RES["pairs"]),
              "int4": any(p["scenario"].startswith("two standard GPTQ-int4") for p in RES["pairs"]),
              "control": any(p["kind"] == "control" for p in RES["pairs"])}
check("pair rows the page .find()s that results.pairs lacks", {k for k, v in need_pairs.items() if not v})
used_gen = set(re.findall(r"\bGEN\.(\w+)", app_js))
check("generations.json fields read by JS but not written", used_gen - set(GEN))
labels = set(re.findall(r'"((?:base|instruct)_\w+)"', app_js))
check("generation labels the page uses that generations.json lacks", labels - set(GEN["outputs"]))
tv = json.load(open(os.path.join(ROOT, "phase2", "demo_testvec.json")))
tlabels = set(re.findall(r'"((?:base|instruct)_\w+)"', rd("phase2/test_demo.mjs")))
check("test vectors the Node test expects that demo_testvec.json lacks", tlabels - set(tv["forward"]))
check("chains the page / test use that the manifest lacks",
      set(re.findall(r'"(delta|family|indep)"', app_js[:app_js.index("const GEN_LABEL")])) - set(MAN["chains"]))

# ---------------------------------------------------------------- 3. worker <-> page messages
posted = set(re.findall(r'post\(\{ type: "(\w+)"', worker_js))
handled = set(re.findall(r'm\.type === "(\w+)"', app_js))
check("message types the worker posts that the page does not handle", posted - handled)
check("message types the page handles that the worker never posts", handled - posted)
cmds_sent = set(re.findall(r'cmd: "(\w+)"', app_js))
cmds_handled = set(re.findall(r'm\.cmd === "(\w+)"', worker_js))
check("commands the page sends that the worker does not handle", cmds_sent - cmds_handled)
res_fields = set(re.findall(r"(\w+)(?:,|:)", worker_js[worker_js.index("const res = {"):worker_js.index("results.push(res)")]))
used_res_fields = set(re.findall(r"\br\.(\w+)", app_js[app_js.index("function onChainDone"):app_js.index("function setButtons")]))
check("per-file result fields the page reads that the worker does not send", used_res_fields - res_fields)

json.dump({"mismatches": out, "failed": fails, "demo_files": [os.path.basename(f) for f in files]},
          open(os.path.join(ROOT, "results", "demo_seam_fields.json"), "w"), indent=1)
print("ZERO MISMATCHES" if not fails else f"{len(fails)} MISMATCH GROUP(S): {fails}")
sys.exit(1 if fails else 0)
