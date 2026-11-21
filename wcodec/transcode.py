"""Re-encode every entropy-coded section of a WCDC file with another coder (lossless: the symbols
are decoded and re-encoded, all other sections are copied byte for byte)."""
from .ans import open_section, is_entropy_section, ans_encode
from .container import Reader, Writer
from . import rans16


def transcode(src, dst, coder="rans16", chunk=None):
    rd = Reader(src)
    wr = Writer(dict(rd.meta, coder=coder, **({"chunk": chunk} if chunk else {})))
    if coder == "rans16":
        enc = (lambda x: rans16.encode(x, chunk)) if chunk else rans16.encode
    else:
        enc = ans_encode
    for r in rd.records:
        sec = {}
        for k in r["sec"]:
            b = rd.section(r, k)
            if is_entropy_section(k):
                b = enc(open_section(b).decode())
            sec[k] = bytes(b)
        wr.add(r["name"], r["kind"], r["shape"], r["params"], sec)
    return wr.write(dst)
