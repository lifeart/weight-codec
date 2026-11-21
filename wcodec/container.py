"""WCDC container: one file = one coded model (or one frame / quality layer of it).

    b"WCDC" | u8 version | u32 header_len | zlib(JSON header) | payload

JSON header = {"meta": {...}, "records": [{"name", "kind", "shape", "params",
                                           "sec": {section_name: [offset, nbytes]}}]}
Offsets are relative to the start of the payload.  EVERY byte of the file -- header, tables,
scales, steps, ANS words -- counts towards the reported size; nothing is kept outside.
"""
import json
import struct
import zlib

MAGIC = b"WCDC"
VERSION = 1


class Writer:
    def __init__(self, meta=None):
        self.meta = dict(meta or {})
        self.records = []
        self.blobs = []
        self.off = 0

    def add(self, name, kind, shape, params=None, sections=None):
        sec = {}
        for k, b in (sections or {}).items():
            b = bytes(b)
            sec[k] = [self.off, len(b)]
            self.blobs.append(b)
            self.off += len(b)
        self.records.append({"name": name, "kind": kind, "shape": list(shape),
                             "params": params or {}, "sec": sec})

    def tobytes(self):
        hdr = zlib.compress(json.dumps({"meta": self.meta, "records": self.records},
                                       separators=(",", ":")).encode(), 9)
        return MAGIC + struct.pack("<BI", VERSION, len(hdr)) + hdr + b"".join(self.blobs)

    def write(self, path):
        b = self.tobytes()
        with open(path, "wb") as f:
            f.write(b)
        return len(b)


class Reader:
    def __init__(self, data):
        if isinstance(data, (str, bytes)) is False and hasattr(data, "read"):
            data = data.read()
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        self.buf = memoryview(data)
        assert bytes(self.buf[:4]) == MAGIC, "not a WCDC file"
        ver, hl = struct.unpack_from("<BI", self.buf, 4)
        assert ver == VERSION, ver
        self.header_bytes = 9 + hl
        h = json.loads(zlib.decompress(bytes(self.buf[9:9 + hl])))
        self.meta = h["meta"]
        self.records = h["records"]
        self.by_name = {r["name"]: r for r in self.records}
        self.base = 9 + hl
        self.nbytes = len(self.buf)

    def section(self, rec, key):
        o, n = rec["sec"][key]
        return self.buf[self.base + o:self.base + o + n]

    def section_bytes(self):
        """{section name: total bytes over all records} -- size breakdown."""
        out = {}
        for r in self.records:
            for k, (_, n) in r["sec"].items():
                out[k] = out.get(k, 0) + n
        return out

    def kind_bytes(self):
        out = {}
        for r in self.records:
            out[r["kind"]] = out.get(r["kind"], 0) + sum(n for _, n in r["sec"].values())
        return out
