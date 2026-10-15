#!/usr/bin/env python3
"""Read a GGUF file's metadata and say whether it can serve /v1/systemone.

Why this exists: a decision model that downloads, loads and chats fine can still
answer 501 "This model is not a decision model", because llama.cpp requires
specific metadata to expose the option logits. This reads that metadata straight
out of the file, so you find out before blaming the build.

    python3 tools/gguf_meta.py ~/models/whatever.gguf
    python3 tools/gguf_meta.py --dump ~/models/whatever.gguf    # every key
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

MAGIC = b"GGUF"

# GGUF metadata value type ids.
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

SCALARS = {
    UINT8: ("<B", 1), INT8: ("<b", 1),
    UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4),
    FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8),
    FLOAT64: ("<d", 8),
}

TYPE_NAMES = {
    UINT8: "uint8", INT8: "int8", UINT16: "uint16", INT16: "int16",
    UINT32: "uint32", INT32: "int32", FLOAT32: "float32", BOOL: "bool",
    STRING: "string", ARRAY: "array", UINT64: "uint64", INT64: "int64",
    FLOAT64: "float64",
}


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes, got {len(b)}")
        return b

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        if n > 64 * 1024 * 1024:
            raise ValueError(f"implausible string length {n}")
        return self.raw(n).decode("utf-8", "replace")

    def value(self, vtype: int, depth: int = 0):
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            if depth > 2:
                raise ValueError("nested arrays deeper than expected")
            elem_type = self.scalar("<I")
            count = self.scalar("<Q")
            if count > 5_000_000:
                raise ValueError(f"implausible array length {count}")
            return [self.value(elem_type, depth + 1) for _ in range(count)]
        if vtype not in SCALARS:
            raise ValueError(f"unknown metadata value type {vtype}")
        fmt, _ = SCALARS[vtype]
        return self.scalar(fmt)


def read_metadata(path: Path) -> tuple[int, int, dict]:
    with path.open("rb") as fh:
        r = Reader(fh)
        if r.raw(4) != MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        version = r.scalar("<I")
        tensor_count = r.scalar("<Q")
        kv_count = r.scalar("<Q")
        meta: dict = {}
        for _ in range(kv_count):
            key = r.string()
            vtype = r.scalar("<I")
            meta[key] = r.value(vtype)
    return version, tensor_count, meta


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def main(argv: list[str]) -> int:
    dump = False
    args = []
    for a in argv:
        if a in ("-h", "--help"):
            print(__doc__)
            return 0
        if a == "--dump":
            dump = True
        else:
            args.append(a)

    if not args:
        print(__doc__)
        return 2

    path = Path(args[0]).expanduser()
    if not path.is_file():
        print(f"no such file: {path}")
        return 2

    size = path.stat().st_size
    try:
        version, tensor_count, meta = read_metadata(path)
    except Exception as exc:
        print(f"could not read {path.name}: {exc}")
        return 1

    print(f"file        {path}")
    print(f"size        {human(size)}")
    print(f"gguf        version {version}, {tensor_count} tensors, {len(meta)} metadata keys")
    print()

    arch = meta.get("general.architecture", "?")
    print(f"architecture  {arch}")
    for key in sorted(meta):
        if key.endswith(".context_length"):
            print(f"context       {meta[key]}  ({key})")
    if "tokenizer.ggml.model" in meta:
        print(f"tokenizer     {meta['tokenizer.ggml.model']}")

    # The part that decides whether /v1/systemone will serve this file.
    hints = {k: v for k, v in meta.items()
             if "decision" in k.lower() or "systemone" in k.lower() or "choice" in k.lower()}
    print()
    if hints:
        print("decision-model metadata found:")
        for key in sorted(hints):
            val = hints[key]
            shown = val if isinstance(val, (int, float, str, bool)) else f"<{TYPE_NAMES.get(ARRAY)} of {len(val)}>"
            print(f"  {key} = {shown}")
        print()
        print("verdict  looks like a decision model. It should serve /v1/systemone.")
    else:
        print("no decision-model metadata found.")
        print()
        print("verdict  probably NOT a decision model. llama.cpp will likely answer")
        print("         501 'This model is not a decision model' for /v1/systemone,")
        print("         even though it loads and chats normally.")

    if dump:
        print()
        print("every metadata key:")
        for key in sorted(meta):
            val = meta[key]
            if isinstance(val, str) and len(val) > 120:
                val = val[:117] + "..."
            if isinstance(val, list) and len(val) > 12:
                val = f"<array of {len(val)}>"
            print(f"  {key} = {val}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
