#!/usr/bin/env python3
"""Embed the TFM1 weights file into a C++ header so the app runs with zero
runtime file I/O.

Reads a weights file (arg 1, default `../cpp/model.bin` relative to
mobile-rn/) and writes `native/include/tfm_model_data.h` containing the raw
bytes as a C++ array in namespace `tfm`.

Usage:
    python3 scripts/embed_weights.py [weights_path]

Output header contract:
    namespace tfm {
      inline const unsigned char kModelBin[] = { ... };
      inline const size_t kModelBinSize = N;
    }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "cpp" / "model.bin"
OUT_HEADER = Path(__file__).resolve().parents[1] / "native" / "include" / "tfm_model_data.h"
PER_LINE = 20


def emit_c_array(data: bytes) -> str:
    lines: list[str] = []
    for i in range(0, len(data), PER_LINE):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + PER_LINE])
        lines.append("  " + chunk + ",")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed TFM1 weights as a C++ header")
    parser.add_argument("weights", nargs="?", default=str(DEFAULT_WEIGHTS), help="path to the TFM1 weights file")
    args = parser.parse_args()

    wpath = Path(args.weights).resolve()
    if not wpath.is_file():
        print(f"error: weights file not found: {wpath}", file=sys.stderr)
        return 1

    data = wpath.read_bytes()
    if data[:4] != b"TFM1":
        print(f"error: {wpath} is not a TFM1 file (magic != 'TFM1')", file=sys.stderr)
        return 1

    body = emit_c_array(data)
    header = (
        "#pragma once\n"
        "#include <cstddef>\n"
        "namespace tfm {\n"
        "inline const unsigned char kModelBin[] = {\n"
        f"{body}\n"
        "};\n"
        f"inline const size_t kModelBinSize = {len(data)};\n"
        "}  // namespace tfm\n"
    )

    OUT_HEADER.parent.mkdir(parents=True, exist_ok=True)
    OUT_HEADER.write_text(header)
    print(f"wrote {OUT_HEADER}")
    print(f"bytes={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())