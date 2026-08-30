#!/usr/bin/env python3
"""Export model_universal.pt to the self-describing TFM1 binary format.

The C++ engine reads float32 weights exactly as PyTorch sees them after
load_model() (which upcasts the stored half-precision checkpoint). Arch
attributes (layers, d_model, nhead, pos_hidden, d_ff, ln_eps) are introspected
from the real module so the file can never drift from the code.

Run: python3 tools/export_weights.py [checkpoint] [output.bin]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.torch_loader import load_model, base_checkpoint_path

MAGIC = b"TFM1"


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else base_checkpoint_path(3)
    out = sys.argv[2] if len(sys.argv) > 2 else "model.bin"

    model, err = load_model(ckpt, 3, device="cpu")
    if not model:
        print(f"load failed: {err}", file=sys.stderr)
        sys.exit(1)

    cfg = {
        "num_layers": len(model.encoder.layers),
        "d_model": model.pos.mlp[-1].out_features,
        "nhead": model.encoder.layers[0].self_attn.num_heads,
        "pos_hidden": model.pos.mlp[0].out_features,
        "d_ff": model.encoder.layers[0].linear1.out_features,
        "ln_eps": float(model.encoder.layers[0].norm1.eps),
    }

    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}

    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5If", cfg["num_layers"], cfg["d_model"],
                            cfg["nhead"], cfg["pos_hidden"], cfg["d_ff"], cfg["ln_eps"]))
        for name, arr in sd.items():
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(arr.astype("<f4", copy=False).tobytes())

    n_params = sum(int(a.size) for a in sd.values())
    print(f"wrote {out}")
    print(f"  arch: {cfg}")
    print(f"  tensors: {len(sd)}  params: {n_params}  size: {Path(out).stat().st_size} bytes")


if __name__ == "__main__":
    main()