#!/usr/bin/env python3
"""Export the trained model weights to a compact binary for the Dart engine.

Layout (all float32 little-endian, in this order):
  header : [d_model i32][num_layers i32][n i32][nhead i32][n_tokens i32]
  cell_embed.weight                 [3 * d]
  pos.pos.weight                    [n^3 * d]
  for each layer:
    in_proj_weight                  [3*d * d]   (q,k,v stacked rows)
    in_proj_bias                    [3*d]
    out_proj.weight                 [d * d]
    out_proj.bias                   [d]
    linear1.weight                  [4d * d]
    linear1.bias                    [4d]
    linear2.weight                  [d * 4d]
    linear2.bias                    [d]
    norm1.weight [d] norm1.bias [d] norm2.weight [d] norm2.bias [d]
  value_head.0.weight [d*d] .bias [d]
  value_head.2.weight [d]   .bias [1]
  policy_head.weight [d]    .bias [1]

Weights are dumped in PyTorch row-major order and transposed on load where
needed (linear/attention weights in PyTorch are (out, in), we store them as-is
and the Dart matmul reads them accordingly).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.torch_loader import load_model, base_checkpoint_path


def main():
    model, err = load_model(base_checkpoint_path(3), 3)
    if err:
        print("load failed:", err)
        sys.exit(1)
    model = model.cpu().eval()

    sd = model.state_dict()
    d = model.cfg()["d_model"]
    layers = model.cfg()["num_layers"]
    n = model.n
    nhead = 8  # fixed in ValuePolicyTransformer
    n_tokens = n**3

    out = struct.pack("<iiiii", d, layers, n, nhead, n_tokens)

    def dump(name):
        nonlocal out
        t = sd[name].detach().numpy().astype("float32").ravel()
        out += struct.pack(f"<{len(t)}f", *t)

    dump("cell_embed.weight")
    dump("pos.pos.weight")
    for i in range(layers):
        for key in (
            f"encoder.layers.{i}.self_attn.in_proj_weight",
            f"encoder.layers.{i}.self_attn.in_proj_bias",
            f"encoder.layers.{i}.self_attn.out_proj.weight",
            f"encoder.layers.{i}.self_attn.out_proj.bias",
            f"encoder.layers.{i}.linear1.weight",
            f"encoder.layers.{i}.linear1.bias",
            f"encoder.layers.{i}.linear2.weight",
            f"encoder.layers.{i}.linear2.bias",
            f"encoder.layers.{i}.norm1.weight",
            f"encoder.layers.{i}.norm1.bias",
            f"encoder.layers.{i}.norm2.weight",
            f"encoder.layers.{i}.norm2.bias",
        ):
            dump(key)
    dump("value_head.0.weight")
    dump("value_head.0.bias")
    dump("value_head.2.weight")
    dump("value_head.2.bias")
    dump("policy_head.weight")
    dump("policy_head.bias")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mobile", "assets", "weights.bin")
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path} ({os.path.getsize(path)} bytes, {len(out)//4 - 5} floats)")


if __name__ == "__main__":
    main()