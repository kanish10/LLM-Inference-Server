"""Export HuggingFace gpt2 (124M) to weights.bin + manifest.json + config.json.

Usage:
    venv/bin/python export_gpt2.py --out ../models

Output layout (all under --out):
    weights.bin    contiguous float32 tensors, back to back, in manifest order
    manifest.json  [{name, dtype, shape, byte_offset, byte_length}, ...]
    config.json    GPT2Config fields the C++ side needs

Design decision on precision: weights.bin stores float32. The M1 CPU oracle
consumes float32 directly (it must match HF fp32 top-1 exactly, so it cannot
afford fp16 quantization error). The M2 GPU loader downcasts to fp16 at
upload time, per the fp16-weights/fp32-accumulation constraint in
gpu-engine/CLAUDE.md. Keeping the interchange file at full precision means
one export serves both consumers.

HF GPT-2 uses Conv1D for c_attn/c_proj/c_fc, whose weight is stored as
(in_features, out_features) -- the opposite of a standard nn.Linear /
row-major GEMM weight (out_features, in_features). Every Conv1D weight below
is transposed before writing.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel


def build_tensor_list(sd: dict, n_layer: int) -> list[tuple[str, np.ndarray]]:
    def t(name: str) -> np.ndarray:
        return sd[name].detach().to(torch.float32).numpy()

    tensors: list[tuple[str, np.ndarray]] = []
    tensors.append(("wte", t("transformer.wte.weight")))
    tensors.append(("wpe", t("transformer.wpe.weight")))

    for i in range(n_layer):
        p = f"transformer.h.{i}."
        tensors.append((f"layer{i}.ln1_g", t(p + "ln_1.weight")))
        tensors.append((f"layer{i}.ln1_b", t(p + "ln_1.bias")))
        # Conv1D: (in, out) -> transpose to (out, in) for a standard row-major GEMM.
        tensors.append((f"layer{i}.qkv_w", t(p + "attn.c_attn.weight").T.copy()))
        tensors.append((f"layer{i}.qkv_b", t(p + "attn.c_attn.bias")))
        tensors.append((f"layer{i}.proj_w", t(p + "attn.c_proj.weight").T.copy()))
        tensors.append((f"layer{i}.proj_b", t(p + "attn.c_proj.bias")))
        tensors.append((f"layer{i}.ln2_g", t(p + "ln_2.weight")))
        tensors.append((f"layer{i}.ln2_b", t(p + "ln_2.bias")))
        tensors.append((f"layer{i}.fc_w", t(p + "mlp.c_fc.weight").T.copy()))
        tensors.append((f"layer{i}.fc_b", t(p + "mlp.c_fc.bias")))
        tensors.append((f"layer{i}.fc_proj_w", t(p + "mlp.c_proj.weight").T.copy()))
        tensors.append((f"layer{i}.fc_proj_b", t(p + "mlp.c_proj.bias")))

    tensors.append(("lnf_g", t("transformer.ln_f.weight")))
    tensors.append(("lnf_b", t("transformer.ln_f.bias")))
    return tensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id (gpt2 = 124M)")
    ap.add_argument("--out", default="../models", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()
    hf_config = model.config
    sd = model.state_dict()

    n_layer = hf_config.n_layer
    n_head = hf_config.n_head
    d_model = hf_config.n_embd
    head_dim = d_model // n_head
    vocab_size = hf_config.vocab_size
    n_ctx = hf_config.n_ctx

    assert d_model == 768 and n_layer == 12 and n_head == 12 and head_dim == 64, (
        f"expected GPT-2 124M dims, got d_model={d_model} n_layer={n_layer} "
        f"n_head={n_head} head_dim={head_dim}"
    )
    assert vocab_size == 50257, f"expected vocab_size=50257, got {vocab_size}"

    tensors = build_tensor_list(sd, n_layer)

    # Weight tying check: lm_head.weight must equal wte.weight exactly.
    lm_head_w = sd["lm_head.weight"].detach().to(torch.float32).numpy()
    wte_w = dict(tensors)["wte"]
    assert np.array_equal(lm_head_w, wte_w), "lm_head is not tied to wte as expected"

    weights_path = out_dir / "weights.bin"
    manifest: list[dict] = []
    sha256 = hashlib.sha256()
    checksum_sum = 0.0

    with open(weights_path, "wb") as f:
        offset = 0
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            raw = arr.tobytes()
            f.write(raw)
            sha256.update(raw)
            checksum_sum += float(np.sum(arr.astype(np.float64)))
            manifest.append(
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": list(arr.shape),
                    "byte_offset": offset,
                    "byte_length": len(raw),
                }
            )
            offset += len(raw)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "tensors": manifest,
                "checksum_sha256": sha256.hexdigest(),
                "checksum_sum": checksum_sum,
                "total_bytes": offset,
            },
            f,
            indent=2,
        )

    config = {
        "n_layer": n_layer,
        "n_head": n_head,
        "d_model": d_model,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "n_ctx": n_ctx,
        "layer_norm_epsilon": hf_config.layer_norm_epsilon,
        "eos_token_id": hf_config.eos_token_id,
        "source_model": args.model,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {weights_path} ({offset:,} bytes)")
    print(f"Wrote {manifest_path} ({len(manifest)} tensors)")
    print(f"Wrote {config_path}")
    print(f"checksum_sha256 = {sha256.hexdigest()}")
    print(f"checksum_sum    = {checksum_sum!r}")


if __name__ == "__main__":
    main()
