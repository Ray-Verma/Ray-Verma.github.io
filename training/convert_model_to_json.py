#!/usr/bin/env python3
"""Convert Cells2Pixels GrowingNCA + SIREN checkpoints to browser JSON/JS.

The current Cells2Pixels PyTorch implementation stores the perception tensor in
channel-major/filter-minor order after depthwise_conv(), while the published
SwissGL demo constructs it in filter-major/channel-minor order. The converter
therefore permutes w1 so the same learned coefficients see the same features.

Example:
    python training/convert_model_to_json.py \
      --model /path/to/model_8.pth \
      --siren /path/to/siren_8.pth \
      --output assets/nca/profile.json

If --siren is omitted, a matching sibling such as siren_8.pth is inferred.
A profile.js wrapper is also emitted beside profile.json for file:// preview.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

EXPECTED_SITE_SHAPES = {
    "nca.w1.weight": [256, 128],
    "nca.w1.bias": [256],
    "nca.w2.weight.T": [256, 32],
    "lppn.net.0.linear.weight": [64, 36],
    "lppn.net.0.linear.bias": [64],
    "lppn.net.1.linear.weight": [64, 64],
    "lppn.net.1.linear.bias": [64],
    "lppn.net.2.linear.weight": [64, 64],
    "lppn.net.2.linear.bias": [64],
    "lppn.net.3.weight": [4, 64],
    "lppn.net.3.bias": [4],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Cells2Pixels model + SIREN checkpoints")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--siren", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-js-output", action="store_true")
    p.add_argument("--no-site-shape-check", action="store_true")
    p.add_argument("--pretty", action="store_true")
    return p.parse_args()


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(obj: Any, label: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    if not isinstance(obj, Mapping):
        raise TypeError(f"{label}: expected mapping, got {type(obj).__name__}")
    for k in ("state_dict", "model_state_dict", "siren_state_dict", "model", "siren"):
        v = obj.get(k)
        if isinstance(v, Mapping) and any(torch.is_tensor(x) for x in v.values()):
            obj = v
            break
    state: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "nca.", "siren.", "lppn."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        state[key] = value
    if not state:
        raise ValueError(f"{label}: no tensors found")
    return state


def is_siren_state(state: Mapping[str, torch.Tensor]) -> bool:
    return "net.0.linear.weight" in state and "net.3.weight" in state


def infer_siren(model_path: Path) -> Path | None:
    stem = model_path.stem
    m = re.match(r"^model(.*)$", stem)
    suffix = m.group(1) if m else ""
    names = [f"siren{suffix}{model_path.suffix}", f"siren{suffix}.pth", f"siren{suffix}.pt", "siren.pth", "siren.pt"]
    seen: set[Path] = set()
    valid: list[Path] = []
    for name in names:
        path = model_path.with_name(name)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            if is_siren_state(unwrap(load_checkpoint(path), str(path))):
                valid.append(path)
        except Exception:
            pass
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise RuntimeError("Multiple SIREN checkpoints match; pass --siren explicitly:\n  " + "\n  ".join(map(str, valid)))
    return None


def squeeze_1x1(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.ndim == 4:
        if tuple(t.shape[-2:]) != (1, 1):
            raise ValueError(f"{name}: expected 1x1 Conv2d, got {list(t.shape)}")
        return t[:, :, 0, 0]
    if t.ndim == 2:
        return t
    raise ValueError(f"{name}: expected 2-D or [out,in,1,1], got {list(t.shape)}")


def required(state: Mapping[str, torch.Tensor], key: str, label: str) -> torch.Tensor:
    if key not in state:
        available = "\n  ".join(sorted(state))
        raise KeyError(f"{label}: missing {key!r}. Available keys:\n  {available}")
    return state[key]


def prepare_tensors(nca: Mapping[str, torch.Tensor], siren: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    w1_pt = squeeze_1x1(required(nca, "w1.weight", "NCA"), "w1.weight")
    b1 = required(nca, "w1.bias", "NCA")
    w2_pt = squeeze_1x1(required(nca, "w2.weight", "NCA"), "w2.weight")

    out_channels = int(w2_pt.shape[0])
    in_perception = int(w1_pt.shape[1])
    if in_perception % out_channels != 0:
        raise ValueError("w1 perception width is not divisible by the NCA channel count")
    perception_kernels = in_perception // out_channels
    if perception_kernels != 4:
        raise ValueError(f"Expected 4 perception kernels, found {perception_kernels}")

    # PyTorch depthwise_conv(): [c0:k0,k1,k2,k3, c1:k0,...]
    # SwissGL demo perc[]:       [k0:all channels, k1:all channels, ...]
    w1_web = (
        w1_pt.reshape(w1_pt.shape[0], out_channels, perception_kernels)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(w1_pt.shape[0], in_perception)
    )

    tensors: dict[str, torch.Tensor] = {
        "nca.w1.weight": w1_web,
        "nca.w1.bias": b1,
        "nca.w2.weight.T": w2_pt.transpose(0, 1).contiguous(),
    }
    mapping = {
        "net.0.linear.weight": "lppn.net.0.linear.weight",
        "net.0.linear.bias": "lppn.net.0.linear.bias",
        "net.1.linear.weight": "lppn.net.1.linear.weight",
        "net.1.linear.bias": "lppn.net.1.linear.bias",
        "net.2.linear.weight": "lppn.net.2.linear.weight",
        "net.2.linear.bias": "lppn.net.2.linear.bias",
        "net.3.weight": "lppn.net.3.weight",
        "net.3.bias": "lppn.net.3.bias",
    }
    for src, dst in mapping.items():
        tensors[dst] = required(siren, src, "SIREN")
    return tensors


def record(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().cpu().to(torch.float32).contiguous()
    return {
        "shape": list(t.shape),
        "data64": base64.b64encode(t.numpy().tobytes(order="C")).decode("ascii"),
    }


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    nca_state = unwrap(load_checkpoint(args.model), str(args.model))
    siren_path = args.siren
    siren_state = None
    if siren_path is not None and siren_path.is_file():
        candidate = unwrap(load_checkpoint(siren_path), str(siren_path))
        if is_siren_state(candidate):
            siren_state = candidate
        else:
            print(f"warning: --siren {siren_path} does not contain SIREN net.* weights; looking for matching sibling", file=sys.stderr)

    if siren_state is None:
        inferred = infer_siren(args.model)
        if inferred is None:
            raise FileNotFoundError("Could not find a matching siren checkpoint. Pass --siren /path/to/siren_*.pth")
        siren_path = inferred
        siren_state = unwrap(load_checkpoint(siren_path), str(siren_path))
        print(f"Using inferred SIREN checkpoint: {siren_path}")

    tensors = prepare_tensors(nca_state, siren_state)
    if not args.no_site_shape_check:
        errors = []
        for key, expected in EXPECTED_SITE_SHAPES.items():
            actual = list(tensors[key].shape)
            if actual != expected:
                errors.append(f"  {key}: expected {expected}, got {actual}")
        if errors:
            raise ValueError("Checkpoint architecture does not match this website runtime:\n" + "\n".join(errors))

    payload = {k: record(v) for k, v in tensors.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":"))
    args.output.write_text(json_text + "\n", encoding="utf-8")

    if not args.no_js_output:
        js_path = args.output.with_suffix(".js")
        js_path.write_text("window.__PROFILE_NCA_MODEL__=" + json_text + ";\n", encoding="utf-8")
        print(f"Wrote: {js_path}")

    print(f"Wrote: {args.output}")
    for key, value in payload.items():
        print(f"  {key}: {value['shape']}")


if __name__ == "__main__":
    main()
