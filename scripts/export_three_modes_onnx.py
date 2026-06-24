
import os
import sys
import gc
import argparse
from collections import Counter
import numpy as np
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import helper
from torch.onnx import register_custom_op_symbolic


# ==========================================================
# Project root
# ==========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet


# ==========================================================
# Custom symbolic for grid_sample
# ==========================================================
def grid_sampler_symbolic(g, input, grid, mode, padding_mode, align_corners):
    """
    Export PyTorch aten::grid_sampler as mmdeploy::grid_sampler first.

    Why:
    - If exported directly as default-domain grid_sampler, PyTorch/ONNX checker fails:
      "No Op registered for grid_sampler".
    - Therefore, we first place it under a custom domain: mmdeploy::grid_sampler.
    - After export, convert it back to default-domain grid_sampler so TensorRT plugin
      libmmdeploy_tensorrt_ops.so can take over on Jetson Nano.

    The attribute names follow mmdeploy TensorRT grid_sampler plugin style:
    - interpolation_mode: 0 = bilinear
    - padding_mode: 0 = zeros
    - align_corners: 0 or 1
    """
    return g.op(
        "mmdeploy::grid_sampler",
        input,
        grid,
        interpolation_mode_i=0,
        padding_mode_i=0,
        align_corners_i=0
    )


register_custom_op_symbolic("aten::grid_sampler", grid_sampler_symbolic, 11)

@contextmanager
def patch_static_coordinate_ops_for_onnx():
    """
    During ONNX export, force coordinate-generation ops to become fixed Constant tensors.
    This avoids exporting Range nodes, which TensorRT 8.2 on Jetson Nano cannot parse reliably.
    This is safe because deployment input size is fixed at 1x3x320x640.
    """
    original_arange = torch.arange
    original_linspace = torch.linspace

    def _to_number(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().reshape(-1)[0].item()
        return x

    def static_arange(*args, **kwargs):
        try:
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", None)

            if len(args) == 1:
                start = 0
                end = _to_number(args[0])
                step = 1
            elif len(args) == 2:
                start = _to_number(args[0])
                end = _to_number(args[1])
                step = 1
            elif len(args) >= 3:
                start = _to_number(args[0])
                end = _to_number(args[1])
                step = _to_number(args[2])
            else:
                start = _to_number(kwargs.get("start", 0))
                end = _to_number(kwargs["end"])
                step = _to_number(kwargs.get("step", 1))

            if dtype is None:
                dtype = torch.int64

            # arange normally excludes end
            values = np.arange(start, end, step)
            return torch.tensor(values, dtype=dtype, device=device)

        except Exception:
            return original_arange(*args, **kwargs)

    def static_linspace(*args, **kwargs):
        try:
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", None)

            if len(args) >= 3:
                start = _to_number(args[0])
                end = _to_number(args[1])
                steps = int(_to_number(args[2]))
            else:
                start = _to_number(kwargs["start"])
                end = _to_number(kwargs["end"])
                steps = int(_to_number(kwargs["steps"]))

            if dtype is None:
                dtype = torch.float32

            values = np.linspace(start, end, steps)
            return torch.tensor(values, dtype=dtype, device=device)

        except Exception:
            return original_linspace(*args, **kwargs)

    torch.arange = static_arange
    torch.linspace = static_linspace

    try:
        yield
    finally:
        torch.arange = original_arange
        torch.linspace = original_linspace
# ==========================================================
# Basic configuration
# ==========================================================
WEIGHT_PATH = "/root/autodl-tmp/project/stereo-matching/checkpoints_ohem+spatial/model_best.pth"
OUTPUT_DIR = "./onnx_three_modes"

INPUT_H = 320
INPUT_W = 640
MAX_DISP = 192

# Jetson Nano / TensorRT 8.2 compatibility
OPSET = 11


MODES = {
    "fast": {
        "use_prn": False,
        "use_gate": False,
        "onnx_raw": "esnet_fast_raw.onnx",
        "onnx_plugin": "esnet_fast_plugin.onnx",
    },
    "balanced": {
        "use_prn": True,
        "use_gate": False,
        "onnx_raw": "esnet_balanced_raw.onnx",
        "onnx_plugin": "esnet_balanced_plugin.onnx",
    },
    "accurate": {
        "use_prn": True,
        "use_gate": True,
        "onnx_raw": "esnet_accurate_raw.onnx",
        "onnx_plugin": "esnet_accurate_plugin.onnx",
    },
}


# ==========================================================
# Utilities
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Export ESNet three deployment modes to ONNX")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["fast", "balanced", "accurate", "all"],
        help="Which mode to export."
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=WEIGHT_PATH,
        help="Checkpoint path."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output ONNX directory."
    )
    return parser.parse_args()


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        cleaned[k.replace("module.", "")] = v
    return cleaned


class ESNetONNXWrapper(nn.Module):
    """
    ONNX export wrapper.

    The original model returns a dictionary. TensorRT deployment should receive
    a single disparity tensor output.

    fast:
        Baseline only. Uses p3 coarse disparity and upsamples it to 320x640.
    balanced:
        Baseline + PRN. Uses disp_init if available.
    accurate:
        Baseline + PRN + Gate. Uses disp_final.
    """

    def __init__(self, model, mode_name):
        super().__init__()
        self.model = model
        self.mode_name = mode_name

    def forward(self, left, right):
        outputs = self.model(left, right)

        if self.mode_name == "fast":
            disp = outputs["all_disparities"]["p3"]

            h = left.shape[-2]
            w = left.shape[-1]
            old_w = disp.shape[-1]

            disp = F.interpolate(
                disp,
                size=(h, w),
                mode="bilinear",
                align_corners=True
            )

            scale = w / old_w
            disp = disp * scale
            return disp

        if self.mode_name == "balanced":
            if "disp_init" in outputs:
                return outputs["disp_init"]
            if "disp_final" in outputs:
                return outputs["disp_final"]
            raise RuntimeError("Balanced mode has no disp_init or disp_final.")

        if self.mode_name == "accurate":
            if "disp_final" in outputs:
                return outputs["disp_final"]
            raise RuntimeError("Accurate mode has no disp_final.")

        raise RuntimeError(f"Unknown mode: {self.mode_name}")


def inspect_onnx(path):
    model = onnx.load(path)
    ops = Counter([n.op_type for n in model.graph.node])
    domains = Counter([n.domain for n in model.graph.node])

    print(f"ONNX inspect: {path}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  GridSample: {ops.get('GridSample', 0)}")
    print(f"  grid_sampler: {ops.get('grid_sampler', 0)}")
    print(f"  Conv: {ops.get('Conv', 0)}")
    print(f"  Domains: {domains}")
    print(f"  Top ops: {ops.most_common(12)}")


def convert_to_trt_plugin_onnx(raw_path, plugin_path):
    """
    Convert ONNX to TensorRT-plugin-ready ONNX.

    Handles:
    1. GridSample -> grid_sampler
    2. mmdeploy::grid_sampler -> default-domain grid_sampler
    3. removes initializers from graph inputs if any
    """

    model = onnx.load(raw_path)
    num_changed = 0

    for node in model.graph.node:
        should_convert = False

        if node.op_type == "GridSample":
            should_convert = True

        if node.op_type == "grid_sampler":
            should_convert = True

        if should_convert:
            num_changed += 1
            print(f"  Found op: domain='{node.domain}', op_type='{node.op_type}', converting to TensorRT plugin grid_sampler.")

            mode = "bilinear"
            padding_mode = "zeros"
            align_corners = 0

            # Read old attributes if they exist.
            for attr in node.attribute:
                if attr.name == "mode" and attr.type == onnx.AttributeProto.STRING:
                    mode = attr.s.decode("utf-8")
                elif attr.name == "padding_mode" and attr.type == onnx.AttributeProto.STRING:
                    padding_mode = attr.s.decode("utf-8")
                elif attr.name == "align_corners":
                    align_corners = int(attr.i)
                elif attr.name == "align_corners" and attr.type == onnx.AttributeProto.INT:
                    align_corners = int(attr.i)

            # Convert string attrs to plugin int attrs.
            interp_int = 0 if mode == "bilinear" else 1
            pad_int = 0 if padding_mode == "zeros" else (1 if padding_mode == "border" else 2)

            node.op_type = "grid_sampler"
            node.domain = ""

            del node.attribute[:]
            node.attribute.extend([
                helper.make_attribute("interpolation_mode", int(interp_int)),
                helper.make_attribute("padding_mode", int(pad_int)),
                helper.make_attribute("align_corners", int(align_corners)),
            ])

    # Remove custom opset import for mmdeploy if no node uses it now.
    used_domains = {node.domain for node in model.graph.node}
    new_imports = []
    for opset in model.opset_import:
        if opset.domain == "mmdeploy" and "mmdeploy" not in used_domains:
            continue
        new_imports.append(opset)

    del model.opset_import[:]
    model.opset_import.extend(new_imports)

    # Remove initializers from graph inputs if present.
    initializer_names = {init.name for init in model.graph.initializer}
    new_inputs = []
    removed_inputs = []

    for graph_input in model.graph.input:
        if graph_input.name in initializer_names:
            removed_inputs.append(graph_input.name)
        else:
            new_inputs.append(graph_input)

    if removed_inputs:
        del model.graph.input[:]
        model.graph.input.extend(new_inputs)

    onnx.save(model, plugin_path)

    print(f"  Saved plugin ONNX: {plugin_path}")
    print(f"  Converted plugin/GridSample nodes: {num_changed}")
    print(f"  Removed initializer inputs: {len(removed_inputs)}")


def export_one_mode(mode_name, mode_cfg, state_dict, device, output_dir):
    print("\n" + "=" * 80)
    print(f"Exporting mode: {mode_name}")
    print(f"use_prn={mode_cfg['use_prn']}, use_gate={mode_cfg['use_gate']}")
    print("=" * 80)

    model = EnhancedStereoNet(
        max_disp=MAX_DISP,
        use_prn=mode_cfg["use_prn"],
        use_gate=mode_cfg["use_gate"],
        debug=False
    ).to(device)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    if len(msg.missing_keys) > 0:
        print("First missing keys:")
        for k in msg.missing_keys[:10]:
            print(" ", k)

    if len(msg.unexpected_keys) > 0:
        print("First unexpected keys:")
        for k in msg.unexpected_keys[:10]:
            print(" ", k)

    model.eval()

    wrapped = ESNetONNXWrapper(model, mode_name).to(device)
    wrapped.eval()

    dummy_l = torch.randn(1, 3, INPUT_H, INPUT_W, device=device)
    dummy_r = torch.randn(1, 3, INPUT_H, INPUT_W, device=device)

    raw_path = os.path.join(output_dir, mode_cfg["onnx_raw"])
    plugin_path = os.path.join(output_dir, mode_cfg["onnx_plugin"])

    print(f"Exporting raw ONNX to: {raw_path}")

    with torch.no_grad(), patch_static_coordinate_ops_for_onnx():
        torch.onnx.export(
            wrapped,
            (dummy_l, dummy_r),
            raw_path,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            input_names=["left", "right"],
            output_names=["disparity"],
            dynamic_axes=None,
            custom_opsets={"mmdeploy": 11}
        )

    print("Checking raw ONNX...")
    raw_model = onnx.load(raw_path)
    onnx.checker.check_model(raw_model)
    print("Raw ONNX check passed.")
    inspect_onnx(raw_path)

    print("Converting to TensorRT plugin ONNX...")
    convert_to_trt_plugin_onnx(raw_path, plugin_path)

    print("Skipping ONNX checker for plugin ONNX because grid_sampler is a TensorRT custom plugin op.")
    inspect_onnx(plugin_path)

    del model
    del wrapped
    del dummy_l
    del dummy_r

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.weight}")
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"Checkpoint not found: {args.weight}")

    ckpt = torch.load(args.weight, map_location=device)
    state_dict = clean_state_dict(extract_state_dict(ckpt))

    if args.mode == "all":
        selected_modes = MODES.items()
    else:
        selected_modes = [(args.mode, MODES[args.mode])]

    for mode_name, mode_cfg in selected_modes:
        export_one_mode(mode_name, mode_cfg, state_dict, device, args.output_dir)

    print("\n" + "=" * 80)
    print("Selected ONNX files exported.")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
