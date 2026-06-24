
import os
import sys
import argparse
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet
from data.dataloader import create_dataloader
from engines.evaluator import EvaluatorFixed
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OHEM ablation checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        cleaned[k.replace("module.", "")] = v
    return cleaned


def sync_loss_config(config):
    """
    与 train_ohem.py 保持一致：
    evaluator 本身不用 loss，但保留结构一致，防止 dataloader/trainer 配置差异。
    """
    if "training" not in config:
        config["training"] = {}
    if "loss" in config:
        config["training"]["loss"] = dict(config["loss"])
        if "weights" in config["loss"]:
            config["training"]["loss_weights"] = dict(config["loss"]["weights"])
    return config


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    config = sync_loss_config(config)

    # 评估必须关闭数据增强
    config.setdefault("data", {})
    config["data"].setdefault("augmentation", {})
    config["data"]["augmentation"]["enabled"] = False

    # 评估 batch size 固定为 1，更稳定，也更接近逐图评估
    config.setdefault("training", {})
    config["training"]["batch_size"] = 1
    config["training"]["num_workers"] = config["training"].get("num_workers", 4)

    model_cfg = config.get("model", {})
    max_disp = model_cfg.get("max_disp", 192)

    print("=" * 80)
    print("Evaluation Setting")
    print("=" * 80)
    print(f"Checkpoint: {args.resume}")
    print(f"use_prn: {model_cfg.get('use_prn', True)}")
    print(f"use_gate: {model_cfg.get('use_gate', True)}")
    print(f"Crop: {config['data'].get('crop_height')} x {config['data'].get('crop_width')}")
    print(f"Batch size: {config['training'].get('batch_size')}")
    print(f"FP16: {args.fp16}")
    print("=" * 80)

    model = EnhancedStereoNet(
        backbone_type=model_cfg.get("backbone_type", "resnet18"),
        max_disp=max_disp,
        use_prn=model_cfg.get("use_prn", True),
        use_gate=model_cfg.get("use_gate", True),
        debug=False
    ).to(device)

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    msg = model.load_state_dict(state_dict, strict=False)

    print("Checkpoint loaded.")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print("First missing keys:", msg.missing_keys[:10])
    if len(msg.unexpected_keys) > 0:
        print("First unexpected keys:", msg.unexpected_keys[:10])

    val_loader = create_dataloader(config, split="val")

    evaluator = EvaluatorFixed(
        max_disp=max_disp,
        metrics=["epe", "bad_1.0", "bad_3.0", "d1_all"],
        device=device
    )

    model.eval()

    with torch.no_grad():
        if args.fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = evaluator.evaluate(model, val_loader)
        else:
            results = evaluator.evaluate(model, val_loader)

    print("\n" + "=" * 80)
    print("FINAL_EVAL_RESULT")
    print("=" * 80)
    print(f"EPE: {results.get('epe', 0):.4f}")
    print(f"Bad_1.0: {results.get('bad_1.0', 0):.4f}")
    print(f"Bad_3.0: {results.get('bad_3.0', 0):.4f}")
    print(f"D1_all: {results.get('d1_all', 0):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()