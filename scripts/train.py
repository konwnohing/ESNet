
import os
import sys
import time
import argparse
import datetime

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet
from data.dataloader import create_dataloader
from engines.trainer import TrainerFixed
from engines.evaluator import EvaluatorFixed
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="OHEM ablation training for ESNet")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--ohem_ratio", type=float, required=True, help="OHEM ratio, e.g. 0.2 / 0.3 / 0.4")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--max-disp", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def colorize_tensor_for_tb(disp_tensor, max_disp):
    disp_np = disp_tensor[0, 0].detach().cpu().numpy()
    norm_disp = disp_np / (float(max_disp) + 1e-6)
    norm_disp = np.clip(norm_disp, 0, 1)
    colormap = plt.get_cmap("jet")
    color_disp = colormap(norm_disp)[..., :3]
    return color_disp


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


def apply_ohem_override(config, ratio):
    """
    关键修复：
    TrainerFixed 实际读取的是 config["training"]["loss"]，
    不是顶层 config["loss"]。
    因此这里必须把顶层 loss 同步到 training.loss。
    """
    if "training" not in config:
        config["training"] = {}
    if "loss" not in config:
        config["loss"] = {}
    if "logging" not in config:
        config["logging"] = {}

    ratio = float(ratio)

    # 顶层记录
    config["training"]["ohem_ratio"] = ratio
    config["loss"]["ohem_fraction"] = ratio

    # 关键：Trainer 读取 training.loss
    config["training"]["loss"] = dict(config["loss"])

    # 关键：Trainer 读取 training.loss_weights
    if "weights" in config["loss"]:
        config["training"]["loss_weights"] = dict(config["loss"]["weights"])

    # optimizer 兼容 trainer.py 的读取方式
    opt_cfg = config["training"].get("optimizer", {})
    if isinstance(opt_cfg, dict):
        if "weight_decay" in opt_cfg:
            config["training"]["weight_decay"] = opt_cfg["weight_decay"]

    # scheduler T_max 和 epochs 保持一致
    if "scheduler" in config["training"] and isinstance(config["training"]["scheduler"], dict):
        if "num_epochs" in config["training"]:
            config["training"]["scheduler"]["T_max"] = config["training"]["num_epochs"]

    # 每个 ratio 独立保存，避免覆盖
    ratio_tag = str(ratio).replace(".", "p")
    config["logging"]["checkpoint_dir"] = f"./checkpoints_ohem_ablation/rho_{ratio_tag}"
    config["logging"]["tensorboard_dir"] = f"./tensorboard_ohem_ablation/rho_{ratio_tag}"

    return config


def get_best_checkpoint_path(checkpoint_dir):
    candidates = [
        os.path.join(checkpoint_dir, "model_best.pth"),
        os.path.join(checkpoint_dir, "best.pth"),
        os.path.join(checkpoint_dir, "checkpoint_last.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    start_time = time.time()
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    config = apply_ohem_override(config, args.ohem_ratio)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    loss_cfg = train_cfg.get("loss", {})
    log_cfg = config.get("logging", {})

    max_disp = args.max_disp if args.max_disp is not None else model_cfg.get("max_disp", 192)
    backbone = args.backbone if args.backbone is not None else model_cfg.get("backbone_type", "resnet18")
    num_epochs = args.epochs if args.epochs is not None else train_cfg.get("num_epochs", 300)

    print("=" * 80)
    print("OHEM Ablation Setting")
    print("=" * 80)
    print(f"OHEM ratio: {loss_cfg.get('ohem_fraction')}")
    print(f"Input crop: {config['data'].get('crop_height')} x {config['data'].get('crop_width')}")
    print(f"Learning rate: {train_cfg.get('learning_rate')}")
    print(f"Batch size: {train_cfg.get('batch_size')}")
    print(f"Epochs: {num_epochs}")
    print(f"use_prn: {model_cfg.get('use_prn', True)}")
    print(f"use_gate: {model_cfg.get('use_gate', True)}")
    print(f"smoothness weight: {loss_cfg.get('smoothness', {}).get('weight')}")
    print(f"spatial_grad enabled: {loss_cfg.get('spatial_grad', {}).get('enabled')}")
    print(f"spatial_grad weight: {loss_cfg.get('spatial_grad', {}).get('weight')}")
    print(f"edge_alpha: {loss_cfg.get('edge_alpha')}")
    print(f"Checkpoint dir: {log_cfg.get('checkpoint_dir')}")
    print(f"TensorBoard dir: {log_cfg.get('tensorboard_dir')}")
    print("=" * 80)

    train_loader = create_dataloader(config, split="train")
    val_loader = create_dataloader(config, split="val")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = EnhancedStereoNet(
        backbone_type=backbone,
        max_disp=max_disp,
        use_prn=model_cfg.get("use_prn", True),
        use_gate=model_cfg.get("use_gate", True),
        debug=args.debug
    ).to(device)

    ckpt_path = args.resume if args.resume else model_cfg.get("checkpoint", None)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading pretrained checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = clean_state_dict(extract_state_dict(checkpoint))
        msg = model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded.")
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")
        if len(msg.missing_keys) > 0:
            print("First missing keys:", msg.missing_keys[:10])
        if len(msg.unexpected_keys) > 0:
            print("First unexpected keys:", msg.unexpected_keys[:10])
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    trainer = TrainerFixed(model, config, device)
    evaluator = EvaluatorFixed(
        max_disp=max_disp,
        metrics=["epe", "bad_1.0", "bad_3.0", "d1_all"],
        device=device
    )

    checkpoint_dir = log_cfg.get("checkpoint_dir", "./checkpoints_ohem_ablation/tmp")
    tb_dir = log_cfg.get("tensorboard_dir", "./tensorboard_ohem_ablation/tmp")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(tb_dir)

    best_metric = float("inf")
    best_epoch = -1
    best_metrics = {}

    print(f"\nStart training for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        epoch_num = epoch + 1
        model.debug = args.debug and epoch_num == 1

        train_losses = trainer.train_epoch(train_loader, epoch, writer)

        model.debug = False
        model.eval()
        with torch.no_grad():
            val_metrics = evaluator.evaluate(model, val_loader)

        cur_d1 = float(val_metrics.get("d1_all", float("inf")))
        is_best = cur_d1 < best_metric

        if is_best:
            best_metric = cur_d1
            best_epoch = epoch_num
            best_metrics = dict(val_metrics)

        trainer.save_checkpoint(epoch_num, checkpoint_dir, is_best=is_best)

        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(val_metrics.get("epe", float("inf")))
            else:
                trainer.scheduler.step()

        current_lr = trainer.optimizer.param_groups[0]["lr"] if hasattr(trainer, "optimizer") else 0.0

        print(
            f"Epoch [{epoch_num}/{num_epochs}] "
            f"Loss: {train_losses.get('total', 0):.4f} | "
            f"EPE: {val_metrics.get('epe', -1):.4f} | "
            f"D1-all: {val_metrics.get('d1_all', -1):.2f}% | "
            f"LR: {current_lr:.6g}"
        )

        writer.add_scalar("Train/loss", train_losses.get("total", 0), epoch_num)
        writer.add_scalar("Val/epe", val_metrics.get("epe", -1), epoch_num)
        writer.add_scalar("Val/d1_all", val_metrics.get("d1_all", -1), epoch_num)
        writer.add_scalar("Train/lr", current_lr, epoch_num)
        writer.add_scalar("Ablation/ohem_ratio", float(args.ohem_ratio), epoch_num)

        try:
            val_batch = next(iter(val_loader))
            v_left = val_batch["left"].to(device)
            v_right = val_batch["right"].to(device)
            v_gt = val_batch["disparity"].to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(v_left, v_right)
                if isinstance(outputs, dict):
                    if "disp_final" in outputs:
                        v_disp = outputs["disp_final"]
                    elif "disp_init" in outputs:
                        v_disp = outputs["disp_init"]
                    elif "all_disparities" in outputs and "p3" in outputs["all_disparities"]:
                        v_disp = outputs["all_disparities"]["p3"]
                    else:
                        v_disp = None
                else:
                    v_disp = outputs

            if v_disp is not None:
                if v_gt.dim() == 3:
                    v_gt = v_gt.unsqueeze(1)

                writer.add_image(
                    "Visuals/1_Prediction",
                    colorize_tensor_for_tb(v_disp, max_disp),
                    epoch_num,
                    dataformats="HWC"
                )
                writer.add_image(
                    "Visuals/2_GroundTruth",
                    colorize_tensor_for_tb(v_gt, max_disp),
                    epoch_num,
                    dataformats="HWC"
                )
        except Exception as e:
            print(f"[Warning] TensorBoard image logging skipped: {e}")

    total_time = time.time() - start_time
    finish_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    summary_path = os.path.join(checkpoint_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ESNet OHEM Ablation Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Finish time: {finish_time}\n")
        f.write(f"Total time: {total_time_str}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Device: {device}\n\n")

        f.write("[Ablation Setting]\n")
        f.write(f"OHEM ratio: {loss_cfg.get('ohem_fraction')}\n")
        f.write(f"PRN: {model_cfg.get('use_prn', True)}\n")
        f.write(f"Gate: {model_cfg.get('use_gate', True)}\n")
        f.write(f"Crop: {config['data'].get('crop_height')} x {config['data'].get('crop_width')}\n")
        f.write(f"Batch size: {train_cfg.get('batch_size')}\n")
        f.write(f"Learning rate: {train_cfg.get('learning_rate')}\n")
        f.write(f"Smoothness weight: {loss_cfg.get('smoothness', {}).get('weight')}\n")
        f.write(f"Spatial grad enabled: {loss_cfg.get('spatial_grad', {}).get('enabled')}\n")
        f.write(f"Spatial grad weight: {loss_cfg.get('spatial_grad', {}).get('weight')}\n")
        f.write(f"Edge alpha: {loss_cfg.get('edge_alpha')}\n")
        f.write(f"Epochs: {num_epochs}\n\n")

        f.write("[Best Validation]\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best D1-all: {best_metrics.get('d1_all', 'N/A')}\n")
        f.write(f"Best EPE: {best_metrics.get('epe', 'N/A')}\n")
        f.write(f"Best Bad 1.0: {best_metrics.get('bad_1.0', 'N/A')}\n")
        f.write(f"Best Bad 3.0: {best_metrics.get('bad_3.0', 'N/A')}\n")

    writer.close()
    print(f"Training finished. Summary saved to: {summary_path}")
    print(f"Best epoch: {best_epoch}, Best D1-all: {best_metric:.4f}")
    best_ckpt = get_best_checkpoint_path(checkpoint_dir)
    print(f"Best checkpoint path: {best_ckpt}")


if __name__ == "__main__":
    main()