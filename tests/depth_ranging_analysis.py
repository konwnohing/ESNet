import os
import sys
import csv
import argparse
import yaml
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet


def read_kitti_disp(path):
    """KITTI disparity png: uint16 / 256.0"""
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(path)
    return disp.astype(np.float32) / 256.0


def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_to_tensor(img_np, normalize, device):
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = normalize(tensor.squeeze(0)).unsqueeze(0)
    return tensor


def collect_kitti_samples(root_dir, split_file=""):
    """
    支持常见 KITTI 2015 结构：
    root_dir/training/image_2
    root_dir/training/image_3
    root_dir/training/disp_occ_0
    """
    left_dir = os.path.join(root_dir, "training", "image_2")
    right_dir = os.path.join(root_dir, "training", "image_3")
    disp_dir_occ = os.path.join(root_dir, "training", "disp_occ_0")
    disp_dir_noc = os.path.join(root_dir, "training", "disp_noc_0")

    if not os.path.exists(left_dir):
        raise FileNotFoundError(f"Cannot find left image dir: {left_dir}")
    if not os.path.exists(right_dir):
        raise FileNotFoundError(f"Cannot find right image dir: {right_dir}")

    if os.path.exists(disp_dir_occ):
        disp_dir = disp_dir_occ
    elif os.path.exists(disp_dir_noc):
        disp_dir = disp_dir_noc
    else:
        raise FileNotFoundError("Cannot find disp_occ_0 or disp_noc_0.")

    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        names = sorted([x for x in os.listdir(disp_dir) if x.endswith(".png")])

    samples = []

    for name in names:
        if not name.endswith(".png"):
            if len(name) == 6:
                name = f"{name}_10.png"
            else:
                name = f"{name}.png"

        left_path = os.path.join(left_dir, name)
        right_path = os.path.join(right_dir, name)
        disp_path = os.path.join(disp_dir, name)

        if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(disp_path):
            samples.append((name, left_path, right_path, disp_path))

    return samples


def parse_calib_file(calib_path):
    """
    读取 KITTI calib 文件，支持 P_rect_02 / P_rect_03 或 P2 / P3。
    返回 focal length f(px) 和 baseline B(m)。
    """
    data = {}

    with open(calib_path, "r") as f:
        for line in f.readlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values = value.strip().split()
            if len(values) == 12:
                data[key.strip()] = np.array([float(x) for x in values], dtype=np.float64).reshape(3, 4)

    p2 = None
    p3 = None

    for k in ["P_rect_02", "P2", "P_rect_2"]:
        if k in data:
            p2 = data[k]
            break

    for k in ["P_rect_03", "P3", "P_rect_3"]:
        if k in data:
            p3 = data[k]
            break

    if p2 is None or p3 is None:
        raise ValueError(f"Cannot find P2/P3 calibration in {calib_path}")

    f = float(p2[0, 0])

    # KITTI rectified projection:
    # Tx = -f * baseline, so camera center x = -P[0,3] / P[0,0]
    cx2 = -p2[0, 3] / p2[0, 0]
    cx3 = -p3[0, 3] / p3[0, 0]
    baseline = abs(cx3 - cx2)

    return f, baseline


def find_calib_file(root_dir, image_name):
    """
    image_name: 000000_10.png
    calib 通常对应 000000.txt
    """
    base = image_name.replace("_10.png", "").replace(".png", "")

    candidates = [
        os.path.join(root_dir, "training", "calib_cam_to_cam", f"{base}.txt"),
        os.path.join(root_dir, "training", "calib", f"{base}.txt"),
        os.path.join(root_dir, "calib_cam_to_cam", f"{base}.txt"),
        os.path.join(root_dir, "calib", f"{base}.txt"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


def get_focal_baseline(root_dir, image_name, calib_cfg):
    """
    优先读取 KITTI calib 文件。
    如果没有 calib 文件，则使用 fallback。
    """
    use_kitti_calib = calib_cfg.get("use_kitti_calib", True)

    if use_kitti_calib:
        calib_path = find_calib_file(root_dir, image_name)
        if calib_path is not None:
            return parse_calib_file(calib_path)

    fallback_f = calib_cfg.get("fallback_focal_length_px", None)
    fallback_b = calib_cfg.get("fallback_baseline_m", None)

    if fallback_f is None or fallback_b is None:
        raise FileNotFoundError(
            f"No calib file found for {image_name}, and fallback focal/baseline are not set. "
            f"Please check KITTI calib folder or set fallback_focal_length_px and fallback_baseline_m."
        )

    return float(fallback_f), float(fallback_b)


def select_disparity_output(outputs, target_h, target_w):
    """
    选择模型输出中的最终 disparity，并在必要时上采样到目标尺寸。
    """
    pred = None

    if isinstance(outputs, dict):
        preferred_keys = [
            "disp_final",
            "final_disp",
            "disp_refined",
            "pred_disp",
            "disparity",
            "disp",
            "disp_init",
        ]

        for k in preferred_keys:
            if k in outputs and torch.is_tensor(outputs[k]):
                pred = outputs[k]
                break

        if pred is None and "all_disparities" in outputs:
            all_disp = outputs["all_disparities"]
            if isinstance(all_disp, dict):
                candidates = [(k, v) for k, v in all_disp.items() if torch.is_tensor(v)]
                if candidates:
                    candidates = sorted(
                        candidates,
                        key=lambda x: x[1].shape[-2] * x[1].shape[-1],
                        reverse=True,
                    )
                    pred = candidates[0][1]

        if pred is None:
            candidates = [(k, v) for k, v in outputs.items() if torch.is_tensor(v)]
            if candidates:
                candidates = sorted(
                    candidates,
                    key=lambda x: x[1].shape[-2] * x[1].shape[-1],
                    reverse=True,
                )
                pred = candidates[0][1]

    elif isinstance(outputs, (list, tuple)):
        tensor_candidates = [x for x in outputs if torch.is_tensor(x)]
        if len(tensor_candidates) == 0:
            raise RuntimeError("No tensor output found.")
        pred = tensor_candidates[-1]

    elif torch.is_tensor(outputs):
        pred = outputs

    if pred is None:
        raise RuntimeError("No valid disparity output found.")

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)

    pred_h, pred_w = pred.shape[-2], pred.shape[-1]

    if pred_h != target_h or pred_w != target_w:
        scale_x = target_w / float(pred_w)
        pred = F.interpolate(
            pred,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ) * scale_x

    return pred


def predict_disp(model, left_img, right_img, normalize, device):
    left_tensor = image_to_tensor(left_img, normalize, device)
    right_tensor = image_to_tensor(right_img, normalize, device)

    _, _, h, w = left_tensor.shape

    top_pad = 32 - (h % 32) if h % 32 != 0 else 0
    right_pad = 32 - (w % 32) if w % 32 != 0 else 0

    left_tensor = F.pad(left_tensor, (0, right_pad, top_pad, 0))
    right_tensor = F.pad(right_tensor, (0, right_pad, top_pad, 0))

    padded_h, padded_w = left_tensor.shape[-2], left_tensor.shape[-1]

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(left_tensor, right_tensor)
        else:
            outputs = model(left_tensor, right_tensor)

    pred = select_disparity_output(outputs, padded_h, padded_w)

    pred = pred.squeeze().detach().cpu().numpy()

    if pred.ndim == 3:
        pred = pred[0]

    pred = pred[top_pad: top_pad + h, 0:w]

    return pred


def load_model(model_cfg, model_item, device):
    backbone_type = model_cfg.get("backbone_type", "resnet18")
    max_disp = model_cfg.get("max_disp", 192)

    name = model_item["name"]
    ckpt_path = model_item["checkpoint_path"]
    use_prn = model_item.get("use_prn", True)
    use_gate = model_item.get("use_gate", True)
    strict_load = model_item.get("strict_load", True)

    print("\n" + "=" * 80)
    print(f"🚀 Loading model: {name}")
    print(f"   checkpoint : {ckpt_path}")
    print(f"   use_prn    : {use_prn}")
    print(f"   use_gate   : {use_gate}")
    print(f"   strict     : {strict_load}")
    print("=" * 80)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = EnhancedStereoNet(
        backbone_type=backbone_type,
        max_disp=max_disp,
        use_prn=use_prn,
        use_gate=use_gate,
        debug=False,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
        if torch.is_tensor(v)
    }

    result = model.load_state_dict(state_dict, strict=strict_load)

    if not strict_load:
        print(f"   missing keys   : {len(result.missing_keys)}")
        print(f"   unexpected keys: {len(result.unexpected_keys)}")

    model.eval()
    print("✅ Model loaded successfully.")

    return model


class RangeAccumulator:
    def __init__(self):
        self.count = 0
        self.abs_depth_sum = 0.0
        self.sq_depth_sum = 0.0
        self.abs_rel_sum = 0.0
        self.bad_count = 0
        self.disp_abs_sum = 0.0

    def update(self, depth_err, depth_gt, disp_err, bad_mask):
        n = depth_err.size
        if n == 0:
            return

        self.count += int(n)
        self.abs_depth_sum += float(np.sum(np.abs(depth_err)))
        self.sq_depth_sum += float(np.sum(depth_err ** 2))
        self.abs_rel_sum += float(np.sum(np.abs(depth_err) / np.maximum(depth_gt, 1e-6)))
        self.bad_count += int(np.sum(bad_mask))
        self.disp_abs_sum += float(np.sum(np.abs(disp_err)))

    def to_row(self, model_name, range_name):
        if self.count == 0:
            return {
                "model": model_name,
                "range": range_name,
                "valid_pixels": 0,
                "depth_mae_m": np.nan,
                "depth_rmse_m": np.nan,
                "abs_rel": np.nan,
                "bad_depth_rate_percent": np.nan,
                "disp_epe_px": np.nan,
            }

        return {
            "model": model_name,
            "range": range_name,
            "valid_pixels": self.count,
            "depth_mae_m": self.abs_depth_sum / self.count,
            "depth_rmse_m": np.sqrt(self.sq_depth_sum / self.count),
            "abs_rel": self.abs_rel_sum / self.count,
            "bad_depth_rate_percent": self.bad_count / self.count * 100.0,
            "disp_epe_px": self.disp_abs_sum / self.count,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    calib_cfg = config.get("calibration", {})
    eval_cfg = config["evaluation"]

    root_dir = data_cfg["root_dir"]
    save_dir = eval_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    normalize = T.Normalize(mean=data_cfg["mean"], std=data_cfg["std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = collect_kitti_samples(root_dir, data_cfg.get("split_file", ""))
    print(f"📊 Found {len(samples)} KITTI samples.")

    if len(samples) == 0:
        print("❌ No KITTI samples found. Please check root_dir or split_file.")
        return

    min_disp = float(eval_cfg.get("min_disp", 0.1))
    max_depth_m = float(eval_cfg.get("max_depth_m", 80.0))
    bad_abs_th = float(eval_cfg.get("bad_depth_abs_threshold_m", 1.0))
    bad_rel_th = float(eval_cfg.get("bad_depth_rel_threshold", 0.05))

    range_cfgs = eval_cfg["depth_ranges"]

    summary_rows = []
    per_image_rows = []

    for model_item in eval_cfg["models"]:
        model_name = model_item["name"]
        model = load_model(model_cfg, model_item, device)

        accumulators = {
            r["name"]: RangeAccumulator()
            for r in range_cfgs
        }
        accumulators["All"] = RangeAccumulator()

        for image_name, left_path, right_path, disp_path in tqdm(samples, desc=model_name):
            left_img = load_image(left_path)
            right_img = load_image(right_path)
            gt_disp = read_kitti_disp(disp_path)

            pred_disp = predict_disp(model, left_img, right_img, normalize, device)

            f_px, baseline_m = get_focal_baseline(root_dir, image_name, calib_cfg)
            fb = f_px * baseline_m

            valid = (
                (gt_disp > min_disp)
                & (pred_disp > min_disp)
                & np.isfinite(gt_disp)
                & np.isfinite(pred_disp)
            )

            gt_depth = np.zeros_like(gt_disp, dtype=np.float32)
            pred_depth = np.zeros_like(pred_disp, dtype=np.float32)

            gt_depth[valid] = fb / gt_disp[valid]
            pred_depth[valid] = fb / pred_disp[valid]

            valid = valid & (gt_depth > 0) & (gt_depth <= max_depth_m) & np.isfinite(gt_depth) & np.isfinite(pred_depth)

            if valid.sum() == 0:
                continue

            depth_err = pred_depth[valid] - gt_depth[valid]
            disp_err = pred_disp[valid] - gt_disp[valid]
            rel_err = np.abs(depth_err) / np.maximum(gt_depth[valid], 1e-6)

            bad_mask = (np.abs(depth_err) > bad_abs_th) & (rel_err > bad_rel_th)

            accumulators["All"].update(
                depth_err=depth_err,
                depth_gt=gt_depth[valid],
                disp_err=disp_err,
                bad_mask=bad_mask,
            )

            # per-image all row
            per_image_rows.append({
                "model": model_name,
                "image": image_name,
                "range": "All",
                "f_px": f_px,
                "baseline_m": baseline_m,
                "valid_pixels": int(valid.sum()),
                "depth_mae_m": float(np.mean(np.abs(depth_err))),
                "depth_rmse_m": float(np.sqrt(np.mean(depth_err ** 2))),
                "abs_rel": float(np.mean(rel_err)),
                "bad_depth_rate_percent": float(np.mean(bad_mask) * 100.0),
                "disp_epe_px": float(np.mean(np.abs(disp_err))),
            })

            for r in range_cfgs:
                r_name = r["name"]
                r_min = float(r["min_m"])
                r_max = float(r["max_m"])

                range_mask_full = valid & (gt_depth >= r_min) & (gt_depth < r_max)

                if range_mask_full.sum() == 0:
                    continue

                de = pred_depth[range_mask_full] - gt_depth[range_mask_full]
                dispe = pred_disp[range_mask_full] - gt_disp[range_mask_full]
                gt_d = gt_depth[range_mask_full]
                rel = np.abs(de) / np.maximum(gt_d, 1e-6)
                bad = (np.abs(de) > bad_abs_th) & (rel > bad_rel_th)

                accumulators[r_name].update(
                    depth_err=de,
                    depth_gt=gt_d,
                    disp_err=dispe,
                    bad_mask=bad,
                )

                per_image_rows.append({
                    "model": model_name,
                    "image": image_name,
                    "range": r_name,
                    "f_px": f_px,
                    "baseline_m": baseline_m,
                    "valid_pixels": int(range_mask_full.sum()),
                    "depth_mae_m": float(np.mean(np.abs(de))),
                    "depth_rmse_m": float(np.sqrt(np.mean(de ** 2))),
                    "abs_rel": float(np.mean(rel)),
                    "bad_depth_rate_percent": float(np.mean(bad) * 100.0),
                    "disp_epe_px": float(np.mean(np.abs(dispe))),
                })

        # summary rows
        for name, acc in accumulators.items():
            summary_rows.append(acc.to_row(model_name, name))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if eval_cfg.get("save_csv", True):
        summary_path = os.path.join(save_dir, "depth_summary.csv")
        per_image_path = os.path.join(save_dir, "depth_per_image.csv")

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        with open(per_image_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

        print("\n" + "=" * 80)
        print(f"📁 Summary saved to: {summary_path}")
        print(f"📁 Per-image results saved to: {per_image_path}")
        print("=" * 80)

    print("\nDepth / Ranging Analysis Summary")
    print("=" * 80)
    for row in summary_rows:
        print(
            f"{row['model']:<25} | {row['range']:<8} | "
            f"MAE: {row['depth_mae_m']:.4f} m | "
            f"RMSE: {row['depth_rmse_m']:.4f} m | "
            f"AbsRel: {row['abs_rel']:.4f} | "
            f"Bad-Z: {row['bad_depth_rate_percent']:.2f}% | "
            f"DispEPE: {row['disp_epe_px']:.4f} px | "
            f"N: {row['valid_pixels']}"
        )


if __name__ == "__main__":
    main()