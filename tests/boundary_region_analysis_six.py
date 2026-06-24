import os
import sys
import argparse
import yaml
import csv
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


def make_edge_mask(left_img, percentile=85, dilation_radius=3):
    """
    基于左图 Sobel 梯度生成边界区域 mask。
    left_img: H x W x 3, float32, [0,1]
    """
    img_u8 = (left_img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    valid_mag = mag[mag > 0]
    if valid_mag.size == 0:
        return np.zeros_like(gray, dtype=bool), mag

    threshold = np.percentile(valid_mag, percentile)
    edge = mag >= threshold

    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)

    return edge, mag


def image_to_tensor(img_np, normalize, device):
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = normalize(tensor.squeeze(0)).unsqueeze(0)
    return tensor


def predict_disp(model, left_img, right_img, normalize, device):
    left_tensor = image_to_tensor(left_img, normalize, device)
    right_tensor = image_to_tensor(right_img, normalize, device)

    _, _, h, w = left_tensor.shape

    top_pad = 32 - (h % 32) if h % 32 != 0 else 0
    right_pad = 32 - (w % 32) if w % 32 != 0 else 0

    left_tensor = F.pad(left_tensor, (0, right_pad, top_pad, 0))
    right_tensor = F.pad(right_tensor, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(left_tensor, right_tensor)
        else:
            outputs = model(left_tensor, right_tensor)

    print("OUTPUT TYPE:", type(outputs))

    if isinstance(outputs, dict):
        print("DICT KEYS:", outputs.keys())
        for k, v in outputs.items():
            if torch.is_tensor(v):
                print(k, v.shape, v.min().item(), v.max().item(), v.mean().item())
            elif isinstance(v, dict):
                print(k, "nested dict:", v.keys())
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        print(" ", kk, vv.shape, vv.min().item(), vv.max().item(), vv.mean().item())

    elif isinstance(outputs, (list, tuple)):
        print("TUPLE/LIST LEN:", len(outputs))
        for i, v in enumerate(outputs):
            if torch.is_tensor(v):
                print(i, v.shape, v.min().item(), v.max().item(), v.mean().item())

    elif torch.is_tensor(outputs):
        print("TENSOR:", outputs.shape, outputs.min().item(), outputs.max().item(), outputs.mean().item())
        
    pred = outputs

    if isinstance(pred, dict):
        pred = pred.get("disp_final", list(pred.values())[0])

    if isinstance(pred, (list, tuple)):
        pred = pred[-1]

    pred = pred.squeeze().detach().cpu().numpy()

    if pred.ndim == 3:
        pred = pred[0]

    pred = pred[top_pad: top_pad + h, 0:w]
    return pred


def compute_metrics(pred, gt, mask):
    if mask.sum() == 0:
        return {
            "epe": np.nan,
            "d1": np.nan,
            "bad1": np.nan,
            "bad3": np.nan,
            "valid_pixels": 0
        }

    abs_err = np.abs(pred[mask] - gt[mask])
    gt_valid = gt[mask]

    epe = float(np.mean(abs_err))

    d1 = float(
        np.mean((abs_err > 3.0) & (abs_err / gt_valid > 0.05)) * 100.0
    )

    bad1 = float(np.mean(abs_err > 1.0) * 100.0)
    bad3 = float(np.mean(abs_err > 3.0) * 100.0)

    return {
        "epe": epe,
        "d1": d1,
        "bad1": bad1,
        "bad3": bad3,
        "valid_pixels": int(mask.sum())
    }


def weighted_average(metric_list, key):
    total = 0.0
    pixels = 0

    for item in metric_list:
        if np.isnan(item[key]):
            continue
        n = item["valid_pixels"]
        total += item[key] * n
        pixels += n

    if pixels == 0:
        return np.nan

    return total / pixels


def collect_kitti_samples(root_dir, split_file=""):
    left_dir = os.path.join(root_dir, "training", "image_2")
    right_dir = os.path.join(root_dir, "training", "image_3")
    disp_dir_occ = os.path.join(root_dir, "training", "disp_occ_0")
    disp_dir_noc = os.path.join(root_dir, "training", "disp_noc_0")

    if os.path.exists(disp_dir_occ):
        disp_dir = disp_dir_occ
    elif os.path.exists(disp_dir_noc):
        disp_dir = disp_dir_noc
    else:
        raise FileNotFoundError("Cannot find disp_occ_0 or disp_noc_0 in KITTI training folder.")

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


def save_edge_visual(left_img, edge_mask, save_path):
    img_u8 = (left_img * 255).astype(np.uint8).copy()
    overlay = img_u8.copy()
    overlay[edge_mask] = [255, 0, 0]
    out = cv2.addWeighted(img_u8, 0.65, overlay, 0.35, 0)
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


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
        debug=False
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    state_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    result = model.load_state_dict(state_dict, strict=strict_load)

    if not strict_load:
        print(f"   missing keys   : {len(result.missing_keys)}")
        print(f"   unexpected keys: {len(result.unexpected_keys)}")

    model.eval()
    print("✅ Model loaded successfully.")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]

    save_dir = eval_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    normalize = T.Normalize(mean=data_cfg["mean"], std=data_cfg["std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_percentile = eval_cfg.get("edge_percentile", 85)
    dilation_radius = eval_cfg.get("dilation_radius", 3)
    max_save_vis = eval_cfg.get("max_save_vis", 5)

    samples = collect_kitti_samples(
        data_cfg["root_dir"],
        data_cfg.get("split_file", "")
    )

    print(f"📊 Found {len(samples)} KITTI samples.")

    if len(samples) == 0:
        print("❌ No KITTI samples found. Please check root_dir or split_file.")
        return

    summary_rows = []
    per_image_rows = []

    for model_item in eval_cfg["models"]:
        model_name = model_item["name"]
        group = model_item.get("group", "")
        loss_type = model_item.get("loss_type", "")

        model = load_model(model_cfg, model_item, device)

        full_metrics_all = []
        edge_metrics_all = []
        nonedge_metrics_all = []

        for idx, (img_name, left_path, right_path, disp_path) in enumerate(
            tqdm(samples, desc=model_name)
        ):
            left_img = load_image(left_path)
            right_img = load_image(right_path)
            gt = read_kitti_disp(disp_path)

            pred = predict_disp(model, left_img, right_img, normalize, device)

            valid_mask = (gt > 0) & np.isfinite(gt)

            edge_mask, _ = make_edge_mask(
                left_img,
                percentile=edge_percentile,
                dilation_radius=dilation_radius
            )

            edge_valid_mask = valid_mask & edge_mask
            nonedge_valid_mask = valid_mask & (~edge_mask)

            full_m = compute_metrics(pred, gt, valid_mask)
            edge_m = compute_metrics(pred, gt, edge_valid_mask)
            nonedge_m = compute_metrics(pred, gt, nonedge_valid_mask)

            full_metrics_all.append(full_m)
            edge_metrics_all.append(edge_m)
            nonedge_metrics_all.append(nonedge_m)

            per_image_rows.append({
                "group": group,
                "model": model_name,
                "loss_type": loss_type,
                "image": img_name,

                "full_epe": full_m["epe"],
                "full_d1": full_m["d1"],
                "full_bad1": full_m["bad1"],
                "full_bad3": full_m["bad3"],
                "full_valid_pixels": full_m["valid_pixels"],

                "edge_epe": edge_m["epe"],
                "edge_d1": edge_m["d1"],
                "edge_bad1": edge_m["bad1"],
                "edge_bad3": edge_m["bad3"],
                "edge_valid_pixels": edge_m["valid_pixels"],

                "nonedge_epe": nonedge_m["epe"],
                "nonedge_d1": nonedge_m["d1"],
                "nonedge_bad1": nonedge_m["bad1"],
                "nonedge_bad3": nonedge_m["bad3"],
                "nonedge_valid_pixels": nonedge_m["valid_pixels"],
            })

            if eval_cfg.get("save_edge_visualization", False) and idx < max_save_vis:
                vis_dir = os.path.join(save_dir, "edge_vis")
                os.makedirs(vis_dir, exist_ok=True)
                save_edge_visual(
                    left_img,
                    edge_mask,
                    os.path.join(
                        vis_dir,
                        f"{model_name}_{img_name.replace('.png', '')}_edge.png"
                    )
                )

        row = {
            "group": group,
            "model": model_name,
            "loss_type": loss_type,

            "full_epe": weighted_average(full_metrics_all, "epe"),
            "full_d1": weighted_average(full_metrics_all, "d1"),
            "full_bad1": weighted_average(full_metrics_all, "bad1"),
            "full_bad3": weighted_average(full_metrics_all, "bad3"),

            "edge_epe": weighted_average(edge_metrics_all, "epe"),
            "edge_d1": weighted_average(edge_metrics_all, "d1"),
            "edge_bad1": weighted_average(edge_metrics_all, "bad1"),
            "edge_bad3": weighted_average(edge_metrics_all, "bad3"),

            "nonedge_epe": weighted_average(nonedge_metrics_all, "epe"),
            "nonedge_d1": weighted_average(nonedge_metrics_all, "d1"),
            "nonedge_bad1": weighted_average(nonedge_metrics_all, "bad1"),
            "nonedge_bad3": weighted_average(nonedge_metrics_all, "bad3"),
        }

        summary_rows.append(row)

        print("\n" + "=" * 80)
        print(f"✅ Boundary Analysis Summary: {model_name}")
        print("=" * 80)
        print(f"Group: {group} | Loss: {loss_type}")
        print(
            f"Full    EPE: {row['full_epe']:.4f} | "
            f"D1: {row['full_d1']:.2f}% | "
            f"Bad1: {row['full_bad1']:.2f}% | "
            f"Bad3: {row['full_bad3']:.2f}%"
        )
        print(
            f"Boundary EPE: {row['edge_epe']:.4f} | "
            f"D1: {row['edge_d1']:.2f}% | "
            f"Bad1: {row['edge_bad1']:.2f}% | "
            f"Bad3: {row['edge_bad3']:.2f}%"
        )
        print(
            f"NonEdge EPE: {row['nonedge_epe']:.4f} | "
            f"D1: {row['nonedge_d1']:.2f}% | "
            f"Bad1: {row['nonedge_bad1']:.2f}% | "
            f"Bad3: {row['nonedge_bad3']:.2f}%"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if eval_cfg.get("save_csv", True):
        summary_path = os.path.join(save_dir, "boundary_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        per_image_path = os.path.join(save_dir, "boundary_per_image.csv")
        with open(per_image_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

        print("\n" + "=" * 80)
        print(f"📁 Summary saved to: {summary_path}")
        print(f"📁 Per-image results saved to: {per_image_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()