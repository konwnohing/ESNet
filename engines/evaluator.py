#evaluator
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from utils.metric import compute_epe, compute_bad_pixels


class EvaluatorFixed:
    """评估器 - 修复版（包含 KITTI D1-all 官方标准评估）"""

    def __init__(
        self,
        max_disp: int = 192,
        metrics: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.max_disp = int(max_disp)
        # 🌟 默认加入 d1_all
        self.metrics = metrics or ["epe", "bad_1.0", "bad_3.0", "bad_5.0", "d1_all"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: Dict[str, float] = {}

    @staticmethod
    def _get_pred_disp(outputs) -> torch.Tensor:
        """兼容 dict/tuple/tensor 三种输出，取最终视差"""
        if isinstance(outputs, dict):
            # 你模型顶层标准输出
            if "disp_final" in outputs and torch.is_tensor(outputs["disp_final"]):
                return outputs["disp_final"]

            # 保险：有些实现只放在 all_disparities 里
            if "all_disparities" in outputs and isinstance(outputs["all_disparities"], dict):
                ad = outputs["all_disparities"]
                if "final" in ad and torch.is_tensor(ad["final"]):
                    return ad["final"]

            # 兜底 key
            for k in ["disparity", "pred", "disp"]:
                if k in outputs and torch.is_tensor(outputs[k]):
                    return outputs[k]

            raise KeyError("模型输出是 dict，但找不到 disp_final / all_disparities.final / disparity / pred / disp")
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 0:
                raise ValueError("模型输出是 tuple/list 但为空")
            return outputs[0]
        if torch.is_tensor(outputs):
            return outputs
        raise TypeError(f"无法解析模型输出类型: {type(outputs)}")

    @staticmethod
    def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
        """统一成 [B,1,H,W]"""
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            # [B,H,W] -> [B,1,H,W]
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError(f"Unsupported tensor dim: {x.dim()}")

    @staticmethod
    def _align_gt_to_pred(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """把 GT resize 到 pred 尺寸（nearest，避免改变数值）"""
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")
        return gt

    def evaluate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.eval()

        all_metrics: Dict[str, List[float]] = {m: [] for m in self.metrics}
        total_time = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 取数据
                if isinstance(batch, dict):
                    left_img = batch["left"].to(self.device, non_blocking=True)
                    right_img = batch["right"].to(self.device, non_blocking=True)
                    disp_gt = batch["disparity"].to(self.device, non_blocking=True)
                else:
                    left_img, right_img, disp_gt = batch
                    left_img = left_img.to(self.device, non_blocking=True)
                    right_img = right_img.to(self.device, non_blocking=True)
                    disp_gt = disp_gt.to(self.device, non_blocking=True)

                bs = left_img.size(0)

                # 推理计时（CUDA需要同步）
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                outputs = model(left_img, right_img)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                total_time += (time.time() - t0)

                num_samples += bs

                pred_disp = self._to_b1hw(self._get_pred_disp(outputs))
                gt_disp = self._to_b1hw(disp_gt)
                # 如果尺寸不一致：GT resize 到 pred（nearest），并把 GT 的 disparity 数值按宽度比例缩放到 pred 的像素单位
                W0 = gt_disp.shape[-1]
                gt_disp = self._align_gt_to_pred(pred_disp, gt_disp)
                scale = float(pred_disp.shape[-1]) / float(W0) if W0 > 0 else 1.0
                if abs(scale - 1.0) > 1e-6:
                    gt_disp = gt_disp * scale
                max_disp_scaled = float(self.max_disp) * scale

                # 有效 mask：0 < gt < max_disp，并且 pred/gt 都是 finite
                valid_mask = (
                    (gt_disp > 0)
                    & (gt_disp < float(max_disp_scaled))
                    & torch.isfinite(gt_disp)
                    & torch.isfinite(pred_disp)
                )

                # 计算指标
                for metric in self.metrics:
                    if metric == "epe":
                        epe = compute_epe(pred_disp, gt_disp, valid_mask)
                        all_metrics["epe"].append(float(epe.item()))
                        
                    elif metric.startswith("bad_"):
                        thr = float(metric.split("_")[1]) * scale
                        bad = compute_bad_pixels(pred_disp, gt_disp, thr, valid_mask)               
                        all_metrics[metric].append(float(bad.item()))
                        
                    # 🌟 核心：新增的 D1-all 计算逻辑！
                    elif metric == "d1_all":
                        if valid_mask.any():
                            # 只抽出有效区域进行计算，防止除以 0 导致 nan
                            pred_valid = pred_disp[valid_mask]
                            gt_valid = gt_disp[valid_mask]
                            
                            abs_err = torch.abs(pred_valid - gt_valid)
                            
                            # 黄金法则：绝对误差 > 3px 且 相对误差 > 5%
                            err_mask = (abs_err > 3.0 * scale) & ((abs_err / gt_valid) > 0.05)
                            
                            d1_all_batch = (err_mask.float().mean()) * 100.0
                        else:
                            d1_all_batch = torch.tensor(0.0, device=self.device)
                            
                        all_metrics["d1_all"].append(float(d1_all_batch.item()))

                if batch_idx % 10 == 0:
                    print(f"评估进度: [{batch_idx}/{len(dataloader)}]")

        # 汇总
        avg_metrics: Dict[str, float] = {}
        for metric, values in all_metrics.items():
            avg_metrics[metric] = float(np.mean(values)) if len(values) > 0 else 0.0

        if num_samples > 0:
            avg_metrics["inference_time"] = total_time / num_samples  # 秒/张
            avg_metrics["fps"] = 1.0 / (avg_metrics["inference_time"] + 1e-12)

        self.results = avg_metrics
        return avg_metrics

    def print_results(self, title: str = "评估结果"):
        if not self.results:
            print("没有可用的结果")
            return

        print("=" * 60)
        print(title)
        print("-" * 60)

        for metric, value in self.results.items():
            if metric == "epe":
                print(f"EPE: {value:.4f} 像素")
            elif metric.startswith("bad_"):
                # compute_bad_pixels 返回百分比(0~100)
                print(f"Bad pixels ({metric.split('_')[1]}px): {value:.2f}%")
            # 🌟 新增 D1-all 打印格式
            elif metric == "d1_all":
                print(f"D1-all (KITTI 标准): {value:.2f}%")
            elif metric == "inference_time":
                print(f"推理时间: {value * 1000:.2f} ms/图像")
            elif metric == "fps":
                print(f"FPS: {value:.2f}")

        print("=" * 60)