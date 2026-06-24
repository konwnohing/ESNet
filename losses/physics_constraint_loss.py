#physics_constaint_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional


def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
    """Ensure disparity is [B,1,H,W]."""
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Unsupported disparity dim: {x.dim()}")


def _infer_device(predictions: Union[torch.Tensor, Dict]) -> torch.device:
    if torch.is_tensor(predictions):
        return predictions.device
    if isinstance(predictions, dict):
        for v in predictions.values():
            if torch.is_tensor(v):
                return v.device
            if isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        return vv.device
    return torch.device("cpu")


def _collect_disparities(predictions: Union[torch.Tensor, Dict]) -> List[Tuple[str, torch.Tensor]]:
    """Collect candidate disparity tensors from outputs."""
    if torch.is_tensor(predictions):
        return [("tensor", predictions)]

    if not isinstance(predictions, dict):
        return []

    items: List[Tuple[str, torch.Tensor]] = []

    ad = predictions.get("all_disparities", None)
    if isinstance(ad, dict):
        for k, v in ad.items():
            if torch.is_tensor(v):
                items.append((str(k), v))

    known_scale_keys = {"c5", "p5", "p4", "p3", "p2", "p1", "full_res", "final"}
    for k, v in predictions.items():
        if not torch.is_tensor(v):
            continue
        ks = str(k)
        if ks.startswith("disp") or ks in known_scale_keys:
            items.append((ks, v))

    seen = set()
    uniq: List[Tuple[str, torch.Tensor]] = []
    for k, v in items:
        key = v.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((k, v))
    return uniq


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失（极其稳健的安全版）
    - 修复了强迫网络产生 80% max_disp 的灾难性幻觉 Bug
    """

    def __init__(self, weight: float = 1.0, mode: str = "all", max_disp: int = 192, verbose: bool = False):
        super().__init__()
        self.weight = float(weight)
        self.mode = str(mode)
        self.max_disp = float(max_disp)
        self.verbose = bool(verbose)

        if self.verbose:
            print(f"[PhysicsConstraintLoss] init: mode={self.mode}, weight={self.weight}, max_disp={self.max_disp}")

    def _compute_single_loss(self, disparity: torch.Tensor) -> torch.Tensor:
        disp = _to_b1hw(disparity)

        # 1) negative disparity penalty (安全写法)
        negative_penalty = F.relu(-disp).mean()

        # 2) smoothness (simple TV without edge-aware, used as a gentle global regularizer)
        grad_x = (disp[:, :, :, 1:] - disp[:, :, :, :-1]).abs()
        grad_y = (disp[:, :, 1:, :] - disp[:, :, :-1, :]).abs()
        smoothness_loss = grad_x.mean() + grad_y.mean()

        # 3) 🚨 修复灾难级 Bug：仅保留溢出惩罚，移除会导致幻觉（Hallucination）的强行边界占比惩罚
        overflow_penalty = F.relu(disp - self.max_disp).mean()
        
        # 将 overflow 视为绝对的红线边界约束
        range_loss = overflow_penalty

        # 🌟 手术修改 2：将 smoothness_loss 的权重从 0.1 改为 0.0，彻底禁用这里的无脑平滑
        total = 0.5 * negative_penalty + 0.0 * smoothness_loss + 0.3 * range_loss
        return total

    def forward(self, predictions: Union[torch.Tensor, Dict]) -> torch.Tensor:
        device = _infer_device(predictions)
        if self.weight <= 0:
            return torch.zeros((), device=device)

        items = _collect_disparities(predictions)
        if not items:
            return torch.zeros((), device=device)

        selected: List[torch.Tensor] = []

        if self.mode == "all":
            selected = [v for _, v in items]
        elif self.mode == "full":
            for k, v in items:
                if k == "full_res" or "full" in k:
                    selected.append(v)
            if not selected:
                for k, v in items:
                    if k == "final" or "final" in k or k == "disp_final":
                        selected.append(v)
        elif self.mode == "final":
            for k, v in items:
                if k == "disp_final" or k == "final" or "final" in k:
                    selected.append(v)
            if not selected:
                for k, v in items:
                    if k == "full_res":
                        selected.append(v)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if not selected:
            return torch.zeros((), device=device)

        losses = [self._compute_single_loss(d) for d in selected]
        loss = torch.stack(losses).mean()
        return loss * self.weight


class NegativeDisparityLoss(nn.Module):
    """负视差惩罚（稳健版）"""

    def __init__(self, weight: float = 0.1, threshold: float = 0.0):
        super().__init__()
        self.weight = float(weight)
        self.threshold = float(threshold)

    def forward(self, predictions: Union[torch.Tensor, Dict]) -> torch.Tensor:
        device = _infer_device(predictions)
        if self.weight <= 0:
            return torch.zeros((), device=device)

        items = _collect_disparities(predictions)
        if not items:
            return torch.zeros((), device=device)

        penalties: List[torch.Tensor] = []
        for _, v in items:
            disp = _to_b1hw(v)
            # 🌟 修复：直接用 F.relu 计算低于阈值的惩罚量，避免索引孤立点造成的梯度爆炸
            penalty = F.relu(self.threshold - disp).mean()
            if penalty > 0:
                penalties.append(penalty)

        if not penalties:
            return torch.zeros((), device=device)
        return torch.stack(penalties).mean() * self.weight


class DisparitySmoothnessLoss(nn.Module):
    """视差平滑性（🌟包含 alpha 的高级 Edge-Aware 版本）"""

    def __init__(self, weight: float = 0.01, edge_alpha: float = 10.0):
        super().__init__()
        self.weight = float(weight)
        self.edge_alpha = float(edge_alpha) 

    def forward(self, predictions: Union[torch.Tensor, Dict], image: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = _infer_device(predictions)
        if self.weight <= 0:
            return torch.zeros((), device=device)

        items = _collect_disparities(predictions)
        if not items:
            return torch.zeros((), device=device)

        losses: List[torch.Tensor] = []
        for _, v in items:
            disp = _to_b1hw(v)

            grad_x = (disp[:, :, :, 1:] - disp[:, :, :, :-1]).abs()
            grad_y = (disp[:, :, 1:, :] - disp[:, :, :-1, :]).abs()

            if image is not None and torch.is_tensor(image):
                img = image
                if img.shape[-2:] != disp.shape[-2:]:
                    img = F.interpolate(img, size=disp.shape[-2:], mode="bilinear", align_corners=False)
                
                img_grad_x = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
                img_grad_y = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
                
                # 🌟 断崖式边缘切割：遇到图像颜色骤变的地方，减弱平滑性要求，允许视差跳跃
                grad_x = grad_x * torch.exp(-self.edge_alpha * img_grad_x)
                grad_y = grad_y * torch.exp(-self.edge_alpha * img_grad_y)

            losses.append(grad_x.mean() + grad_y.mean())

        return torch.stack(losses).mean() * self.weight