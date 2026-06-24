#multileve_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from math import exp

class FixedMultiscaleLoss(nn.Module):
    """
    多尺度损失（对齐 EnhancedStereoNet 的 outputs / all_disparities 结构）
    🌟 终极安全进化版：
    1. 包含 Edge-Aware Smoothness Loss (alpha=15.0 极致边缘切割)
    2. 包含 Exponential Depth-Aware Loss (指数级逆深度加权)
    3. 包含 OHEM 困难样本挖掘 (专攻全班倒数 30% 的错题)
    4. 🌟 修复：梯度平摊的安全负视差惩罚
    """

    def __init__(
        self,
        max_disp: int = 192,
        weights: Optional[Dict[str, float]] = None,
        reduction: str = "mean",
        negative_penalty_weight: float = 0.1,
        smoothness_weight: float = 0.1,    
        use_ssim: bool = False,
        monitor_negative: bool = True,
        distribution_weight: float = 0.05,
        distribution_temperature: float = 0.1,
        spatial_grad_weight: float = 0.5,  
        edge_alpha: float = 15.0,           # 🌟 手术修改 1：从 4.0 提升到 15.0，大幅强化边缘切割能力！
        relative_loss_weight: float = 2.0, 
        ohem_fraction: float = 0.3,        
    ):
        super().__init__()
        self.max_disp = int(max_disp)

        self.weights = weights or {
            "final": 1.2,
            "full_res": 0.7,
            "p1": 0.5,
            "p2": 0.4,
            "p3": 0.5,
            "p4": 0.0,
            "p5": 0.0,
            "c5": 0.0,
            "disp_coarse": 0.0,
        }

        self.negative_penalty_weight = float(negative_penalty_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.spatial_grad_weight = float(spatial_grad_weight)
        self.use_ssim = bool(use_ssim)
        self.monitor_negative = bool(monitor_negative)

        self.distribution_weight = float(distribution_weight)
        self.distribution_temperature = float(distribution_temperature)
        self.edge_alpha = float(edge_alpha) 
        self.relative_loss_weight = float(relative_loss_weight) 
        self.ohem_fraction = float(ohem_fraction) 

        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.smooth_l1 = nn.SmoothL1Loss(reduction=reduction)
        self.reduction = reduction

        if self.use_ssim:
            self.ssim_loss = SSIMLoss()

    @staticmethod
    def _ensure_b1hw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError(f"Unsupported tensor dim: {x.dim()}")

    def resize_gt_and_mask(
        self,
        pred_disp: torch.Tensor,
        disp_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        pred_disp = self._ensure_b1hw(pred_disp)
        disp_gt = self._ensure_b1hw(disp_gt)
        mask = self._ensure_b1hw(mask)

        _, _, H_pred, W_pred = pred_disp.shape
        _, _, H_gt, W_gt = disp_gt.shape

        gt_resized = F.interpolate(disp_gt, size=(H_pred, W_pred), mode="nearest")
        mask_resized = F.interpolate(mask.float(), size=(H_pred, W_pred), mode="nearest").bool()

        scale = float(W_pred) / float(max(W_gt, 1))  
        gt_scaled = gt_resized * scale

        return gt_scaled, mask_resized, scale

    def apply_scale_maxdisp_mask(
        self,
        gt_scaled: torch.Tensor,
        mask_resized: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        max_disp_s = float(self.max_disp) * float(scale)
        return mask_resized & (gt_scaled > 0) & (gt_scaled < max_disp_s)

    @staticmethod
    def compute_negative_penalty(disparity: torch.Tensor) -> torch.Tensor:
        # 🌟 修复：使用 F.relu 平摊梯度，彻底解决极个别像素导致的梯度爆炸
        return F.relu(-disparity).mean()

    def compute_base_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """结合了指数深度加权与 OHEM（困难样本挖掘）的最强回归损失"""
        abs_error = torch.abs(pred - gt)
        
        # 1. 深度加权
        #depth_weight = torch.exp(-gt / 20.0) 
        #weighted_error = abs_error * (1.0 + self.relative_loss_weight * depth_weight)
        weighted_error = abs_error
        # 2. OHEM 核心逻辑
        valid_pixels = weighted_error.numel()
        if valid_pixels == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        num_hard = max(1, int(valid_pixels * self.ohem_fraction))
        
        # 🌟 优化：使用 reshape(-1) 替代 view(-1)，防止内存不连续报错
        hard_loss, _ = torch.topk(weighted_error.reshape(-1), k=num_hard)
        
        # 3. 只对这部分最难的像素求平均梯度
        return hard_loss.mean()

    def compute_smoothness_loss(self, disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if image is None or image.shape[-2:] != disparity.shape[-2:]:
            return torch.tensor(0.0, device=disparity.device)

        grad_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        grad_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)

        weight_x = torch.exp(-self.edge_alpha * grad_img_x)
        weight_y = torch.exp(-self.edge_alpha * grad_img_y)

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return (smoothness_x.mean() + smoothness_y.mean()) * 0.5

    @staticmethod
    def compute_spatial_gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]

        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        loss_dx = F.l1_loss(pred_dx[mask_dx], gt_dx[mask_dx]) if mask_dx.any() else torch.tensor(0.0, device=pred.device)
        loss_dy = F.l1_loss(pred_dy[mask_dy], gt_dy[mask_dy]) if mask_dy.any() else torch.tensor(0.0, device=pred.device)

        return loss_dx + loss_dy

    def compute_disparity_distribution_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        if attention_weights is None or attention_weights.dim() != 4:
            device = attention_weights.device if attention_weights is not None else "cpu"
            return torch.tensor(0.0, device=device)

        _, D, _, _ = attention_weights.shape
        avg_attn = attention_weights.mean(dim=(0, 2, 3)) 

        avg_attn_safe = torch.clamp(avg_attn, min=1e-4)
        log_prob = torch.log(avg_attn_safe)

        target = torch.ones_like(log_prob) / float(D)
        dist_loss = F.kl_div(log_prob, target, reduction="sum")
        return dist_loss

    @staticmethod
    def analyze_disparity_distribution(attention_weights: torch.Tensor, prefix: str = "attn_") -> Dict[str, float]:
        if attention_weights is None or attention_weights.dim() != 4:
            return {}

        _, D, _, _ = attention_weights.shape
        avg_attn = attention_weights.mean(dim=(0, 2, 3)).detach().cpu().numpy()

        boundary_usage = float(avg_attn[0] + avg_attn[-1])
        return {
            f"{prefix}boundary_usage": boundary_usage,
            f"{prefix}boundary_ratio": boundary_usage / max(D, 1),
            f"{prefix}min_attn": float(avg_attn.min()),
            f"{prefix}max_attn": float(avg_attn.max()),
            f"{prefix}range_width": float(avg_attn.max() - avg_attn.min()),
        }

    def forward(
        self,
        outputs: dict,
        disp_gt: torch.Tensor,
        left_image: Optional[torch.Tensor] = None,
    ) -> dict:
        losses: Dict[str, object] = {}
        total_terms = []

        disp_gt = self._ensure_b1hw(disp_gt)
        mask_full = (disp_gt > 0) & (disp_gt < float(self.max_disp))
        scales = outputs.get("all_disparities", {}) or {}

        # ========== 1) final 监督 ==========
        disp_final = scales.get("final", outputs.get("disp_final", None))
        if disp_final is not None and self.weights.get("final", 0) > 0:
            disp_final = self._ensure_b1hw(disp_final)

            gt_final, mask_final, scale_final = self.resize_gt_and_mask(disp_final, disp_gt, mask_full)
            mask_final = self.apply_scale_maxdisp_mask(gt_final, mask_final, scale_final)

            if mask_final.sum() > 0:
                base = self.compute_base_loss(disp_final[mask_final], gt_final[mask_final])
                neg_pen = self.compute_negative_penalty(disp_final)

                smt = torch.tensor(0.0, device=disp_final.device)
                if self.smoothness_weight > 0 and left_image is not None:
                    if left_image.shape[-2:] != disp_final.shape[-2:]:
                        img_rs = F.interpolate(left_image, size=disp_final.shape[-2:], mode="bilinear", align_corners=False)
                    else:
                        img_rs = left_image
                    smt = self.compute_smoothness_loss(disp_final, img_rs)

                grad_loss = torch.tensor(0.0, device=disp_final.device)
                if self.spatial_grad_weight > 0:
                    grad_loss = self.compute_spatial_gradient_loss(disp_final, gt_final, mask_final)

                loss_final = base + self.negative_penalty_weight * neg_pen + self.smoothness_weight * smt + self.spatial_grad_weight * grad_loss
                loss_final = loss_final * float(self.weights["final"])

                losses["final"] = loss_final
                total_terms.append(loss_final)

                losses["final_base"] = float(base.detach().item())
                losses["final_neg"] = float(neg_pen.detach().item())
                if self.spatial_grad_weight > 0:
                    losses["final_grad"] = float(grad_loss.detach().item()) 
                if self.monitor_negative:
                    losses["final_neg_ratio"] = float((disp_final < 0).float().mean().detach().item())
            else:
                losses["final"] = torch.tensor(0.0, device=disp_gt.device)

        # ========== 2) multi-scale 监督 ==========
        supervised_scales = ["c5", "p5", "p4", "p3", "p2", "p1", "full_res"]
        for s in supervised_scales:
            if s in scales and self.weights.get(s, 0) > 0:
                disp_s = self._ensure_b1hw(scales[s])

                gt_s, mask_s, scale_s = self.resize_gt_and_mask(disp_s, disp_gt, mask_full)
                mask_s = self.apply_scale_maxdisp_mask(gt_s, mask_s, scale_s)

                if mask_s.sum() > 0:
                    base = self.compute_base_loss(disp_s[mask_s], gt_s[mask_s])
                    neg_pen = self.compute_negative_penalty(disp_s)
                    loss_s = (base + self.negative_penalty_weight * neg_pen) * float(self.weights[s])

                    losses[s] = loss_s
                    total_terms.append(loss_s)

                    losses[f"{s}_base"] = float(base.detach().item())
                    losses[f"{s}_neg"] = float(neg_pen.detach().item())
                    if self.monitor_negative:
                        losses[f"{s}_neg_ratio"] = float((disp_s < 0).float().mean().detach().item())
                else:
                    losses[s] = torch.tensor(0.0, device=disp_gt.device)

        # ========== 3) coarse 监督 ==========
        if "disp_coarse" in outputs and self.weights.get("disp_coarse", 0) > 0:
            disp_c = self._ensure_b1hw(outputs["disp_coarse"])

            gt_c, mask_c, scale_c = self.resize_gt_and_mask(disp_c, disp_gt, mask_full)
            mask_c = self.apply_scale_maxdisp_mask(gt_c, mask_c, scale_c)

            if mask_c.sum() > 0:
                base = self.compute_base_loss(disp_c[mask_c], gt_c[mask_c])
                neg_pen = self.compute_negative_penalty(disp_c)
                loss_c = (base + self.negative_penalty_weight * neg_pen) * float(self.weights["disp_coarse"])

                losses["disp_coarse"] = loss_c
                total_terms.append(loss_c)

                losses["disp_coarse_base"] = float(base.detach().item())
                losses["disp_coarse_neg"] = float(neg_pen.detach().item())
                if self.monitor_negative:
                    losses["disp_coarse_neg_ratio"] = float((disp_c < 0).float().mean().detach().item())
            else:
                losses["disp_coarse"] = torch.tensor(0.0, device=disp_gt.device)

        # ========== 4) attention 分布正则（可选） ==========
        attn = outputs.get("attention_weights", None)
        if attn is not None and self.distribution_weight > 0:
            dist_loss = self.compute_disparity_distribution_loss(attn) * float(self.distribution_weight)
            losses["distribution"] = dist_loss
            total_terms.append(dist_loss)
            losses["distribution_value"] = float(dist_loss.detach().item())
            losses.update(self.analyze_disparity_distribution(attn, prefix="attn_"))
        else:
            losses["distribution"] = torch.tensor(0.0, device=disp_gt.device)
            losses["distribution_value"] = 0.0

        # ========== 5) total ==========
        if len(total_terms) > 0:
            losses["total"] = torch.stack(total_terms).sum()
        else:
            losses["total"] = torch.tensor(0.0, device=disp_gt.device)

        return losses


class SSIMLoss(nn.Module):
    """SSIM loss（可选）"""
    def __init__(self, window_size=11, sigma=1.5, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        # 🌟 优化：纯 PyTorch 操作，避免 Python list 初始化带来的性能消耗和 UserWarning
        x = torch.arange(window_size).float()
        gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1d = self.gaussian(window_size, self.sigma).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            return 1 - ssim_map.mean()
        return 1 - ssim_map.mean(1).mean(1).mean(1)