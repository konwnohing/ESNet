# utils/warp.py

import torch
import torch.nn.functional as F


def spatial_warp(
    x: torch.Tensor,
    disp: torch.Tensor,
    clamp_disp: bool = True,
    padding_mode: str = 'zeros',
    align_corners: bool = True
) -> torch.Tensor:
   
    B, C, H, W = x.shape
    device = x.device

    # -------------------------------
    # Step 1: 将 disp 上采样到与 x 分辨率一致
    # -------------------------------
    if disp.shape[2:] != (H, W):
        disp = F.interpolate(
            disp,
            size=(H, W),
            mode='bilinear',
            align_corners=align_corners  # 必须开启
        )

    # -------------------------------
    # Step 2: 强制非负视差（防止反向偏移）
    # -------------------------------
    if clamp_disp:
        disp = torch.clamp(disp, min=0.0)

    # -------------------------------
    # Step 3: 创建网格坐标 [-1, 1]
    # -------------------------------
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)

    # 扩展为 [B, H, W, 1]
    y_grid = y_coords.view(1, -1,1, 1).expand(B, H, W, 1)
    x_grid = x_coords.view(1, 1, -1,1).expand(B, H, W, 1)

    # 应用视差偏移：x ← x - disp
    x_warped = x_grid - disp.permute(0, 2, 3, 1)  # [B,H,W,1]

    # 归一化到 [-1, 1]
    x_warped_norm = 2.0 * x_warped / (W - 1) - 1.0
    y_norm = 2.0 * y_grid / (H - 1) - 1.0

    # 构造最终 grid [B, H, W, 2]
    grid = torch.cat([x_warped_norm, y_norm], dim=-1)

    # -------------------------------
    # Step 4: 使用 grid_sample 进行双线性采样
    # -------------------------------
    try:
        output = F.grid_sample(
            input=x,
            grid=grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=align_corners
        )
    except Exception as e:
        print(f"[Error in spatial_warp] Failed to sample: {e}")
        raise

    return output


# -------------------------------
# ✅ 辅助函数：批量 warp 多个尺度（可选）
# -------------------------------

def spatial_warp_multiscale(
    features_list: list[torch.Tensor],
    disps_list: list[torch.Tensor],
    **kwargs
) -> list[torch.Tensor]:
    """
    对多尺度特征列表执行 warp
    例如：p2/p3/p4 特征 + 对应 disp_p2/disp_p3/disp_p4

    Args:
        features_list: [feat_p2, feat_p3, ...]
        disps_list: [disp_p2, disp_p3, ...]
        **kwargs: 传递给 spatial_warp 的参数

    Returns:
        warped_features: warp 后的特征列表
    """
    return [
        spatial_warp(feat, disp, **kwargs)
        for feat, disp in zip(features_list, disps_list)
    ]


# -------------------------------
# ✅ 调试工具：检查 disp 是否合理
# -------------------------------

def validate_disparity(disp: torch.Tensor, name: str = "disparity"):
    """
    调试用：打印并验证视差图是否正常
    """
    if not isinstance(disp, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")

    if torch.isnan(disp).any():
        print(f"❌ {name} contains NaN!")
        return False
    if torch.isinf(disp).any():
        print(f"❌ {name} contains Inf!")
        return False

    print(f"[Debug] {name}: mean={disp.mean().item():.4f}, "
          f"min={disp.min().item():.4f}, max={disp.max().item():.4f}")

    return True
