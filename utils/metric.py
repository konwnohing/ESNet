import torch
from typing import Optional, Union


def _is_number(x) -> bool:
    return isinstance(x, (int, float))


def compute_epe(
    pred_disp: torch.Tensor,
    gt_disp: torch.Tensor,
    mask_or_maxdisp: Optional[Union[torch.Tensor, int, float]] = None,
    max_disp: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """
    兼容两种调用：
      1) compute_epe(pred, gt, mask=valid_mask)
      2) compute_epe(pred, gt, max_disp)   # 老代码常见写法
      3) compute_epe(pred, gt, mask_or_maxdisp, max_disp=xxx)
    """
    # ---- 兼容：第三参是 max_disp 的老写法 ----
    if _is_number(mask_or_maxdisp) and max_disp is None:
        max_disp = float(mask_or_maxdisp)
        mask = None
    else:
        mask = mask_or_maxdisp

    # ---- finite ----
    finite = torch.isfinite(pred_disp) & torch.isfinite(gt_disp)

    # ---- base valid from max_disp ----
    if max_disp is not None:
        valid = (gt_disp > 0) & (gt_disp < float(max_disp)) & finite
    else:
        valid = finite

    # ---- combine external mask ----
    if mask is not None:
        valid = valid & mask.bool()

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_disp.device)

    epe_map = (pred_disp - gt_disp).abs()
    return epe_map[valid].mean()


def compute_bad_pixels(
    pred_disp: torch.Tensor,
    gt_disp: torch.Tensor,
    threshold: float = 3.0,
    mask_or_maxdisp: Optional[Union[torch.Tensor, int, float]] = None,
    max_disp: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """
    返回百分比 0~100
    🌟 已升级为 KITTI / SceneFlow 国际通用 D1-all 标准
    (必须同时满足：绝对误差 > 3.0 且 相对误差 > 5%)
    
    同样兼容老写法：
      compute_bad_pixels(pred, gt, thr, max_disp)
    """
    if _is_number(mask_or_maxdisp) and max_disp is None:
        max_disp = float(mask_or_maxdisp)
        mask = None
    else:
        mask = mask_or_maxdisp

    finite = torch.isfinite(pred_disp) & torch.isfinite(gt_disp)

    if max_disp is not None:
        valid = (gt_disp > 0) & (gt_disp < float(max_disp)) & finite
    else:
        valid = finite

    if mask is not None:
        valid = valid & mask.bool()

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_disp.device)

    err = (pred_disp - gt_disp).abs()
    
    # 🌟 国际标准 D1-all 核心双约束
    bad_abs = err > float(threshold)
    # 加上 1e-7 防止极其微小的 0 导致除以 0 的计算崩溃
    bad_rel = err / (gt_disp.abs() + 1e-7) > 0.05 
    
    # 必须同时满足 绝对误差>阈值 AND 相对误差>5% 才被判定为真正的坏点
    bad = bad_abs & bad_rel & valid
    
    return bad.sum().float() / valid.sum().float() * 100.0