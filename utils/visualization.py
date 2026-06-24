import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import cv2

def visualize_disparity(disparity: torch.Tensor, 
                       max_disp: Optional[float] = None,
                       cmap: str = 'viridis') -> np.ndarray:
    """
    可视化视差图
    Args:
        disparity: 视差图 [H, W] 或 [1, H, W]
        max_disp: 最大视差值（用于归一化）
        cmap: 颜色映射
    Returns:
        彩色视差图 [H, W, 3]
    """
    if isinstance(disparity, torch.Tensor):
        disparity = disparity.detach().cpu().numpy()
    
    # 处理维度
    if disparity.ndim == 3:
        disparity = disparity[0]  # 取第一个通道
    
    # 归一化
    if max_disp is None:
        max_disp = np.max(disparity)
    
    disparity_norm = np.clip(disparity / max_disp, 0, 1)
    
    # 应用颜色映射
    cmap_obj = plt.get_cmap(cmap)
    colored_disp = cmap_obj(disparity_norm)[:, :, :3]
    colored_disp = (colored_disp * 255).astype(np.uint8)
    
    return colored_disp

def visualize_confidence(confidence: torch.Tensor,
                        cmap: str = 'RdYlGn') -> np.ndarray:
    """
    可视化置信度图
    Args:
        confidence: 置信度图 [H, W] 或 [1, H, W]
        cmap: 颜色映射
    Returns:
        彩色置信度图 [H, W, 3]
    """
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy()
    
    # 处理维度
    if confidence.ndim == 3:
        confidence = confidence[0]  # 取第一个通道
    
    # 应用颜色映射
    cmap_obj = plt.get_cmap(cmap)
    colored_conf = cmap_obj(confidence)[:, :, :3]
    colored_conf = (colored_conf * 255).astype(np.uint8)
    
    return colored_conf

def overlay_disparity_on_image(image: np.ndarray,
                              disparity: np.ndarray,
                              alpha: float = 0.6) -> np.ndarray:
    """
    将视差图叠加在原图上
    Args:
        image: 原图 [H, W, 3]
        disparity: 彩色视差图 [H, W, 3]
        alpha: 叠加透明度
    Returns:
        叠加后的图像
    """
    # 确保图像在0-255范围内
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, disparity, alpha, 0)
    
    return overlay