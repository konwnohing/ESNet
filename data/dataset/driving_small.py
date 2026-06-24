# data/datasets/driving_small.py（修复版）
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional

# 导入PFM工具
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm_utils import read_pfm

class DrivingSmallDataset(Dataset):
    """用于快速过拟合的Driving Small数据集"""
    def __init__(self, root_dir: str, 
                 split: str = 'train',
                 transform: Optional[Any] = None,
                 max_disp: int = 192):
        self.root_dir = root_dir
        self.transform = transform
        self.max_disp = max_disp
        self.split = split
        
        # 构建样本列表
        self.samples = []
        left_dir = os.path.join(root_dir, 'left')
        
        for filename in sorted(os.listdir(left_dir)):
            if filename.endswith('.png'):
                base_name = os.path.splitext(filename)[0]
                left_path = os.path.join(left_dir, filename)
                right_path = os.path.join(root_dir, 'right', filename)
                disp_path = os.path.join(root_dir, 'disparity', f'{base_name}.pfm')
                
                if os.path.exists(right_path) and os.path.exists(disp_path):
                    self.samples.append({
                        'left': left_path,
                        'right': right_path,
                        'disparity': disp_path,
                        'id': base_name
                    })
        
        print(f"[DrivingSmall] 加载了 {len(self.samples)} 个样本")
        if len(self.samples) > 0:
            print(f"示例: {self.samples[0]['id']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载左右图像
        left_img = Image.open(sample['left']).convert('RGB')
        right_img = Image.open(sample['right']).convert('RGB')
        
        # 加载PFM视差图（已测试成功）
        disp = read_pfm(sample['disparity'])  # shape: [H, W], dtype: float32
        
        # 关键修复：确保数组是连续的
        if not disp.flags['C_CONTIGUOUS']:
            disp = np.ascontiguousarray(disp)
        
        # SceneFlow Driving数据集：视差可能需要除以缩放因子
        # 通常SceneFlow PFM存储的是实际视差值，但我们可以检查范围
        if disp.max() > 1000:  # 如果值过大，可能需要缩放
            # SceneFlow FlyingThings3D: 除以128
            disp = disp / 128.0
        
        # 限制视差范围
        disp = np.clip(disp, 0, self.max_disp)
        
        # 转换为PyTorch Tensor
        disp_tensor = torch.from_numpy(disp).float()
        
        # 如果是2D数组，添加通道维度
        if disp_tensor.dim() == 2:
            disp_tensor = disp_tensor.unsqueeze(0)  # [1, H, W]
        
        # 应用变换
        if self.transform:
            left_img, right_img, disp_tensor = self.transform(left_img, right_img, disp_tensor)
        
        return {
            'left': left_img,
            'right': right_img,
            'disparity': disp_tensor,
            'id': sample['id']
        }