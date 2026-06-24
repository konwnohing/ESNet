import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfm_utils import read_pfm  

class SceneFlowDataset(Dataset):
    """
    Scene Flow 合并版数据集 (适配 OpenDataLab 统一树状结构)
    """
    def __init__(self, root_dir: str, split: str = 'train',
                 transform: Optional[Any] = None,
                 max_disp: int = 192):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_disp = max_disp
        
        self.samples = self._build_sample_list()
        
    def _build_sample_list(self):
        samples = []
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"找不到数据根目录: {self.root_dir}")
            
        print(f"[INFO] 开始扫描 SceneFlow 统一目录... 模式: {self.split.upper()}")
        
        # 直接映射到官方帮你合并好的 TRAIN 或 TEST 文件夹
        official_split = 'TRAIN' if self.split == 'train' else 'TEST'
        search_dir = os.path.join(self.root_dir, 'frames_cleanpass', official_split)

        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"找不到分割目录: {search_dir}，请确认路径是否正确。")

        # 极速扫描所有的左图
        pattern = os.path.join(search_dir, '**', 'left', '*.png')
        all_left_imgs = glob.glob(pattern, recursive=True)
        all_left_imgs.sort() # 保证顺序一致
            
        if not all_left_imgs:
            raise RuntimeError(f"在 {search_dir} 中没有扫描到任何左图！")
            
        for left_path in all_left_imgs:
            # 1. 右图路径：/left/ 换成 /right/
            right_path = left_path.replace(f'{os.sep}left{os.sep}', f'{os.sep}right{os.sep}')
            
            # 2. 视差图路径：frames_cleanpass 换成 disparity，.png 换成 .pfm
            disp_path = left_path.replace('frames_cleanpass', 'disparity')
            disp_path = os.path.splitext(disp_path)[0] + '.pfm'
            
            if os.path.exists(right_path) and os.path.exists(disp_path):
                rel_path = os.path.relpath(left_path, self.root_dir)
                base_name = rel_path.replace(os.sep, '_') 
                
                samples.append({
                    'left': left_path,
                    'right': right_path,
                    'disparity': disp_path,
                    'base_name': base_name
                })
        
        print(f"✅ SceneFlow [{self.split.upper()}] 成功配对 {len(samples)} 个完美样本对.")
        return samples
    
    def _load_disparity(self, disp_path: str) -> np.ndarray:
        ext = os.path.splitext(disp_path)[1].lower()
        if ext == '.pfm':
            disp = read_pfm(disp_path)
            if isinstance(disp, tuple):
                disp = disp[0]
            # 替换非法的无穷大或NaN值
            disp = np.ascontiguousarray(disp, dtype=np.float32)
            disp[np.isinf(disp)] = 0
            disp[np.isnan(disp)] = 0
            return disp
        elif ext == '.png':
            disp = np.array(Image.open(disp_path)).astype(np.float32)
            if disp.max() > 1000:  
                disp = disp / 256.0
            return disp
        else:
            raise ValueError(f"Unsupported disparity format: {ext}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        left_img = Image.open(sample['left']).convert('RGB')
        right_img = Image.open(sample['right']).convert('RGB')
        disp = self._load_disparity(sample['disparity'])
        
        disp = np.clip(disp, 0, self.max_disp)
        disp_tensor = torch.from_numpy(disp).float()
        
        if disp_tensor.dim() == 2:
            disp_tensor = disp_tensor.unsqueeze(0) 
        
        if self.transform:
            left_img, right_img, disp_tensor = self.transform(left_img, right_img, disp_tensor)
        
        return {
            'left': left_img,
            'right': right_img,
            'disparity': disp_tensor,
            'left_path': sample['left'],
            'right_path': sample['right'],
            'base_name': sample['base_name']
        }