import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

# 导入PFM工具
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from pfm_utils import read_pfm
except ImportError:
    # 兼容性处理
    def read_pfm(path): return None

class KITTIDataset(Dataset):
    """KITTI立体匹配数据集 (兼容 2012 和 2015)"""
    def __init__(self, root_dir: str, split: str = 'train',
                 transform: Optional[Any] = None,
                 max_disp: int = 192,
                 year: int = 2015):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_disp = max_disp
        self.year = year
        
        self.samples = self._build_sample_list()
        
    def _build_sample_list(self):
        samples = []
        
        # 路径映射：训练/验证都去 training 文件夹
        if self.split in ['train', 'val', 'train_all']:
            actual_folder = 'training'
        elif self.split in ['test', 'testing']:
            actual_folder = 'testing'
        else:
            actual_folder = self.split
        
        # 🌟 根据年份确定子文件夹名称
        if self.year == 2015:
            image_dir = os.path.join(self.root_dir, actual_folder, 'image_2')
            disp_dir = os.path.join(self.root_dir, actual_folder, 'disp_occ_0')
            obj_dir = os.path.join(self.root_dir, actual_folder, 'obj_map') # 🌟 新增: obj_map 路径
        elif self.year == 2012:
            image_dir = os.path.join(self.root_dir, 'kitti2012', actual_folder, 'colored_0')
            disp_dir = os.path.join(self.root_dir, 'kitti2012', actual_folder, 'disp_occ')
            obj_dir = '' # 2012 没有官方细分的 obj_map
        else:
            raise ValueError(f"Unsupported KITTI year: {self.year}")
        
        if not os.path.exists(image_dir):
            print(f"Warning: Image directory {image_dir} does not exist")
            return samples
        
        # 遍历图像
        all_samples = []
        img_list = sorted(os.listdir(image_dir))
        
        for img_name in img_list:
            if '_10.png' in img_name:
                base_name = img_name.replace('_10.png', '')
                left_path = os.path.join(image_dir, img_name)
                
                # 左右图路径对齐
                if self.year == 2015:
                    right_path = left_path.replace('image_2', 'image_3')
                else:
                    right_path = left_path.replace('colored_0', 'colored_1')
                
                # 查找视差图
                disp_paths = [
                    os.path.join(disp_dir, f'{base_name}_10.pfm'),
                    os.path.join(disp_dir, f'{base_name}_10.png'),
                    os.path.join(disp_dir, img_name)
                ]
                disp_path = next((dp for dp in disp_paths if os.path.exists(dp)), '')

                # 🌟 查找前景掩码图 (仅 2015)
                obj_map_path = ''
                if self.year == 2015 and obj_dir:
                    potential_obj = os.path.join(obj_dir, img_name)
                    if os.path.exists(potential_obj):
                        obj_map_path = potential_obj
                
                if os.path.exists(right_path):
                    all_samples.append({
                        'left': left_path,
                        'right': right_path,
                        'disparity': disp_path,
                        'obj_map': obj_map_path,  # 🌟 将掩码路径加入 sample
                        'base_name': base_name
                    })
        
        num_samples = len(all_samples)
        train_idx = int(num_samples*0.8)

        if self.split == 'train':
            samples = all_samples[:train_idx]
            print(f"[KITTI {self.year}] 划分模式: 训练集 (取前 {len(samples)} 张)")
        elif self.split == 'val':
            samples = all_samples[train_idx:]
            print(f"[KITTI {self.year}] 划分模式: 验证集 (取后 {len(samples)} 张)")
        else:
            samples = all_samples
            print(f"[KITTI {self.year}] 划分模式: 全量数据集 ({self.split}, 共 {len(samples)} 张)")
        
        return samples
    
    def _load_disparity(self, disp_path: str) -> Optional[np.ndarray]:
        if not disp_path or not os.path.exists(disp_path):
            return None
            
        ext = os.path.splitext(disp_path)[1].lower()
        if ext == '.pfm':
            disp = read_pfm(disp_path)
            return disp.astype(np.float32) if disp is not None else None
        elif ext == '.png':
            disp = np.array(Image.open(disp_path)).astype(np.float32)
            return disp / 256.0
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        left_img = Image.open(sample['left']).convert('RGB')
        right_img = Image.open(sample['right']).convert('RGB')
        
        # 1. 加载视差
        disp_data = self._load_disparity(sample['disparity'])
        if disp_data is None:
            w, h = left_img.size
            disp_data = np.zeros((h, w), dtype=np.float32)
        disp_data = np.clip(disp_data, 0, self.max_disp)
        disp_tensor = torch.from_numpy(disp_data).float().unsqueeze(0) # [1, H, W]

        # 🌟 2. 加载物体前景掩码 (obj_map)
        obj_map_path = sample.get('obj_map', '')
        if obj_map_path and os.path.exists(obj_map_path):
            obj_arr = np.array(Image.open(obj_map_path))
            # KITTI 中 >0 的像素为前景(动态车辆等)，=0 为背景
            obj_mask = (obj_arr > 0).astype(np.float32)
            obj_tensor = torch.from_numpy(obj_mask).unsqueeze(0) # [1, H, W]
        else:
            obj_tensor = None
        
        # 3. 经过数据增强 / Pad
        if self.transform:
            left_img, right_img, disp_tensor = self.transform(left_img, right_img, disp_tensor)
            
            # 🌟 万无一失的对齐机制：如果 transform 改变了图像尺寸 (如 pad)，同步缩放 obj_mask
            if obj_tensor is not None and obj_tensor.shape != disp_tensor.shape:
                # 使用 nearest 保证掩码依然是纯净的 0 和 1
                obj_tensor = F.interpolate(
                    obj_tensor.unsqueeze(0), 
                    size=disp_tensor.shape[-2:], 
                    mode='nearest'
                ).squeeze(0)
        
        # 4. 组装返回字典
        res = {
            'left': left_img, 
            'right': right_img, 
            'disparity': disp_tensor,
            'left_path': sample['left'], 
            'base_name': sample['base_name']
        }
        
        # 把对齐后的掩码加进去供评估脚本调用
        if obj_tensor is not None:
            res['obj_mask'] = obj_tensor
            
        return res