import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from .datasets.eth3d import ETH3DDataset
from .datasets.sceneflow import SceneFlowDataset
from .datasets.kitti import KITTIDataset
from .datasets.driving_small import DrivingSmallDataset
from .transforms import StereoTransform

def create_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    data_config = config.get('data', {})
    dataset_name = data_config.get('dataset', 'sceneflow')
    root_dir = data_config.get('root_dir', '')

    if root_dir == '':
        print("[WARN] data.root_dir 为空，请确认你的 config.yaml 中已正确设置数据路径。")

    # ====== 模式开关 ======
    debug_mode = bool(config.get("debug", {}).get("enable_debug_mode", False))
    overfit_mode = bool(config.get("training", {}).get("overfit", False))

    # ====== 数据增强 (仅在 train 且非 overfit 模式下开启) ======
    aug_cfg = data_config.get('augmentation', {})
    augment = (split == 'train') and aug_cfg.get('enabled', False) and (not overfit_mode)

    transform = StereoTransform(
        crop_height=data_config.get('crop_height', 256),
        crop_width=data_config.get('crop_width', 512),
        mean=data_config.get('mean', [0.485, 0.456, 0.406]),
        std=data_config.get('std', [0.229, 0.224, 0.225]),
        augment=augment
    )

    dataset_name_lower = dataset_name.lower()
    max_disp = config.get('model', {}).get('max_disp', 192)

    print(f"\n[INFO] 正在构建 {split.upper()} Dataloader...")
    print(f"[INFO] 目标数据集: {dataset_name_lower} | 路径: {root_dir}")

    # ====== 实例化 Dataset ======
    if dataset_name_lower == 'sceneflow':
        dataset = SceneFlowDataset(root_dir=root_dir, split=split, transform=transform, max_disp=max_disp)
    elif dataset_name_lower == 'kitti':
        dataset = KITTIDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            max_disp=max_disp,
            year=data_config.get('kitti_year', 2015)
        )
    elif dataset_name_lower == 'driving_small':
        dataset = DrivingSmallDataset(root_dir=root_dir, split=split, transform=transform, max_disp=max_disp)
    elif dataset_name_lower == 'eth3d':
        dataset = ETH3DDataset(root_dir=root_dir, split=split, transform=transform, max_disp=max_disp)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    # ====== 参数配置 ======
    training_cfg = config.get('training', {})
    batch_size = training_cfg.get('batch_size', 4)

    # Dataloader 参数 (兼容 debug 和 overfit)
    num_workers = training_cfg.get('num_workers', 4)
    if debug_mode or overfit_mode:
        num_workers = 0

    shuffle = (split == 'train')
    if overfit_mode:
        shuffle = False

    drop_last = (split == 'train')
    if overfit_mode:
        drop_last = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    
    print(f"[INFO] {split.upper()} Dataloader 构建完成，样本总数: {len(dataset)}，Batch Size: {batch_size}\n")
    return dataloader