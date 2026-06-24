import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

TensorLikeDisp = Union[torch.Tensor, np.ndarray]

class StereoTransform:
    """
    立体图像对变换（抗过拟合猛药版）
    - train: random crop + 强力非对称 color aug + 垂直翻转
    - val/test: center crop
    - 移除危险的水平翻转（stereo_horizontal_flip 极易导致负视差 bug）
    """

    def __init__(
        self,
        crop_height: int = 256,
        crop_width: int = 512,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
        color_aug: bool = True,
        geometric_aug: bool = False, 
        hflip_prob: float = 0.0,     # 强制设为 0，防止破坏极线约束
        vflip_prob: float = 0.5,     # 🌟 猛药1：垂直翻转（对立体匹配完全安全）
        asymmetric_color_prob: float = 0.4, # 🌟 猛药2：非对称颜色抖动概率
        # color aug ranges
        brightness: Tuple[float, float] = (0.8, 1.2),
        contrast: Tuple[float, float] = (0.8, 1.2),
        saturation: Tuple[float, float] = (0.8, 1.2),
        # padding value
        pad_value: int = 0,
        disp_pad_value: float = 0.0,
    ):
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)

        self.mean = mean
        self.std = std

        self.augment = augment
        self.color_aug_enabled = bool(color_aug)
        self.geometric_aug_enabled = bool(geometric_aug)
        
        self.hflip_prob = 0.0
        self.vflip_prob = float(vflip_prob)
        self.asymmetric_color_prob = float(asymmetric_color_prob)

        self.brightness_range = brightness
        self.contrast_range = contrast
        self.saturation_range = saturation

        self.pad_value = int(pad_value)
        self.disp_pad_value = float(disp_pad_value)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(
        self,
        left_img: Image.Image,
        right_img: Image.Image,
        disp: Optional[TensorLikeDisp] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # 1) 确保输入是RGB
        left_img = self._ensure_rgb(left_img)
        right_img = self._ensure_rgb(right_img)

        # 2) 视差统一成 torch.Tensor [1,H,W] 或 None
        disp_t = self._to_disp_tensor(disp)

        # 3) 如果尺寸小于crop，先pad（图像与视差一致）
        left_img, right_img, disp_t = self._pad_to_min_size(left_img, right_img, disp_t)

        # 4) crop（train随机 / eval中心）
        if self.augment:
            left_img, right_img, disp_t = self.random_crop(left_img, right_img, disp_t)
        else:
            left_img, right_img, disp_t = self.center_crop(left_img, right_img, disp_t)

        # 5) augmentation
        if self.augment:
            if self.color_aug_enabled:
                left_img, right_img = self.color_augmentation(left_img, right_img)
            
            # 🌟 猛药1执行：垂直翻转 (Vertical Flip)
            # 这不会改变左右目的相对关系，只会让画面倒过来，极其增加几何泛化性
            if random.random() < self.vflip_prob:
                left_img = TF.vflip(left_img)
                right_img = TF.vflip(right_img)
                if disp_t is not None:
                    disp_t = TF.vflip(disp_t)

        # 6) to tensor + normalize
        left_tensor = self.normalize(self.to_tensor(left_img))
        right_tensor = self.normalize(self.to_tensor(right_img))

        return left_tensor, right_tensor, disp_t

    # ----------------------------
    # Basic helpers
    # ----------------------------
    def _ensure_rgb(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _to_disp_tensor(self, disp: Optional[TensorLikeDisp]) -> Optional[torch.Tensor]:
        if disp is None:
            return None

        if isinstance(disp, np.ndarray):
            disp = torch.from_numpy(disp)

        if not isinstance(disp, torch.Tensor):
            raise TypeError(f"Unsupported disp type: {type(disp)}")

        if disp.dim() == 2:
            disp = disp.unsqueeze(0)
        elif disp.dim() == 3:
            pass
        elif disp.dim() == 4:
            disp = disp[0]
        else:
            raise ValueError(f"Unsupported disp shape: {tuple(disp.shape)}")

        if disp.dtype not in (torch.float16, torch.float32, torch.float64):
            disp = disp.float()
        else:
            disp = disp.to(torch.float32)

        return disp

    def _pad_to_min_size(
        self,
        left_img: Image.Image,
        right_img: Image.Image,
        disp: Optional[torch.Tensor],
    ) -> Tuple[Image.Image, Image.Image, Optional[torch.Tensor]]:
        w, h = left_img.size
        pad_w = max(0, self.crop_width - w)
        pad_h = max(0, self.crop_height - h)
        if pad_w == 0 and pad_h == 0:
            return left_img, right_img, disp

        padding = (0, 0, pad_w, pad_h)  # (left, top, right, bottom)
        left_img = TF.pad(left_img, padding, fill=self.pad_value)
        right_img = TF.pad(right_img, padding, fill=self.pad_value)

        if disp is not None and disp.numel() > 0:
            disp = TF.pad(
                disp,
                padding,
                fill=self.disp_pad_value,
            )

        return left_img, right_img, disp

    # ----------------------------
    # Crop
    # ----------------------------
    def random_crop(
        self,
        left_img: Image.Image,
        right_img: Image.Image,
        disp: Optional[torch.Tensor] = None,
    ) -> Tuple[Image.Image, Image.Image, Optional[torch.Tensor]]:
        w, h = left_img.size
        if w < self.crop_width or h < self.crop_height:
            left_img, right_img, disp = self._pad_to_min_size(left_img, right_img, disp)
            w, h = left_img.size

        x = random.randint(0, w - self.crop_width)
        y = random.randint(0, h - self.crop_height)

        left_crop = left_img.crop((x, y, x + self.crop_width, y + self.crop_height))
        right_crop = right_img.crop((x, y, x + self.crop_width, y + self.crop_height))

        if disp is not None and disp.numel() > 0:
            disp_crop = disp[:, y : y + self.crop_height, x : x + self.crop_width]
        else:
            disp_crop = disp

        return left_crop, right_crop, disp_crop

    def center_crop(
        self,
        left_img: Image.Image,
        right_img: Image.Image,
        disp: Optional[torch.Tensor] = None,
    ) -> Tuple[Image.Image, Image.Image, Optional[torch.Tensor]]:
        w, h = left_img.size
        if w < self.crop_width or h < self.crop_height:
            left_img, right_img, disp = self._pad_to_min_size(left_img, right_img, disp)
            w, h = left_img.size

        x = (w - self.crop_width) // 2
        y = (h - self.crop_height) // 2

        left_crop = left_img.crop((x, y, x + self.crop_width, y + self.crop_height))
        right_crop = right_img.crop((x, y, x + self.crop_width, y + self.crop_height))

        if disp is not None and disp.numel() > 0:
            disp_crop = disp[:, y : y + self.crop_height, x : x + self.crop_width]
        else:
            disp_crop = disp

        return left_crop, right_crop, disp_crop

    # ----------------------------
    # Augmentations
    # ----------------------------
    def color_augmentation(
        self,
        left_img: Image.Image,
        right_img: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        
        # 1. 先应用对称的基础抖动（保证两张图的整体环境光一致）
        b = random.uniform(*self.brightness_range)
        c = random.uniform(*self.contrast_range)
        s = random.uniform(*self.saturation_range)

        left_img = TF.adjust_brightness(left_img, b)
        right_img = TF.adjust_brightness(right_img, b)

        left_img = TF.adjust_contrast(left_img, c)
        right_img = TF.adjust_contrast(right_img, c)

        left_img = TF.adjust_saturation(left_img, s)
        right_img = TF.adjust_saturation(right_img, s)

        # 🌟 猛药2执行：非对称颜色抖动 (Asymmetric Color Jitter)
        # 模拟现实中两个摄像头光圈/白平衡不一致的情况，逼迫网络学习几何特征而不是死记颜色
        if random.random() < self.asymmetric_color_prob:
            # 给右图额外加一点点随机偏移
            b_shift = random.uniform(-0.05, 0.05)
            c_shift = random.uniform(-0.05, 0.05)
            s_shift = random.uniform(-0.05, 0.05)
            
            right_img = TF.adjust_brightness(right_img, max(0, 1.0 + b_shift))
            right_img = TF.adjust_contrast(right_img, max(0, 1.0 + c_shift))
            right_img = TF.adjust_saturation(right_img, max(0, 1.0 + s_shift))

        return left_img, right_img