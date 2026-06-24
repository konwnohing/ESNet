import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# 🌟 重命名：将 Transformer 语义改为循环相关匹配 (RCM)
from .backbone import BackboneWithFPN
from .rcm_head import RecurrentCorrelationMatcher # 原 TransformerCostVolumeMatcher
from .progressive_refinement import ProgressiveRefinementNetwork

class GateRefinementLayer(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_dim: int = 32, max_res: float = 3.0):
        super().__init__()
        self.max_res = max_res
        cat_channels = in_channels + 1 + 3

        self.shared_encoder = nn.Sequential(
            nn.Conv2d(cat_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid() 
        )

        self.res_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, disp_init: torch.Tensor, left_feat: torch.Tensor, left_img: torch.Tensor, edge_mode: bool = False):
        if edge_mode:
            return disp_init, torch.ones_like(disp_init)

        H, W = disp_init.shape[2:]
        if left_feat.shape[2:] != (H, W):
            left_feat = F.interpolate(left_feat, size=(H, W), mode='bilinear', align_corners=True)
        if left_img.shape[2:] != (H, W):
            left_img = F.interpolate(left_img, size=(H, W), mode='bilinear', align_corners=True)

        concat_feat = torch.cat([left_feat, disp_init, left_img], dim=1)
        shared_feat = self.shared_encoder(concat_feat)

        conf = self.conf_head(shared_feat)
        base_res = self.res_head(shared_feat) * self.max_res

        # 🌟 核心创新：门控残差修正
        disp_final = disp_init + (1.0 - conf) * base_res
        disp_final = F.relu(disp_final) 

        return disp_final, conf


class EnhancedStereoNet(nn.Module):
    def __init__(self, backbone_type: str = 'resnet18', max_disp: int = 192, 
                 use_prn: bool = True, use_gate: bool = True, debug: bool = False):
        super().__init__()
        self.max_disp = max_disp
        self.debug = debug
        self.edge_mode = False 
        
        self.use_prn = use_prn
        self.use_gate = use_gate

        # 1. 特征提取
        self.backbone = BackboneWithFPN(backbone_type=backbone_type)

        # 2. 循环相关匹配头 (RCM) - 原 Transformer 模块
        self.rcm_matcher = RecurrentCorrelationMatcher(
            feature_channels=128, max_disp=max_disp, debug=debug
        )

        # 3. 渐进式精修 (PRN)
        if self.use_prn:
            self.progressive_refiner = ProgressiveRefinementNetwork(
                max_disp=max_disp, fpn_channels=128, debug=debug
            )

        # 4. 门控精修层 (Gate)
        if self.use_gate:
            self.gate_refiner = GateRefinementLayer(
                in_channels=128, hidden_dim=32, max_res=4.0
            )

    def forward(self, left_img, right_img) -> Dict[str, torch.Tensor]:
        left_feats = self.backbone(left_img)
        right_feats = self.backbone(right_img)

        # Stage 2: RCM @ 1/8
        matcher_out = self.rcm_matcher(left_feats['p3'], right_feats['p3'])
        disp_p3 = matcher_out['disp_lowres']
        disp_coarse = matcher_out.get('disp_coarse', disp_p3)

        # Stage 3: Cascade / PRN
        if self.use_prn:
            disp_init, inter_disps = self.progressive_refiner(
                disp_p3, left_feats, right_feats, start_level='p3'
            )
        else:
            # 如果不消融 PRN，则简单双线性上采样对齐尺寸
            disp_init = F.interpolate(disp_p3, size=left_img.shape[-2:], mode='bilinear', align_corners=True)
            disp_init = disp_init * (left_img.shape[-1] / disp_p3.shape[-1])
            inter_disps = {}

        # Stage 4: Gate Refinement
        if self.use_gate:
            disp_final, confidence = self.gate_refiner(
                disp_init=disp_init, 
                left_feat=left_feats['full_res'], 
                left_img=left_img,
                edge_mode=self.edge_mode
            )
        else:
            disp_final = disp_init
            confidence = torch.ones_like(disp_init)

        outputs = {
            'disp_final': disp_final,
            'confidence': confidence,
            'disp_init': disp_init,
            'all_disparities': {
                'disp_coarse': disp_coarse,
                'p3': disp_p3,
                **inter_disps,
                'final': disp_final
            }
        }
        return outputs