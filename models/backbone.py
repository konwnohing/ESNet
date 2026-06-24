#backbone
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict

from .fpn import MultiOutputFPN, ASPPModule


class BackboneWithFPN(nn.Module):
    """Backbone: ResNet + ASPP + FPN, 输出统一通道=fpn_channels(默认128)"""

    def __init__(self, backbone_type: str = 'resnet18', fpn_channels: int = 128, aspp_dropout_p: float = 0.1):
        super().__init__()

        if backbone_type == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone_channels = [64, 64, 128, 256]  # c1,c2,c3,c4
            self.c5_channels = 512
        elif backbone_type == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone_channels = [64, 256, 512, 1024]
            self.c5_channels = 2048
        else:
            raise ValueError(f"不支持的backbone类型: {backbone_type}")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.aspp = ASPPModule(in_channels=self.c5_channels, out_channels=fpn_channels, dropout_p=aspp_dropout_p)

        self.fpn = MultiOutputFPN(
            backbone_channels=self.backbone_channels + [fpn_channels],
            fpn_channels=fpn_channels
        )

        self.full_res_recon = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=fpn_channels),
            nn.ReLU(inplace=False)
        )

        self.p1_enhance = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=fpn_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=fpn_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape

        x = self.relu(self.bn1(self.conv1(x)))  # [B,64,H/2,W/2]
        c1 = x

        x = self.maxpool(x)                     # [B,64,H/4,W/4]
        c2 = self.layer1(x)                     # [B,*,H/4,W/4]
        c3 = self.layer2(c2)                    # [B,*,H/8,W/8]
        c4 = self.layer3(c3)                    # [B,*,H/16,W/16]
        c5 = self.layer4(c4)                    # [B,*,H/32,W/32]

        c5_aspp = self.aspp(c5)                 # [B,128,H/32,W/32]

        fpn_out = self.fpn([c1, c2, c3, c4, c5_aspp])

        p1_enhanced = self.p1_enhance(fpn_out['p1'])      # [B,128,H/2,W/2]
        full_res = self.full_res_recon(p1_enhanced)       # [B,128,H,W]

        return {
            'p1': p1_enhanced,
            'p2': fpn_out['p2'],
            'p3': fpn_out['p3'],
            'p4': fpn_out['p4'],
            'p5': fpn_out['p5'],
            'full_res': full_res,
            'c5_aspp': c5_aspp,
        }
