#fpn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class ASPPModule(nn.Module):
    """ASPP"""

    def __init__(self, in_channels: int, out_channels: int = 256, dilations: List[int] = [1, 6, 12, 18], dropout_p: float = 0.1):
        super().__init__()

        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            ))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

        total = out_channels * (len(dilations) + 1)
        drop = nn.Dropout2d(float(dropout_p)) if float(dropout_p) > 0 else nn.Identity()

        self.fusion = nn.Sequential(
            nn.Conv2d(total, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feats = [conv(x) for conv in self.convs]

        g = self.global_pool(x)
        g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=True)
        feats.append(g)

        out = torch.cat(feats, dim=1)
        return self.fusion(out)


class MultiOutputFPN(nn.Module):
    """输出 p1..p5 的 FPN"""

    def __init__(self, backbone_channels: List[int], fpn_channels: int = 128):
        super().__init__()
        num_levels = len(backbone_channels)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_channels, kernel_size=1)
            for in_ch in backbone_channels
        ])

        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        laterals = [lat(feat) for lat, feat in zip(self.lateral_convs, features)]

        pyramid = []
        for i in reversed(range(len(laterals))):
            if i == len(laterals) - 1:
                fused = self.fusion_convs[i](laterals[i])
            else:
                up = F.interpolate(pyramid[-1], size=laterals[i].shape[2:], mode='nearest')
                fused = self.fusion_convs[i](up + laterals[i])
            pyramid.append(fused)

        pyramid.reverse()  # [p1,p2,p3,p4,p5]
        return {
            'p1': pyramid[0],
            'p2': pyramid[1],
            'p3': pyramid[2],
            'p4': pyramid[3],
            'p5': pyramid[4],
        }
