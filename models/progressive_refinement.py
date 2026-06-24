#progressive_refinement
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ConvexUpsample2x(nn.Module):
    """
    🌟 RAFT 风格的高频特征引导凸上采样 (Convex Upsampling)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 36, kernel_size=1)
        )
        nn.init.zeros_(self.mask_conv[-1].weight)
        nn.init.zeros_(self.mask_conv[-1].bias)
        self.mask_conv[-1].bias.data[16:20] = 2.0

    def forward(self, coarse_disp: torch.Tensor, coarse_feat: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        B, _, H, W = coarse_disp.shape
        Ht, Wt = target_size

        mask = self.mask_conv(coarse_feat) 
        mask = mask.view(B, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2) 

        disp_unfold = F.unfold(coarse_disp, kernel_size=3, padding=1) 
        disp_unfold = disp_unfold.view(B, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * disp_unfold, dim=2) 

        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3).contiguous() 
        up_disp = up_disp.view(B, 1, H * 2, W * 2)

        if up_disp.shape[2:] != (Ht, Wt):
            up_disp = up_disp[:, :, :Ht, :Wt]

        scale_factor = float(Wt) / float(W)
        return up_disp * scale_factor


class RefinementStage(nn.Module):
    def __init__(self, in_channels: int, level_name: str, max_disp: int, debug: bool = False):
        super().__init__()
        self.level_name = level_name
        self.max_disp = max_disp
        self.debug = debug

        concat_channels = in_channels * 2 + 1
        self.res_net = nn.Sequential(
            nn.Conv2d(concat_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.res_net[-1].weight)
        nn.init.zeros_(self.res_net[-1].bias)

    def _warp(self, x: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        xx = torch.arange(0, W, device=x.device, dtype=x.dtype).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        
        grid_x = xx - disp
        grid_y = yy
        
        grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
        grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
        
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1) 
        # 🌟 修复点 3：padding_mode='border' 避免左侧边缘出现断层黑洞信号
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    def forward(self, disp: torch.Tensor, left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        warped_right = self._warp(right_feat, disp)
        concat_feat = torch.cat([left_feat, warped_right, disp], dim=1)
        residual = self.res_net(concat_feat)
        refined_disp = disp + residual
        return refined_disp


class ProgressiveRefinementNetwork(nn.Module):
    def __init__(self, max_disp: int = 192, fpn_channels: int = 128, debug: bool = False):
        super().__init__()
        self.max_disp = max_disp
        self.debug = debug
        
        self.stages = nn.ModuleDict({
            'p5': RefinementStage(fpn_channels, 'p5', max_disp // 32, debug=debug),
            'p4': RefinementStage(fpn_channels, 'p4', max_disp // 16, debug=debug),
            'p3': RefinementStage(fpn_channels, 'p3', max_disp // 8,  debug=debug),
            'p2': RefinementStage(fpn_channels, 'p2', max_disp // 4,  debug=debug),
            'p1': RefinementStage(fpn_channels, 'p1', max_disp // 2,  debug=debug),
            'full_res': RefinementStage(fpn_channels, 'full_res', max_disp, debug=debug),
        })

        self.upsamplers = nn.ModuleDict({
            'p5': ConvexUpsample2x(fpn_channels),
            'p4': ConvexUpsample2x(fpn_channels),
            'p3': ConvexUpsample2x(fpn_channels),
            'p2': ConvexUpsample2x(fpn_channels),
            'p1': ConvexUpsample2x(fpn_channels),
            'full_res': ConvexUpsample2x(fpn_channels),
        })

    def forward(
        self, 
        init_disp: torch.Tensor, 
        left_features: Dict[str, torch.Tensor], 
        right_features: Dict[str, torch.Tensor],
        start_level: str = 'p3'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        current = init_disp
        inter = {}
        all_scales = ['p5', 'p4', 'p3', 'p2', 'p1', 'full_res']
        
        try:
            start_idx = all_scales.index(start_level)
            run_scales = all_scales[start_idx:]
        except ValueError:
            run_scales = all_scales

        prev_lf = left_features[start_level]

        for i, s in enumerate(run_scales):
            lf = left_features[s]
            rf = right_features[s]
            Ht, Wt = lf.shape[2:]
            
            if current.shape[2:] != (Ht, Wt):
                current = self.upsamplers[s](current, prev_lf, target_size=(Ht, Wt))

            current = self.stages[s](current, lf, rf)
            inter[s] = current
            prev_lf = lf
            
        return current, inter