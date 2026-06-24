#transformer_stereo_head
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ==============================================================================
# 🌟 核心 1：轻量级 ConvGRU 更新算子 (模拟人类“看图修改”的直觉)
# ==============================================================================
class ConvGRUUpdate(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64):
        super().__init__()
        # input_dim = 上下文特征 + 采样出的相关性 + 当前视差预测
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        
        # 视差残差预测头 (极其轻量)
        self.disp_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        delta_disp = self.disp_head(h)
        return h, delta_disp

# ==============================================================================
# 🌟 核心 2：重构的匹配头 (彻底抛弃 3D CNN，算力狂降，精度狂飙)
# ==============================================================================
class RecurrentCorrelationMatcher(nn.Module):
    def __init__(self, feature_channels: int = 128, max_disp: int = 192, debug: bool = False):
        super().__init__()
        # 注意：这里的特征是 1/8 分辨率传入的，所以底层最大视差要除以 8
        self.max_disp = max_disp // 8 
        self.debug = debug
        self.hidden_dim = 64
        self.context_dim = 64
        
        self.corr_levels = 4      # 相关性金字塔层数
        self.corr_radius = 4      # 每次查表的搜索半径 (-4 到 +4)

        # 1. 特征降维：拆分为 匹配特征(fmap) 和 上下文特征(cmap)
        self.proj_fmap = nn.Sequential(
            nn.Conv2d(feature_channels, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        self.proj_cmap = nn.Sequential(
            nn.Conv2d(feature_channels, self.hidden_dim + self.context_dim, 1),
            nn.GroupNorm(8, self.hidden_dim + self.context_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 迭代核心 GRU
        corr_out_dim = self.corr_levels * (self.corr_radius * 2 + 1) # 4 * 9 = 36
        self.gru = ConvGRUUpdate(
            hidden_dim=self.hidden_dim, 
            input_dim=self.context_dim + corr_out_dim + 1 # context + corr + disp
        )
        self.test_iters = 6
    def corr_volume(self, fmap1, fmap2):
        """构建全对全 1D 极线相关性矩阵 (一次矩阵乘法解决所有匹配计算)"""
        B, C, H, W = fmap1.shape
        # [B, H, W, C] @ [B, H, C, W] -> [B, H, W, W]
        fmap1 = fmap1.permute(0, 2, 3, 1).contiguous() 
        fmap2 = fmap2.permute(0, 2, 1, 3).contiguous() 
        corr = torch.matmul(fmap1, fmap2) / (C ** 0.5) 
        return corr # 左图每个像素与右图同一行所有像素的内积相似度

    def _sample_level(self, corr, disp, W1, W2_level, scale_factor):
        """在单层相关性金字塔中采样"""
        B, H, _, _ = corr.shape
        corr_reshaped = corr.view(B*H, 1, W1, W2_level)
        
        # 构建搜索范围: [-4, -3, ..., 4]
        dx = torch.linspace(-self.corr_radius, self.corr_radius, 2*self.corr_radius+1, device=corr.device)
        x_left = torch.arange(W1, device=corr.device).view(1, W1, 1)
        disp_bh = disp.view(B*H, W1, 1)

        # 🌟 核心：目标位置 = (左图位置 - 视差) / 缩放比例 + 搜索偏移
        center = (x_left - disp_bh) / scale_factor
        x_right = center + dx.view(1, 1, -1) # [B*H, W1, 9]

        # 归一化到 [-1, 1] 用于 grid_sample
        x_grid = 2.0 * x_right / max(W2_level - 1, 1) - 1.0
        y_grid = 2.0 * x_left.expand(B*H, -1, 2*self.corr_radius+1) / max(W1 - 1, 1) - 1.0
        grid = torch.stack([x_grid, y_grid], dim=-1) # [B*H, W1, 9, 2]

        sampled = F.grid_sample(corr_reshaped, grid, align_corners=True, padding_mode='zeros')
        return sampled.view(B, H, W1, -1).permute(0, 3, 1, 2) # [B, 9, H, W1]

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = left_feat.shape

        # 1. 提取匹配特征与上下文隐藏态
        fmap1 = self.proj_fmap(left_feat)
        fmap2 = self.proj_fmap(right_feat)
        cmap = self.proj_cmap(left_feat)
        
        net, inp = torch.split(cmap, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)  # GRU 的初始隐藏状态
        inp = torch.relu(inp)  # 提供给 GRU 的静态图像上下文信息

        # 2. 构建静态的相关性矩阵 (极其省显存，无 3D CNN)
        corr_matrix = self.corr_volume(fmap1, fmap2)

        # 3. 初始化视差为 0
        disp = torch.zeros(B, 1, H, W, device=left_feat.device)
        
        # 4. RAFT 迭代引擎 (像人眼一样，不断查表并修正视差)
        iters = 6 if self.training else self.test_iters  # 训练时迭代少点省显存，推理时拉满精度
        disp_predictions = []
        
        scale_factor = 1.0
        current_corr = corr_matrix
        
        for i in range(iters):
            out_corrs = []
            tmp_corr = current_corr
            tmp_scale = 1.0
            
            # 建立多尺度查表 (获取局部细节 + 全局视野)
            for lvl in range(self.corr_levels):
                sampled = self._sample_level(tmp_corr, disp, W, tmp_corr.shape[-1], tmp_scale)
                out_corrs.append(sampled)
                tmp_corr = F.avg_pool2d(tmp_corr, kernel_size=(1, 2), stride=(1, 2))
                tmp_scale *= 2.0
                
            sampled_corr = torch.cat(out_corrs, dim=1) # [B, 36, H, W]
            
            # 将 [上下文, 查表结果, 当前视差] 打包喂给 GRU
            gru_input = torch.cat([inp, sampled_corr, disp], dim=1)
            net, delta_disp = self.gru(net, gru_input)
            
            # 修正视差！
            disp = disp + delta_disp
            disp_predictions.append(disp)

        # 保证不输出负视差
        disp_final = torch.clamp(disp_predictions[-1], 0.0, float(self.max_disp))

        return {
            # 完美对接你外层的 EnhancedStereoNet
            'disp_lowres': disp_final,  
            'disp_coarse': disp_predictions[0] if self.training else disp_final,
            'disp_predictions': disp_predictions # 方便以后用来算迭代 Loss
        }