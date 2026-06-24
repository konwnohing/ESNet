import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 0. 挂载环境与导入
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet 

# 🌟 导入你封装好的 dataloader 构建函数
from data import create_dataloader  

# ==========================================
# 1. 核心：KITTI 官方 D1 误差计算函数
# ==========================================
def compute_kitti_metrics(pred_disp, gt_disp, valid_mask, fg_mask=None):
    """
    严格按照 KITTI 2015 官方标准计算 D1 误差：
    Outlier 定义：绝对误差 > 3px 且 相对误差 > 5%
    """
    pred_disp = pred_disp[valid_mask]
    gt_disp = gt_disp[valid_mask]
    
    # 计算绝对误差和相对误差
    abs_err = torch.abs(pred_disp - gt_disp)
    rel_err = abs_err / gt_disp
    
    # 识别错误像素 (Error Mask)
    err_mask = (abs_err > 3.0) & (rel_err > 0.05)
    
    # D1-all (所有有效像素的错误率)
    d1_all = err_mask.float().mean().item() * 100.0
    
    d1_bg, d1_fg = 0.0, 0.0
    if fg_mask is not None:
        # 提取当前 valid 区域内的 fg (前景/动态) 和 bg (背景/静态) 掩码
        fg_valid = fg_mask[valid_mask].bool()
        bg_valid = ~fg_valid
        
        if fg_valid.sum() > 0:
            d1_fg = err_mask[fg_valid].float().mean().item() * 100.0
            
        if bg_valid.sum() > 0:
            d1_bg = err_mask[bg_valid].float().mean().item() * 100.0
            
    return d1_all, d1_bg, d1_fg

# ==========================================
# 2. 评估主逻辑
# ==========================================
@torch.no_grad()
def evaluate(model, val_loader, device, use_fp16=True):
    model.eval()
    
    total_d1_all = 0.0
    total_d1_bg, total_d1_fg = 0.0, 0.0
    valid_samples_all = 0
    valid_samples_fg, valid_samples_bg = 0, 0
    
    pbar = tqdm(val_loader, desc=f"Evaluating KITTI ({'FP16' if use_fp16 else 'FP32'})")
    
    for batch in pbar:
        # 根据你的 Dataset 返回字典提取数据 (这里适配了常见命名，如遇到 key error 请微调)     
        # 有的数据集可能返回 'left_img', 这里假设是 'left'
        left_img = batch['left'].to(device)      
        right_img = batch['right'].to(device)    
        
        # 获取真实视差 (Ground Truth)
        if 'disparity' in batch:
            gt_disp = batch['disparity'].to(device)
        elif 'disp_gt' in batch:
            gt_disp = batch['disp_gt'].to(device)
        else:
            raise KeyError("未在 batch 中找到视差图，请检查 Dataset 的返回值 key")
        
        # KITTI 2015 提供的物体掩码 (用于区分 fg 和 bg)
        fg_mask = batch.get('obj_mask', None) 
        if fg_mask is not None:
            fg_mask = fg_mask.to(device)
        
        # 统一 GT 的维度到 [B, 1, H, W]
        if gt_disp.dim() == 3:
            gt_disp = gt_disp.unsqueeze(1)
            
        # --- 核心推理阶段 (开启混合精度) ---
        if use_fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(left_img, right_img)
        else:
            outputs = model(left_img, right_img)
            
        pred_disp = outputs['disp_final']
        
        # 如果推理输出尺寸与 GT 不一致，恢复回 GT 尺寸
        if pred_disp.shape[-2:] != gt_disp.shape[-2:]:
            pred_disp = F.interpolate(pred_disp, size=gt_disp.shape[-2:], mode='bilinear', align_corners=True)
            
        # KITTI 官方的有效掩码 (0 < D < 192)
        valid_mask = (gt_disp > 0) & (gt_disp < 192)
        
        if valid_mask.sum() == 0:
            continue
            
        # 计算误差
        d1_all, d1_bg, d1_fg = compute_kitti_metrics(pred_disp, gt_disp, valid_mask, fg_mask)
        
        total_d1_all += d1_all
        valid_samples_all += 1
        
        if fg_mask is not None:
            total_d1_bg += d1_bg
            total_d1_fg += d1_fg
            valid_samples_bg += 1
            valid_samples_fg += 1
            
        # 动态更新进度条
        pbar.set_postfix({'D1-all': f"{total_d1_all/valid_samples_all:.3f}%"})

    # 计算整个验证集的平均误差
    avg_d1_all = total_d1_all / valid_samples_all
    avg_d1_bg = (total_d1_bg / valid_samples_bg) if valid_samples_bg > 0 else 0.0
    avg_d1_fg = (total_d1_fg / valid_samples_fg) if valid_samples_fg > 0 else 0.0
    
    print("\n" + "="*55)
    print(f"🏆 [KITTI 2015 Validation Report] ({'FP16' if use_fp16 else 'FP32'})")
    print("="*55)
    print(f"🔹 D1-all (Overall Error) : {avg_d1_all:.3f} %")
    if valid_samples_fg > 0:
        print(f"🔹 D1-bg  (Background)    : {avg_d1_bg:.3f} %")
        print(f"🔹 D1-fg  (Foreground)    : {avg_d1_fg:.3f} %")
    print("="*55)
    
    return avg_d1_all

# ==========================================
# 3. 主函数执行
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("📦 正在初始化模型 (EnhancedStereoNet)...")
    model = EnhancedStereoNet(max_disp=192).to(device)
    
    # 🌟 暴力锁死测试迭代次数为 6 (Fast Mode)
    if hasattr(model, 'transformer_matcher'):
        model.transformer_matcher.test_iters = 6
        print("⚡ 已强制锁定 GRU 迭代次数: test_iters = 6")
    
    # 加载权重
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints_submittest/model_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"📥 正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        # 兼容 DDP 训练保存的权重
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print("❌ 未找到权重文件，请检查路径！")
        sys.exit(1)

    # 🌟 构造专用于评估的 Config 字典
    print("🚧 正在准备 DataLoader 配置...")
    eval_config = {
        'data': {
            'dataset': 'kitti',
            # ⚠️⚠️⚠️ 请务必修改下面这个路径为你服务器上的真实 KITTI 路径！
            'root_dir': '/root/autodl-tmp/kitti_dataset', 
            'kitti_year': 2015,
            'crop_height': 384,   
            'crop_width': 1248,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'augmentation': {'enabled': False} # 验证集绝对不能开数据增强
        },
        'training': {
            'batch_size': 1,      # 验证集 Batch Size 必须为 1
            'num_workers': 4,
            'overfit': False
        },
        'model': {
            'max_disp': 192
        },
        'debug': {
            'enable_debug_mode': False
        }
    }

    print("📦 正在构建 KITTI 验证集 Dataloader...")
    val_loader = create_dataloader(eval_config, split='val')
    
    if len(val_loader) > 0:
        # 开始评估 (默认开启 FP16)
        evaluate(model, val_loader, device, use_fp16=True)
    else:
        print("❌ Dataloader 为空，请检查数据集路径是否正确！")