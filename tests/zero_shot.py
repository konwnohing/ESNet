import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T  # 🌟 新增：用于标准归一化
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 确保能找到 models 文件夹
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet

# KITTI 路径
KITTI_PATH = "/root/autodl-tmp/kitti_dataset/training" 
WEIGHT_PATH = "/root/autodl-tmp/project/stereo-matching/checkpoints_sceneflow_pretrain_1/model_best.pth" # 确保路径正确

# 🌟 ImageNet 归一化标准 (必须与训练时严格一致)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def evaluate_kitti():
    device = torch.device('cuda')
    
    print("🚀 正在加载 EnhancedStereoNet 模型 (Max Disp=192)...")
    model = EnhancedStereoNet(max_disp=192).to(device)
    
    # 1. 挂载权重文件
    if not os.path.exists(WEIGHT_PATH):
        print(f"\n❌ 致命错误：找不到权重文件！\n请检查 WEIGHT_PATH 路径：{WEIGHT_PATH}")
        return

    try:
        state_dict = torch.load(WEIGHT_PATH, map_location=device)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 权重加载成功！准备进入闭卷考试...")
    except Exception as e:
        print(f"⚠️ 权重解析失败: {e}")
        return

    model.eval()

    # 2. 梳理测试集路径
    image2_dir = os.path.join(KITTI_PATH, 'image_2')
    image3_dir = os.path.join(KITTI_PATH, 'image_3')
    disp_dir = os.path.join(KITTI_PATH, 'disp_occ_0')

    img_names = sorted([f for f in os.listdir(image2_dir) if f.endswith('_10.png')])
    
    print(f"\n📊 开始进行 KITTI 2015 Zero-Shot 评估，共计 {len(img_names)} 张真实驾驶场景图...")
    
    total_epe_sum = 0.0
    total_d1_sum = 0.0
    total_valid_pixels = 0

    # 3. 逐图推理与计算
    with torch.no_grad():
        for img_name in tqdm(img_names, desc="🚗 跑图中 (Evaluating)"):
            # 读取左右图 (H, W, 3)
            left_img = np.array(Image.open(os.path.join(image2_dir, img_name)).convert('RGB'), dtype=np.float32) / 255.0
            right_img = np.array(Image.open(os.path.join(image3_dir, img_name)).convert('RGB'), dtype=np.float32) / 255.0
            
            # 读取真实视差图
            disp_gt_img = Image.open(os.path.join(disp_dir, img_name))
            disp_gt = np.ascontiguousarray(disp_gt_img, dtype=np.float32) / 256.0
            
            # 转换为 Tensor: [1, 3, H, W]
            left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).to(device)
            right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 🌟 核心修复：执行 ImageNet 归一化！
            left_tensor = normalize(left_tensor.squeeze(0)).unsqueeze(0)
            right_tensor = normalize(right_tensor.squeeze(0)).unsqueeze(0)
            
            _, _, h, w = left_tensor.shape
            
            # Pad 填充
            top_pad = 32 - (h % 32) if h % 32 != 0 else 0
            right_pad = 32 - (w % 32) if w % 32 != 0 else 0
            
            left_tensor = F.pad(left_tensor, (0, right_pad, top_pad, 0))
            right_tensor = F.pad(right_tensor, (0, right_pad, top_pad, 0))

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_disp = model(left_tensor, right_tensor)
                
            # 🌟 核心修复：精准捕捉 Gate Refinement 层的输出
            if isinstance(pred_disp, dict):
                if 'disp_final' in pred_disp:  # 我们模型的最终输出 key
                    pred_disp = pred_disp['disp_final']
                else:
                    pred_disp = list(pred_disp.values())[0]
                    
            if isinstance(pred_disp, (list, tuple)):
                pred_disp = pred_disp[-1] 
                
            pred_disp = pred_disp.squeeze().cpu().numpy()
            if pred_disp.ndim == 3:
                pred_disp = pred_disp[0]
            
            # 🌟 工程安全修复：使用绝对尺寸裁剪，避免 `:-0` 的空切片 Bug
            pred_disp = pred_disp[top_pad : top_pad + h, 0 : w]

            # 计算指标
            mask = (disp_gt > 0) & (disp_gt < 192)
            
            if mask.sum() == 0:
                continue
            
            valid_pixels = mask.sum()
            abs_err = np.abs(pred_disp[mask] - disp_gt[mask])
            
            total_epe_sum += np.sum(abs_err)
            
            d1_mask = (abs_err > 3) & (abs_err / disp_gt[mask] > 0.05)
            total_d1_sum += np.sum(d1_mask)
            
            total_valid_pixels += valid_pixels

    # 计算全局平均指标
    final_epe = total_epe_sum / total_valid_pixels
    final_d1_all = (total_d1_sum / total_valid_pixels) * 100.0

    print("\n" + "="*55)
    print("🏆 KITTI 2015 Zero-Shot 跨域评估终极成绩单 🏆")
    print("="*55)
    print(f"🎯 D1-all (三像素错误率): {final_d1_all:.2f} %")
    print(f"📏 EPE (平均端点误差):  {final_epe:.4f} px")
    print("="*55)

if __name__ == "__main__":
    evaluate_kitti()