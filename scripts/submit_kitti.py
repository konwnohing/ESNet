import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import yaml

# ==========================================
# 1. 环境准备
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet 

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

print("🚀 正在装载终极打榜兵器...")

# ==========================================
# 2. 加载配置与模型
# ==========================================
# 🌟 建议直接指向你最后一次训练用的 YAML
config_path = os.path.join(PROJECT_ROOT, "configs/kitti_ohem.yaml")
config = load_config(config_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_disp = config['model'].get('max_disp', 192)

# 初始化模型
model = EnhancedStereoNet(max_disp=max_disp).to(device)

# 🌟 指向你 300 轮全量训练的最优权重
checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints_submitlast/checkpoint_last.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)

# 自动处理分布式训练的 module. 前缀
state_dict = checkpoint['model_state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"✅ 模型与权重装载完毕！当前最大视差限制: {max_disp}")

# ==========================================
# 3. 路径与输出准备
# ==========================================
# 测试集路径
test_left_dir = "/root/autodl-tmp/kitti_dataset/testing/image_2"
test_right_dir = "/root/autodl-tmp/kitti_dataset/testing/image_3"

# 官方要求文件夹名：disp_0
output_dir = os.path.join(PROJECT_ROOT, "kitti_submission/disp_0") 
os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(test_left_dir) if f.endswith('_10.png')])

# ==========================================
# 4. 执行推理
# ==========================================
mean = config['data'].get('mean', [0.485, 0.456, 0.406])
std = config['data'].get('std', [0.229, 0.224, 0.225])

with torch.no_grad():
    for i, img_name in enumerate(image_files, 1):
        left_pil = Image.open(os.path.join(test_left_dir, img_name)).convert('RGB')
        right_pil = Image.open(os.path.join(test_right_dir, img_name)).convert('RGB')
        orig_w, orig_h = left_pil.size
        
        # 🌟 动态计算 Padding (确保是 32 的倍数)
        pad_h = (orig_h + 31) // 32 * 32 - orig_h
        pad_w = (orig_w + 31) // 32 * 32 - orig_w
        
        left_padded = TF.pad(left_pil, (0, 0, pad_w, pad_h), padding_mode='edge')
        right_padded = TF.pad(right_pil, (0, 0, pad_w, pad_h), padding_mode='edge')
        
        left_t = TF.normalize(TF.to_tensor(left_padded), mean, std).unsqueeze(0).to(device)
        right_t = TF.normalize(TF.to_tensor(right_padded), mean, std).unsqueeze(0).to(device)
        
        # 推理
        outputs = model(left_t, right_t)
        disp_pred = outputs['disp_final'] if isinstance(outputs, dict) else outputs
        
        # 裁剪回原始尺寸并移动到 CPU
        disp_pred = disp_pred[0, 0, :orig_h, :orig_w].cpu().numpy()
        
        # 🌟 关键：物理截断 + 官方 16-bit 编码
        disp_pred = np.clip(disp_pred, 0, max_disp)
        disp_kitti = (disp_pred * 256.0).astype(np.uint16)
        
        # 保存
        cv2.imwrite(os.path.join(output_dir, img_name), disp_kitti)
        
        if i % 20 == 0:
            print(f"⏳ 进度: {i}/200 | 已生成: {img_name}")

print(f"\n🎉 测试图生成完毕！结果保存在: {output_dir}")