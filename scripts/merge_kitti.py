import os
import shutil
from tqdm import tqdm

# 配置原始路径和目标路径
KITTI_2015_DIR = "/root/autodl-tmp/kitti_dataset/training"
KITTI_2012_DIR = "/root/autodl-tmp/kitti_dataset/kitti2012/training"
MIX_DIR = "/root/autodl-tmp/kitti_mix/training"

def merge_datasets():
    print("🚀 开始融合 KITTI 2012 与 2015 数据集...")
    
    target_img_left = os.path.join(MIX_DIR, "image_2")
    target_img_right = os.path.join(MIX_DIR, "image_3")
    target_disp = os.path.join(MIX_DIR, "disp_occ_0")
    
    os.makedirs(target_img_left, exist_ok=True)
    os.makedirs(target_img_right, exist_ok=True)
    os.makedirs(target_disp, exist_ok=True)

    # ================= 1. 搬运 KITTI 2015 =================
    print("📦 正在处理 KITTI 2015...")
    # 🌟 核心修复：死死盯住 _10.png，绝不碰 _11.png
    img2015_names = [f for f in os.listdir(os.path.join(KITTI_2015_DIR, 'image_2')) if f.endswith('_10.png')]
    for name in tqdm(img2015_names):
        new_name = f"2015_{name}"
        shutil.copy(os.path.join(KITTI_2015_DIR, 'image_2', name), os.path.join(target_img_left, new_name))
        shutil.copy(os.path.join(KITTI_2015_DIR, 'image_3', name), os.path.join(target_img_right, new_name))
        
        # 加上防爆护盾：只拷贝存在的视差图
        disp_path = os.path.join(KITTI_2015_DIR, 'disp_occ_0', name)
        if os.path.exists(disp_path):
            shutil.copy(disp_path, os.path.join(target_disp, new_name))

    # ================= 2. 搬运 KITTI 2012 =================
    print("📦 正在处理 KITTI 2012...")
    # 🌟 同样过滤 _10.png
    img2012_names = [f for f in os.listdir(os.path.join(KITTI_2012_DIR, 'colored_0')) if f.endswith('_10.png')]
    for name in tqdm(img2012_names):
        new_name = f"2012_{name}"
        shutil.copy(os.path.join(KITTI_2012_DIR, 'colored_0', name), os.path.join(target_img_left, new_name))
        shutil.copy(os.path.join(KITTI_2012_DIR, 'colored_1', name), os.path.join(target_img_right, new_name))
        
        disp_path = os.path.join(KITTI_2012_DIR, 'disp_occ', name)
        if os.path.exists(disp_path):
            shutil.copy(disp_path, os.path.join(target_disp, new_name))

    print(f"✅ 完美搞定！融合后的纯净训练集总计 {len(img2015_names) + len(img2012_names)} 张图像！")
    print(f"存放路径为: {MIX_DIR}")

if __name__ == "__main__":
    merge_datasets()