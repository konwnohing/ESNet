# data/pfm_utils.py
import numpy as np

def read_pfm(file_path: str) -> np.ndarray:
    """
    读取PFM格式文件（SceneFlow/KITTI视差图格式）。
    返回一个numpy数组。
    """
    with open(file_path, 'rb') as f:
        # 读取文件头
        header = f.readline().decode('latin-1').strip()
        if header not in ['PF', 'Pf']:
            raise ValueError(f'无效的PFM文件头: {header}')
        
        # 读取宽度和高度（跳过可能的注释行）
        line = f.readline().decode('latin-1')
        while line.startswith('#'):
            line = f.readline().decode('latin-1')
        width, height = map(int, line.strip().split())
        
        # 读取缩放因子
        scale_line = f.readline().decode('latin-1')
        scale = float(scale_line.strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        # 读取二进制数据
        data = np.fromfile(f, dtype=endian + 'f')
        
        # 重塑数据
        if header == 'PF':
            data = data.reshape((height, width, 3))
        else:  # 'Pf'
            data = data.reshape((height, width))
        
        # 翻转（PFM是上下颠倒存储的）
        data = np.flipud(data)
        
        return data

# 兼容别名，供其他模块导入
def readPFM(file_path: str):
    """兼容旧导入名称"""
    return read_pfm(file_path)