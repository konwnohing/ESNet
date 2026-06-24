import torch
import torch.nn as nn
from typing import Dict, List

class WeightFixer:
    """权重修复器"""
    
    @staticmethod
    def fix_transformer_channels(checkpoint: Dict, target_channels: int = 128) -> Dict:
        """
        修复Transformer匹配器的权重通道数
        
        参数:
            checkpoint: 检查点字典
            target_channels: 目标输入通道数
        
        返回:
            修复后的检查点
        """
        if 'model_state_dict' not in checkpoint:
            print("⚠️ 检查点中没有model_state_dict")
            return checkpoint
        
        state_dict = checkpoint['model_state_dict']
        fixed_keys = []
        
        # 要修复的层
        layers_to_fix = [
            'transformer_matcher.feature_projection.0.weight',
            'transformer_matcher.feature_projection.0.bias'
        ]
        
        for key in list(state_dict.keys()):
            for layer_pattern in layers_to_fix:
                if layer_pattern in key:
                    if 'weight' in key:
                        weight = state_dict[key]
                        # [out_channels, in_channels, kH, kW]
                        out_channels, in_channels, kH, kW = weight.shape
                        
                        if in_channels != target_channels:
                            print(f"  修复权重 {key}: {weight.shape} -> [{out_channels}, {target_channels}, {kH}, {kW}]")
                            
                            if in_channels > target_channels:
                                # 取前target_channels个通道
                                new_weight = weight[:, :target_channels, :, :].clone()
                            else:
                                # 扩展权重，用零填充
                                new_weight = torch.zeros(out_channels, target_channels, kH, kW, 
                                                       device=weight.device, dtype=weight.dtype)
                                new_weight[:, :in_channels, :, :] = weight.clone()
                            
                            state_dict[key] = new_weight
                            fixed_keys.append(key)
                    
                    elif 'bias' in key:
                        # 偏置保持不变
                        pass
        
        # 如果有修复，更新检查点
        if fixed_keys:
            checkpoint['model_state_dict'] = state_dict
            print(f"✅ 修复了 {len(fixed_keys)} 个权重")
        
        return checkpoint
    
    @staticmethod
    def check_compatibility(model: nn.Module, checkpoint_path: str) -> bool:
        """
        检查模型与检查点的兼容性
        """
        print(f"检查检查点兼容性: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' not in checkpoint:
                print("⚠️ 检查点格式错误")
                return False
            
            state_dict = checkpoint['model_state_dict']
            model_state_dict = model.state_dict()
            
            # 检查关键层
            incompatibilities = []
            
            for key in state_dict.keys():
                if key in model_state_dict:
                    expected_shape = model_state_dict[key].shape
                    actual_shape = state_dict[key].shape
                    
                    if expected_shape != actual_shape:
                        incompatibilities.append((key, expected_shape, actual_shape))
                else:
                    print(f"  ⚠️ 检查点中有但模型中无: {key}")
            
            if incompatibilities:
                print(f"⚠️ 发现 {len(incompatibilities)} 个不兼容的层:")
                for key, expected, actual in incompatibilities:
                    print(f"  {key}: 期望 {expected}, 实际 {actual}")
                return False
            
            print("✅ 检查点兼容")
            return True
            
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            return False