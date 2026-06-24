import yaml
import os
import time
import copy
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
        
    # 设置默认值
    config = set_default_config(config)
    
    # 验证配置
    validate_config(config)
    
    return config

def _ensure_dict(parent_dict: Dict[str, Any], key: str) -> Dict[str, Any]:
    """【防御性编程】确保字典中某个键对应的值是一个字典，如果是 None 或缺失，则替换为空字典"""
    val = parent_dict.get(key)
    if not isinstance(val, dict):
        val = {}
        parent_dict[key] = val
    return val

def set_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """设置默认配置值（加入防 None 崩溃保护）"""
    
    # 模型配置默认值
    model_config = _ensure_dict(config, 'model')
    model_config.setdefault('backbone_type', 'resnet18')
    model_config.setdefault('max_disp', 192)
    
    # 负视差惩罚默认配置
    negative_config = _ensure_dict(model_config, 'negative_penalty')
    negative_config.setdefault('enabled', True)
    negative_config.setdefault('penalty_weight', 0.05)
    negative_config.setdefault('clip_negative', False)
    
    # 训练配置默认值
    train_config = _ensure_dict(config, 'training')
    train_config.setdefault('batch_size', 4)
    train_config.setdefault('num_epochs', 50)
    train_config.setdefault('learning_rate', 0.001)
    train_config.setdefault('weight_decay', 0.0001)
    train_config.setdefault('lr_step_size', 10)
    train_config.setdefault('lr_gamma', 0.5)
    train_config.setdefault('gradient_clip_norm', 1.0)
    
    # 优化器默认配置
    train_config.setdefault('optimizer', 'adam')
    train_config.setdefault('scheduler', 'step')
    
    # 梯度惩罚默认配置
    gradient_config = _ensure_dict(train_config, 'gradient_penalty')
    gradient_config.setdefault('enabled', True)
    gradient_config.setdefault('max_norm', 1.0)
    gradient_config.setdefault('penalty_type', 'norm')
    
    # 负视差惩罚策略默认配置
    negative_strategy = _ensure_dict(train_config, 'negative_disparity')
    negative_strategy.setdefault('enabled', True)
    negative_strategy.setdefault('penalty_strategy', 'adaptive')
    negative_strategy.setdefault('monitor_threshold', 0.01)
    
    # 多阶段惩罚权重默认值
    stage_weights = _ensure_dict(negative_strategy, 'stage_weights')
    stage_weights.setdefault('stage1_epochs', 5)
    stage_weights.setdefault('stage1_weight', 0.05)
    stage_weights.setdefault('stage2_epochs', 15)
    stage_weights.setdefault('stage2_weight', 0.1)
    stage_weights.setdefault('stage3_weight', 0.15)
    
    # 多尺度损失权重默认值
    loss_weights = _ensure_dict(train_config, 'loss_weights')
    loss_weights.setdefault('final', 1.0)
    loss_weights.setdefault('midres', 0.5)
    loss_weights.setdefault('lowres', 0.2)
    
    # 损失函数详细配置默认值
    loss_config = _ensure_dict(train_config, 'loss')
    loss_config.setdefault('type', 'multilevel')
    
    # 负视差惩罚默认配置
    neg_penalty_config = _ensure_dict(loss_config, 'negative_penalty')
    neg_penalty_config.setdefault('enabled', True)
    neg_penalty_config.setdefault('weight', 0.1)
    neg_penalty_config.setdefault('type', 'l1')
    
    # 平滑性约束默认配置
    smooth_config = _ensure_dict(loss_config, 'smoothness')
    smooth_config.setdefault('enabled', True)
    smooth_config.setdefault('weight', 0.05)
    smooth_config.setdefault('edge_aware', True)
    
    # SSIM默认配置
    ssim_config = _ensure_dict(loss_config, 'ssim')
    ssim_config.setdefault('enabled', False)
    ssim_config.setdefault('weight', 0.85)
    ssim_config.setdefault('window_size', 11)
    
    # 物理约束默认配置
    physics_config = _ensure_dict(loss_config, 'physics_constraint')
    physics_config.setdefault('enabled', True)
    physics_config.setdefault('weight', 0.05)
    
    # 一致性约束默认配置
    consistency_config = _ensure_dict(loss_config, 'consistency')
    consistency_config.setdefault('enabled', True)
    consistency_config.setdefault('weight', 0.1)
    consistency_config.setdefault('type', 'left-right')
    
    # 数据配置默认值
    data_config = _ensure_dict(config, 'data')
    data_config.setdefault('dataset', 'driving_small')
    data_config.setdefault('root_dir', 'data/driving_small')
    data_config.setdefault('crop_height', 256)
    data_config.setdefault('crop_width', 512)
    data_config.setdefault('mean', [0.485, 0.456, 0.406])
    data_config.setdefault('std', [0.229, 0.224, 0.225])
    
    # 数据增强默认配置
    aug_config = _ensure_dict(data_config, 'augmentation')
    aug_config.setdefault('enabled', True)
    aug_config.setdefault('color_aug', False)
    aug_config.setdefault('geometric_aug', True)
    aug_config.setdefault('crop_scale', [0.8, 1.0])
    aug_config.setdefault('hflip_prob', 0.5)
    
    # 日志配置默认值
    log_config = _ensure_dict(config, 'logging')
    log_config.setdefault('log_dir', './logs')
    log_config.setdefault('checkpoint_dir', './checkpoints')
    log_config.setdefault('tensorboard_dir', './tensorboard')
    log_config.setdefault('save_freq', 10)
    log_config.setdefault('val_freq', 1)
    
    # 负视差监控默认配置
    neg_monitor_config = _ensure_dict(log_config, 'negative_monitoring')
    neg_monitor_config.setdefault('enabled', True)
    neg_monitor_config.setdefault('log_frequency', 10)
    neg_monitor_config.setdefault('save_threshold', 0.02)
    
    # 可视化默认配置
    vis_config = _ensure_dict(log_config, 'visualization')
    vis_config.setdefault('enabled', True)
    vis_config.setdefault('save_disparity_maps', True)
    vis_config.setdefault('save_negative_masks', True)
    vis_config.setdefault('frequency', 50)
    
    # 评估配置默认值
    eval_config = _ensure_dict(config, 'evaluation')
    eval_config.setdefault('metrics', ['epe', 'bad_3.0', 'bad_1.0'])
    eval_config.setdefault('max_disp', 192)
    
    # 负视差评估默认配置
    neg_eval_config = _ensure_dict(eval_config, 'negative_evaluation')
    neg_eval_config.setdefault('enabled', True)
    neg_eval_config.setdefault('metrics', ['negative_ratio', 'negative_mean', 'negative_std'])
    neg_eval_config.setdefault('threshold', 0.0)
    
    # 验证集默认配置
    val_config = _ensure_dict(eval_config, 'validation')
    val_config.setdefault('compute_3px_error', True)
    val_config.setdefault('compute_1px_error', True)
    val_config.setdefault('compute_negative_stats', True)
    
    # 调试配置默认值
    debug_config = _ensure_dict(config, 'debug')
    debug_config.setdefault('enable_debug_mode', False)
    debug_config.setdefault('print_gradients', False)
    debug_config.setdefault('print_disparity_stats', True)
    
    # 负视差调试默认配置
    neg_debug_config = _ensure_dict(debug_config, 'negative_debug')
    neg_debug_config.setdefault('print_negative_ratios', True)
    neg_debug_config.setdefault('visualize_negative_regions', False)
    neg_debug_config.setdefault('log_gradients_on_negative', False)
    
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """验证配置的合法性"""
    if 'model' not in config:
        raise ValueError("配置文件中必须包含'model'部分")
    
    backbone_type = config['model'].get('backbone_type', 'resnet18')
    if backbone_type not in ['resnet18', 'resnet50']:
        print(f"⚠️ 警告: backbone_type '{backbone_type}' 不是推荐值，使用默认值'resnet18'")
        config['model']['backbone_type'] = 'resnet18'
    
    max_disp = config['model'].get('max_disp', 192)
    if not isinstance(max_disp, int) or max_disp <= 0:
        raise ValueError(f"max_disp必须是正整数，当前为: {max_disp}")
    
    crop_height = config['data'].get('crop_height', 256)
    crop_width = config['data'].get('crop_width', 512)
    
    if crop_height % 32 != 0 or crop_width % 32 != 0:
        print(f"⚠️ 警告: 输入尺寸{crop_height}x{crop_width}不是32的倍数，可能导致尺寸不匹配")
    
    neg_weight = config['training']['loss']['negative_penalty'].get('weight', 0.1)
    if neg_weight < 0 or neg_weight > 1:
        print(f"⚠️ 警告: 负视差惩罚权重{neg_weight}应该在[0, 1]范围内")
    
    lr = config['training'].get('learning_rate', 0.001)
    if lr <= 0:
        raise ValueError(f"学习率必须为正数，当前为: {lr}")
    
    print("✅ 配置验证通过")
    
def get_negative_penalty_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取负视差惩罚相关的配置"""
    negative_config = {
        'enabled': config['training']['negative_disparity']['enabled'],
        'strategy': config['training']['negative_disparity']['penalty_strategy'],
        'threshold': config['training']['negative_disparity']['monitor_threshold'],
        'stage_weights': config['training']['negative_disparity']['stage_weights'],
        'loss_weight': config['training']['loss']['negative_penalty']['weight'],
        'smoothness_weight': config['training']['loss']['smoothness']['weight'],
        'physics_weight': config['training']['loss']['physics_constraint']['weight'],
        'use_ssim': config['training']['loss']['ssim']['enabled'],
    }
    return negative_config

def print_config_summary(config: Dict[str, Any]) -> None:
    """打印配置摘要"""
    print("\n" + "="*50)
    print("配置摘要")
    print("="*50)
    
    print(f"📦 模型配置:")
    print(f"  - Backbone: {config['model']['backbone_type']}")
    print(f"  - 最大视差: {config['model']['max_disp']}")
    print(f"  - 负视差惩罚: {'启用' if config['model']['negative_penalty']['enabled'] else '禁用'}")
    
    print(f"🚀 训练配置:")
    print(f"  - Batch大小: {config['training']['batch_size']}")
    print(f"  - 训练轮数: {config['training']['num_epochs']}")
    print(f"  - 学习率: {config['training']['learning_rate']}")
    print(f"  - 优化器: {config['training']['optimizer']}")
    print(f"  - 调度器: {config['training']['scheduler']}")
    
    neg_config = get_negative_penalty_config(config)
    print(f"🎯 负视差惩罚配置:")
    print(f"  - 策略: {neg_config['strategy']}")
    print(f"  - 惩罚权重: {neg_config['loss_weight']}")
    print(f"  - 平滑性权重: {neg_config['smoothness_weight']}")
    print(f"  - 警告阈值: {neg_config['threshold']:.1%}")
    
    print(f"📊 损失函数配置:")
    print(f"  - 类型: {config['training']['loss']['type']}")
    print(f"  - 最终权重: {config['training']['loss_weights']['final']}")
    print(f"  - 中分辨率权重: {config['training']['loss_weights'].get('midres', 'N/A')}")
    
    print(f"📁 数据配置:")
    print(f"  - 数据集: {config['data']['dataset']}")
    print(f"  - 输入尺寸: {config['data']['crop_height']}x{config['data']['crop_width']}")
    print(f"  - 数据增强: {'启用' if config['data']['augmentation']['enabled'] else '禁用'}")
    
    print("="*50 + "\n")

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """保存配置到文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 配置已保存到: {save_path}")

def create_experiment_config(base_config: Dict[str, Any], 
                           experiment_name: str,
                           negative_weight: Optional[float] = None,
                           smoothness_weight: Optional[float] = None) -> Dict[str, Any]:
    """创建实验配置"""
    config = copy.deepcopy(base_config)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{experiment_name}_{timestamp}"
    
    config['logging']['log_dir'] = f"{experiment_dir}/logs"
    config['logging']['checkpoint_dir'] = f"{experiment_dir}/checkpoints"
    config['logging']['tensorboard_dir'] = f"{experiment_dir}/tensorboard"
    
    if negative_weight is not None:
        config['training']['loss']['negative_penalty']['weight'] = negative_weight
    if smoothness_weight is not None:
        config['training']['loss']['smoothness']['weight'] = smoothness_weight
        
    return config, experiment_dir