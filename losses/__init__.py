import torch.nn as nn
from .multilevel_loss import FixedMultiscaleLoss
from .physics_constraint_loss import PhysicsConstraintLoss

# 2. 确保负视差惩罚在损失函数中生效
class CombinedStereoLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multiscale = FixedMultiscaleLoss(
            max_disp=config['model']['max_disp'],
            weights=config['training']['loss_weights'],
            negative_penalty_weight=config['training']['loss']['negative_penalty']['weight'],
            smoothness_weight=config['training']['loss']['smoothness']['weight'],
            monitor_negative=True
        )
        
        # 额外的物理约束（可选）
        self.physics_loss = PhysicsConstraintLoss(
            weight=config['training']['loss']['physics_constraint']['weight']
        )
    
    def forward(self, outputs, disp_gt, left_img=None):
        # 主损失
        loss_dict = self.multiscale(outputs, disp_gt, left_img)
        total_loss = loss_dict['total']
        
        # 额外的负视差惩罚
        extra_penalty = 0.0
        for key, value in outputs.items():
            if 'disp' in key:
                extra_penalty += self.physics_loss(value)
        
        total_loss += extra_penalty
        loss_dict['physics_extra'] = extra_penalty.item()
        
        return total_loss, loss_dict