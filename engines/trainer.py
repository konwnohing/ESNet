import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union

from losses.multilevel_loss import FixedMultiscaleLoss
from losses.physics_constraint_loss import PhysicsConstraintLoss

class AverageMeter:
    """计算并存储平均值"""
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

def compute_gate_loss(disp_init: torch.Tensor, disp_final: torch.Tensor, 
                      conf_pred: torch.Tensor, disp_gt: torch.Tensor, 
                      valid_mask: torch.Tensor, alpha: float = 1.0):
    """
    置信度感知损失函数 (包含 Confidence Loss 与 Focal Residual Loss)
    """
    # 如果该 batch 没有有效像素，直接返回 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=disp_init.device), torch.tensor(0.0, device=disp_init.device)

    # 1. 生成置信度伪标签 (无梯度)
    with torch.no_grad():
        error_init = torch.abs(disp_init - disp_gt)
        # 巧妙设计：误差为 0 时目标置信度为 1；误差越大，目标置信度呈指数衰减趋近于 0
        conf_target = torch.exp(-alpha * error_init)

    # 2. 置信度损失 (BCE Loss) - 教会网络认识自己的错误
    loss_conf = F.binary_cross_entropy(conf_pred[valid_mask], conf_target[valid_mask])

    # 3. 焦点残差损失 (Focal Smooth L1)
    # 我们希望网络在精修阶段，不要去碰那些已经对的像素(conf_target接近1)，
    # 而是把所有的 Loss 惩罚都集中在那些初始误差大(conf_target接近0)的地方！
    residual_weight = 1.0 - conf_target 
    l1_error = F.smooth_l1_loss(disp_final[valid_mask], disp_gt[valid_mask], reduction='none')
    
    # 加权求均值
    loss_disp_weighted = (l1_error * residual_weight[valid_mask]).mean()

    return loss_conf, loss_disp_weighted


class TrainerFixed:
    """训练器 - 支持门控精修机制与各种消融参数"""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        train_cfg = config.get('training', {})
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})

        self.crop_height = data_cfg.get('crop_height', 256)
        self.crop_width = data_cfg.get('crop_width', 512)
        self.max_disp = model_cfg.get('max_disp', 192)

        self.grad_clip_norm = float(train_cfg.get('gradient_clip_norm', 1.0))
        self.freeze_bn = bool(train_cfg.get('freeze_bn', int(train_cfg.get('batch_size', 1)) == 1))

        # ========== 损失配置 ==========
        loss_cfg = train_cfg.get('loss', {})
        use_physics_loss = bool(loss_cfg.get('physics_constraint', {}).get('enabled', True))
        negative_penalty_weight = float(loss_cfg.get('negative_penalty', {}).get('weight', 0.1))
        smoothness_weight = float(loss_cfg.get('smoothness', {}).get('weight', 0.05))
        use_ssim = bool(loss_cfg.get('ssim', {}).get('enabled', False))
        
        # 🌟🌟 新增：专门解析 spatial_grad 边缘切割配置 🌟🌟
        spatial_grad_cfg = loss_cfg.get('spatial_grad', {})
        self.spatial_grad_enabled = bool(spatial_grad_cfg.get('enabled', False))
        self.spatial_grad_weight = float(spatial_grad_cfg.get('weight', 0.1)) if self.spatial_grad_enabled else 0.0
        
        # 🌟 消融实验专用超参数读取
        edge_alpha = float(loss_cfg.get('edge_alpha', 15.0))
        relative_loss_weight = float(loss_cfg.get('relative_loss_weight', 2.0))
        ohem_fraction = float(loss_cfg.get('ohem_fraction', 0.3))

        # 🌟🌟 Gate Refinement 专属超参数读取 🌟🌟
        gate_cfg = loss_cfg.get('gate_refinement', {})
        self.gate_enabled = bool(gate_cfg.get('enabled', True))
        self.gate_loss_weight = float(gate_cfg.get('weight', 1.0)) # 门控Loss在总Loss中的比重
        self.gate_alpha = float(gate_cfg.get('alpha', 1.0))        # 置信度伪标签的指数衰减系数
        
        dist_cfg = loss_cfg.get('distribution', {})
        dist_enabled = bool(dist_cfg.get('enabled', False))
        dist_weight = float(dist_cfg.get('weight', 0.0)) if dist_enabled else 0.0
        dist_temp = float(dist_cfg.get('temperature', 0.1))
            
        loss_weights = train_cfg.get('loss_weights', {
            'final': 1.0, 'full_res': 0.8, 'p3': 0.5, 'disp_coarse': 0.2
        })

        print(f"[Trainer] 配置解析:")
        print(f"  max_disp: {self.max_disp}")
        print(f"  平滑性权重: {smoothness_weight}, Edge Alpha: {edge_alpha}")
        print(f"  OHEM 挖掘比例: {ohem_fraction}")
        print(f"  EDW 深度加权: {relative_loss_weight}")
        print(f"  🌟 门控精修机制: {'启用' if self.gate_enabled else '禁用'} (Weight={self.gate_loss_weight}, Alpha={self.gate_alpha})")
        
        # 🌟 打印空间梯度的状态
        if self.spatial_grad_enabled:
            print(f"  🔨 空间梯度(边缘切割)机制: 启用 (Weight={self.spatial_grad_weight})")
        else:
            print(f"  🔨 空间梯度(边缘切割)机制: 关闭")

        self.criterion = FixedMultiscaleLoss(
            max_disp=self.max_disp,
            weights=loss_weights,
            negative_penalty_weight=negative_penalty_weight,
            smoothness_weight=smoothness_weight,
            use_ssim=use_ssim,
            monitor_negative=True,
            distribution_weight=dist_weight,
            distribution_temperature=dist_temp,
            spatial_grad_weight=self.spatial_grad_weight,  # 传入解析后的权重
            edge_alpha=edge_alpha,                    
            relative_loss_weight=relative_loss_weight,
            ohem_fraction=ohem_fraction                
        )

        if use_physics_loss:
            physics_weight = float(loss_cfg.get('physics_constraint', {}).get('weight', 0.05))
            self.physics_criterion = PhysicsConstraintLoss(
                weight=physics_weight, mode='final', max_disp=self.max_disp
            )
        else:
            self.physics_criterion = None

        # ========== 优化器与调度器 ==========
        optimizer_type = str(train_cfg.get('optimizer', 'adam')).lower()
        learning_rate = float(train_cfg.get('learning_rate', 1e-4))
        weight_decay = float(train_cfg.get('weight_decay', 1e-4))

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(train_cfg.get('num_epochs', 50)), eta_min=1e-6
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        logging_cfg = config.get('logging', {})
        self.checkpoint_dir = logging_cfg.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._loss_weight_keys = set(loss_weights.keys())

    @staticmethod
    def _freeze_bn_layers(m: nn.Module):
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

    def _flatten_outputs_for_loss(self, outputs: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        if isinstance(outputs, dict) and 'all_disparities' in outputs and isinstance(outputs['all_disparities'], dict):
            return {**outputs, **outputs['all_disparities']}
        return outputs

    def _sanity_check_outputs(self, outputs_for_loss: Dict[str, Any]):
        pass 

    def save_checkpoint(self, epoch: int, checkpoint_dir: Optional[str] = None, is_best: bool = False):
        if checkpoint_dir is None: checkpoint_dir = self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            'epoch': int(epoch),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'max_disp': int(self.max_disp),
        }
        ckpt_path = os.path.join(checkpoint_dir, 'checkpoint_last.pth')
        torch.save(state, ckpt_path)
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        if not os.path.exists(checkpoint_path): return 0
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                # DataParallel handling
                model_has_module = any(k.startswith('module.') for k in model_dict.keys())
                ckpt_has_module = any(k.startswith('module.') for k in pretrained_dict.keys())
                if (not model_has_module) and ckpt_has_module:
                    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
                elif model_has_module and (not ckpt_has_module):
                    pretrained_dict = {'module.' + k: v for k, v in pretrained_dict.items()}
                
                # Strict=False to avoid minor mismatches crashing
                self.model.load_state_dict({k: v for k, v in pretrained_dict.items() if k in model_dict}, strict=False)
            
            self.current_epoch = 0 
            self.global_step = 0
            self.best_loss = float('inf')
            print(f"✓ checkpoint 恢复成功: 准备开启 Fine-tuning!")
            return self.current_epoch
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return 0

    def train_epoch(self, train_loader: DataLoader, epoch: int, writer=None) -> Dict[str, float]:
        self.model.train()
        if getattr(self, 'freeze_bn', False):
            self.model.apply(self._freeze_bn_layers)

        losses = {
            'total': AverageMeter('Total', ':.4f'),
            'gate_conf': AverageMeter('GateConf', ':.4f'),
            'gate_res': AverageMeter('GateRes', ':.4f')
        }
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                left_img = batch['left'].to(self.device)
                right_img = batch['right'].to(self.device)
                disp_gt = batch['disparity'].to(self.device)
            else:
                left_img, right_img, disp_gt = batch
                left_img, right_img, disp_gt = left_img.to(self.device), right_img.to(self.device), disp_gt.to(self.device)

            bs = left_img.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(left_img, right_img)
            outputs_for_loss = self._flatten_outputs_for_loss(outputs)

            # 1. 计算原有多尺度/物理损失
            loss_dict = self.criterion(outputs_for_loss, disp_gt, left_img)
            total_loss = loss_dict['total']

            if self.physics_criterion is not None:
                total_loss += self.physics_criterion(outputs_for_loss)

            # 🌟🌟 2. 核心：计算并融合置信度门控损失 🌟🌟
            if self.gate_enabled and 'confidence' in outputs and 'disp_init' in outputs:
                valid_mask = (disp_gt > 0) & (disp_gt < self.max_disp)
                
                loss_conf, loss_disp_weighted = compute_gate_loss(
                    disp_init=outputs['disp_init'], 
                    disp_final=outputs['disp_final'], 
                    conf_pred=outputs['confidence'], 
                    disp_gt=disp_gt, 
                    valid_mask=valid_mask,
                    alpha=self.gate_alpha
                )
                
                # 累加到总损失中
                total_loss += self.gate_loss_weight * (loss_conf + loss_disp_weighted)
                
                # 更新监控器
                losses['gate_conf'].update(loss_conf.item(), bs)
                losses['gate_res'].update(loss_disp_weighted.item(), bs)

            # 3. 反向传播
            total_loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            losses['total'].update(total_loss.item(), bs)

            if batch_idx % 20 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                log_str = f"Epoch[{epoch:03d}] [{batch_idx:03d}/{len(train_loader):03d}] "
                log_str += f"total_loss={float(total_loss.item()):.4f} "
                if self.gate_enabled:
                    log_str += f"(conf_loss={float(loss_conf.item()):.4f}, res_loss={float(loss_disp_weighted.item()):.4f}) "
                log_str += f"lr={lr:.6g}"
                print(log_str)

            self.global_step += 1

        self.current_epoch = epoch + 1
        return {'total': losses['total'].avg}