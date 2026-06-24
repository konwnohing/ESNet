import torch
import os
from typing import Dict, Any, Optional

def save_checkpoint(state: Dict[str, Any], 
                   checkpoint_dir: str,
                   filename: str = 'checkpoint.pth'):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(checkpoint_path: str,
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """加载检查点"""
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint