# utils/training_monitor.py
import time

class TrainingMonitor:
    """训练监控器"""
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.train_losses = []
        self.val_losses = []
        
    def start_epoch(self):
        self.start_time = time.time()
    
    def end_epoch(self):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.epoch_times.append(elapsed)
            return elapsed
        return 0
    
    def get_avg_epoch_time(self):
        if not self.epoch_times:
            return 0
        return sum(self.epoch_times) / len(self.epoch_times)
    
    def estimate_remaining_time(self, current_epoch, total_epochs):
        avg_time = self.get_avg_epoch_time()
        remaining_epochs = total_epochs - current_epoch - 1
        return avg_time * remaining_epochs
    
    def record_loss(self, train_loss=None, val_loss=None):
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
    
    def get_summary(self):
        """获取训练总结"""
        summary = {
            'total_epochs': len(self.epoch_times),
            'total_time': sum(self.epoch_times),
            'avg_epoch_time': self.get_avg_epoch_time(),
            'min_train_loss': min(self.train_losses) if self.train_losses else None,
            'min_val_loss': min(self.val_losses) if self.val_losses else None,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None
        }
        return summary