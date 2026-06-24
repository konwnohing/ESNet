import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(
    log_dir: str,
    name: str = "stereo_matching",
    console_level: int = logging.WARNING,   # 控制台默认只显示 WARNING+，训练不会刷屏
    file_level: int = logging.INFO,         # 文件里可以记录 INFO
    max_bytes: int = 5 * 1024 * 1024,       # 单个日志文件 5MB
    backup_count: int = 3,                  # 最多保留 3 个旧日志 -> 总计 ~20MB
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # root 收集所有级别，handler 决定输出

    # 清除旧 handler（避免重复打印）
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 文件：滚动写入
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # 控制台：少输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    root_logger.addHandler(console_handler)

    return log_file

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
