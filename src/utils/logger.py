"""独立的日志记录器模块 - 重新设计"""
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class BaseLogger(ABC):
    """日志记录器基类 - 清晰的回调接口"""

    @abstractmethod
    def log_time(self, message: str):
        """记录时间信息"""
        pass

    @abstractmethod
    def log_loss(self, phase: str, epoch: int, batch_idx: int, loss: float):
        """记录损失信息"""
        pass

    @abstractmethod
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """记录评估指标"""
        pass

    @abstractmethod
    def log_images(self, epoch: int, data, output, target, prefix: str):
        """记录图像"""
        pass

class SegLogger(BaseLogger):
    """分割任务专用日志记录器 - 简化实现"""

    def __init__(self, save_dir: str, total_epochs: int, log_interval: int = 10):
        self.save_dir = save_dir
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.time_format = '[%Y年%m月%d日 %H:%M:%S]'

        # 创建目录
        os.makedirs(f"{save_dir}/log", exist_ok=True)
        os.makedirs(f"{save_dir}/image", exist_ok=True)
        self.log_file = f"{save_dir}/log/log.txt"
        self.image_dir = f"{save_dir}/image"

        # 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write("=== 训练日志开始 ===\n")

    def log_time(self, message: str):
        """记录时间信息"""
        timestamp = datetime.now().strftime(self.time_format)
        msg = f"{timestamp} {message}"
        with open(self.log_file, 'a') as f:
            f.write(f"{msg}\n")
        print(msg)

    def log_loss(self, phase: str, epoch: int, batch_idx: int, loss: float):
        """记录损失信息"""
        if phase == 'train' and batch_idx % self.log_interval == 0:
            msg = (f"Epoch {epoch}/{self.total_epochs} | "
                   f"batch: {batch_idx} | {phase} Loss: {loss:.4f}")
            self.log_time(msg)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """记录评估指标"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        msg = (f"Epoch {epoch}/{self.total_epochs} | "
               f"Train Loss: {metrics.get('train_loss', 0.0):.4f} | {metrics_str}")
        self.log_time(msg)

    def log_images(self, epoch: int, data, output, target, prefix: str):
        """记录图像"""
        # try:
        #     for j in range(min(2, data.shape[0])):  # 保存前2个样本
        #         plot_results(
        #             data[j], target[j], output[j],
        #             f'{self.image_dir}/{prefix}_epoch_{epoch}_sample_{j}.png'
        #         )
        # except Exception as e:
        #     print(f"可视化保存失败: {e}")
        pass


class SimpleConsoleLogger(BaseLogger):
    """简单控制台日志记录器 - 用于快速调试"""

    def log_time(self, message: str):
        """记录时间信息"""
        print(message)

    def log_loss(self, phase: str, epoch: int, batch_idx: int, loss: float):
        """记录损失信息"""
        if phase == 'train' and batch_idx % 10 == 0:
            msg = (f"Epoch {epoch} | "
                   f"batch: {batch_idx} | {phase} Loss: {loss:.4f}")
            self.log_time(msg)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """记录评估指标"""
        metrics_str = " ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
        msg = f"Epoch {epoch} | Train Loss: {metrics.get('train_loss', 0.0):.4f} | Val: {metrics_str}"
        self.log_time(msg)

    def log_images(self, epoch: int, data, output, target, prefix: str):
        """记录图像"""
        pass  # 控制台日志不需要图像
