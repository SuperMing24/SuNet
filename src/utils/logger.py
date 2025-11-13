"""独立的日志记录器模块 - 重新设计"""
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class BaseLogger(ABC):
    """日志记录器基类 - 清晰的回调接口"""

    @abstractmethod
    def _log_message(self, message: str):
        pass

    @abstractmethod
    def log_time(self, message: str, **kwargs):
        """记录时间信息"""
        pass

    @abstractmethod
    def log_loss(self, phase: str, epoch: int, batch_idx: int, batch_total: int, loss: float, **kwargs):
        """记录损失信息"""
        pass

    @abstractmethod
    def log_metrics(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """记录评估指标"""
        pass

    @abstractmethod
    def log_images(self, epoch: int, data, output, target, prefix: str, **kwargs):
        """记录图像"""
        pass

class SegLogger(BaseLogger):
    """分割任务专用日志记录器 """

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

    def _log_message(self, message: str):
        """统一的消息记录方法"""
        timestamp = datetime.now().strftime(self.time_format)
        full_message = f"{timestamp} {message}"

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{full_message}\n")

        # 输出到控制台
        print(full_message)

    def log_time(self, message: str, **kwargs):
        """记录时间信息 - 自动处理额外参数"""
        if kwargs:
            # 如果有额外参数，构建完整的消息
            extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
            message = f"{message} | {extra_info}"
        self._log_message(message)

    def log_loss(self, phase: str, epoch: int, batch_idx: int, batch_total: int, loss: float, **kwargs):
        """记录损失信息"""
        if (phase == 'train' or phase == 'val') and batch_idx % self.log_interval == 0:
            # 基础消息
            msg = (f"Epoch {epoch}/{self.total_epochs} | "
                   f"batch: {batch_idx}/{batch_total} | {phase} Loss: {loss:.4f}")

            # 如果有额外参数，自动添加
            if kwargs:
                extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
                msg += f" | {extra_info}"

            self._log_message(msg)
        elif phase != 'train' and phase != 'val':
            msg = f'Not Train or Val！'
            self._log_message(msg)

    def log_metrics(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """记录评估指标"""
        # 构建指标字符串
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        # 基础消息
        msg = (f"Epoch {epoch}/{self.total_epochs} | "
               f"{metrics_str}")

        # 如果有额外参数，自动添加
        if kwargs:
            extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
            msg += f" | {extra_info}"

        self._log_message(msg)

    def log_images(self, epoch: int, data, output, target, prefix: str, **kwargs):
        """记录图像 - 独立的业务逻辑"""
        try:
            for j in range(min(2, data.shape[0])):  # 保存前2个样本
                plot_results(
                    data[j], target[j], output[j],
                    f'{self.image_dir}/{prefix}_epoch_{epoch}_sample_{j}.png'
                )
                # 基础消息
                msg = f"保存可视化结果: {prefix}_epoch_{epoch}_sample_{j}.png"

                # 如果有额外参数，自动添加
                if kwargs:
                    extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
                    msg += f" | {extra_info}"

                self._log_message(msg)
        except Exception as e:
            self._log_message(f"可视化保存失败: {e}")



class SimpleConsoleLogger(BaseLogger):
    """简单控制台日志记录器 - 用于快速调试"""

    def log_time(self, message: str, **kwargs):
        """记录时间信息"""
        print(message)

    def log_loss(self, phase: str, epoch: int, batch_idx: int, loss: float, **kwargs):
        """记录损失信息"""
        if phase == 'train' and batch_idx % 10 == 0:
            msg = (f"Epoch {epoch} | "
                   f"batch: {batch_idx} | {phase} Loss: {loss:.4f}")
            self.log_time(msg)

    def log_metrics(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """记录评估指标"""
        metrics_str = " ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
        msg = f"Epoch {epoch} | Train Loss: {metrics.get('train_loss', 0.0):.4f} | Val: {metrics_str}"
        self.log_time(msg)

    def log_images(self, epoch: int, data, output, target, prefix: str, **kwargs):
        """记录图像"""
        pass  # 控制台日志不需要图像
