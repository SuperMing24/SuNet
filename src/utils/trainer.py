"""训练器 - 协调所有组件"""
import os
import time
from abc import abstractmethod
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ModelManager:
    """模型管理器 - 专注模型保存加载"""

    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model: torch.nn.Module, optimizer: optim.Optimizer,
             epoch: int, filename: str, **kwargs):
        """保存模型检查点"""
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            **kwargs
        }
        torch.save(state, os.path.join(self.save_dir, filename))

    def load(self, model: torch.nn.Module, optimizer: optim.Optimizer,
             filepath: str) -> Optional[Dict[str, Any]]:
        """加载模型检查点 - 健壮实现"""
        filepath = os.path.join(self.save_dir, filepath)
        if not os.path.exists(filepath):
            print(f"检查点不存在: '{filepath}'")
            return None

        try:
            # 自动设备映射
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
            state = torch.load(filepath, map_location=map_location)

            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])

            print(f"成功加载检查点 (epoch {state.get('epoch', 'unknown')})")
            return {'epoch': state.get('epoch', 0), **state}
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None

class Trainer:
    prompt = '(trainer) '

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 初始化模型
        self.model = config['model'].to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-4))
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        # 损失函数 - 血管分割默认使用clDice
        self.loss_fn = config.get('loss_fn')
        # 评估系统
        self.metric_evaluator = config.get('metric_evaluator')

        # 日志系统
        self.loggers = config('loggers')
        self.model_manager = ModelManager(config('model_dir'))

        # 训练状态
        self.current_epoch = 1
        self.best_metric = float('inf')  # 默认监控损失
        self.train_loader = None
        self.val_loader = None

        # 早停机制
        self.patience = config.get('patience', 10)
        self.early_stop_counter = 0

        # 从检查点恢复
        if 'checkpoint' in config:
            self._load_checkpoint(config['checkpoint'])

        self.trainning = False

    def _log(self, method_name: str, *args, **kwargs):
        """统一的日志记录方法"""
        for logger in self.loggers:
            if hasattr(logger, method_name):
                log_method = getattr(logger, method_name)
                try:
                    log_method(*args, **kwargs)
                except Exception as e:
                    print(f"日志记录错误 ({method_name}): {e}")

    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        state = self.model_manager.load(self.model, self.optimizer, checkpoint_path)
        if state:
            self.current_epoch = state['epoch'] + 1  # 从下一轮开始
            self.best_metric = state.get('best_metric', self.best_metric)
            print(f"从epoch {state['epoch']}恢复训练")

    def _train_epoch(self) -> float:
        """训练阶段 - 只计算损失"""
        self.model.train()
        total_loss = 0.0
        loss_fn = self.config['loss_fn']

        for batch_idx, (data, target) in enumerate(self.train_loader):
            start_time = time.time()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            use_time = time.time() - start_time

            # 记录时间
            self._log('log_time', f"训练batch {batch_idx} 耗时: {use_time:.2f}s")

            # 记录损失
            self._log('log_loss', 'train', self.current_epoch, batch_idx, loss.item())

            if not self.trainning:
                break

        return total_loss / len(self.train_loader)

    def _validate(self) -> Dict[str, float]:
        """验证阶段 - 计算所有评估指标"""
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                start_time = time.time()
                # 同时移动data和target到设备
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())

                use_time = time.time() - start_time

                # 记录时间
                self._log('log_time', f"验证batch {batch_idx} 耗时: {use_time:.2f}s")

                # 记录验证损失（可选）
                val_loss = self.loss_fn(output, target).item()
                self._log('log_loss', 'val', self.current_epoch, batch_idx, val_loss)

        # 合并计算评估指标
        combined_output = torch.cat(all_outputs, dim=0)
        combined_target = torch.cat(all_targets, dim=0)

        if self.metric_evaluator:
            val_metrics = self.metric_evaluator.evaluate(combined_output, combined_target)
            # 记录评估指标
            self._log('log_metrics', self.current_epoch, val_metrics)
            return val_metrics
        return {}

    def start_train(self):
        """开始训练 - 添加早停机制"""
        self.train_loader = self.config['train_loader']
        self.val_loader = self.config['val_loader']
        self.trainning = True

        for epoch in range(self.current_epoch, self.config['epochs'] + 1):
            self.current_epoch = epoch

            # 记录epoch开始时间
            epoch_start_time = time.time()

            # 训练阶段
            train_loss = self._train_epoch()
            # 验证阶段
            val_metrics = self._validate()

            # 更新学习率
            self.scheduler.step(train_loss)

            # 记录epoch总耗时
            epoch_use_time = time.time() - epoch_start_time
            self._log('log_time', f"Epoch {epoch} 总耗时: {epoch_use_time:.2f}s")
            # 模型保存和早停检查
            if train_loss < self.best_metric:
                self.best_metric = train_loss
                self.early_stop_counter = 0
                self.model_manager.save(
                    self.model, self.optimizer, epoch,
                    'best_model.pth',
                    best_metric=self.best_metric,
                    val_metrics=val_metrics
                )
                self._log('log_time', f"💾 保存最佳模型 (loss: {train_loss:.4f})")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    self._log('log_time', f"🛑 早停触发，连续{self.patience}个epoch未改善")
                    break
            # 定期保存
            if epoch % 10 == 0:
                self.save()

            if not self.trainning:
                break

        self._log('log_time', "训练完成!")

    def save(self):
        filename = f'checkpoint_epoch_{self.current_epoch}.pth'
        self.model_manager.save(self.model, self.optimizer, self.current_epoch, filename)
        msg = f'save model to {filename}'
        print(msg)
        return msg

    def stop(self, arg):
        self.save()
        self.trainning = False
        print("训练停止")
