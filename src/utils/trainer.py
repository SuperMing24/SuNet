"""训练器 - 协调所有组件"""
import os
import time
import math
import json
import csv
from typing import Dict, Any, Optional, Union
from datetime import datetime

import torch
import torch.optim as optim
from IPython.conftest import work_path
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

class TrainingResultManager:
    """训练结果管理器 - 专注结构化数据保存"""

    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # CSV文件路径
        self.csv_file = os.path.join(save_dir, "training_results.csv")
        self._init_csv_file()

        # JSON备份文件路径
        self.json_file = os.path.join(save_dir, "training_results.json")

    def _init_csv_file(self):
        """初始化CSV文件表头"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 定义表头
                headers = [
                    'epoch', 'timestamp', 'train_loss', 'val_loss',
                    'learning_rate', 'epoch_duration', 'gradient_norm',
                    # 评估指标
                    'dice', 'cl_dice', 'hausdorff', 'accuracy',
                    # 模型指标
                    'best_metric'
                ]
                writer.writerow(headers)

    def save_epoch_result(self, epoch_data: Dict[str, Any]):
        """保存epoch训练结果到CSV和JSON"""
        # 保存到CSV
        self._save_to_csv(epoch_data)

        # 保存到JSON（追加模式）
        # self._save_to_json(epoch_data)

    def _save_to_csv(self, data: Dict[str, Any]):
        """保存到CSV文件"""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 按表头顺序提取数据
            row = [
                data.get('epoch', ''),
                data.get('timestamp', ''),
                data.get('train_loss', ''),
                data.get('val_loss', ''),
                data.get('learning_rate', ''),
                data.get('epoch_duration', ''),
                data.get('gradient_norm', ''),
                # 评估指标
                data.get('eval_metrics', {}).get('dice', ''),
                data.get('eval_metrics', {}).get('cl_dice', ''),
                data.get('eval_metrics', {}).get('hausdorff', ''),
                data.get('eval_metrics', {}).get('accuracy', ''),
                # 模型状态
                data.get('best_metric', '')
            ]
            writer.writerow(row)

    def _save_to_json(self, data: Dict[str, Any]):
        """保存到JSON文件（追加模式）"""
        # 读取现有数据
        all_data = []
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            except (json.JSONDecodeError, Exception):
                all_data = []

        # 添加新数据
        all_data.append(data)

        # 写回文件
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

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
        self.loss_id = config.get('loss_id')

        # 评估系统
        self.metric_evaluator = config.get('metric_evaluator')
        self.best_metric = float('inf')

        # 日志系统
        self.loggers = config.get('loggers')
        self.model_manager = ModelManager(config.get('model_dir'))

        #可视化工具
        self.visualizer = config.get('visualizer')
        self.interval_visualize = config.get('interval_visualize')
        self.image_dir = os.path.join(config.get('work_dir'), 'image')

        # 训练结果管理器
        self.result_manager = TrainingResultManager(config.get('model_dir'))

        # 训练状态
        self.current_epoch = 1
        self.train_loader = None
        self.val_loader = None
        self.train_losses = []  # 记录每个epoch的训练损失

        self.training_actions = config.get('training_actions')

        # 早停机制
        self.patience = config.get('patience', 10)
        self.early_stop_counter = 0

        # 参数存档点检查，与恢复
        if 'checkpoint' in config:
            self._load_checkpoint(config['checkpoint'])

        # 评估质量检查
        self.enable_quality_checks = config.get('enable_quality_checks', False)
        # 训练状态检查
        self.training = False

    def _log(self, method_name: str, *args, **kwargs):
        """统一的日志记录方法"""
        for logger in self.loggers:
            if hasattr(logger, method_name):
                log_method = getattr(logger, method_name)
                try:
                    # 同时传递位置参数和关键字参数
                    log_method(*args, **kwargs)
                except TypeError:
                    # 参数不匹配，尝试简化调用
                    try:
                        log_method(*args)  # 只传位置参数
                    except Exception as e:
                        print(f"日志记录错误 ({method_name}): {e}")
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
        """训练阶段 - 使用kwargs参数"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            start_time = time.time()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()

            # 计算梯度范数（用于监控）
            total_norm = self._compute_gradient_norm()

            self.optimizer.step()

            total_loss += loss.item()
            use_time = time.time() - start_time

            # 增加kwargs传递信息
            self._log('log_loss', 'train',
                    self.current_epoch, batch_idx, len(self.train_loader),loss.item(),
                    耗时=f"{use_time:.2f}s",
                    梯度范数=f"{total_norm:.4f}",
                    设备=str(self.device),
                    主要评估=f"{self.best_metric}")

            # 可视化
            if batch_idx % self.interval_visualize == 0:
                self.visualizer.plot(data[0], target[0], output[0],
                                     f'{self.image_dir}/epoch_{self.current_epoch}_batch_{batch_idx}.png', (8, 8))

            if not self.training:
                break

        # 返回整个epoch的平均训练损失
        avg_train_loss = total_loss / len(self.train_loader)
        return avg_train_loss

    def _validate(self):
        """验证阶段 - 计算所有评估指标"""
        self.model.eval()
        all_outputs = []
        all_targets = []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                start_time = time.time()
                # 同时移动data和target到设备
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())

                # 记录验证损失和耗时等信息
                val_loss = self.loss_fn(output, target).item()
                total_val_loss += val_loss
                use_time = time.time() - start_time
                self._log('log_loss', 'val',
                        self.current_epoch, batch_idx, len(self.val_loader), val_loss,
                        耗时=f"{use_time:.2f}s",
                        批次大小=f"{data.size(0)}")

        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(self.val_loader)

        # 合并计算评估指标，不包含损失值
        eval_metrics = {}
        if self.metric_evaluator:
            combined_output = torch.cat(all_outputs, dim=0)
            combined_target = torch.cat(all_targets, dim=0)
            eval_metrics = self.metric_evaluator.evaluate(combined_output, combined_target)

            # 记录评估指标
            self._log('log_metrics', self.current_epoch, eval_metrics,
                    学习率=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    验证样本总数=f"{combined_output.size(0)}",
                    最佳指标=f"{self.best_metric:.4f}",)

        # 返回验证损失和所有指标
        return avg_val_loss, eval_metrics

    def _compute_gradient_norm(self) -> float:
        """计算梯度L2范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2范数
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _should_save_model(self, loss_metrics: Dict[str, float], eval_metrics: Dict[str, float]) -> bool:
        """判断是否应该保存模型"""

        # 1. 首先检查是否有改进
        if not self._is_improvement(loss_metrics):
            return False

        # 2. 如果启用了质量检查，返回质量检查结果；否则直接返回True
        return self._check_quality_issue(eval_metrics) if self.enable_quality_checks else True

    def _is_improvement(self, loss_metrics: Dict[str, float]) -> bool:
        """有改进的标准：验证阶段损失值更小"""
        val_loss = loss_metrics['val_loss']
        return val_loss < self.best_metric

    def _check_quality_issue(self, eval_metrics: Dict[str, float]) -> bool:
        """质量检查 - 严格的多指标检查"""

        # 检查1: 基础数值有效性
        if not self._check_numerical_validity(eval_metrics):
            return False

        # 检查2: 关键指标合理性
        if not self._check_key_metrics_reasonable(eval_metrics):
            return False

        # 检查3: 指标间一致性
        if not self._check_metrics_consistency(eval_metrics):
            return False

        # 检查4: 任务特定检查
        if not self._check_task_specific_rules(eval_metrics):
            return False

        return True

    def _save_better_model(self, epoch: int, loss_metrics: Dict[str, float]):
        """保存最佳模型 - 只关注模型优化"""
        self.best_metric = loss_metrics['val_loss']

        # 保存模型
        self.model_manager.save(
            self.model, self.optimizer, epoch,
            'best_model.pth',
            best_metric=self.best_metric
        )

        # 记录保存信息
        metric_info = f"{self.loss_id} | val_loss: {self.best_metric:.4f}"
        # 考虑是否要再加train_loss？？
        self._log('log_time', f"💾 保存最佳模型 ({metric_info})")

    def _check_early_stop(self) -> bool:
        """检查是否应该早停"""
        self.early_stop_counter += 1
        if self.early_stop_counter >= self.patience:
            self._log('log_time', f"🛑 早停触发，连续{self.patience}个epoch未改善")
            return True
        return False

    def _update_learning_rate(self, val_loss: float):
        """更新学习率 - 使用验证损失"""
        if not math.isnan(val_loss) and val_loss != float('inf'):
            self.scheduler.step(val_loss)

            # 记录学习率变化（可选）
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log('log_time', f"📉 学习率现为: {current_lr:.2e}")

    def _save_epoch_results(self, loss_metrics: Dict[str, float], eval_metrics: Dict[str, float],
                        epoch_duration: float):
        """保存epoch训练结果 - 结构化数据"""
        epoch_data = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'train_loss': loss_metrics['train_loss'],
            'val_loss': loss_metrics['val_loss'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epoch_duration': epoch_duration,
            'gradient_norm': self._compute_gradient_norm(),
            'eval_metrics': eval_metrics,
            'best_metric': self.best_metric,
        }

        # 使用结果管理器保存
        self.result_manager.save_epoch_result(epoch_data)

        # 同时记录到日志
        self._log('log_time',
                 f"📊 Epoch {self.current_epoch} 结果已保存 | "
                 f"Train: {loss_metrics['train_loss']:.4f} | Val: {loss_metrics['val_loss']:.4f} | ")

    def _evaluate_training_progress(self, epoch: int,
            loss_metrics: Dict[str, float], eval_metrics: Dict[str, float]) -> Dict[str, Any]:
        """评估训练进度 - 只收集状态信息，不执行操作"""

        progress_info = {
            'epoch': epoch,
            'should_update_lr': self.training_actions.get('update_lr_every_epoch', True),
            'should_save_model': self._should_save_model(loss_metrics, eval_metrics),
            'should_early_stop': False,
        }

        # 检查早停
        if not progress_info['should_save_model'] and self.training_actions.get('enable_early_stop'):
            progress_info['should_early_stop'] = self._check_early_stop()

        return progress_info

    def _execute_training_actions(self, progress_info: Dict[str, Any],
                loss_metrics: Dict[str, float], eval_metrics: Dict[str, float], epoch_duration: float):
        """配置驱动的训练动作执行"""

        # 1. 更新学习率
        if progress_info['should_update_lr']:
            self._update_learning_rate(loss_metrics['val_loss'])

        # 2. 保存最佳模型（模型优化策略）
        if progress_info['should_save_model']:
            self._save_better_model(
                progress_info['epoch'],
                loss_metrics
            )
            self.early_stop_counter = 0

        # 3. 保存定期结果档案（结果分析策略）
        if self.training_actions.get('should_save_checkpoint', True):
            # 定期保存检查点
            if progress_info['epoch'] % self.training_actions.get('save_checkpoint_interval') == 0:
                # 保存训练结果数据
                self._save_epoch_results(loss_metrics, eval_metrics, epoch_duration)

        # 4. 处理早停
        if progress_info.get('should_early_stop', False):
            self._log('log_time', f"🛑 早停触发，连续{self.patience}个epoch未改善")

    def start_train(self):
        """开始训练 - 使用验证指标选择最佳模型"""
        self.train_loader = self.config['train_loader']
        self.val_loader = self.config['val_loader']
        self.training = True

        # 记录训练开始时间
        total_start_time = time.time()
        self._log('log_time', "🚀 训练开始!", 总epoch数=f"{self.config['epochs']}")

        for epoch in range(self.current_epoch, self.config['epochs'] + 1):
            self.current_epoch = epoch
            loss_metrics = {}

            # 记录epoch开始时间
            epoch_start_time = time.time()

            # 训练阶段
            loss_metrics['train_loss'] = self._train_epoch()

            # 验证阶段
            val_loss, eval_metrics = self._validate()
            loss_metrics['val_loss'] = val_loss

            # 记录epoch总耗时
            epoch_duration = time.time() - epoch_start_time
            self._log('log_time', f"Epoch {epoch} 总耗时: {epoch_duration:.2f}s")

            # 评估训练进度
            progress_info = self._evaluate_training_progress(epoch, loss_metrics, eval_metrics)
            # 并执行相应操作
            self._execute_training_actions(progress_info, loss_metrics, eval_metrics, epoch_duration)

            if progress_info.get('should_early_stop', False):
                break

            if not self.training:
                break

        # 记录训练总耗时
        total_time = time.time() - total_start_time
        self._log('log_time', f"训练完成! 总耗时: {total_time:.2f}s")

        # 保存最终模型和结果
        self._save_final_results(total_time)

    def _save_final_results(self, total_time: float):
        """保存最终训练结果"""
        final_data = {
            'final_epoch': self.current_epoch,
            'total_training_time': total_time,
            'best_metric': self.best_metric,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'training_completed': True,
            'completion_time': datetime.now().isoformat()
        }

        # 保存到JSON
        final_file = os.path.join(self.config['model_dir'], "final_training_summary.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        self._log('log_time', f"📄 最终训练摘要已保存: {final_file}")

    def save(self):
        """保存检查点 - 结果分析策略"""
        filename = f'checkpoint_epoch_{self.current_epoch}.pth'
        self.model_manager.save(self.model, self.optimizer, self.current_epoch, filename)
        msg = f'save model to {filename}'
        print(msg)
        self._log('log_time', f"💾 检查点已保存: {filename}")
        return msg

    def stop(self):
        """停止训练并保存当前状态"""
        self.save()
        self.training = False
        self._log('log_time', "训练停止")
