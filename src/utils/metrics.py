"""评估指标定义 - 专注性能评估"""
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseMetric(ABC):
    """评估指标基类"""

    @abstractmethod
    def compute(self, output: torch.Tensor, target: torch.Tensor) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class DiceScore(BaseMetric):
    """Dice系数评估 - 分割重叠度"""

    @property
    def name(self) -> str:
        return "dice"

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> float:
        pred = (torch.sigmoid(output) > 0.5).float()
        smooth = 1e-6
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + smooth) / (union + smooth).item()


class CLDiceScore(BaseMetric):
    """中心线Dice评估 - 血管结构匹配度"""

    @property
    def name(self) -> str:
        return "cldice"

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> float:
        pred = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
        true = target.cpu().numpy()

        batch_scores = []
        for i in range(pred.shape[0]):
            pred_center = self._skeletonize(pred[i, 0])
            true_center = self._skeletonize(true[i, 0])

            intersection = np.logical_and(pred_center, true_center).sum()
            union = pred_center.sum() + true_center.sum()

            if union == 0:
                batch_scores.append(1.0 if intersection == 0 else 0.0)
            else:
                batch_scores.append(2. * intersection / union)

        return float(np.mean(batch_scores))

    def _skeletonize(self, binary_image):
        """使用scikit-image的骨架化"""
        try:
            from skimage.morphology import skeletonize
            return skeletonize(binary_image > 0.5)
        except ImportError:
            # 回退到简化版本
            return binary_image > 0.5


class HausdorffDistance(BaseMetric):
    """Hausdorff距离 - 边界匹配精度"""

    @property
    def name(self) -> str:
        return "hausdorff"

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> float:
        pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
        true = target.cpu().numpy()

        batch_distances = []
        for i in range(pred.shape[0]):
            pred_coords = np.argwhere(pred[i, 0] > 0)
            true_coords = np.argwhere(true[i, 0] > 0)

            if len(pred_coords) == 0 or len(true_coords) == 0:
                batch_distances.append(np.inf)
                continue

            dist1 = directed_hausdorff(pred_coords, true_coords)[0]
            dist2 = directed_hausdorff(true_coords, pred_coords)[0]
            batch_distances.append(max(dist1, dist2))

        return float(np.mean([d for d in batch_distances if d != np.inf]))


class AccuracyScore(BaseMetric):
    """像素准确率 - 分类任务基础指标"""

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> float:
        pred = (torch.sigmoid(output) > 0.5).float()
        correct = (pred == target).float()
        return correct.sum().item() / target.numel()


class MetricEvaluator:
    """指标评估器 - 简洁的多指标计算"""

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    def evaluate(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """计算所有配置的指标"""
        results = {}
        for metric in self.metrics:
            try:
                results[metric.name] = metric.compute(output, target)
            except Exception as e:
                print(f"计算指标 {metric.name} 失败: {e}")
                results[metric.name] = 0.0
        return results


# 指标创建助手函数
def create_metrics(metric_names: List[str]) -> MetricEvaluator:
    """创建指标评估器 - 简洁的配置方式"""
    metric_classes = {
        'dice': DiceScore,
        'cldice': CLDiceScore,
        'hausdorff': HausdorffDistance,
        'accuracy': AccuracyScore
    }

    metrics = []
    for name in metric_names:
        if name in metric_classes:
            metrics.append(metric_classes[name]())
        else:
            print(f"警告: 未知指标 {name}")

    return MetricEvaluator(metrics)
