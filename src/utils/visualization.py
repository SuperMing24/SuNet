from typing import List, Tuple, Optional, Union
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置 Matplotlib 使用 Agg 后端
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class SegmentationVisualizer:
    """统一的分割结果可视化器"""
    
    def __init__(self, task_type: str = 'binary', class_colors: Optional[List[Tuple]] = None):
        """
        参数:
            task_type: 'binary' 或 'multiclass'
            class_colors: 多分类时的类别颜色，格式为 [(R,G,B), ...]
        """
        self.task_type = task_type
        self.class_colors = class_colors or self._get_default_colors()
        
    def _get_default_colors(self):
        """获取默认颜色映射"""
        return [
            (255, 0, 0),    # 红色 - 类别0
            (0, 255, 0),    # 绿色 - 类别1
            (0, 0, 255),    # 蓝色 - 类别2
            (255, 255, 0),  # 黄色 - 类别3
            (255, 0, 255),  # 品红 - 类别4
            (0, 255, 255),  # 青色 - 类别5
        ]
    
    def _prepare_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """准备输入图像显示"""
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)
        return image
    
    def _process_binary_prediction(self, pred_mask: torch.Tensor) -> np.ndarray:
        """处理二分类预测"""
        return (torch.sigmoid(pred_mask).squeeze().detach().cpu().numpy() > 0.5).astype(int)
    
    def _process_multiclass_prediction(self, pred_mask: torch.Tensor) -> np.ndarray:
        """处理多分类预测"""
        return pred_mask.argmax(dim=0).cpu().numpy()
    
    def _mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """将类别掩码转换为彩色图像"""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in enumerate(self.class_colors):
            if class_id < len(self.class_colors):
                colored[mask == class_id] = color
        return colored
    
    def _compute_error_map(self, true_mask: np.ndarray, pred_mask: np.ndarray, 
                          base_image: np.ndarray) -> np.ndarray:
        """计算误差图"""

        # 确保基础图像是3通道
        if base_image.ndim == 3 and base_image.shape[2] == 1:  # [H, W, 1] → [H, W, 3]
            base_image = np.repeat(base_image, 3, axis=2)

        error_map = base_image.copy()

        # 计算误差位置
        error_positions = true_mask != pred_mask

        error_map[error_positions] = [255, 0, 0]  # 错误区域标红
        return error_map
    
    def plot(self, image: torch.Tensor, true_mask: torch.Tensor, 
             pred_mask: torch.Tensor, save_path: str, size: Union[int, Tuple] = (5, 5)):
        """
        绘制分割结果对比图
        
        参数:
            image: 输入图像 [C, H, W]
            true_mask: 真实掩码 
            pred_mask: 预测掩码
            save_path: 保存路径
            size: 图像大小
        """
        # 准备图像
        prepared_image = self._prepare_image(image)
        true_mask_np = true_mask.squeeze().cpu().numpy()
        
        # 处理预测
        if self.task_type == 'binary':
            pred_mask_np = self._process_binary_prediction(pred_mask)
            true_display = true_mask_np
            pred_display = pred_mask_np
        else:
            pred_mask_np = self._process_multiclass_prediction(pred_mask)
            true_display = self._mask_to_color(true_mask_np)
            pred_display = self._mask_to_color(pred_mask_np)
        
        # 计算误差图
        error_map = self._compute_error_map(true_mask_np, pred_mask_np, prepared_image)
        
        # 创建子图
        size = (size * 4, size) if isinstance(size, int) else (4 * size[0], size[1])
        fig, axes = plt.subplots(1, 4, figsize=size,
                               subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False})
        
        # 绘制子图
        titles = ['Input Image', 'Ground Truth', 'Prediction', 'Error Map']
        displays = [prepared_image, true_display, pred_display, error_map]
        cmaps = [None, 'gray' if self.task_type == 'binary' else None, 
                'gray' if self.task_type == 'binary' else None, None]
        
        for ax, title, display, cmap in zip(axes, titles, displays, cmaps):
            ax.set_title(title)
            ax.imshow(display, cmap=cmap)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
