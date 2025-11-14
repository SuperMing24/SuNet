import nibabel as nib
import numpy as np
import torch
from utils.segutil import SegDataset
from typing import Dict, List, Tuple, Any, Optional
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置 Matplotlib 使用 Agg 后端
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
import json
import os


class DataInspector:
    """
    数据检查器：负责数据的统计分析和检查
    """

    @staticmethod
    def get_shape_info(data: np.ndarray, name: str = "data") -> Dict[str, Any]:
        """获取形状信息的统一方法"""
        return {
            f"{name}_shape": data.shape,
            f"{name}_ndim": data.ndim,
            f"{name}_dtype": str(data.dtype),
            f"{name}_size": data.size
        }

    @staticmethod
    def get_image_statistical(data: np.ndarray, name: str = "image") -> Dict[str, Any]:
        """获取统计信息的统一方法"""
        return {
            f"{name}_min": float(data.min()),
            f"{name}_max": float(data.max()),
            f"{name}_mean": float(data.mean()),
            f"{name}_std": float(data.std()),
            f"{name}_median": float(np.median(data)),
            f"{name}_has_nan": np.isnan(data).any(),
            f"{name}_has_inf": np.isinf(data).any()
        }

    @staticmethod
    def get_mask_distribution(mask: np.ndarray, name: str = "mask") -> Dict[str, Any]:
        """获取掩码分布信息"""
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size

        distribution = {}
        for val, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            distribution[f"{name}_class_{int(val)}_pixels"] = int(count)
            distribution[f"{name}_class_{int(val)}_percentage"] = float(percentage)

        return distribution

    @staticmethod
    def visualize_sample(image: np.ndarray, mask: np.ndarray, save_path: Optional[str] = None):
        """可视化样本数据"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示图像
        if image.ndim == 3 and image.shape[2] in [1, 3]:
            display_image = image if image.shape[2] == 3 else image[:, :, 0]
            axes[0].imshow(display_image, cmap='gray')
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title('输入图像')
        axes[0].axis('off')

        # 显示掩码
        axes[1].imshow(mask.squeeze(), cmap='jet')
        axes[1].set_title('真实掩码')
        axes[1].axis('off')

        # 显示叠加效果
        axes[2].imshow(display_image, cmap='gray')
        axes[2].imshow(mask.squeeze(), cmap='jet', alpha=0.5)
        axes[2].set_title('图像+掩码叠加')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 样本可视化已保存: {save_path}")

        plt.close()


class DataReporter:
    """数据报告生成器 - 负责结构化输出检查结果"""

    @staticmethod
    def print_complete_report(dataset_stats: Dict[str, Any], show_individual_samples = False):
        """打印数据集统计信息"""
        if not dataset_stats:
            print("❌ 没有可用的统计数据")
            return

        print("\n" + "=" * 80)
        print("📊 NIfTI 数据集检查报告")
        print("=" * 80)

        # 1. 整体概览
        DataReporter.print_overview(dataset_stats)

        # 2. 文件级别统计摘要
        DataReporter.print_file_stats(dataset_stats)

        DataReporter.print_anomaly_check(dataset_stats)

        # 3. 总体样本统计
        DataReporter.print_statistical_summary(dataset_stats)

        # 4. 单个样本统计
        if show_individual_samples:
            DataReporter._print_individual_sample_stats(dataset_stats)

        print("\n" + "=" * 80)
        print("🎉 数据检查报告生成完成!")
        print("=" * 80)


    @staticmethod
    def print_overview(dataset_stats: Dict[str, Any]):
        # 文件基本信息概览
        total_files = dataset_stats.get('total_files', 0)
        total_samples = dataset_stats.get('total_samples', 0)

        print(f"\n📁 数据集基本信息:")
        print(f"  ├── 文件对数量: {total_files}")
        print(f"  ├── 总样本数: {total_samples}")

        if total_files > 0 and total_samples > 0:
            avg_samples_per_file = total_samples / total_files
            print(f"  └── 平均每文件样本数: {avg_samples_per_file:.1f}")

    @staticmethod
    def print_file_stats(dataset_stats: Dict[str, Any], max_files: int = 10):
        """打印文件级别统计信息"""
        file_pairs = dataset_stats.get('file_pairs', [])
        if not file_pairs:
            return

        print(f"\n📄 文件统计 (显示前 {min(max_files, len(file_pairs))} 个文件):")
        print("-" * 60)

        for i, file_info in enumerate(file_pairs[:max_files]):
            print(f"\n🔹 文件对 {i + 1}:")
            print(f"   图像文件: {os.path.basename(file_info['image_file'])}")
            print(f"   掩码文件: {os.path.basename(file_info['mask_file'])}")

            # 形状信息
            image_shape = file_info['image_shape']
            mask_shape = file_info['mask_shape']
            print(f"   原始图像形状: {image_shape['原始图像_shape']}")
            print(f"   原始掩码形状: {mask_shape['原始掩码_shape']}")

    @staticmethod
    def print_statistical_summary(dataset_stats: Dict[str, Any]):
        """打印统计摘要"""
        file_pairs = dataset_stats.get('file_pairs', [])
        if not file_pairs:
            return

        print(f"\n📈 统计摘要 (基于 {len(file_pairs)} 个文件):")
        print("-" * 60)

        # 收集所有统计信息
        all_raw_image_stats = []
        all_processed_image_stats = []
        all_raw_mask_stats = []
        all_processed_mask_stats = []

        for file_info in file_pairs:
            all_raw_image_stats.append(file_info['image_raw_stats'])
            all_processed_image_stats.append(file_info['image_processed_stats'])
            all_raw_mask_stats.append(file_info['mask_raw_stats'])
            all_processed_mask_stats.append(file_info['mask_processed_stats'])

        # 图像统计
        print("\n🎯 图像数值统计:")
        DataReporter._print_image_statistical(
            all_raw_image_stats, all_processed_image_stats, "图像"
        )

        # 掩码统计
        print("\n🎯 掩码分布统计:")
        DataReporter._print_mask_statical(all_raw_mask_stats, all_processed_mask_stats)

    @staticmethod
    def _print_image_statistical(raw_image_stats_list: List[Dict], processed_image_stats_list: List[Dict], data_type: str):
        """打印统计对比信息"""
        if not raw_image_stats_list or not processed_image_stats_list:
            return

        # 原始数据统计
        raw_means = [stats[f'原始{data_type}_mean'] for stats in raw_image_stats_list]
        raw_stds = [stats[f'原始{data_type}_std'] for stats in raw_image_stats_list]
        raw_mins = [stats[f'原始{data_type}_min'] for stats in raw_image_stats_list]
        raw_maxs = [stats[f'原始{data_type}_max'] for stats in raw_image_stats_list]

        # 处理后数据统计
        processed_means = [stats[f'归一化后{data_type}_mean'] for stats in processed_image_stats_list]
        processed_stds = [stats[f'归一化后{data_type}_std'] for stats in processed_image_stats_list]
        processed_mins = [stats[f'归一化后{data_type}_min'] for stats in processed_image_stats_list]
        processed_maxs = [stats[f'归一化后{data_type}_max'] for stats in processed_image_stats_list]

        print(f"  原始数据:")
        print(f"    ├── 均值范围: [{min(raw_means):.4f}, {max(raw_means):.4f}]")
        print(f"    ├── 标准差范围: [{min(raw_stds):.4f}, {max(raw_stds):.4f}]")
        print(f"    ├── 最小值范围: [{min(raw_mins):.4f}, {max(raw_mins):.4f}]")
        print(f"    └── 最大值范围: [{min(raw_maxs):.4f}, {max(raw_maxs):.4f}]")

        print(f"  归一化后:")
        print(f"    ├── 均值范围: [{min(processed_means):.4f}, {max(processed_means):.4f}]")
        print(f"    ├── 标准差范围: [{min(processed_stds):.4f}, {max(processed_stds):.4f}]")
        print(f"    ├── 最小值范围: [{min(processed_mins):.4f}, {max(processed_mins):.4f}]")
        print(f"    └── 最大值范围: [{min(processed_maxs):.4f}, {max(processed_maxs):.4f}]")

        print(f"  总平均量与变化趋势：")
        print(f"    ├── 均值: [{np.mean(raw_means)} ——> {np.mean(processed_means)}] "
              f"{'↓减小' if np.mean(processed_means) < np.mean(raw_means) else '↑增大'}")
        print(f"    ├── 标准差: [{np.mean(raw_stds)} ——> {np.mean(processed_stds)}] "
              f"{'↓减小' if np.mean(processed_stds) < np.mean(raw_stds) else '↑增大'}")
        print(f"    ├── 最小值: [{np.mean(raw_mins)} ——> {np.mean(processed_mins)}] "
              f"{'↓减小' if np.mean(processed_mins) < np.mean(raw_mins) else '↑增大'}")
        print(f"    ├── 最大值: [{np.mean(raw_maxs)} ——> {np.mean(processed_maxs)}] "
              f"{'↓减小' if np.mean(processed_maxs) < np.mean(raw_maxs) else '↑增大'}")

    @staticmethod
    def _print_mask_statical(raw_mask_stats_list: List[Dict], processed_mask_stats_list: List[Dict]):
        """打印掩码分布摘要"""
        # 收集前景比例
        raw_foreground_ratios = []
        processed_foreground_ratios = []

        for raw_stats, processed_stats in zip(raw_mask_stats_list, processed_mask_stats_list):
            # 类别0是前景
            if '原始掩码_class_0_percentage' in raw_stats:
                raw_foreground_ratios.append(raw_stats['原始掩码_class_0_percentage'])
            if '归一化后掩码_class_0_percentage' in processed_stats:
                processed_foreground_ratios.append(processed_stats['归一化后掩码_class_0_percentage'])

        if raw_foreground_ratios:
            print(f"  前景像素比例:")
            print(f"    ├── 原始数据: {np.min(raw_foreground_ratios):5.2f}%~{np.max(raw_foreground_ratios):5.2f}% (平均: {np.mean(raw_foreground_ratios):5.2f}%)")
            print(f"    └── 处理后数据: {np.min(processed_foreground_ratios):5.2f}%~{np.max(processed_foreground_ratios):5.2f}% (平均: {np.mean(processed_foreground_ratios):5.2f}%)")


    @staticmethod
    def _print_individual_sample_stats(dataset_stats: Dict[str, Any]):
        """打印单个样本的详细统计"""
        file_pairs = dataset_stats.get('file_pairs', [])
        if not file_pairs:
            return

        print(f"\n🔍 单个样本详细统计:")
        print("=" * 80)

        sample_count = 0
        for file_idx, file_info in enumerate(file_pairs):
            image_file = os.path.basename(file_info['image_file'])

            # 只显示前几个样本的详细信息（避免输出过长）
            if sample_count >= 10:  # 最多显示10个样本的详细信息
                print(f"... 还有 {len(file_pairs) - 10} 个文件的统计信息未显示")
                break

            print(f"\n📁 文件 {file_idx + 1}: {image_file}")
            print("-" * 50)

            # 原始数据统计
            print("🔹 原始数据:")
            raw_image_stats = file_info['image_raw_stats']
            print(f"   形状: {file_info['image_shape']['原始图像_shape']}")
            print(f"   范围: [{raw_image_stats['原始图像_min']:6.4f}, {raw_image_stats['原始图像_max']:6.4f}]")
            print(f"   均值: {raw_image_stats['原始图像_mean']:6.4f}")
            print(f"   标准差: {raw_image_stats['原始图像_std']:6.4f}")

            # 处理后数据统计
            print("🔹 处理后数据:")
            processed_image_stats = file_info['image_processed_stats']
            print(f"   范围: [{processed_image_stats['归一化后图像_min']:6.4f}, {processed_image_stats['归一化后图像_max']:6.4f}]")
            print(f"   均值: {processed_image_stats['归一化后图像_mean']:6.4f}")
            print(f"   标准差: {processed_image_stats['归一化后图像_std']:6.4f}")

            # 掩码统计
            print("🔹 掩码分布:")
            raw_mask_stats = file_info['mask_raw_stats']
            processed_mask_stats = file_info['mask_processed_stats']

            # 显示各类别比例
            for k, v in raw_mask_stats.items():
                if 'percentage' in k:
                    class_name = k.split('_')[2]  # 提取类别名
                    print(f"   类别{class_name}: {v:5.2f}% (原始)")

            for k, v in processed_mask_stats.items():
                if 'percentage' in k:
                    class_name = k.split('_')[2]
                    print(f"   类别{class_name}: {v:5.2f}% (处理后)")

            sample_count += 1

    @staticmethod
    def print_anomaly_check(dataset_stats: Dict[str, Any]):
        """打印异常检查结果"""
        file_pairs = dataset_stats.get('file_pairs', [])
        if not file_pairs:
            return

        print(f"\n⚠️  异常检查:")
        print("-" * 60)

        has_nan_files = []
        has_inf_files = []

        for i, file_info in enumerate(file_pairs):
            raw_stats = file_info['image_raw_stats']
            processed_stats = file_info['image_processed_stats']

            if raw_stats.get('原始图像_has_nan', False) or processed_stats.get('归一化后图像_has_nan', False):
                has_nan_files.append(os.path.basename(file_info['image_file']))

            if raw_stats.get('原始图像_has_inf', False) or processed_stats.get('归一化后图像_has_inf', False):
                has_inf_files.append(os.path.basename(file_info['image_file']))

        if has_nan_files:
            print(f"❌ 发现 NaN 值的文件: {has_nan_files}")
        else:
            print("✅ 未发现 NaN 值")

        if has_inf_files:
            print(f"❌ 发现 Inf 值的文件: {has_inf_files}")
        else:
            print("✅ 未发现 Inf 值")



class NIIDataProcessor:
    """医学图像数据处理器 - 支持动态切片数"""

    @staticmethod
    def trans_datatype_float32(data):
        return data.astype(np.float32)

    @staticmethod
    def concatenated_multislice(volume, center_slice, num_slices=3):
        """
        提取多个相邻切片，用于跨切片上下文

        参数:
            volume: 3D体积数据 [H, W, D]
            center_slice: 中心切片索引
            num_slices: 提取的切片数量（必须为奇数）

        返回:
            多切片数据 [H, W, num_slices]
        """
        if num_slices % 2 == 0:
            raise ValueError("num_slices 必须为奇数以保证对称")

        half = num_slices // 2
        total_slices = volume.shape[2]
        slices = []

        # 计算需要提取的切片索引范围
        target_indices = []
        for offset in range(-half, half + 1):
            target_idx = center_slice + offset
            # 边界处理：超出范围的索引映射到最近的边界
            if target_idx < 0:
                target_idx = 0
            elif target_idx >= total_slices:
                target_idx = total_slices - 1
            target_indices.append(target_idx)

        # 提取切片
        for idx in target_indices:
            slices.append(volume[:, :, idx])

        return np.stack(slices, axis=-1)  # [H, W, num_slices]

    @staticmethod
    def nii_normalize(images, masks):

        # 对每个切片的通道进行归一化
        means = images.mean(axis=(0, 1))  # 每个切片的均值
        stds = images.std(axis=(0, 1))    # 每个切片的标准差

        processed_images = (images - means) / (stds + 1e-8)
        processed_masks = (masks > 0).astype(np.float32)

        return processed_images, processed_masks


class CatNiiDataset(SegDataset):
    """
    CatNiiDataset - 保持跨切片处理并兼容transforms
    """

    def __init__(self, img_dir: str, mask_dir: str, transforms: List = [],
                 check: str = 'none', num_slices: int = 3, enable_inspection: bool = False):
        """
        参数:
            img_dir: 图像目录
            mask_dir: 掩码目录
            transforms: 数据转换列表
            check: 检查模式
            num_slices: 跨切片数量
            enable_inspection: 是否启用数据检查
        """
        self.num_slices = num_slices
        self.enable_inspection = enable_inspection
        self.dataset_stats = {}  # 数据集级别统计信息

        # 调用父类初始化
        super().__init__(img_dir, mask_dir, transforms, check)

        # 初始化完成后进行数据检查
        # if enable_inspection:
        #     self._perform_dataset_analysis()

    def _resolve_ids(self, img_dir, mask_dir, check='none'):
        """文件解析，加载NIfTI数据"""
        image_mask_pairs = super()._resolve_ids(img_dir, mask_dir, check)
        self.load_datas = {}

        # 记录数据集基本信息
        if self.enable_inspection:
            self.dataset_stats['total_files'] = len(image_mask_pairs)
            self.dataset_stats['file_pairs'] = []

        # 预加载所有NIfTI文件，保持原始float32精度
        for image_file, mask_file in image_mask_pairs:
            # 加载图像数据
            image_raw_data = nib.load(image_file).get_fdata()
            mask_raw_data = nib.load(mask_file).get_fdata()

            # 记录文件级别信息 - 原始加载数据
            file_info = {
                'image_file': image_file,
                'mask_file': mask_file
            }

            # 只有在启用检查时才填充统计信息
            if self.enable_inspection:
                # 原始数据统计
                file_info.update({
                    'image_shape': DataInspector.get_shape_info(image_raw_data, "原始图像"),
                    'mask_shape': DataInspector.get_shape_info(mask_raw_data, "原始掩码"),
                    'image_raw_stats': DataInspector.get_image_statistical(image_raw_data, "原始图像"),
                    'mask_raw_stats': DataInspector.get_mask_distribution(mask_raw_data, "原始掩码")
                })

            # 基础类型转换处理
            image_processed_data = NIIDataProcessor.trans_datatype_float32(image_raw_data)
            mask_processed_data = NIIDataProcessor.trans_datatype_float32(mask_raw_data)

            # nii数据的归一化处理
            image_processed_data, mask_processed_data = NIIDataProcessor.nii_normalize(image_processed_data, mask_processed_data)

            # 收录数据
            self.load_datas[image_file] = image_processed_data
            self.load_datas[mask_file] = mask_processed_data

            # 添加处理后信息
            if self.enable_inspection:
                processed_info = {
                    'image_processed_stats': DataInspector.get_image_statistical(image_processed_data, "归一化后图像"),
                    'mask_processed_stats': DataInspector.get_mask_distribution(mask_processed_data, "归一化后掩码")
                }
                file_info.update(processed_info)  # 将处理后的信息合并到原有字典      ### ？提示“局部变量 'file_info' 可能在赋值前引用 ”，这个设计其实感觉不太舒服，但如果不在前后都加“self.enable_inspection”又不对，待究
                self.dataset_stats['file_pairs'].append(file_info)


        # 创建样本索引对 (文件, 文件, 切片层)
        id_pairs = []

        for image_file, mask_file in image_mask_pairs:
            if self.load_datas[image_file].shape[2] != self.load_datas[mask_file].shape[2]:
                print(f"⚠️ 警告: {image_file} 和 {mask_file} 切片数量不匹配")
                num_layers = min(self.load_datas[image_file].shape[2], self.load_datas[mask_file].shape[2])
            else:
                num_layers = self.load_datas[image_file].shape[2]
            for layer in range(num_layers):
                id_pairs.append((image_file, mask_file, layer))
        self.dataset_stats['total_samples'] = len(id_pairs)
        return id_pairs

    def _load_datas(self, id):
        """重写数据加载，保持跨切片处理"""
        image_file, mask_file, layer = id
        image_data = self.load_datas[image_file]
        mask_data = self.load_datas[mask_file]

        # 提取多个相邻切片用于上下文信息
        processed_images = NIIDataProcessor.concatenated_multislice(
            image_data, layer, self.num_slices
        )
        # 掩码只使用中心切片，但保持相同的处理接口
        processed_masks = mask_data[:, :, [layer]]  # [H, W, 1]

        return processed_images, processed_masks

    def get_dataset_stats(self):
        """获取数据集统计信息"""
        stats = self.dataset_stats.copy()
        if stats:
            return stats
        else:
            return None