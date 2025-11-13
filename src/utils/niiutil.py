import nibabel as nib
import numpy as np
import torch
from utils.segutil import SegDataset

class NIIDataProcessor:
    """医学图像数据处理器 - 支持动态切片数"""

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
    """CatNiiDataset，保持跨切片处理并兼容transforms"""

    def __init__(self, img_dir, mask_dir, transforms=[], check='none', num_slices=3):
        """
        参数:
            num_slices: 跨切片数量，默认为3个切片
        """
        self.num_slices = num_slices
        self.processor = NIIDataProcessor()
        super().__init__(img_dir, mask_dir, transforms, check)

    def _resolve_ids(self, img_dir, mask_dir, check='none'):
        """重写文件解析，加载NIfTI数据"""
        image_pairs = super()._resolve_ids(img_dir, mask_dir, check)
        self.load_datas = {}

        # 预加载所有NIfTI文件，保持原始float32精度
        for image_file, mask_file in image_pairs:
            self.load_datas[image_file] = nib.load(image_file).get_fdata().astype(np.float32)
            self.load_datas[mask_file] = nib.load(mask_file).get_fdata().astype(np.float32)

        id_pairs = []
        for image_file, mask_file in image_pairs:
            for layer in range(self.load_datas[image_file].shape[2]):
                id_pairs.append((image_file, mask_file, layer))
        return id_pairs

    def _load_datas(self, id):
        """重写数据加载，保持跨切片处理"""
        image_file, mask_file, layer = id
        image_data = self.load_datas[image_file]
        mask_data = self.load_datas[mask_file]

        # 提取多个相邻切片用于上下文信息
        processed_images = self.processor.concatenated_multislice(
            image_data, layer, self.num_slices
        )
        # 掩码只使用中心切片，但保持相同的处理接口
        processed_masks = mask_data[:, :, [layer]]  # [H, W, 1]

        # nii数据的归一化处理
        processed_images, processed_masks = self.processor.nii_normalize(processed_images, processed_masks)

        return processed_images, processed_masks
