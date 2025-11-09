import nibabel as nib
import numpy as np
import torch
from PIL import Image
from utils.segutil import SegDataset, tongbu_trans
from scipy.ndimage import zoom

class MedicalDataProcessor:
    """医学图像数据处理器 - 保持跨切片处理"""

    @staticmethod
    def extract_multislice_data(volume, center_slice, num_slices=3):
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
        slices = []

        for i in range(-half, half + 1):
            slice_idx = max(0, min(center_slice + i, volume.shape[2] - 1))
            slices.append(volume[:, :, slice_idx])

        return np.stack(slices, axis=-1)  # [H, W, num_slices]

    @staticmethod
    def numpy_to_pil_compatible(array, is_mask=False):
        """
        将numpy数组转换为PIL兼容的格式，同时保持跨切片信息

        参数:
            array: numpy数组 [H, W, C]
            is_mask: 是否为掩码数据

        返回:
            PIL图像对象
        """
        # 确保数据在0-255范围内
        if array.dtype != np.uint8:
            if is_mask:
                # 对于掩码，直接二值化
                array = (array > 0).astype(np.uint8) * 255
            else:
                # 对于图像，进行归一化到0-255
                array = (array - array.min()) / (array.max() - array.min() + 1e-8) * 255
                array = array.astype(np.uint8)

        # 处理多通道情况（跨切片）
        if array.shape[2] == 1:
            return Image.fromarray(array[:, :, 0], 'L')
        elif array.shape[2] == 3:
            return Image.fromarray(array, 'RGB')
        else:
            # 如果切片数不是1或3，取前3个通道或进行其他处理
            # 这里可以根据需要调整
            if array.shape[2] > 3:
                array = array[:, :, :3]  # 取前3个切片
            return Image.fromarray(array, 'RGB')

    @staticmethod
    def pil_to_medical_tensor(pil_image, original_dtype=np.float32):
        """
        将PIL图像转换回医学图像格式的Tensor

        参数:
            pil_image: PIL图像对象
            original_dtype: 原始数据类型

        返回:
            PyTorch Tensor [C, H, W]
        """
            # 🔧 关键修复：检查输入类型
        if isinstance(pil_image, torch.Tensor):
            # 如果已经是Tensor，确保维度顺序正确
            if pil_image.dim() == 3:
                # [C, H, W] 顺序，直接返回
                return pil_image
            elif pil_image.dim() == 4:
                # [B, C, H, W] 顺序，去掉batch维度
                return pil_image.squeeze(0)
            else:
                # 其他情况，保持原样
                return pil_image
        # 转换为numpy数组
        array = np.array(pil_image)

        # 调整维度顺序
        if array.ndim == 2:  # 灰度图像
            array = array[:, :, np.newaxis]  # [H, W] → [H, W, 1]

        # 转换回原始数值范围（近似）
        # 注意：这里会有精度损失，但保持了跨切片信息
        if original_dtype == np.float32:
            array = array.astype(np.float32) / 255.0

        # 🔧 关键修复：确保正确的维度顺序 [C, H, W]
        if array.shape[2] in [1, 3]:  # 单通道或3通道
            tensor = torch.from_numpy(array.transpose(2, 0, 1))  # [H, W, C] → [C, H, W]
        else:
            # 其他通道数，保持原样
            tensor = torch.from_numpy(array)

        return tensor

class Cat25Dataset(SegDataset):
    """修改后的Cat25Dataset，保持跨切片处理并兼容transforms"""

    def __init__(self, img_dir, mask_dir, transforms=[], check='none', num_slices=3):
        """
        参数:
            num_slices: 跨切片数量，默认为3个切片
        """
        self.num_slices = num_slices
        self.processor = MedicalDataProcessor()
        super().__init__(img_dir, mask_dir, transforms, check)

    def _resolve_ids(self, img_dir, mask_dir, check):
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
        processed_images = self.processor.extract_multislice_data(
            image_data, layer, self.num_slices
        )

        # 对每个切片的通道进行归一化
        means = processed_images.mean(axis=(0, 1))  # 每个切片的均值
        stds = processed_images.std(axis=(0, 1))    # 每个切片的标准差
        processed_images = (processed_images - means) / (stds + 1e-8)

        # 掩码只使用中心切片，但保持相同的处理接口
        processed_masks = mask_data[:, :, [layer]]  # [H, W, 1]
        processed_masks = (processed_masks > 0).astype(np.float32)

        # 转换为PIL兼容格式以支持transforms
        image_pil = self.processor.numpy_to_pil_compatible(processed_images, is_mask=False)
        mask_pil = self.processor.numpy_to_pil_compatible(processed_masks, is_mask=True)

        return image_pil, mask_pil

    def __getitem__(self, idx):
        """重写getitem以在转换后恢复医学图像格式"""
        id = self.image_mask_pairs[idx]
        image_pil, mask_pil = self._load_datas(id)

        # 应用标准的torchvision transforms
        for transform, tb in self.transforms:
            if tb:
                image_pil, mask_pil = tongbu_trans(transform, image_pil, mask_pil)
            else:
                image_pil = transform(image_pil)
                mask_pil = transform(mask_pil)

        # 转换回医学图像格式的Tensor
        image_tensor = self.processor.pil_to_medical_tensor(image_pil, np.float32)
        mask_tensor = self.processor.pil_to_medical_tensor(mask_pil, np.float32)

        return image_tensor, mask_tensor
