import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    img_types = ('.png', '.jpg', '.tif','.nii')

    def __init__(self, img_dir, mask_dir, transforms=[], check='none'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = [[t, False] if not isinstance(t, (list, tuple)) else t for t in transforms]
        self.check = check

        self.image_mask_pairs = self._resolve_ids(img_dir, mask_dir, check)

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        id = self.image_mask_pairs[idx]
        image, mask = self._load_datas(id)

        for transform, tb in self.transforms:
            if tb:
                image, mask = tongbu_trans(transform, image, mask)
            else:
                image = transform(image)
                mask = transform(mask)
        return image, mask

    def _resolve_ids(self, img_dir, mask_dir, check='none'):
        # 收集所有符合条件的图像文件
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(SegDataset.img_types)])

        image_mask_pairs = []
        missing_masks = []  # 用于收集找不到的掩码文件

        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            found = False

            # 循环检查所有可能的扩展名
            for ext in SegDataset.img_types:
                mask_file = base_name + ext
                mask_path = os.path.join(mask_dir, mask_file)

                if os.path.exists(mask_path):
                    image_mask_pairs.append((img_file, mask_file))
                    found = True
                    break  # 找到后停止检查

            if not found:
                if check == 'single':
                    raise FileNotFoundError(f"Missing mask for image file: {img_file}")
                missing_masks.append(img_file)

        # 处理缺失的掩码文件
        if missing_masks:
            if check == 'all':
                error_msg = "Missing masks for the following files:\n" + "\n".join(missing_masks)
                raise FileNotFoundError(error_msg)
            elif check == 'none':
                print("Warning: Some images are missing corresponding mask files. "
                      "These files have been skipped.")
            else:
                raise ValueError(f"Invalid value for 'check': {check}. Use 'single', 'all', or 'none'.")

        image_mask_pairs = [
            (os.path.join(self.img_dir, img), os.path.join(self.mask_dir, mask))
            for img, mask in image_mask_pairs
        ]

        return image_mask_pairs

    def _load_datas(self, id):
        img_file, mask_file = id
        image = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file).convert('L')
        return image, mask


def tongbu_trans(transform, image, mask):
    """
    对图像和掩码应用相同的变换。

    参数:
        transform: 要应用的变换（如 torchvision.transforms 中的变换）
        image (torch.Tensor): 输入图像数据，可能的形状为 [C, H, W] 或 [H, W, C]
        mask (torch.Tensor): 输入掩码数据，可能的形状为 [C, H, W]、[H, W] 或 [H, W, C]

    返回:
        tuple: 包含转换后的图像和掩码
    """

    # 如果还不是张量的形式
    if not hasattr(image, 'shape'):
        return transform(image), transform(mask)

    # 记录原始形状
    original_image_shape = image.shape
    original_mask_shape = mask.shape

    # 确保图像和掩码是三维张量，并调整通道位置到第一维
    def _prepare_tensor(x):
        if x.ndim == 2:
            x = x.unsqueeze(0)  # 如果是二维数组，则增加通道维度
        elif x.ndim == 3:
            if x.shape[0] not in [1, 3]:
                x = x.permute(2, 0, 1)  # 如果通道不在第一维，则交换维度
        return x

    # 将图像和掩码恢复到原始形状
    def _restore_shape(x, original_shape):
        if len(original_shape) == 2:  # 如果原始是二维数组，则去掉通道维度
            x = x.squeeze(0)
        elif len(original_shape) == 3:
            if original_shape[0] not in [1, 3]:  # 如果原始通道在最后一维
                x = x.permute(1, 2, 0)
        return x

    image_prepared = _prepare_tensor(image)
    mask_prepared = _prepare_tensor(mask)

    # 拼接图像和掩码 应用变换
    combined = torch.cat((image_prepared, mask_prepared), dim=0)
    combined_transformed = transform(combined)

    # 分离图像和掩码
    image_channels = image_prepared.shape[0]
    transformed_image = combined_transformed[:image_channels, :, :]
    transformed_mask = combined_transformed[image_channels:, :, :]

    transformed_image = _restore_shape(transformed_image, original_image_shape)
    transformed_mask = _restore_shape(transformed_mask, original_mask_shape)

    return transformed_image, transformed_mask


def get_dataset(data_dir="./data", transforms=[], dataset_class=None):
    # 数据路径
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')

    if dataset_class is None:
        dataset_class = SegDataset

    # 创建数据集
    dataset = dataset_class(images_dir, masks_dir, transforms)
    return dataset


def split_dataset(dataset, train_ratio=0.8, random_seed=42):
    """划分数据集"""
    # 验证参数
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio必须在0和1之间，得到: {train_ratio}")

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(f"数据集划分无效: train_size={train_size}, val_size={val_size}")

    # 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(random_seed))
    return train_dataset, val_dataset


def get_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)
    return dataloader

