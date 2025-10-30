import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def tongbutrans(transform, image, mask):
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
    def prepare_tensor(x):
        if x.ndim == 2:
            x = x.unsqueeze(0)  # 如果是二维数组，则增加通道维度
        elif x.ndim == 3:
            if x.shape[0] not in [1, 3]:
                x = x.permute(2, 0, 1)  # 如果通道不在第一维，则交换维度
        return x

    # 将图像和掩码恢复到原始形状
    def restore_shape(x, original_shape):
        if len(original_shape) == 2:  # 如果原始是二维数组，则去掉通道维度
            x = x.squeeze(0)
        elif len(original_shape) == 3:
            if original_shape[0] not in [1, 3]:  # 如果原始通道在最后一维
                x = x.permute(1, 2, 0)
        return x

    image_prepared = prepare_tensor(image)
    mask_prepared = prepare_tensor(mask)

    # 拼接图像和掩码 应用变换
    combined = torch.cat((image_prepared, mask_prepared), dim=0)
    combined_transformed = transform(combined)

    # 分离图像和掩码
    image_channels = image_prepared.shape[0]
    transformed_image = combined_transformed[:image_channels, :, :]
    transformed_mask = combined_transformed[image_channels:, :, :]

    transformed_image = restore_shape(transformed_image, original_image_shape)
    transformed_mask = restore_shape(transformed_mask, original_mask_shape)

    return transformed_image, transformed_mask


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
                image, mask = tongbutrans(transform, image, mask)
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


def get_data_iter(data_dir="./data", transform=[], train_size=None, batch_size=4, num_workers=0, dataset_class=None):
    # 数据路径
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')

    if dataset_class is None:
        dataset_class = SegDataset
    # 创建数据集
    dataset = dataset_class(images_dir, masks_dir, transform)

    if train_size is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, 1 - train_size])

    # 创建数据加载器
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def plot_results(image, true_mask, pred_mask, save_path, size=(5, 5)):
    size = (size * 4, size) if isinstance(size, int) else (4 * size[0], size[1])
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=size,
        subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False}
    )

    image = image.permute(1, 2, 0).cpu().numpy()
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    ax1.set_title('Input Image')
    ax1.imshow(image)

    true_mask = true_mask.squeeze().cpu().numpy()
    ax2.set_title('Ground Truth')
    ax2.imshow(true_mask, cmap='gray')

    pred_mask = (torch.sigmoid(pred_mask).squeeze().detach().cpu().numpy() > 0.5).astype(int)
    ax3.set_title('Prediction')
    ax3.imshow(pred_mask, cmap='gray')

    diff = true_mask != pred_mask
    overlay = np.full((image.shape), 255)
    overlay[diff] = [255, 0, 0]
    ax4.set_title('Error Map')
    ax4.imshow(overlay)

    plt.savefig(save_path)
    plt.close()


def plot_multi_results(image, true_mask, pred_mask, class_colors, save_path, size=(5, 5)):
    size = (size * 4, size) if isinstance(size, int) else (4 * size[0], size[1])
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=size,
        subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False}
    )

    image = image.permute(1, 2, 0).cpu().numpy()
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    ax1.set_title('Input Image')
    ax1.imshow(image)

    # 真实标签可视化
    true_mask = true_mask.cpu().numpy()
    true_color = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        true_color[true_mask == class_id] = color
    ax2.set_title('Ground Truth')
    ax2.imshow(true_color)

    # 预测结果可视化
    pred_mask = pred_mask.argmax(dim=0).cpu().numpy()
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        pred_color[pred_mask == class_id] = color
    ax3.set_title('Prediction')
    ax3.imshow(pred_color)

    # 差异对比图
    diff = true_mask != pred_mask
    overlay = image.copy()
    overlay[diff] = [255, 0, 0]  # 错误区域标红
    ax4.set_title('Error Map')
    ax4.imshow(overlay)

    plt.savefig(save_path)
    plt.close()


def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
