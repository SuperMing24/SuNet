import nibabel as nib
import numpy as np

from utils.segutil import SegDataset


class Cat25Dataset(SegDataset):
    def __init__(self, img_dir, mask_dir, transforms=[], check='none'):
        super().__init__(img_dir, mask_dir, transforms, check)

    def _resolve_ids(self, img_dir, mask_dir, check):
        image_pairs = super()._resolve_ids(img_dir, mask_dir, check)
        self.load_datas = {}
        for image_file, mask_file in image_pairs:
            self.load_datas[image_file] = nib.load(image_file).get_fdata().astype(np.float32)
            self.load_datas[mask_file] = nib.load(mask_file).get_fdata().astype(np.float32)

        id_pairs = []
        for image_file, mask_file in image_pairs:
            for layer in range(self.load_datas[image_file].shape[2]):
                id_pairs.append((image_file, mask_file, layer))
        return id_pairs

    def _load_datas(self, id):
        image_file, mask_file, layer = id
        image_data = self.load_datas[image_file]
        mask_data = self.load_datas[mask_file]

        zs = [max(layer - 1, 0), layer, min(layer + 1, image_data.shape[2] - 1)]
        processed_images = image_data[:, :, zs]
        # 计算每个切片的均值和标准差
        means = processed_images.mean(axis=(0, 1))  # 形状: (3,)
        stds = processed_images.std(axis=(0, 1))  # 形状: (3,)
        # z-score批量归一化
        processed_images = (processed_images - means) / (stds + 1e-8)

        processed_masks = mask_data[:, :, [layer]]
        processed_masks = (processed_masks > 0).astype(np.float32)
        return processed_images, processed_masks
        # return_value = (processed_images, processed_masks)
        # print(f"返回元组的长度: {len(return_value)}")
        # return return_value
