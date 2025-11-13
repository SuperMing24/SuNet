import os
import torch
from torchvision import transforms

from unet_model import UNet

from utils import trainer
from utils import forward_hook
from utils import visualization

from utils.niiutil import CatNiiDataset
from utils.segutil import get_dataset, get_dataloader
from utils.forward_hook import ForwardHookCaller, ModuleNode, Tuple, Any
from utils.losses import LOSS_REGISTRY
from utils.metrics import create_metrics
from utils.logger import SegLogger


if __name__ == '__main__':
    model_origin = UNet(1, 1)

    # X = torch.randn(1, 3, 256, 256)
    # forward_hook.show_model_structure(model_origin, True)

    base_dir = 'E:/Surora/Azure_work/AI'
    data_dir = os.path.join(base_dir, 'datasets/fcdnet')
    model_dir = os.path.join(base_dir, 'model/fcdnet')
    log_dir = os.path.join(base_dir, 'work/fcdnet')
    batch_size = 4
    # 图像预处理
    tf = [
        (transforms.Resize(256), True),
        transforms.ToTensor(),
        (transforms.RandomCrop(size=256), True)
    ]

    train_dataset = get_dataset(data_dir + '/train', tf, dataset_class=CatNiiDataset, num_slices=1)
    val_dataset = get_dataset(data_dir + '/val', tf, dataset_class=CatNiiDataset, num_slices=1)
    train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_dataloader = get_dataloader(val_dataset, batch_size, shuffle=True, num_workers=0)


    # 配置参数
    config = {
        'model': model_origin,
        'epochs': 500,
        'lr': 1e-3,
        'device': torch.device('cuda:0'),

        # 数据
        'train_loader': train_dataloader,
        'val_loader': val_dataloader,

        # 日志记录器
        'loggers': [SegLogger(log_dir, total_epochs=500, log_interval=5)],

        # 可视化工具
        'visualizer': visualization.SegmentationVisualizer(),
        'interval_visualize': 100,

        # 损失函数 - 血管分割专用
        'loss_id': 'dice',
        'loss_fn': LOSS_REGISTRY['dice'],
        # 评估指标 - 简洁配置
        'metric_evaluator': create_metrics(['iou', 'dice', 'cl_dice', 'hausdorff', 'accuracy']),

        # 训练动作配置
        'training_actions': {
            'update_lr_every_epoch': True,
            'update_lr_based_on': 'val_loss',
            'save_checkpoint_interval': 1,
            'should_save_checkpoint': True,
            'enable_early_stop': True,
            'enable_quality_checks': True  # 只有质量合格的改进才保存
        },

        # 其他配置
        'model_dir': model_dir,
        'checkpoint': 'best_model.pth',  # 可选：恢复训练
    }

    trainer = trainer.Trainer(config)
    trainer.start_train()