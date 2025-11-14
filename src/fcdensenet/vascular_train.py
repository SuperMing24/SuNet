import os

import torch
from IPython.conftest import work_path
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2

from utils.niiutil import CatNiiDataset
from fcdensenet.fc_dn_model import FCDenseNet
from utils import trainer
from utils import visualization
from utils.segutil import get_dataset, get_dataloader
from utils.forward_hook import ForwardHookCaller, ModuleNode, Tuple, Any

from utils.losses import LOSS_REGISTRY
from utils.metrics import create_metrics
from utils.logger import SegLogger


class MyHookCaller(ForwardHookCaller):
    """打印执行过程的钩子调用器"""

    def before_forward(self, node: ModuleNode, inputs: Tuple[Any, ...]) -> None:
        if node.is_leaf:
            if 'conv' in node.local_name:
                X = inputs[0]
                print('before ' + node.full_path,
                      X[0, 0, X.shape[2] // 2:X.shape[2] // 2 + 5, X.shape[3] // 2:X.shape[3] // 2 + 5])

    def after_forward(self, node: ModuleNode, inputs: Tuple[Any, ...], outputs: Any) -> None:
        X = outputs[0]
        print('after ' + node.name, X[0, 0, X.shape[2] // 2:X.shape[2] // 2 + 5, X.shape[3] // 2:X.shape[3] // 2 + 5])


if __name__ == '__main__':
    model_rafat = FCDenseNet(1, 48, [4, 5, 7, 9, 11, 13], growth_rate=18)
    # model_check = FCDenseNet(3, 48, [4, 5, 7, 10, 12, 15], growth_rate=16)
    # model_rafat_cat_attention = FCDenseNet(3, 48, [4, 5, 7, 9, 11, 13], growth_rate=18)

    # X = torch.randn(1, 3, 256, 256)
    # forward_hook.show_model_structure(model, True)
    # with forward_hook.Forward_Hook(model, MyHookCaller()):
    #     model(X)
    # input('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx程序已暂停，请按Enter键继续xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # with Tracer():
    #     result = model.forward(X)
    # sys.exit(0)

    base_dir = 'E:/Surora/Azure_work/AI'
    data_dir = os.path.join(base_dir, 'datasets/fcdnet')
    model_dir = os.path.join(base_dir, 'model/fcdnet')
    work_dir = os.path.join(base_dir, 'work/fcdnet')

    tf = [transforms.ToTensor()]
    batch_size = 1

    train_dataset = get_dataset(data_dir + '/train', tf, dataset_class=CatNiiDataset, num_slices=1)
    val_dataset = get_dataset(data_dir + '/val', tf, dataset_class=CatNiiDataset, num_slices=1)
    train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_dataloader = get_dataloader(val_dataset, batch_size, shuffle=True, num_workers=0)


    # 配置参数
    config = {
        'model': model_rafat,
        'epochs': 500,
        'lr': 1e-3,
        'device': torch.device('cuda:0'),

        # 数据
        'train_loader': train_dataloader,
        'val_loader': val_dataloader,

        # 日志记录器
        'loggers': [SegLogger(work_dir, total_epochs=500, log_interval=5)],

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
            'update_lr_based_on': 'val_loss',  # 或 'cl_dice'
            'save_checkpoint_interval': 1,
            'should_save_checkpoint': True,
            'enable_early_stop': True,
            'enable_quality_checks': True  # 只有质量合格的改进才保存
        },

        # 其他配置
        'model_dir': model_dir,
        'work_dir': work_dir,
        'checkpoint': 'best_model.pth',  # 可选：恢复训练
    }

    trainer = trainer.Trainer(config)
    # command.CommandServer(trainer).start()
    # trainer.start_train()