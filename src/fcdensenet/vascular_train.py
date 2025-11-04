import glob
import os
import sys
import nibabel as nib
import numpy as np
from datetime import datetime

import torch
from torchvision import transforms

from fc_dn_tools import Cat25Dataset
from fcdensenet.fc_dn_model import FCDenseNet
from utils import command
from utils import trainer
from utils.segutil import get_data_iter
from utils.forward_hook import ForwardHookCaller, ModuleNode, Tuple, Any

from utils.losses import LOSS_REGISTRY
from utils.metrics import create_metrics
from utils.logger import SegLogger, SimpleConsoleLogger


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
    model = FCDenseNet(3, 48, [4, 5, 7, 10, 12, 15], growth_rate=16)
    # model_rafat = FCDenseNet(3, 48, [4, 5, 7, 9, 11, 13], growth_rate=18)
    X = torch.randn(1, 3, 256, 256)

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
    log_dir = os.path.join(base_dir, 'work/fcdnet')

    # tf = [transforms.ToTensor(), (transforms.RandomCrop(size=256), True)]
    tf = [transforms.ToTensor()]
    batch_size = 1

    # train_itr, val_itr = get_data_iter(data_dir + '/train', tf, train_size=0.8, batch_size=batch_size,
    #                                    dataset_class=Cat25Dataset)
    train_loader = get_data_iter(data_dir + '/train', tf, train_size=None, batch_size=batch_size,
                              dataset_class=Cat25Dataset)
    val_loader = get_data_iter(data_dir + '/val', tf, train_size=None, batch_size=batch_size, dataset_class=Cat25Dataset)


    # 配置参数
    config = {
        'model': model,
        'epochs': 500,
        'lr': 1e-4,
        'device': torch.device('cuda:0'),

        # 数据
        'train_loader': train_loader,
        'val_loader': val_loader,

        # 日志记录器
        'loggers': [SegLogger(log_dir, total_epochs=500, log_interval=2)],

        # 损失函数 - 血管分割专用
        'loss_id': 'cl_dice',
        'loss_fn': LOSS_REGISTRY['cl_dice'],
        # 评估指标 - 简洁配置
        'metric_evaluator': create_metrics(['dice', 'cl_dice', 'hausdorff', 'accuracy']),

        # 训练动作配置
        'training_actions': {
            'update_lr_every_epoch': True,
            'update_lr_based_on': 'val_loss',  # 或 'cl_dice'
            'save_checkpoint_interval': 5,
            'should_save_checkpoint': True,
            'enable_early_stop': True,
            'enable_quality_checks': True  # 只有质量合格的改进才保存
        },

        # 其他配置
        'model_dir': model_dir,
        'checkpoint': 'best_model.pth',  # 可选：恢复训练
    }

    trainer = trainer.Trainer(config)
    # command.CommandServer(trainer).start()
    trainer.start_train()