import torch
from torch import nn

from utils import putil
from utils import putil as putil_old

def train(model, train_loader, val_loader, num_epochs, lr):
    """
    Args:

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    timer, num_batches = putil.Timer2(), len(train_loader)
    print(f'开始训练，共{num_epochs}个epoch')

    for epoch in range(num_epochs):
        metric = putil.Accumulator(3)
        model.train()  # 设置为训练模式

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            timer.start()  # 开始计时
            optimizer.zero_grad()

            # 前向传播
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)

            # 反向传播
            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(fc-densenet.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            timer.stop()  # 停止计时

            # 记录训练，不计算梯度
            with torch.no_grad():
                metric.add(batch_loss.item() * inputs.size(0), putil.accuracy(outputs, targets), inputs.size(0))

            # 计算当前批次的训练损失和准确率
            train_loss = metric[0] / metric[2]  # 平均训练损失
            train_accuracy = metric[1] / metric[2]  # 平均训练准确率

            # 定期打印训练进度
            if (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
                print(f'Epoch {epoch + 1}, batch {batch_idx + 1}/{num_batches}, '
                      f'train loss: {train_loss:.3f}, train acc: {train_accuracy:.3f}')

        scheduler.step()  # 学习率衰减
        # 每个epoch结束后在验证集上评估模型（不是测试集）
        val_accuracy = putil.evaluate_accuracy(model, val_loader)
        print(f'Epoch {epoch + 1} completed, validation acc: {val_accuracy:.3f}')

    # 所有epoch循环运行结束后的统计
    print(f'loss {train_loss:.3f}, train acc {train_accuracy:.3f}, '
          f'test acc {val_accuracy:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.3f} examples/sec '
          f'on {str(device)}')

if __name__ == '__main__':
    model_test = densenet121(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model_test(x)
    print(out.shape)

    # 加载数据加载器
    dataset_manager = DatasetManager()
    train_dataset, val_dataset = dataset_manager.get_datasets('CIFAR10')
    train_loader, val_loader = dataset_manager.get_dataloaders((train_dataset, val_dataset), batch_size=128, num_workers=4)

    # # 设置训练
    lr, num_epochs = 0.01, 10
    model = densenet121(num_classes=100, in_channels=3, init_feature_channels = 24, growth_rate=12)
    # train(fc-densenet, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, lr=lr)

    # 使用简化版本测试
    simple_model = SimpleDenseNet(num_classes=10)
    X = torch.randn(size=(1, 3, 96, 96))
    # putil_old.xavier_init_weights(fc-densenet)
    putil_old.exec_with_detail(model, X)
    # train(simple_model, train_loader, val_loader, num_epochs=500, lr=0.01)