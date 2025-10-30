# 定义一个训练器，它充当主题（Subject）和模板方法
class Trainer:
    def __init__(self):
        self.callbacks = []
    def add_callbacks(self, callback):
        self.callbacks.append(callback)
    def _call_events(self, method_name, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)
    def fit(self, model, train_loader, val_loader, epochs, **kwargs):
        self._call_events('train_begin')
        for epoch in range(epochs):
            self._call_events('epoch_begin', epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
                self._call_events('batch_begin')
                loss = model.train_(data, target)
                self._call_events('batch_end', batch_idx, loss=loss)
            self._call_events('epoch_end', epoch)
        self._call_events('train_end', )


# 定义回调的抽象基类（可选，但推荐）
from abc import ABC, abstractmethod

class Callback(ABC):
    @abstractmethod
    def train_begin(self):
        pass
    def train_end(self):
        pass
    def epoch_begin(self, epoch):
        pass
    def epoch_end(self, epoch):
        pass
    def batch_begin(self):
        pass
    def batch_end(self):
        pass

# 具体的回调实现（策略1：日志记录）
class LoggingCallback(Callback):
    def epoch_begin(self, epoch):
        print(f"{epoch}")
    def batch_end(self):
        print(f"")

# 策略2：学习率调整
class LrAdjusterCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 每个epoch结束后调整学习率
        # 这里简单示例，实际可能根据损失调整
        if epoch % 10 == 0:
            print("调整学习率")
        # 实际调整学习率的代码...

# 使用工厂模式创建回调（可选）
def create_callbacks(config):
    callbacks = []
    if config.get('logging', False):
        callbacks.append(LoggingCallback())
    if config.get('lradjustment', False):
        callbacks.append(LrAdjusterCallback())
    return callbacks

# 使用
config = {'logging': True, 'adjust_lr': True}
trainer = Trainer()
callbacks = create_callbacks(config)
for callback in callbacks:
    trainer.add_callback(callback)

# 然后开始训练
# trainer.fit(model, data_loader, epochs=10)
