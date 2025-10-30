import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import warnings

import matplotlib.pyplot as plt
import numpy as np
import time
import atexit
import threading
from typing import List, Tuple, Union, Dict, Set, Any
from collections import OrderedDict


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Timer2:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.begin = time.time()
    def stop(self):
        duration = time.time() - self.begin
        self.times.append(duration)
        return duration
    def sum(self):
        return sum(self.times)

class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 主要accuracy函数（处理标准格式）
def accuracy(outputs, targets):
    """标准accuracy函数，要求多输出格式"""
    return (outputs.argmax(dim=-1) == targets).sum().item()     # 就考虑最后一个维度即可统一（单样本和多样本的情况）
# 特殊情况处理函数
def accuracy_binary(outputs, targets):
    """单输出二分类专用"""
    return ((outputs > 0).long() == targets).sum().item()


def evaluate_accuracy(model, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(model, nn.Module):
        # 防御性编程，检查是否是nn.Module，打开评估模式(1、关闭Dropout 2、禁用求导机制 3、固定BatchNorm)
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
    # 正确预测的数量，总预测的数量，没有loss，因为一般给验证阶段（或者测试集）的测评，只需要准确率
    metric = Accumulator(2)

    # 在不计算梯度的情况下进行预测
    with torch.no_grad():  # 不计算梯度，节省内存和计算
        for X, y in data_iter:
            # 数据迁移到设备，需要进行todevice的原因是，dataloader本身并没有进行，每次用到其中的数据X,y时，都需要检查下是否匹配设备
            if isinstance(X, list):
                # 处理多输入模型（如BERT等）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 累加正确预测数和总样本数
            metric.add(accuracy(model(X), y), y.numel())

    # 防御性除法
    if metric[1] == 0:
        warnings.warn("总样本数为0，无法计算准确率")
        return 0.0
    return metric[0] / metric[1]


# 定义LineDefine类来表示每条线的属性
class LineDefine:
    def __init__(self, name, color, style):
        self.name = name
        self.color = color
        self.style = style


line_defs = Union[LineDefine, str, List[LineDefine], List[str]]


class Animator:
    _instances = []

    def __init__(self, xlabel='', ylabel='', legend: line_defs = None, size=(8, 6), xlim=(0, 1000), ylim=(0, 10),
                 grid=True, grid_style='--', grid_alpha=0.5, grid_color='lightgray'):
        cs = [
            # 基础色缩写
            'r', 'g', 'b',  # 红、绿、蓝
            'c', 'm', 'y', 'k',  # 青、品红、黄、黑

            # 常用颜色名称（Matplotlib支持的全称）
            'orange',  # 橙色
            'purple',  # 紫色
            'pink',  # 粉红
            'brown',  # 棕色
            'lime',  # 亮绿
            'teal',  # 青绿
            'navy',  # 深蓝

            # 现代可视化常用色
            'darkorange',  # 暗橙
            'royalblue',  # 宝蓝
            'forestgreen',  # 森林绿
            'darkviolet',  # 深紫

            # 高对比度补充色
            'gold',  # 金色
            'dodgerblue'  # 道奇蓝
        ]

        styles = ['-', '-', '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']
        i = -1

        def get_leg(g):
            nonlocal i
            if isinstance(g, LineDefine):
                return g
            else:
                i += 1
                name = g  # 把非LineDefine的对象转为字符串作为名称
                color = cs[i % len(cs)]
                style = styles[i % len(styles)]
                return LineDefine(name, color, style)

        if isinstance(legend, list):
            self.line_defs = [get_leg(g) for g in legend]
        else:
            self.line_defs = [get_leg(legend)]

        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=size)  # 创建图表
        self.lines = []
        for line_def in self.line_defs:
            line, = self.ax.plot([], [], lw=2, color=line_def.color, linestyle=line_def.style, label=line_def.name)
            self.lines.append(line)

        # 新增网格线配置
        if grid:
            self.ax.grid(
                visible=True,
                linestyle=grid_style,
                alpha=grid_alpha,
                color=grid_color,
                which='both'  # 同时显示主次网格线
            )
        else:
            self.ax.grid(visible=False)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()

        self.fig.canvas.draw()
        plt.pause(0.1)

        # 注册实例到全局列表
        Animator._instances.append(self)

    def add(self, x, y_values):
        for line, y in zip(self.lines, y_values):
            if y:
                new_x = np.append(line.get_xdata(), x)
                new_y = np.append(line.get_ydata(), y)
                line.set_data(new_x, new_y)
                # 更新后验证
                assert len(new_x) == len(new_y), f"数据更新导致维度不一致:{len(new_x)}<>{len(new_y)}"

        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize(self):
        """阻塞窗口并关闭交互模式"""
        if plt.isinteractive():
            plt.ioff()
        plt.show(block=True)  # 阻塞直到窗口手动关闭

    @classmethod
    def _exit_handler(cls):
        """程序退出时自动调用所有实例的finalize方法"""
        for instance in cls._instances:
            if plt.fignum_exists(instance.fig.number):
                instance.finalize()


# 注册退出回调
atexit.register(Animator._exit_handler)


def trace_model(model: nn.Module, model_name: str = '', deep: int = 0):
    wrap_forward_with_trace(model, model_name, deep)
    for name, child in get_direct_submodules(model):
        full_name = f"{model_name}.{name}" if model_name else name
        wrap_forward_with_trace(child, full_name, deep + 1)
        trace_model(child, full_name, deep + 1)


def un_trace_model(model: nn.Module):
    forward = getattr(model, 'original_forward', None)
    if forward:
        model.forward = forward
        model.original_forward = None
    for _, child in get_direct_submodules(model):
        un_trace_model(child)


def wrap_forward_with_trace(module: nn.Module, model_name: str = "", deep: int = 0):
    if getattr(module, 'original_forward', None):
        return

    original_forward = module.forward
    leaf = len(get_direct_submodules(module)) == 0
    prefix = '\t' * deep
    type_name = module.__class__.__name__
    model_name = type_name + ':' + model_name if model_name else type_name
    if leaf:
        def wrapped_forward(*args, **kwargs):
            input = args[-1]
            msg = f'[{model_name}]:{list(input.shape)}====>'
            output = original_forward(*args, **kwargs)
            print(prefix + msg + f"{list(output.shape)}")
            return output

        module.forward = wrapped_forward
    else:
        def wrapped_forward(*args, **kwargs):
            input = args[-1]
            print(prefix + f'[{model_name}]:{list(input.shape)}====>')
            output = original_forward(*args, **kwargs)
            print(prefix + f'[{model_name}]:====>{list(output.shape)}')
            return output

        module.forward = wrapped_forward
    module.original_forward = original_forward

def get_direct_submodules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    visited: Set[int] = set()  # 用于去重，记录模块 id
    result: List[Tuple[str, nn.Module]] = []

    for attr_name in dir(module):
        if attr_name.startswith('_'):
            continue  # 忽略私有属性

        attribute = getattr(module, attr_name)

        # 情况1：单个模块
        if isinstance(attribute, nn.Module):
            if id(attribute) not in visited:
                visited.add(id(attribute))
                result.append((attr_name, attribute))

        # 情况2：容器类型 list / tuple / ModuleList
        elif isinstance(attribute, (list, tuple, nn.ModuleList)):
            for idx, item in enumerate(attribute):
                if isinstance(item, nn.Module) and id(item) not in visited:
                    visited.add(id(item))
                    result.append((f"{attr_name}.{idx}", item))

        # 情况3：字典或 ModuleDict
        elif isinstance(attribute, (dict, nn.ModuleDict)):
            for key, item in attribute.items():
                if isinstance(item, nn.Module) and id(item) not in visited:
                    visited.add(id(item))
                    result.append((f"{attr_name}.{key}", item))

    return result


def exec_with_detail(model, X):
    trace_model(model)
    Y = model(X)
    un_trace_model(model)
    return Y


class ForwardTracer:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        trace_model(self.model)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        un_trace_model(self.model)
        return False


def xavier_init_weights(model, debug=False):
    if type(model) == nn.Sequential:
        for cmodel in model:
            if debug:
                print(f'init weight of child: {type(cmodel)}')
            xavier_init_weights(cmodel, debug)
    else:
        if type(model) == nn.Linear or type(model) == nn.Conv2d:
            if debug:
                print(f'init weight of model: {type(model)} shape={model.weight.shape}')
            nn.init.xavier_uniform_(model.weight)
        else:
            if isinstance(model, nn.Module):
                for name, param in model.named_parameters():
                    if name.endswith('.weight') and param.dim() >= 2:
                        if debug:
                            print(f'init weight of name: {name} shape={param.shape}')
                        nn.init.xavier_uniform_(param)


def param_count(model):
    pc = 0
    for name, param in model.named_parameters():
        i = 1
        for s in param.shape:
            i *= s
        print(f'{name}:\t{param.shape}\t=\t{i}')
        pc += i
    print('total param count:', pc)


def init_weights(m):
    # 按层类型分别处理
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        # BatchNorm：weight=1, bias=0
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.Linear, nn.Conv1d)):
        # 全连接和1D卷积：Xavier初始化 + 偏置归零
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        # 2D卷积：Kaiming初始化 + 偏置归零
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

import os
import nibabel as nib
import numpy as np
import glob
def analysis_images_discrepancy(base_dir):
    data_dir = os.path.join(base_dir, 'images')
    image_files = glob.glob(os.path.join(data_dir, "*.nii"))

    print("详细分析images数据:")
    print("=" * 80)

    for i, i_file in enumerate(image_files):
        image_file = nib.load(i_file)
        data = image_file.get_fdata()

        # 转换为PyTorch tensor检查
        tensor_data = torch.from_numpy(data).float()

        # 更详细的统计
        print(f"第{i + 1}个图像:")
        print(f"  NumPy范围: [{data.min():8.4f}, {data.max():8.4f}]")
        print(f"  Tensor范围: [{tensor_data.min():8.4f}, {tensor_data.max():8.4f}]")
        print(f"  均值: {data.mean():8.4f}, 标准差: {data.std():8.4f}")
        print(f"  数据类型: {data.dtype}, Tensor类型: {tensor_data.dtype}")

        # 检查数据分布
        unique_vals = np.unique(data)
        print(f"  唯一值数量: {len(unique_vals)}")
        if len(unique_vals) < 20:  # 如果唯一值很少，显示具体值
            print(f"  唯一值: {unique_vals}")

        print("-" * 50)


def analyze_masks_discrepancy(base_dir):
    data_dir = os.path.join(base_dir, 'masks')
    mask_files = glob.glob(os.path.join(data_dir, "*.nii"))

    print("分析masks数据差异:")
    print("=" * 80)

    for i, m_file in enumerate(mask_files):
        mask_file = nib.load(m_file)
        data = mask_file.get_fdata()

        print(f"第{i + 1}个mask:")
        print(f"  范围: [{data.min():8.1f}, {data.max():8.1f}]")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  文件大小: {os.path.getsize(m_file) / 1024:.1f} KB")

        # 检查标签分布
        unique, counts = np.unique(data, return_counts=True)
        print(f"  标签分布: {dict(zip(unique.astype(int), counts))}")

        print("-" * 50)