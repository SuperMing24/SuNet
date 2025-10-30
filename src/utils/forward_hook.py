import torch.nn as nn
from typing import List, Tuple, Set, Optional, Callable, Any, Dict
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ModuleNode:
    """模型节点描述"""
    module: nn.Module
    name: str                    # 完整路径名
    local_name: str             # 在当前层级中的名字
    depth: int                  # 深度（根节点为0）
    parent: Optional['ModuleNode'] = None
    children: List['ModuleNode'] = None
    is_leaf: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def module_type(self) -> str:
        return self.module.__class__.__name__
    
    @property
    def full_path(self) -> str:
        return self.name

class ForwardHookCaller:
    """前向传播钩子调用器"""
    
    def before_forward(self, node: ModuleNode, inputs: Tuple[Any, ...]) -> None:
        """前向传播前调用"""
        pass
    
    def after_forward(self, node: ModuleNode, inputs: Tuple[Any, ...], outputs: Any) -> None:
        """前向传播后调用"""
        pass

class PrintHookCaller(ForwardHookCaller):
    """打印执行过程的钩子调用器"""
    
    def before_forward(self, node: ModuleNode, inputs: Tuple[Any, ...]) -> None:
        if node.is_leaf:
            input_tensor = inputs[0] if len(inputs) > 0 else None
            if hasattr(input_tensor, 'shape'):
                print(f"{'  ' * node.depth}[{node.full_path}] input: {list(input_tensor.shape)}")
        else:
            print(f"{'  ' * node.depth}[{node.full_path}] forward start")
    
    def after_forward(self, node: ModuleNode, inputs: Tuple[Any, ...], outputs: Any) -> None:
        if node.is_leaf:
            if hasattr(outputs, 'shape'):
                print(f"{'  ' * node.depth}[{node.full_path}] output: {list(outputs.shape)}")
        else:
            print(f"{'  ' * node.depth}[{node.full_path}] forward end")

class ModelTreeBuilder:
    """模型树构建器"""
    
    @staticmethod
    def build_model_tree(model: nn.Module, root_name: str = 'model') -> ModuleNode:
        """
        构建完整的模型树
        
        Args:
            model: 根模块
            root_name: 根节点名称
            
        Returns:
            模型树的根节点
        """
        visited: Set[int] = set()
        root_node = ModuleNode(
            module=model,
            name=root_name,
            local_name=root_name,
            depth=0
        )
        
        ModelTreeBuilder._build_tree_recursive(model, root_node, visited)
        ModelTreeBuilder._mark_leaf_nodes(root_node)
        
        return root_node
    
    @staticmethod
    def _build_tree_recursive(module: nn.Module, parent_node: ModuleNode, visited: Set[int]) -> None:
        """递归构建模型树"""
        for local_name, child_module in ModelTreeBuilder._get_direct_submodules(module, visited):
            full_name = f"{parent_node.name}.{local_name}" if parent_node.name else local_name
            
            child_node = ModuleNode(
                module=child_module,
                name=full_name,
                local_name=local_name,
                depth=parent_node.depth + 1,
                parent=parent_node
            )
            
            parent_node.children.append(child_node)
            ModelTreeBuilder._build_tree_recursive(child_module, child_node, visited)
    
    @staticmethod
    def _mark_leaf_nodes(node: ModuleNode) -> None:
        """标记叶子节点"""
        if not node.children:
            node.is_leaf = True
        else:
            for child in node.children:
                ModelTreeBuilder._mark_leaf_nodes(child)
    
    @staticmethod
    def _get_direct_submodules(module: nn.Module, visited: Set[int]) -> List[Tuple[str, nn.Module]]:
        """获取直接子模块（避免重复和循环引用）"""
        result = []
        local_visited = set()  # 当前模块内的去重

        # 特殊处理容器类型的模块
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            # 对于 ModuleList 和 ModuleDict，直接遍历它们的内容
            if isinstance(module, nn.ModuleList):
                for idx, item in enumerate(module):
                    if isinstance(item, nn.Module) and id(item) not in visited:
                        visited.add(id(item))
                        result.append((f"[{idx}]", item))
            elif isinstance(module, nn.ModuleDict):
                for key, item in module.items():
                    if isinstance(item, nn.Module) and id(item) not in visited:
                        visited.add(id(item))
                        result.append((f"['{key}']", item))
            return result

        # 对于普通模块，遍历其属性
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue

            attribute = getattr(module, attr_name)
            module_id = id(attribute)

            # 跳过已访问的模块（避免循环引用）
            if module_id in visited:
                continue

            # 单个模块
            if isinstance(attribute, nn.Module):
                if module_id not in local_visited:
                    local_visited.add(module_id)
                    visited.add(module_id)
                    result.append((attr_name, attribute))

            # 容器类型
            elif isinstance(attribute, (list, tuple)):
                for idx, item in enumerate(attribute):
                    if isinstance(item, nn.Module) and id(item) not in visited:
                        visited.add(id(item))
                        result.append((f"{attr_name}[{idx}]", item))

            # 字典类型
            elif isinstance(attribute, dict):
                for key, item in attribute.items():
                    if isinstance(item, nn.Module) and id(item) not in visited:
                        visited.add(id(item))
                        result.append((f"{attr_name}['{key}']", item))

        return result


class ForwardHookManager:
    """前向传播钩子管理器"""
    
    @staticmethod
    def attach_forward_hooks(model_tree: ModuleNode, hook_caller: ForwardHookCaller) -> None:
        """
        为模型树附加前向传播钩子
        
        Args:
            model_tree: 模型树根节点
            hook_caller: 钩子调用器
        """
        ForwardHookManager._attach_hooks_recursive(model_tree, hook_caller)
    
    @staticmethod
    def attach_forward_hooks_to_model(model: nn.Module, hook_caller: ForwardHookCaller, root_name: str = 'model') -> ModuleNode:
        """
        直接为模型附加前向传播钩子
        
        Args:
            model: 模型
            hook_caller: 钩子调用器
            root_name: 根节点名称
            
        Returns:
            模型树根节点（用于后续解除钩子）
        """
        model_tree = ModelTreeBuilder.build_model_tree(model, root_name)
        ForwardHookManager.attach_forward_hooks(model_tree, hook_caller)
        return model_tree
    
    @staticmethod
    def detach_forward_hooks(model_tree: ModuleNode) -> None:
        """
        解除模型树的前向传播钩子
        
        Args:
            model_tree: 模型树根节点
        """
        ForwardHookManager._detach_hooks_recursive(model_tree)
    
    @staticmethod
    def _attach_hooks_recursive(node: ModuleNode, hook_caller: ForwardHookCaller) -> None:
        """递归附加钩子"""
        if hasattr(node.module, '_original_forward'):
            return  # 避免重复附加
        
        original_forward = node.module.forward
        
        def wrapped_forward(*args, **kwargs):
            # 前向传播前
            hook_caller.before_forward(node, args)
            
            # 执行原始前向传播
            output = original_forward(*args, **kwargs)
            
            # 前向传播后
            hook_caller.after_forward(node, args, output)
            
            return output
        
        # 保存原始方法并替换
        node.module._original_forward = original_forward
        node.module.forward = wrapped_forward
        
        # 递归处理子节点
        for child in node.children:
            ForwardHookManager._attach_hooks_recursive(child, hook_caller)
    
    @staticmethod
    def _detach_hooks_recursive(node: ModuleNode) -> None:
        """递归解除钩子"""
        if hasattr(node.module, '_original_forward'):
            node.module.forward = node.module._original_forward
            delattr(node.module, '_original_forward')
        
        for child in node.children:
            ForwardHookManager._detach_hooks_recursive(child)

# 便捷函数
def with_forward_hooks(model: nn.Module, hook_caller: ForwardHookCaller, root_name: str = 'model'):
    """
    为模型附加前向传播钩子的便捷函数
    
    Args:
        model: 模型
        hook_caller: 钩子调用器
        root_name: 根节点名称
        
    Returns:
        模型树根节点
    """
    return ForwardHookManager.attach_forward_hooks_to_model(model, hook_caller, root_name)

def remove_forward_hooks(model_tree: ModuleNode):
    """
    解除模型前向传播钩子的便捷函数
    
    Args:
        model_tree: 模型树根节点
    """
    ForwardHookManager.detach_forward_hooks(model_tree)


class Forward_Hook:
    def __init__(self, model:ModuleNode,hook_caller: ForwardHookCaller,root_name:str= 'model'):
        self.model = model
        self.hook_caller = hook_caller
        self.root_name = root_name

    def __enter__(self):
        with_forward_hooks(self.model,self.hook_caller,self.root_name)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_forward_hooks(ModelTreeBuilder.build_model_tree(self.model))
        return False

#---------------------------------------------------------------------------------------
def print_model_tree(model: nn.Module, root_name: str = 'model', max_depth: Optional[int] = None):
    """
    打印模型树结构

    Args:
        model: 要打印的模型
        root_name: 根节点名称
        max_depth: 最大打印深度，None表示不限制
    """
    model_tree = ModelTreeBuilder.build_model_tree(model, root_name)
    _print_tree_recursive(model_tree, max_depth)


def _print_tree_recursive(node: ModuleNode, max_depth: Optional[int], current_depth: int = 0):
    """递归打印模型树"""
    if max_depth is not None and current_depth > max_depth:
        return

    # 缩进前缀
    indent = '  ' * current_depth

    # 节点信息
    module_type = node.module_type
    local_name = node.local_name
    is_leaf = node.is_leaf

    # 叶子节点标记
    leaf_marker = " 🍃" if is_leaf else ""

    # 打印当前节点
    print(f"{indent}├── {local_name} ({module_type}){leaf_marker}")

    # 打印参数信息（如果可用）
    param_count = _count_parameters(node.module)
    if param_count > 0:
        print(f"{indent}│   └── Parameters: {param_count:,}")

    # 递归打印子节点
    for child in node.children:
        _print_tree_recursive(child, max_depth, current_depth + 1)


def _count_parameters(module: nn.Module) -> int:
    """计算模块的参数数量"""
    return sum(p.numel() for p in module.parameters())


def print_detailed_model_tree(model: nn.Module, root_name: str = 'model', max_depth: Optional[int] = None):
    """
    打印详细的模型树结构，包含更多信息

    Args:
        model: 要打印的模型
        root_name: 根节点名称
        max_depth: 最大打印深度
    """
    model_tree = ModelTreeBuilder.build_model_tree(model, root_name)
    _print_detailed_tree_recursive(model_tree, max_depth)


def _print_detailed_tree_recursive(node: ModuleNode, max_depth: Optional[int], current_depth: int = 0):
    """递归打印详细模型树"""
    if max_depth is not None and current_depth > max_depth:
        return

    indent = '  ' * current_depth
    module_type = node.module_type
    local_name = node.local_name
    full_path = node.full_path
    is_leaf = node.is_leaf

    # 节点类型标记
    if is_leaf:
        marker = "🍃"
    elif len(node.children) == 0:
        marker = "🌱"  # 实际上没有子节点的叶子
    else:
        marker = "📁"

    # 打印当前节点
    print(f"{indent}├── {marker} {local_name}")
    print(f"{indent}│   ├── Type: {module_type}")
    print(f"{indent}│   ├── Path: {full_path}")
    print(f"{indent}│   ├── Depth: {node.depth}")

    # 参数信息
    param_count = _count_parameters(node.module)
    trainable_count = _count_trainable_parameters(node.module)
    print(f"{indent}│   ├── Parameters: {param_count:,}")
    print(f"{indent}│   └── Trainable: {trainable_count:,}")

    # 递归打印子节点
    for i, child in enumerate(node.children):
        if i == len(node.children) - 1:
            # 最后一个子节点
            new_indent = indent + "    "
        else:
            new_indent = indent + "│   "

        # 临时修改缩进用于递归
        _print_detailed_tree_recursive(child, max_depth, current_depth + 1)


def _count_trainable_parameters(module: nn.Module) -> int:
    """计算可训练参数数量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# 便捷函数
def show_model_structure(model: nn.Module, detailed: bool = False, max_depth: Optional[int] = None):
    """
    显示模型结构的便捷函数

    Args:
        model: 要显示的模型
        detailed: 是否显示详细信息
        max_depth: 最大显示深度
    """
    print("=" * 60)
    print("Model Structure")
    print("=" * 60)

    if detailed:
        print_detailed_model_tree(model, max_depth=max_depth)
    else:
        print_model_tree(model, max_depth=max_depth)

    # 打印总体统计
    total_params = _count_parameters(model)
    trainable_params = _count_trainable_parameters(model)
    print("=" * 60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print("=" * 60)


@contextmanager
def forward_tracing(model: nn.Module, hook_caller: Optional[ForwardHookCaller] = None, root_name: str = 'model'):
    """
    前向传播追踪的上下文管理器
    
    Args:
        model: 模型
        hook_caller: 钩子调用器（默认为PrintHookCaller）
        root_name: 根节点名称
    """
    if hook_caller is None:
        hook_caller = PrintHookCaller()
    
    model_tree = None
    try:
        model_tree = with_forward_hooks(model, hook_caller, root_name)
        yield model_tree
    finally:
        if model_tree is not None:
            remove_forward_hooks(model_tree)

# 使用示例
def example_usage():
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # 使用方法1：上下文管理器
    print("=== 使用方法1：上下文管理器 ===")
    with forward_tracing(model):
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
    
    print("\n=== 使用方法2：自定义钩子 ===")
    
    # 自定义钩子
    class CustomHookCaller(ForwardHookCaller):
        def before_forward(self, node: ModuleNode, inputs: Tuple[Any, ...]) -> None:
            if node.is_leaf:
                print(f"{'->' * node.depth} {node.local_name} ({node.module_type})")
    
    # 手动附加和解除钩子
    model_tree = with_forward_hooks(model, CustomHookCaller())
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    remove_forward_hooks(model_tree)

if __name__ == "__main__":
    import torch
    example_usage()
