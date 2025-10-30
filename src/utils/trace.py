import sys
import inspect
import functools


class Tracer:
    current_module_name = 'current_module'

    def __init__(self, modules=[], max_str_len=20, max_list_len=5):
        self.max_str_len = max_str_len
        self.max_list_len = max_list_len

        if len(modules) == 0:
            modules = [Tracer.current_module_name]

        frame = inspect.currentframe().f_back
        def t_module_name(name):
            if name == Tracer.current_module_name:
                return frame.f_globals['__name__']
            else:
                return name

        modules = [t_module_name(module) for module in modules] if isinstance(modules, list) else [t_module_name(modules)]
        modules = list(dict.fromkeys(modules))

        self._modules = [sys.modules[module] for module in modules]
        self._original_methods = {}
        self._depth = 0
        self._has_inner_calls = {}

    def __enter__(self):
        self.wrap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.un_warp()

    def wrap(self):
        for module in self._modules:
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    key = module.__name__ + '.' + name
                    self._original_methods[key] = obj
                    setattr(module, name, self._wrap_method(obj, name))

    def un_warp(self):
        for module in self._modules:
            prefix = module.__name__ + '.'
            for full_name, original in self._original_methods.items():
                if full_name.startswith(prefix):
                    name = full_name[len(prefix):]
                    setattr(module, name, original)
        self._original_methods.clear()

    def _wrap_method(self, method, method_name):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            current_depth = self._depth
            self._has_inner_calls[current_depth] = False

            # 获取调用信息
            caller_frame = inspect.currentframe().f_back.f_back
            caller_info = ""
            if caller_frame:
                caller_line = caller_frame.f_lineno
                caller_file = caller_frame.f_code.co_filename
                caller_info = f" @ {caller_file}:{caller_line}"

            indent = '\t' * min(current_depth, 20)
            args_str = self._format_args(args, kwargs)

            # 打印调用入口信息
            call_info = f"{indent}[{current_depth}]{method_name}({args_str})"
            print('\n' + call_info, end='')

            try:
                self._depth += 1
                result = method(*args, **kwargs)
                # 判断是否是最底层调用
                if not self._has_inner_calls[current_depth]:
                    print(f" ==> {self._format_arg(result)}", end='' if current_depth > 0 else '\n')
                else:
                    print(f"\n{call_info} ==> {self._format_arg(result)}", end='' if current_depth > 0 else '\n')
                return result
            except Exception as e:
                error_info = f"{indent}[{current_depth}]{method_name} ==> RAISED {type(e).__name__}: {str(e)}"
                print(error_info)
                raise
            finally:
                if current_depth > 0:
                    self._has_inner_calls[current_depth - 1] = True
                # 恢复状态
                self._depth -= 1

        return wrapper

    def _format_args(self, args, kwargs):
        args_str = ', '.join([self._format_arg(arg) for arg in args])
        kwargs_str = ', '.join([f'{k}={self._format_arg(v)}' for k, v in kwargs.items()])
        return ', '.join(filter(None, [args_str, kwargs_str]))

    def _format_arg(self, arg):
        if isinstance(arg, (int, float, bool, type(None))):
            return str(arg)
        elif isinstance(arg, str):
            return f'"{arg[:self.max_str_len]}..."' if len(arg) > self.max_str_len else f'"{arg}"'
        elif isinstance(arg, (list, tuple)):
            if len(arg) > self.max_list_len:
                return f'{type(arg).__name__}({arg[:self.max_list_len]}...), len={len(arg)}'
            return f'{type(arg).__name__}({arg})'
        elif hasattr(arg, 'shape'):
            return f'{type(arg).__name__}(shape={arg.shape})'
        else:
            return f'{type(arg).__name__}'
