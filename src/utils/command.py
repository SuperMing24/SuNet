import cmd
import shlex
import socket
import threading
from typing import Union, Dict, Any, Optional


class CommandProcessor:
    default_key: str = "default"

    def __init__(self, objects: Union[Dict[str, Any], Any]):
        super().__init__()
        self.objects = {CommandProcessor.default_key: objects} if not isinstance(objects, dict) else objects

    def add(self, obj, key='default'):
        self.objects[key] = obj

    def _resolve_object_path(self, path: str) -> Optional[Any]:
        """解析对象路径，返回最终对象和方法名"""
        parts = path.split('.')
        current = None

        # 处理默认对象情况
        if parts[0] not in self.objects:
            if self.default_key in self.objects:
                parts.insert(0, self.default_key)
            else:
                return None, None

        try:
            current = self.objects[parts[0]]
            for part in parts[1:-1]:
                current = getattr(current, part)
            method_name = parts[-1]
            return current, method_name
        except (AttributeError, IndexError):
            return None, None

    def _parse_args(self, arg_str: str) -> tuple:
        """解析参数字符串为Python对象"""
        if not arg_str.strip():
            return ()

        try:
            # 使用shlex处理带引号的参数
            args = shlex.split(arg_str)
            parsed_args = []
            for arg in args:
                # 尝试转换为数字
                try:
                    parsed_args.append(int(arg))
                except ValueError:
                    try:
                        parsed_args.append(float(arg))
                    except ValueError:
                        # 保持为字符串
                        parsed_args.append(arg)
            return tuple(parsed_args)
        except ValueError:
            return (arg_str,)

    def execute(self, line: str):
        """处理所有输入命令"""
        if not line.strip():
            return 'no command'

        # 分离命令和参数
        parts = line.split(maxsplit=1)
        cmd_path = parts[0]
        arg_str = parts[1] if len(parts) > 1 else ""

        obj, method_name = self._resolve_object_path(cmd_path)
        if obj is None or method_name is None:
            msg = f"Error: Invalid command path '{cmd_path}'"
            print(msg)
            return msg

        method = getattr(obj, method_name, None)
        if not callable(method):
            msg = f"Error: '{method_name}' is not a callable method"
            print(msg)
            return msg

        try:
            args = self._parse_args(arg_str)
            result = method(*args)
            if result is None: result = 'None'
            print(result)
            return result

        except Exception as e:
            msg = f"Error executing command: {e}"
            print(msg)
            return msg


class CmdProcessor(cmd.Cmd):

    def __init__(self, objects: Union[Dict[str, Any], Any]):
        super().__init__()
        objects = {CommandProcessor.default_key: objects} if not isinstance(objects, dict) else objects
        self.processor = CommandProcessor(objects)
        self.prompt = "> "
        self._stop_event = threading.Event()
        self._thread = None

    def add(self, obj, key=CommandProcessor.default_key):
        self.processor.add(obj, key)

    def start_async(self):
        """异步启动命令行界面"""
        self._thread = threading.Thread(target=self.cmdloop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止命令行界面"""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def default(self, line: str):
        """处理所有输入命令"""
        if not line.strip():
            return

        if line == 'exit':
            self.stop()

        r = self.processor.execute(line)
        print(r)


class CommandServer:
    CloseMark = 'closed'

    def __init__(self, objects: Union[Dict[str, Any], Any], host='localhost', port=9999):
        super().__init__()
        objects = {CommandProcessor.default_key: objects} if not isinstance(objects, dict) else objects
        self.processor = CommandProcessor(objects)
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._running = False

    def add(self, obj, key=CommandProcessor.default_key):
        self.processor.add(obj, key)

    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self._running = True
        print(f"Command server listening on {self.host}:{self.port}")

        # 将监听逻辑放到一个单独的线程中运行
        server_thread = threading.Thread(target=self._server_loop, daemon=True)
        server_thread.start()

    def _server_loop(self):
        while self._running:
            try:
                client_sock, addr = self.server_socket.accept()
                print(f"Accepted connection from {addr}")
                threading.Thread(
                    target=self.handle_client,
                    args=(client_sock, addr),
                    daemon=True
                ).start()
            except OSError as e:
                if self._running:
                    print(f"Server socket error: {e}")
                else:
                    break
        print("Server stopped.")

    def handle_client(self, sock: socket.socket, addr):
        with sock:
            print(f"New connection from {addr}")
            while True:
                try:
                    data = sock.recv(4096).decode('utf-8').strip()
                    if not data:
                        break

                    if data == 'exit':
                        print(f"Closing connection with {sock.getpeername()}")  # 日志记录
                        sock.sendall(CommandServer.encode_msg(CommandServer.CloseMark))
                        break

                    result = self.processor.execute(data)
                    response = CommandServer.encode_msg(result)
                    sock.sendall(response)
                except (ConnectionResetError, BrokenPipeError):
                    break
                except Exception as e:
                    sock.sendall(f"Error: {str(e)}\n".encode('utf-8'))
            print(f"Connection closed {addr}")

    @staticmethod
    def encode_msg(msg):
        return str(msg).encode('utf-8') + b'\n'


class CommandClient(cmd.Cmd):
    def __init__(self, host='localhost', port=9999):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.prompt = "remote> "
        self._connect()

    def _connect(self):
        try:
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            print("Error: Server not available")
            exit(1)

    def default(self, line):
        try:
            self.socket.sendall(line.encode('utf-8') + b'\n')
            response = self.socket.recv(4096).decode('utf-8')
            if response.strip() == CommandServer.CloseMark:
                exit()
            print(response, end='')
        except (ConnectionResetError, BrokenPipeError):
            print("Connection lost")
            exit(1)



    def emptyline(self):
        pass


if __name__ == "__main__":
    # 示例用法
    class Database:
        def query(self, sql: str):
            return f"Executing: {sql}"


    class Logger:
        def info(self, message: str):
            str = f"INFO: {message}"
            print(str)
            return str

        def xyz(self, a: str, b: int, c: int):
            str = f'{a} {b} {c}'
            print(str)
            return str


    objects = {
        "database": Database(),
        "logger": Logger(),
        "default": Logger()  # 默认对象
    }

    # cli = CmdProcessor(objects)
    # cli.start_async()

    server = CommandServer(objects)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()

    # 主线程可以继续做其他事情
    while True:
        pass
