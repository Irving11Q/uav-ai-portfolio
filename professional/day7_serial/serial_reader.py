"""
串口数据读取与解析。

从串口读取一行飞行参数（电池电压、电机电流、机身温度、飞行高度），
按逗号分隔解析为数值并打印。通过 MockSerial 模拟串口连接，
接口与 pyserial 保持一致，便于后续替换为真实设备。
"""

import socket


class MockSerial:
    """模拟串口，接口与 pyserial.Serial 一致（readline / close）。"""

    def __init__(self, host: str, port: int) -> None:
        self._conn = socket.create_connection((host, port))
        self._buffer = b""

    def readline(self) -> bytes:
        """读取一行数据，直到换行符。"""
        while b"\n" not in self._buffer:
            chunk = self._conn.recv(1024)
            if not chunk:
                return b""
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line + b"\n"

    def close(self) -> None:
        self._conn.close()


def parse_line(line: bytes) -> list | None:
    """将一行 "24.6,1.2,38.5,120" 解析为数值列表；格式错误返回 None。"""
    text = line.decode("utf-8").strip()
    parts = text.split(",")
    try:
        return [float(p) for p in parts]
    except ValueError:
        print(f"数据格式异常，跳过该行: {text}")
        return None


def main() -> None:
    host = "127.0.0.1"
    port = 9999

    ser = MockSerial(host, port)
    print(f"已连接串口 {host}:{port}，开始读取数据（Ctrl+C 停止）")

    try:
        while True:
            line = ser.readline()
            values = parse_line(line)
            if values is not None:
                print(f"电压={values[0]}V  电流={values[1]}A  "
                      f"温度={values[2]}°C  高度={values[3]}m")
    except KeyboardInterrupt:
        print("\n已停止读取")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
