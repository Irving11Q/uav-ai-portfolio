"""
模拟设备数据源。

通过本地 TCP 连接每 1 秒发送一行飞行参数（电池电压、电机电流、
机身温度、飞行高度），模拟真实下位机经串口上报数据的格式。
与 serial_reader.py 配套使用，用于在没有真实硬件时联调读取逻辑。
"""

import random
import socket
import time


def main() -> None:
    host = "127.0.0.1"
    port = 9999

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"模拟设备已启动，监听 {host}:{port}")

    conn, _ = server.accept()
    print("已连接读取端，开始发送数据")

    try:
        while True:
            values = [
                round(random.uniform(23.0, 25.0), 1),   # 电池电压 V
                round(random.uniform(0.5, 2.0), 1),     # 电机电流 A
                round(random.uniform(30.0, 60.0), 1),   # 机身温度 °C
                round(random.uniform(50, 150), 0),      # 飞行高度 m
            ]
            line = ",".join(str(v) for v in values) + "\n"
            conn.sendall(line.encode("utf-8"))
            time.sleep(1)
    except (ConnectionError, BrokenPipeError):
        print("读取端已断开，模拟设备停止。")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
