import os
import time
from dotenv import load_dotenv
from controller import MiLampController

def main():
    # 从 .env 文件加载环境变量
    load_dotenv()
    # 从环境变量中获取 IP 和 Token
    lamp_ip = os.getenv("LAMP_IP")
    lamp_token = os.getenv("LAMP_TOKEN")

    if not lamp_ip or not lamp_token:
        print("错误：请确保在 .env 文件中设置了 LAMP_IP 和 LAMP_TOKEN。")
        return

    # 初始化台灯控制器
    lamp = MiLampController(ip=lamp_ip, token=lamp_token)


    print("\n--- 台灯控制演示 ---")

    # 开灯
    lamp.turn_on()
    time.sleep(1)

    # 设置亮度
    lamp.set_brightness(90)
    time.sleep(1)

    # 设置色温 (2700-5100)
    lamp.set_color_temperature(5000)
    time.sleep(1)

    # 关灯
    lamp.turn_off()

    return


main()
