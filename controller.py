import logging
from miio import MiotDevice
from miio.exceptions import DeviceException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MiLampController:
    MIN_COLOR_TEMP = 2700
    MAX_COLOR_TEMP = 5100

    def __init__(self, ip: str, token: str):
        self.ip = ip
        self.token = token
        self.device = None
        self._connect()

    def _connect(self):
        try:
            logging.info(f"正在尝试连接到台灯 {self.ip}...")
            self.device = MiotDevice(self.ip, self.token)
            status = self.device.info()
            logging.info(f"成功连接到台灯: {status}")
        except DeviceException as e:
            logging.error(f"无法连接到台灯 {self.ip}: {e}")
            self.device = None

    def turn_on(self) -> bool:
        if not self.device:
            logging.warning("设备未连接，无法开灯。")
            return False
        try:
            self.device.set_property_by(siid=2, piid=1, value=True)
            logging.info("台灯已打开。")
            return True
        except DeviceException as e:
            logging.error(f"开灯失败: {e}")
            return False

    def turn_off(self) -> bool:
        if not self.device:
            logging.warning("设备未连接，无法关灯。")
            return False
        try:
            self.device.set_property_by(siid=2, piid=1, value=False)
            logging.info("台灯已关闭。")
            return True
        except DeviceException as e:
            logging.error(f"关灯失败: {e}")
            return False

    def set_brightness(self, level: int) -> bool:
        if not self.device:
            logging.warning("设备未连接，无法设置亮度。")
            return False
        if not 1 <= level <= 100:
            logging.warning(f"无效的亮度值: {level}. 请输入 1-100 之间的整数。")
            return False
        try:
            self.device.set_property_by(siid=2, piid=2, value=level)
            logging.info(f"亮度已设置为 {level}%. ")
            return True
        except DeviceException as e:
            logging.error(f"设置亮度失败: {e}")
            return False

    def set_color_temperature(self, kelvin: int) -> bool:
        if not self.device:
            logging.warning("设备未连接，无法设置色温。")
            return False
        clamped_kelvin = max(self.MIN_COLOR_TEMP, min(kelvin, self.MAX_COLOR_TEMP))
        if clamped_kelvin != kelvin:
            logging.warning(f"色温值 {kelvin}K 超出范围，将使用最接近的值 {clamped_kelvin}K. ")

        try:
            self.device.set_property_by(siid=2, piid=3, value=clamped_kelvin)
            logging.info(f"色温已设置为 {clamped_kelvin}K. ")
            return True
        except DeviceException as e:
            logging.error(f"设置色温失败: {e}")
            return False

    def get_status(self) -> dict:
        if not self.device:
            logging.warning("设备未连接，无法获取状态。")
            return {}
        try:
            properties = [
                {'did': str(self.device.device_id), 'siid': 2, 'piid': 1}, # Switch
                {'did': str(self.device.device_id), 'siid': 2, 'piid': 2}, # Brightness
                {'did': str(self.device.device_id), 'siid': 2, 'piid': 3}  # Color Temp
            ]

            logging.info("获取状态功能需根据具体型号适配")
            return properties

        except Exception as e:
            logging.error(f"获取状态失败: {e}")
            return {}

