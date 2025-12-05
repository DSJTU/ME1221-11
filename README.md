# ME1221-11 情绪感应智能灯
课程《工程学导论》第二小组的项目代码。

## 主程序
#!/usr/bin/env python3
"""
智能情绪感应灯 - 完整版
硬件配置：
  - 树莓派4B + 500万像素摄像头模块
  - 米家智能台灯 MJTD06YL (蓝牙连接)
功能流程：
  摄像头捕获 -> DeepFace情绪识别 -> 情绪映射 -> 蓝牙控制台灯
"""

import sys
import time
import logging
import cv2
import numpy as np
from bluepy import btle
from deepface import DeepFace

# ============ 第一部分：配置参数 ============
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 摄像头配置
CAMERA_WIDTH = 640      # 图像宽度，降低分辨率以提高处理速度
CAMERA_HEIGHT = 480     # 图像高度
FRAME_SKIP = 5          # 每5帧处理一次，减少计算负担

# 蓝牙设备配置（必须修改！）
DEVICE_MAC = "AA:BB:CC:DD:EE:FF"  # 替换为您的台灯蓝牙MAC地址
SERVICE_UUID = "0000fe01-0000-1000-8000-00805f9b34fb"  # 蓝牙服务UUID
CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"     # 控制特征值UUID

# ============ 第二部分：蓝牙台灯控制器 ============

class BluetoothLampController:
    """米家床头灯2蓝牙控制器"""
    
    def __init__(self):
        self.device = None
        self.control_char = None
        self._connect()
    
    def _connect(self):
        """建立蓝牙连接"""
        try:
            logger.info(f"正在连接蓝牙设备 {DEVICE_MAC}...")
            self.device = btle.Peripheral(DEVICE_MAC, addrType=btle.ADDR_TYPE_RANDOM)
            
            # 获取服务和控制特征
            service = self.device.getServiceByUUID(SERVICE_UUID)
            self.control_char = service.getCharacteristics(CHAR_UUID)[0]
            
            logger.info("✅ 蓝牙设备连接成功")
        except Exception as e:
            logger.error(f"❌ 蓝牙连接失败: {e}")
            sys.exit(1)
    
    def set_light(self, brightness=50, rgb=(255, 255, 255)):
        """
        设置灯光参数
        
        参数：
          brightness: 亮度 (0-100)
          rgb: 颜色元组 (R, G, B)
        """
        try:
            # 构建控制指令（需要根据实际协议调整）
            # 示例格式：[起始符, 亮度, R, G, B]
            cmd = bytearray([0xAA])  # 假设起始符
            cmd.append(brightness)   # 亮度
            cmd.append(rgb[0])       # 红色
            cmd.append(rgb[1])       # 绿色
            cmd.append(rgb[2])       # 蓝色
            
            # 发送指令
            self.control_char.write(cmd)
            logger.debug(f"灯光设置: 亮度{brightness}%, RGB{rgb}")
            
        except Exception as e:
            logger.error(f"控制指令失败: {e}")
            # 尝试重新连接
            try:
                self._connect()
            except:
                logger.error("重新连接失败")

# ============ 第三部分：情绪识别与映射 ============

class EmotionProcessor:
    """情绪处理器：识别情绪并映射到灯光参数"""
    
    def __init__(self):
        # DeepFace支持的7种基本情绪
        self.supported_emotions = ['angry', 'disgust', 'fear', 'happy', 
                                   'sad', 'surprise', 'neutral']
        
        # 情绪到灯光参数的映射（基于您的要求）
        self.emotion_to_light = {
            # DeepFace原生情绪 -> (亮度, RGB颜色)
            'happy':     (85, (255, 200, 100)),    # 开心: 85%亮度, 暖橙色
            'neutral':   (65, (220, 230, 255)),    # 平静/中性: 65%亮度, 淡蓝色
            'sad':       (45, (150, 180, 255)),    # 低落: 45%亮度, 冷蓝色
            'angry':     (55, (255, 100, 100)),    # 烦躁: 55%亮度, 浅红色
            # 其他情绪的默认映射
            'surprise':  (70, (255, 255, 200)),    # 惊讶: 70%亮度, 淡黄色
            'fear':      (40, (100, 100, 200)),    # 恐惧: 40%亮度, 蓝色
            'disgust':   (50, (150, 200, 100)),    # 厌恶: 50%亮度, 黄绿色
            'default':   (65, (255, 255, 255))     # 默认: 65%亮度, 白色
        }
        
        # 专注和疲惫状态检测（基于连续帧分析）
        self.emotion_history = []  # 存储最近的情绪历史
        self.history_max_len = 20  # 历史记录最大长度
    
    def detect_emotion(self, frame):
        """
        使用DeepFace检测图像中的情绪
        
        参数:
          frame: 摄像头捕获的图像帧
          
        返回:
          情绪字符串 (如 'happy', 'sad', 'neutral' 等)
        """
        try:
            # 使用DeepFace分析情绪
            # enforce_detection=False 允许在没有检测到人脸时继续运行
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True  # 减少输出
            )
            
            # 提取主要情绪
            if isinstance(analysis, list) and len(analysis) > 0:
                emotion_result = analysis[0]
                dominant_emotion = emotion_result.get('dominant_emotion', 'neutral')
                
                # 获取情绪置信度
                emotion_scores = emotion_result.get('emotion', {})
                confidence = emotion_scores.get(dominant_emotion, 0)
                
                # 只接受置信度高于阈值的识别结果
                if confidence > 20:  # 置信度阈值20%
                    logger.debug(f"检测到情绪: {dominant_emotion} (置信度: {confidence:.1f}%)")
                    
                    # 更新情绪历史
                    self._update_emotion_history(dominant_emotion)
                    
                    # 分析连续情绪状态（检测专注/疲惫）
                    special_state = self._analyze_emotion_pattern()
                    if special_state:
                        return special_state
                    
                    return dominant_emotion
            
            return 'neutral'
            
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return 'neutral'
    
    def _update_emotion_history(self, emotion):
        """更新情绪历史记录"""
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.history_max_len:
            self.emotion_history.pop(0)
    
    def _analyze_emotion_pattern(self):
        """
        分析情绪历史，检测特殊状态
        
        返回:
          'focused': 专注状态（连续多帧中性/平静）
          'tired': 疲惫状态（连续多帧低落/变化缓慢）
        """
        if len(self.emotion_history) < 10:
            return None
        
        # 检查是否专注（最近10帧中有8帧以上是中性/平静）
        recent = self.emotion_history[-10:]
        neutral_count = recent.count('neutral')
        calm_count = recent.count('happy')  # 将开心也视为平静状态
        
        if neutral_count + calm_count >= 8:
            return 'focused'
        
        # 检查是否疲惫（情绪变化缓慢，多为低落）
        sad_count = recent.count('sad')
        if sad_count >= 6:
            return 'tired'
        
        return None
    
    def map_emotion_to_light(self, emotion):
        """
        将情绪映射到灯光参数
        
        参数:
          emotion: 情绪字符串
          
        返回:
          (亮度, RGB颜色) 元组
        """
        # 特殊状态映射
        if emotion == 'focused':
            return (95, (255, 255, 255))  # 专注: 95%亮度, 纯白色
        elif emotion == 'tired':
            return (45, (255, 180, 80))   # 疲惫: 45%亮度, 暖黄色
        
        # 基本情绪映射
        return self.emotion_to_light.get(emotion, self.emotion_to_light['default'])

# ============ 第四部分：主程序 ============

def main():
    """主控制循环"""
    
    logger.info("=" * 60)
    logger.info("智能情绪感应灯 - 完整系统启动")
    logger.info(f"摄像头分辨率: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    logger.info("按 'q' 键退出程序")
    logger.info("=" * 60)
    
    # 初始化组件
    logger.info("初始化摄像头...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        sys.exit(1)
    
    logger.info("初始化情绪处理器...")
    emotion_processor = EmotionProcessor()
    
    logger.info("初始化蓝牙台灯控制器...")
    lamp_controller = BluetoothLampController()
    
    # 状态变量
    frame_count = 0
    last_emotion = None
    last_light_params = None
    last_emotion_time = time.time()
    
    logger.info("✅ 系统初始化完成，开始情绪感应...")
    
    try:
        while True:
            # 1. 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 2. 定期进行情绪识别（跳过一些帧以提高性能）
            current_emotion = None
            if frame_count % FRAME_SKIP == 0:
                # 转换颜色空间（DeepFace需要RGB）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 识别情绪
                current_emotion = emotion_processor.detect_emotion(rgb_frame)
                
                # 3. 如果检测到新情绪，控制台灯
                if current_emotion and current_emotion != last_emotion:
                    # 情绪映射到灯光参数
                    brightness, color = emotion_processor.map_emotion_to_light(current_emotion)
                    
                    # 控制台灯
                    lamp_controller.set_light(brightness=brightness, rgb=color)
                    
                    # 更新状态
                    last_emotion = current_emotion
                    last_light_params = (brightness, color)
                    last_emotion_time = time.time()
                    
                    logger.info(f"情绪变化: {current_emotion} -> 亮度{brightness}%")
            
            # 4. 在画面上显示信息
            display_frame = frame.copy()
            
            # 显示当前情绪
            if last_emotion:
                emotion_text = f"Emotion: {last_emotion}"
                cv2.putText(display_frame, emotion_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示灯光状态
            if last_light_params:
                brightness, color = last_light_params
                light_text = f"Light: {brightness}%"
                cv2.putText(display_frame, light_text, (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 显示颜色预览
                color_preview = np.zeros((30, 30, 3), dtype=np.uint8)
                color_preview[:, :] = color
                display_frame[20:50, CAMERA_WIDTH-50:CAMERA_WIDTH-20] = color_preview
                cv2.rectangle(display_frame, 
                             (CAMERA_WIDTH-50, 20), 
                             (CAMERA_WIDTH-20, 50), 
                             (255, 255, 255), 2)
            
            # 显示帧率
            fps_text = f"FPS: {int(frame_count / (time.time() - last_emotion_time + 0.001))}"
            cv2.putText(display_frame, fps_text, (CAMERA_WIDTH-150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
            # 显示提示
            cv2.putText(display_frame, "Press 'q' to quit", (20, CAMERA_HEIGHT-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # 5. 显示画面
            cv2.imshow('Emotion Sensing Light', display_frame)
            
            # 6. 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("收到退出信号")
                break
            
            # 7. 性能调节：控制帧率
            time.sleep(0.01)  # 小延迟，避免过度占用CPU
            
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
    finally:
        # 清理资源
        logger.info("正在清理资源...")
        
        # 设置柔和灯光
        lamp_controller.set_light(brightness=30, rgb=(255, 255, 200))
        
        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info("程序结束")

# ============ 程序入口 ============
if __name__ == "__main__":
    # 安装依赖指南：
    # 1. 更新系统: sudo apt-get update && sudo apt-get upgrade
    # 2. 安装OpenCV: sudo apt-get install python3-opencv
    # 3. 安装DeepFace: pip install deepface
    # 4. 安装蓝牙库: sudo apt-get install bluez && pip install bluepy
    # 5. 启用摄像头: sudo raspi-config -> Interface Options -> Camera -> Enable
    # 6. 获取蓝牙MAC: sudo hcitool lescan
    # 7. 运行程序: python3 emotion_camera_light.py
    
    print("智能情绪感应灯 - 完整系统")
    print("=" * 50)
    print("确保已安装以下依赖：")
    print("  - OpenCV (python3-opencv)")
    print("  - DeepFace (pip install deepface)")
    print("  - bluepy (pip install bluepy)")
    print("  - 已启用树莓派摄像头")
    print("=" * 50)
    
    main()



## 前置代码
# 更新系统
sudo apt-get update && sudo apt-get upgrade

# 安装OpenCV
sudo apt-get install python3-opencv

# 安装DeepFace（可能需要较长时间）
pip install deepface

# 安装蓝牙库
sudo apt-get install bluez
pip install bluepy  

sudo raspi-config
# 选择 Interface Options -> Camera -> Enable

# 扫描蓝牙设备
sudo hcitool lescan
# 找到您的台灯并记录MAC地址

# 查找服务UUID（可能需要蓝牙调试工具）

python3 emotion_camera_light.py
