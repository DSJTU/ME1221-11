import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import os

# --- 配置 ---
MODEL_TFLITE_PATH = 'emotion_model_new.tflite'  # 确保此文件存在于当前目录
DLIB_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    print(f"错误：无法加载 Haar 级联文件 {HAAR_CASCADE_PATH}。请检查路径。")
    exit()

EMOTION_LABELS = np.array(["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
IMG_SIZE = (48, 48)  # 您的 Mini-Xception 模型要求的输入尺寸


# ---dlib 初始化 ---
try:
    # dlib只使用predictor，配合Haar的人脸框。
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
except RuntimeError:
    print(f"错误：无法加载 dlib 模型文件 {HAAR_CASCADE_PATH}。请检查路径。")
    exit()

# --- TFLite 解释器初始化 ---
try:
    # 1. 初始化 TFLite 解释器
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    print("TFLite 模型加载成功.")
except Exception as e:
    print(f"错误：TFLite 模型加载失败。请检查 {MODEL_TFLITE_PATH} 文件是否存在且完整。")
    print(f"详细错误: {e}")
    exit()

# 获取输入和输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 检查输入数据类型和形状
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
print(f"模型输入形状: {input_shape}, 数据类型: {input_dtype}")

INTERVAL_SECONDS = 10.0
emotion_counts = {label: 0 for label in EMOTION_LABELS}
start_time = time.time()
current_dominant_emotion = "Initializing"

TIRED_THRESHOLD = 0.23
TIRED_CONSEC_FRAMES = 3
tired_counter = 0
tired_sum = 0

# --- 辅助函数：计算眼睛纵横比 (EAR) ---
def eye_aspect_ratio(eye):
    # 计算垂直眼部地标之间的距离
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])

    # 计算水平眼部地标之间的距离
    c = np.linalg.norm(eye[0] - eye[3])

    # EAR 公式：(A + B) / (2 * C)
    ear = (a + b) / (2.0 * c)
    return ear

# --- 预处理函数 (适应 TFLite 的输入要求) ---
def preprocess_face_tflite(face_img):
    """将裁剪的人脸图像转换为模型所需的格式 (48x48 灰度, 归一化)。"""

    # 转换为灰度图 (确保)
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # 缩放到目标尺寸
    resized_face = cv2.resize(face_img, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 归一化和添加维度 (1, 48, 48, 1)
    # TFLite 模型的输入通常是 float32
    processed_face = resized_face.astype(np.float32) / 255.0
    processed_face = np.expand_dims(processed_face, axis=0)
    processed_face = np.expand_dims(processed_face, axis=-1)

    return processed_face

# --- 视频捕获初始化 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头。请检查摄像头连接和权限。")
    exit()

print("开始 PC 实时情绪检测...")
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 图像翻转，让画面更自然（可选）
    frame = cv2.flip(frame, 1)

    # 1. 人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_haar = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 缩放系数
        minNeighbors=5,  # 最小邻居数
        minSize=(30, 30)  # 最小检测尺寸
    )

    dominant_emotion = "No Face"
    max_prob = 0.0
    is_tired_in_frame = False

    for (x, y, w, h) in faces_haar:

        # 将 Haar 格式 (x, y, w, h) 转换为 dlib 格式的边界
        x1, y1, x2, y2 = x, y, x + w, y + h

        # 用于疲倦判定的Dlib特征点分析
        rect = dlib.rectangle(x1, y1, x2, y2)
        landmarks = predictor(gray, rect)

        # 转换为 NumPy 数组
        landmarks_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])

        # 提取左右眼点位 (Dlib 点位 36-41 是左眼, 42-47 是右眼)
        left_eye = landmarks_points[36:42]
        right_eye = landmarks_points[42:48]

        # 计算 EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < TIRED_THRESHOLD:
            is_tired_in_frame = True
            tired_counter += 1

        # 裁剪人脸区域（增加填充）
        margin = 20
        x1_p = max(0, x1 - margin)
        y1_p = max(0, y1 - margin)
        x2_p = min(frame.shape[1], x2 + margin)
        y2_p = min(frame.shape[0], y2 + margin)

        face_region = gray[y1_p:y2_p, x1_p:x2_p]

        if face_region.size == 0:
            continue

        # 1. 预处理
        processed_face = preprocess_face_tflite(face_region)

        # 2. *** TFLite 推理 ***
        interpreter.set_tensor(input_details[0]['index'], processed_face)
        interpreter.invoke()

        # 3. 获取输出结果
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # 获取最高概率的类别
        emotion_index = np.argmax(predictions)
        emotion_text = EMOTION_LABELS[emotion_index]
        confidence = predictions[emotion_index]

        # 记录当前帧检测到的情绪
        emotion_counts[emotion_text] += 1

        # 4. 绘制结果
        # 绘制人脸矩形框 (蓝色)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        realtime_text = f"{emotion_text}: {confidence:.2f}"

        cv2.putText(frame, realtime_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if not is_tired_in_frame:
        tired_sum = tired_counter
        tired_counter = 0

    elapsed_time = time.time() - start_time

    if elapsed_time >= INTERVAL_SECONDS:

        if tired_sum >= TIRED_CONSEC_FRAMES:
            current_dominant_emotion = "Tired"
            print(f"!!! 主导情绪判定为: Tired (连续闭眼 {tired_sum} 帧)")

        # 1. 找出占比最高的、持续时间最长的情绪
        elif any(emotion_counts.values()):
            max_count = 0
            dominant_emotion = "Neutral"  # 默认值

            # 遍历计数，找到最大值
            for emotion, count in emotion_counts.items():
                if count > max_count:
                    max_count = count
                    dominant_emotion = emotion

            # 更新要显示的主导情绪
            current_dominant_emotion = dominant_emotion

            print(f"--- 10s 周期结束 ---")
            print(f"情绪统计: {emotion_counts}")
            print(tired_sum)
            print(f"主导情绪更新为: {current_dominant_emotion}")

        else:
            # 如果 10 秒内没有检测到人脸
            current_dominant_emotion = "No Face Detected"

        # 2. 重置计数器和计时器
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        start_time = time.time()

    # --- 绘制主导情绪结果和倒计时 ---

    # 1. 主导情绪状态
    dominant_text = f"DOMINANT: {current_dominant_emotion}"
    cv2.putText(frame, dominant_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # 显示实时视频流
    cv2.imshow('Real-time Emotion Detector (TFLite)', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 资源释放
cap.release()
cv2.destroyAllWindows()