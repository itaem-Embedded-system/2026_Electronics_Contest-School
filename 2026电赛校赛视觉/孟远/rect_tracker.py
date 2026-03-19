# 导入必要的模块
import sensor        # OpenMV摄像头传感器模块，用于配置和控制摄像头
import time          # 时间模块，用于延时、计时等操作
from machine import UART
from ulab import numpy as np

# ======================== 卡尔曼滤波类实现 ========================
class KalmanFilter:
    def __init__(self, initial_x=0, initial_y=0):
        # 初始化状态：x坐标, y坐标, x速度, y速度
        self.x = initial_x
        self.y = initial_y
        self.vx = 0.0
        self.vy = 0.0

        # 状态协方差矩阵P，初始化为单位矩阵，描述状态的不确定性
        self.P = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

        # 过程噪声协方差矩阵Q，控制模型噪声的大小，值越小越信任模型
        self.Q = np.array([[0.1, 0.0, 0.0, 0.0],
                           [0.0, 0.1, 0.0, 0.0],
                           [0.0, 0.0, 0.1, 0.0],
                           [0.0, 0.0, 0.0, 0.1]])

        # 测量噪声协方差矩阵R（默认值），控制测量噪声的大小，值越小越信任测量值
        self.R_default = np.array([[0.1, 0.0],
                                   [0.0, 0.1]])
        self.R = self.R_default.copy()

        # 状态转移矩阵A，描述状态随时间的变化规律
        self.A = np.array([[1.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 1.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

        # 测量矩阵H，将状态空间映射到测量空间
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

    def predict(self):
        # 预测下一时刻的状态
        state = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        # 修复：@替换为np.dot
        state = np.dot(self.A, state)

        # 更新状态值
        self.x = state[0][0]
        self.y = state[1][0]
        self.vx = state[2][0]
        self.vy = state[3][0]

        # 预测协方差矩阵，更新状态的不确定性
        # 修复：连续矩阵乘法嵌套np.dot，保证顺序不变
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return (self.x, self.y)

    def update(self, meas_x, meas_y):
        # 构造测量值向量
        z = np.array([[meas_x], [meas_y]])

        # 计算卡尔曼增益，平衡模型预测和测量值的权重
        # 修复：@替换为np.dot
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 使用测量值更新状态
        state = np.array([[self.x], [self.y], [self.vx], [self.vy]])
        # 修复：@替换为np.dot
        state = state + np.dot(K, (z - np.dot(self.H, state)))

        self.x = state[0][0]
        self.y = state[1][0]
        self.vx = state[2][0]
        self.vy = state[3][0]

        # 更新协方差矩阵，降低状态的不确定性
        # 修复：@替换为np.dot
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

        return (self.x, self.y)

        #重置R矩阵到默认值，在没有检测到矩形时使用
    def reset_R(self):
        self.R = self.R_default.copy()

# ======================== 空心矩形检测函数 ========================  (检测矩形是否为空心矩形通过比较矩形边缘和内部的亮度差异来判断)

def check_hollow_rect(img, rect, edge_ratio=0.15, brightness_thresh=80):
    x, y, w, h = rect.rect()

    # 计算内部区域（去除边缘）
    edge_w = int(w * edge_ratio)
    edge_h = int(h * edge_ratio)

    inner_x = x + edge_w
    inner_y = y + edge_h
    inner_w = w - 2 * edge_w
    inner_h = h - 2 * edge_h

    # 边界检查，确保ROI在图像范围内
    inner_x = max(0, min(inner_x, img.width() - 1))
    inner_y = max(0, min(inner_y, img.height() - 1))
    inner_w = min(inner_w, img.width() - inner_x)
    inner_h = min(inner_h, img.height() - inner_y)

    # 如果内部区域太小，无法判断
    if inner_w <= 3 or inner_h <= 3:
        return False, None

    # 获取内部区域平均亮度
    stats = img.get_statistics(roi=(inner_x, inner_y, inner_w, inner_h))
    inner_brightness = stats.mean()

    # 判断是否为空心矩形：内部亮度高于阈值（内部较亮，说明是空心）
    is_hollow = inner_brightness > brightness_thresh

    return is_hollow, (inner_x, inner_y, inner_w, inner_h)

# ======================== 全局变量配置 ========================
# 定义画面中心坐标 (QQVGA分辨率为160x120，中心为80,60)
CENTER_X = 80
CENTER_Y = 60

# 定义是否开启自动曝光和自动增益
AUTO_EXPOSURE = True  # True: 开启自动曝光和增益，False: 关闭自动曝光和增益，使用手动设置的参数
# 定义手动曝光时间（微秒）和增益（dB），仅在AUTO_EXPOSURE=False时生效
EXPOSURE_TIME_US = 40000  # 曝光时间
GAIN_DB = 20              # 增益值

# 定义矩形置信度过滤参数，过滤低置信度的矩形
RECT_THRESHOLD = 8000
# 定义矩形最小尺寸过滤参数，过滤过小的矩形，减少误识别
MIN_RECT_SIZE = 25
# 定义矩形最大尺寸过滤参数，过滤过大的矩形，减少误识别
MAX_RECT_SIZE = 80
# 定义宽高比过滤参数，过滤宽高比不合理的矩形，进一步减少误识别
ASPECT_RATIO_MIN = 0.3          # 最小宽高比（防止过于细长的矩形）
ASPECT_RATIO_MAX = 3.0          # 最大宽高比（防止过于扁平的矩形）

# ======================== 空心矩形检测参数 ========================
ONLY_HOLLOW_RECT = True          # 是否只检测空心矩形
HOLLOW_EDGE_RATIO = 0.15         # 边缘比例，用于计算内部区域
HOLLOW_BRIGHTNESS_THRESH = 80    # 内部亮度阈值，高于此值认为是空心

# 定义数据发送相关参数
# CHECK_TICK: 每隔多少帧发送一次平均偏差数据
CHECK_TICK = 5
send_x = 0
send_y = 0
send_tick = 0
# 偏差纠正值（通信偏移量）
OFFSET = 0

# 初始化卡尔曼滤波器，初始位置设为画面中心
kalman = KalmanFilter(initial_x=CENTER_X, initial_y=CENTER_Y)

# ======================== 摄像头初始化配置 ========================
# 重置摄像头
sensor.reset()
# 设置像素格式为灰度模式 (GRAYSCALE)，减少数据量，提升处理速度
sensor.set_pixformat(sensor.GRAYSCALE)
# 设置帧大小为QQVGA (160x120)，兼顾分辨率和处理效率
sensor.set_framesize(sensor.QQVGA)
# 跳过初始的2000ms帧，让摄像头完成自动曝光/白平衡校准，稳定图像
sensor.skip_frames(time=2000)
# 创建时钟对象，用于计算帧率(FPS)
clock = time.clock()

if AUTO_EXPOSURE:
    # 开启自动曝光
    sensor.set_auto_exposure(True)
    # 开启自动增益
    sensor.set_auto_gain(True)
    # 再次跳过2000ms帧，让曝光/增益参数生效
    sensor.skip_frames(time=2000)
else:
    # 关闭自动曝光，手动设置曝光时间
    sensor.set_auto_exposure(False, exposure_us=EXPOSURE_TIME_US)
    # 关闭自动增益，手动设置增益
    sensor.set_auto_gain(False, gain_db=GAIN_DB)
    # 再次跳过2000ms帧，让手动曝光/增益参数生效
    sensor.skip_frames(time=2000)

# ======================== 串口初始化 ========================
uart = UART(3, 9600)

# ======================== 主循环：矩形识别与跟踪 ========================
while True:
    clock.tick()  # 更新时钟，用于计算帧率
    # 拍摄一张图像
    img = sensor.snapshot()
    rect = None          # 存储识别到的最大矩形对象
    rect_max_mag = 0     # 存储最大矩形的匹配度(置信度)
    inner_roi = None     # 用于存储空心矩形的内部区域，便于可视化

    # 遍历图像中所有识别到的矩形
    for r in img.find_rects(threshold=RECT_THRESHOLD):
        # 过滤掉宽度或高度小于MIN_RECT_SIZE的小矩形，减少误识别
        if r.w() < MIN_RECT_SIZE or r.h() < MIN_RECT_SIZE:
            continue
        # 过滤掉宽度或高度大于MAX_RECT_SIZE的大矩形，减少误识别
        if r.w() > MAX_RECT_SIZE or r.h() > MAX_RECT_SIZE:
            continue
        # 计算宽高比并进行双向过滤，排除过于细长或扁平的矩形
        aspect = r.w() / r.h()
        if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX:
            continue

        # ======================== 空心矩形检测 ========================
        if ONLY_HOLLOW_RECT:
            # 调用空心矩形检测函数，判断当前矩形是否为空心
            is_hollow, current_inner_roi = check_hollow_rect(
                img, r,
                edge_ratio=HOLLOW_EDGE_RATIO,
                brightness_thresh=HOLLOW_BRIGHTNESS_THRESH
            )
            # 如果不是空心矩形，则跳过
            if not is_hollow:
                continue
            # 保存内部区域坐标，用于可视化
            inner_roi = current_inner_roi

        # 只保留匹配度最高的矩形
        if r.magnitude() > rect_max_mag:
            rect = r                  # 更新为当前匹配度最高的矩形
            rect_max_mag = r.magnitude()  # 更新最大匹配度

    # 卡尔曼滤波预测下一时刻的位置
    pred_x, pred_y = kalman.predict()
    # 初始化滤波后的坐标为预测值
    filtered_x, filtered_y = pred_x, pred_y

    # 可以根据矩形大小动态调整测量噪声
    if rect:
        # 矩形越大，测量越可信，噪声越小
        size_factor = min(1.0, (rect.w() * rect.h()) / (MAX_RECT_SIZE * MAX_RECT_SIZE))
        kalman.R = np.array([[0.1 * (1 - size_factor), 0.0],
                             [0.0, 0.1 * (1 - size_factor)]])
    else:
        # 没有检测到矩形时，将R矩阵重置为默认值，避免使用错误的噪声参数
        kalman.reset_R()

    # 如果识别到有效矩形
    if rect:
        # 计算识别到的矩形中心X坐标 (矩形左上角X + 宽度/2)
        rect_center_x = rect.x() + rect.w()/2
        # 计算识别到的矩形中心Y坐标 (矩形左上角Y + 高度/2)
        rect_center_y = rect.y() + rect.h()/2

        # 使用测量值更新卡尔曼滤波器，得到滤波后的位置
        filtered_x, filtered_y = kalman.update(rect_center_x, rect_center_y)

        # ======================== 计算矩形中心与画面中心的偏差 ========================
        # X轴偏差：画面中心X - 滤波后的矩形中心X (正值表示矩形在画面左侧，负值表示在右侧)
        relative_coordinate_x = CENTER_X - filtered_x
        # Y轴偏差：滤波后的矩形中心Y - 画面中心Y (正值表示矩形在画面下方，负值表示在上方)
        relative_coordinate_y = filtered_y - CENTER_Y

        # ======================== 数据发送 ========================
        send_tick += 1  # 计数器增加
        send_x += relative_coordinate_x  # 累加X轴偏差值
        send_y += relative_coordinate_y  # 累加Y轴偏差值
        if send_tick >= CHECK_TICK:
            # 计算平均偏差值
            send_x_avg = send_x / send_tick
            send_y_avg = send_y / send_tick

            # 打印平均偏差值，用于调试
            print(f"平均偏差: dx={send_x_avg:.1f}, dy={send_y_avg:.1f}")
            # 重置计数器和累加值
            send_tick = 0
            send_x = 0
            send_y = 0

        # 发送当前偏差数据
        x_pos = int(relative_coordinate_x) + OFFSET
        y_pos = int(relative_coordinate_y) + OFFSET
        uart.write(f"{x_pos},{y_pos},{OFFSET}\r\n")

        # ======================== 图像可视化标注 (调试用) ========================
        # 用白色矩形框标出识别到的矩形
        img.draw_rectangle(rect.rect(), color=255)

        # ======================== 新增：绘制内部区域 ========================
        # 如果启用了空心矩形检测，用灰色矩形框标出内部检测区域
        if ONLY_HOLLOW_RECT and inner_roi:
            img.draw_rectangle(inner_roi, color=127)

        # 用灰色圆圈标出矩形的四个角点
        for p in rect.corners():
            img.draw_circle(p[0], p[1], 5, color=127)
        # 用黑色圆圈标出滤波后的矩形中心
        img.draw_circle(int(filtered_x), int(filtered_y), 5, color=0)
    else:
        # 没有识别到矩形时，使用预测的位置计算偏差
        relative_coordinate_x = CENTER_X - pred_x
        relative_coordinate_y = pred_y - CENTER_Y
        # 发送预测的偏差数据
        x_pos = int(relative_coordinate_x) + OFFSET
        y_pos = int(relative_coordinate_y) + OFFSET
        uart.write(f"{x_pos},{y_pos},{OFFSET}\r\n")
        # 用深灰色圆圈标出预测的中心
        img.draw_circle(int(pred_x), int(pred_y), 5, color=64)

    # 绘制画面中心十字线，方便对比
    img.draw_cross(CENTER_X, CENTER_Y, color=127, size=40)

    # # 打印当前帧率（可取消注释用于调试）
    # print(f"FPS: {clock.fps():.1f}")
