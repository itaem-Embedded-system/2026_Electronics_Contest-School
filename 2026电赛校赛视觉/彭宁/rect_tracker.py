# 导入必要的模块
import sensor        # OpenMV摄像头传感器模块，用于配置和控制摄像头
import time          # 时间模块，用于延时、计时等操作
from machine import UART
import json

# ======================== 全局变量配置 ========================
# 定义画面中心坐标 QQVGA 160,120
CENTER_X = 80
CENTER_Y = 60

# 定义是否开启自动曝光和自动增益
AUTO_EXPOSURE = True  # True: 开启自动曝光和增益，False: 关闭自动曝光和增益，使用手动设置的参数
# 定义手动曝光时间（微秒）和增益（dB），仅在AUTO_EXPOSURE=False时生效
EXPOSURE_TIME_US = 40000  # 曝光
GAIN_DB = 20              # 增益

# 定义矩形置信度过滤参数
RECT_THRESHOLD = 10000
# 定义矩形最小尺寸过滤参数
MIN_RECT_SIZE = 25

# 定义数据发送相关参数
# CHECK_TICK: 每隔多少帧发送一次数据
CHECK_TICK = 5
send_x = 0
send_y = 0
send_tick = 0
# 偏差纠正值（通信偏移量）
OFFSET = 100
# ======================== 摄像头初始化配置 ========================
# 重置摄像头
sensor.reset()
# 设置像素格式为灰度模式 (GRAYSCALE)，减少数据量，提升处理速度
sensor.set_pixformat(sensor.GRAYSCALE)
# 设置帧大小为QVGA (320x240)，兼顾分辨率和处理效率
sensor.set_framesize(sensor.QQVGA)
# 跳过初始的2000ms帧，让摄像头完成自动曝光/白平衡校准，稳定图像
sensor.skip_frames(time=2000)
# 创建时钟对象，用于计算帧率(FPS)
clock = time.clock()

if AUTO_EXPOSURE:
    # 开启自动曝光，手动设置曝光时间为20000微秒(20ms)，避免光线变化影响识别
    sensor.set_auto_exposure(True)
    # 开启自动增益，手动设置增益为10dB，固定图像亮度，提升识别稳定性
    sensor.set_auto_gain(True)
    # 再次跳过2000ms帧，让曝光/增益参数生效
    sensor.skip_frames(time=2000)
else:
    # 关闭自动曝光，手动设置曝光时间为20000微秒(20ms)，避免光线变化影响识别
    sensor.set_auto_exposure(False, exposure_us=40000)
    # 关闭自动增益，手动设置增益为10dB，固定图像亮度，提升识别稳定性
    sensor.set_auto_gain(False, gain_db=20)
    # 再次跳过2000ms帧，让手动曝光/增益参数生效
    sensor.skip_frames(time=2000)
#===================     串口     ==============================
def send_coord_packet(x, y):
    """构建坐标数据包"""
    packet = bytearray()
    packet.append(0x7E)  # 起始符
    # X坐标 (0-255范围内)
    packet.append(x & 0xFF)
    # Y坐标
    packet.append(y & 0xFF)
    # 结束符
    packet.append(0x7F)  # 结束符
    return packet

uart = UART(3, 9600)
# ======================== 主循环：矩形识别与可视化 ========================
while True:
    clock.tick()  # 更新时钟，用于计算帧率
    # 拍摄一张图像
    img = sensor.snapshot()
    rect = None          # 存储识别到的最大矩形对象
    rect_max_mag = 0     # 存储最大矩形的匹配度(置信度)

    # 遍历图像中所有识别到的矩形 (threshold=10000：过滤低置信度的矩形)
    for r in img.find_rects(threshold=RECT_THRESHOLD):
        # 过滤掉宽度或高度小于30像素的小矩形，减少误识别
        if r.w() < MIN_RECT_SIZE or r.h() < MIN_RECT_SIZE:
            continue
        # 只保留匹配度最高的矩形
        if r.magnitude() > rect_max_mag:
            rect = r                  # 更新为当前匹配度最高的矩形
            rect_max_mag = r.magnitude()  # 更新最大匹配度

    # 如果没有识别到有效矩形
    if not rect:
        # print("no rect")          # 打印提示信息
        continue                  # 跳过后续逻辑，进入下一次循环

    # ======================== 计算矩形中心与画面中心的偏差 ========================
    # 计算识别到的矩形中心X坐标 (矩形左上角X + 宽度/2)
    rect_center_x = rect.x() + rect.w()/2
    # 计算识别到的矩形中心Y坐标 (矩形左上角Y + 高度/2)
    rect_center_y = rect.y() + rect.h()/2

    # 计算X轴偏差：矩形中心X - 画面中心X (正值偏右，负值偏左)
    relative_coordinate_x = CENTER_X - rect_center_x
    # 计算Y轴偏差：矩形中心Y - 画面中心Y (正值偏下，负值偏上)
    relative_coordinate_y = rect_center_y - CENTER_Y

    # # # 打印偏差值，用于调试
    # print(f"矩形中心坐标: ({rect_center_x:.1f}, {rect_center_y:.1f}) | 偏差: dx={relative_coordinate_x:.1f}, dy={relative_coordinate_y:.1f}")

    # ======================== 数据发送 ========================
    send_tick += 1  # 计数器增加
    send_x += relative_coordinate_x  # 累加X轴偏差值
    send_y += relative_coordinate_y  # 累加Y轴偏差值
    if send_tick >= CHECK_TICK:
        # 计算平均偏差值
        send_x = send_x / send_tick
        send_y = send_y / send_tick

        # 打印平均偏差值，用于调试
        print(f"平均偏差: dx={send_x:.1f}, dy={send_y:.1f}")
        # 重置计数器和累加值
        send_tick = 0
        send_x = 0
        send_y = 0
    
    # 通信
    x_pos = int(relative_coordinate_x) + OFFSET
    y_pos = int(relative_coordinate_y) + OFFSET
    uart.write(f"{x_pos},{y_pos},{OFFSET}\r\n")
    # ======================== 图像可视化标注 (调试用) ========================
    # 用红色矩形框标出识别到的矩形
    img.draw_rectangle(rect.rect(), color=(255, 0, 0))
    # 用绿色圆圈标出矩形的四个角点
    for p in rect.corners():
        img.draw_circle(p[0], p[1], 5, color=(0, 255, 0))
    # 绘制画面中心十字线，方便对比
    # img.draw_cross(CENTER_X, CENTER_Y, color=(255, 255, 0), size=40)

    # 打印帧率
    # print(f"FPS: {clock.fps():.1f}")



