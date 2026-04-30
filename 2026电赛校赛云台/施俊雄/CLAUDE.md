# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

2025 全国大学生电子设计竞赛 E 题——简易自瞄准装置。STM32F103C8T6 驱动的双轴步进电机云台，搭载 OpenMV 摄像头和激光笔，视觉闭环跟踪矩形靶标。

## 构建命令

```bash
# Debug 构建
cmake --preset Debug && cmake --build build/Debug

# Release 构建
cmake --preset Release && cmake --build build/Release
```

工具链：`arm-none-eabi-gcc`，构建系统：CMake + Ninja，预设配置在 [CMakePresets.json](CMakePresets.json)。

另支持 Keil MDK-ARM 打开 `MDK-ARM/25EYunTai.uvprojx`（Armcc V5.06，`MDK-ARM/` 已在 .gitignore 中排除）。

## 工程架构

```
App/Serial/    视觉协议解析（USART3，OpenMV） + 搜索状态机
App/Chassis/   底盘转弯通信（USART1，单字节 'T'/'E' 协议）
App/PID/       PID 结构体与 Date() 增量式控制器（当前跟踪链不使用 PID）
App/OLED/      SSD1306 I2C 模拟驱动（PB8/PB9）
App/Vofa/      [废弃] VOFA+ 上位机调参（USART1 已挪作底盘通信）
Stepper/       步进电机抽象层（stepper_core + stepper_port_hal）
Core/          CubeMX 生成的 HAL 代码（main.c 是核心控制逻辑）
Drivers/       CMSIS + STM32F1xx HAL 驱动库
cmake/         工具链文件 + CubeMX 生成的 CMake 子目录
```

**main.c 是宇宙中心**：所有控制逻辑（跟踪、搜索、转弯、看门狗、冷却）都在 `main()` 的 while(1) 循环中，使用非阻塞步进电机 API。

## 关键引脚

| 功能 | 引脚 |
|------|------|
| Y轴 STEP/DIR/EN | PA0 / PA1 / PA6 |
| X轴 STEP/DIR/EN | PA2 / PA3 / PA5 |
| USART1 TX/RX (底盘转弯) | PA9 / PA10 |
| USART3 TX/RX (视觉) | PB10 / PB11 |
| OLED SCL/SDA | PB8 / PB9 |

## 视觉数据协议（USART3, 115200bps）

OpenMV 下发格式：`X,Y,const\r\n`，其中 X/Y 为光斑到靶心的像素偏差（浮点），const=1 表示识别到矩形靶，const=0 表示未识别。

解析状态机在 `serial_process_byte()`（[Serial.c:97-198](App/Serial/Serial.c#L97-L198)）：5 态解析，支持浮点小数点和负号，State 4 校验（|X|<160, |Y|<120）通过后原子写入全局 `Serial_X/Serial_Y/Serial_Const` 并置 `Serial_RxFlag=1`。ISR 使用双缓冲临时变量，防止无效帧污染全局值。

## 主循环控制流程（main.c while(1)）

每个主循环迭代按以下顺序执行：

1. **底盘转弯命令消费**：消费 USART1 的 'T'/'E' 单字节帧，'T' 进入强制转弯（X 轴盲走 TURN_RPM），'E' 退出转弯恢复视觉跟踪
2. **视觉帧消费**（`Serial_GetRxFlag()`）：首帧到达启动搜索，const=1 时 `error_to_rpm()` 纯比例映射 + D 项阻尼驱动电机，const≠1 不在此处处理
3. **通信看门狗**：首帧后 500ms 无帧 → 强制 `Serial_Const=0`
4. **搜索/丢靶冷却**：搜索态调 `Search_Task()`，丢靶态冷却 0.5 秒后重启搜索，转弯脱靶时 Y 停车 + X 盲走
5. **转弯兜底完成检测**：'E' 信号丢失时位置差值自动结束转弯
6. **`run_nonblock()` ×2**：无条件驱动 X/Y 电机脉冲

`run_nonblock()` 在所有条件块外部调用——若放在 `Serial_GetRxFlag` 内部，脉冲频率会被 OpenMV 帧率（~50Hz）限制。

## 控制策略

**error_to_rpm()**（[main.c:148-161](Core/Src/main.c#L148-L161)）：纯比例映射，死区 0px，比例段 0→70px 对应 0→30 RPM，70px+ 饱和 30 RPM。方向永远正比于误差符号。

**D 项**：一阶差分 `KD*(e(k)-e(k-1))` 叠加在 RPM 上，KD 默认 0.02（[main.c:70](Core/Src/main.c#L70) `PID_K[2]` 和 `PID_K[5]`），`d_history_ready` 标志在搜索锁定/转弯进出时复位。

**搜索（[Serial.c:263-377](App/Serial/Serial.c#L263-L377)）**：定速旋转扫描，Y 轴不动，X 轴 20 RPM 单方向扫描 3.4s 后蛇形折返换向。搜索函数不读 `Serial_Const`，由主循环判断何时停止。

**丢靶冷却**（[main.c:101](Core/Src/main.c#L101) `LOST_COOLDOWN_US`）：0.5 秒冷却，冷却期间电机减速到一半，到期后重启搜索。

**步进电机**：3600 步/转（1/16 细分），`set_speed_nonblock()` 将 RPM 转换为脉冲间隔，`run_nonblock()` 按间隔产生 STEP 脉冲。相位仅在停止→启动或方向翻转且旧间隔到期时重置。钳位参数：min=0.3 RPM, max=3000 RPM。

## 时序基础设施

TIM2（72MHz/72=1MHz，1 tick=1μs）提供 `get_tick_us()`：16 位计数器 + 高字累加 → 32 位时间戳。约束：两次调用间隔 < 65ms（否则漏溢出）。所有超时比较使用 `(int32_t)(a - b) < 0` 模式防溢出。

## 与参考工程的关键差异

参考工程使用 SPL + Keil MDK + PID(Date) + speed_adjust 四级调速，RPM 范围 0~33.75。当前工程使用 HAL + CMake + GCC + 纯比例映射，RPM 范围 0~30（70px 饱和），新增搜索/冷却/转弯/看门狗/UART 错误恢复等机制。

## 注意事项

- `Serial_Const=1` 仅表示视觉模块识别到矩形靶，不保证光斑落在靶面上
- `Serial_RxPacket[]` 在 `Serial.h:8` 声明但无定义（历史遗留），不可引用
- `MDK-ARM/` 和 `.claude/` 在 .gitignore 中，不会被提交
- HAL_UART_ErrorCallback 自动恢复 ORE/FE 等硬件错误后的接收中断
