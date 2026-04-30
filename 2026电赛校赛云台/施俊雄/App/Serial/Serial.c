#include "Serial.h"
#include "stm32f1xx_hal.h"
#include "usart.h"
#include <stdarg.h>
#include <stdio.h>

uint8_t Serial_TxPacket[5];

volatile float Serial_X = 0.0f;
volatile float Serial_Y = 0.0f;
volatile int32_t Serial_Const = 0;
volatile uint8_t Serial_RxFlag;

void Serial_Init(void) {
  /* 初始化内容已由 CubeMX (MX_USART3_UART_Init) 处理，这里留空以兼容主函数调用
   */
}

void Serial_SendByte(uint8_t Byte) {
  HAL_UART_Transmit(&huart3, &Byte, 1, 1000);
}

void Serial_SendArray(uint8_t *Array, uint16_t Length) {
  uint16_t i;
  for (i = 0; i < Length; i++) {
    Serial_SendByte(Array[i]);
  }
}

void Serial_SendString(char *String) {
  uint8_t i;
  for (i = 0; String[i] != '\0'; i++) {
    Serial_SendByte(String[i]);
  }
}

uint32_t Serial_Pow(uint32_t X, uint32_t Y) {
  uint32_t Result = 1;
  while (Y--) {
    Result *= X;
  }
  return Result;
}

void Serial_SendNumber(uint32_t Number, uint8_t Length) {
  uint8_t i;
  for (i = 0; i < Length; i++) {
    Serial_SendByte(Number / Serial_Pow(10, Length - i - 1) % 10 + '0');
  }
}

/* 如果与 HAL 的 syscalls.c 中的 _write 发生冲突，需要注释掉本函数 */
/*
int fputc(int ch, FILE *f)
{
    Serial_SendByte(ch);
    return ch;
}

void Serial_Printf(char *format, ...)
{
    char String[100];
    va_list arg;
    va_start(arg, format);
    vsprintf(String, format, arg);
    va_end(arg);
    Serial_SendString(String);
}
*/
void Serial_SendPacket(void) {
  Serial_SendArray(Serial_TxPacket, 5);
  Serial_SendByte('\r');
  Serial_SendByte('\n');
}

uint8_t Serial_GetRxFlag(void) {
  if (Serial_RxFlag == 1) {
    Serial_RxFlag = 0;
    return 1;
  }
  return 0;
}

static uint8_t s_rxState = 0;
static float s_curFloat = 0.0f;
static float s_fracWeight = 1.0f;
static uint8_t s_isNeg = 0;
/* 双缓冲临时变量：ISR 在 State 1-3 写入临时变量，
   State 4 校验通过后才原子提交到全局 Serial_X/Y/Const（防无效帧污染） */
static float s_tmpX = 0.0f;
static float s_tmpY = 0.0f;
static int32_t s_tmpConst = 0;

#define X_VALID_MAX 160.0f
#define Y_VALID_MAX 120.0f

void serial_process_byte(uint8_t RxData) {
  float digit;

  switch (s_rxState) {
  case 0:
    if (RxData >= '0' && RxData <= '9') {
      s_curFloat = (float)(RxData - '0');
      s_fracWeight = 1.0f;
      s_isNeg = 0;
      s_rxState = 1;
    } else if (RxData == '-') {
      s_curFloat = 0.0f;
      s_fracWeight = 1.0f;
      s_isNeg = 1;
      s_rxState = 1;
    }
    break;

  case 1:
    if (RxData >= '0' && RxData <= '9') {
      digit = (float)(RxData - '0');
      if (s_fracWeight >= 1.0f) {
        s_curFloat = s_curFloat * 10.0f + digit;
      } else {
        s_curFloat += digit * s_fracWeight;
        s_fracWeight *= 0.1f;
      }
    } else if (RxData == '.') {
      s_fracWeight = 0.1f;
    } else if (RxData == ',') {
      s_tmpX = s_isNeg ? -s_curFloat : s_curFloat;
      s_curFloat = 0.0f;
      s_fracWeight = 1.0f;
      s_isNeg = 0;
      s_rxState = 2;
    } else if (RxData == '\r') {
      s_rxState = 0;
    } else {
      s_rxState = 0;
    }
    break;

  case 2:
    if (RxData >= '0' && RxData <= '9') {
      digit = (float)(RxData - '0');
      if (s_fracWeight >= 1.0f) {
        s_curFloat = s_curFloat * 10.0f + digit;
      } else {
        s_curFloat += digit * s_fracWeight;
        s_fracWeight *= 0.1f;
      }
    } else if (RxData == '.') {
      s_fracWeight = 0.1f;
    } else if (RxData == '-') {
      s_curFloat = 0.0f;
      s_fracWeight = 1.0f;
      s_isNeg = 1;
    } else if (RxData == ',') {
      s_tmpY = s_isNeg ? -s_curFloat : s_curFloat;
      s_curFloat = 0.0f;
      s_fracWeight = 1.0f;
      s_isNeg = 0;
      s_rxState = 3;
    } else if (RxData == '\r') {
      s_rxState = 0;
    } else {
      s_rxState = 0;
    }
    break;

  case 3:
    if (RxData >= '0' && RxData <= '9') {
      s_curFloat = s_curFloat * 10.0f + (float)(RxData - '0');
    } else if (RxData == '-') {
      s_curFloat = 0.0f;
      s_fracWeight = 1.0f;
      s_isNeg = 1;
    } else if (RxData == '\r') {
      s_tmpConst = (int32_t)(s_isNeg ? -s_curFloat : s_curFloat);
      s_rxState = 4;
    } else {
      s_rxState = 0;
    }
    break;

  case 4:
    if (RxData == '\n') {
      if (s_tmpX > -X_VALID_MAX && s_tmpX < X_VALID_MAX &&
          s_tmpY > -Y_VALID_MAX && s_tmpY < Y_VALID_MAX) {
        Serial_X = s_tmpX;
        Serial_Y = s_tmpY;
        Serial_Const = s_tmpConst;
        Serial_RxFlag = 1;
      }
    }
    s_rxState = 0;
    break;

  default:
    s_rxState = 0;
    break;
  }
}

/* ======================== 搜索任务 ======================== */
#include "stepper_core.h"

/* ===== [废弃] 旧 Lissajous 搜索 =====
 *
 * 原搜索方案：X正弦 + Y三角波 Lissajous 轨迹，两阶段（小范围→大范围）
 * 已被单行定速旋转替代（见下方新代码）。保留以供回退参考。
 *
 * 恢复方法：取消本注释块，删除下方新搜索代码。
 *
 * #include <math.h>
 *
 * typedef enum {
 *     SEARCH_IDLE = 0,
 *     SEARCH_PHASE1,
 *     SEARCH_PHASE2
 * } search_state_t;
 *
 * #define P1_AMP_X         50
 * #define P1_AMP_Y         20
 * #define P1_X_PERIOD      6.0f
 * #define P1_SWEEP_COUNT   1
 * #define P2_AMP_X         80
 * #define P2_AMP_Y         35
 * #define P2_X_PERIOD      8.0f
 * #define SEARCH_RPM_X    10.0f
 * #define SEARCH_RPM_Y    10.0f
 * #define SEARCH_DEADZONE     2
 * #ifndef M_PI
 * #define M_PI 3.14159265f
 * #endif
 *
 * static search_state_t s_searchState = SEARCH_IDLE;
 * static uint32_t s_searchStartUs = 0;
 *
 * static float triangle_wave(float phase) {
 *     if (phase < 0.25f)      return 4.0f * phase;
 *     else if (phase < 0.75f) return 2.0f - 4.0f * phase;
 *     else                    return 4.0f * phase - 4.0f;
 * }
 *
 * void Search_Init(void) {
 *     s_searchState = SEARCH_IDLE;
 *     s_searchStartUs = 0;
 * }
 *
 * void Search_Stop(void *motorX, void *motorY) {
 *     s_searchState = SEARCH_IDLE;
 *     stepper_core_set_speed_nonblock(motorX, 0.0f);
 *     stepper_core_set_speed_nonblock(motorY, 0.0f);
 * }
 *
 * uint8_t Search_IsRunning(void) {
 *     return (s_searchState != SEARCH_IDLE) ? 1 : 0;
 * }
 *
 * void Search_Task(void *motorX, void *motorY) {
 *     // ... 完整旧代码见 git 历史 ...
 * }
 */
/* ===== [废弃] 旧 Lissajous 搜索 END ===== */

/* ===== 新搜索：定速旋转扫描 ===== */
/*
 * 策略：Y轴不动（赛前标定好俯仰角），X轴定速旋转覆盖360°方位角。
 * 视觉模块持续检测 → 检测到靶子 → 主循环调 Search_Stop → 切跟踪。
 * 搜索函数不读 Serial_Const，职责仅限"转"。
 *
 * 扩展多行弓字形：在超时切行处实现 Y 轴定时移动（代码待实现）。
 * 当前 SEARCH_LINES=2，但 Y 轴移动代码尚为注释骨架，两条线在同一高度折返。
 */

/* ======== 搜索可调参数 ======== */

/*
 * SEARCH_LINES — 扫描行数
 *   搜索时 X 轴来回扫几条水平线（蛇形弓字折返）。
 *   当前 Y 轴移动代码未实现（电机俯仰角赛前已标定），所有行在同一高度折返，
 *   所以此值 >1 仅控制"扫几条线后重置行号"，实际效果和 =1 等价。
 *   调法：Y 轴移动实现后，每加一行多覆盖一条水平带。行数越多覆盖越密，
 *   但完整扫完一轮的耗时也越长（= SEARCH_LINES × SEARCH_TIMEOUT_S）。
 *   典型值：1~4。
 */
#define SEARCH_LINES 2

/*
 * SEARCH_RPM — 搜索时 X 轴旋转速度（单位：转/分钟）
 *   越高扫得越快，但 OpenMV 帧率有限（约 50Hz），如果转速过高，相邻帧之间
 *   光斑移动距离过大（视场角固定时），摄像头可能漏检。
 *   一圈耗时 = 60 / SEARCH_RPM 秒。当前 20 RPM → 一圈约 3.0 秒。
 *   调法：在"不漏检"和"扫得快"之间折中。可逐步提高，直到现场出现漏检再回退。
 *   典型值：20~80。
 */
#define SEARCH_RPM 20.0f

/*
 * SEARCH_TIMEOUT_S — 每条扫描线持续时长（单位：秒）
 *   X 轴朝一个方向旋转这么长时间后，反转方向进入下一条线（蛇形折返）。
 *   当前 3.4s × 20 RPM = 每线约 1.13 圈，确保至少覆盖一整圈。
 *   调法：至少大于 60/SEARCH_RPM（一整圈时间），否则可能转向太快覆盖不全。
 *   如果现场发现某个方向搜不到、反向才搜到 → 把这个值加大，确保每线多转几圈。
 *   典型值：1.5~5.0。
 */
#define SEARCH_TIMEOUT_S 3.4f

typedef enum { SEARCH_IDLE = 0, SEARCH_SCANNING } search_state_t;

static search_state_t s_searchState = SEARCH_IDLE;
static int8_t s_row = 0;
static int8_t s_dir = 1; /* X方向: 1=正转, -1=反转 */
static uint32_t s_deadline_us = 0;

/* -------- API（函数签名与旧代码完全一致） -------- */

void Search_Init(void) {
  s_searchState = SEARCH_IDLE;
  s_row = 0;
  s_dir = 1;
}

void Search_Stop(void *motorX, void *motorY) {
  s_searchState = SEARCH_IDLE;
  stepper_core_set_speed_nonblock(motorX, 0.0f);
  stepper_core_set_speed_nonblock(motorY, 0.0f);
}

uint8_t Search_IsRunning(void) {
  return (s_searchState != SEARCH_IDLE) ? 1 : 0;
}

/*
 * 调用前提：主循环保证只在 Serial_Const != 1 时调用本函数。
 * 本函数不读 Serial_Const，只负责 X 轴旋转超时换向。
 */
void Search_Task(void *motorX, void *motorY) {
  /* IDLE → 启动搜索 */
  if (s_searchState == SEARCH_IDLE) {
    s_searchState = SEARCH_SCANNING;
    s_row = 0;
    s_dir = 1;

    /* Y轴不动（Y轴移动代码待实现，赛前已标定好俯仰角） */
    stepper_core_set_speed_nonblock(motorY, 0.0f);

    /* X轴开始旋转 */
    s_deadline_us = get_tick_us() + (uint32_t)(SEARCH_TIMEOUT_S * 1000000.0f);
    stepper_core_set_speed_nonblock(motorX, (float)s_dir * SEARCH_RPM);
    return;
  }

  /* SCANNING：检查当前行是否超时（有符号差值比较，防无符号下溢误判） */
  uint32_t now = get_tick_us();
  if ((int32_t)(now - s_deadline_us) < 0) {
    return; /* 未超时，继续旋转，无需任何操作 */
  }

  /* 超时 → 切下一行 */
  s_row++;
  if (s_row >= SEARCH_LINES) {
    s_row = 0; /* 所有行扫完，从头循环 */
  }
  s_dir = -s_dir; /* 蛇形折返 */

  /*
   * 多行扩展点（Y 轴移动代码待实现）：
   * 当前 SEARCH_LINES=2 会执行到此处的状态循环（切行/换向），
   * 但 Y 轴移动逻辑仍为注释骨架，两条线在同一高度折返。
   * 如需开启 Y 轴移动，取消下方注释：
   *   float y_target = SEARCH_BASE_Y + (s_row - (SEARCH_LINES-1)/2.0f) *
   * SEARCH_ROW_SPACING; stepper_core_set_speed_nonblock(motorY, y_direction *
   * Y_MOVE_RPM);
   *   // 需要额外的定时止停逻辑
   */

  s_deadline_us = now + (uint32_t)(SEARCH_TIMEOUT_S * 1000000.0f);
  stepper_core_set_speed_nonblock(motorX, (float)s_dir * SEARCH_RPM);
}
