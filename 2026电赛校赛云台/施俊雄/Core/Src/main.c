/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "gpio.h"
#include "tim.h"
#include "usart.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "OLED.h"
#include "PID.h"
#include "Serial.h"
/* [废弃] VOFA+ 上位机调参已停用，USART1 挪作循迹通信
#include "bsp_usart.h"
#include "vofa_usart.h"
*/
#include "Chassis.h"      /* 底盘转弯通信（USART1） */
#include "stepper_core.h" /* 步进电机核心层 */
#include <stdbool.h>
#include <string.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
#define Y_PIXEL_OFFSET 2
#define X_PIXEL_OFFSET 1
/* 步进电机句柄 */
static stepper_handle_t motor1 = NULL;
static stepper_handle_t motor2 = NULL;

extern float err_x;
extern float err_y;
extern PID_t pid_x, pid_y;
/* [废弃] extern uint8_t Usart_RxData; — VOFA 调参用，USART1 已挪作循迹通信 */

float PID_K[6] = {0.5f, 0.00f, 0.02f, 0.6f, 0.0f, 0.02f};
/* [废弃] uint8_t PID_index = 0; — VOFA 调参用，USART1 已挪作循迹通信 */

/* D 项历史误差（一阶差分 Kd*(e(k)-e(k-1))，独立于 PID 结构体） */
static float d_prev_err_x = 0.03f;
static float d_prev_err_y = 0.0f;
static uint8_t d_history_ready = 0; /* 0=复位, 1=正式开启 D 项 */

/* ===== 底盘转弯控制 ===== */
static uint8_t chassis_in_turn = 0;    /* 1=正在执行强制转弯 */
static uint8_t chassis_turn_state = 0; /* 当前弯编号 0-3（每圈 4 个直角弯） */
static int32_t turn_start_pos = 0;     /* 转弯起始步数 */
static int32_t turn_steps_abs = 0;     /* 本次目标步数绝对值 */

/* 每弯硬编码参数（四个弯可独立标定） */
#define TURN_STEPS_0 800 /* 弯1: +800步 = +90° CW */
#define TURN_STEPS_1 800 /* 弯2 */
#define TURN_STEPS_2 800 /* 弯3 */
#define TURN_STEPS_3 800 /* 弯4 */
#define TURN_RPM 15.0f   /* 强制转弯转速 */

static const int32_t turn_steps[4] = {TURN_STEPS_0, TURN_STEPS_1, TURN_STEPS_2,
                                      TURN_STEPS_3};
/* ===== 底盘转弯控制 END ===== */

/* [已清理] 以下旧变量已无引用，已删除：
 * volatile int16_t x_target_speed, y_target_speed;
 * uint8_t system_running;
 * float out_x, out_y, rpmX, rpmY;
 */

#define LOST_COOLDOWN_US 500000 /* 丢靶后冷却 0.5 秒再重启搜索 */
static uint8_t s_lost_cooldown_active = 0;
static uint32_t s_lost_cooldown_deadline = 0;
static uint32_t s_last_frame_us = 0;      /* 最后一帧到达时刻，通信看门狗用 */
static uint8_t s_first_frame_arrived = 0; /* 首帧到达标志，上电等待视觉就绪 */

static uint8_t uart1_rx_byte; /* USART1=循迹通信接收缓冲（原VOFA已废弃） */
static uint8_t uart3_rx_byte;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
extern uint32_t get_tick_us(void);

/* [保留] 原分级调速函数（存在×1000硬跳变，已被 error_to_rpm 替代）
float speed_adjust(float output, float err) {
  float abs_err = fabs(err);
  if (abs_err > 3.0f) {
    return output * 1000.0f;
  } else if (abs_err > 1.0f) {
    return output;
  } else {
    return output * 0.8f;
  }
}
*/

/* ===== 连续增益：误差→RPM 直接映射 ===== */
#define DEAD_ZONE 0.0f        // 死区范围(px)
#define TRANSITION_HIGH 70.0f // 比例段终点(px)，此后饱和
#define MAX_RPM 30.0f         // 最大转速
/* [废弃] 旧分段参数，纯比例映射不再使用
#define MIN_RPM 1.0f          // 最小有效转速
#define TRANSITION_LOW 20.0f  // 线性段起点(px)
*/

/* 纯比例映射：rpm = (|err|/TRANSITION_HIGH) * MAX_RPM
   死区边界(0.4px)处 rpm=(0.4/70)*30=0.171，低于 stepper_core 钳位(0.3)，
   死区内误差无法突破 min_speed_rpm 钳位，电机不转。方向永远正比于误差。
*/
float error_to_rpm(float error) {
  float abs_err = fabs(error);
  float rpm;

  if (abs_err < DEAD_ZONE) {
    rpm = 0.0f;
  } else if (abs_err < TRANSITION_HIGH) {
    rpm = (abs_err / TRANSITION_HIGH) * MAX_RPM;
  } else {
    rpm = MAX_RPM;
  }

  return (error > 0) ? rpm : -rpm;
}

/* [废弃] VOFA_Send_Wave — USART1 已挪作循迹通信，VOFA 波形发送停用
void VOFA_Send_Wave(float ch1, float ch2, float ch3, float ch4, float ch5,
                    float ch6, float ch7, float ch8) {
  uint8_t send_buf[36] = {0};
  float data_buf[8] = {ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8};
  const uint8_t tail[4] = {0x00, 0x00, 0x80, 0x7F};

  memcpy(send_buf, data_buf, 32);
  memcpy(send_buf + 32, tail, 4);

  Usart_SendArray(DEBUG_USARTx, send_buf, 36);
}
===== [废弃] VOFA_Send_Wave END ===== */
/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick.
   */

  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */

  stepper_platform_init();
  stepper_port_pins_init(0, 1, 6); /* Y轴: PA0=STEP, PA1=DIR, PA6=EN */
  stepper_port_pins_init(2, 3, 5); /* X轴: PA2=STEP, PA3=DIR, PA5=EN */

  motor1 = stepper_core_create(0, 1, 6, 3200); /* Y轴, 3200步/转（1/16细分） */
  motor2 = stepper_core_create(2, 3, 5, 3200); /* X轴, 3200步/转（1/16细分） */

  if (motor1 == NULL || motor2 == NULL) {
    Error_Handler();
  }

   //OLED_Init();

  // OLED_ShowHexNum(4, 14, 4, 2);

  stepper_core_set_speed_nonblock(motor2, 0.0f);
  stepper_core_set_speed_nonblock(motor1, 0.0f);

  PID_Init(&pid_x, PID_K[0], PID_K[1], PID_K[2], 10000.0f, 200.0f);
  PID_Init(&pid_y, PID_K[3], PID_K[4], PID_K[5], 10000.0f, 200.0f);

   //OLED_ShowHexNum(4, 14, 4, 2);
   //OLED_ShowChar(1, 8, 'Y');
   //OLED_ShowChar(2, 8, 'X');

  stepper_core_enable(motor1, true);
  stepper_core_enable(motor2, true);

  /* USART1=循迹通信接收启动（原VOFA已废弃） */
  HAL_UART_Receive_IT(&huart1, &uart1_rx_byte, 1);
  HAL_UART_Receive_IT(&huart3, &uart3_rx_byte, 1);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  Search_Init();
  /* 搜索推迟到首帧到达后启动，避免摄像头未就绪时盲转 */
  while (1) {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    /* 底盘转弯命令消费（镜像 Serial_GetRxFlag 模式） */
    if (Chassis_GetRxFlag()) {
      uint8_t cmd = Chassis_RxCmd;
      if (cmd == 'T' && !chassis_in_turn) {
        if (Search_IsRunning())
          Search_Stop(motor2, motor1);
        chassis_in_turn = 1;
        d_history_ready = 0; /* 进入转弯复位历史 */
        turn_start_pos = stepper_core_get_position(motor2);
        turn_steps_abs = (turn_steps[chassis_turn_state] >= 0)
                             ? turn_steps[chassis_turn_state]
                             : -turn_steps[chassis_turn_state];
        float rpm =
            (turn_steps[chassis_turn_state] >= 0) ? TURN_RPM : -TURN_RPM;
        stepper_core_set_speed_nonblock(motor2, rpm);
      } else if (cmd == 'E' && chassis_in_turn) {
        chassis_in_turn = 0;
        chassis_turn_state = (chassis_turn_state + 1) % 4;
        stepper_core_set_speed_nonblock(motor2, 0.0f);
        d_history_ready = 0; /* 出弯后复位历史 */
      }
    }

    /* [废弃] VOFA 调参帧处理 — USART1 已挪作循迹通信
    if (Usart_GetRxFlag() == 1) {
      PID_index = Get_id_Flag();
      if (PID_index >= 1 && PID_index <= 6) {
        PID_K[PID_index - 1] = RxPacket_Data_Handle();

        pid_x.Kp = PID_K[0];
        pid_x.Ki = PID_K[1];
        pid_x.Kd = PID_K[2];
        pid_y.Kp = PID_K[3];
        pid_y.Ki = PID_K[4];
        pid_y.Kd = PID_K[5];

        OLED_ShowChar(1, 10, '!');
        Usart_SendString(DEBUG_USARTx, "OK\r\n");
      }
    }
    ===== [废弃] VOFA 调参帧处理 END ===== */

    if (Serial_GetRxFlag() == 1) {
      /* 禁中断快照 volatile 变量，确保同一迭代内所有逻辑基于同一帧数据 */
      __disable_irq();
      const float sx = Serial_X;
      const float sy = Serial_Y;
      const int32_t sc = Serial_Const;
      __enable_irq();
      s_last_frame_us = get_tick_us(); /* 刷新通信看门狗时间戳 */

      /* 首帧到达：若未检测到靶子，立即启动搜索（跳过3秒冷却） */
      if (!s_first_frame_arrived) {
        s_first_frame_arrived = 1;
        if (sc == 0 && !Search_IsRunning() && !chassis_in_turn) {
          Search_Task(motor2, motor1);
        }
      }

      /* 无论搜索还是跟踪，都更新OLED显示 */
      // OLED_ShowSignedNum(1, 1, (int32_t)Serial_Y, 4);
       //OLED_ShowSignedNum(2, 1, (int32_t)Serial_X, 4);
      // OLED_ShowSignedNum(3, 1, Serial_Const, 4);

      if (sc == 1) {
        /* ===== 找到靶子 → PID跟踪 ===== */
        if (Search_IsRunning()) {
          Search_Stop(motor2, motor1);
          PID_Init(&pid_x, PID_K[0], PID_K[1], PID_K[2], 10000.0f, 200.0f);
          PID_Init(&pid_y, PID_K[3], PID_K[4], PID_K[5], 10000.0f, 200.0f);
          d_history_ready = 0; /* 重锁时废弃 D 项旧历史 */
        }

        err_x = sx + X_PIXEL_OFFSET;
        err_y = sy + Y_PIXEL_OFFSET;

        /* [保留] PID + speed_adjust 旧逻辑（已被 error_to_rpm 替代）
         * 如需恢复，取消以下注释并删除 error_to_rpm 调用
         *
         * float pid_out_x = Date(&pid_x, err_x);
         * float pid_out_y = Date(&pid_y, err_y);
         * pid_out_x = speed_adjust(pid_out_x, err_x);
         * pid_out_y = speed_adjust(pid_out_y, err_y);
         * float local_rpmX = (pid_out_x * 60.0f) / 800.0f;
         * float local_rpmY = (pid_out_y * 60.0f) / 800.0f;
         */

        /* X 轴：转弯中若识别到靶子，视觉接管替代盲走 TURN_RPM */
        float local_rpmX = error_to_rpm(err_x);
        if (d_history_ready >= 1) {
          float d_x = PID_K[2] * (err_x - d_prev_err_x);
          local_rpmX += d_x;
        }
        d_prev_err_x = err_x;
        stepper_core_set_speed_nonblock(motor2, local_rpmX);

        /* Y 轴：始终正常跟踪 */
        float local_rpmY = error_to_rpm(err_y);
        if (d_history_ready >= 1) {
          float d_y = PID_K[5] * (err_y - d_prev_err_y);
          local_rpmY += d_y;
        }
        d_prev_err_y = err_y;

        if (d_history_ready < 1)
          d_history_ready++;
        stepper_core_set_speed_nonblock(motor1, local_rpmY);

        /* [废弃] VOFA波形发送 — USART1 已挪作循迹通信
        static uint32_t last_t = 0;
        if (get_tick_us() - last_t >= 10000) {
          last_t = get_tick_us();
          VOFA_Send_Wave(err_x, err_y, local_rpmX, local_rpmY, 0, 0, DEAD_ZONE,
                         MAX_RPM);
        }
        ===== [废弃] VOFA波形发送 END ===== */
      }
    }

    /* 通信超时看门狗：500ms 无帧到达 → 强制丢靶
       移除 !chassis_in_turn 限制，转弯期间通信断开也应丢靶保护 */
    if (s_first_frame_arrived && !Search_IsRunning() &&
        (int32_t)(get_tick_us() - s_last_frame_us) >= 500000) {
      Serial_Const = 0;
    }

    /* 搜索超时检查/换向 + 丢靶后冷却重启 */
    if (Search_IsRunning() && !chassis_in_turn) {
      Search_Task(motor2, motor1);
      s_lost_cooldown_active = 0;
    } else if (s_first_frame_arrived && Serial_Const != 1) {
      if (chassis_in_turn) {
        /* 转弯期间脱靶/断连：Y轴停车，X轴回退盲走 TURN_RPM */
        stepper_core_set_speed_nonblock(motor1, 0.0f);
        float rpm = (turn_steps[chassis_turn_state] >= 0) ? TURN_RPM : -TURN_RPM;
        stepper_core_set_speed_nonblock(motor2, rpm);
        s_lost_cooldown_active = 0;
      } else {
        if (!s_lost_cooldown_active) {
          s_lost_cooldown_deadline = get_tick_us() + LOST_COOLDOWN_US;
          s_lost_cooldown_active = 1;
          /* 丢靶瞬间：Y轴停车，X轴减速到一半 */
          stepper_core_set_speed_nonblock(motor1, 0.0f);
          stepper_state_t x_state;
          stepper_core_get_state(motor2, &x_state);
          float signed_rpm_x = (x_state.direction == STEPPER_DIR_CW)
                                   ? x_state.speed_rpm
                                   : -x_state.speed_rpm;
          stepper_core_set_speed_nonblock(motor2, signed_rpm_x / 2.0f);
        } else if ((int32_t)(get_tick_us() - s_lost_cooldown_deadline) >= 0) {
          s_lost_cooldown_active = 0;
          Search_Task(motor2, motor1);
        }
      }
    } else {
      s_lost_cooldown_active = 0;
    }

    /* 强制转弯步数完成检测（'E' 信号丢失时的兜底停止） */
    if (chassis_in_turn) {
      int32_t cur = stepper_core_get_position(motor2);
      int32_t delta = (cur >= turn_start_pos) ? (cur - turn_start_pos)
                                              : (turn_start_pos - cur);
      if (delta >= turn_steps_abs) {
        chassis_in_turn = 0;
        chassis_turn_state = (chassis_turn_state + 1) % 4;
        stepper_core_set_speed_nonblock(motor2, 0.0f);
        d_history_ready = 0;
      }
    }

    /* 无条件驱动脉冲：无论跟踪/搜索/空闲，每次主循环都执行 */
    stepper_core_run_nonblock(motor2);
    stepper_core_run_nonblock(motor1);
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance == USART1) {
    chassis_process_byte(uart1_rx_byte);
    HAL_UART_Receive_IT(&huart1, &uart1_rx_byte, 1);
  } else if (huart->Instance == USART3) {
    serial_process_byte(uart3_rx_byte);
    HAL_UART_Receive_IT(&huart3, &uart3_rx_byte, 1);
  }
}

/* 串口错误回调：防止 ORE/FE 等硬件错误导致接收中断永久挂死 */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance == USART1) {
    HAL_UART_Receive_IT(&huart1, &uart1_rx_byte, 1);
  } else if (huart->Instance == USART3) {
    HAL_UART_Receive_IT(&huart3, &uart3_rx_byte, 1);
  }
}
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1) {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line) {
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line
     number, ex: printf("Wrong parameters value: file %s on line %d\r\n", file,
     line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
