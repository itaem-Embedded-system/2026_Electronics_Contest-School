#include "PID.h"
#include "Serial.h"
#include "stm32f1xx_hal.h" /* HAL版：替换 stm32f10x.h */
#include <math.h>

PID_t pid_x, pid_y;
float err_x = 0.0f;
float err_y = 0.0f;

void PID_Init(PID_t *pid, float Kp, float Ki, float Kd, float out_max,
              float out_min) {
  pid->Kp = Kp;
  pid->Ki = Ki;
  pid->Kd = Kd;
  pid->out_max = out_max;
  pid->out_min = out_min;
  pid->prev_error = 0.0f;
  pid->integral = 0.0f;
}

float Date(PID_t *pid, float error) {
  /* 1. 比例项 */
  float p_term = pid->Kp * error;

  /* 2. 积分项 (加入限幅防止积分饱和) */
  pid->integral += error;
  if (pid->integral > 2000.0f)
    pid->integral = 2000.0f;
  if (pid->integral < -2000.0f)
    pid->integral = -2000.0f;
  float i_term = pid->Ki * pid->integral;

  /* 3. 微分项 */
  float d_term = pid->Kd * (error - pid->prev_error);

  /* 4. 计算总输出 */
  float output = p_term + i_term + d_term;

  /* 5. 输出限幅 */
  if (output > pid->out_max)
    output = pid->out_max;
  if (output < -pid->out_max)
    output = -pid->out_max;

  /* 6. 死区/最小转速处理 */
  if (fabs(output) < pid->out_min && output != 0)
    output = 0;

  /* 7. 更新历史误差 */
  pid->prev_error = error;

  return output;
}
