#ifndef __PID_H_
#define __PID_H_

#include "stm32f1xx_hal.h"
#include <math.h>

typedef struct {
  float Kp;
  float Ki;
  float Kd;
  float out_max;
  float out_min;
  float prev_error;
  float integral;
} PID_t;

void PID_Init(PID_t *pid, float Kp, float Ki, float Kd, float out_max,
              float out_min);
float Date(PID_t *pid, float error);

#endif
