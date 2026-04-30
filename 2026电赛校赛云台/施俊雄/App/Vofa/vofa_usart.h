#ifndef __VOFA_USART_H
#define __VOFA_USART_H

#include "stm32f1xx_hal.h"   /* HAL版：替换 stm32f10x.h */
#include <stdarg.h>
#include <stdio.h>

uint8_t Usart_GetRxFlag(void);
uint8_t Usart_GetRxData(void);
uint8_t Get_id_Flag(void);
float   RxPacket_Data_Handle(void);

#endif
