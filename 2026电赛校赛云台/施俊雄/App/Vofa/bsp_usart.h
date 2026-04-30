#ifndef __BSP_USART_H
#define __BSP_USART_H

#include "stm32f1xx_hal.h"   /* HAL版：替换 stm32f10x.h */
#include "usart.h"           /* 引用 huart1 句柄（CubeMX生成） */
#include <stdio.h>

/* VOFA调参串口固定为 USART1（DEBUG_USARTx宏保留供 main.c 使用） */
#define DEBUG_USARTx   USART1

/* 发送函数声明（内部实现使用 HAL_UART_Transmit） */
void Usart_SendByte(USART_TypeDef *pUSARTx, uint8_t ch);
void Usart_SendArray(USART_TypeDef *pUSARTx, uint8_t *array, uint16_t num);
void Usart_SendString(USART_TypeDef *pUSARTx, char *str);
void Usart_SendHalfWord(USART_TypeDef *pUSARTx, uint16_t ch);

/* USART1 已由 CubeMX 的 MX_USART1_UART_Init() 完成初始化 */

void vofa_process_byte(uint8_t data);

#endif /* __BSP_USART_H */
