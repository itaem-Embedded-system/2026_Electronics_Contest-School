#ifndef __SERIAL_H
#define __SERIAL_H

#include <stdint.h>
#include <stdio.h>

extern uint8_t Serial_TxPacket[];
extern uint8_t Serial_RxPacket[];
extern volatile float Serial_X;
extern volatile float Serial_Y;
extern volatile int32_t Serial_Const;
extern volatile uint8_t Serial_RxFlag;

void Serial_Init(void);
void Serial_SendByte(uint8_t Byte);
void Serial_SendArray(uint8_t *Array, uint16_t Length);
void Serial_SendString(char *String);
void Serial_SendNumber(uint32_t Number, uint8_t Length);
void Serial_Printf(char *format, ...);

void Serial_SendPacket(void);
uint8_t Serial_GetRxFlag(void);
void serial_process_byte(uint8_t RxData);

/* 搜索任务 */
void Search_Init(void);
void Search_Task(void *motorX, void *motorY);
void Search_Stop(void *motorX, void *motorY);
uint8_t Search_IsRunning(void);

#endif
