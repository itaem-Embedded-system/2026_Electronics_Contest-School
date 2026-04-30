#ifndef __CHASSIS_H
#define __CHASSIS_H

#include <stdint.h>

/* 底盘发送的命令字节 */
#define CHASSIS_CMD_TURN_IN   'T'   /* 进入转弯 */
#define CHASSIS_CMD_TURN_OUT  'E'   /* 离开转弯 */

extern volatile uint8_t Chassis_RxFlag;   /* 1=收到完整命令，ISR 写 / 主循环读 */
extern volatile uint8_t Chassis_RxCmd;    /* 当前命令字节 */

uint8_t Chassis_GetRxFlag(void);          /* 消费 RxFlag（clear-on-read） */
void chassis_process_byte(uint8_t data);  /* 字节状态机，ISR 中调用 */

#endif
