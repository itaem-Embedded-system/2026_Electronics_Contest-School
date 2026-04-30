#include "Chassis.h"

volatile uint8_t Chassis_RxFlag;
volatile uint8_t Chassis_RxCmd;

static uint8_t s_rxState = 0;

uint8_t Chassis_GetRxFlag(void)
{
    if (Chassis_RxFlag == 1)
    {
        Chassis_RxFlag = 0;
        return 1;
    }
    return 0;
}

void chassis_process_byte(uint8_t data)
{
    switch (s_rxState)
    {
        case 0:  /* IDLE：仅识别 'T'/'E'，其余当作线路噪声丢弃 */
            if (data == 'T' || data == 'E')
            {
                Chassis_RxCmd = data;
                Chassis_RxFlag = 1;
            }
            /* 单字节帧，无帧尾，保持在 IDLE */
            break;

        default:
            s_rxState = 0;
            break;
    }
}
