#include "stm32f1xx_hal.h"   /* HAL版：替换 stm32f10x.h */
#include "OLED_Font.h"

/* 引脚宏：PB8=SCL，PB9=SDA（软件模拟I2C，开漏输出）
 * PB8/PB9 已由 CubeMX / MX_GPIO_Init() 配置为开漏输出，无需重新初始化 */
#define OLED_W_SCL(x)  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, (x) ? GPIO_PIN_SET : GPIO_PIN_RESET)
#define OLED_W_SDA(x)  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, (x) ? GPIO_PIN_SET : GPIO_PIN_RESET)

/* 引脚初始化：PB8/PB9 已由 CubeMX 配置，只设置初始电平 */
void OLED_I2C_Init(void)
{
    OLED_W_SCL(1);
    OLED_W_SDA(1);
}

/**
  * @brief  I2C开始
  */
void OLED_I2C_Start(void)
{
    OLED_W_SDA(1);
    OLED_W_SCL(1);
    OLED_W_SDA(0);
    OLED_W_SCL(0);
}

/**
  * @brief  I2C停止
  */
void OLED_I2C_Stop(void)
{
    OLED_W_SDA(0);
    OLED_W_SCL(1);
    OLED_W_SDA(1);
}

/**
  * @brief  I2C发送一个字节
  */
void OLED_I2C_SendByte(uint8_t Byte)
{
    uint8_t i;
    for (i = 0; i < 8; i++)
    {
        OLED_W_SDA(!!(Byte & (0x80 >> i)));
        OLED_W_SCL(1);
        OLED_W_SCL(0);
    }
    OLED_W_SCL(1);  /* 额外时钟，不处理应答信号 */
    OLED_W_SCL(0);
}

/**
  * @brief  OLED写命令
  */
void OLED_WriteCommand(uint8_t Command)
{
    OLED_I2C_Start();
    OLED_I2C_SendByte(0x78);    /* 从机地址 */
    OLED_I2C_SendByte(0x00);    /* 写命令 */
    OLED_I2C_SendByte(Command);
    OLED_I2C_Stop();
}

/**
  * @brief  OLED写数据
  */
void OLED_WriteData(uint8_t Data)
{
    OLED_I2C_Start();
    OLED_I2C_SendByte(0x78);    /* 从机地址 */
    OLED_I2C_SendByte(0x40);    /* 写数据 */
    OLED_I2C_SendByte(Data);
    OLED_I2C_Stop();
}

/**
  * @brief  OLED设置光标位置
  * @param  Y 向下方向坐标，范围：0~7
  * @param  X 向右方向坐标，范围：0~127
  */
void OLED_SetCursor(uint8_t Y, uint8_t X)
{
    OLED_WriteCommand(0xB0 | Y);
    OLED_WriteCommand(0x10 | ((X & 0xF0) >> 4));
    OLED_WriteCommand(0x00 | (X & 0x0F));
}

/**
  * @brief  OLED清屏
  */
void OLED_Clear(void)
{
    uint8_t i, j;
    for (j = 0; j < 8; j++)
    {
        OLED_SetCursor(j, 0);
        for (i = 0; i < 128; i++)
        {
            OLED_WriteData(0x00);
        }
    }
}

/**
  * @brief  OLED显示一个字符
  * @param  Line   行位置，范围：1~4
  * @param  Column 列位置，范围：1~16
  */
void OLED_ShowChar(uint8_t Line, uint8_t Column, char Char)
{
    uint8_t i;
    OLED_SetCursor((Line - 1) * 2, (Column - 1) * 8);
    for (i = 0; i < 8; i++)
    {
        OLED_WriteData(OLED_F8x16[Char - ' '][i]);
    }
    OLED_SetCursor((Line - 1) * 2 + 1, (Column - 1) * 8);
    for (i = 0; i < 8; i++)
    {
        OLED_WriteData(OLED_F8x16[Char - ' '][i + 8]);
    }
}

/**
  * @brief  OLED显示字符串
  */
void OLED_ShowString(uint8_t Line, uint8_t Column, char *String)
{
    uint8_t i;
    for (i = 0; String[i] != '\0'; i++)
    {
        OLED_ShowChar(Line, Column + i, String[i]);
    }
}

/**
  * @brief  次方函数
  */
uint32_t OLED_Pow(uint32_t X, uint32_t Y)
{
    uint32_t Result = 1;
    while (Y--)
    {
        Result *= X;
    }
    return Result;
}

/**
  * @brief  OLED显示数字（十进制，正数）
  */
void OLED_ShowNum(uint8_t Line, uint8_t Column, uint32_t Number, uint8_t Length)
{
    uint8_t i;
    for (i = 0; i < Length; i++)
    {
        OLED_ShowChar(Line, Column + i, Number / OLED_Pow(10, Length - i - 1) % 10 + '0');
    }
}

/**
  * @brief  OLED显示数字（十进制，带符号数）
  */
void OLED_ShowSignedNum(uint8_t Line, uint8_t Column, int32_t Number, uint8_t Length)
{
    uint8_t i;
    uint32_t Number1;
    if (Number >= 0)
    {
        OLED_ShowChar(Line, Column, '+');
        Number1 = Number;
    }
    else
    {
        OLED_ShowChar(Line, Column, '-');
        Number1 = -Number;
    }
    for (i = 0; i < Length; i++)
    {
        OLED_ShowChar(Line, Column + i + 1, Number1 / OLED_Pow(10, Length - i - 1) % 10 + '0');
    }
}

/**
  * @brief  OLED显示数字（十六进制，正数）
  */
void OLED_ShowHexNum(uint8_t Line, uint8_t Column, uint32_t Number, uint8_t Length)
{
    uint8_t i, SingleNumber;
    for (i = 0; i < Length; i++)
    {
        SingleNumber = Number / OLED_Pow(16, Length - i - 1) % 16;
        if (SingleNumber < 10)
        {
            OLED_ShowChar(Line, Column + i, SingleNumber + '0');
        }
        else
        {
            OLED_ShowChar(Line, Column + i, SingleNumber - 10 + 'A');
        }
    }
}

/**
  * @brief  OLED显示数字（二进制，正数）
  */
void OLED_ShowBinNum(uint8_t Line, uint8_t Column, uint32_t Number, uint8_t Length)
{
    uint8_t i;
    for (i = 0; i < Length; i++)
    {
        OLED_ShowChar(Line, Column + i, Number / OLED_Pow(2, Length - i - 1) % 2 + '0');
    }
}

/**
  * @brief  OLED初始化
  */
void OLED_Init(void)
{
    uint32_t i, j;

    for (i = 0; i < 1000; i++)     /* 上电延时 */
    {
        for (j = 0; j < 1000; j++);
    }

    OLED_I2C_Init();            /* 设置引脚初始电平 */

    OLED_WriteCommand(0xAE);    /* 关闭显示 */
    OLED_WriteCommand(0xD5);    /* 设置显示时钟分频比/振荡器频率 */
    OLED_WriteCommand(0x80);
    OLED_WriteCommand(0xA8);    /* 设置多路复用率 */
    OLED_WriteCommand(0x3F);
    OLED_WriteCommand(0xD3);    /* 设置显示偏移 */
    OLED_WriteCommand(0x00);
    OLED_WriteCommand(0x40);    /* 设置显示开始行 */
    OLED_WriteCommand(0xA1);    /* 设置左右方向，0xA1正常 */
    OLED_WriteCommand(0xC8);    /* 设置上下方向，0xC8正常 */
    OLED_WriteCommand(0xDA);    /* 设置COM引脚硬件配置 */
    OLED_WriteCommand(0x12);
    OLED_WriteCommand(0x81);    /* 设置对比度控制 */
    OLED_WriteCommand(0xCF);
    OLED_WriteCommand(0xD9);    /* 设置预充电周期 */
    OLED_WriteCommand(0xF1);
    OLED_WriteCommand(0xDB);    /* 设置VCOMH取消选择级别 */
    OLED_WriteCommand(0x30);
    OLED_WriteCommand(0xA4);    /* 设置整个显示打开/关闭 */
    OLED_WriteCommand(0xA6);    /* 设置正常/倒转显示 */
    OLED_WriteCommand(0x8D);    /* 设置充电泵 */
    OLED_WriteCommand(0x14);
    OLED_WriteCommand(0xAF);    /* 开启显示 */

    OLED_Clear();               /* OLED清屏 */
}
