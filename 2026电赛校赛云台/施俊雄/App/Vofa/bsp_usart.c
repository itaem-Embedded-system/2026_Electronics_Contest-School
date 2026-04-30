#include "bsp_usart.h"

/* USART1 NVIC 已由 CubeMX 配置，USART1 已由 MX_USART1_UART_Init() 初始化 */

/**
  * @brief  发送一个字节
  * @note   内部改用 HAL_UART_Transmit（有超时保护）
  *         pUSARTx 参数保留但忽略，本工程固定使用 USART1（huart1）
  */
void Usart_SendByte(USART_TypeDef *pUSARTx, uint8_t ch)
{
    (void)pUSARTx;  /* 本工程仅通过 DEBUG_USARTx=USART1 调用，固定用 huart1 */
    HAL_UART_Transmit(&huart1, &ch, 1, 1000);
}

/**
  * @brief  发送数组
  */
void Usart_SendArray(USART_TypeDef *pUSARTx, uint8_t *array, uint16_t num)
{
    (void)pUSARTx;
    HAL_UART_Transmit(&huart1, array, num, 1000);
    /* HAL_UART_Transmit 在发送完成后才返回，无需额外等待 */
}

/**
  * @brief  发送字符串
  */
void Usart_SendString(USART_TypeDef *pUSARTx, char *str)
{
    (void)pUSARTx;
    uint16_t len = 0;
    while (str[len] != '\0') len++;
    HAL_UART_Transmit(&huart1, (uint8_t *)str, len, 1000);
}

/**
  * @brief  发送半字（2字节，高字节在前）
  */
void Usart_SendHalfWord(USART_TypeDef *pUSARTx, uint16_t ch)
{
    (void)pUSARTx;
    uint8_t buf[2];
    buf[0] = (ch & 0xFF00) >> 8;   /* 高字节 */
    buf[1] = ch & 0xFF;            /* 低字节 */
    HAL_UART_Transmit(&huart1, buf, 2, 1000);
}
