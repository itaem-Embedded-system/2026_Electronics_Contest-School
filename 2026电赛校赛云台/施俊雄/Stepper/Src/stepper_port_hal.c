
/**
 * @file stepper_port_hal.c
 * @brief 步进电机硬件移植层 — STM32F1xx HAL 版
 *
 *   - GPIO 操作：寄存器(BSRR/BRR) → HAL_GPIO_WritePin()
 *   - 时间戳：SysTick @1μs  → TIM2 自由计数器 @1μs (Prescaler=71)
 *   - 微秒延时：NOP软件循环 → TIM2 计数器轮询（精确）
 *   - 毫秒延时：NOP循环     → HAL_Delay()
 */

#include "interfaces.h"
#include "stepper_core.h"
#include "stm32f1xx_hal.h"
#include "tim.h"      /* 引用 htim2（TIM2句柄，由 CubeMX 生成） */
#include <stddef.h>


/* =========================================================================
 * ① pin_map — 16项数组，全显式初始化
 *    HAL宏 GPIO_PIN_x 与标准库 GPIO_Pin_x 数值相同（均为位掩码）
 * ========================================================================= */
static const uint16_t pin_map[16] = {
    GPIO_PIN_0,  GPIO_PIN_1,  GPIO_PIN_2,  GPIO_PIN_3,
    GPIO_PIN_4,  GPIO_PIN_5,  GPIO_PIN_6,  GPIO_PIN_7,
    GPIO_PIN_8,  GPIO_PIN_9,  GPIO_PIN_10, GPIO_PIN_11,
    GPIO_PIN_12, GPIO_PIN_13, GPIO_PIN_14, GPIO_PIN_15
};


/* =========================================================================
 * ② GPIO 操作函数 — 逻辑不变，API 替换为 HAL
 * ========================================================================= */
static void hal_gpio_write(uint8_t pin, gpio_level_t level)
{
    HAL_GPIO_WritePin(GPIOA, pin_map[pin],
        level == GPIO_LEVEL_HIGH ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static gpio_level_t hal_gpio_read(uint8_t pin)
{
    return (HAL_GPIO_ReadPin(GPIOA, pin_map[pin]) == GPIO_PIN_SET)
           ? GPIO_LEVEL_HIGH : GPIO_LEVEL_LOW;
}

static void hal_gpio_toggle(uint8_t pin)
{
    HAL_GPIO_TogglePin(GPIOA, pin_map[pin]);
}


/* =========================================================================
 * ③ 定时函数 — 微秒精确延时基于 TIM2 计数器，毫秒用 HAL_Delay
 *    TIM2 配置：Prescaler=71（72MHz/72=1MHz），即 1 tick = 1 μs
 *    Period=65535（16位自由运行），无溢出中断，只读计数值
 * ========================================================================= */
static void hal_delay_us(uint32_t us)
{
    uint32_t start = get_tick_us();
    while ((get_tick_us() - start) < us);
}

static void hal_delay_ms(uint32_t ms)
{
    HAL_Delay(ms);
}

static uint32_t hal_get_tick(void)
{
    /* timer_if_t.get_tick 在核心层未实际使用，返回0 */
    return 0;
}


/* =========================================================================
 * ④ get_tick_us() — 供 stepper_core_run_nonblock() / set_speed_nonblock()
 *    调用。32位溢出安全时间戳：每次读取检测TIM2 16位回绕并累加高字，
 *    返回 true 32-bit μs 时间戳（详见下方实现）。
 * ========================================================================= */
/* 32位溢出安全时间戳：TIM2为16位计数器@1MHz，每次读取检测溢出并累加高位。
 * 注意：两次调用间隔必须 < 65ms，否则会漏溢出。主循环足够快，满足此条件。
 * 此函数不可重入，请勿在中断中调用（仅在主循环中使用）。
 */
uint32_t get_tick_us(void)
{
    static uint16_t last_count = 0;
    static uint32_t high_word  = 0;

    uint16_t cur = (uint16_t)__HAL_TIM_GET_COUNTER(&htim2);
    if (cur < last_count) {
        high_word += 0x10000U; /* TIM2 16位溢出，高字累加 */
    }
    last_count = cur;
    return high_word | cur;
}


/* =========================================================================
 * ⑤ 接口结构体注册表
 * ========================================================================= */
static const gpio_if_t hal_gpio_if = {
    .write  = hal_gpio_write,
    .read   = hal_gpio_read,
    .toggle = hal_gpio_toggle
};

static const timer_if_t hal_timer_if = {
    .delay_us = hal_delay_us,
    .delay_ms = hal_delay_ms,
    .get_tick = hal_get_tick
};


/* =========================================================================
 * ⑥ stepper_port_pins_init() — API 替换为 HAL
 *    调用示例：
 *      stepper_port_pins_init(0, 1, 6);  // Y轴: PA0 PA1 PA6
 *      stepper_port_pins_init(2, 3, 5);  // X轴: PA2 PA3 PA5
 * ========================================================================= */
void stepper_port_pins_init(uint8_t step_pin, uint8_t dir_pin, uint8_t enable_pin)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    uint16_t pins = 0;

    /* 使能 GPIOA 时钟 */
    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* 组合三根引脚的位掩码 */
    pins = pin_map[step_pin] | pin_map[dir_pin] | pin_map[enable_pin];

    /* 推挽输出，50MHz */
    GPIO_InitStruct.Pin   = pins;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* 初始电平拉低 */
    HAL_GPIO_WritePin(GPIOA, pins, GPIO_PIN_RESET);
}


/* =========================================================================
 * ⑦ stepper_platform_init() — 注册接口，并启动 TIM2 自由计数
 *    必须在 stepper_port_pins_init() 和 stepper_core_create() 之前调用
 * ========================================================================= */
void stepper_platform_init(void)
{
    /* 启动 TIM2 自由运行计数（MX_TIM2_Init 已配置好，只需 Start） */
    HAL_TIM_Base_Start(&htim2);

    /* 注册 GPIO 和 Timer 接口 */
    stepper_core_register_interfaces(&hal_gpio_if, &hal_timer_if);
}
