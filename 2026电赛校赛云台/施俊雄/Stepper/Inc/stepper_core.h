
#ifndef __STEPPER_CORE_H
#define __STEPPER_CORE_H

#include "interfaces.h"
#include "stm32f1xx_hal.h" /* HAL 版：替换原来的 stm32f10x.h */
#include <stdbool.h>
#include <stdint.h>

typedef void *stepper_handle_t;

void stepper_core_register_interfaces(const gpio_if_t *gpio_if,
                                      const timer_if_t *timer_if);

stepper_handle_t stepper_core_create(uint8_t step_pin, uint8_t dir_pin,
                                     uint8_t enable_pin,
                                     uint32_t steps_per_rev);

void stepper_core_destroy(stepper_handle_t handle);

error_t stepper_core_enable(stepper_handle_t handle, bool en);

error_t stepper_core_move(stepper_handle_t handle, int32_t steps);

error_t stepper_core_set_speed(stepper_handle_t handle, float rpm);

error_t stepper_core_stop(stepper_handle_t handle);

int32_t stepper_core_get_position(stepper_handle_t handle);

error_t stepper_core_get_state(stepper_handle_t handle, stepper_state_t *state);

void stepper_core_reset_position(stepper_handle_t handle);

error_t stepper_core_set_speed_nonblock(void *handle, float rpm);

error_t stepper_core_run_nonblock(void *handle);

/* get_tick_us() 由 stepper_port_hal.c 提供（基于TIM2），此处只声明 */
uint32_t get_tick_us(void);

/* sys_tick_init() 在 HAL 工程中不使用，声明已移除 */

/* 移植层函数（由 stepper_port_hal.c 实现） */
void stepper_platform_init(void);
void stepper_port_pins_init(uint8_t step_pin, uint8_t dir_pin,
                            uint8_t enable_pin);

#endif /* __STEPPER_CORE_H */
