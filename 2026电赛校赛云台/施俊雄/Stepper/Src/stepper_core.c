
/**
 * @file stepper_core.c
 * @brief 步进电机核心层 — HAL 版
 */

#include "interfaces.h"
#include "stepper_core.h"    /* 含 get_tick_us() 声明 */
#include "stm32f1xx_hal.h"   /* HAL 版头文件 */
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef struct {

    const gpio_if_t* gpio;
    const timer_if_t* timer;


    uint8_t step_pin;
    uint8_t dir_pin;
    uint8_t enable_pin;


    struct {
        uint32_t steps_per_rev;
        float max_speed_rpm;
        float min_speed_rpm;
        uint32_t pulse_width_us;
    } config;


    struct {
        int32_t position;
        float speed_rpm;
        stepper_dir_t direction;
        bool enabled;
        bool moving;
    } state;

    uint8_t initialized;

    struct {
        uint32_t pulse_interval_us;
        uint32_t last_pulse_tick;
    } nonblock;
} motor_ctx_t;



static const gpio_if_t* g_gpio = NULL;
static const timer_if_t* g_timer = NULL;


void stepper_core_register_interfaces(const gpio_if_t* gpio_if, const timer_if_t* timer_if)
{
    if (gpio_if) g_gpio = gpio_if;
    if (timer_if) g_timer = timer_if;
}



static uint32_t rpm_to_interval(float rpm, uint32_t steps_per_rev)
{
    if (rpm <= 0 || steps_per_rev == 0) return 0;
    float pulses_per_sec = (rpm / 60.0f) * steps_per_rev;
    return (uint32_t)(1000000.0f / pulses_per_sec);
}

static error_t motor_generate_pulse(motor_ctx_t* motor)
{
    if (!motor->state.enabled) return ERR_BUSY;

    motor->gpio->write(motor->step_pin, GPIO_LEVEL_HIGH);
    motor->timer->delay_us(motor->config.pulse_width_us);
    motor->gpio->write(motor->step_pin, GPIO_LEVEL_LOW);

    return ERR_OK;
}




void* stepper_core_create(uint8_t step_pin, uint8_t dir_pin, uint8_t enable_pin,
                          uint32_t steps_per_rev)
{
    motor_ctx_t* motor;

    if (!g_gpio || !g_timer) return NULL;

    motor = (motor_ctx_t*)malloc(sizeof(motor_ctx_t));
    if (!motor) return NULL;


    memset(motor, 0, sizeof(motor_ctx_t));


    motor->gpio = g_gpio;
    motor->timer = g_timer;


    motor->step_pin = step_pin;
    motor->dir_pin = dir_pin;
    motor->enable_pin = enable_pin;


    motor->config.steps_per_rev = steps_per_rev;
    motor->config.max_speed_rpm = 3000.0f;
    motor->config.min_speed_rpm = 0.3f; /* 安全钳位，死区边界(0.4px)处 rpm=0.171 < 0.3，死区内不会误动 */
    motor->config.pulse_width_us = 10;


    motor->state.speed_rpm = 0.0f;
    motor->initialized = 1;


    motor->gpio->write(step_pin, GPIO_LEVEL_LOW);
    motor->gpio->write(dir_pin, GPIO_LEVEL_LOW);
    motor->gpio->write(enable_pin, GPIO_LEVEL_LOW);

    motor->nonblock.pulse_interval_us = 0;
    motor->nonblock.last_pulse_tick = 0;

    return (void*)motor;
}


void stepper_core_destroy(void* handle)
{
    if (handle) {
        free(handle);
    }
}



error_t stepper_core_enable(void* handle, bool en)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor || !motor->initialized) return ERR_PARAM;

    motor->gpio->write(motor->enable_pin, en ? GPIO_LEVEL_HIGH : GPIO_LEVEL_LOW);
    motor->state.enabled = en;

    if (!en) {
        motor->nonblock.pulse_interval_us = 0;
        motor->nonblock.last_pulse_tick = 0;
        motor->state.moving = false;
    }

    return ERR_OK;
}

error_t stepper_core_move(void* handle, int32_t steps)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor || !motor->initialized) return ERR_PARAM;
    if (!motor->state.enabled) return ERR_BUSY;


    if (steps > 0) {
        motor->gpio->write(motor->dir_pin, GPIO_LEVEL_HIGH);
        motor->state.direction = STEPPER_DIR_CW;
    } else {
        motor->gpio->write(motor->dir_pin, GPIO_LEVEL_LOW);
        motor->state.direction = STEPPER_DIR_CCW;
        steps = -steps;
    }


    uint32_t interval_us = rpm_to_interval(motor->state.speed_rpm, motor->config.steps_per_rev);
    if (interval_us == 0) return ERR_PARAM;


    motor->state.moving = true;
    for (uint32_t i = 0; i < (uint32_t)steps; i++) {
        error_t err = motor_generate_pulse(motor);
        if (err != ERR_OK) {
            motor->state.moving = false;
            return err;
        }

        if (motor->state.direction == STEPPER_DIR_CW) motor->state.position++;
        else motor->state.position--;

        if (interval_us > motor->config.pulse_width_us) {
            motor->timer->delay_us(interval_us - motor->config.pulse_width_us);
        }
    }
    motor->state.moving = false;

    return ERR_OK;
}

error_t stepper_core_set_speed(void* handle, float rpm)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor || !motor->initialized) return ERR_PARAM;

    if (rpm > motor->config.max_speed_rpm) rpm = motor->config.max_speed_rpm;
    if (rpm < motor->config.min_speed_rpm && rpm > 0) rpm = motor->config.min_speed_rpm;

    motor->state.speed_rpm = rpm;
    return ERR_OK;
}

error_t stepper_core_stop(void* handle)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor) return ERR_PARAM;
    motor->state.moving = false;
    return ERR_OK;
}

int32_t stepper_core_get_position(void* handle)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    return motor ? motor->state.position : 0;
}

void stepper_core_reset_position(stepper_handle_t handle)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (motor) motor->state.position = 0;
}

error_t stepper_core_get_state(void* handle, stepper_state_t* state)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor || !state) return ERR_PARAM;

    state->position = motor->state.position;
    state->speed_rpm = motor->state.speed_rpm;
    state->direction = motor->state.direction;
    state->enabled = motor->state.enabled;
    state->moving = motor->state.moving;
    return ERR_OK;
}

/* -----------------------------------------------------------------------
 * get_tick_us() 由 stepper_port_hal.c 基于 TIM2 计数器提供。
 * ----------------------------------------------------------------------- */

error_t stepper_core_set_speed_nonblock(void* handle, float rpm)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor || !motor->initialized) return ERR_PARAM;
    if (!motor->state.enabled) return ERR_BUSY;

    if (rpm == 0) {
        motor->nonblock.pulse_interval_us = 0;
        motor->state.speed_rpm = 0;
        motor->state.moving = false;
        return ERR_OK;
    }

    if (rpm > motor->config.max_speed_rpm) rpm = motor->config.max_speed_rpm;
    if (rpm < -motor->config.max_speed_rpm) rpm = -motor->config.max_speed_rpm;
    if (fabs(rpm) < motor->config.min_speed_rpm && rpm != 0) {
        rpm = rpm > 0 ? motor->config.min_speed_rpm : -motor->config.min_speed_rpm;
    }

    uint32_t interval_us = rpm_to_interval(fabs(rpm), motor->config.steps_per_rev);
    if (interval_us == 0) return ERR_PARAM;

    stepper_dir_t new_dir = (rpm > 0.0f) ? STEPPER_DIR_CW : STEPPER_DIR_CCW;
    uint32_t now_tick = get_tick_us();

    bool interval_expired = ((now_tick - motor->nonblock.last_pulse_tick)
                             >= motor->nonblock.pulse_interval_us);
    bool dir_reversed = (motor->state.speed_rpm != 0.0f)
                        && (motor->state.direction != new_dir);

    if (motor->nonblock.pulse_interval_us == 0
        || (dir_reversed && interval_expired)) {
        motor->nonblock.last_pulse_tick = now_tick;
    }

    motor->state.direction = new_dir;
    motor->gpio->write(motor->dir_pin,
                       (new_dir == STEPPER_DIR_CW) ? GPIO_LEVEL_HIGH : GPIO_LEVEL_LOW);

    motor->nonblock.pulse_interval_us = interval_us;
    motor->state.speed_rpm = fabs(rpm);
    motor->state.moving = true;

    return ERR_OK;
}


error_t stepper_core_run_nonblock(void* handle)
{
    motor_ctx_t* motor = (motor_ctx_t*)handle;
    if (!motor || !motor->initialized) return ERR_PARAM;
    if (!motor->state.enabled) return ERR_BUSY;

    if (motor->nonblock.pulse_interval_us == 0) {
        motor->state.moving = false;
        return ERR_OK;
    }

    uint32_t now_tick = get_tick_us();
    if ((now_tick - motor->nonblock.last_pulse_tick) >= motor->nonblock.pulse_interval_us) {
        motor_generate_pulse(motor);

        if (motor->state.direction == STEPPER_DIR_CW) {
            motor->state.position++;
        } else {
            motor->state.position--;
        }

        motor->nonblock.last_pulse_tick += motor->nonblock.pulse_interval_us;
        motor->state.moving = true;
    }

    return ERR_OK;
}
