// codes for messages we get from Serial
#pragma once
#include <Arduino.h>
enum class VocalCommandCodes: u_int8_t {
  WELCOME_MUSIC = 0x7F,
  WELCOME_TEXT,
  MODE_SELECT_BUTTONS,
  FREEPLAY_MODE_CONFIRM,
  LEARNING_MODE_SELECTED,
  CALIB_VELO_NO_PRESS,
  CALIB_SOFT_PRESS,
  CALIB_HARD_PRESS,
  HOW_TO_CHANGE_MODE,
  CONFIRM_SONG,
  CONFIRM_SELECTION,
  DEBUG = 0xFE,
  INVALID = 0xFF,
};
