// codes for messages we get from Serial
#pragma once
#include <Arduino.h>


enum class Hand : uint8_t 
{
  Audio = 0xFD,
  Left = 0xFE,
  Right
};

enum class PlayingMode: u_int8_t {
  LEARNING_MODE = 0,
  FREEPLAY_MODE = 1,
};

enum class VocalCommandCodes: u_int8_t {
  WELCOME_MUSIC = 16,
  WELCOME_TEXT = 17,
  MODE_SELECT_BUTTONS = 18,
  SELECT_SONG = 19,
  FREEPLAY_MODE_CONFIRM = 20,
  LEARNING_MODE_CONFIRM = 21,
  CALIB_VELO_NO_PRESS = 22,
  CALIB_SOFT_PRESS = 23,
  CALIB_HARD_PRESS = 24,
  HOW_TO_CHANGE_MODE = 25,
  HOW_TO_RESET_SONG = 26,
  FLUSH = 27,
  DEBUG = 28,
  INVALID = 29,
  CALIBRATING = 30,
  CALIBRATING_SINE_WAVE = 31,
  CALIBRATION_FAILED = 32,
  CALIBRATION_SUCCESS = 33,
  FIX_POSTURE = 34,
};
