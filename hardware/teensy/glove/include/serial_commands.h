#pragma once
#include <Arduino.h>

enum class Hand : uint8_t 
{
  Audio = OxFD,
  Left = 0xFE,
  Right
};

enum class SensorValue : uint8_t
{
  Pressed=0,
  Released,
};

enum class VoiceCommands : uint8_t{
  WELCOME_MUSIC = 16, // bootup sound
  WELCOME_TEXT = 17, // "welcome to giano"
  MODE_SELECT_BUTTONS = 18, // "press right button one time for freeplay mode, press twice for learning mode"
  SELECT_SONG= 19, // "press the left button to select song"
  FREEPLAY_MODE_CONFIRM = 20, // " freeplay mode"
  LEARNING_MODE_CONFIRM = 21, // "learning mode"
  CALIB_VELO_NO_PRESS = 22, // "beginning start: calibrate the velostat with no press"
  CALIB_SOFT_PRESS = 23, // "calibrate the velostat with soft press"
  CALIB_HARD_PRESS = 24, // "calibrate the velostat with hard press"
  HOW_TO_CHANGE_MODE = 25, // "press the right button to change mode"
  HOW_TO_RESET_SONG = 26, // "press the left button to restart the song"
  FLUSH = 27, // internal command: flush the input serial buffer python side
  DEBUG = 28, // self explanatory 
  INVALID = 29,// another debug message?
  CALIBRATING = 30, // "calibrating"
  CALIBRATE_SINE_WAVE = 31, // "sine wave effect 7 seconds"
  CALIBRATION_FAILED = 32, // "calibration failed". to recall calibration
  CALIBRATION_SUCCESS = 33, // self explanatory
  FIX_POSTURE = 34 // "INCORRECT POSTURE" "use for interrup"
};
