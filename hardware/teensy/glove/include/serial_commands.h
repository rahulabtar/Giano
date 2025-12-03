#pragma once

enum Hand
{
  Left = 0xFE,
  Right
};

enum SensorValue
{
  Pressed=0,
  Released,
};

enum VoiceCommands {
 WELCOME_MUSIC = 16,
  WELCOME_TEXT = 17,
  MODE_SELECT_BUTTONS = 18,
  FREEPLAY_MODE_CONFIRM = 19,
  LEARNING_MODE_SELECTED = 20,
  CALIB_VELO_NO_PRESS = 21,
  CALIB_SOFT_PRESS = 22,
  CALIB_HARD_PRESS = 23,
  HOW_TO_CHANGE_MODE = 24,
  CONFIRM_SONG = 25,
  CONFIRM_SELECTION = 26,
  DEBUG = 254,
  INVALID = 255
};