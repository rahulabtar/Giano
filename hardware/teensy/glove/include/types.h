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
  BOOTED = 16,
  WELCOME,
  SELSONG,
  LEARNMOD,
  HOWRS,
  HOWMOD,
  FREEMOD,
  DEBUG,
  CONFSONG,
  CONFIRM,
};