#pragma once

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>
#include <serial_commands.h>

// Use these with the Teensy Audio Shield
//def don't change these :3
#define SDCARD_CS_PIN    10
#define SDCARD_MOSI_PIN  7   // Teensy 4 ignores this
#define SDCARD_SCK_PIN   14  // Teensy 4 ignores this

// #define BOOTED_SD "BOOTED.wav"   // Must be uppercase 8.3
// #define WELCOME_SD "WELCOME.wav"
// #define SELSONG_SD "SELSONG.wav"
// #define LEARNMOD_SD "LEARNMOD.wav"
// #define HOW_RESET_SD "HOWRS.wav"
// #define HOW_MODE_SD "HOWMOD.wav"
// #define FREE_MODE_SD "FREEMOD.wav"
// #define DEBUG_SD "DEBUG.wav"
// #define CONFSONG_SD "CONFSONG.wav"
// #define CONFIRM_SD "CONFIRM.wav"

#define BOOTED_SD                 "BOOTED.wav"
#define WELCOME_SD                "WELCOME.wav"
#define SELSONG_SD                "SELSONG.wav"
#define MODE_SELECT_BUTTONS_SD    "SELMOD.wav"
#define SELECT_SONG_SD            "SELSONG.wav"
#define FREEPLAY_MODE_CONFIRM_SD  "FREEMOD.wav"
#define LEARNING_MODE_CONFIRM_SD  "LEARNMOD.wav"
#define CALIB_VELO_NO_PRESS_SD    "BEGCAL.wav"
#define CALIB_SOFT_PRESS_SD       "CALSOFT.wav"
#define CALIB_HARD_PRESS_SD       "CALHARD.wav"
#define HOW_TO_CHANGE_MODE_SD     "HOWMOD.wav"
#define HOW_TO_RESET_SONG_SD      "HOWRS.wav"
#define CONFSONG_SD               "CONFSONG.wav"
#define CONFIRM_SD                "CONFIRM.wav"
#define DEBUG_SD                  "DEBUG.wav"
#define INVALID_SD                "INVALID"
#define CALIBRATING_SD            "CAL.wav"
#define CALIBRATING_SINE_WAVE_SD  "CALSINE.wav"
#define CALIBRATION_FAILED_SD     "CALFAIL.wav"
#define CALIBRATION_SUCCESS_SD    "CALSUCC.wav"
#define FIX_POSTURE_SD            "BADPOST.wav"
#define FLUSH_SD                  "FLUSH"



class VoiceCommands {
  public:
    VoiceCommands() {
      isSetup_ = false;
    }

    const char* getFileName(VocalCommandCodes command); 
    
    //setup function
    int setUpSD();

    // play an instruction from the SD card
    void playInstruction(const char *filename);

    // SDCard playwav object
    AudioPlaySdWav playWav1;
  
    ~VoiceCommands() {}
  private:
    bool isSetup_;

};
