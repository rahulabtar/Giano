#pragma once

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

// Use these with the Teensy Audio Shield
//def don't change these :3
#define SDCARD_CS_PIN    10
#define SDCARD_MOSI_PIN  7   // Teensy 4 ignores this
#define SDCARD_SCK_PIN   14  // Teensy 4 ignores this

#define WELCOME_MUSIC "BOOTED.wav"   // Must be uppercase 8.3
#define WELCOME_SD "WELCOME.wav"
#define SELECT_SONG_SD "SELSONG.wav"
#define LEARNING_MODE_SELECTED_SD "LEARNMOD.wav"
#define HOW_TO_RESET_SD "HOWRS.wav"
#define HOW_TO_CHANGE_MODE_SD "HOWMOD.wav"
#define FREEPLAY_MODE_SELECTED_SD "FREEMOD.wav"
#define DEBUG_SD "DEBUG.wav"
#define CONFIRM_SONG_SD "CONFSONG.wav" // DON'T NEED CONFIRM_SONG
#define CONFIRM_SD "CONFIRM.wav" // DON'T NEED CONFIRM_SELECTION

class VoiceCommands {
  public:
    VoiceCommands() {
      isSetup_ = false;
    }
    
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
