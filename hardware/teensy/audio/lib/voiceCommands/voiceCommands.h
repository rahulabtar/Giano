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

#define BOOTED_SD "BOOTED.wav"   // Must be uppercase 8.3
#define WELCOME_SD "WELCOME.wav"
#define SELSONG_SD "SELSONG.wav"
#define LEARNMOD_SD "LEARNMOD.wav"
#define HOW_RESET_SD "HOWRS.wav"
#define HOW_MODE_SD "HOWMOD.wav"
#define FREE_MODE_SD "FREEMOD.wav"
#define DEBUG_SD "DEBUG.wav"
#define CONFSONG_SD "CONFSONG.wav"
#define CONFIRM_SD "CONFIRM.wav"

class VoiceCommands {
  public:
    VoiceCommands() {
      isSetup_ = false;
    }
    
    //setup function
    void setUpSD();

    // play an instruction from the SD card
    void playInstr(const char *filename);

    // SDCard playwav object
    AudioPlaySdWav playWav1;
  
    ~VoiceCommands() {}
  private:
    bool isSetup_;

};
