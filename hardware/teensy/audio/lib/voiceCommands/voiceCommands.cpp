// Simple WAV file player + test sine tone in setup()
#include "voiceCommands.h"


int VoiceCommands::setUpSD(){
  // Assume this is done elsewhere
  // AudioMemory(12);

  SPI.setMOSI(SDCARD_MOSI_PIN);
  SPI.setSCK(SDCARD_SCK_PIN);
  if (!SD.begin(SDCARD_CS_PIN)) {
    isSetup_ = false;
    return -1;
  }
  isSetup_ = true;
  //AudioPlaySdWav playWav1;
  return 0;
}

void VoiceCommands::playInstruction(const char *filename)
{
  if (isSetup_) {  
    Serial.print("Playing file: ");
    Serial.println(filename);
    if (playWav1.isPlaying()) {
      playWav1.stop(); //stop currently playing file
    }

    playWav1.play(filename); //if a wav is already playing, stop currently playing file and plays new filename
    
    uint32_t length = playWav1.lengthMillis();
    
    // Blocking version for now: wait here until file is done playing
    // TODO: figure out how to make non-blocking
    delay(length + 100); //add a small buffer to ensure full playback

    Serial.print("Played for (ms): ");
    Serial.println(length);

  }
  // without this while loop, the playInstr function is non-blocking
  // while (playWav1.isPlaying()) {
  //   // optional volume pot code here
  // }
}

const char* VoiceCommands::getFileName(VocalCommandCodes command) {
    Serial.print("Getting filename for command: ");
    switch(command) {
        case VocalCommandCodes::WELCOME_MUSIC:          return WELCOME_SONG_SD;
        case VocalCommandCodes::WELCOME_TEXT:           return SELSONG_SD;
        case VocalCommandCodes::MODE_SELECT_BUTTONS:    return MODE_SELECT_BUTTONS_SD;
        case VocalCommandCodes::SELECT_SONG:            return SELECT_SONG_SD;
        case VocalCommandCodes::FREEPLAY_MODE_CONFIRM:  return FREEPLAY_MODE_CONFIRM_SD;
        case VocalCommandCodes::LEARNING_MODE_CONFIRM:  return LEARNING_MODE_CONFIRM_SD;
        case VocalCommandCodes::CALIB_VELO_NO_PRESS:    return CALIB_VELO_NO_PRESS_SD;
        case VocalCommandCodes::CALIB_SOFT_PRESS:       return CALIB_SOFT_PRESS_SD;
        case VocalCommandCodes::CALIB_HARD_PRESS:       return CALIB_HARD_PRESS_SD;
        case VocalCommandCodes::HOW_TO_CHANGE_MODE:     return HOW_TO_CHANGE_MODE_SD;
        case VocalCommandCodes::HOW_TO_RESET_SONG:      return HOW_TO_RESET_SONG_SD;
        case VocalCommandCodes::DEBUG:                  return DEBUG_SD;
        case VocalCommandCodes::INVALID:                return INVALID_SD;
        case VocalCommandCodes::FLUSH:                  return FLUSH_SD;
        case VocalCommandCodes::CALIBRATING:            return CALIBRATING_SD;
        case VocalCommandCodes::CALIBRATING_SINE_WAVE:  return CALIBRATING_SINE_WAVE_SD;
        case VocalCommandCodes::CALIBRATION_FAILED:     return CALIBRATION_FAILED_SD;
        case VocalCommandCodes::CALIBRATION_SUCCESS:    return CALIBRATION_SUCCESS_SD;
        case VocalCommandCodes::FIX_POSTURE:            return FIX_POSTURE_SD;
        default:                                        return WELCOME_SONG_SD;
    }
}


