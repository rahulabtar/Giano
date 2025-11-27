// Simple WAV file player + test sine tone in setup()
#include "voiceCommands.h"



void VoiceCommands::setUpSD(){
  // Assume this is done elsewhere
  // AudioMemory(12);

  SPI.setMOSI(SDCARD_MOSI_PIN);
  SPI.setSCK(SDCARD_SCK_PIN);
  SD.begin(SDCARD_CS_PIN);
  isSetup_ = true;
}

void VoiceCommands::playInstr(const char *filename)
{
  Serial.print("Playing file: ");
  Serial.println(filename);
  playWav1.play(filename); //if a wav is already playing, stop currently playing file and plays new filename
  delay(25); //delay should be here to allow time to read the wav header 
  // without this while loop, the playInstr function is non-blocking
  // while (playWav1.isPlaying()) {
  //   // optional volume pot code here
  // }
}
