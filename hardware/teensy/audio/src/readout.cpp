/*

// Simple WAV file player + test sine tone in setup()

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

AudioPlaySdWav           playWav1;
AudioSynthWaveform       testSine;       // <-- added for test tone
AudioOutputI2S           audioOutput;

// Patch cords
AudioConnection          patchCord1(playWav1, 0, audioOutput, 0);
AudioConnection          patchCord2(playWav1, 1, audioOutput, 1);

AudioControlSGTL5000     sgtl5000_1;

// Use these with the Teensy Audio Shield
//def don't change these :3
#define SDCARD_CS_PIN    10
#define SDCARD_MOSI_PIN  7   // Teensy 4 ignores this
#define SDCARD_SCK_PIN   14  // Teensy 4 ignores this

void setup() {
  Serial.begin(9600);
  AudioMemory(12); //probably should leave this as is

  sgtl5000_1.enable();
  sgtl5000_1.volume(0.5);

  SPI.setMOSI(SDCARD_MOSI_PIN);
  SPI.setSCK(SDCARD_SCK_PIN);

  if (!(SD.begin(SDCARD_CS_PIN))) {
    while (1) {
      Serial.println("Unable to access the SD card");
      delay(500);
    }
  }
}

void setUpSD(){
  AudioMemory(12);
}

void playInstr(const char *filename)
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

void loop() {
  playInstr("BOOTED.wav");   // Must be uppercase 8.3
  //check play Instr note
  delay(10000);
}

// playInstr("BOOTED.wav");   // Must be uppercase 8.3
//   playInstr("WELCOME.wav");
//   playInstr("SELSONG.wav");
//   playInstr("LEARNMOD.wav");
//   playInstr("HOWRS.wav");
//   playInstr("HOWMOD.wav");
//   playInstr("FREEMOD.wav");
//   playInstr("DEBUG.wav");
//   playInstr("CONFSONG.wav");
//   playInstr("CONFIRM.wav");

*/