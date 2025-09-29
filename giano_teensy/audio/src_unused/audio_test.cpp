#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>

// Audio objects
AudioSynthWaveformSine   sine1;     // Generate a sine wave
AudioOutputI2S           i2s1;      // Audio Shield output (I2S)
AudioConnection          patchCord1(sine1, 0, i2s1, 0);
AudioConnection          patchCord2(sine1, 0, i2s1, 1);
AudioControlSGTL5000     sgtl5000_1; // Control object for Audio Shield

void setup() {
  AudioMemory(12);                 // Allocate audio blocks
  sgtl5000_1.enable();             // Enable the audio shield
  sgtl5000_1.volume(0.5);          // Headphone volume (0.0–1.0)

}

void loop() {
  // nothing to do; Audio library runs in the background
  sine1.frequency(440);            // A4 = 440 Hz
  sine1.amplitude(0.5);   
  delay(500);
  sine1.amplitude(0);
  delay(500);
}
