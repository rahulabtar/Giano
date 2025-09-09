#ifdef TEENSYDUINO

#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>
#include <Bounce.h>
// put function declarations here:

AudioSynthWaveform waveform1;
AudioSynthWaveformSineModulated sine1;
AudioOutputI2S i2s_1;
AudioConnection myConnection(sine1, i2s_1);
void setup() {
  
  Serial.begin(115200);
  sine1.frequency(440);
  i2s_1.begin();
  F

}

void loop() {
  // put your main code here, to run repeatedly:
  
}

#endif