#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

// GUItool: begin automatically generated code
AudioSynthWaveform synth_1;
AudioSynthWaveformSine   sine1;          //xy=49.33332824707031,183.3333282470703
AudioOutputI2S           i2s1;           //xy=187.33332443237305,181.3333396911621
AudioConnection          patchCord1(synth_1, 0, i2s1, 0);
AudioConnection          patchCord2(synth_1, 0, i2s1, 1);
AudioControlSGTL5000 sgtl5000_1;

// GUItool: end automatically generated code


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  while (! Serial);
  
  sine1.amplitude(1.0);
  sine1.frequency(440.0);
  AudioMemory(10);

  sgtl5000_1.enable();
  // 0.8 IS THE max
  sgtl5000_1.volume(0.5);
  
  synth_1.frequency(440.0);
  synth_1.amplitude(1.0);
  synth_1.begin(WAVEFORM_SQUARE);

  Serial.println("Done setting up!");
  
}

void loop() {
  Serial.println(sine1.isActive());
  
}

