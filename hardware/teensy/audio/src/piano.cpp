#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <math.h>

// Audio objects
AudioSynthWaveformModulated fm_piano;
AudioSynthWaveform mod1; 
AudioSynthWaveform mod2; 
AudioMixer4 modMix; 
AudioEffectEnvelope env;
AudioOutputI2S i2s1;

AudioConnection patchCord1(mod1, 0, modMix, 0);
AudioConnection patchCord2(mod2, 0, modMix, 1);
AudioConnection patchCord3(modMix, 0, fm_piano, 0);
AudioConnection patchCord4(fm_piano, env);
AudioConnection patchCord5(env, 0, i2s1, 0);
AudioConnection patchCord6(env, 0, i2s1, 1);

AudioControlSGTL5000 sgtl5000_1;
int note = 60; 
int vel = 10;

float midiNoteToFreq(uint8_t note) {
  return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

float velocityToAmp(byte velocity, unsigned int scaling_factor){
  if (scaling_factor > 10) scaling_factor = 10;
  return logf(1.0f + scaling_factor * velocity / 127.0f) /
         logf(1.0f + scaling_factor);
}

void noteOn(byte channel, byte note, byte velocity) {
  Serial.printf("Note ON: Ch %d, Note %d, Vel %d\n", channel, note, velocity);
  float freq = midiNoteToFreq(note);
  float amp = velocityToAmp(velocity, 5);

  fm_piano.amplitude(amp);
  //fm_piano.frequencyModulation(freq * 1.5);
  fm_piano.frequency(freq);
  mod1.frequency(freq * 1.0);
  mod1.amplitude(0.2 + 0.5 * (velocity / 127.0));
  // mod2.frequency(freq * 1);
  // mod2.amplitude(amp * 0.5); 
  env.noteOn();

  Serial.printf("Playing %.2f Hz\n", freq);
}

void noteOff(byte channel, byte note, byte velocity){
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  env.noteOff();
}

void setup() {
  Serial.begin(115200);
  while(!Serial);

  AudioMemory(20);
  sgtl5000_1.enable();
  sgtl5000_1.volume(0.5);

  env.attack(10);
  env.decay(100);
  env.sustain(0.2);
  env.release(200);

  fm_piano.begin(WAVEFORM_SINE);
  mod1.begin(WAVEFORM_SINE);
  mod2.begin(WAVEFORM_SINE);
  modMix.gain(0, 1.0);
  modMix.gain(1, 1.0); 
}

void loop() {
  if (note > 71) {note = 60; 
  vel = 10;}
  vel += 10; 
  note++; 

  noteOn(1, note, vel);
  delay(1000);
  noteOff(1, note, vel);
  delay(1000);
}
