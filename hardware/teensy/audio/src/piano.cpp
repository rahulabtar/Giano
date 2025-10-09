#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

// Audio objects
AudioSynthWaveformSine   fmCarrier;
AudioSynthWaveformSine   fmModulator;
AudioSynthWaveformSine   fmHarmonic1;
AudioSynthWaveformSine   fmHarmonic2; 
AudioEffectEnvelope      env;      
AudioEffectEnvelope      env1;
AudioEffectEnvelope      env2;
AudioMixer4              mixer;
AudioOutputI2S           i2sOutput;

// Audio connections
AudioConnection patch1(fmCarrier, 0, env, 0);   // Send carrier through envelope
AudioConnection patch2(env, 0, mixer, 0);       // Envelope output into mixer
AudioConnection patch3(fmModulator, 0, fmCarrier, 0); // FM connection

AudioConnection patch6(fmHarmonic1, 0, env1, 0);
AudioConnection patch7(fmHarmonic2, 0, env2, 0);
AudioConnection patch8(env1, 0, mixer, 1);
AudioConnection patch9(env2, 0, mixer, 2);

AudioConnection patch4(mixer, 0, i2sOutput, 0);
AudioConnection patch5(mixer, 0, i2sOutput, 1);



AudioControlSGTL5000 audioShield;

void setup() {
  AudioMemory(20);
  audioShield.enable();
  audioShield.volume(0.5);

  // Envelope setup
  env.attack(2);    // milliseconds
  env.decay(500);    // milliseconds
  env.sustain(1);  // fully decay
  env.release(100);    // not used if single note

  env1.attack(2);    // milliseconds
  env1.decay(200);    // milliseconds
  env1.sustain(0.25);  // fully decay
  env1.release(50);    // not used if single note

  env2.attack(2);    // milliseconds
  env2.decay(200);    // milliseconds
  env2.sustain(0.25);  // fully decay
  env2.release(20);    // not used if single note
  
}

int note = 70; 
int vel = 120;

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
  env.decay(1 + 0.01 * freq);
  float amp = velocityToAmp(velocity, 5);

  // Carrier & modulator setup
  fmCarrier.frequency(freq);    // Carrier frequency
  fmCarrier.amplitude(amp);    // Full amplitude; envelope will scale it

  fmModulator.frequency(freq * 2);  // Modulator frequency
  fmModulator.amplitude(amp);   // Modulation index

  fmHarmonic1.frequency(freq * 2);
  fmHarmonic1.amplitude(amp * 0.2); 
  
  fmHarmonic2.frequency(freq * 3);
  fmHarmonic2.amplitude(amp * 0.3);
 
  Serial.printf("Playing %.2f Hz\n", freq);
  env.noteOn();      // trigger the envelope
  env1.noteOn();
  env2.noteOn();
}

void noteOff(byte channel, byte note, byte velocity){
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  env.noteOff(); 
  env1.noteOff();
  env2.noteOff(); 
}


void loop() {
  vel = 102;
  noteOn(1, note, vel);
  delay(1000);
  noteOff(1, note, vel);
  delay(1000);
}
