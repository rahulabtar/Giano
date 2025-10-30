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
AudioSynthWaveformSine   fmHarmonic3;
AudioSynthWaveformSine   fmHarmonic4;
AudioSynthWaveformSine   fmHarmonic5;

AudioEffectEnvelope      env;      
AudioEffectEnvelope      env1;
AudioEffectEnvelope      env2;
AudioEffectEnvelope      env3;
AudioEffectEnvelope      env4;
AudioEffectEnvelope      env5;

AudioMixer4              mixer;
AudioOutputI2S           i2sOutput;

// Audio connections
AudioConnection patch1(fmCarrier, 0, env, 0);   // Send carrier through envelope
AudioConnection patch2(env, 0, mixer, 0);       // Envelope output into mixer
AudioConnection patch3(fmModulator, 0, fmCarrier, 0); // FM connection

AudioConnection patch6(fmHarmonic1, 0, env1, 0);
AudioConnection patch7(fmHarmonic2, 0, env2, 0);
AudioConnection patch8(fmHarmonic3, 0, env3, 0);
//AudioConnection patch9(fmHarmonic4, 0, env4, 0);
//AudioConnection patch10(fmHarmonic5, 0, env5, 0);


AudioConnection patch11(env1, 0, mixer, 1);
AudioConnection patch12(env2, 0, mixer, 2);

AudioConnection patch13(mixer, 0, i2sOutput, 0);
AudioConnection patch14(mixer, 0, i2sOutput, 1);

int note [] = {60, 62, 64, 65, 67, 69, 71, 72}; 
int vel = 70;
int hold_time=3000;
float decayTime = 500;
float sustainLevel = 0.5;

AudioControlSGTL5000 audioShield;

void setup() {
  AudioMemory(20);
  audioShield.enable();
  audioShield.volume(0.4);
  


  // Envelope setup
 // env.attack(2);     // milliseconds
  //other option for attack:
  env.attack(2 - (vel / 70)); // softer notes have slower attack
  env.decay(500);    // milliseconds
  env.sustain(sustainLevel);  
  env.release(300);    // not used if single note

  env.attack(2 - (vel / 70));  // milliseconds
  env1.decay(500);    // milliseconds
  env1.sustain(sustainLevel);  
  env1.release(200);    // not used if single note

  env.attack(2 - (vel / 70)); // milliseconds
  env2.decay(500);    // milliseconds
  env2.sustain(sustainLevel);  
  env2.release(100);    // not used if single note
  
  env.attack(2 - (vel / 70)); // milliseconds
  env3.decay(500);    // milliseconds
  env3.sustain(sustainLevel);  
  env3.release(50);    // not used if single note

  /*with below sounds like an organ i dont like
  env.attack(2); // softer notes have slower attack// milliseconds
  env4.decay(200);    // milliseconds
  env4.sustain(0.25);  // fully decay
  env4.release(50);    // not used if single note

  env.attack(2); // softer notes have slower attack// milliseconds
  env5.decay(200);    // milliseconds
  env5.sustain(0.25);  // fully decay
  env5.release(50);    // not used if single note
  */
  
}


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

  fmCarrier.frequency(freq);
  fmCarrier.amplitude(amp);
  fmModulator.frequency(freq * 2);
  fmModulator.amplitude(amp);
  fmHarmonic1.frequency(freq * 2);
  fmHarmonic1.amplitude(amp * 0.2);
  fmHarmonic2.frequency(freq * 3);
  fmHarmonic2.amplitude(amp * 0.3);
  fmHarmonic3.frequency(freq * 4);
  fmHarmonic3.amplitude(amp * 0.4);

  env.noteOn();
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
  
  for (int i = 0; i < 8; i++) {
    
    /* int delayTime = 5000; // how long to hold the note
    float sustainLevel = sustainFromDelay(delayTime);

    // dynamically set sustain based on delay
    env.sustain(sustainLevel);
    env1.sustain(sustainLevel);
    env2.sustain(sustainLevel);
*/  
/*
    if (hold_time<400) sustainLevel=0.8;
    else if (hold_time<1000) sustainLevel=0.5;
    else if (hold_time<2000) sustainLevel=0.3;
    else sustainLevel=0.15; //assuming very long hold time
*/
    /*env.decay(200);
    env1.decay(200);
    env2.decay(200);
    env3.decay(200);
    
    env.sustain(sustainLevel);
    env1.sustain(sustainLevel);
    env2.sustain(sustainLevel);
    env3.sustain(sustainLevel);
    */
    decayTime = hold_time * 0.7;  // roughly fade for 70% of the hold time
    if (decayTime > 4000) decayTime = 4000; // cap it so it doesn't last forever
    if (decayTime < 200)  decayTime = 200;
    env.decay(decayTime);
    env1.decay(decayTime);
    env2.decay(decayTime);
    env3.decay(decayTime);

   
    noteOn(1, note[i], vel);
    delay(hold_time);
    noteOff(1, note[i], vel);
    delay(500);
    

  }
}

