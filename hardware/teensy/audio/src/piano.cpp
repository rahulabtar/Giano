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
AudioConnection patch9(fmHarmonic4, 0, env4, 0);
AudioConnection patch10(fmHarmonic5, 0, env5, 0);


AudioConnection patch11(env, 0, mixer, 0);
AudioConnection patch12(env1, 0, mixer, 1);
AudioConnection patch13(env2, 0, mixer, 2);
AudioConnection patch14(env3, 0, mixer, 3);
AudioConnection patch15(env4, 0, mixer, 4);
AudioConnection patch16(env5, 0, mixer, 5);
AudioConnection patch17(mixer, 0, i2sOutput, 0);
AudioConnection patch18(mixer, 0, i2sOutput, 1);

int note [] = {60, 62, 64, 65, 67, 69, 71, 72}; 
int vel = 60;
int hold_time=1000;
float decayTime = 500;
float sustainLevel = 0;
float releaseTime = 800;
AudioControlSGTL5000 audioShield;

void setup() {
  AudioMemory(20);
  audioShield.enable();
  audioShield.volume(0.4);
  


  // Envelope setup
  env.attack(3 - (vel / 70)); // softer notes have slower attack
  env.decay(500);    // milliseconds
  env.sustain(sustainLevel);  
  env.release(800);    

  env1.attack(3 - (vel / 70));  // milliseconds
  env1.decay(500);    // milliseconds
  env1.sustain(sustainLevel);  
  env1.release(800);    

  env2.attack(3 - (vel / 70)); // milliseconds
  env2.decay(500);    // milliseconds
  env2.sustain(sustainLevel);  
  env2.release(800);    
  
  env3.attack(3 - (vel / 70)); // milliseconds
  env3.decay(500);    // milliseconds
  env3.sustain(sustainLevel);  
  env3.release(800);    
  
/*
  //with below sounds like an organ i dont like
  env4.attack(2 - (vel / 70)); // softer notes have slower attack// milliseconds
  env4.decay(200);    // milliseconds
  env4.sustain(0.25);  // fully decay
  env4.release(50);    

  env5.attack(2 - (vel / 70)); // softer notes have slower attack// milliseconds
  env5.decay(200);    // milliseconds
  env5.sustain(0.25);  // fully decay
  env5.release(50);    
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

  fmCarrier.frequency(freq); //fundamental freq is set for each note
  fmCarrier.amplitude(amp); //amp set for each note; determined by velocity (which is set to a fixed # for now)
  fmModulator.frequency(freq*100); //should typically be 1x or 2x carrier freq but exagerrating here bc it sounds good
  fmModulator.amplitude(amp*0.0001);  //should be multiplied a very long
  fmHarmonic1.frequency(freq * 2); // first harmonic
  fmHarmonic1.amplitude(amp * 0.4); //all harmonic amps should ideally be n/2, set to lower bc sounds better (higher harmonic amplitudes are lower)
  fmHarmonic2.frequency(freq * 3); //second harmonic
  fmHarmonic2.amplitude(amp * 0.25);
  fmHarmonic3.frequency(freq * 4); //third harmonic
  fmHarmonic3.amplitude(amp * 0.15);

  env.noteOn();
  env1.noteOn();
  env2.noteOn();
  env3.noteOn();
  env4.noteOn(); //not used rn
  env5.noteOn(); //not used rn

}


void noteOff(byte channel, byte note, byte velocity){
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  env.noteOff(); 
  env1.noteOff();
  env2.noteOff(); 
  env3.noteOff();
  env4.noteOff(); //not used rn
  env5.noteOff(); //not used rn
}




void loop() {
  
  for (int i = 0; i < 8; i++) {
    
 
//change sustain pt of ADSR based on hold time
    if (hold_time<400) sustainLevel=0.8;
    else if (hold_time<1000) sustainLevel=0.5;
    else if (hold_time<2000) sustainLevel=0.3;
    else sustainLevel=0.05; //assuming very long hold time
    env.sustain(sustainLevel);
    env1.sustain(sustainLevel);
    env2.sustain(sustainLevel);
    env3.sustain(sustainLevel);
    env4.sustain(sustainLevel); //not used rn
    env5.sustain(sustainLevel);//not used rn
    
    //change decay pt of ADSR based on hold time
    
    decayTime = hold_time * 0.7;  // roughly fade for 70% of the hold time
    if (decayTime > 4000) decayTime = 4000; // cap it so it doesn't last forever
    if (decayTime < 200)  decayTime = 200;
    env.decay(decayTime);
    env1.decay(decayTime);
    env2.decay(decayTime);
    env3.decay(decayTime);
    env4.decay(decayTime);//not used rn
    env5.decay(decayTime);//not used rn
/*
    //change release pt of ADSR based on hold time
    if (hold_time < 1000) releaseTime = 300;
    else if (hold_time <2000) releaseTime = 600;
    else if (hold_time>9000) releaseTime = 6000;
    env.release(releaseTime);
    env1.release(releaseTime);
    env2.release(releaseTime);
    env3.release(releaseTime);
    env4.release(releaseTime);
    env5.release(releaseTime);
    */


   
    noteOn(1, note[i], vel);
    delay(hold_time); //controls how long note stays in sustain stage
    noteOff(1, note[i], vel); //begins release stage
    delay(releaseTime+200);
    

  }
}

