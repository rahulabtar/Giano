#include "TeensyPianoVoice.h"

void TeensyPianoVoice::setup() {
  // setup FM carrier and modulator defaults
  // Use the modulated sine carrier: the modulator audio output will be
  // connected into the carrier's modulation input (see below).
  
  //Modulator frequency should be >> carrier (14x)
  fmModulator.frequency(1.0f);
  fmModulator.amplitude(1.0f);
  
  // Modulator -> carrier
  fmCarrier.frequency(1.0f);
  fmCarrier.amplitude(0.0f);

  
  // setup envelopes
  
  carrierEnv_.attack(10.0f);
  // 4 second decay
  carrierEnv_.decay(4000.0f);
  carrierEnv_.sustain(0.0f);
  carrierEnv_.release(100.0f);

  modulatorEnv_.attack(2.0f);
  modulatorEnv_.decay(100.0f);
  modulatorEnv_.sustain(0.1);
  modulatorEnv_.release(50.0f);

  env1_.attack(3.0f);
  env1_.decay(500);
  env1_.sustain(0.5);
  env1_.release(800);

  env2_.attack(3.0f);
  env2_.decay(500);
  env2_.sustain(0.5);
  env2_.release(800);

  env3_.attack(3.0f);
  env3_.decay(500);
  env3_.sustain(0.5);
  env3_.release(800);

  // setup connections

  //fm carrier is connected to  carrierEnv_
  patch1_.connect(fmCarrier, 0, carrierEnv_, 0);
 
  // carrier envelope is connected to output
  patch9_.connect(carrierEnv_, 0, mixer_out, 0);


  // route modulator -> amplifier -> carrier modulation input
  patchModToAmp_.connect(fmModulator, 0, modAmp, 0);
  patchAmpToEnv_.connect(modAmp, 0, modulatorEnv_, 0);
  patchAmpEnvToCarrier_.connect(modulatorEnv_, 0, fmCarrier, 0);

  // patch6_.connect(fmHarmonic1, 0, env1_, 0);
  // patch7_.connect(fmHarmonic2, 0, env2_, 0);
  // patch8_.connect(fmHarmonic3, 0, env3_, 0);

  patch10_.connect(env1_, 0, mixer_out, 1);
  patch11_.connect(env2_, 0, mixer_out, 2);
  patch12_.connect(env3_, 0, mixer_out, 3);  

  // Old code
  // patch9_.connect  carrierEnv_, 0, mixer, 1);
  // patch10_.connect(env1_, 0, mixer, 2);
  // patch11_.connect(env2_, 0, mixer, 3);
  // patch12_.connect(env3_, 0, mixer, 3);


}


float TeensyPianoVoice::midiNoteToFreq(uint8_t note) {
      return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

float TeensyPianoVoice::velocityToAmp(u_int8_t velocity, unsigned int scaling_factor) {
  if (scaling_factor > 10) scaling_factor = 10;
  return logf(1.0f + scaling_factor * velocity / 127.0f) /
        logf(1.0f + scaling_factor);
}

void TeensyPianoVoice::noteOn(u_int8_t note, u_int8_t velocity) {
    
  // convert midi note number to frequency
  float freq = midiNoteToFreq(note);

  //convert velocity input to amplitude
  float amp = velocityToAmp(velocity, 5);
  
  AudioNoInterrupts();

  
  fmCarrier.frequency(freq);
  fmCarrier.amplitude(0.5*amp);

  // set modulator to modulatorRatio (high harmonic) of the carrier; keep modulator amplitude at 1.0
  fmModulator.frequency(freq * kModRatio);
  fmModulator.amplitude(1.0f);
  // control modulation depth via the amplifier gain (modAmp)
  // modDepth is a simple scalar (tweak as needed). Scale by velocity (0..1).
  modAmp.gain(kModDepth * amp);
  
  // change ADSR
  carrierEnv_.attack(10.0f - (8.0f * amp));
  carrierEnv_.decay(3000.0f + (1000.0f * amp));
  modulatorEnv_.attack(2.0f - (1.0f * amp));

  // REMOVED ADDITIVE STUFF
  // env1_.attack(3.0f - (velocity/ 70));
  // env1_.decay(500);
  // env1_.sustain(0.5);
  // env1_.release(800);

  // env2_.attack(3 - (velocity/ 70));
  // env2_.decay(500);
  // env2_.sustain(0.5);
  // env2_.release(800);

  // env3_.attack(3 - (velocity/ 70));
  // env3_.decay(500);
  // env3_.sustain(0.5);
  // env3_.release(800);

  // fmHarmonic1.frequency(freq * 2);
  // fmHarmonic1.amplitude(amp * 0.4);
  // fmHarmonic2.frequency(freq * 3);
  // fmHarmonic2.amplitude(amp * 0.25);
  // fmHarmonic3.frequency(freq * 4);
  // fmHarmonic3.amplitude(amp * 0.15);
  // fmHarmonic1.frequency(freq * 2);
  // fmHarmonic1.amplitude(0.0f);
  // fmHarmonic2.frequency(freq * 3);
  // fmHarmonic2.amplitude(0.0f);
  // fmHarmonic3.frequency(freq * 4);
  // fmHarmonic3.amplitude(0.0f);

  carrierEnv_.noteOn();
  modulatorEnv_.noteOn();
  // env1_.noteOn();
  // env2_.noteOn();
  // env3_.noteOn();

  AudioInterrupts();
  voiceOn_ = true;
  pitch_ = note;

}


void TeensyPianoVoice::noteOff() {
  AudioNoInterrupts();
  carrierEnv_.noteOff();
  modulatorEnv_.noteOff();

  // commented out to save cpu
  // env1_.noteOff();
  // env2_.noteOff();
  // env3_.noteOff();

  AudioInterrupts();

  voiceOn_ = false;
  pitch_ = -1; 
}

bool TeensyPianoVoice::isVoiceOn() {
  return voiceOn_;
}

int TeensyPianoVoice::getPitch() {
  return pitch_;
}

TeensyPianoVoice::~TeensyPianoVoice() {}