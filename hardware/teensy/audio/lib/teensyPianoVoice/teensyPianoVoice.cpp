#include "TeensyPianoVoice.h"

void TeensyPianoVoice::setup() {
  // setup envelopes
  

  env_.attack(3 - (vel / 70));
  env_.decay(500);
  env_.sustain(0.5);
  env_.release(800);

  env1_.attack(3 - (vel / 70));
  env1_.decay(500);
  env1_.sustain(0.5);
  env1_.release(800);

  env2_.attack(3 - (vel / 70));
  env2_.decay(500);
  env2_.sustain(0.5);
  env2_.release(800);

  env3_.attack(3 - (vel / 70));
  env3_.decay(500);
  env3_.sustain(0.5);
  env3_.release(800);

  // setup connections
  patch1_.connect(fmCarrier, 0, env_, 0);
  patch3_.connect(fmModulator, 0, fmCarrier, 0);

  patch6_.connect(fmHarmonic1, 0, env1_, 0);
  patch7_.connect(fmHarmonic2, 0, env2_, 0);
  patch8_.connect(fmHarmonic3, 0, env3_, 0);

  patch9_.connect(env_, 0, mixer_out, 0);
  patch10_.connect(env1_, 0, mixer_out, 1);
  patch11_.connect(env2_, 0, mixer_out, 2);
  patch12_.connect(env3_, 0, mixer_out, 3);  

  // Old code
  // patch9_.connect(env_, 0, mixer, 1);
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

  fmCarrier.frequency(freq);
  fmCarrier.amplitude(amp);
  fmModulator.frequency(freq * 100);
  fmModulator.amplitude(amp * 0.0001);
  fmHarmonic1.frequency(freq * 2);
  fmHarmonic1.amplitude(amp * 0.4);
  fmHarmonic2.frequency(freq * 3);
  fmHarmonic2.amplitude(amp * 0.25);
  fmHarmonic3.frequency(freq * 4);
  fmHarmonic3.amplitude(amp * 0.15);

  env_.noteOn();
  env1_.noteOn();
  env2_.noteOn();
  env3_.noteOn();
  voiceOn_ = true;

}

  void TeensyPianoVoice::noteOff() {
    env_.noteOff();
    env1_.noteOff();
    env2_.noteOff();
    env3_.noteOff();
    voiceOn_ = false;
  }
  
  bool TeensyPianoVoice::isVoiceOn() {
    return voiceOn_;
  }

  TeensyPianoVoice::~TeensyPianoVoice() {}