#pragma once
#include "teensyPianoVoice.h"
#include <vector>
#include <math.h>

class TeensyPiano{
  public:
    TeensyPiano() {}
    int setup(unsigned int n_voices);

    //this method will turn on voice {idx}
    void voiceOn(unsigned int idx, byte midiNote, u_int8_t vel);
    void voiceOff(unsigned int idx);
    AudioMixer4 piano_mixer_out;

  
  private:
    std::vector<TeensyPianoVoice> voices_;
    std::vector<AudioConnection> voice_connections_;

    std::vector<AudioMixer4> mixers_;
    std::vector<AudioConnection> mixer_connections_;
};