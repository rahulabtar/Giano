#pragma once

#include "teensyPianoVoice.h"
#include <vector>
#include <math.h>

class TeensyPiano{
  public:
    TeensyPiano() {}
    int setup(unsigned int n_voices);

    //this method will turn on voice {idx}
    void voiceOn(unsigned int idx, u_int8_t midiNote, u_int8_t vel);
    
    // this method will turn off voice {idx}
    void voiceOff(unsigned int idx);

    bool areVoicesOn();
    AudioMixer4 piano_mixer_out;
    ~TeensyPiano();
  
  private:
    std::vector<TeensyPianoVoice> voices_;
    std::vector<AudioConnection> voice_connections_;

    std::vector<AudioMixer4> mixers_;
    std::vector<AudioConnection> mixer_connections_;
};