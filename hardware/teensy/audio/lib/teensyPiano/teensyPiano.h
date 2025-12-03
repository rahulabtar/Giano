#pragma once

#include "teensyPianoVoice.h"
#include <vector>
#include <math.h>
#include <array>
#include "consts.h"

class TeensyPiano{
  public:
    TeensyPiano() {}
    // number of voices is fixed at compile time as NUM_VOICES
    int setup();

    //this method will turn on voice {idx}
    void voiceOn(unsigned int idx, u_int8_t midiNote, u_int8_t vel);
    
    // this method will turn off voice {idx}
    void voiceOff(unsigned int idx);

    //
    std::array<bool, NUM_VOICES> areVoicesOn();
    std::array<int, NUM_VOICES> getVoicePitches();
    AudioMixer4 piano_mixer_out;
    ~TeensyPiano();
  
  private:
    std::array<TeensyPianoVoice, NUM_VOICES> voices_;
    std::vector<AudioConnection> voice_connections_;

    std::vector<AudioMixer4> mixers_;
    std::vector<AudioConnection> mixer_connections_;

    // array with the status of voices
    std::array<bool, NUM_VOICES> areVoicesOn_;
    std::array<int, NUM_VOICES> voicePitches_;
};