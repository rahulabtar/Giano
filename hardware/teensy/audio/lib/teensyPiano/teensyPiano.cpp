#include "teensyPiano.h"

int TeensyPiano::setup() {
  // NO MORE THAN 16 VOICES
  if (NUM_VOICES > 16)
    { return -1; }
  
  
  
  // we just resize the voices_ and voice_connections_
  // vectors to the number of voices
  
  // TODO: change to static array
  voice_connections_.resize(NUM_VOICES);

  // this the number of mixers/connections we will need to connect the voices
  unsigned int n_mixers = ceilf((float)NUM_VOICES / 4.0f);
  
  mixers_.resize(n_mixers);
  mixer_connections_.resize(n_mixers);

  for (unsigned int i = 0; i < NUM_VOICES; i++) {
    unsigned int mixer_idx = floorf(i/4.0f); 
    unsigned int mixer_channel = i % 4;
    
    // set voice[i] up!
    voices_[i].setup();

    // connect voice[i] to mixer[floor(i/4)] channel i % 4
    voice_connections_[i].connect(voices_[i].mixer_out, 0, mixers_[mixer_idx], mixer_channel);
  }

  for (unsigned int i = 0; i < mixers_.size(); i++) {
    mixer_connections_[i].connect(mixers_[i], 0, piano_mixer_out, i);
  }

  // return 0 if successful
  return 0;
}

void TeensyPiano::voiceOn(unsigned int idx, u_int8_t midiNote, u_int8_t vel) {
  voices_[idx].noteOn(midiNote, vel);
  areVoicesOn_[idx] = true;
  voicePitches_[idx] = midiNote;
}

void TeensyPiano::voiceOff(unsigned int idx) {
  voices_[idx].noteOff();
  areVoicesOn_[idx] = false;
  voicePitches_[idx] = -1;
}

std::array<bool, NUM_VOICES> TeensyPiano::areVoicesOn() {
  return areVoicesOn_;
}

std::array<int, NUM_VOICES> TeensyPiano::getVoicePitches() {
  return voicePitches_;
}

TeensyPiano::~TeensyPiano() {
  
}