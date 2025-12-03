#pragma once
#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include "consts.h"

// base FM depth; adjust to taste
const float kModDepth = 2.0f; 

// base ratio of modulator to fundamental
const float kModRatio = 14.0f;

class TeensyPianoVoice
{
  public:
    //methods
    TeensyPianoVoice() {}
    void setup();

    // Note on function
    void noteOn(u_int8_t note, u_int8_t velocity);
    void noteOff();

    // Returns true if noteOn was called without accompanying noteOff
    bool isVoiceOn();

    // mapping functions
    float midiNoteToFreq(uint8_t note);
    float velocityToAmp(u_int8_t velocity, unsigned int scaling_factor);

    int vel = 70; // velocity (you may change)

    // objects
    // output mixer object
    AudioMixer4              mixer_out;

    ~TeensyPianoVoice();


  private:
    // methods

    // voiceOn property
    bool voiceOn_;
    float modulatorRatio_;
    AudioSynthWaveformSineModulated fmCarrier;
    // Waveform objects
    AudioSynthWaveformSine   fmModulator;
    AudioSynthWaveformSine   fmHarmonic1;
    AudioSynthWaveformSine   fmHarmonic2; 
    AudioSynthWaveformSine   fmHarmonic3;
    
    //ADSR envelopes
    AudioEffectEnvelope      carrierEnv_; 
    AudioEffectEnvelope      modulatorEnv_;

    AudioEffectEnvelope      env1_;
    AudioEffectEnvelope      env2_;
    AudioEffectEnvelope      env3_;

    // connections between objects
    AudioConnection patch1_;
    AudioConnection patch3_;

    AudioConnection patch6_;
    AudioConnection patch7_;
    AudioConnection patch8_;

    // amplifier to control modulation depth and its connections
    AudioAmplifier           modAmp;
    AudioConnection         patchModToAmp_;
    AudioConnection         patchAmpToEnv_;
    AudioConnection         patchAmpEnvToCarrier_;

    AudioConnection patch9_;
    AudioConnection patch10_;
    AudioConnection patch11_;
    AudioConnection patch12_;
    AudioConnection patch13_;
    AudioConnection patch14_;


    // Auto note-off timing
    unsigned long noteOnTime[NUM_SENSORS] = {0};


};
