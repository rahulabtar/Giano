#include <Arduino.h>
#include <usbMIDI.h>
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

AudioMixer4              mixer;
AudioOutputI2S           i2sOutput;

// Audio connections
AudioConnection patch1(fmCarrier, 0, env, 0);   // Send carrier through envelope
AudioConnection patch2(env, 0, mixer, 0);       // Envelope output into mixer
AudioConnection patch3(fmModulator, 0, fmCarrier, 0); // FM connection

AudioConnection patch6(fmHarmonic1, 0, env1, 0);
AudioConnection patch7(fmHarmonic2, 0, env2, 0);
AudioConnection patch8(fmHarmonic3, 0, env3, 0);

AudioConnection patch11(env, 0, mixer, 0);
AudioConnection patch12(env1, 0, mixer, 1);
AudioConnection patch13(env2, 0, mixer, 2);
AudioConnection patch14(env3, 0, mixer, 3);
AudioConnection patch17(mixer, 0, i2sOutput, 0);
AudioConnection patch18(mixer, 0, i2sOutput, 1);

int note [] = {60, 62, 64, 65, 67, 69, 71, 72}; 
int vel = 60;
int hold_time=1000; // work on this - need a long delay. note ON sent, then hold until note OFF - not a set time, - sky
float decayTime = 500;
float sustainLevel = 0;
float releaseTime = 800;
AudioControlSGTL5000 audioShield;




void setup() {
    Serial.begin(115200); // receive midi over serial from raspi

  // Envelope setup - audio hat
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
  
}

// Audio Hat/ FM Helper functions

/**
 * Convert MIDI note number to frequency in Hz.
 */
float midiNoteToFreq(uint8_t note) {
  return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

/**
 * Convert MIDI velocity (0-127) to amplitude (0.0 - 1.0) using logarithmic scaling.
 */
float velocityToAmp(byte velocity, unsigned int scaling_factor){
  if (scaling_factor > 10) scaling_factor = 10;
  return logf(1.0f + scaling_factor * velocity / 127.0f) /
         logf(1.0f + scaling_factor);
}

/**
 * Handles note ON command
 * parameters: MIDI channel, note number, velocity
 */
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
}

/**
 * Handles note OFF command
 * parameters: MIDI channel, note number, velocity
 */
void noteOff(byte channel, byte note, byte velocity){
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  env.noteOff(); 
  env1.noteOff();
  env2.noteOff(); 
  env3.noteOff();
}


// main loop: reads in MIDI note data serially from raspi and calls noteOn/noteOff functions
void loop() {
    if(usbMIDI.read()) {
        Serial.print("NOTE RECEIVED: PROCESSING NOW");
        processMIDIData();
    }
    Serial.print("ERRROR: NO MIDI RECEIVED"); // debug statement
}


/**
  Processing Function: breaks MIDI data into pitch, velocity, and note ON vs. note OFF 
  Note ON/OFF is key, and will trigger the audio output stop/start reactions.
*/
void processMIDIData(){

  // gives us channel of message received - unsure if we need this
  int channel = usbMIDI.getChannel();
  // gives us status updates of MIDI, we only care about ON/OFF
  int onStatus = usbMIDI.getType();
  // gives us first data byte associated with pitch
  int pitch = usbMIDI.getData1();
  // gives us second data byte associate with velocity
  int velocity = usbMIDI.getData2();
  // gives us port/cable data byte, helpful to control MIDI routing
  int cable = usbMIDI.getCable();


  // switch case regarding if the MIDI message is ON or OFF
  // NOTE: there are other cases we can play around with if needed!
  switch(onStatus) {
    case usbMIDI.NoteOn:
      noteOn(channel, pitch, velocity);
      Serial.print("Note OFF, ch=");
      Serial.print(channel, DEC);
      Serial.print(", note=");
      Serial.print(pitch, DEC);
      Serial.print(", velocity=");
      Serial.println(velocity, DEC);
      break;

    case usbMIDI.NoteOff:
      noteOff(channel, pitch, velocity);
      Serial.print("Note OFF, ch=");
      Serial.print(channel, DEC);
      Serial.print(", note=");
      Serial.print(pitch, DEC);
      Serial.print(", velocity=");
      Serial.println(velocity, DEC);
      break;
  }
}






