#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>
#include <usb_midi.h>

//ONLY PUSH THIS CODE TO TEENSY WITH AUDIO HAT
// GUItool: begin automatically generated code
AudioSynthWaveform       synth_1;
AudioSynthWaveformSine   sine1;          //xy=49.33332824707031,183.3333282470703
AudioOutputI2S           i2s1;           //xy=187.33332443237305,181.3333396911621
AudioConnection          patchCord1(synth_1, 0, i2s1, 0); //connect synth to i2s channel 0
AudioConnection          patchCord2(synth_1, 0, i2s1, 1); //connect synth to i2s channel 1
AudioControlSGTL5000     sgtl5000_1;

// GUItool: end automatically generated code

// Note frequencies for MIDI notes (A4 = 440Hz = MIDI note 69)
float midiNoteToFreq(uint8_t note) {
  return 440.0 * pow(2.0, (note - 69) / 12.0);
}

void noteOnCallback(byte channel, byte note, byte velocity) {
  Serial.printf("Note ON: Ch %d, Note %d, Vel %d\n", channel, note, velocity);
  
  // Convert MIDI note to frequency
  float freq = midiNoteToFreq(note);
  synth_1.frequency(freq);
  
  // Convert velocity (0-127) to amplitude (0.0-0.8)
  // Using 0.8 as max to avoid clipping/distortion
  float amplitude = (velocity / 127.0) * 0.5;
  synth_1.amplitude(amplitude);
  
  Serial.printf("Playing %.2f Hz at %.2f amplitude\n", freq, amplitude);
}

void noteOffCallback(byte channel, byte note, byte velocity) {
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  
  // Turn off the sound by setting amplitude to 0
  
  synth_1.amplitude(0.0);
}

void setup() {
  Serial.begin(115200);
  while (! Serial);
  
  // Audio setup
  sine1.amplitude(1.0);
  sine1.frequency(440.0);
  AudioMemory(10);
  sgtl5000_1.enable();
  sgtl5000_1.volume(0.5); // 0.8 is the MAX volume you should use, it will hurt ur ears

  
  synth_1.frequency(440.0);
  synth_1.amplitude(0.0); // Start silent
  synth_1.begin(WAVEFORM_SQUARE);
  Serial.println("Audio out setup complete");

  // Set up MIDI callbacks
  usbMIDI.setHandleNoteOn(noteOnCallback);
  usbMIDI.setHandleNoteOff(noteOffCallback);
  
  Serial.println("USB MIDI Synthesizer Ready!");
}

void loop() {
  // Check for incoming MIDI messages
  // This will trigger the callbacks when MIDI data is received
  usbMIDI.read();
  
}

