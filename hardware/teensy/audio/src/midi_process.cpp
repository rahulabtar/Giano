#include <Arduino.h>
#include <usb_midi.h>
#include "midi_process.h"
#include "piano_globals.h"
#include "serial_commands.h"

/**
  Function to pass off audio processing/handling for ON
*/
void playAudioHat(uint8_t pitch, uint8_t velocity) {
  // FILLER STUB FOR NOW, just to check if it reaches here
  Serial.println(" Congrats! You played a note!");
  std::array<bool, NUM_VOICES> voices = piano.areVoicesOn();
  for (int i = 0; i < NUM_VOICES; i++) {
    if (!voices[i]) {
      piano.voiceOn(i, pitch, velocity); // always getting assigned to zero for some reason
      return;
    }
  }
  //piano.voiceOn(0, pitch, velocity);
}
/**
  Function to pass off audio processing/handling for OFF
*/
void terminateAudioHat(int pitch) {
  // FILLER STUB FOR NOW, just to check if it reaches here
  Serial.println(" Note stopped! Play the next!");
  for (int i = 0; i < NUM_VOICES; i++) {
    std::array<int, NUM_VOICES> voicePitches = piano.getVoicePitches();
    if (voicePitches[i] == pitch) {
      piano.voiceOff(i);
      return;
    }
  }
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
    Serial.println(pitch);
      if (pitch >= 60){
        playAudioHat(pitch, velocity); //only plays for notes in the octaves supported 
      } 
      else{
        Serial.println(" Playing voice command...");
        VocalCommandCodes command = static_cast<VocalCommandCodes>(pitch);
        const char* filename = voiceCmds.getFileName(command);
        voiceCmds.playInstruction(filename);
      }
      break;

    case usbMIDI.NoteOff:
      if (pitch >= 60){
        terminateAudioHat(pitch);
      }
      else{
        Serial.println(" Stopping voice command...");
        voiceCmds.playWav1.stop();  
      }
      break;
  }
}



