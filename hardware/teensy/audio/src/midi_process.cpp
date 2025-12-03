#include <Arduino.h>
#include <usb_midi.h>

/**
  Function to pass off audio processing/handling for ON
*/
void playAudioHat() {
  // FILLER STUB FOR NOW, just to check if it reaches here
  Serial.println("Congrats! You played a note!");
}

/**
  Function to pass off audio processing/handling for OFF
*/
void terminateAudioHat() {
  // FILLER STUB FOR NOW, just to check if it reaches here
  Serial.println("Note stopped! Play the next!");
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
      playAudioHat();
      Serial.print("Note OFF, ch=");
      Serial.print(channel, DEC);
      Serial.print(", note=");
      Serial.print(pitch, DEC);
      Serial.print(", velocity=");
      Serial.println(velocity, DEC);
      break;

    case usbMIDI.NoteOff:
      terminateAudioHat();
      Serial.print("Note OFF, ch=");
      Serial.print(channel, DEC);
      Serial.print(", note=");
      Serial.print(pitch, DEC);
      Serial.print(", velocity=");
      Serial.println(velocity, DEC);
      break;
  }
}



