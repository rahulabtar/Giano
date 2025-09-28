#include <Arduino.h>
#include <usbMIDI.h>

// RASPI <-> TEENSY MIDI COMMUNICATION

/**
  Simple Script to read data over the USB connection
  This breaks it down into note on, note off, etc. 
  I have created filler functions (likely commented out for now) 
  that will mimick the audio output handoff for the case of note on versus note off commands
  for Nik/Rahul to mess around with.

  Happy Teensying!!! - Sky
*/


void setup() {
  Serial.begin(115200); // we can change BAUDRATE as needed
}

/**
  Main Loop: Constant Datastream of reading USB
  Note: Each MIDI interaction is processed before the next one is read
  each MIDI interaction cannot be processed unless it is succesfully read 
  (will help us catch errors for now, can be converted into a timeout and retry later)
*/
void loop() {
  if(usbMIDI.read()) {
    Serial.print("NOTE RECEIVED: PROCESSING NOW");
    processMIDIData();
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

