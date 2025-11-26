#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include "teensyPiano.h"
#include <vector>

TeensyPiano piano;
std::vector<AudioConnection> pianoConnectors;

AudioControlSGTL5000 audioShield;
AudioOutputI2S i2sOutput;

AudioConnection i2sCord;

bool gIsNoteOn[NUM_SENSORS] = {false};

// not needed for end code
int sensorToNote[NUM_SENSORS] = {
  60, 
  71  
};

enum AudioStates {
  AUDIO_HAT_WELCOME = 0,
  MODE_SELECT,
  VELOSTAT_CALIBRATION,
  PLAYING_MODE_FREEPLAY,
  PLAYING_MODE_LEARNING
};

AudioStates gCurSystemState = AUDIO_HAT_WELCOME;

void setup() {

  AudioMemory(64);
  

  Serial.begin(115200);
  Serial1.begin(9600);

  Serial.println("Receiver Ready");

  // setup the piano with 5 voices
  piano.setup(5);
  delay(100);
  
  i2sCord.connect(piano.piano_mixer_out, 0, i2sOutput, 0);
  i2sCord.connect(piano.piano_mixer_out, 0, i2sOutput, 1);

  // audioShield.enable();
  // audioShield.volume(0.4);

}


void loop() {

  switch(gCurSystemState) {
    case(AUDIO_HAT_WELCOME):
      
      break;

    case(MODE_SELECT):
      break;
    
    case(VELOSTAT_CALIBRATION):
      break;

    case(PLAYING_MODE_FREEPLAY):
      break;

    case(PLAYING_MODE_LEARNING):
      break;


  };
  
  // checkSerial1ForSensorMessages();
  //autoNoteOffHandler();

}

void readSerialMessages() {
  while (Serial.available()) {
    byte c = Serial.read();
    static String msg = "";
    if (c == '\n') {
      msg.trim();

      // NOTE ON
      if (msg.startsWith("Sensor ")) {
        int idx = msg.substring(7).toInt();
        if (idx >= 0 && idx < NUM_SENSORS && !gIsNoteOn[idx]) { 
          int midiNote = sensorToNote[idx];
          //idk how to do this yet
          // Serial.printf("Note ON: Ch %d, Note %d, Vel %d\n", channel, note, velocity);
          u_int8_t vel = 70;
          Serial.printf("Trigger from Sensor %d → Note %d\n", idx, midiNote);
          piano.voiceOn(idx, midiNote, vel);

        }
      }


      // NOTE OFF
      else if (msg.startsWith("SensorReleased ")) {
        int idx = msg.substring(15).toInt();
        if (idx >= 0 && idx < NUM_SENSORS && gIsNoteOn[idx]) { 
          int midiNote = sensorToNote[idx];
          Serial.printf("Release from Sensor %d → Note %d\n", idx, midiNote);
          piano.voiceOff(idx);
          
          // Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);

          gIsNoteOn[idx] = false;
        }
      }

          msg = ""; // reset for next message
        } else {
          msg += c;
        }
    }
}



/*void autoNoteOffHandler() {
  unsigned long now = millis();

  for (int i = 0; i < NUM_SENSORS; i++) {
    if (isNoteOn[i] && (now - noteOnTime[i] >= NOTE_DURATION)) {
      int midiNote = sensorToNote[i];
      noteOff(1, midiNote, 0);
      isNoteOn[i] = false;
    }
  }
}
*/

