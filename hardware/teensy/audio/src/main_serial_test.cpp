#include "midi_process.h"
#include <Arduino.h>
#include "teensyPiano.h"

TeensyPiano piano; 
AudioControlSGTL5000 sgtl5000_1;
AudioOutputI2S audioOutput;
AudioConnection patchOut;

// TODO: remember you have stashed changes
void setup() {
    AudioMemory(60); //probably should leave this as is
    sgtl5000_1.enable();
    sgtl5000_1.volume(0.7);

    Serial.begin(115200);
    delay(100);
    int success = piano.setup(); 
    Serial.print("Piano setup returned: ");
    Serial.println(success);
    patchOut.connect(piano.piano_mixer_out, 0, audioOutput, 0);
}

void loop() {
    if(usbMIDI.read()) {
    Serial.print("NOTE RECEIVED: PROCESSING NOW");
    processMIDIData();
    }
    // piano.voiceOn(0, 60, 100);
    // Serial.println("piano ON");
    // delay(1000);
    // piano.voiceOff(0);
    // Serial.println("piano OFF");

    delay(1000);
}