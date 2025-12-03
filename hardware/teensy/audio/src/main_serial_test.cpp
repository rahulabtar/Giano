#include "midi_process.h"
#include <Arduino.h>
#include "teensyPiano.h"
#include "voiceCommands.h"

TeensyPiano piano; 
VoiceCommands voiceCmds;
AudioControlSGTL5000 sgtl5000_1;
AudioMixer4 masterMixer;
AudioOutputI2S audioOutput;
AudioConnection patchOut;
AudioConnection instrOut; 
AudioConnection masterPatch1; 
AudioConnection masterPatch2;

// TODO: remember you have stashed changes
void setup() {
    AudioMemory(350); //probably should leave this as is
    sgtl5000_1.enable();
    sgtl5000_1.volume(0.7);

    Serial.begin(115200);
    delay(100);
    Serial.println("Setting up SD Card...");
    int sdSuccess = voiceCmds.setUpSD();
    Serial.print("SD setup returned: ");
    Serial.println(sdSuccess);
    int success = piano.setup(); 
    Serial.print("Piano setup returned: ");
    Serial.println(success);
    patchOut.connect(piano.piano_mixer_out, 0, masterMixer, 0);
    instrOut.connect(voiceCmds.playWav1, 0, masterMixer, 1);
    masterPatch1.connect(masterMixer, 0, audioOutput, 0);
    masterPatch2.connect(masterMixer, 0, audioOutput, 1);
}

void loop() {
    if(usbMIDI.read()) {
    Serial.print("NOTE RECEIVED: PROCESSING NOW");
    processMIDIData();
    }
    //delay(1000);
}