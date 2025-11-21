#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

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

AudioConnection patch1(fmCarrier, 0, env, 0);
AudioConnection patch2(env, 0, mixer, 0);
AudioConnection patch3(fmModulator, 0, fmCarrier, 0);

AudioConnection patch6(fmHarmonic1, 0, env1, 0);
AudioConnection patch7(fmHarmonic2, 0, env2, 0);
AudioConnection patch8(fmHarmonic3, 0, env3, 0);

AudioConnection patch11(env, 0, mixer, 1);
AudioConnection patch12(env1, 0, mixer, 2);
AudioConnection patch13(env2, 0, mixer, 3);
AudioConnection patch14(env3, 0, mixer, 3);
AudioConnection patch17(mixer, 0, i2sOutput, 0);
AudioConnection patch18(mixer, 0, i2sOutput, 1);



const int NUM_SENSORS = 2;

int sensorToNote[NUM_SENSORS] = {
  60, 
  71  
};

int vel = 100; // velocity (you may change)

// Auto note-off timing
const unsigned long NOTE_DURATION = 200; // ms
unsigned long noteOnTime[NUM_SENSORS] = {0};
bool isNoteOn[NUM_SENSORS] = {false};

float midiNoteToFreq(uint8_t note) {
  return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

float velocityToAmp(byte velocity, unsigned int scaling_factor){
  if (scaling_factor > 10) scaling_factor = 10;
  return logf(1.0f + scaling_factor * velocity / 127.0f) /
         logf(1.0f + scaling_factor);
}

void noteOn(byte channel, byte note, byte velocity) {
  Serial.printf("Note ON: Ch %d, Note %d, Vel %d\n", channel, note, velocity);
  float freq = midiNoteToFreq(note);
  float amp = velocityToAmp(velocity, 5);

  fmCarrier.frequency(freq);
  fmCarrier.amplitude(amp);
  fmModulator.frequency(freq * 100);
  fmModulator.amplitude(amp * 0.0001);
  fmHarmonic1.frequency(freq * 2);
  fmHarmonic1.amplitude(amp * 0.4);
  fmHarmonic2.frequency(freq * 3);
  fmHarmonic2.amplitude(amp * 0.25);
  fmHarmonic3.frequency(freq * 4);
  fmHarmonic3.amplitude(amp * 0.15);

  env.noteOn();
  env1.noteOn();
  env2.noteOn();
  env3.noteOn();
}

void noteOff(byte channel, byte note, byte velocity) {
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  env.noteOff();
  env1.noteOff();
  env2.noteOff();
  env3.noteOff();
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);

  Serial.println("Receiver Ready");

  env.attack(3 - (vel / 70));
  env.decay(500);
  env.sustain(0.5);
  env.release(800);

  env1.attack(3 - (vel / 70));
  env1.decay(500);
  env1.sustain(0.5);
  env1.release(800);

  env2.attack(3 - (vel / 70));
  env2.decay(500);
  env2.sustain(0.5);
  env2.release(800);

  env3.attack(3 - (vel / 70));
  env3.decay(500);
  env3.sustain(0.5);
  env3.release(800);
}


void loop() {
  checkSerial1ForSensorMessages();
  autoNoteOffHandler();
}

void checkSerial1ForSensorMessages() {
  if (!Serial1.available()) return;

  String msg = Serial1.readStringUntil('\n');
  msg.trim();

  if (msg.startsWith("Sensor")) {
    int idx = msg.substring(7).toInt();

    if (idx >= 0 && idx < NUM_SENSORS) {
      int midiNote = sensorToNote[idx];
      Serial.printf("Trigger from Sensor %d → Note %d\n", idx, midiNote);

      noteOn(1, midiNote, vel);

      isNoteOn[idx] = true;
      noteOnTime[idx] = millis();
    }
  }
}

void autoNoteOffHandler() {
  unsigned long now = millis();

  for (int i = 0; i < NUM_SENSORS; i++) {
    if (isNoteOn[i] && (now - noteOnTime[i] >= NOTE_DURATION)) {
      int midiNote = sensorToNote[i];
      noteOff(1, midiNote, 0);
      isNoteOn[i] = false;
    }
  }
}
