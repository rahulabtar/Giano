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

AudioEffectEnvelope      env;      
AudioEffectEnvelope      env1;
AudioEffectEnvelope      env2;
AudioEffectEnvelope      env3;

AudioMixer4              mixer; //CONTINUE HERE; TRY TO CHANNEL TWO AUDIOMIXER4 TO ADD MORE INHARMONIC HARMONICS; MUST CHANGE MEMORY TO 30 I THINK
AudioOutputI2S           i2sOutput;

//AudioFilterStateVariable body;



// Audio connections
AudioConnection patch1(fmCarrier, 0, env, 0);   // Send carrier through envelope
AudioConnection patch2(fmModulator, 0, fmCarrier, 0); // FM connection

AudioConnection patch3(fmHarmonic1, 0, env1, 0); //send harmonic 1 to env1
AudioConnection patch4(fmHarmonic2, 0, env2, 0);// send harmonic 2 to env2
AudioConnection patch5(fmHarmonic3, 0, env3, 0); //send harmonic 3 to env3

AudioConnection patch6(env, 0, mixer, 0); //send env to mixer audio input 0
AudioConnection patch7(env1, 0, mixer, 1); //send env to mixer audio input 1
AudioConnection patch8(env2, 0, mixer, 2);  //send env to mixer audio input 2
AudioConnection patch9(env3, 0, mixer, 3); //send env to mixer audio input 3 (note: teensy audio mixer has inputs 0-3)

AudioConnection patch10(mixer, 0, i2sOutput, 0); //take mixer and send it to i2s output, channel 0; left channel (of speaker/ headphones)
AudioConnection patch11(mixer, 0, i2sOutput, 1); //take mixer and send it to i2s output, channel 1; right channel (of speaker/ headphones)
//AudioConnection pBody(mixer, 1, body, 0);
//AudioConnection pOutL(body, 0, i2sOutput, 0);
//AudioConnection pOutR(body, 0, i2sOutput, 1);

int note [] = {60, 62, 64, 65, 67, 69, 71, 72}; 
int vel = 70; //fixed for now
//int hold_time=1000;
float decayTime = 500;
float sustainLevel = 0.9;
float releaseTime = 800;
bool noteIsOn = false;

AudioControlSGTL5000 audioShield;

void setup() {
  AudioMemory(20);
  audioShield.enable();
  audioShield.volume(0.4);
  
  //body.frequency(2000); 
  //body.resonance(6);


  // Envelope setup
  env.attack(0.8); // fixed attack for now, can always integrate velocity here later using something like env.attack(3 - vel/70)
  env.decay(800);    // milliseconds
  env.sustain(sustainLevel); //sustain level is amplitude percentage
  env.release(800);    

  env1.attack(0.8);  // milliseconds
  env1.decay(600);    // milliseconds
  env1.sustain(sustainLevel);  //sustain level is amplitude percentage
  env1.release(800);    

  env2.attack(0.8); // milliseconds
  env2.decay(400);    // milliseconds
  env2.sustain(sustainLevel);  //sustain level is amplitude percentage
  env2.release(800);    
  
  env3.attack(0.8); // milliseconds
  env3.decay(200);    // milliseconds
  env3.sustain(sustainLevel);  //sustain level is amplitude percentage
  env3.release(800);    
  

}

//converts a midi note into its respective freq
float midiNoteToFreq(uint8_t note) {
  return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

//converts velocity you give it to an amplitude
float velocityToAmp(byte velocity, unsigned int scaling_factor){
  if (scaling_factor > 10) scaling_factor = 10;
  return logf(1.0f + scaling_factor * velocity / 127.0f) /
         logf(1.0f + scaling_factor);
}

//when a note is played, turn it ON
void noteOn(byte channel, byte note, byte velocity) {
  noteIsOn = true;

  //sustainLevel = 0.8; //resets to this every time a new note is played
  Serial.printf("Note ON: Ch %d, Note %d, Vel %d\n", channel, note, velocity);
  float freq = midiNoteToFreq(note);
  float amp = velocityToAmp(velocity, 5);

  float B = 0.0008;

  //note: higher harmonics should decay faster (set to lower amplitude) to create piano like sound
  fmCarrier.frequency(freq); //fundamental freq is set for each note
  fmCarrier.amplitude(amp); //amp set for each note; determined by velocity (which is set to a fixed # for now)
  fmModulator.frequency(freq*1); //should typically be a 1:1 or 4:1 ratio btwn carrier and modulator
  fmModulator.amplitude(amp*0.3);  
  fmHarmonic1.frequency(2*freq * sqrtf(1+B*4)); // first harmonic
  fmHarmonic1.amplitude(amp * 0.1); //all harmonic amps should ideally be n/2, set to lower bc sounds better (higher harmonic amplitudes are lower)
  fmHarmonic2.frequency(3*freq * sqrtf(1+B*9)); //second harmonic
  fmHarmonic2.amplitude(amp * 0.05);
  fmHarmonic3.frequency(4*freq * sqrtf(1+B*16)); //third harmonic
  fmHarmonic3.amplitude(amp * 0.015);
 


//turn on all envelopes
  env.noteOn(); //diff than this noteOn void function, acc envelope method
  env1.noteOn(); 
  env2.noteOn();
  env3.noteOn();

//keep on changing sustain &/ release part of ADSR until noteOff is called


    /* initial logic, needs to change bc cannot depend on hold_time
 if (hold_time<400) sustainLevel=0.8; 
    else if (hold_time<1000) sustainLevel=0.5;
    else if (hold_time<2000) sustainLevel=0.3;
    else sustainLevel=0.05; //assuming very long hold time
    env.sustain(sustainLevel);
    env1.sustain(sustainLevel);
    env2.sustain(sustainLevel);
    env3.sustain(sustainLevel);
    */
   
}

//when user releases the key, turn note OFF
void noteOff(byte channel, byte note, byte velocity){
  noteIsOn = false;
  Serial.printf("Note OFF: Ch %d, Note %d\n", channel, note);
  env.noteOff(); //turn off all envelopes
  env1.noteOff();
  env2.noteOff(); 
  env3.noteOff();
}


void updateSustain(){
  if (noteIsOn){
    float decayFactor = 0.99; //has to be below 1.0, otherwise it is not decreasing the sustain
    sustainLevel*= decayFactor;

    if (sustainLevel < 0.05) sustainLevel = 0.05; //FIX

    env.sustain(sustainLevel);
    env1.sustain(sustainLevel);
    env2.sustain(sustainLevel);
    env3.sustain(sustainLevel);
  }
}



void loop() {
  for (int i=0; i<8; i++){
    noteOn(1, note[i], vel);
    delay(1000); // for testing purposes only, this depends on how long user presses key for
    noteOff(1, note[i], vel);
    delay(2000);
  }
  
  
  //noteOff(1, 60, 60);

  /*
  for (int i = 0; i < 8; i++) {
    
 
//change sustain pt of ADSR based on hold time
    if (hold_time<400) sustainLevel=0.8;
    else if (hold_time<1000) sustainLevel=0.5;
    else if (hold_time<2000) sustainLevel=0.3;
    else sustainLevel=0.05; //assuming very long hold time
    env.sustain(sustainLevel);
    env1.sustain(sustainLevel);
    env2.sustain(sustainLevel);
    env3.sustain(sustainLevel);
    
    //change decay pt of ADSR based on hold time
    
    decayTime = hold_time * 0.7;  // roughly fade for 70% of the hold time
    if (decayTime > 4000) decayTime = 4000; // cap it so it doesn't last forever
    if (decayTime < 200)  decayTime = 200;
    env.decay(decayTime);
    env1.decay(decayTime);
    env2.decay(decayTime);
    env3.decay(decayTime);
/*
    //change release pt of ADSR based on hold time
    if (hold_time < 1000) releaseTime = 300;
    else if (hold_time <2000) releaseTime = 600;
    else if (hold_time>9000) releaseTime = 6000;
    env.release(releaseTime);
    env1.release(releaseTime);
    env2.release(releaseTime);
    env3.release(releaseTime);
    env4.release(releaseTime);
    env5.release(releaseTime);
    */

    /*
  noteOn(1, note[i], vel);
  delay(hold_time); //controls how long note stays in sustain stage (keeps note on until note off is called) -> ideally will not have this delay, the delay will be when the note off is called
    noteOff(1, note[i], vel); //begins release stage
    delay(releaseTime+200);

    */

  }


//hold time logic needs to change bc we cannot take in hold time as a variable
//will need to call note on when a note is played and the note will stop playing when the noteoff function is called
//took out env4and5 from patches bc no longer needed
//have on off switch for the note
//mixer.vol (0.8) etc to set it instead of hold_time
//everything should be in the note on and note off functions ideally






    /*
    //change decay pt of ADSR based on hold time
    
    decayTime = hold_time * 0.7;  // roughly fade for 70% of the hold time
    if (decayTime > 4000) decayTime = 4000; // cap it so it doesn't last forever
    if (decayTime < 200)  decayTime = 200;
    env.decay(decayTime);
    env1.decay(decayTime);
    env2.decay(decayTime);
    env3.decay(decayTime);
    */