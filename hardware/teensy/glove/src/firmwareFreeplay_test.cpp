/**
FREEPLAY MODE Firmware
 Basically just continually tests for a button press and then spits out whatever note 
 sends a flag to the python to check CV because a note was pressed.
 Doesnt care what finger, just that SOME finger was detected


 For testing purposes: will assign a random note to each finger just so we know that 
 it works and directly outputs to audio

*/

#include <Arduino.h>
#include <types.h>

// HAND RELATED ENUM


// THIS WILL BE FOR LEFT HAND
#define TEENSY_HAND Hand::Left

// BUTTON / MODE RELATED VARIABLES
const int BUTTON_MODE = 2;
const int BUTTON_SONG = 3;



bool gFreeplayMode = true; // THIS IS JUST FOR TESTING THE INTERRUPT OF THIS VERSION. IN THE REAL FIRMWARE WE WILL JUST WIND UP SWITCHING MODES NOT EXITING


// VELOSTAT VARIABLES

const int NUM_VELOSTAT = 2; // num of fingers eventually
const int VELOSTAT_PINS[NUM_VELOSTAT] = {A0, A1}; // pinouts for velostat
const int THRESHOLD = 45; // threshold for pressed/unpressed
const int ADC_BITS = 12;

// to collect baseline readings 
int gBaseline[NUM_VELOSTAT];

bool gPressed = false;

bool gSensorState[NUM_VELOSTAT] = {false, false};

// ADD HAPTICS SHIT HERE FOR LEARNING MODE

void setup() {
  Serial.begin(9600);
  Serial1.begin(9600); // just for testing this - we will wind up using USB -> RASPI -> USB, but this allows us to do communication between the TX and RX teensy to teensy

  delay(3000);

  // send one confirmation byte to the RasPi 
  Serial.println(TEENSY_HAND);

  // BUTTON SETUP: INTEGRATE THIS WITH VOICE COMMANDS PLS AND THANKS
  pinMode(BUTTON_MODE, INPUT_PULLUP);
  pinMode(BUTTON_SONG, INPUT_PULLUP);

  analogReadResolution(ADC_BITS); // to easily change the number of bits needed for the ADC if we choose tgo change it
  analogReadAveraging(8); // mild data smoothing


  // VELOSTAT SETUP: CAPTURE THE BASLINE VALUES OF SENSORS

  Serial.println("Calibrating Velostats");
  for(int i = 0; i < NUM_VELOSTAT; i++) {
    long sum = 0; 
    const int SAMPLERATE = 200; 
    for(int j = 0; j < SAMPLERATE; j++) {
      sum += analogRead(VELOSTAT_PINS[i]);
      delay(5);
    }

    gBaseline[i] = sum / SAMPLERATE; 

    Serial.print("Baseline for sensor");
    Serial.print(i);
    Serial.print(" = ");
    Serial.println(gBaseline[i]);
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  if(!gFreeplayMode) {
    // this is just a debug statement, should NEVER reach this if setup is properly communicated.
    Serial.println("NOT IN FREEPLAY MODE SOMETHING IS WRONG for this test");
  }

  if(digitalRead(BUTTON_MODE) == LOW) {
    Serial.println("SWITCHING MODE");
    if(gFreeplayMode) {
      gFreeplayMode = false; // if true set to false
    } else if (!gFreeplayMode) {
      gFreeplayMode = true; // if false set it to true
    }
    delay(200); // debounce time for button logic
  }

  checkFingerPress();

}


// TODO: Write equation for determining velostat resistance
void checkFingerPress() {
  for (unsigned int i = 0; i < NUM_VELOSTAT; i++) {
    int raw = analogRead(VELOSTAT_PINS[i]);
    bool currentlyPressed = raw >= (gBaseline[i] + THRESHOLD);

    // transition: unpressed -> pressed
    if (currentlyPressed && !gSensorState[i]) {
      Serial1.print("Sensor ");
      Serial1.println(i);     // send "note on" to receiver
    
      // SENDS TO RASPI
      Serial.write((u_int8_t)TEENSY_HAND);
      Serial.write(SensorValue::Pressed);
      Serial.write(i);

      gSensorState[i] = true;  // remember it's pressed
    } 
    // transition: pressed -> released
    else if (!currentlyPressed && gSensorState[i]) {
      Serial1.print("SensorReleased ");
      Serial1.println(i);     // send "note off" to receiver

      // SENDING IT TO RASPI
      Serial.write((u_int8_t)TEENSY_HAND);
      Serial.write(SensorValue::Released);
      Serial.write(i);     // send "note off" to receiver

      gSensorState[i] = false; // update state
    }

    // if pressed and already marked as pressed, do nothing
  }

  delay(50); // optional debounce/stability
}






