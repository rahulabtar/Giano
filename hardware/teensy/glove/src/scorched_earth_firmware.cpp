#include <Arduino.h>
#include <serial_commands.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

#define TEENSY_HAND Hand::Left

const int NUM_VELOSTAT = 5; 

//LEFT PINS:
const int VELOSTAT_PINS[NUM_VELOSTAT] = {14, 18, 19, 21, 20}; 


// set default state of pressed vs unpressed to be unpressed
bool gPressed = false; 
// array to hold state of each sensor (pressed or unpressed)
bool gSensorState[NUM_VELOSTAT] = {false, false, false, false, false}; 
const int THRESHOLD = 45; 
int gBaseline[NUM_VELOSTAT];
const int ADC_BITS = 12;
#define FLEX_WRIST 22

/**
 * Velostat Calibration Function
 * Calibrates all velostat sensors based on 3 levels of pressure: open, soft press, hard press
 * Sets the baseline values for each velostat sensor based on calibration algorithm.
 */
void calibrateVelostat(unsigned int SAMPLE_COUNT = 200, unsigned int SAMPLE_PERIOD = 50) {

  int open_means[NUM_VELOSTAT];
  int open_stdevs[NUM_VELOSTAT];
  int soft_means[NUM_VELOSTAT];
  int soft_stdevs[NUM_VELOSTAT];
  int hard_means[NUM_VELOSTAT];
  int hard_stdevs[NUM_VELOSTAT];

  long sum;
  long sumSq;
  float mean; 
  float stdev; 

  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_VELO_NO_PRESS));
  delay(5500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  delay(2500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  delay(8500);

  //Serial.println(" Velostat Calibration for Open ...");
  //delay(1000); 
  //Serial.println("Please make sure all fingers are open (no pressure) Scrunch hands in and out");
  //delay(2000);
  //Serial.println("Starting now...");

  for (int finger = 0; finger < NUM_VELOSTAT; finger++) {

    //Serial.print("\nCalibrating finger ");
    //Serial.println(finger);

    sum = 0;
    sumSq = 0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
      int reading = analogRead(VELOSTAT_PINS[finger]);

      sum += reading;
      sumSq += (long)reading * (long)reading;

      delay(SAMPLE_PERIOD);
    }

    mean = (float)sum / SAMPLE_COUNT;
    stdev = sqrt(((float)sumSq / SAMPLE_COUNT) - (mean * mean));

    open_means[finger] = mean; 
    open_stdevs[finger] = stdev;
  }

  delay(200);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_SOFT_PRESS));
  delay(5500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  delay(2500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  delay(8500);
  //Serial.println(" Velostat Calibration for closed (light press)...");
  //delay(1000); 
  //Serial.println("Please hold all fingertips against surface lightly, like you are petting a cat :D");
  //delay(2000);
  //Serial.println("Starting now...");

  for (int finger = 0; finger < NUM_VELOSTAT; finger++) {

    //Serial.print("\nCalibrating finger ");
    //Serial.println(finger);

    sum = 0;
    sumSq = 0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
      int reading = analogRead(VELOSTAT_PINS[finger]);

      sum += reading;
      sumSq += (long)reading * (long)reading;

      delay(SAMPLE_PERIOD);
    }

    mean = (float)sum / SAMPLE_COUNT;
    stdev = sqrt(((float)sumSq / SAMPLE_COUNT) - (mean * mean));

    soft_means[finger] = mean; 
    soft_stdevs[finger] = stdev;
  }

  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_HARD_PRESS));
  delay(5500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  delay(2500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  delay(8500);
  //Serial.println(" Velostat Calibration for closed (hard press)...");
  //delay(1000);
  //Serial.println("Please hold all fingertips against surface hard :D");
  //delay(2000);
  //Serial.println("Starting now...");

  for (int finger = 0; finger < NUM_VELOSTAT; finger++) {

    //Serial.print("\nCalibrating finger ");
    //Serial.println(finger);

    sum = 0;
    sumSq = 0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
      int reading = analogRead(VELOSTAT_PINS[finger]);

      sum += reading;
      sumSq += (long)reading * (long)reading;

      delay(SAMPLE_PERIOD);
    }

    mean = (float)sum / SAMPLE_COUNT;
    stdev = sqrt(((float)sumSq / SAMPLE_COUNT) - (mean * mean));

    hard_means[finger] = mean; 
    hard_stdevs[finger] = stdev;
  }

  // implementing calibration algorithm now 
  int maxPress[NUM_VELOSTAT];
  for (int finger = 0; finger < NUM_VELOSTAT; finger++){
  gBaseline[finger] = open_means[finger] + 2 * open_stdevs[finger];
  maxPress[finger] = hard_means[finger] + hard_stdevs[finger];

  // TODO:  HOW DO WE EVEN ADDRESS THIS
  if (gBaseline[finger] >= maxPress[finger]) 
  {
    Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATION_FAILED));
    delay(2500);
    calibrateVelostat();
  } else {
    Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATION_SUCCESS));
    delay(2500);
    // Serial.println(finger);
    // Serial.print("Baseline: ");
    // Serial.println(gBaseline[finger]);
    // Serial.print("Max Press: ");
    // Serial.println(maxPress[finger]);
  }
}
}


/**
 * CHECK FINGER PRESS FUNCTION
 * Checks the state of each velostat sensor and sends press/release events to RasPi.
 * Primary call for freeplay mode, embedded call for learning mode.
 */
void checkFingerPress() {
  for (unsigned int i = 0; i < NUM_VELOSTAT; i++) {
    int raw = analogRead(VELOSTAT_PINS[i]);
    
    bool currentlyPressed = raw >= (gBaseline[i] + THRESHOLD);

    // transition: unpressed -> pressed
    if (currentlyPressed && !gSensorState[i]) {
      //Serial1.print("Sensor ");
      //Serial1.println(i);     // send "note on" to receiver
    
      // SENDS TO RASPI
      Serial.write(static_cast<uint8_t>(TEENSY_HAND));
      Serial.write(static_cast<uint8_t>(SensorValue::Pressed));
      Serial.write(static_cast<uint8_t>(i));     // send "note on" to receiver

      gSensorState[i] = true;  // remember it's pressed
    } 
    // transition: pressed -> released
    else if (!currentlyPressed && gSensorState[i]) {
      Serial1.print("SensorReleased ");
      Serial1.println(i);     // send "note off" to receiver

      // SENDING IT TO RASPI
      Serial.write(static_cast<uint8_t>(TEENSY_HAND));
      Serial.write(static_cast<uint8_t>(SensorValue::Released));
      Serial.write(static_cast<uint8_t>(i));     // send "note off" to receiver

      gSensorState[i] = false; // update state
    }

    // if pressed and already marked as pressed, do nothing
  }

  delay(50); // optional debounce/stability
}

/**
 * SETUP FUNCTION TO RUN AT STARTUP AND AID IN SELECTING SONG/ MODE
 */
void setup() {

    // Step 1: Initialize Serial Communication
    Serial.begin(115200); // for raspi <-> communication via USB
    delay(100); // just a buffer delay
    calibrateVelostat(); // velostat and flex sensor

}

void loop() {
    checkFingerPress();
}

