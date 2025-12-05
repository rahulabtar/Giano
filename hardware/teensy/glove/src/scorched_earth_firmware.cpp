#include <Arduino.h>
#include <serial_commands.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

#define TEENSY_HAND Hand::Left

const int PRESS_THRESHOLD = 45;   // threshold to register a press
const int RELEASE_THRESHOLD = 25; // threshold to register a release (hysteresis) needs a lesser value 

const int NUM_VELOSTAT = 5; 

//LEFT PINS:
//const int VELOSTAT_PINS[NUM_VELOSTAT] = {14, 18, 19, 20, 21}; 

//RIGHT PINS
const int VELOSTAT_PINS[NUM_VELOSTAT] = {21, 20, 19, 18, 14}; 


int maxPress[NUM_VELOSTAT];

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
void calibrateVelostat(unsigned int SAMPLE_COUNT = 150, unsigned int SAMPLE_PERIOD = 10) {

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

  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_VELO_NO_PRESS));
  // delay(5500);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  // delay(2500);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  // delay(8500);

  Serial.println(" Velostat Calibration for Open ...");
  delay(1000); 
  Serial.println("Please make sure all fingers are open (no pressure) Scrunch hands in and out");
  delay(2000);
  Serial.println("Starting now...");

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

  // delay(200);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_SOFT_PRESS));
  // delay(5500);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  // delay(2500);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  // delay(8500);
  Serial.println(" Velostat Calibration for closed (light press)...");
  delay(1000); 
  Serial.println("Please hold all fingertips against surface lightly, like you are petting a cat :D");
  delay(2000);
  Serial.println("Starting now...");

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

  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_HARD_PRESS));
  // delay(5500);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  // delay(2500);
  // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  // delay(8500);
  Serial.println(" Velostat Calibration for closed (hard press)...");
  delay(1000);
  Serial.println("Please hold all fingertips against surface hard :D");
  delay(2000);
  Serial.println("Starting now...");

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
  for (int finger = 0; finger < NUM_VELOSTAT; finger++){
  gBaseline[finger] = open_means[finger] + 2 * open_stdevs[finger];
  maxPress[finger] = hard_means[finger] + hard_stdevs[finger];

  // TODO:  HOW DO WE EVEN ADDRESS THIS
  if (gBaseline[finger] >= maxPress[finger]) 
  {
    // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATION_FAILED));
    delay(2500);
    calibrateVelostat();
    Serial.println("Calibration failed, restarting...");
  } else {
    // Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATION_SUCCESS));
    Serial.println("Calibration successful!");
    delay(2500);
    // Serial.println(finger);
    // Serial.print("Baseline: ");
    // Serial.println(gBaseline[finger]);
    // Serial.print("Max Press: ");
    // Serial.println(maxPress[finger]);
  }

  Serial.println("DEBUG: 1");
}
  Serial.println("DEBUG: 2");
}


int getVelocity(bool currentlyPressed, int raw, int fingerIndex){
  int velocity = 0; //if it prints this something is wrong
  Serial.println("Velocity reading: ");
  if (currentlyPressed){  
    velocity = map(raw, gBaseline[fingerIndex], maxPress[fingerIndex], 1, 127);
    if(velocity > 127) {
      velocity = 127;
    }
    if (velocity < 0) {
      velocity = 0;
    }

    Serial.println(velocity);
  }
  else {
    velocity = 0;
    Serial.println(velocity);
  }
  return velocity;
}

/**
 * CHECK FINGER PRESS FUNCTION
 * Checks the state of each velostat sensor and sends press/release events to RasPi.
 * Primary call for freeplay mode, embedded call for learning mode.
 */

void checkFingerPress() {
    unsigned long now = millis();

    for (unsigned int i = 0; i < NUM_VELOSTAT; i++) {
        int raw = analogRead(VELOSTAT_PINS[i]);
        bool currentlyPressed = gSensorState[i]; // default: maintain previous state

        // Hysteresis logic
        if (!gSensorState[i] && raw >= gBaseline[i] + PRESS_THRESHOLD) {
            currentlyPressed = true; // transition: unpressed -> pressed
        } 
        else if (gSensorState[i] && raw <= gBaseline[i] + RELEASE_THRESHOLD) {
            currentlyPressed = false; // transition: pressed -> released
        }

        // Only act on state change
        if (currentlyPressed != gSensorState[i]) {
            gSensorState[i] = currentlyPressed; // update state

            if (currentlyPressed) {
                Serial.print("Press - Hand: ");
                Serial.print(static_cast<int>(TEENSY_HAND));
                Serial.print(", Finger: ");
                Serial.print(i);
                Serial.print(", Velocity: ");
                Serial.println(getVelocity(true, raw, i));
            } 
            else {
                Serial.print("Release - Hand: ");
                Serial.print(static_cast<int>(TEENSY_HAND));
                Serial.print(", Finger: ");
                Serial.print(i);
                Serial.print(", Velocity: ");
                Serial.println(getVelocity(false, raw, i));
            }
        }
    }
    // debounce delay
    delay(50);
}


/**
 * SETUP FUNCTION TO RUN AT STARTUP AND AID IN SELECTING SONG/ MODE
 */
void setup() {

    // Step 1: Initialize Serial Communication
    Serial.begin(115200); // for raspi <-> communication via USB
    delay(100); // just a buffer delay
    Serial.println("Scorched Earth Firmware Starting...");
    calibrateVelostat(); // velostat and flex sensor
    Serial.println("DEBUG: EXITING CALIBRATION");

    delay(100);

    Serial.println("DEBUG: ENTERING HAPTIC CALIBRATION");
    calibrateHaptics();

}

int i = 0; 
void loop() {
    static unsigned long lastPrint = 0;
    static int i = 0;
    
    checkFingerPress();

    if (millis() - lastPrint >= 200) {  // print every 200ms
        Serial.print("Counter Loop: ");
        Serial.println(i);
        i++;
        lastPrint = millis();
    }
}

