#include <Arduino.h>
#include <types.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

/**
 * Working toward putting setups for both modes in here - freeplay and learning.
 * Calls proper calibration script, as well as finger press checking function.
 */

// Setting Type for the hand: *must manually changed before uploading to each hand*
#define TEENSY_HAND Hand::Left


// Set button pins for left glove - CHECK THESE PLEASEEEE!!!!
const int BUTTON_MODE = 10; 
const int BUTTON_SONG = 11; 

// Global variable for setting mode - default to freeplay mode
bool gFreeplayMode = true;


// VELOSTAT SETUP VARIABLES
// number of velostat sensors
const int NUM_VELOSTAT = 5; 
// array of size of # of velostat sensors, sets their pins - ADD THIS
// thumb index 0, pinky at 4
const int VELOSTAT_PINS[NUM_VELOSTAT] = {A0, A1, A2, A3, A4}; 
// set default state of pressed vs unpressed to be unpressed
bool gPressed = false; 
// array to hold state of each sensor (pressed or unpressed)
bool gSensorState[NUM_VELOSTAT] = {false, false, false, false, false}; 
// threshold for determining pressed vs unpressed - CAN BE ADJUSTED
const int THRESHOLD = 45; 
// to collect baseline readings of velostat sensors
int gBaseline[NUM_VELOSTAT];
// ADC bits for analog read resolution
const int ADC_BITS = 12;

// FLEX SETUP VARIABLES
// #define FLEX_POINTER 15
// #define FLEX_RING 23
#define FLEX_WRIST 22


// HAPTIC SETUP VARIABLES
const int NUM_HAPTICS = 7; 
// SET THUMB at index 0, Pinky at 4, left wrist 5, right wrist 6
// ie for right hand it might be reversed 6 -> 0 
// THIS IS ACTUALLY SETUP FOR SELECT LINES
const int HAPTIC_PINS[NUM_HAPTICS] = {0, 1, 2, 3, 4, 5, 6};
// TCA MUX ADDRESS
#define TCAADDR 0x77
// INSTANTIATE HAPTIC DRIVER OBJECT
Adafruit_DRV2605 drv;


/**
 * SETUP FUNCTION TO RUN AT STARTUP AND AID IN SELECTING SONG/ MODE
 */
void setup {

    // Step 1: Initialize Serial Communication
    Serial.begin(9600); // for raspi <-> communication via USB
    Wire.begin();
    delay(1000); // just a buffer delay

    // Step 1.5: Send confirmation byte to RasPi of which hand this is
    Serial.write(TEENSY_HAND);

    // Step 2: Initialize all necessary sensors/button components for setup logic
    // we only want buttons to work if on left hand
    if(TEENSY_HAND == Hand::Left) { 
        pinMode(BUTTON_MODE, INPUT_PULLUP);
        pinMode(BUTTON_SONG, INPUT_PULLUP);
    }
    analogReadResolution(ADC_BITS);
    analogReadAveraging(8); // mild data smoothing

    // Step 3: Setup, mode, and song selection logic! 
    // send notif to PYTHON to play the initial setup message of 

    // "welcome to GIANO, please select the mode by pressing the leftmost button. once 
    // for freeplay, twice for learning mode."
    
    // wait adequate time for user to press button, detect how many times it was pressed 
    // in that window of time, and set mode accordingly
    if(TEENSY_HAND == Hand::Left) {
        int modePressCount = buttonPressCount(5000, BUTTON_MODE); // 5 second window to press button
        if(modePressCount == 1) {
            gFreeplayMode = true; 
        } else if (modePressCount >= 2) {
            gFreeplayMode = false; 
        }
        // write the mode selected back to python for confirmation
        Serial.print(gFreeplayMode);
    }

    // Step 4: Call song selection calibration function to calibrate all sensors BASED ON MODE
    // if freeplay mode only call velostat and flex sensor calibration
    // if learning mode call velostat, flex, and haptic calibration (needs mux)
    switch(gFreeplayMode) {
        case true:
            gFreeplayMode = true;
            calibrateVelostat(); // only calibrate velostat and flex sensors
            break;
        case false:
            gFreeplayMode = false;
            calibrateVelostat(); // velostat and flex sensors
            calibrateHaptics(); // haptics calibration function
            // send notice to python to hash out song selection
            /**
             * You have selected learning mode. Please select a song by pressing the rightmost button
             * once for Song 1, twice for Song 2, thrice for Song 3.
             */
            // detect and wait to see how many times button was pressed, send that 
            // number back to the python unit - only for left hand
            if(TEENSY_HAND == Hand::Left) {
                int songPressCount = buttonPressCount(5000, BUTTON_SONG); // 5 second window to press button
                Serial.print(songPressCount); // send selected song number back to python
            }
            break;
    }

    // Step 5: Final confirmation message to python that setup is complete and game can begin.
    // really a message that says "if u press the mode button at any point during gameplay,
    // you can change modes."

    // final confirmation message to python - this cues it to play out 
    // final message on audio hat and then clear its input buffer or whatever needs to be done
    Serial.print("COMPLETE");

}


void loop() {
    // NEED TO INTEGRATE A BUTTON INTERRUPT: IF AT ANY POINT THE MODE BUTTON IS PRESSED,
    // RETURN TO SETUP LOGIC TO CHANGE MODES??? 

    // now, depending on the mode, call the proper logic.
    // if in freeplay mode, call finger press checking function because thats all we need 
    if(gFreeplayMode) {
        checkFingerPress();
    }
    else if(!gFreeplayMode) {
        guideFingerPress();
    }
}



/**
 * HELPER FUNCTIONS
 * Various helper functions for calibration, tca select line function, checking finger presses, etc.
 */


/**
 * TCA SELECT LINE FUNCTION
 * Selects the proper line on the TCA MUX for haptic motor control
 */
void tcaSelect(uint8_t i) {
  if (i > 7) return;  
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

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

  Serial.println(" Velostat Calibration for Open ...");
  delay(1000); 
  Serial.println("Please make sure all fingers are open (no pressure) Scrunch hands in and out");
  delay(2000);
  Serial.println("Starting now...");

  for (int finger = 0; finger < NUM_VELOSTAT; finger++) {

    Serial.print("\nCalibrating finger ");
    Serial.println(finger);

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

  Serial.println(" Velostat Calibration for closed (light press)...");
  delay(1000); 
  Serial.println("Please hold all fingertips against surface lightly, like you are petting a cat :D");
  delay(2000);
  Serial.println("Starting now...");

  for (int finger = 0; finger < NUM_VELOSTAT; finger++) {

    Serial.print("\nCalibrating finger ");
    Serial.println(finger);

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

  Serial.println(" Velostat Calibration for closed (hard press)...");
  delay(1000);
  Serial.println("Please hold all fingertips against surface hard :D");
  delay(2000);
  Serial.println("Starting now...");

  for (int finger = 0; finger < NUM_VELOSTAT; finger++) {

    Serial.print("\nCalibrating finger ");
    Serial.println(finger);

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

  if (gBaseline[finger] >= maxPress[finger]) 
  {
    Serial.print("Calibration failed for finger ");
    Serial.println(finger);
    Serial.println("Please redo calibration with better finger presses");
  } else {
    Serial.print("Calibration successful for finger ");
    Serial.println(finger);
    Serial.print("Baseline: ");
    Serial.println(gBaseline[finger]);
    Serial.print("Max Press: ");
    Serial.println(maxPress[finger]);
  }
}
}

/**
 * Haptics Calibration Function
 * Calibrates all haptic motors by initializing them via the TCA MUX
 * and playing a test effect to ensure they are working.
 */
void calibrateHaptics() {
  Serial.println("Starting Haptics Calibration...");

  for (int i = 0; i < NUM_HAPTICS; i++) {
    Serial.print("Calibrating Haptic Motor at MUX line ");
    Serial.println(i);

    tcaSelect(HAPTIC_PINS[i]);
    if (!drv.begin()) {
      Serial.print("Failed to initialize haptic motor at MUX line ");
      Serial.println(i);
      continue;
    }
    drv.setMode(DRV2605_MODE_INTTRIG);

    // Play a test effect to ensure motor is working - do we even need this
    drv.setWaveform(0, 1); // simple click effect
    drv.setWaveform(1, 0);
    drv.go();
    delay(1000); // wait for effect to finish

    Serial.print("Haptic Motor at MUX line ");
    Serial.print(i);
    Serial.println(" calibrated successfully.");
  }

  Serial.println("Haptics Calibration Complete.");
}

/**
 * BUTTON PRESS COUNT FUNCTION
 * Counts the number of button presses within a specified time window.
 */
int buttonPressCount(unsigned long timeWindow, int buttonPin) {
    unsigned long start = millis();
    bool lastState = HIGH;  // INPUT_PULLUP default - does this make sense?
    int count = 0;

    while (millis() - start < timeWindow) {
        bool currentState = digitalRead(buttonPin);

        // detect falling edge to count a valid pres
        if (lastState == HIGH && currentState == LOW) {
            count++;
            delay(50);  // basic debounce
        }

        lastState = currentState;
    }

    return count;
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

/**
 * Function to help guide finger presses in learning mode. 
 * Uses haptics to signal each finger and where it should press. 
 */
void guideFingerPress() {
    // Step 1: Get the set of finger instructions that need to happen

    // Step 2: Apply it using haptics 

    // Step 3: detect a finger press, when that happens send back to Raspi for 
    // confirmation

}


