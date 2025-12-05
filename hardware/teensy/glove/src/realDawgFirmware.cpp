#include <Arduino.h>
#include <serial_commands.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"

// LEFT HAND VARIABLES: UNCOMMENT THESE ALL FOR LEFT HAND

#define TEENSY_HAND Hand::Left
const int VELOSTAT_PINS[NUM_VELOSTAT] = {14, 18, 19, 20, 21}; 
const int HAPTIC_PINS[NUM_HAPTICS] = {2, 1, 0, 6, 5, 4, 3};

// RIGHT HAND VARIABLES: UNCOMMENT THESE ALL FOR RIGHT HAND

//#define TEENSY_HAND Hand::Right
//const int VELOSTAT_PINS[NUM_VELOSTAT] = {21, 20, 19, 18, 14}; 
// const int HAPTIC_PINS[NUM_HAPTICS] = {5, 6, 0, 1, 2, 4, 3};

// VELOSTAT SETUP VARIABLES
const int PRESS_THRESHOLD = 45;   // threshold to register a press
const int RELEASE_THRESHOLD = 25; // threshold to register a release (hysteresis) needs a lesser value 
const int NUM_VELOSTAT = 5; 
int maxPress[NUM_VELOSTAT];
// set default state of pressed vs unpressed to be unpressed
bool gPressed = false; 
// array to hold state of each sensor (pressed or unpressed)
bool gSensorState[NUM_VELOSTAT] = {false, false, false, false, false}; 
int gBaseline[NUM_VELOSTAT];
const int ADC_BITS = 12;


// FLEX SETUP VARIABLES
#define FLEX_WRIST 22

// HAPTIC SETUP VARIABLES
const int NUM_HAPTICS = 7; 
#define TCAADDR 0x77
Adafruit_DRV2605 drv;

// BUTTON SETUP FOR LEFT GLOVE
// LEFTMOST BUTTON CONTROLS SONG, RIGHT CONTROLS MODE
const int BUTTON_MODE = 10; 
const int BUTTON_SONG = 11; 

// Global variable for setting mode - default to freeplay mode
volatile bool gFreeplayMode = true;

// Global variable for changing mode- FLAG TO CHECK IF MODE TOGGLE REQUESTED
volatile bool gModeToggleRequested = false;

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
  Wire1.beginTransmission(TCAADDR);
  Wire1.write(1 << i);
  Wire1.endTransmission();
}

/**
 * INTERRUPT SERVICE PROTOCOL FUNCTION
 * Sets flag of toggle mode requested to true
 */
void modeButtonISR() {
    gFreeplayMode = !gFreeplayMode;
    gModeToggleRequested = true;
}


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

  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIB_VELO_NO_PRESS));
  delay(5500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATING));
  delay(2500);
  Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATE_SINE_WAVE));
  delay(8500);

  // Serial.println(" Velostat Calibration for Open ...");
  // delay(1000); 
  // Serial.println("Please make sure all fingers are open (no pressure) Scrunch hands in and out");
  // delay(2000);
  // Serial.println("Starting now...");

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
  // Serial.println(" Velostat Calibration for closed (light press)...");
  // delay(1000); 
  // Serial.println("Please hold all fingertips against surface lightly, like you are petting a cat :D");
  // delay(2000);
  // Serial.println("Starting now...");

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
  // Serial.println(" Velostat Calibration for closed (hard press)...");
  // delay(1000);
  // Serial.println("Please hold all fingertips against surface hard :D");
  // delay(2000);
  // Serial.println("Starting now...");

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

  if (gBaseline[finger] >= maxPress[finger]) 
  {
    Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATION_FAILED));
    delay(2500);
    calibrateVelostat();
    // Serial.println("Calibration failed, restarting...");
  } else {
    Serial.write(static_cast<uint8_t>(VoiceCommands::CALIBRATION_SUCCESS));
    // Serial.println("Calibration successful!");
    delay(2500);
    // Serial.println(finger);
    // Serial.print("Baseline: ");
    // Serial.println(gBaseline[finger]);
    // Serial.print("Max Press: ");
    // Serial.println(maxPress[finger]);
  }

  // Serial.println("DEBUG: 1");
}
  // Serial.println("DEBUG: 2");
}

/**
 * Haptics Calibration Function
 * Calibrates all haptic motors by initializing them via the TCA MUX
 * and playing a test effect to ensure they are working.
 */
void calibrateHaptics() {
  // Serial.println("Starting Haptics Calibration...");

  for (int i = 0; i < NUM_HAPTICS; i++) {
    // Serial.print("Calibrating Haptic Motor at MUX line ");
    // Serial.println(i);

    tcaSelect(HAPTIC_PINS[i]);
    delay(100); // give I2C time to switch
    
    if (!drv.begin(&Wire1)) {
      // Serial.print("Failed to initialize haptic motor at MUX line ");
      // Serial.println(i);
      continue;
    }
    
    // Serial.println("  - DRV2605 initialized");
    drv.selectLibrary(1); // Use library 1 for better effects
    drv.setMode(DRV2605_MODE_INTTRIG);
    
    // Clear any previous waveforms
    drv.setWaveform(0, 0);
    drv.setWaveform(1, 0);
    
    // Play a test effect to ensure motor is working
    drv.setWaveform(0, 47); // strong click effect
    drv.setWaveform(1, 0);
    drv.go();
    // Serial.println("  - Effect triggered");
    
    delay(100); // wait longer for effect to complete and be audible

    // Serial.print("Haptic Motor at MUX line ");
    // Serial.print(i);
    // Serial.println(" calibrated successfully.");
  }

  // Serial.println("Haptics Calibration Complete.");
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
 *  GET VELOCITY FUNCTION RETURNS VELOCITY OF FINGER PRESS
 */
int getVelocity(bool currentlyPressed, int raw, int fingerIndex){
  int velocity = 0; //if it prints this something is wrong
  // Serial.println("Velocity reading: ");
  if (currentlyPressed){  
    velocity = map(raw, gBaseline[fingerIndex], maxPress[fingerIndex], 1, 127);
    if(velocity > 127) {
      velocity = 127;
    }
    if (velocity < 0) {
      velocity = 0;
    }

    // Serial.println(velocity);
  }
  else {
    velocity = 0;
    // Serial.println(velocity);
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
                Serial.write(static_cast<uint8_t>(TEENSY_HAND));
                Serial.write(static_cast<uint8_t>(i));
                Serial.write(static_cast<uint8_t>(SensorValue::PRESSED));
                Serial.write(static_cast<uint8_t>(getVelocity(true, raw, i)));
                Serial.flush();


                // Serial.print("Press - Hand: ");
                // Serial.print(static_cast<int>(TEENSY_HAND));
                // Serial.print(", Finger: ");
                // Serial.print(i);
                // Serial.print(", Velocity: ");
                // Serial.println(getVelocity(true, raw, i));
            } 
            else {
                Serial.write(static_cast<uint8_t>(TEENSY_HAND));
                Serial.write(static_cast<uint8_t>(i));
                Serial.write(static_cast<uint8_t>(SensorValue::RELEASED));
                Serial.write(static_cast<uint8_t>(getVelocity(true, raw, i)));
                Serial.flush();
                // Serial.print("Release - Hand: ");
                // Serial.print(static_cast<int>(TEENSY_HAND));
                // Serial.print(", Finger: ");
                // Serial.print(i);
                // Serial.print(", Velocity: ");
                // Serial.println(getVelocity(false, raw, i));
            }
        }
    }
    // debounce delay
    delay(50);
}

/**
 * BUZZ MOTOR FUNCTION, allows us to long buzz the motor for debug.
 */
void buzzMotor(int sensorIndex) {
  tcaSelect(sensorIndex);
    delay(100); // give I2C time to switch
    
    if (!drv.begin(&Wire1)) {
      continue;
    }
    
    drv.selectLibrary(1); // Use library 1 for better effects
    drv.setMode(DRV2605_MODE_INTTRIG);
    
    // Clear any previous waveforms
    drv.setWaveform(0, 0);
    drv.setWaveform(1, 0);
    
    // Play a test effect to ensure motor is working
    drv.setWaveform(0, 47); // strong click effect
    drv.setWaveform(1, 0);
    drv.go();
    // Serial.println("  - Effect triggered");
    
    delay(100); // wait longer for effect to complete and be audible
}


// /**
//  * Function to help guide finger presses in learning mode. 
//  * Uses haptics to signal each finger and where it should press. 
//  */
// void guideFingerPress() {
//     // Step 1: Get the set of finger instructions that need to happen

//     // Step 2: Apply it using haptics 

//     // Step 3: detect a finger press, when that happens send back to Raspi for 
//     // confirmation

// }


/**
 * SETUP FUNCTION TO RUN AT STARTUP AND AID IN SELECTING SONG/ MODE
 */
void setup() {

    // Step 1: Initialize Serial and wait for USB enumeration
  Serial.begin(115200);
  Wire1.begin();
  
  // Wait for USB CDC to be ready (crucial!)
  while (!Serial) {
    delay(10);
  }
  delay(100);  // Additional stabilization time
  
  // Clear any stale data that arrived during boot
  while (Serial.available()) {
    Serial.read();
  }
  
  // ========== ROBUST HANDSHAKE PROTOCOL ==========
  const uint8_t HANDSHAKE_REQUEST = 0xAA;
  const uint8_t HANDSHAKE_ACK = 0x55;
  
  // Wait for handshake with timeout
  bool handshake_complete = false;
  unsigned long handshake_start = millis();
  const unsigned long HANDSHAKE_TIMEOUT = 30000;  // 30 second timeout
  
  while (!handshake_complete && (millis() - handshake_start < HANDSHAKE_TIMEOUT)) {
    // Check if we received handshake request
    if (Serial.available() > 0) {
      uint8_t received = Serial.read();
      
      if (received == HANDSHAKE_REQUEST) {
        // Clear any additional bytes
        delay(10);
        while (Serial.available()) {
          Serial.read();
        }
        
        // Send ACK
        Serial.write(HANDSHAKE_ACK);
        Serial.flush();  // Ensure it's sent
        delay(50);
        
        // Send hand identifier
        Serial.write(static_cast<uint8_t>(TEENSY_HAND));
        Serial.flush();
        delay(50);
        
        // Wait for confirmation from Pi
        unsigned long confirm_start = millis();
        while (millis() - confirm_start < 10000) {  // 10 second timeout
          if (Serial.available() > 0) {
            uint8_t confirm = Serial.read();
            if (confirm == HANDSHAKE_ACK) {
              handshake_complete = true;
              break;
            }
          }
          delay(10);
        }
        
        if (handshake_complete) {
          break;  // Exit handshake loop
        }
      }
    }
    
    delay(50);  // Small delay between checks
  }
  
  // If handshake failed, blink LED as error indicator
  if (!handshake_complete) {
    pinMode(LED_BUILTIN, OUTPUT);
    while (true) {  // Stuck in error state
      digitalWrite(LED_BUILTIN, HIGH);
      delay(200);
      digitalWrite(LED_BUILTIN, LOW);
      delay(200);
    }
  }
  
  // Clear buffers after successful handshake
  delay(100);
  while (Serial.available()) {
    Serial.read();
  }

  // CALIBRATE HAPTICS ON BOTH HANDS!! 
  calibrateHaptics();
  
  // ========== NOW PROCEED WITH SETUP ==========
    // Step 2: Initialize all necessary sensors/button components for setup logic
    // we only want buttons to work if on left hand
    if(TEENSY_HAND == Hand::Left) { 
        pinMode(BUTTON_MODE, INPUT_PULLUP);
        pinMode(BUTTON_SONG, INPUT_PULLUP);
    }
    analogReadResolution(ADC_BITS);
    analogReadAveraging(8); // mild data smoothing


    if(TEENSY_HAND == Hand::Left) {
        // startup song
        // BOOTUP SOUND
        while (!Serial.availableForWrite()) {
          delay(5);
        }
        Serial.write(static_cast<uint8_t>(VoiceCommands::WELCOME_MUSIC));
        
        delay(6000);

        // Step 3: Setup, mode, and song selection logic! 
        // send notif to PYTHON to play the initial setup message of 

        // "welcome to GIANO, please select the mode by pressing the leftmost button. once 
        // for freeplay, twice for learning mode."
        
        
        Serial.write(static_cast<uint8_t>(VoiceCommands::WELCOME_TEXT));
        delay(2500);
        Serial.write(static_cast<uint8_t>(VoiceCommands::MODE_SELECT_BUTTONS));
        delay(5500);

        // wait adequate time for user to press button, detect how many times it was pressed 
        // in that window of time, and set mode accordingly

        int modePressCount = buttonPressCount(5000, BUTTON_MODE); // 5 second window to press button
        
        if(modePressCount == 1) {
            gFreeplayMode = true; 
            Serial.write(static_cast<uint8_t>(VoiceCommands::FREEPLAY_MODE_CONFIRM));
            delay(1500);
        } else if (modePressCount >= 2) {
            gFreeplayMode = false; 
            Serial.write(static_cast<uint8_t>(VoiceCommands::LEARNING_MODE_CONFIRM));
            delay(1500);
        } else {
        gFreeplayMode = true;  // default to freeplay
        }
        // write the mode selected back to python for confirmation
        Serial.write(static_cast<uint8_t>(gFreeplayMode));
    }

    // IF this is the right hand, we need to receive and set the mode from the Python brain
    if(TEENSY_HAND == Hand::Right) {
        // read from the serial input and set the mode!!
        while(!Serial.available()) {
        delay(5);
      }
      uint8_t modeByte = Serial.read();
      if (modeByte == static_cast<uint8_t>(5)) {
        gFreeplayMode = 0;
      } else if (modeByte == static_cast<uint8_t>(6) {
        gFreeplayMode = 1;
      }
    }

    //LEARNING MODE IS 5  



    // Step 4: Call calibration function to calibrate all sensors!
    // Since we already did haptics, it can do velostat regardless of mode
    calibrateVelostat();

    // if in LEARNING MODE - move on to song selection
    if (!gFreeplayMode) {
      /**
      * You have selected learning mode. Please select a song by pressing the rightmost button
      * once for Song 1, twice for Song 2, thrice for Song 3.
      */
      if (TEENSY_HAND == Hand::Left) {
          // Step 5: SONG SELECTION 
          if (Serial.availableForWrite()) {Serial.write(static_cast<uint8_t>(VoiceCommands::SELECT_SONG));}
          delay(7500);

        // detect and wait to see how many times button was pressed, send that 
        // number back to the python unit - only for left hand

          int songPressCount = buttonPressCount(6000, BUTTON_SONG); // 6 second window to press button
          Serial.write(static_cast<uint8_t>(songPressCount)); // send selected song number back to python
          delay(10);
      } 
    } 

    // Step 6: Final confirmation message to python that setup is complete and game can begin.
    // really a message that says "if u press the mode button at any point during gameplay,
    // you can change modes."

    Serial.write(static_cast<uint8_t>(VoiceCommands::HOW_TO_CHANGE_MODE));
    delay(5500);
    Serial.write(static_cast<uint8_t>(VoiceCommands::HOW_TO_RESET_SONG));
    delay(5500);


    //SETUP INTERRUPT FOR LEFT HAND SWITCHING MODE BUTTON
    if (TEENSY_HAND == Hand::Left) {
      attachInterrupt(
        digitalPinToInterrupt(BUTTON_MODE),
        modeButtonISR,
        FALLING
      );
    }


    // final confirmation message to python - this cues it to play out 
    // final message on audio hat and then clear its input buffer or whatever needs to be done
    Serial.write(static_cast<uint8_t>(VoiceCommands::FLUSH));


    // MIDDLE FINGER VIBRATES ONCE SETUP IS DONE
    buzzMotor(2);
}


void loop() {

  volatile bool curGFreeplay = gFreeplayMode;

  if (gModeToggleRequested) { // THIS WILL ONLY BE CALLED ON LEFT HAND

    // buzz the left side of wrist entering interrupt  
    buzzMotor(5);

    noInterrupts();                    
    gModeToggleRequested = false;


    // past state == learning mode, want to switch to freeplay
    if(curGFreeplay == 0) {
      Serial.write(static_cast<uint8_t>(6));
      Serial.write(static_cast<uint8_t>(6));
      Serial.write(static_cast<uint8_t>(6));
      Serial.write(static_cast<uint8_t>(6));
    }
    if(curGFreeplay == 1) {
      Serial.write(static_cast<uint8_t>(5));
      Serial.write(static_cast<uint8_t>(5));
      Serial.write(static_cast<uint8_t>(5));
      Serial.write(static_cast<uint8_t>(5));
    }

    // PYTHON SIDE: WILL CHANGE MODE

        // FOR BOTH GLOVES, BEFORE RUNNING THIS SHIT, WE WANT TO CHECK
      // IN EACH LOOP IF THE MODE IS BEING CHANGED
      // TODO FIX THIS FOR RIGHT GLOVE SOMEHOW
    while(true){
      modeRes = Serial.read();
      if(modeRes == static_cast<uint8_t>(6)) {
        gFreeplayMode = 1; // we switch into freeplay!
      } else if (modeRes == static_cast<uint8_t>(5)) {
        gFreeplayMode = 0; // we switch into learning!
      }
    }
    interrupts();
  }

  // buzz right side of write 
  buzzMotor(6);

      // If we JUST switched into LEARNING MODE, run song selection
   if (curGFreeplay == 1 && gFreeplayMode == 0) {

        if (TEENSY_HAND == Hand::Left) {

            Serial.write(static_cast<uint8_t>(VoiceCommands::SELECT_SONG));
            delay(7500);

            int songPressCount = buttonPressCount(6000, BUTTON_SONG);
            Serial.write(static_cast<uint8_t>(songPressCount));
        }
    }

    // now, depending on the mode, call the proper logic.
    // if in freeplay mode, call finger press checking function because thats all we need 
    if(gFreeplayMode) {
        checkFingerPress();
    }
    else if(!gFreeplayMode) {
        guideFingerPress();
    }
}

