/**
FREEPLAY MODE Firmware
 Basically just continually tests for a button press and then spits out whatever note 
 sends a flag to the python to check CV because a note was pressed.
 Doesnt care what finger, just that SOME finger was detected


 For testing purposes: will assign a random note to each finger just so we know that 
 it works and directly outputs to audio

*/

// BUTTON / MODE RELATED VARIABLES
const int BUTTON_MODE = 2;
const int BUTTON_SONG = 3;



bool freeplayMode = true; // THIS IS JUST FOR TESTING THE INTERRUPT OF THIS VERSION. IN THE REAL FIRMWARE WE WILL JUST WIND UP SWITCHING MODES NOT EXITING


// VELOSTAT VARIABLES

const int NUM_VELOSTAT = 2; // num of fingers eventually
const int VELOSTAT_PINS[NUM_VELOSTAT] = {A0, A1}; // pinouts for velostat
const int THRESHOLD = 45; // threshold for pressed/unpressed
const int ADC_BITS = 12;

// to collect baseline readings 
int baseline[NUM_VELOSTAT];

bool pressed = false;

bool sensorState[NUM_VELOSTAT] = {false, false};

// ADD HAPTICS SHIT HERE FOR LEARNING MODE

void setup() {
  Serial.begin(9600);
  Serial1.begin(9600); // just for testing this - we will wind up using USB -> RASPI -> USB, but this allows us to do communication between the TX and RX teensy to teensy

  delay(3000);


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

    baseline[i] = sum / SAMPLERATE; 

    Serial.print("Baseline for sensor");
    Serial.print(i);
    Serial.print(" = ");
    Serial.println(baseline[i]);
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  if(!freeplayMode) {
    Serial.println("NOT IN FREEPLAY MODE SOMETHING IS WRONG for this test");
  }

  if(digitalRead(BUTTON_MODE) == LOW) {
    Serial.println("SWITCHING MODE");
    if(freeplayMode) {
      freeplayMode = false; // if true set to false
    } else if (!freeplayMode) {
      freeplayMode = true; // if false set it to true
    }
    delay(200); // debounce time for button logic
  }

  checkFingerPress();

}



void checkFingerPress() {
  for (int i = 0; i < NUM_VELOSTAT; i++) {
    int raw = analogRead(VELOSTAT_PINS[i]);
    bool currentlyPressed = raw >= (baseline[i] + THRESHOLD);

    // transition: unpressed -> pressed
    if (currentlyPressed && !sensorState[i]) {
      Serial1.print("Sensor ");
      Serial1.println(i);     // send "note on" to receiver
      Serial.print("Sensor ");
      Serial.println(i);      // debug

      sensorState[i] = true;  // remember it's pressed
    } 
    // transition: pressed -> released
    else if (!currentlyPressed && sensorState[i]) {
      Serial1.print("SensorReleased ");
      Serial1.println(i);     // send "note off" to receiver
      Serial.println("Sensor released"); // debug

      sensorState[i] = false; // update state
    }
    // if pressed and already marked as pressed, do nothing
  }

  delay(50); // optional debounce/stability
}
