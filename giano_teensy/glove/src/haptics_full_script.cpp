// Chat we won't know if this works until we have all the gloves built and connected to pi
// # pray # vibes, - Sky

#include <Wire.h>
#include <Adafruit_DRV2605.h>

#define TCAADDR 0x70
Adafruit_DRV2605 drv;

// setting select lines for each of the haptic motors

const int MOTOR_THUMB  = 0;
const int MOTOR_INDEX  = 1;
const int MOTOR_MIDDLE = 2;
const int MOTOR_RING   = 3;
const int MOTOR_PINKY  = 4;
const int MOTOR_LEFT   = 5;
const int MOTOR_RIGHT  = 6;


// used to check that all motors are reachable - for future debugging
bool motorPresent[7];

// setting flags for octave motors as we want continuous vibration until state change
bool leftVibrating = false;
bool rightVibrating = false;


// tca select line basic function for each motor driver
void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  // buffer time for setup - we can alter after but would prefer to have some sort of buffer
  delay(100);

  // initialize each motor channel
  for (int i = 0; i < 7; i++) {
    tcaselect(i);
    if (drv.begin()) {
      motorPresent[i] = true;
      drv.selectLibrary(1);
      drv.useERM();
      drv.setMode(0x00); 
      Serial.print("Motor "); Serial.print(i); Serial.println(" found.");
    } else {
      motorPresent[i] = false;
      Serial.print("Motor "); Serial.print(i); Serial.println(" not found!");
    }
    delay(10);
  }
  Serial.println("MOTORS READY");
}

void loop() {
  if (Serial.available()) {
    int cmd = Serial.read();  // read one byte (4 bits aka our concerned number) from the pi
    stopContinuous();         // stop any side vibration first
    handleCommand(cmd & 0x0F);
  }
}

/**
 * NOTE FOR SKY - ADD ON TWO MORE CODES TO STOP CONTINUOUS OCTAVE VIBRATION 
 * AKA LIKE VALIDATION - STOP VIBRATION WHEN OVER RIGHT NOTE
 */
void handleCommand(uint8_t code) {
  switch (code) {
    case 0b0000: vibrateShort(MOTOR_THUMB); break;
    case 0b0001: vibrateShort(MOTOR_INDEX); break;
    case 0b0010: vibrateShort(MOTOR_MIDDLE); break;
    case 0b0011: vibrateShort(MOTOR_RING); break;
    case 0b0100: vibrateShort(MOTOR_PINKY); break;
    case 0b0101: startContinuous(MOTOR_RIGHT); rightVibrating = true; break;
    case 0b0110: startContinuous(MOTOR_LEFT); leftVibrating = true; break;
    case 0b0111: vibrateTriple(MOTOR_THUMB); break;
    case 0b1000: vibrateTriple(MOTOR_INDEX); break;
    case 0b1001: vibrateTriple(MOTOR_MIDDLE); break;
    case 0b1010: vibrateTriple(MOTOR_RING); break;
    case 0b1011: vibrateTriple(MOTOR_PINKY); break;
    case 0b1100: vibrateAll(); break;
    default: Serial.println("Unknown code");
  }
}

// HELPERS FOR HAPTIC

void vibrateShort(int motor, int durationMs = 200) {
  if (!motorPresent[motor]) return;
  tcaselect(motor);
  drv.setWaveform(0, 47); 
  drv.setWaveform(1, 0);
  drv.setMode(0x00);
  drv.go();
  delay(durationMs);
  drv.stop();
}

void vibrateTriple(int motor) {
  if (!motorPresent[motor]) return;
  tcaselect(motor);
  for (int i = 0; i < 3; i++) {
    drv.setWaveform(0, 47);
    drv.setWaveform(1, 0);
    drv.setMode(0x00);
    drv.go();
    delay(150);
  }
  drv.stop();
}

void startContinuous(int motor, uint8_t intensity = 120) {
  if (!motorPresent[motor]) return;
  tcaselect(motor);
  drv.setMode(0x05);
  drv.setRealtimeValue(intensity);
}

void stopContinuous() {
  // stop both left & right side if active
  if (leftVibrating && motorPresent[MOTOR_LEFT]) {
    tcaselect(MOTOR_LEFT);
    drv.setRealtimeValue(0);
    drv.setMode(0x00);
    leftVibrating = false;
  }
  if (rightVibrating && motorPresent[MOTOR_RIGHT]) {
    tcaselect(MOTOR_RIGHT);
    drv.setRealtimeValue(0);
    drv.setMode(0x00);
    rightVibrating = false;
  }
}

void vibrateAll() {
  // all 5 fingers
  for (int i = 0; i < 5; i++) {
    if (motorPresent[i]) {
      tcaselect(i);
      drv.setWaveform(0, 47);
      drv.setWaveform(1, 0);
      drv.setMode(0x00);
      drv.go();
    }
  }
  delay(500);
  for (int i = 0; i < 5; i++) {
    if (motorPresent[i]) {
      tcaselect(i);
      drv.stop();
    }
  }
}
