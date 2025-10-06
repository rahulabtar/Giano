// Partial Test Code to loop through some of the haptics effects with multiplexer
// Adaptation of the test code Ajith ran! - Sky 

#include <Wire.h>
#include "Adafruit_DRV2605.h"


// MUX adr
#define TCAADDR 0x70

// assigning select lines
#define MOTOR_1 0   
#define MOTOR_2 1   

Adafruit_DRV2605 drv;

// tca select line unction
void tcaSelect(uint8_t i) {
  if (i > 7) return;  
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  delay(100);

  Serial.println("Haptics Test with MUX");

  // make sure both haptics are working
  tcaSelect(MOTOR_1);
  if (!drv.begin()) {
    Serial.println("Failed channel 0");
    while (1);
  }
  drv.setMode(DRV2605_MODE_INTTRIG); //sets internal trigger like old code

  tcaSelect(MOTOR_2);
  if (!drv.begin()) {
    Serial.println("Failed channel 1");
    while (1);
  }
  drv.setMode(DRV2605_MODE_INTTRIG);

  Serial.println("Setup complete.");
}

void loop() {
  Serial.println("testing channel 0");
  for (uint8_t i = 1; i <= 5; i++) {
    tcaSelect(MOTOR_1);
    Serial.print("Playing Effect");
    Serial.println(i);
    drv.setWaveform(0, i);
    drv.setWaveform(1, 0);
    drv.go();
    delay(600);
  }

  Serial.println("testing channel 1");
  for (uint8_t i = 1; i <= 5; i++) {
    tcaSelect(MOTOR_2);
    Serial.print("Playing effect ");
    Serial.println(i);
    drv.setWaveform(0, i);
    drv.setWaveform(1, 0);
    drv.go();
    delay(600);
  }

  Serial.println("testing multiline select");
  for (uint8_t i = 1; i <= 5; i++) {
    Serial.print("Playing effect ");
    Serial.println(i);

    tcaSelect(MOTOR_1);
    drv.setWaveform(0, i);
    drv.setWaveform(1, 0);
    drv.go();

    tcaSelect(MOTOR_2);
    drv.setWaveform(0, i);
    drv.setWaveform(1, 0);
    drv.go();

    delay(800);
  }

  Serial.println("successful!");
  delay(5000);
}

