// Partial Test Code to loop through some of the haptics effects with multiplexer
// Adaptation of the test code Ajith ran! - Sky 

#include <Arduino.h>
#include <Wire.h>
#include "Adafruit_DRV2605.h"


// MUX adr
#define TCAADDR 0x77 //ALL HIGH ON OUR PCB! 0x70 for the breakout module -Rahul

// assigning select lines
#define MOTOR_1 0 // setting a value of -1 or 7 causes to fail for some reason
// Which test points correspond to which MOTOR value. Refer to schematic. Enum should be implement to make this less confusing
//T1 = SDA0 = MUX0 = OUT0
//T2 = SDA1 = MUX1 = OUT1
//T3 = SDA2 = MUX2 = OUT2
//T4 = SDA3 = MUX4 = OUT3
//T5 = SDA4 = MUX5 = OUT4
//T6 = SDA5 = MUX6 = OUT5
//T7 = SDA6 = MUX3 = OUT6
#define RESET 12

Adafruit_DRV2605 drv;

// tca select line unction
void tcaSelect(uint8_t i) {
  if (i > 7) return;  
  Wire1.beginTransmission(TCAADDR);
  Serial.printf("%d\n", (0x1 << i));
  Wire1.write(0x1 << i);
  Wire1.endTransmission();
  Serial.printf("TCA Channel %d selected\n", i);
}

void readControlRegister() {
  Wire1.beginTransmission(TCAADDR);
  unsigned int result = Wire1.read();
  Serial.printf("Control Register ", result);
  Wire1.endTransmission();
}

void setup() { 
  Serial.begin(9600);
  Serial.println("Starting...");
  Serial.println("MUX RESET");
  pinMode(RESET, OUTPUT);
  digitalWrite(RESET, LOW);
  delay(2500);
  digitalWrite(RESET, HIGH);
  Serial.println("MUX out of RESET");
  Wire1.begin();
  delay(100);

  Serial.println("Haptics Test with MUX");

  // make sure both haptics are working
  tcaSelect(MOTOR_1);
  if (!drv.begin(&Wire1)) {
    Serial.println("Failed channel 0");
    while (1) { delay(100); }
  }
  delay(10);
  
  readControlRegister();
  delay(1000);

  drv.setMode(DRV2605_MODE_INTTRIG);

  Serial.println("Setup complete.");
}

void loop() {

  for (uint8_t j = 0; j < 8; j++) {
    Serial.print("testing channel ");
    Serial.println(j);

    tcaSelect(j);

    if (!drv.begin(&Wire1)) {
        Serial.print("DRV2605 failed on channel ");
        Serial.println(j);
        continue;
    }

    drv.setMode(DRV2605_MODE_INTTRIG);
    drv.setWaveform(0, 47);
    drv.setWaveform(1, 0);   // end of waveform
    drv.go();

    delay(2000);
}

}
