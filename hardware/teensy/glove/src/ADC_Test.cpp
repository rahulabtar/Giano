#include <Arduino.h>
#include <TCA9548.h>
#include <consts.h>
#include <Adafruit_DRV2605.h>
#include <SPI.h>

#define PIN_CS   10 //Pin for Chip Select only specified, else are hard coded due to teensys hardware SPI
#define VREF 5
#define WAIT 100
#define CHAN_8_PIN A0 //Analog pin for channel 90
#define CHAN_9_PIN A1 //Analog pin for channel 10

uint16_t vals[8];
uint8_t iter = 0;

void setup() {
  Serial.begin(115200);
  SPI.begin(); 
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
}

uint16_t readMCP3208(uint8_t channel) {
    if (channel > 7) return 0;

    uint8_t data[3] = {0b00000110 | ((channel & 0x04) >> 2), (channel & 0x03) << 6, 0x00};
    
    digitalWrite(PIN_CS, LOW);
    SPI.transfer(data, 3);
    digitalWrite(PIN_CS, HIGH);

    uint16_t result = ((data[1] & 0x0F) << 8) | data[2];
    return result;
}

uint16_t readResistiveSensor(uint8_t channel){
  if (channel >= 0 && channel < 8) return readMCP3208(channel);
  else if (channel == 8)  return digitalRead(CHAN_8_PIN); 
  else if (channel == 9)  return digitalRead(CHAN_9_PIN); 
  return 0; // invalid channel selected 
}


void loop() {
  if (iter > 7) iter = 0;
  vals[iter] = readMCP3208(iter);  // read CH0
  float voltage = (vals[iter] * VREF) / 4096.0; // 12-bit resolution
  Serial.printf("CH%4d: %4d (%.3f V)\n", iter, vals[iter], voltage);
  delay(WAIT);
  //iter++;
}
