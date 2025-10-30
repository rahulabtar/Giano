#include <Arduino.h>
#include <TCA9548.h>
#include <consts.h>
#include <Adafruit_DRV2605.h>
#include <SPI.h>

#define HAND_RIGHT  true
#define PIN_CS   10 //Pin for Chip Select only specified, else are hard coded due to teensys hardware SPI
#define VREF 3.3
#define WAIT 100
#define ADC_CHAN_8_PIN A0 //Analog pin for channel 90
#define ADC_CHAN_9_PIN A1 //Analog pin for channel 10

#if HAND_RIGHT
  typedef enum{
    FINGER_PINKY = 5,
    FINGER_RING = 4,
    FINGER_MIDDLE = 3,
    FINGER_INDEX = 2, 
    FINGER_THUMB = 1, 
    FINGER_INVALID = 6, 
}Finger;
#else
  typedef enum{
    FINGER_PINKY = 1,
    FINGER_RING = 2,
    FINGER_MIDDLE = 3,
    FINGER_INDEX = 4, 
    FINGER_THUMB = 5, 
    FINGER_INVALID = 6, 

}FingerMap;
#endif
//WE SHOULD HAVE FINGER VALS BE THE SAME AS PIANO NOTIATION RATHER THAN ALWAYS CALLING THUMB 1 


#if HAND_RIGHT
  typedef enum{
    PINKY_VEL = 0,
    PINKY_FLEX = 1,
    RING_VEL = 2,
    RING_FLEX = 3,
    MIDDLE_VEL = 4,
    MIDDLE_FLEX = 5,
    INDEX_VEL = 6,
    INDEX_FLEX = 7,
    THUMB_VEL = 8,
    THUMB_FLEX = 9,
    RES_INVALID = 13, 
}FingerResChan;
#else
  typedef enum{
    THUMB_VEL = 0,
    THUMB_FLEX = 1,
    INDEX_VEL = 2,
    INDEX_FLEX = 3,
    MIDDLE_VEL = 4,
    MIDDLE_FLEX = 5,
    RING_VEL = 6,
    RING_FLEX = 7,
    PINKY_VEL = 8,
    PINKY_FLEX = 9,
    RES_INVALID = 13; 
}FingerResChan;
#endif

struct Resistor_Values {
  uint16_t flex_val = 0; 
  uint16_t vel_val = 0; 
  bool press = false; 
}; 

void setup() {
  Serial.begin(115200);
  SPI.begin(); 
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  analogReadResolution(12); // set ADC resolution to 12 bits
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

uint16_t readResistiveSensor(FingerResChan channel){
  if (channel >= 0 && channel < 8) return readMCP3208(channel);
  else if (channel == 8)  return analogRead(ADC_CHAN_8_PIN); 
  else if (channel == 9)  return analogRead(ADC_CHAN_9_PIN); 
  return -1; // invalid channel selected 
}

bool isPress(int vel_val, int flex_val){
  int threshold = 2000; 
  return (vel_val >= threshold); 
}

void getFingerChans(Finger f, FingerResChan &flex_chan, FingerResChan &vel_chan){
  switch (f){
    case FINGER_THUMB: {
      flex_chan = THUMB_FLEX; 
      vel_chan = THUMB_VEL; 
      break; 
    }
    case FINGER_INDEX: {
      flex_chan = INDEX_FLEX;
      vel_chan = INDEX_VEL; 
      break; 
    }
    case FINGER_MIDDLE: {
      flex_chan = MIDDLE_FLEX;
      vel_chan = MIDDLE_VEL;
      break;
    }
    case FINGER_RING: {
      flex_chan = RING_FLEX;
      vel_chan = RING_VEL;
      break;
    }
    case FINGER_PINKY: {
      flex_chan = PINKY_FLEX;
      vel_chan = PINKY_VEL;
      break;
    }
    default: {
      flex_chan = RES_INVALID; 
      vel_chan = RES_INVALID; 
      break; 
    }
  }
}

Finger getFingerVals(unsigned int A){

  //Creates a Resistor Values Struct 
  Resistor_Values result; 
  //Makes Sure Requested Finger, A, is an Actual Finger
  if (A < 0 || A >= 5) return FINGER_INVALID; 
  //Based on the finger, get the corresponding ADC Channels needed to read based on pin out enum FingerResChan
  Finger f = static_cast<Finger>(A);
  FingerResChan flex_chan; 
  FingerResChan vel_chan;  
  if (f == FINGER_INVALID) return f; //checks again incase case goes bad
  getFingerChans(f, flex_chan, vel_chan); //get the channels based on pin out enum 

  //read the chans and get the adc vals
  result.flex_val = readResistiveSensor(flex_chan); 
  result.vel_val = readResistiveSensor(vel_chan);
  //calcs whether we think this finger is pressed or not
  result.press = isPress(result.flex_val, result.vel_val); 

  //if we think the key is pressed return the finger (same as A), else return an invalid finger (6)
  if (result.press) return f; 
  else return FINGER_INVALID;
}

void loop() {
  //ima just die bro 
}
