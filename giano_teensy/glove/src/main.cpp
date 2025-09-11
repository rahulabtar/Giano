#include <Arduino.h>
#include <TCA9548.h>
#include <consts.h>
#include <Adafruit_DRV2605.h>

TCA9548 MP(0x70);
Adafruit_DRV2605 drv;

void setup() 
{
  Serial.begin(115200);
  
  // Begin mux
  if (MP.begin() == false)
  {
    Serial.println("COULD NOT CONNECT TO MUX");
  }
  // Look for a device with this address on the mux
  // TODO: replace with address of motor controller
  // TODO: Make a motor driver class
  int driverChannels = MP.find(0x50);
  MP.selectChannel(0);
  
  // begin using default I2C ports,
  // can specify different ones
  // using &Wire1 
  drv.begin();
}

void loop() 
{

}