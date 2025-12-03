// #include <Arduino.h>
// #include <TCA9548.h>
// #include <consts.h>
// #include <Adafruit_DRV2605.h>

// TCA9548 MP(0x70);
// Adafruit_DRV2605 drv;
// const float R1 = 740.0;
// const int veloPin = 8;


// void setup() 
// {
//   Serial.begin(115200);
  
  
//   // multiplex code
//   /*
//   // Begin mux
//   if (MP.begin() == false)
//   {
//     Serial.println("COULD NOT CONNECT TO MUX");
//   } 
//   int driverChannels = MP.find(0x50);
//   MP.selectChannel(0);
//   */
//   // test commit 2
//   // Look for a device with this address on the mux
//   // TODO: replace with address of motor controller
//   // TODO: Make a motor driver class
  
  
//   // Driver code
  
//   /* begin using default I2C ports,
//   can specify different ones
//   using &Wire1 
//   */
//   if (! drv.begin())
//   { 
//     Serial.println("Could not find DRV2605");
//   }
  
//   drv.selectLibrary(1);
//   drv.setMode(DRV2605_MODE_INTTRIG);

//   // setup analog in
// }
// uint8_t effect = 1;

// void loop() {
// //  Serial.print("Effect #"); Serial.println(effect);

//   effect = 119;
  
//   if (effect == 1) {
//     Serial.println(F("1 − Strong Click - 100%"));
//   }
//   if (effect == 118) {
//     Serial.println(F("118 − Long buzz for programmatic stopping – 100%"));
//   }
//   if (effect == 119) {
//     Serial.println(F("119 − Smooth Hum 1 (No kick or brake pulse) – 50%"));
//   }
//   if (effect == 120) {
//     Serial.println(F("120 − Smooth Hum 2 (No kick or brake pulse) – 40%"));
//   }
//   if (effect == 121) {
//     Serial.println(F("121 − Smooth Hum 3 (No kick or brake pulse) – 30%"));
//   }
//   if (effect == 122) {
//     Serial.println(F("122 − Smooth Hum 4 (No kick or brake pulse) – 20%"));
//   }
//   if (effect == 123) {
//     Serial.println(F("123 − Smooth Hum 5 (No kick or brake pulse) – 10%"));
//   }

//   // set the effect to play
//   drv.setWaveform(0, effect);  // play effect 
//   drv.setWaveform(1, 0);       // end waveform

//   // play the effect!
//   drv.go();
//   int veloValue = analogRead(veloPin);
//   float veloVoltage = map((float)veloValue, 0, 1023, 0, 3.3);
//   // velostat resistance
//   float veloR = R1*(3.3/veloVoltage - 1);
//   Serial.printf("Velostat resistance is %f\n", veloR);  
//   // wait a bit
//   delay(100);

//   // effect++;
//   if (effect > 117) effect = 1;
// }