#include <Arduino.h>
// Teensy Solder Joint Tester
// Cycles through all digital pins, setting each HIGH one at a time
// Connect your multimeter GND lead to Teensy GND, and probe each pin
int TEST_PIN = 0; 

void setup() {
    Serial.begin(115200);
    while (! Serial);
    pinMode(TEST_PIN, OUTPUT);
}

void loop() {
    Serial.println(TEST_PIN);
    Serial.println("Setting pin High "); 
    digitalWrite(TEST_PIN, HIGH);

    // Leave it HIGH for 2 seconds so you can measure with multimeter
    delay(10000);

    // Turn it back LOW before moving on
    digitalWrite(TEST_PIN, LOW);
    Serial.println("Setting pin Low "); 

    delay(10000);
    if (TEST_PIN > 23) TEST_PIN = 0; // Cycle back to pin 0 after pin 13
    TEST_PIN++;
}
