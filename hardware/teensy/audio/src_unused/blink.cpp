#include <Arduino.h>

// The built-in LED pin varies by Teensy model.
// On Teensy 4.x it's pin 13.
const int LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);  // Configure LED pin as output
}

void loop() {
  digitalWrite(LED_PIN, HIGH); // Turn LED on
  delay(100);                  // Wait 500 ms
  digitalWrite(LED_PIN, LOW);  // Turn LED off
  delay(100);                  // Wait 500 ms
}
