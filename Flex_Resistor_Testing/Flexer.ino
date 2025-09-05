// Arduino Uno: Read 5 analog pins and print to Serial
void setup() {
  Serial.begin(9600);  // must match Python baudrate
}

void loop() {
  int v0 = analogRead(A0);
  int v1 = analogRead(A1);
  int v2 = analogRead(A2);
  int v3 = analogRead(A3);
  int v4 = analogRead(A4);

  // Map 10-bit ADC (0–1023) down to 0–255
  v0 = map(v0, 0, 1023, 0, 255);
  v1 = map(v1, 0, 1023, 0, 255);
  v2 = map(v2, 0, 1023, 0, 255);
  v3 = map(v3, 0, 1023, 0, 255);
  v4 = map(v4, 0, 1023, 0, 255);

  // Print comma-separated values, one line
  Serial.print(v0);
  Serial.print(",");
  Serial.print(v1);
  Serial.print(",");
  Serial.print(v2);
  Serial.print(",");
  Serial.print(v3);
  Serial.print(",");
  Serial.println(v4);  // println ends with newline

  delay(50); // Adjust rate (50ms → ~20 samples/sec)
}
