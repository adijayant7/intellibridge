// IntelliBridge danger-level demo for ESP32 (works in the Wokwi simulator).
// Files in the project: intellibridge_demo.ino (this file) + intellibridge_danger.h
// LEDs: green GPIO 25, yellow GPIO 26, red GPIO 27 (each through a resistor to GND).
// Serial monitor: type 8 numbers and press Enter:
//   vibration_mg  freq_hz  tilt_deg  crack_mm  water_cm  strain_ue  temp_c  humidity_pct
// example:  3 4 14 0 20 0 20 65
#include "intellibridge_danger.h"
const int LED_G = 25, LED_Y = 26, LED_R = 27;

void report(const char *name, float v, float f, float t, float c, float w, float s, float T) {
  int cls; int lv = ib_danger(v, f, t, c, w, s, T, &cls);
  const char *txt[] = {"SAFE", "WARNING", "DANGER"};
  Serial.printf("%-28s danger level %3d / 100  ->  %s\n", name, lv, txt[cls]);
  digitalWrite(LED_G, cls == 0); digitalWrite(LED_Y, cls == 1); digitalWrite(LED_R, cls == 2);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_G, OUTPUT); pinMode(LED_Y, OUTPUT); pinMode(LED_R, OUTPUT);
  Serial.println("\nIntelliBridge on ESP32 (model trained on simulated data)");
  report("Normal day",            3.0, 4.0, 0.05, 0.0, 20, 0,   20); delay(1500);
  report("Heavy truck, hot day",  9.5, 3.9, 0.05, 0.3, 20, 130, 38); delay(1500);
  report("Early damage",          3.1, 3.5, 0.4,  0.5, 15, 90,  22); delay(1500);
  report("Flood",                 3.0, 4.0, 0.05, 0.0, 210, 0,  20); delay(1500);
  report("Severe damage",         4.0, 3.1, 1.4,  1.8, 20, 220, 20); delay(1500);
  Serial.println("\nType 8 numbers: vib freq tilt crack water strain temp humidity");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  float v[8];
  if (sscanf(line.c_str(), "%f %f %f %f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 8)
    report("Your reading", v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
  else Serial.println("Please type 8 numbers separated by spaces.");
}
