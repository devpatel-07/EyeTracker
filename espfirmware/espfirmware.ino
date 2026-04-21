/**
 * @file espfirmware.ino
 * @brief Serial test firmware — drive BASE servo (PCA9685) from typed commands.
 *
 * Same hardware stack as the manual keyboard sketch: Wire + Adafruit_PWMServoDriver @ 50 Hz.
 * Source of truth in repo: ../espfirmware.cpp (keep in sync or edit one and copy).
 *
 * Serial: 115200, newline-terminated lines.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const int BASE_SERVO_CHANNEL = 0;

const int SERVO_MIN_PULSE   = 600;
const int SERVO_MAX_PULSE   = 2400;
const int SERVO_START_PULSE = 1500;

const int STEP_US_DEFAULT = 50;

int basePulseUs = SERVO_START_PULSE;

void i2cScan() {
  Serial.println("I2C bus scan (expect PCA9685 at 0x40 if ADDR jumpers default):");
  int found = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  device at 0x");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  NO devices — fix SDA/SCL (often GPIO21/22), GND, and PCA9685 VCC.");
  }
  Serial.println();
}

void sweepChannels() {
  Serial.println("Sweeping channels 0..15 — watch which connector moves your servo:");
  for (int c = 0; c < 16; c++) {
    Serial.print(">>> CHANNEL ");
    Serial.println(c);
    pwm.writeMicroseconds(c, 1000);
    delay(450);
    pwm.writeMicroseconds(c, 2000);
    delay(450);
    pwm.writeMicroseconds(c, 1500);
    delay(250);
  }
  Serial.println("Sweep done. Use that channel number in BASE_SERVO_CHANNEL or command w <ch> <us>.");
  Serial.println();
}

void applyBasePulse(int us) {
  us = constrain(us, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
  basePulseUs = us;
  pwm.writeMicroseconds(BASE_SERVO_CHANNEL, basePulseUs);
  Serial.print("BASE ch");
  Serial.print(BASE_SERVO_CHANNEL);
  Serial.print(" -> ");
  Serial.print(basePulseUs);
  Serial.println(" us");
}

void printHelp() {
  Serial.println();
  Serial.println("--- Base servo serial test ---");
  Serial.println("  <number>     pulse width in microseconds (" +
                 String(SERVO_MIN_PULSE) + ".." + String(SERVO_MAX_PULSE) + ")");
  Serial.println("  + [n]       bump +n us (default " + String(STEP_US_DEFAULT) + ")");
  Serial.println("  - [n]       bump -n us (default " + String(STEP_US_DEFAULT) + ")");
  Serial.println("  c           center (" + String(SERVO_START_PULSE) + " us)");
  Serial.println("  m           min pulse");
  Serial.println("  x           max pulse");
  Serial.println("  i2c         scan I2C bus (see 0x40 for PCA9685)");
  Serial.println("  sweep       wiggle each channel 0-15 in order");
  Serial.println("  w <ch> <us> test one channel, e.g. w 3 1500");
  Serial.println("  ?           this help");
  Serial.println("Current: " + String(basePulseUs) + " us");
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("Base servo test — initializing PCA9685...");

  pwm.begin();
  pwm.setPWMFreq(50);

  for (int i = 0; i < 16; i++) {
    pwm.writeMicroseconds(i, SERVO_START_PULSE);
    delay(20);
  }

  applyBasePulse(SERVO_START_PULSE);
  i2cScan();
  printHelp();
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;
  }

  if (line.equalsIgnoreCase("i2c")) {
    i2cScan();
    return;
  }

  if (line.equalsIgnoreCase("sweep")) {
    sweepChannels();
    return;
  }

  if (line.length() > 2 && (line.charAt(0) == 'w' || line.charAt(0) == 'W') && line.charAt(1) == ' ') {
    int sp = line.indexOf(' ', 2);
    if (sp > 2) {
      int ch = line.substring(2, sp).toInt();
      long pus = line.substring(sp + 1).toInt();
      if (ch >= 0 && ch <= 15 && pus >= SERVO_MIN_PULSE && pus <= SERVO_MAX_PULSE) {
        pwm.writeMicroseconds(ch, (int)pus);
        Serial.print("OK ch ");
        Serial.print(ch);
        Serial.print(" -> ");
        Serial.print(pus);
        Serial.println(" us");
      } else {
        Serial.println("Usage: w <0-15> <us in range>");
      }
    } else {
      Serial.println("Usage: w <ch> <us>  example: w 7 1500");
    }
    return;
  }

  char c0 = line.charAt(0);

  if (c0 == '?' || c0 == 'h' || c0 == 'H') {
    printHelp();
    return;
  }

  if (c0 == 'c' || c0 == 'C') {
    applyBasePulse(SERVO_START_PULSE);
    return;
  }

  if (c0 == 'm' || c0 == 'M') {
    applyBasePulse(SERVO_MIN_PULSE);
    return;
  }

  if (c0 == 'x' || c0 == 'X') {
    applyBasePulse(SERVO_MAX_PULSE);
    return;
  }

  if (c0 == '+') {
    int step = STEP_US_DEFAULT;
    if (line.length() > 1) {
      step = line.substring(1).toInt();
      if (step <= 0) {
        step = STEP_US_DEFAULT;
      }
    }
    applyBasePulse(basePulseUs + step);
    return;
  }

  if (c0 == '-') {
    int step = STEP_US_DEFAULT;
    if (line.length() > 1) {
      step = line.substring(1).toInt();
      if (step <= 0) {
        step = STEP_US_DEFAULT;
      }
    }
    applyBasePulse(basePulseUs - step);
    return;
  }

  if (c0 >= '0' && c0 <= '9') {
    long v = line.toInt();
    if (v < SERVO_MIN_PULSE || v > SERVO_MAX_PULSE) {
      Serial.print("Out of range (");
      Serial.print(SERVO_MIN_PULSE);
      Serial.print("..");
      Serial.print(SERVO_MAX_PULSE);
      Serial.println("). Ignored.");
      return;
    }
    applyBasePulse((int)v);
    return;
  }

  Serial.println("Unknown command. Type ? for help.");
}
