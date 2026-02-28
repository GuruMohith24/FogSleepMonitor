#include <Wire.h>

const int MPU_ADDR = 0x68; // I2C address of the MPU-6050
int16_t AcX, AcY, AcZ; // Accelerometer variables
int heartRatePin = A0; // Heart rate sensor analog pin
int pulseValue; // Heart pulse reading

// Variables for timing
unsigned long previousMillis = 0;
const long interval = 100; // 100ms sampling -> 10Hz

void setup() {
  Serial.begin(115200); // High baud rate for fast data transfer
  
  // Initialize I2C communication with MPU6050
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
  
  // Allow sensor to stabilize
  delay(100);
  Serial.println("System Ready. Starting Data Acquisition...");
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Read Accelerometer values (X, Y, Z)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); // Starting with register 0x3B (ACCEL_XOUT_H)
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true); // Request 6 registers
    
    if (Wire.available() >= 6) {
      AcX = Wire.read() << 8 | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)    
      AcY = Wire.read() << 8 | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
      AcZ = Wire.read() << 8 | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
    }

    // Read Heart Rate Sensor value
    pulseValue = analogRead(heartRatePin);

    // Send the readings as a comma-separated format for easy parsing at the Fog Node
    // Format: Timestamp, AcX, AcY, AcZ, PulseValue
    Serial.print(currentMillis);
    Serial.print(",");
    Serial.print(AcX);
    Serial.print(",");
    Serial.print(AcY);
    Serial.print(",");
    Serial.print(AcZ);
    Serial.print(",");
    Serial.println(pulseValue);
  }
}
