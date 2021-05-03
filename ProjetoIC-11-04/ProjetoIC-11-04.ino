#define MQ2pin (0)
#define DHTPIN 2
#define DHTTYPE DHT11

#include "Wire.h"
#include "DHT.h"
#include <MPU6050_light.h>

DHT dht(DHTPIN, DHTTYPE);
MPU6050 mpu(Wire);

long timer = 0;
float sensorValue;  //variable to store sensor value

void setup() {
  Serial.begin(9600);
  dht.begin();
  Wire.begin();
  
  byte status = mpu.begin();
  while(status!=0){ }
  delay(1000);
  mpu.calcOffsets(true,true);
}

void loop() {
  mpu.update();

  if(millis() - timer > 1000){

    float h = dht.readHumidity();
    float t = dht.readTemperature();
    
    sensorValue = analogRead(MQ2pin); // read analog input pin 0
    Serial.print(F(":temp:"));Serial.print(mpu.getTemp());
    Serial.print(":Gas:"); Serial.print(sensorValue);
    Serial.print(F(":accX:"));Serial.print(mpu.getAccX());
    Serial.print(":accY:");Serial.print(mpu.getAccY());
    Serial.print(":accZ:");Serial.print(mpu.getAccZ());
    Serial.print(F(":gyroAngleX:"));Serial.print(mpu.getGyroX());
    Serial.print(":gyroAngleY:");Serial.print(mpu.getGyroY());
    Serial.print(":gyroAngleZ:");Serial.print(mpu.getGyroZ());
    Serial.print(F(":tempDHT11:"));Serial.print(t);
    Serial.print(F(":Humidity:"));Serial.print(h);
    Serial.println(":");
    
    timer = millis();
    delay(2000);
    
  }

}
