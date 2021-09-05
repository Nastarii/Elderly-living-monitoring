#define MQ2pin (0)
#define DHTPIN 2
#define DHTTYPE DHT11

#include "Wire.h"
#include "DHT.h"
#include <MPU6050_light.h>

DHT dht(DHTPIN, DHTTYPE);
MPU6050 mpu(Wire);

long timer = 0;
float gasSensor;

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
    
    gasSensor = analogRead(MQ2pin);
    Serial.print(":");Serial.print(mpu.getTemp());
    Serial.print(":"); Serial.print(gasSensor);
    Serial.print(":");Serial.print(mpu.getAccX());
    Serial.print(":");Serial.print(mpu.getAccY());
    Serial.print(":");Serial.print(mpu.getAccZ());
    Serial.print(":");Serial.print(mpu.getGyroX());
    Serial.print(":");Serial.print(mpu.getGyroY());
    Serial.print(":");Serial.print(mpu.getGyroZ());
    Serial.print(":");Serial.print(t);
    Serial.print(":");Serial.print(h);
    Serial.println(":");
    
    timer = millis();
    delay(40);
    
  }

}
