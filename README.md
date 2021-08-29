# Elderly Living Monitoring System
This project pretend build an system to detect fall and fires in home enviromment, for help elderly living people. Using a microcontroller (Arduino UNO), with some
sensors attached, the body velocity of the elderly and temperature data will be collected and stored in a database, called RRDTool. These data are sent to a computer,which performs
graphical analyses and computer vision techniques, to confirm the results of the sensors.
## Materials List
Arduino           | Arduino UNO

Acelerometer      | MPU-6050

Bluetooth Module  | HC-05

Gas Sensor        | MQ-02

Temperature Sensor| DHT11

Power Supply      | 9V Baterry + Plug P4 Male

Resistors         | 1kΩ and 2kΩ

## Set up Arduino
First, it's necessary to configure the sensor HC-05 using these conexions and the file hc05_config.ino:

Vcc --> 5V

GND --> GND

EN --> 3.3V (Enable AT commands)

TX  --> TX board pin

RX --> 2kΩ + Rx Pin/ 1kΩ + GND (Voltage divider)

With the file opened, open the serial monitor and use the AT commands:

Type AT using the baud rate 9600,38400,115200. If 

AT+NAME=device_name(Choose the bluetooth name)

AT+PSWD=1234(Choose the password)

AT+CMODE (Check if it's configured in slave mode)


The arduino code is avaiable in the file main.ino
Before start 
