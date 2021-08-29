# Elderly Living Monitoring System
This project pretend build an system to detect fall and fires in home enviromment, for help elderly living people. Using a microcontroller (Arduino UNO), with some
sensors attached, the body velocity of the elderly and temperature data will be collected and stored in a database, called RRDTool. These data are sent to a computer,which performs graphical analyses and computer vision techniques, to confirm the results of the sensors.

## Materials List
Arduino           | Arduino UNO

Acelerometer      | MPU-6050

Bluetooth Module  | HC-05

Gas Sensor        | MQ-02

Temperature Sensor| DHT11

Power Supply      | 9V Baterry + Plug P4 Male

Resistors         | 1kΩ and 2kΩ

## Configure HC-05
First, it's necessary to configure the sensor HC-05 using these connections and the file hc05_config.ino:

Vcc --> 5V

GND --> GND

EN --> 3.3V (Enable AT commands)

TX  --> TX board pin

RX --> 2kΩ + Rx Pin/ 1kΩ + GND (Voltage divider)

With the file opened and uploaded, open the serial monitor and use the AT commands:

Type AT using the baud rate 9600,38400,115200. Wait until you get the answer 'OK', use this baud rate to type the rest of AT commands

AT+NAME=device_name(Choose the bluetooth name)

AT+PSWD=1234(Choose the password)

AT+ROLE=0 (Configure to slave mode, i.e, others devices can connect in the HC-05)

AT+ADDR? (Get bluetooth adress and copy it)

All commands are avaiable at https://s3-sa-east-1.amazonaws.com/robocore-lojavirtual/709/HC-05_ATCommandSet.pdf

Remove the EN wire to use the module.

## Set up Arduino

The arduino code is avaiable in the file main.ino and the connection diagram is shown in the image below.

![Pins_connections](https://user-images.githubusercontent.com/76565870/131266401-bd6474ac-76ac-4e51-9163-35cc515d7142.png)

