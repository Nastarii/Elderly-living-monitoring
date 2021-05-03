import serial
import datetime
import sqlite3
from time import sleep

comSerial = serial.Serial('/dev/ttyUSB0',9600)
connec = sqlite3.connect('moniSys.db')


class Sensor(object):
    def __init__(self,name,lenght,*args):
        self.name = name
        self.lenght = lenght
        self.data = {d:[] for d in args}
        self.threshold = {t:[] for t in args} 
        self.time = {'Tempo':[datetime.datetime.now().strftime('%H:%M:%S')]}

    def update(self,source,val):
        if datetime.datetime.now().strftime('%H:%M:%S') not in self.time['Tempo']:
            self.time['Tempo'].append(datetime.datetime.now().strftime('%H:%M:%S'))
        self.data[source].append(val)
        if len(self.time['Tempo']) == self.lenght:
            self.time['Tempo'].pop(0)
            self.data[source].pop(0)

    def set_threshold(self,th,source):
        self.threshold[source] = [x + th for x in self.data[source][:-1]]
            

    def model(self):
        if self.name == 'DHT11':

            inc = 0

            for i in range(len(self.time['Tempo'][-6:]) - 1):
                if self.data['Temperatura'][i + 1] >= self.data['Temperatura'][i] + 2 and self.data['Umidade'][i + 1] <= self.data['Umidade'][i] - 2:
                    inc = inc + 20

    

while True:

    serialPrint = str(SerialComunication.readline())
    valSensor = serialPrint.split(":")

    sleep(2)