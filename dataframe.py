from random import uniform
from os import makedirs
from time import sleep
import pathlib
import csv

class Dataframe():

    def __init__(self) -> None:
        self.dic = {}
        self.cont = 0

    
    def put(self,data,frame):
        self.dic[frame] = data
        if len(self.dic) == 28:
            min_frame = min(self.dic.keys())
            del self.dic[min_frame]
    
    def insert(self,values):
        position = min(self.dic.keys())
        for value in values:
            self.dic[int(position) + self.cont].append(value)
            self.cont = self.cont + 1
            if self.cont == len(values):
                self.cont = 0
        return self.dic

    def get(self):
        
        diret = str(pathlib.Path(__file__).parent.resolve()) +'/Dataframe_cache'
        makedirs(diret)
        
        with open(diret + 'Dataframe.csv', 'w') as output:
            writer = csv.writer(output,delimiter=";")
            
            for value in self.dic.values():
                writer.writerow([value])
