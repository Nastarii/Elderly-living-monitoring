import csv
from time import sleep
from random import uniform

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

        diret = '/home/lucas/Documentos/Testes/'
        with open(diret + 'Dataframe.csv', 'w') as output:
            writer = csv.writer(output,delimiter=";")
            
            for value in self.dic.values():
                writer.writerow([value])
'''     
cont = 0
df = Dataframe()

while True:

    dados = [round(uniform(0,15),2) for _ in range(11)]
    df.put(dados,cont)
    cont = cont + 1
    sleep(1)
    if cont == 5:
        break


a = [i for i in range(11)]
b = [j for j in range(18)]
df.insert2(a)
df.insert2(b)


df.get()

def insert(self,value):
        position = min(self.dic.keys())
        self.dic[int(position) + self.cont].append(value)
        self.cont = self.cont + 1
        if self.cont == len(self.dic):
            self.cont = 0
        return self.dic'''