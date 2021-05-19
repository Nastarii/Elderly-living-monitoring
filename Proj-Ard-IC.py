import rrdtool
import serial
from os import path
from threading import Thread
from time import sleep
from math import sqrt
import cv2

#sudo su
#rfcomm bind 0 98:D3:71:F6:23:02
SerialComunication = serial.Serial('/dev/rfcomm0',9600)

#Criando os bancos de dados caso ja não tenham sido criados
if not path.isfile('temp-gas.rrd'):
    rrdtool.create('temp-gas.rrd',
                   '--start','now',
                   '--step','1',
                   'DS:Temperatura:GAUGE:600:-40:85',
                   'DS:Fumaca:GAUGE:600:0:400',
                   'RRA:LAST:0.5:1s:12s',
                   'RRA:AVERAGE:0.5:1m:1h',
                   'RRA:AVERAGE:0.5:5m:1d')

if not path.isfile('acelerometro.rrd'):
    rrdtool.create('acelerometro.rrd',
                   '--start','now',
                   '--step','2',
                   'DS:AccX:GAUGE:600:-3:3',
                   'DS:AccY:GAUGE:600:-3:3',
                   'DS:AccZ:GAUGE:600:-3:3',
                   'DS:Acc:GAUGE:600:-3:3',
                   'RRA:LAST:0.5:2s:1800s',
                   'RRA:AVERAGE:0.5:1h:1d')

if not path.isfile('giroscopio.rrd'):
    rrdtool.create('giroscopio.rrd',
                   '--start','now',
                   '--step','2',
                   'DS:GirX:GAUGE:600:-180:180',
                   'DS:GirY:GAUGE:600:-180:180',
                   'DS:GirZ:GAUGE:600:-180:180',
                   'DS:Gir:GAUGE:600:-180:180',
                   'RRA:LAST:0.5:2s:1800s',
                   'RRA:AVERAGE:0.5:1h:1d')

# As propriedades dos gráficos
def prop(nome,inicio,periodo,fim,titulo,eixoY,altura,largura,minimo,maximo):
    return [nome,'--start',inicio,'--step',periodo,'--end',fim,'--height='+altura,
    '--width='+largura,'--title',titulo,'--vertical-label',eixoY,'-N','-l','0','-i','--rigid',
    '--slope-mode','--color', 'BACK#FFFFFF','-l',minimo,'-u',maximo]

def dados_dos_sensores():
    while True:

        #Inserindo os valores do monitor serial em um banco de dados RRD
        serialPrint = str(SerialComunication.readline())
        val = serialPrint.split(':')

        acc = sqrt(float(val[6])**2 + float(val[8])**2 + float(val[10])**2)
        gir = sqrt(float(val[12])**2 + float(val[14])**2 + float(val[16])**2)
        cam_in[0] = acc
        cam_in[1] = gir

        rrdtool.update('temp-gas.rrd','N:' + val[2] + ':' + val[4])
        rrdtool.update('acelerometro.rrd','N:' + val[6] + ':' + val[8] + ':' + val[10] + ':' + str(acc))
        rrdtool.update('giroscopio.rrd','N:' + val[12] + ':' + val[14] + ':' + val[16] + ':' + str(gir))

        rrdtool.graph(prop('ModeloInc.png','now - 12s','1','now','Modelo - Detecção de incêndio', \
                  'Temperatura[°C]','220','450','0','65'),
                  '--x-grid', 'SECOND:2:SECOND:2:SECOND:2:0:%X',
                  'DEF:t=temp-gas.rrd:Temperatura:LAST','DEF:g=temp-gas.rrd:Fumaca:LAST',
                  'CDEF:lt=t,1.95,+','SHIFT:lt:2',
                  'CDEF:compT=lt,t,LT,t,UNKN,IF',
                  'CDEF:at=COUNT,2,%,0,EQ,t,UNKN,IF',
                  'CDEF:trisc=COUNT,2,%,0,EQ,compT,UNKN,IF',
                  'CDEF:contT=lt,t,LT,10,UNKN,IF',
                  'VDEF:total=contT,TOTAL',
                  'CDEF:temp-gas=contT,POP,total,100,EQ,1,UNKN,IF',
                  'VDEF:dinc=temp-gas,MAXIMUM',
                  'CDEF:fum=g,300,GT,g,UNKN,IF','VDEF:dfum=fum,MAXIMUM',
                  'VDEF:valG=g,MAXIMUM',
                  'GPRINT:valG:Nível de fumaça\:%2.1lf \l',
                  'COMMENT:Valor de Risco > 300 \\n',
                  'COMMENT:Detecção de um possível incêndio\:',
                  'GPRINT:total:%2.1lf %S%%\l',
                  'GPRINT:dinc:Temperatura de um possível incêndio\: %c \\n:strftime',
                  'GPRINT:dfum:Presença de fumaça detectada\: %c \\n:strftime',
                  'AREA:at#0000FF95:Temperatura',
                  'AREA:trisc#FFFF50:Temperatura de risco detectada',
                  'LINE2:lt#FF000095:Temperatura de risco')

        rrdtool.graph(prop('accMpu.png','now  - 300s','2s','now','','Accelerometro[g]','220','500','0','4'),
                  'DEF:acc=acelerometro.rrd:Acc:LAST',
                  'CDEF:norAcc=acc,1.5,LT,acc,UNKN,IF','CDEF:altAcc=acc,1.5,GT,acc,UNKN,IF',
                  'LINE2:norAcc#00BB0095:Dados do acelerometro','LINE2:altAcc#FFFF0095:Possível queda',
                  'HRULE:1.5#FF000095:Limiar de risco do accelerômetro')

        rrdtool.graph(prop('girMpu.png','now  - 300s','2s','now','','Giroscópio[°]','220','500','-10','180'),
                  'DEF:gir=giroscopio.rrd:Gir:LAST',
                  'CDEF:norGir=gir,90,LT,gir,UNKN,IF','CDEF:altGir=gir,90,GT,gir,UNKN,IF',
                  'LINE2:norGir#0000FF95:Dados do Giroscópio','LINE2:altGir#FFFF0095:Possível queda',
                  'HRULE:90#FF000095:Limiar de risco do giroscópio')

        sleep(2)

def grafico_h():
    while True:
        rrdtool.graph(prop('temph.png','now - 1h','1m','now','Monitoramento de temperatura na ultima hora', \
                 'Temperatura[°C]','220','800','0','60'),'DEF:t=temp-gas.rrd:Temperatura:AVERAGE',
                 'AREA:t#0000DD85:Temperatura')
        
        sleep(60)

def grafico_d():
    while True:
        rrdtool.graph(prop('tempDiaria.png','now - 1d','5m','now','Monitoramento Diário de temperatura', \
                 'Temperatura[°C]','220','800','0','60'),'DEF:t=temp-gas.rrd:Temperatura:AVERAGE',
                 'AREA:t#0000DD85:Temperatura')
    
        sleep(300)

def camera():
    
    cap = cv2.VideoCapture(0)

    while True:
        if cam_in[0] >= 1.5 or cam_in[1] >= 90:
            ret, frame = cap.read()
            cv2.imshow('Vídeo', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break
        else:
            c = cv2.waitKey(1)
            if c == 27:
                break
        
    cap.release()
    cv2.destroyAllWindows()

cam_in = [0,0]
th0 = Thread(target = dados_dos_sensores)
th0.start()
th1 = Thread(target = grafico_h)
th1.start()
th2 = Thread(target = grafico_d)
th2.start()
#th3 = Thread(target = camera)
#th3.start()


'''
    rrdtool.graph(prop('accMpu.png','now  - 300s','2s','now','Dados do acelerometro','','220','500','-2','2'),
                 'DEF:x=acelerometro.rrd:AccX:LAST',
                 'DEF:y=acelerometro.rrd:AccY:LAST',
                 'DEF:z=acelerometro.rrd:AccZ:LAST',
                 'DEF:acc=acelerometro.rrd:Acc:LAST',
                 'LINE2:x#FF0000:Eixo X',
                 'LINE2:y#00FF00:Eixo Y',
                 'LINE2:z#0000FF:Eixo Z',
                 'LINE3:acc#000000:')
                 
    rrdtool.graph(prop('girMpu.png','now  - 300s','2s','now','Dados do giroscópio','Ângulo[°]', \
                 '220','500','-360','360'),
                 'DEF:x=giroscopio.rrd:GirX:LAST',
                 'DEF:y=giroscopio.rrd:GirY:LAST',
                 'DEF:z=giroscopio.rrd:GirZ:LAST',
                 'LINE2:x#FF0000:Eixo X',
                 'LINE2:y#00FF00:Eixo Y',
                 'LINE2:z#0000FF:Eixo Z')'''

