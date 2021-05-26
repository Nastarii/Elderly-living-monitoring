from os import path
from threading import Thread
from time import sleep
from math import sqrt
import rrdtool
import serial
import numpy as np
import cv2 as cv
import os

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


#Extração do vídeo no momento da queda
def vidExtract():
    # Inicialização da captura da câmera
    cap = cv.VideoCapture(0)

    # Declaração das variáveis necessárias para o funcionamento do programa
    imgs = []
    cont1 = 0
    cont2 = 0

    # Inicialização da leitura dos dados da câmera
    while True:

        #Lê a captura frame a frame
        _,frame = cap.read()

        #Redimensionamento do frame
        frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)

        # Se os dados dos sensores estão alterados
        if cam_in[0] > 1.5 or cam_in[1] > 90:

            #Guarda 10 frames após detectada a alteração
            if cont2 <= 9:
                cv.imwrite('/home/lucas/Documentos/Python/Camera/frame'+str(cont2 + 10)+'.jpg',frame)
                cont2 = cont2 + 1
            else:
                break
                    

        # Se não há alterações nos sensores
        else:
            # Guarda os 10 últimos frames
            cv.imwrite('/home/lucas/Documentos/Python/Camera/frame'+str(cont1)+'.jpg',frame)
            if cont1 <= 9:
                cont1 = cont1 + 1
            else:
                for k in range (1,11):
                    os.rename('/home/lucas/Documentos/Python/Camera/frame'+str(k)+'.jpg',
                    '/home/lucas/Documentos/Python/Camera/frame'+str(k-1)+'.jpg')
    
    # Cria um vídeo com os frames de antes e depois da queda
    #for filename in glob.glob('/home/lucas/Documentos/Python/Camera/*.jpg'):
    for cont in range(len(os.listdir('/home/lucas/Documentos/Python/Camera'))):
        filename = '/home/lucas/Documentos/Python/Camera/frame'+str(cont)+'.jpg'
        img = cv.imread(filename)
        h ,w, l= img.shape
        tam = (w,h)
        imgs.append(img)

    out = cv.VideoWriter('video-extraido.avi',cv.VideoWriter_fourcc(*'DIVX'),5,tam)
    for x in range(len(imgs)):
        out.write(imgs[x])
    out.release()
    cap.release()
    KNNsubtractor()
    mobilenetSSD()

def KNNsubtractor():
    # Inicialização da captura do video extraido
    cap = cv.VideoCapture('video-extraido.avi')

    # Salva a analise do método KNN
    out = cv.VideoWriter('video-extraido-KNN.avi',cv.VideoWriter_fourcc(*'DIVX'),5,(640,412))

    # Criação do método de subtração de fundo
    subFundo = cv.createBackgroundSubtractorKNN(history=300,dist2Threshold=500)

    el_cor = (np.random.randint(256),0,0)

    while cap.isOpened():

        ret,frame = cap.read()
        
        if ret:
            # Aplicação de técnicas de pré-processamento 
            blurFrame = cv.GaussianBlur(frame,(5,5),0)

            # Aplicação do métodos de subtração de fundo
            masc = subFundo.apply(blurFrame)

            # Detecção de contornos nos objetos identificados 
            contornos,_ = cv.findContours(masc, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # Inicialização de uma lista com as áreas dos objetos identificados
            areas = []

            # Adicionando valores a lista areas
            for contorno in contornos:
                areas.append(cv.contourArea(contorno))


            # Identifica o objeto com maior área e circunscreve o contorno do objeto em uma elipse
            try:
                max_contorno = contornos[areas.index(max(areas))]
                ellipse = cv.fitEllipse(max_contorno)
                (xc,yc),(d1,d2),angulo = ellipse
                cv.ellipse(frame,ellipse,el_cor, 2)
                label = "Angle : {:.2f}".format(angulo)
                y = round(yc - d2/2) + 15 if (yc - d2/2) > 30 else yc - d2/2 + 15
               
                cv.putText(frame, label, (round(xc),y),cv.FONT_HERSHEY_SIMPLEX, 0.5, el_cor, 2)
                try:
                    coef_mov = sqrt((x2c -xc)**2 + (y2c-yc)**2)
                    label2 = "Coef. de mov.: {:.2f}".format(coef_mov)
                    y2 = round(yc - d2/2) + 30 if (yc - d2/2) > 30 else yc - d2/2 + 30
                    cv.putText(frame, label2, (round(xc),y2),cv.FONT_HERSHEY_SIMPLEX, 0.5, el_cor, 2)
                except:
                    pass
                (x2c,y2c) = (xc,yc)

            except:
                print('Nenhum objeto encontrado!')

            
            
            out.write(frame)
        else:
            break
    
    out.release()
    cap.release()

def mobilenetSSD():
    RESIZED_DIMENSIONS = (300, 300) # Dimensoes que o SSD é treinado
    IMG_NORM_RATIO = 0.007843 # Define a escala de cinza entre os valores de 0 a 255
    
    # Carrega a rede neural pré treinada
    neural_network = cv.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
            'MobileNetSSD_deploy.caffemodel')
    
    # Lista de categorias e classes
    categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
                4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
                9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
                13: 'horse', 14: 'motorbike', 15: 'person', 
                16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
                19: 'train', 20: 'tvmonitor'}
    
    classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
                "bus", "car", "cat", "chair", "cow", 
            "diningtable",  "dog", "horse", "motorbike", "person", 
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                        
    # Declara a variável que define a cor do objeto
    bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))

    # Carrega o video extraído
    cap = cv.VideoCapture('video-extraido.avi')

    # Salva a analise utilizando redes neurais
    out = cv.VideoWriter('video-extraido-MNSSD.avi',cv.VideoWriter_fourcc(*'DIVX'),5,(640,412))
    
    def med(x,y):
        return (x + y)/2

    # Processamento do video
    while True:
         
        # Captura frame por frame
        ret, frame = cap.read() 
    
        # Se for identificado video prossegue o processamento
        if ret:
            
            # Captura o tamanho dos frames
            (h, w) = frame.shape[:2]

            # Cria um blob, grupo de pixels conectados em um frame binário que possuí propriedades comuns
            # Preporcessamento do frame para a classificação do modelo de deep learn
            frame_blob = cv.dnn.blobFromImage(cv.resize(frame, RESIZED_DIMENSIONS), 
                            IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
            
            # Seleciona a entrada para a rede neural
            neural_network.setInput(frame_blob)
        
            # Saída da rede neural
            neural_network_output = neural_network.forward()
        
            # Avalia classe por classe
            for i in np.arange(0, neural_network_output.shape[2]):
                    
                confidence = neural_network_output[0, 0, i, 2]
            
                # Acurácia da detecção deve ser maior de 50% para ser mostrada       
                if confidence > 0.50:
                        
                    idx = int(neural_network_output[0, 0, i, 1])
                    
                    # Ignora classes diferentes de pessoa,sofa e cadeira
                    if classes[idx] != 'person' and 'sofa' and 'chair':
                        continue

                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                    [w, h, w, h])
                
                
                    (sX, sY, eX, eY) = bounding_box.astype("int")
        
                    label = "{}: {:.2f}%".format(classes[idx], confidence * 100) 
                
                    cv.rectangle(frame, (sX, sY), (
                    eX, eY), bbox_colors[idx], 2)
                    
                                
                    y = sY - 15 if sY - 15 > 30 else sY + 15   
                    cv.putText(frame, label, (sX, y),cv.FONT_HERSHEY_SIMPLEX, 
                    0.5, bbox_colors[idx], 2)
                    try:
                        y2 = sY - 30 if sY - 15 > 30 else sY + 30
                        coef_mov = sqrt((med(sX2,eX2) - med(sX,eX))**2 + (med(sY2,eY2) - med(sY,eY))**2)
                        label2 = "Coef. de mov.: {:.2f}".format(coef_mov)
                        cv.putText(frame, label2, (sX, y2),cv.FONT_HERSHEY_SIMPLEX, 
                    0.5, bbox_colors[idx], 2)
                    except:
                        pass
                    
                    (sX2,sY2,eX2,eY2) = (sX,sY,eX,eY)

                # Redimensionamento do frame
                frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)
            
            out.write(frame)
        else:
            break

    out.release()
    cap.release()
    


cam_in = [0,0]
th0 = Thread(target = dados_dos_sensores)
th0.start()
th1 = Thread(target = grafico_h)
th1.start()
th2 = Thread(target = grafico_d)
th2.start()
th3 = Thread(target = vidExtract)
th3.start()


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

