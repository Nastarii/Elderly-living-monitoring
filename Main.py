from centroidtracker import CentroidTracker
from dataframe import Dataframe
from multiprocessing import Process,Array
from collections import defaultdict
from threading import Thread
from time import sleep
from math import sqrt
from os import path,remove,listdir
from statistics import median,pstdev
import numpy as np
import cv2 as cv
import rrdtool
import serial

#sudo su
#rfcomm bind 0 98:D3:71:F6:23:02
#Inicia a comunicação serial com o Arduino
SerialComunication = serial.Serial('/dev/rfcomm0',9600)


def main(dados):

    def receber_serial(dados):

        while True:

            #Lê os dados do arduino
            serialPrint = str(SerialComunication.readline())

            #Realiza os ajustes dos dados recebidos na serial
            vals = serialPrint.split(':')
            vals = [float(vals[x]) for x in range(1,11)]

            #Faz o cálculo da aceleração e rotação corporal com os 3 eixos
            acc = sqrt(vals[2]**2 + vals[3]**2 + vals[4]**2)
            gir = sqrt(vals[5]**2 + vals[6]**2 + vals[7]**2)
            vals.append(round(acc,2))
            vals.append(round(gir,2))
            
            #Passa os dados para serem utilizados em outras linhas de processamento
            for i in range(len(dados)):
                dados[i] = vals[i]


    def atualizar_dados(sensor):

        while True:
            
            #Atualiza os dados dos sensores em no banco de dados RRDTOOL
            rrdtool.update('temp-gas.rrd','N:' + str(sensor[0]) + ':' + str(sensor[1]))
            rrdtool.update('acelerometro.rrd','N:' + str(sensor[2]) + ':' + str(sensor[3]) + ':' + str(sensor[4]) + ':' + str(sensor[10]))
            rrdtool.update('giroscopio.rrd','N:' + str(sensor[5]) + ':' + str(sensor[6]) + ':' + str(sensor[7]) + ':' + str(sensor[11]))
            
            sleep(1)

    #Função com as propriedades básicas dos gráficos
    def prop(nome,inicio,periodo,fim,titulo,eixoY,altura,largura,minimo,maximo):
        return [nome,'--start',inicio,'--step',periodo,'--end',fim,'--height='+altura,
        '--width='+largura,'--title',titulo,'--vertical-label',eixoY,'-N','-l','0','-i','--rigid',
        '--slope-mode','--color', 'BACK#FFFFFF','-l',minimo,'-u',maximo]

    def grafico_tr():

        while True:
            #Cria os gráficos que são atualizados em tempo real
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

            rrdtool.graph(prop('accMpu.png','now  - 300s','1s','now','','Accelerometro[g]','220','500','0','4'),
                    'DEF:acc=acelerometro.rrd:Acc:AVERAGE',
                    'CDEF:norAcc=acc,1.5,LT,acc,UNKN,IF','CDEF:altAcc=acc,1.5,GT,acc,UNKN,IF',
                    'LINE2:acc#00BB0095:Dados do acelerometro'#,'LINE2:altAcc#FFFF0095:Possível queda',
                    #'HRULE:1.5#FF000095:Limiar de risco do accelerômetro'
                    )

            rrdtool.graph(prop('girMpu.png','now  - 300s','1s','now','','Giroscópio[°]','220','500','-10','180'),
                    'DEF:gir=giroscopio.rrd:Gir:AVERAGE',
                    #'CDEF:norGir=gir,90,LT,gir,UNKN,IF','CDEF:altGir=gir,90,GT,gir,UNKN,IF',
                    #'LINE2:norGir#0000FF95:Dados do Giroscópio','LINE2:altGir#FFFF0095:Possível queda'
                    #'HRULE:90#FF000095:Limiar de risco do giroscópio')
                    'LINE2:gir#0000FF95:Dados do Giroscópio')

            rrdtool.graph(prop('3eAcc.png','now  - 300s','1s','now','Dados do acelerometro','','220','500','-1','3'),
                        'DEF:x=acelerometro.rrd:AccX:AVERAGE',
                        'DEF:y=acelerometro.rrd:AccY:AVERAGE',
                        'DEF:z=acelerometro.rrd:AccZ:AVERAGE',
                        'LINE2:x#FF0000:Eixo X',
                        'LINE2:y#00FF00:Eixo Y',
                        'LINE2:z#0000FF:Eixo Z')
                        
            rrdtool.graph(prop('3eGir.png','now  - 300s','1s','now','Dados do giroscópio','Ângulo[°]', \
                        '220','500','-360','360'),
                        'DEF:x=giroscopio.rrd:GirX:AVERAGE',
                        'DEF:y=giroscopio.rrd:GirY:AVERAGE',
                        'DEF:z=giroscopio.rrd:GirZ:AVERAGE',
                        'LINE2:x#FF0000:Eixo X',
                        'LINE2:y#00FF00:Eixo Y',
                        'LINE2:z#0000FF:Eixo Z')
                        
            sleep(2)

    def grafico_h():

        while True:
            #Cria o gráfico que realiza o monitoramento da última hora
            rrdtool.graph(prop('temph.png','now - 1h','1m','now','Monitoramento de temperatura na ultima hora', \
                    'Temperatura[°C]','220','800','0','60'),'DEF:t=temp-gas.rrd:Temperatura:AVERAGE',
                    'AREA:t#0000DD85:Temperatura')
            
            sleep(60)

    def grafico_d():

        while True:
            #Cria o gŕafico que é atualizado diariamente
            rrdtool.graph(prop('tempDiaria.png','now - 1d','5m','now','Monitoramento Diário de temperatura', \
                    'Temperatura[°C]','220','800','0','60'),'DEF:t=temp-gas.rrd:Temperatura:AVERAGE',
                    'AREA:t#0000DD85:Temperatura')
        
            sleep(300)

    #Inicia as Linhas de processamento das funções contidas na função principal
    th1 = Thread(target=receber_serial,args=(dados,))
    th1.start()

    th2 = Thread(target=atualizar_dados,args=(dados,))
    th2.start()

    th3 = Thread(target = grafico_tr)
    th3.start()

    th4 = Thread(target = grafico_h)
    th4.start()

    th5 = Thread(target = grafico_d)
    th5.start()

#Extração do vídeo no momento da queda
def extrairVideo(sensor):

    #Função responsável por salvar as imagens
    def guardarImg(num):
        cv.imwrite(dir + "/frame" + str(num) + '.jpg',frame)
        dirs.append(dir + "/frame" + str(num) + '.jpg')

    # Declaração das variáveis necessárias para o funcionamento do programa
    imgs,dirs,cont1,cont2,num_frame,alter = [],[],0,0,0,False
    dir = '/home/lucas/Documentos/Python/Camera'
    df = Dataframe()
    
    print("Arrumando sistema...")
    
    #Remove as imagens de detecções anteriores
    try:
        for arq in listdir(dir):
            remove(path.join(dir,arq))
    except:
        pass
    # Inicialização da captura da câmera
    print("Inicializando Captura da câmera...")
    cap = cv.VideoCapture(0)

    amostras = 14 #Metade do número de imagens a serem salvas

    # Inicialização da leitura dos vals da câmera
    while True:
        
        #Lê a captura frame a frame
        _,frame = cap.read()

        #Redimensionamento do frame
        frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)
        
        #Salva os dados dos sensores em uma planilha
        df.put(list(sensor),num_frame)

        # Se os vals dos sensores estão alterados
        if sensor[10] > 1.38 or alter == True:    

            alter = True

            #Guarda 10 frames após detectada a alteração
            if cont2 < amostras:
                cont2 = cont2 + 1
                guardarImg(cont1 + cont2)
            else:
                print("Compilando as imagens...")
                break
                    

        # Se não há alterações nos sensores
        else:
            cont1 = cont1 + 1

            # Guarda os últimos frames
            guardarImg(cont1)
            if cont1 < amostras:
                pass
            else:
                remove(dir +"/frame" + str(cont1 - (amostras - 1)) + '.jpg')
                dirs.pop(0)

        num_frame = num_frame + 1
    
    # Cria um vídeo com os frames de antes e depois da queda
    for arquivo in dirs:
        img = cv.imread(arquivo)
        h ,w, l= img.shape
        tam = (w,h)
        imgs.append(img)

    #Cria uma saida de video
    out = cv.VideoWriter('teste-extraido.mp4',0x31637661,5,tam)

    # Preenche a saída de video com as imagens coletadas
    for x in range(len(imgs)):
        out.write(imgs[x])

    out.release()
    cap.release()
    print("Vídeo compilado com sucesso!")


    mobilenetSSD(df)
    CNTsubtractor(df)

    df.get()

def mobilenetSSD(dataframe):

    def moldura(cor):
        cv.rectangle(frame, (sX, y2), (sX + 60, sY), cor, -1)
        cv.rectangle(frame, (sX, sY), (eX, eY), cor, 2)
        cv.rectangle(frame, (eX, sY + 90), (eX + 85, sY), cor, -1)
        cv.putText(frame, "pessoa", (sX, y),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
        cv.putText(frame,"precisao",(eX,sY + 15),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
        cv.putText(frame,prec,(eX,sY + 35),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)

    print("Processando video...")

    RESIZED_DIMENSIONS = (300, 300) # Dimensoes que o SSD é treinado
    IMG_NORM_RATIO = 0.007843 # Define a escala de cinza entre os valores de 0 a 255
    
    # Carrega a rede neural pré treinada
    neural_network = cv.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
            'MobileNetSSD_deploy.caffemodel')
    
    # Lista de classes
    
    classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable",  "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Carrega o video extraído
    cap = cv.VideoCapture('teste-extraido.mp4')

    # Salva a analise utilizando redes neurais
    out = cv.VideoWriter('teste-MNSSD.mp4',0x31637661,5,(640,412))

    ct = CentroidTracker()
    cms = defaultdict()
    for i in range(5):
        cms.update({i:[]})

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

            num_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            rects = []

            # Avalia classe por classe
            for i in np.arange(0, neural_network_output.shape[2]):
                    
                confidence = neural_network_output[0, 0, i, 2]
            
                # Acurácia da detecção deve ser maior de 35% para ser mostrada       
                if confidence > 0.35:
                        
                    idx = int(neural_network_output[0, 0, i, 1])
                    
                    # Ignora classes diferentes de pessoa
                    if classes[idx] != 'person':
                        continue

                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                    [w, h, w, h])
                
                    rects.append(bounding_box.astype("int"))
                    (sX, sY, eX, eY) = bounding_box.astype("int")

                    prec = "{:.2f}%".format(confidence * 100) 
                                
                    y = sY - 5 if sY - 15 > 30 else sY + 15
                    y2 = sY - 20 if sY - 15 > 30 else sY + 20
                    moldura(verde)
            
                    objs = ct.update(rects)
                    
                    try:
                        for objectID in objs.keys():
                            coef_mov = sqrt((objs[objectID][0] - p_objs[objectID][0])**2 + (objs[objectID][1] - p_objs[objectID][1])**2)

                            if coef_mov > 60:
                                moldura(amarelo)
                                cv.rectangle(frame, (sX, objs[objectID][1] - 15), (eX, objs[objectID][1] + 5), amarelo, -1)
                                cv.putText(frame, "Possivel Acidente!", (sX + 10,objs[objectID][1]),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)

                            if coef_mov != 0:
                                cms[objectID].append(round(coef_mov,2))
                                text_cm = "{:.2f} id:{}".format(coef_mov,objectID)
                                cv.putText(frame,"coef. mov.", (eX,sY + 55),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                                cv.putText(frame,text_cm, (eX,sY + 75),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                    except:
                        pass

                    p_objs = objs.copy()
                    
            
            try:
                pessoas_detectadas.append(len(objs))
                num_pessoas = "Pessoas detectadas: {}".format(len(objs))
                cv.putText(frame, num_pessoas, (15,397),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
            except:
                pass
            
            if num_frame == 27:
                max_ids = max(pessoas_detectadas)
                for i in range(max_ids,5):
                    cms.pop(i,None)
                for valores in cms.values():
                    dataframe.insert(list(valores))
                    
                    
                
            # Redimensionamento do frame
            frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)
        
            out.write(frame)
           
        else:
            break

    out.release()
    cap.release()

def CNTsubtractor(dataframe):

    def molduraSub(cor):
        cv.rectangle(frame, (xc, yc - d1 - 15), (xc + 80, yc - d1 + 70), cor, -1)
        cv.ellipse(frame,ellipse,cor, 2)

    angulos = defaultdict()
    quedas = [False] * max(pessoas_detectadas)
    for i in range(max(pessoas_detectadas)):
        angulos.update({i:[]})

    # Inicialização da captura do video extraido
    cap = cv.VideoCapture('teste-extraido.mp4')

    # Salva a analise do método KNN
    out = cv.VideoWriter('teste-CNT.mp4',0x31637661,5,(640,412))

    # Criação do método de subtração de fundo
    subFundo = cv.bgsegm.createBackgroundSubtractorCNT(5,True)
    
    while True:

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

            num_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            
             # Identifica o objeto com maior área e circunscreve o contorno do objeto em uma elipse
            try:
                
                if pessoas_detectadas[num_frame] == 0:
                    pessoas_detectadas[num_frame] = 1
                
                for id_pessoa in range(pessoas_detectadas[num_frame]):

                    if max(areas) < 220000:
                        m_area = areas.index(max(areas))
                        max_contorno = contornos[m_area]
                        ellipse = cv.fitEllipse(max_contorno)
                        (xc,yc),(d1,d2),angulo = ellipse
                        xc,yc,d1,ang = round(xc),round(yc),round(d1),round(angulo,2)
                        angulos[id_pessoa].append(ang)
                        
                        if angulo > 80 and angulo < 130:
                            molduraSub(amarelo)
                            cv.rectangle(frame,(xc - 85,yc - 15),(xc + 85,yc + 5),amarelo,-1)
                            cv.putText(frame, "Possivel Acidente!", (xc - 75,yc),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                            
                        else:
                            molduraSub(verde)
                        
                        cv.putText(frame, "pessoa", (xc,yc - d1),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                        cv.putText(frame, str(id_pessoa), (xc,yc - d1 + 20),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                        cv.putText(frame, "angulo", (xc,yc - d1 + 40),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                        cv.putText(frame, str(ang), (xc,yc - d1 + 60),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                            
                        areas[m_area] = 0

                    else:
                        pass

            except:
                pass
            
            
            if num_frame == 27:
                c = 0
                try:
                    
                    for valores in angulos.values():

                        dataframe.insert(list(valores))
                        mediana = median(valores[-11:])
                        desvio_padrao = pstdev(valores[-11:])
                        modelo = sqrt((mediana-90)**2 + (desvio_padrao - 40)**2)
                        
                        if modelo <= 50:
                            quedas[c] = True
                        c = c + 1
                except:
                    pass

                if any(x for x in quedas) == True:
                    cv.rectangle(frame,(0,185),(640,205),vermelho,-1)
                    cv.putText(frame, "Queda detectada!", (280,200),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                else:
                    cv.rectangle(frame,(0,185),(640,205),verde,-1)
                    cv.putText(frame, "Queda nao detectada!", (280,200),cv.FONT_HERSHEY_SIMPLEX, 0.5, preto, 2)
                    
            
            frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)

            out.write(frame)
        else:
            break
    
    dataframe.get()
    out.release()
    cap.release()
    
    print("Métodos Aplicados...")

if __name__ == '__main__':

    pessoas_detectadas = []
    verde,amarelo,vermelho,preto = (0,255,128),(0,255,255),(0,0,255),(0,0,0)
    #Criando os bancos de dados caso ja não tenham sido criados

    if not path.isfile('temp-gas.rrd'):
        rrdtool.create('temp-gas.rrd',
                        '--start','now',
                        '--step','1s',
                        'DS:Temperatura:GAUGE:600:-40:85',
                        'DS:Fumaca:GAUGE:600:-1:400',
                        'RRA:LAST:0.5:1s:12s',
                        'RRA:AVERAGE:0.5:1m:1h',
                        'RRA:AVERAGE:0.5:5m:1d')

    if not path.isfile('acelerometro.rrd'):
        rrdtool.create('acelerometro.rrd',
                        '--start','now',
                        '--step','1s',
                        'DS:AccX:GAUGE:600:-3:3',
                        'DS:AccY:GAUGE:600:-3:3',
                        'DS:AccZ:GAUGE:600:-3:3',
                        'DS:Acc:GAUGE:600:-3:3',
                        'RRA:AVERAGE:0.5:1s:3600s',
                        'RRA:AVERAGE:0.5:1h:1d')

    if not path.isfile('giroscopio.rrd'):
        rrdtool.create('giroscopio.rrd',
                        '--start','now',
                        '--step','1s',
                        'DS:GirX:GAUGE:600:-180:180',
                        'DS:GirY:GAUGE:600:-180:180',
                        'DS:GirZ:GAUGE:600:-180:180',
                        'DS:Gir:GAUGE:600:-180:180',
                        'RRA:AVERAGE:0.5:1s:3600s',
                        'RRA:AVERAGE:0.5:1h:1d')

    dados = Array('f',[0]*12)
 
    p1 = Process(target= main,args=(dados, ))
    p1.start()

    p2 = Process(target= extrairVideo,args=(dados, ))
    p2.start()