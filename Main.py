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

#Begin serial comunication with Arduino
SerialComunication = serial.Serial('/dev/rfcomm0',9600)


def main(data):

    def read_serial(data):

        while True:

            #Read Arduino Data
            serialPrint = str(SerialComunication.readline())

            #Adjust data values
            vals = serialPrint.split(':')
            vals = [float(vals[x]) for x in range(1,11)]

            #Get the body rotation and acceleration
            acc = sqrt(vals[2]**2 + vals[3]**2 + vals[4]**2)
            gir = sqrt(vals[5]**2 + vals[6]**2 + vals[7]**2)
            
            #Append MPU data into the values list
            vals.append(round(acc,2))
            vals.append(round(gir,2))
            
            #Pass data to others threads
            for i in range(len(data)):
                data[i] = vals[i]


    def update_data(sensor):

        while True:
            
            #Update data into the database RRDTool
            rrdtool.update('temp-gas.rrd','N:' + str(sensor[0]) + ':' + str(sensor[1]))
            rrdtool.update('acelerometro.rrd','N:' + str(sensor[2]) + ':' + str(sensor[3]) + ':' + str(sensor[4]) + ':' + str(sensor[10]))
            rrdtool.update('giroscopio.rrd','N:' + str(sensor[5]) + ':' + str(sensor[6]) + ':' + str(sensor[7]) + ':' + str(sensor[11]))
            
            sleep(1)

    #Graphics properties
    def properties(name,start,step,end,title,Yaxis,height,width,minimum,maximum):
        return [name,'--start',start,'--step',step,'--end',end,'--height='+height,
        '--width='+width,'--title',title,'--vertical-label',Yaxis,'-N','-i','--rigid',
        '--slope-mode','--color', 'BACK#FFFFFF','-l',minimum,'-u',maximum]
    
    #Create and update this graphics in real time
    def grafico_rt():

        while True:
            
            rrdtool.graph(properties('ModeloInc.png','now - 12s','1','now','Modelo - Detecção de incêndio', \
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
        
            rrdtool.graph(properties('accMpu.png','now  - 300s','1s','now','','Accelerometro[g]','220','500','0','4'),
                    'DEF:acc=acelerometro.rrd:Acc:AVERAGE',
                    'CDEF:norAcc=acc,1.5,LT,acc,UNKN,IF','CDEF:altAcc=acc,1.5,GT,acc,UNKN,IF',
                    'LINE2:acc#00BB0095:Dados do acelerometro','LINE2:altAcc#FFFF0095:Possível queda',
                    'HRULE:1.42#FF000095:Limiar de risco do accelerômetro')

            rrdtool.graph(properties('girMpu.png','now  - 300s','1s','now','','Giroscópio[°]','220','500','-10','180'),
                    'DEF:gir=giroscopio.rrd:Gir:AVERAGE',
                    'LINE2:gir#0000FF95:Dados do Giroscópio')

            rrdtool.graph(properties('3eAcc.png','now  - 300s','1s','now','Dados do acelerometro','','220','500','-1','3'),
                        'DEF:x=acelerometro.rrd:AccX:AVERAGE',
                        'DEF:y=acelerometro.rrd:AccY:AVERAGE',
                        'DEF:z=acelerometro.rrd:AccZ:AVERAGE',
                        'LINE2:x#FF0000:Eixo X',
                        'LINE2:y#00FF00:Eixo Y',
                        'LINE2:z#0000FF:Eixo Z')
                        
            rrdtool.graph(properties('3eGir.png','now  - 300s','1s','now','Dados do giroscópio','Ângulo[°]', \
                        '220','500','-360','360'),
                        'DEF:x=giroscopio.rrd:GirX:AVERAGE',
                        'DEF:y=giroscopio.rrd:GirY:AVERAGE',
                        'DEF:z=giroscopio.rrd:GirZ:AVERAGE',
                        'LINE2:x#FF0000:Eixo X',
                        'LINE2:y#00FF00:Eixo Y',
                        'LINE2:z#0000FF:Eixo Z')
                        
            sleep(2)

    #Last hour data graphic
    def grafico_h():

        while True:
            rrdtool.graph(properties('temph.png','now - 1h','1m','now','Monitoramento de temperatura na ultima hora', \
                    'Temperatura[°C]','220','800','0','60'),'DEF:t=temp-gas.rrd:Temperatura:AVERAGE',
                    'AREA:t#0000DD85:Temperatura')
            
            sleep(60)
    
    #Daily data graphic
    def grafico_d():

        while True:
            rrdtool.graph(properties('tempDiaria.png','now - 1d','5m','now','Monitoramento Diário de temperatura', \
                    'Temperatura[°C]','220','800','0','60'),'DEF:t=temp-gas.rrd:Temperatura:AVERAGE',
                    'AREA:t#0000DD85:Temperatura')
        
            sleep(300)

    #Create threads for each graphic
    th1 = Thread(target=read_serial,args=(data,))
    th1.start()

    th2 = Thread(target=update_data,args=(data,))
    th2.start()

    th3 = Thread(target = grafico_rt)
    th3.start()

    th4 = Thread(target = grafico_h)
    th4.start()

    th5 = Thread(target = grafico_d)
    th5.start()

#Video extraction in the fall moment
def videoExtraction(sensor):

    #Accumulate images (Create cache)
    def accImg(num):
        cv.imwrite(dir + "/frame" + str(num) + '.jpg',frame)
        dirs.append(dir + "/frame" + str(num) + '.jpg')

    # Declaração das variáveis necessárias para o funcionamento do programa
    imgs,dirs,cont1,cont2,num_frame,changes = [],[],0,0,0,False
    dir = '/home/lucas/Documentos/Python/Camera'
    df = Dataframe()
    
    print("Organize sistem...")
    
    #Clear images cache
    try:
        for arq in listdir(dir):
            remove(path.join(dir,arq))
    except:
        pass
    
    print("Booting Camera Capture...")
    cap = cv.VideoCapture(0)

    samples = 14 #Half images to be stored in cache
    
    #Start infinite loop
    while True:
        
        #read camera frame to frame
        _,frame = cap.read()

        #Resizing frame
        frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)
        
        #Save data into csv file
        df.put(list(sensor),num_frame)

        # Set condition with sensor value
        if sensor[10] > 1.42 or changes == True:    

            changes = True

            #Accumulates 14 frames after accident
            if cont2 < samples:
                cont2 = cont2 + 1
                accImg(cont1 + cont2)
            else:
                print("Compiling images...")
                break
                    

        # When the values from sensor are normal
        else:
            cont1 = cont1 + 1

            # Accumulates 14 frames before changes in sensor value
            accImg(cont1)
            
            if cont1 < samples:
                pass
            else:
                remove(dir +"/frame" + str(cont1 - (samples - 1)) + '.jpg')
                dirs.pop(0)

        num_frame = num_frame + 1
    
    # Collect all accumulate image
    for file in dirs:
        img = cv.imread(file)
        h ,w, l= img.shape
        tam = (w,h)
        imgs.append(img)

    #Create a video output
    out = cv.VideoWriter('teste-extraido.mp4',0x31637661,5,tam)

    # Create a video with the accumulates images
    for x in range(len(imgs)):
        out.write(imgs[x])
        
    #Release camera capture and the output video
    out.release()
    cap.release()
    print("Sucess to compile the video...")


    mobilenetSSD(df)
    CNTsubtractor(df)

    df.get()

def mobilenetSSD(dataframe):

    def molding(color):
        cv.rectangle(frame, (sX, y2), (sX + 60, sY), color, -1)
        cv.rectangle(frame, (sX, sY), (eX, eY), color, 2)
        cv.rectangle(frame, (eX, sY + 90), (eX + 85, sY), color, -1)
        cv.putText(frame, "person", (sX, y),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
        cv.putText(frame,"precision",(eX,sY + 15),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
        cv.putText(frame,prec,(eX,sY + 35),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)

    print("Processing video...")

    RESIZED_DIMENSIONS = (300, 300) # Train data dimensions
    IMG_NORM_RATIO = 0.007843 # Define gray scale ratio
    
    # Load neural network
    neural_network = cv.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
            'MobileNetSSD_deploy.caffemodel')
    
    # Classify data
    classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable",  "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Load the video extracted in the possible fall moment
    cap = cv.VideoCapture('teste-extraido.mp4')

    # Create the neural network output
    out = cv.VideoWriter('teste-MNSSD.mp4',0x31637661,5,(640,412))
    
    # Create the centroid object and the dict to upload the dataframe
    ct = CentroidTracker()
    cms = defaultdict()
    for i in range(5):
        cms.update({i:[]})

    # Processing the video
    while True:
         
        # Capture each frame
        ret, frame = cap.read() 

        # if has video continue the capture
        if ret:
            
            # Capture the frame length
            (h, w) = frame.shape[:2]

            # Create a blob, pre-processing neural network
            frame_blob = cv.dnn.blobFromImage(cv.resize(frame, RESIZED_DIMENSIONS), 
                            IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
            
            # Select blob input to the neural network
            neural_network.setInput(frame_blob)
        
            # Get the output
            neural_network_output = neural_network.forward()

            # Get the momentary frame
            num_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            rects = []

            # Evaluate all classes
            for i in np.arange(0, neural_network_output.shape[2]):
                    
                confidence = neural_network_output[0, 0, i, 2]
            
                # Set the confidence of the detection     
                if confidence > 0.35:
                        
                    idx = int(neural_network_output[0, 0, i, 1])
                    
                    # Ignore diferents classes
                    if classes[idx] != 'person':
                        continue

                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                    [w, h, w, h])
                
                    rects.append(bounding_box.astype("int"))
                    (sX, sY, eX, eY) = bounding_box.astype("int")

                    prec = "{:.2f}%".format(confidence * 100) 
                                
                    y = sY - 5 if sY - 15 > 30 else sY + 15
                    y2 = sY - 20 if sY - 15 > 30 else sY + 20
                    molding(green)
            
                    objs = ct.update(rects)
                    
                    #Compare the centroid distance between two frames for each object detected 
                    try:
                        for objectID in objs.keys():
                            cm = sqrt((objs[objectID][0] - p_objs[objectID][0])**2 + (objs[objectID][1] - p_objs[objectID][1])**2)

                            if cm > 60:
                                molding(yellow)
                                cv.rectangle(frame, (sX, objs[objectID][1] - 15), (eX, objs[objectID][1] + 5), yellow, -1)
                                cv.putText(frame, "Possible Accident", (sX + 10,objs[objectID][1]),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)

                            if cm != 0:
                                cms[objectID].append(round(cm,2))
                                text_cm = "{:.2f} id:{}".format(cm,objectID)
                                cv.putText(frame,"coef. mov.", (eX,sY + 55),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                                cv.putText(frame,text_cm, (eX,sY + 75),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                    except:
                        pass

                    p_objs = objs.copy()
                    
            
            try:
                num_person.append(len(objs))
                num_person_frame = "Pessoas detectadas: {}".format(len(objs))
                cv.putText(frame, num_person_frame, (15,397),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
            except:
                pass
            
            if num_frame == 27:
                max_ids = max(num_person)
                for i in range(max_ids,5):
                    cms.pop(i,None)
                for value in cms.values():
                    dataframe.insert(list(value))
                    
                    
                
            # Resize frame
            frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)
        
            out.write(frame)
           
        else:
            break

    out.release()
    cap.release()

def CNTsubtractor(dataframe):

    def secondMolding(color):
        cv.rectangle(frame, (xc, yc - d1 - 15), (xc + 80, yc - d1 + 70), color, -1)
        cv.ellipse(frame,ellipse,color, 2)

    angles = defaultdict()
    falls = [False] * max(num_person)
    for i in range(max(num_person)):
        angles.update({i:[]})

    # Get video extracted
    cap = cv.VideoCapture('teste-extraido.mp4')

    # Create the output for subtraction method
    out = cv.VideoWriter('teste-CNT.mp4',0x31637661,5,(640,412))

    # Create method
    backSub = cv.bgsegm.createBackgroundSubtractorCNT(5,True)
    
    while True:

        ret,frame = cap.read()
        
        if ret:
            # Pre-processing filters
            blurFrame = cv.GaussianBlur(frame,(5,5),0)

            # Apply mask
            masc = backSub.apply(blurFrame)

            # Detect contours 
            contours,_ = cv.findContours(masc, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # Create a list with each object area
            areas = []

            # Fill area list
            for contour in contours:
                areas.append(cv.contourArea(contour))

            num_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            
            # Start analysis
            try:
                
                # Detect at least 1 object
                if num_person[num_frame] == 0:
                    num_person[num_frame] = 1
                
                # For each person
                for id_person in range(num_person[num_frame]):
                    
                    # Area length condition
                    if max(areas) < 220000:
                        
                        # Get greater area
                        m_area = areas.index(max(areas))
                        max_contour = contours[m_area]
                        ellipse = cv.fitEllipse(max_contour)
                        (xc,yc),(d1,d2),angle = ellipse
                        xc,yc,d1,ang = round(xc),round(yc),round(d1),round(angle,2)
                        angles[id_person].append(ang)
                        
                        if angle > 80 and angle < 130:
                            secondMolding(yellow)
                            cv.rectangle(frame,(xc - 85,yc - 15),(xc + 85,yc + 5),yellow,-1)
                            cv.putText(frame, "Possivel Acidente!", (xc - 75,yc),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                            
                        else:
                            secondMolding(green)
                        
                        cv.putText(frame, "person", (xc,yc - d1),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                        cv.putText(frame, str(id_person), (xc,yc - d1 + 20),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                        cv.putText(frame, "angle", (xc,yc - d1 + 40),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                        cv.putText(frame, str(ang), (xc,yc - d1 + 60),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                            
                        areas[m_area] = 0

                    else:
                        pass

            except:
                pass
            
            
            if num_frame == 27:
                c = 0
                try:
                    
                    for values in angles.values():

                        dataframe.insert(list(values))
                        med = median(values[-11:])
                        std_dev = pstdev(values[-11:])
                        model = sqrt((med-90)**2 + (std_dev - 40)**2)
                        
                        if model <= 50:
                            falls[c] = True
                        c = c + 1
                except:
                    pass

                if any(x for x in falls) == True:
                    cv.rectangle(frame,(0,185),(640,205),red,-1)
                    cv.putText(frame, "Fall detected", (280,200),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                else:
                    cv.rectangle(frame,(0,185),(640,205),green,-1)
                    cv.putText(frame, "Fall not detected", (280,200),cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
                    
            
            frame = cv.resize(frame, (640,412), interpolation=cv.INTER_NEAREST)

            out.write(frame)
        else:
            break
    
    dataframe.get()
    out.release()
    cap.release()
    
    print("Methods Applied...")

if __name__ == '__main__':

    num_person = []
    green,yellow,red,black = (0,255,128),(0,255,255),(0,0,255),(0,0,0)
    
    #Create database
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

    data = Array('f',[0]*12)
 
    p1 = Process(target= main,args=(data, ))
    p1.start()

    p2 = Process(target= videoExtraction,args=(data, ))
    p2.start()
