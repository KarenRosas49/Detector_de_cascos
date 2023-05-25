#Código para probar el modelo entrenado

import cv2
import os
import mediapipe as mp
import urllib.request #para abrir y leer URL
import paho.mqtt.client as mqtt

url = 'http://192.168.100.26' #Se coloca el url de la esp32cam

client=mqtt.Client()
client.connect("broker.hivemq.com",1883,60)

mp_face_detection = mp.solutions.face_detection #Se manda a llamar la solución de la librería mediapipe para la detección de rostros

LABELS = ["Con_casco", "Sin_casco"] #Se coloca el nombre de los dos grupos entrenados con el programa train.py

# Leer el modelo
face_helmet = cv2.face.EigenFaceRecognizer_create()
face_helmet.read("modelo_cascos1.xml")

#Lectura del video

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Si se desea usar la cámara de la computadora, descomentar esta línea
cap = cv2.VideoCapture(url + ":81/stream") #Se lee el url de la esp32cam
#cap = cv2.VideoCapture("capture.mp4") #Descomentar esta l línea se se desea usar un video existente




with mp_face_detection.FaceDetection(
     min_detection_confidence=0.5) as face_detection:

     while True:
          ret, frame = cap.read()
          if ret == False: break
          frame = cv2.flip(frame, 1) #Se invierte la imagen


          #Se cambia la imagen a RGB
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
          results = face_detection.process(frame_rgb) #Se observan los resultados

          #Se dibuja el cuadrado que detecta el rostro
          if results.detections is not None:
               for detection in results.detections:
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    if xmin < 0 and ymin < 0:
                         continue
                    
                    #Se extrae la información de los rostros y se aplica escala de grises
                    face_image = frame[ymin : ymin + h, xmin : xmin + w]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC) #Se redimenciona el tama;o de imagen a 72x72 pixeles
                    
                    #Se aplica el modelo entrenado
                    result = face_helmet.predict(face_image)
                    
                    #Dibuja el rectángulo y le coloca una etiqueta
                    if result[1] < 5700:                     
                         color = (0, 255, 0) 
                         client.publish("proyecto/parte2/detector_cascos","TRUE") if LABELS[result[0]] == "Con_casco"  else (0, 0, 255)
                         cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                         cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)

          cv2.imshow("Frame", frame)
          k = cv2.waitKey(1)
          if k == 27:
               break

cap.release()
cv2.destroyAllWindows()