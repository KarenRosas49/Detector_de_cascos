#Código para el entrenamiento del reconocedor

#Se importan las librerías
import cv2
import os 
import numpy as np

dataPath = "/home/karen/Documents/GitHub/Detector_de_cascos/Dataset" #Especificar la ruta donde se encuentra almacenado el dataset
dir_list = os.listdir(dataPath) #Se listan las carpetas que se encuentran en la carpeta especificada
print("Lista archivos:", dir_list) #Se muestra en la consola el nombre de las carpetas dentro del directorio donde se encuentran los datasets
 
labels = [] #Arrays que se utilizará para almacenar el valor correspondiente a "con_casco" y un valor correspondiente a "sin_casco"
facesData = [] #Array que se utilizará para almacenar la etiqueta con_"casco" y la etiqueta "sin_casco"
label = 0 #Contador que se iniciará en cero

#Lectura de cada una de las imágenes almacenadas en los datasets
     #El primer "for" hace lectura de cada una de las carpetas con los datasets seleccionados
     #El segundi "for" lee cada una de las imágenes guardadas en cada carpeta
for name_dir in dir_list:
     dir_path = dataPath + "/" + name_dir
     
     for file_name in os.listdir(dir_path):
          image_path = dir_path + "/" + file_name
          print(image_path)
          image = cv2.imread(image_path, 0) #Se lee la imagen y con el 0 se especifica que es escala de grises
          
          facesData.append(image) #Se almacena cada imagen en escala de grises
          labels.append(label) #Se almacena el valor correspondiente a cada rostro, es decir, en cada imagen se coloca si correspode a "Con_casco" o si corresponde a "Sin_casco"
     label += 1

print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0)) #Se cuentan cuántos ceros hay
print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1)) #Se cuentan cuántos unos hay

# Eigen FaceRecognizer
face_mask = cv2.face.EigenFaceRecognizer_create()

# Entrenamiento
print("Entrenando, por favor espera...")
face_mask.train(facesData, np.array(labels))

# Almacenar modelo
face_mask.write("modelo_cascos1.xml")
print("Modelo almacenado exitosamente") 