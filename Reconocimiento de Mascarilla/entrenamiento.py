import cv2 
import os 
import numpy as np
#---- Importamos las fotos tomadas anteriormente--- 
direccion = 'C:/Users/Ronald/Desktop/Cursos del tercer semestre/Arquitectura de computadoras/Modelo de entrenamiento' 
lista = os.listdir (direccion)
etiquetas = [] 
rostros = [] 
con = 0
for nameDir in lista:
    nombre = direccion + '/' + nameDir           #Leeremos las fotos tomadas de los rostros

    for fileName in os.listdir (nombre):        #Asignaremos las etiquetas a cada foto
        etiquetas.append (con)                  #Valor de la etiqueta ( asignamos O a la primera etiqueta y a la segunda) 
        rostros.append (cv2.imread (nombre + '/' + fileName, 0)) #Añadimos las imagenes en jpg
        
    con = con + 1
#- Creamos el modelo 
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#Empezamos a entrenarlo con las fotos 
reconocimiento.train(rostros, np.array(etiquetas))

#Guardamos el modelo 
reconocimiento.write('modeloLBP.xml') 
print ("Modelo Creado")
