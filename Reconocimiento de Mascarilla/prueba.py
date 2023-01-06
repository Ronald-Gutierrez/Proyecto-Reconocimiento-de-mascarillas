import cv2
import os
from playsound import playsound
from mtcnn.mtcnn import MTCNN

# importamos la libreria
direccion = 'C:/Users/Ronald/Desktop/Cursos del tercer semestre/Arquitectura de computadoras/Modelo de entrenamiento' 
dire_img = os.listdir(direccion)
print("Nombre: ",dire_img)
#-------------------------------- Llamamos el modelo de reconocimiento ------ 
reconocimiento = cv2. face. LBPHFaceRecognizer_create()

#-- Leemos el modelo 
reconocimiento.read('modeloLBP.xml')
#- Capturamos el video en tiempo real 
detector = MTCNN() #Creamos el objeto que va a detectar 
cap = cv2.VideoCapture (0)
while True:
        ret, frame = cap.read() 
        if ret == False: break 
        gris = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY) 
        copia = frame.copy () 
        copia2 = gris.copy ()
        caras = detector.detect_faces (copia)
        for i in range (len (caras)):
            x1, y1, ancho, alto = caras[i]['box']
            x2, y2 = x1 + ancho, y1 + alto 
            cara_reg = copia2 [y1:y2, x1:x2] 
            cara_rec = cv2.resize (cara_reg, (150, 200), interpolation = cv2.INTER_CUBIC) #Ajustamos la imagen con un tamaño de 150x200 
            resultado = reconocimiento.predict(cara_rec) 

            #  Vamos a mostrar en pantalla los resultados 
            if resultado [0] == 0:
                cv2.putText (frame, 'CON TAPABOCAS'.format (dire_img[0]), (x1, y1-5),1,1.3, (0,255,0),1, cv2.LINE_AA)
                cv2.rectangle (frame, (x1, y1), (x1+ancho, y1+alto), (0,255,0), 2)
                
 
            if resultado [0] == 1:
                cv2.putText (frame, 'SIN TAPABOCA'.format (dire_img[1]), (x1, y1-5), 1, 1.3, (0,0,255), 1, cv2.LINE_AA)
                cv2.rectangle (frame, (x1, y1), (x1+ancho, y1+alto), (0,0,255), 2)
                playsound('C:\\Users\\Ronald\\Desktop\\CIENCIA COMPUTACION\\Proyect_Arquitect\\sound\\alarma.mp3') 
                
        cv2.imshow ('Reconocimiento del Tapabocas', frame)
        t = cv2.waitKey(1) 
        if t == 27:
            break
cap.release () 
cv2.destroyAllWindows()
