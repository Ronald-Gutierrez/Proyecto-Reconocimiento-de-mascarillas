import cv2
import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #usaremos la camara

colorBajo = np.array([175, 100, 20], np.uint8) # reconoce el color 
colorAlto = np.array([179,255,255], np.uint8)


x1 = None
y1 = None
imAux = None

while True:

	ret,frame = cap.read()
	if ret==False: break

	frame = cv2.flip(frame,1)
	frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	if imAux is None: imAux = np.zeros(frame.shape,dtype=np.uint8)
	
	# Detección del color celeste
	maskColor = cv2.inRange(frameHSV, colorBajo, colorAlto)
	maskColor = cv2.erode(maskColor,None,iterations = 1)
	maskColor = cv2.dilate(maskColor,None,iterations = 2)
	maskColor = cv2.medianBlur(maskColor, 13)
	cnts,_ = cv2.findContours(maskColor, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
	
	for c in cnts:
		area = cv2.contourArea(c)
		if area > 1000:
			x,y2,w,h = cv2.boundingRect(c)
			cv2.rectangle(frame,(x,y2),(x+w,y2+h),(0,0,255),2)
			x2 = x + w//2
			
		else:
			x1, y1 = None, None
	
	
	#cv2.imshow('imAux',imAux)
	cv2.imshow('frame', frame)
	cv2.imshow('maskColor', maskColor)

	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
