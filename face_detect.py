import cv2
import numpy

face_cascade = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')

capL = cv2.VideoCapture(0)

while(True):
	retL, imgL = capL.read()
	
	gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 2)
	for (x,y,w,h) in faces:
		imgL = cv2.rectangle(imgL,(x,y),(x+w,y+h),(255,0,0),2)
		print 'detected'
	
	cv2.imshow('img',imgL)
	if cv2.waitKey(1) & 0xFF == ord('q'):
                break
	
cv2.destroyAllWindows()
