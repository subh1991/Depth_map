import numpy as np
import cv2

capR = cv2.VideoCapture(1)
capL = cv2.VideoCapture(0)
i = 0

while(True):
	# Capture frame-by-frame
	retL, frameL = capL.read()
	retR, frameR = capR.read()
	
	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Display the resulting frame
	cv2.imshow('frameL', frameL)
	cv2.imshow('frameR',frameR)
        		
	if cv2.waitKey(1) & 0xFF == ord('c'):
		nameL = './resources/left_image/left_img'+ str(i)+'.png'
		cv2.imwrite(nameL, frameL)
		nameR = './resources/right_image/right_img'+ str(i)+'.png'
		cv2.imwrite(nameR, frameR)
		print "%s , %s captured..." %(nameL, nameR)
		i = i+1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
capL.release()
capR.release()
cv2.destroyAllWindows()
