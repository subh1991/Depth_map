import numpy as np
import cv2
import disparity_functions as df
import sys
from matplotlib import pyplot as plt

vid_name = sys.argv[1]
cap = cv2.VideoCapture(vid_name)
ret, frame = cap.read()
height, width, depth = frame.shape
print height, width, depth 

Q = np.float32([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

############################################################################
while(cap.isOpened()):
	ret, frame = cap.read()
	
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	left_img = frame[:,0:(width/2)]
	right_img = frame[:,(width/2):width]
	disparity_img1 = df.disparity_with_filter(left_img, right_img, 16, 5)
	disparity_img2, max_val = df.disparity_without_filter(left_img, right_img, 64, 9)

	disparity_img1 = cv2.normalize(src=disparity_img1, dst=disparity_img1, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
	disparity_img1 = np.uint8(disparity_img1)
	depthMapImg = cv2.reprojectImageTo3D(disparity_img1, Q)	

#	i = i+1
#	if (i==30):
#		cv2.imwrite('car_depth.png', depthMapImg)
#		cv2.imwrite('car.png', left_img)
#
#	print max_val/16, np.amax(disparity_img2), np.amax(disparity_img1)	#max_val/16 gives disparities in pixel
#
#	hist1 = df.draw_histogram(disparity_img1)
#	hist2 = df.draw_histogram(disparity_img2)
#	hist3 = df.draw_histogram(left_img)
#
#	cv2.imshow('frame1', left_img)
#	cv2.imshow('frame2', hist3)
	cv2.imshow('frame3', disparity_img1)
#	cv2.imshow('frame4', hist1)
#	cv2.imshow('frame5', disparity_img2)
#	cv2.imshow('frame6', hist2)
	cv2.imshow('frame6', depthMapImg)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
















