import disparity_functions as df
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import scipy.io as sio

Q = np.loadtxt('./matrix_dir/Q.txt') 
cameraMatrixL = np.loadtxt('./matrix_dir/stereo_cameraMatrixL.txt')
cameraMatrixR = np.loadtxt('./matrix_dir/stereo_cameraMatrixR.txt')
distCoeffsL = np.loadtxt('./matrix_dir/stereo_distCoeffsL.txt')
distCoeffsR = np.loadtxt('./matrix_dir/stereo_distCoeffsR.txt')
R1 = np.loadtxt('./matrix_dir/R1.txt')
R2 = np.loadtxt('./matrix_dir/R2.txt')
P1 = np.loadtxt('./matrix_dir/P1.txt')
P2 = np.loadtxt('./matrix_dir/P2.txt')

left_maps = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (640,480), cv2.CV_16SC2)
right_maps= cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (640,480), cv2.CV_16SC2)

capR = cv2.VideoCapture(1)
capL = cv2.VideoCapture(0)

while(True):
        # Capture frame-by-frame
        retL, imgL = capL.read()
        retR, imgR = capR.read()
	
	left_img = cv2.remap(imgL, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
	right_img = cv2.remap(imgR, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
	cv2.imshow('rectified img', left_img)

	left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
	right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

	disparity_img = df.disparity_with_filter(left_img_gray, right_img_gray, 160, 5)
	disparity_img = disparity_img/16

#	depth_img = (724*20)/disparity_img

	filtered_img = left_img
	filtered_img[:,:,0][(disparity_img < 140) | (disparity_img > 150)] = 0
	filtered_img[:,:,1][(disparity_img < 140) | (disparity_img > 150)] = 0
	filtered_img[:,:,2][(disparity_img < 140) | (disparity_img > 150)] = 0

	disparity_img = cv2.normalize(src=disparity_img, dst=disparity_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
	disparity_img = np.uint8(disparity_img)
#	depth_img = cv2.normalize(src=depth_img, dst=depth_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#	depth_img = np.uint8(depth_img)
	cv2.imshow('95cm to 105cm', filtered_img)
	cv2.imshow('disparity img', disparity_img)
#	cv2.imshow('depth img', depth_img)
#	cv2.imshow('histogram filtered', disparity_img_hist_filt)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
capL.release()
capR.release()
cv2.destroyAllWindows()



