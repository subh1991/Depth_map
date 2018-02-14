import disparity_functions as df
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import scipy.io as sio
import math

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

# mouse callback function
def single_click(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print xyz[y][x][0], xyz[y][x][1], xyz[y][x][2], disparity_img[y][x], x, y
# Create a window and bind the function to window
cv2.namedWindow('image to click')
cv2.setMouseCallback('image to click',single_click)

xyz_init = np.zeros((480,640,3))
for x in range(640):
        for y in range(480):
                xyz_init[y][x][0] = x - 320
                xyz_init[y][x][1] = 240 - y

angle_adjust = math.cos(math.atan2(26,37))

capR = cv2.VideoCapture(1)
capL = cv2.VideoCapture(0)

while(True):
        # Capture frame-by-frame
	retL, imgL = capL.read()
	retR, imgR = capR.read()
	
	left_img = cv2.remap(imgL, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
	right_img = cv2.remap(imgR, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
#	cv2.imshow('rectified img', left_img)

	left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
	right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
	left_img_hist_eq = cv2.equalizeHist(left_img_gray)
	right_img_hist_eq = cv2.equalizeHist(right_img_gray)

#	disparity_img = df.disparity_with_filter(left_img_gray, right_img_gray, 160, 5)
	disparity_img = df.disparity_with_filter(left_img, right_img, 160, 5)
#	disparity_img = df.disparity_with_filter(left_img_hist_eq, right_img_hist_eq, 160, 3)

	disparity_img = disparity_img/16
	disparity_img[disparity_img < 1] = 1

	filtered_img = np.copy(left_img)
	filtered_img[:,:,:][(disparity_img <  140) | (disparity_img > 150)] = 0
#	filtered_img[:,:,1][(disparity_img < 140) | (disparity_img > 150)] = 0
#	filtered_img[:,:,2][(disparity_img < 140) | (disparity_img > 150)] = 0

	depth_img = ((724*20)/disparity_img) * angle_adjust 
	xyz = np.copy(xyz_init)
#	print np.amin(xyz_init), np.amax(xyz_init)
	xyz[:,:,0] = np.multiply(xyz[:,:,0], depth_img)
	xyz[:,:,0] = xyz[:,:,0] / 724
	xyz[:,:,1] = np.multiply(xyz[:,:,1], depth_img)
	xyz[:,:,1] = xyz[:,:,1] / 724
	xyz[:,:,2] = depth_img



	disparity_img = cv2.normalize(src=disparity_img, dst=disparity_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
	disparity_img = np.uint8(disparity_img)
#	depth_img = cv2.normalize(src=depth_img, dst=depth_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#	depth_img = np.uint8(depth_img)
	cv2.imshow('image to click', left_img)
	cv2.imshow('filtered image', filtered_img)
	cv2.imshow('disparity img', disparity_img)
#	cv2.imshow('depth img', depth_img)
#	cv2.imshow('histogram filtered', disparity_img_hist_filt)
#	cv2.imshow('image to click', left_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
capL.release()
capR.release()
cv2.destroyAllWindows()



