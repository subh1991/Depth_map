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
        print xyz[x][y][0], xyz[x][y][1], xyz[x][y][2] 
# Create a window and bind the function to window
cv2.namedWindow('left image')
cv2.setMouseCallback('left image',single_click)

nameL = sys.argv[1]
nameR = sys.argv[2]

imgL = cv2.imread(nameL)
imgR = cv2.imread(nameR)

left_img = cv2.remap(imgL, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
right_img = cv2.remap(imgR, right_maps[0], right_maps[1], cv2.INTER_LINEAR)

left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

disparity_img = df.disparity_with_filter(left_img_gray, right_img_gray, 160, 5)
#disparity_img = disparity_img/16
disparity_img[disparity_img < 1] = 1

depthMapImg = cv2.reprojectImageTo3D(disparity_img, Q)
depthMapImg[depthMapImg == (np.inf)] = 0
depthMapImg[depthMapImg == (-np.inf)] = 0


#cv2.imshow('to_matlab_left_image.png', left_img)
#cv2.imshow('to_matlab_disparity.png', disparity_img)
#cv2.imshow('depth', depth_img)
print np.amin(depthMapImg)
print np.amax(depthMapImg)
print type(depthMapImg)
print depthMapImg.shape
print np.amin(depthMapImg[:,:,0])
print np.amax(depthMapImg[:,:,0])
print np.amin(depthMapImg[:,:,1])
print np.amax(depthMapImg[:,:,1])
print np.amin(depthMapImg[:,:,2])
print np.amax(depthMapImg[:,:,2])

cv2.imwrite('to_matlab_left_image.png', left_img)
depthMapImg[(depthMapImg < 0) | (depthMapImg > 255)] = 0
cv2.imwrite('to_matlab_depth.png',depthMapImg)
#print xyz
#cv2.imshow('xyz', xyz)
cv2.waitKey(0)






















