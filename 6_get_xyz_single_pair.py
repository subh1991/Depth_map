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

xyz = np.zeros((480,640,3))
dst = np.zeros((480, 640))
for x in range(640):
	for y in range(480):
		xyz[y][x][0] = x
		xyz[y][x][1] = y

#xyz[:,:,1] = cv2.normalize(src=xyz[:,:,1], dst=dst, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#dst = np.uint8(dst)

#cv2.imshow('Y', dst)
#cv2.waitKey(0)

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
disparity_img = disparity_img/16
disparity_img[disparity_img < 1] = 1

filtered_img = np.copy(left_img)
filtered_img[:,:,:][disparity_img < 57] = 0
#       filtered_img[:,:,1][(disparity_img < 140) | (disparity_img > 150)] = 0


#depth_img = (724*20)/disparity_img
depth_img = ((724*20)/disparity_img) * math.cos(math.atan2(26,37))

xyz[:,:,0] = np.multiply(xyz[:,:,0], depth_img)
xyz[:,:,0] = xyz[:,:,0] / 724
xyz[:,:,1] = np.multiply(xyz[:,:,1], depth_img)
xyz[:,:,1] = xyz[:,:,1] / 724
xyz[:,:,2] = depth_img


cv2.imwrite('to_matlab_left_image.png', left_img)
cv2.imshow('filtered image', filtered_img)
cv2.imshow('to_matlab_disparity.png', disparity_img)
cv2.imshow('depth', depth_img)
print np.amin(disparity_img)
print np.amax(disparity_img)
#print xyz
#cv2.imshow('xyz', xyz)
cv2.waitKey(0)






















