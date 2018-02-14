import numpy as np
import cv2
import glob
import pickle
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*7.1

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = []

imagesL = glob.glob('./resources/left_image/*.png')

for fnameL in imagesL:
	imgL = cv2.imread(fnameL)
	grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
#	print fname
	fnameR = fnameL.replace('left', 'right')
	imgR = cv2.imread(fnameR)
#	print fnameR, imgR.shape
	grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
	
	# Find the chess board corners
	retL, cornersL = cv2.findChessboardCorners(grayL, (8,6),None)
	retR, cornersR = cv2.findChessboardCorners(grayR, (8,6),None)
#	print ret
	
	# If found, add object points, image points (after refining them)
	if (retL == True) & (retR == True):
		objpoints.append(objp)
		
		corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
		imgpointsL.append(corners2L)
		
		corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
		imgpointsR.append(corners2R)

		# Draw and display the corners
		imgL = cv2.drawChessboardCorners(imgL, (8,6), corners2L,retL)
		cv2.imshow('imgL',imgL)
		imgR = cv2.drawChessboardCorners(imgR, (8,6), corners2R,retR)
		cv2.imshow('imgR',imgR)

		cv2.waitKey(100)
	else:
		print fnameL, fnameR
cv2.destroyAllWindows()

print 'calibrating... Please wait...'

retL, M_L, distortion_L, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1],None,None)
retR, M_R, distortion_R, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1],None,None)

with open('./matrix_dir/objPoints.txt', 'wb') as f1:
    pickle.dump(objpoints, f1)
with open('./matrix_dir/imgPoints_Left.txt', 'wb') as f2:
    pickle.dump(imgpointsL, f2)
with open('./matrix_dir/imgPoints_Right.txt', 'wb') as f3:
    pickle.dump(imgpointsR, f3)

print imgpointsR
np.savetxt('./matrix_dir/cameraMatrix_left.txt',M_L)
np.savetxt('./matrix_dir/cameraMatrix_right.txt',M_R)
np.savetxt('./matrix_dir/distCoeff_left.txt', distortion_L)
np.savetxt('./matrix_dir/distCoeff_right.txt', distortion_R)

