import cv2
import numpy as np
import pickle
import time

objpoints = []
imgpointsL = []
imgpointsL = []

with open('./matrix_dir/objPoints.txt', 'rb') as f1:
	objpoints = pickle.load(f1)
with open('./matrix_dir/imgPoints_Left.txt', 'rb') as f2:
	imgpointsL = pickle.load(f2)
with open('./matrix_dir/imgPoints_Right.txt', 'rb') as f3:
	imgpointsR = pickle.load(f3)

M_L = np.loadtxt('./matrix_dir/cameraMatrix_left.txt')
M_R = np.loadtxt('./matrix_dir/cameraMatrix_right.txt')
distortion_L = np.loadtxt('./matrix_dir/distCoeff_left.txt')
distortion_R = np.loadtxt('./matrix_dir/distCoeff_right.txt')

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO#| cv2.CALIB_FIX_INTRINSIC# | cv2.CALIB_ZERO_TANGENT_DIST| cv2.CALIB_FIX_INTRINSIC# | cv2.CALIB_FIX_FOCAL_LENGTH
# | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |

stereocalib_retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR,M_L,distortion_L, M_R, distortion_R, (640,480), criteria = stereocalib_criteria, flags = stereocalib_flags)

print 'stereocalib_retval'
print stereocalib_retval
print 'cameraMatrixL'
print cameraMatrixL
np.savetxt('./matrix_dir/stereo_cameraMatrixL.txt', cameraMatrixL)
print 'distCoeffsL'
print distCoeffsL
np.savetxt('./matrix_dir/stereo_distCoeffsL.txt', distCoeffsL)
print 'cameraMatrixR'
print cameraMatrixR
np.savetxt('./matrix_dir/stereo_cameraMatrixR.txt', cameraMatrixR)
print 'distCoeffsR'
print distCoeffsR
np.savetxt('./matrix_dir/stereo_distCoeffsR.txt', distCoeffsR)
print 'R'
print R
np.savetxt('./matrix_dir/R.txt', R)
print 'T'
print T
np.savetxt('./matrix_dir/T.txt', T)
print 'E'
print E
np.savetxt('./matrix_dir/E.txt', E)
print 'F'
print F
np.savetxt('./matrix_dir/F.txt', F)


print "Rectifying Image...."

#stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]) -> R1, R2, P1, P2, Q, validPixROI1, validPixROI2


R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, (640,480), R, T)

print 'Q'
print Q
np.savetxt('./matrix_dir/Q.txt', Q)
np.savetxt('./matrix_dir/R1.txt', R1)
np.savetxt('./matrix_dir/R2.txt', R2)
np.savetxt('./matrix_dir/P1.txt', P1)
np.savetxt('./matrix_dir/P2.txt', P2)


#    {"initUndistortRectifyMap", (PyCFunction)pyopencv_cv_initUndistortRectifyMap, METH_VARARGS | METH_KEYWORDS, "initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2"},

left_maps = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (640,480), cv2.CV_16SC2)
right_maps= cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (640,480), cv2.CV_16SC2)

#    {"remap", (PyCFunction)pyopencv_cv_remap, METH_VARARGS | METH_KEYWORDS, "remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst"},
#new_imgL = cv2.imread('./resources/left_image/left_img18.png')
#new_imgR = cv2.imread('./resources/right_image/right_img18.png')
new_imgL = cv2.imread('./my_photo-1.jpg')
new_imgR = cv2.imread('./my_photo-2.jpg')
print '**********************************'
print 'camera Matrix'
print cameraMatrixL
print 'M_L'
print M_L

start_time = time.time()
left_img_remap = cv2.remap(new_imgL, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
right_img_remap = cv2.remap(new_imgR, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
print 'Time for both remap: ', time.time()-start_time
#print type(left_img_remap), left_img_remap[100][100][2], type(left_img_remap[100][100][2]), left_img_remap.shape

#undist_L = cv2.undistortPoints(new_imgL, cameraMatrixL, distCoeffsL, R1, P1)
cv2.imwrite('left_rectified.png', left_img_remap)
cv2.imwrite('right_rectified.png', right_img_remap)
#cv2.waitKey(0)


