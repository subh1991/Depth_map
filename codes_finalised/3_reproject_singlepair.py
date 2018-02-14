import disparity_functions as df
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import scipy.io as sio

Q = np.float32([[1, 0, 0, -3.77753695e+02],
                [0, 1, 0, -2.06903945e+02],
                [0, 0, 0,  6.74158207e+02],
                [0, 0, 4.97033865e-02, 0]])

left_img = cv2.imread(sys.argv[1])
right_img = cv2.imread(sys.argv[2])

#print sys.argv[0]
#print left_img.shape

disparity_img1 = df.disparity_with_filter(left_img, right_img, 160, 5)
#disparity_img2, max_val = df.disparity_without_filter(left_img, right_img, 32, 9)

#disparity_img1 = cv2.normalize(src=disparity_img1, dst=disparity_img1, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#disparity_img1 = np.uint8(disparity_img1)

print np.amax(disparity_img1)
print np.amin(disparity_img1)
disparity_img1 = disparity_img1/16
print np.amax(disparity_img1)
print np.amin(disparity_img1)
plt.hist(disparity_img1.ravel(),256,[0,256])
plt.show()
disparity_img1[disparity_img1 < 140] = 0 
disparity_img1[disparity_img1 > 150] = 0 
plt.imshow(disparity_img1, 'gray')
plt.show()
left_img[:,:,0][(disparity_img1 < 145) | (disparity_img1 > 146)] = 0
left_img[:,:,1][(disparity_img1 < 145) | (disparity_img1 > 146)] = 0
left_img[:,:,2][(disparity_img1 < 145) | (disparity_img1 > 146)] = 0
cv2.imshow('1.5 meter', left_img)
cv2.waitKey(0)

#print 'disparity img 155 465'
#print disparity_img1[210][373]


depthMapImg = cv2.reprojectImageTo3D(disparity_img1/16, Q)
depthMapImg[depthMapImg == (np.inf)] = 0
depthMapImg[depthMapImg == (-np.inf)] = 0

#depthMapImg = cv2.normalize(src=depthMapImg, dst=depthMapImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#depthMapImg = np.uint8(depthMapImg)

#print np.amin(depthMapImg)
#print np.amax(depthMapImg)
#print 'depth reprojected img 155 465'
#print depthMapImg[210][373][0], depthMapImg[210][373][1],depthMapImg[155][465][2]

#cv2.imshow('depth map', depthMapImg)
#cv2.waitKey(0)
#cv2.imwrite('depth_of_image.png', depthMapImg)


#print depthMapImg.shape
#print depthMapImg[193][600][0]
#print depthMapImg[193][600][1]
#print depthMapImg[193][600][2]

#cv2.imwrite('disparity_map.png', disparity_img1/16)
#cv2.imshow('without filter', disparity_img2)
#cv2.imshow(' depth map',depthMapImg)
#cv2.imwrite('depthmap.png', depthMapImg)
#cv2.waitKey(0)
