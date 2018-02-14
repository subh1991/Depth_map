import disparity_functions as df
import numpy as np
import cv2
import sys


Q = np.float32([[1, 0, 0, 4.95838611e+02],
                [0, 1, 0, 2.24740481e+01],
                [0, 0, 0, 4.23789559e+00],
                [0, 0, 4.66947240e-03, 0]])

left_img = cv2.imread('resources/left_image/left_img6.png')
right_img = cv2.imread('resources/right_image/right_img6.png')

disparity_img1 = df.disparity_with_filter(left_img, right_img, 32, 5)
disparity_img2, max_val = df.disparity_without_filter(left_img, right_img, 32, 9)

disparity_img1 = cv2.normalize(src=disparity_img1, dst=disparity_img1, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
disparity_img1 = np.uint8(disparity_img1)

disparity_img2 = cv2.normalize(src=disparity_img2, dst=disparity_img2, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
disparity_img2 = np.uint8(disparity_img2)

cv2.imshow('with filter', disparity_img1)
cv2.imshow('without filter', disparity_img2)
cv2.waitKey(0)
