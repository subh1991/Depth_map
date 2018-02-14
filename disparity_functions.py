import numpy as np
import cv2

def disparity_with_filter(imgL, imgR, MAX_no_DISP, BLOCK_SIZE):

        # SGBM Parameters -----------------
        window_size = 5                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities= MAX_no_DISP,             # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize=BLOCK_SIZE,
                P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=0,
                speckleRange=2,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2
        visual_multiplier = 1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
#	filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#	filteredImg = np.uint8(filteredImg)
	
	return filteredImg 

def disparity_without_filter(imgL, imgR, NUM_of_DISPARITY, BLOCK_SIZE):
	stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=NUM_of_DISPARITY, blockSize=BLOCK_SIZE)
	disparity = stereo.compute(imgL,imgR)
#	max_val = np.amax(disparity)
#	disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#	disparity = np.uint8(disparity )
	#disparity = disparity - np.amin(disparity)
	#disparity = disparity.astype(np.uint8)

	
	return disparity


def show_histogram(img):
	img = cv2.imread('left_image.bmp')
#	img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
	height, width, depth = img.shape
	h = np.zeros((width,height,depth))
	b,g,r = cv2.split(img)
	bins = np.arange(256).reshape(256,1)
	color = [ (255,0,0),(0,255,0),(0,0,255) ]
	
	for item,col in zip([b,g,r],color):
		hist_item = cv2.calcHist([img],[0],None,[256],[0,255])
		cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
		hist=np.int32(np.around(hist_item))
		pts = np.column_stack((bins,hist))
		cv2.polylines(h,[pts],False,(255,0,0))
	
	h=np.flipud(h)
	
	return h	



def draw_histogram(img):
        if len(img.shape) == 3:
                height, width, depth = img.shape

                h = np.zeros((height, width, depth))

                bins = np.arange(256).reshape(256,1)
                color = [ (255,0,0),(0,255,0),(0,0,255) ]

                for ch, col in enumerate(color):
                        hist_item = cv2.calcHist([img],[ch],None,[256],[0,255])
                        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
                        hist=np.int32(np.around(hist_item))
                        pts = np.column_stack((bins,hist))
                        cv2.polylines(h,[pts],False,col)

                h=np.flipud(h)
        else:
                height, width = img.shape

                h = np.zeros((height, width, 1))

                bins = np.arange(256).reshape(256,1)

                hist_item = cv2.calcHist([img],[0],None,[256],[0,255])
                cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
                hist=np.int32(np.around(hist_item))
                pts = np.column_stack((bins,hist))
                cv2.polylines(h,[pts],False,(255,0,0))

                h=np.flipud(h)


        return h















