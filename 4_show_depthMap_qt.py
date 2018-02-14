from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import disparity_functions as df
import numpy as np
import cv2
import sys
import math

#QT app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

#initialize some points data
pos = np.zeros((1,3))

sp2 = gl.GLScatterPlotItem(pos=pos)
w.addItem(sp2)

#capR = cv2.VideoCapture(0)
#capL = cv2.VideoCapture(1)
#ret, frame = capR.read()
#height, width, depth = frame.shape
#print height, width, depth

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

xyz_init = np.zeros((480,640,3))
dst = np.zeros((480, 640))
for x in range(640):
        for y in range(480):
                xyz_init[y][x][0] = x
                xyz_init[y][x][1] = y

capR = cv2.VideoCapture(1)
capL = cv2.VideoCapture(0)

def update():
	colors = ((1.0, 1.0, 1.0, 1.0))

#	retL, imgL = capL.read()
#        retR, imgR = capR.read()
	
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
        disparity_img[disparity_img < 50] = 50
	
	depth_img = ((724*20)/disparity_img) * math.cos(math.atan2(26,37))
	
	xyz = np.copy(xyz_init)
	xyz[:,:,0] = np.multiply(xyz[:,:,0], depth_img)
	xyz[:,:,0] = xyz[:,:,0] / 724
	xyz[:,:,1] = np.multiply(xyz[:,:,1], depth_img)
	xyz[:,:,1] = xyz[:,:,1] / 724
	xyz[:,:,2] = depth_img
	
	
	colors = left_img.astype(float)
	colors = colors/255
	temp_clr = colors[:,:,0]
	colors[:,:,0] = colors[:,:,2]
	colors[:,:,2] = temp_clr
	sp2.setData(pos=np.array(xyz, dtype=np.float64), color=colors, size=2)



t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

sys.exit(0)
