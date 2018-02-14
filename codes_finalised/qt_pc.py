from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import disparity_functions as df
import numpy as np
import cv2
import sys

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

capR = cv2.VideoCapture(0)
capL = cv2.VideoCapture(1)
ret, frame = capR.read()
height, width, depth = frame.shape
print height, width, depth

Q = np.float32([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

def update():
	colors = ((1.0, 1.0, 1.0, 1.0))
	
	
#	out = cv2.imread('car_depth.png')
#	out = out.astype(float)
#	out = out/16
#	colors = cv2.imread('car.png')
#	colors = colors.astype(float)
#	colors = colors/255
#	ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	retL, left_img = capL.read()
        retR, right_img = capR.read()

#        left_img = frame[:,0:(width/2)]
#        right_img = frame[:,(width/2):width]
        disparity_img1 = df.disparity_with_filter(left_img, right_img, 32, 5)
        disparity_img2, max_val = df.disparity_without_filter(left_img, right_img, 64, 9)

        disparity_img1 = cv2.normalize(src=disparity_img1, dst=disparity_img1, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        disparity_img1 = np.uint8(disparity_img1)
        depthMapImg = cv2.reprojectImageTo3D(disparity_img1, Q)

	colors = left_img.astype(float)
	colors = colors/255
	sp2.setData(pos=np.array(depthMapImg, dtype=np.float64), color=colors, size=2)



t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

sys.exit(0)
