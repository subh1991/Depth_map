import vtk
import numpy as np
#from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import cv2
 
print('loading images...')
imgL =cv2.imread('/home/subhasis/Desktop/chair0.png') 
imgR = cv2.imread('/home/subhasis/Desktop/chair1.png')
#imgL = cv2.imread('left_image.bmp')  # downscale images for faster processing
#imgR = cv2.imread('right_image.bmp')
Q = np.identity(4)

h, w, d = imgL.shape

#imgL = cv2.imread('chair0.png')  # downscale images for faster processing
#imgR = cv2.imread('chair1.png')
#imgL = cv2.imread('chair0.png')  
#imgR = cv2.imread('chair1.png')
#imgL = cv2.imread('IMG_20180101_155749.jpg')
#imgR = cv2.imread('IMG_20180101_155751.jpg')
#imgL = cv2.imread('IMG_20180101_160103.jpg')
#imgR = cv2.imread('IMG_20180101_160101.jpg')
#imgL = cv2.pyrDown(imgL)
#imgR = cv2.pyrDown(imgR)
 
# SGBM Parameters -----------------
window_size = 5                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*4,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
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
 
print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
 
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)

depthMapImg = cv2.reprojectImageTo3D(filteredImg, Q)
cv2.imwrite('chair_depth.png', depthMapImg)
print depthMapImg.shape
depthMapImg = depthMapImg.reshape((h*w, 3), order = 'F')
print depthMapImg.shape



class VtkPointCloud:

#    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=6666666):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        #mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetColor(0, 0, 0)
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            #self.vtkDepth.InsertNextValue(point[0])
            self.vtkDepth.InsertNextTupleValue([5])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

pointCloud = VtkPointCloud()

for k in xrange(h * w):
    point = depthMapImg[k,:]
    pointCloud.addPoint(point)
pointCloud.addPoint([0,0,0])
pointCloud.addPoint([0,0,0])
pointCloud.addPoint([0,0,0])
pointCloud.addPoint([0,0,0])

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(pointCloud.vtkActor)
renderer.SetBackground(.2, .3, .4)
renderer.ResetCamera()

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Begin Interaction
renderWindow.Render()
renderWindowInteractor.Start()


#plt.hist(filteredImg.ravel(),256,[0,256])
#plt.show()

#filteredImg[filteredImg < 53] = 0 
#filteredImg[filteredImg > 100] = 0 

#plt.imshow(filteredImg, 'gray')
#plt.imshow(imgR, 'gray')
#plt.show()

#cv2.imshow('Disparity Map', filteredImg)
cv2.waitKey(0)
#cv2.destroyAllWindows()





