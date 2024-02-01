import cv2 as cv
import numpy as np
from mylib import Scale, FovToFocalLength, ElapsedTime, GetDepthFromUint24DepthMap, DepthToDisparity, DisparityToDepth, FindEffectiveDisparity, CustomDisparityMap, ConvertDisparityToDepth
from matplotlib import pyplot as plt
from time import time
import math


# Zadanie 1
# Odczytaj wartości z mapy głębi z pliku PNG. Przyjmij następujące parametry:
# Maksymalna odległość mapy głębi: 1000 m
# Minimalna odległość mapy głębi: 0V
# Kodowanie 24 bity (uint24)
def task1():
    MAX_DEPTH = 1000 #m
    MIN_DEPTH = 0

    depthMap = cv.imread('t0z1a/depth.png')
    depthMap = cv.cvtColor(depthMap, cv.COLOR_BGR2RGB)
    depthMap, SCALE = Scale(depthMap, 250)
    #MAX_DEPTH *= SCALE

    depth = GetDepthFromUint24DepthMap(depthMap, X = 100, Y = 100, max_depth = MAX_DEPTH)

    plt.subplot(1, 1, 1)
    plt.title('Deepth Map')
    plt.axis('off') 
    plt.imshow(depthMap)
    plt.show()


# Zadanie 2
# Wyznacz mapę rozbieżności (disparity map) na podstawie mapy głębi (depth map). Przyjmij poniższe założenia.
# Kanoniczny układ kamer stereo
# Poziome pole widzenia kamery (FoV): 120 stopni
# Odległość pomiędzy kamerami (baseline): 0,6 m
# Maksymalna odległość mapy głębi: 1000 m
# Minimalna odległość mapy głębi: 0
# 24 bitowe kodowanie (uint24)
# Wynikową mapę rozbieżności zapisz w postaci 8 bitowego obrazu PNG.
def task2():
    FOV = 120 #degrees
    BASELINE = 0.6 #m
    MAX_DEPTH = 1000 #m
    MIN_DEPTH = 0 #m
    depthMap = cv.imread('t0z1a/depth.png')
    depthMap = cv.cvtColor(depthMap, cv.COLOR_BGR2RGB)

    height, width = depthMap.shape[:2]
    FOCAL_LENGTH = FovToFocalLength(width, FOV)

    depthMap, SCALE = Scale(depthMap, 250)
    #MAX_DEPTH *= SCALE

    disparityMap = DepthToDisparity(depthMap, BASELINE, FOCAL_LENGTH)
    disparityMap_gray = cv.cvtColor(disparityMap, cv.COLOR_BGR2GRAY)
    plt.subplot(1, 1, 1)
    plt.title('Disparity Map')
    plt.axis('off') 
    plt.imshow(disparityMap_gray, cmap='gray')
    plt.show()
    cv.imwrite('t0z1a/disparity_map_8bit.png', disparityMap_gray)

    




def task3():


    cam0 = np.array([[1733.74, 0, 792.27],      #fx 0 cx
                    [0, 1733.74, 541.89],       #0 fy cy
                    [0, 0, 1]])
    doffs=0
    BASELINE=536.62 #mm
    BASELINE/=1000 #m

    width=1920
    height=1080
    ndisp=170
    vmin=55
    vmax=142

    fx = 1733.74 

    FOV = 2 * math.atan(width / (2 * fx))
    FOV = math.degrees(FOV)
    #fov_vertical = 2 * math.atan(height / (2 * fy))
    print(f'FoV: {FOV:.2f}')
    FOCAL_LENGTH = FovToFocalLength(width, FOV)

    disparity_gt = cv.imread('t0z3a/disp0.pfm', cv.IMREAD_UNCHANGED)
    disparity_gt = np.asarray(disparity_gt)
    disparity_gt = disparity_gt / 256

    depth_gt = DisparityToDepth(disparity_gt, BASELINE, FOCAL_LENGTH)
    




    left_image = cv.imread('t0z3a/im0.png')
    left_image = cv.cvtColor(left_image, cv.COLOR_BGR2RGB)
    #left_image, _ = Scale(left_image, 200)

    right_image = cv.imread('t0z3a/im1.png')
    right_image = cv.cvtColor(right_image, cv.COLOR_BGR2RGB)
    #right_image, _ = Scale(right_image, 200)

    MAX_DISPARITY = FindEffectiveDisparity(no_samples = 1000, 
                                        leftImg = left_image, 
                                        rightImg= right_image,
                                        searchWindow = 100)
    
    block_size = 11
    stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=ndisp,
    blockSize=block_size,
    uniquenessRatio=5,
    speckleWindowSize=200,
    speckleRange=2,
    disp12MaxDiff=0,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)

    # Wyznacz mapę rozbieżności
    disparity = stereo.compute(left_image, right_image)

    disparity = cv.normalize(disparity, disparity, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
    disparity = np.uint8(disparity)
    #disparity = 255 - disparity
    disparity[disparity == 0] = 255

    # Skonwertuj mapę rozbieżności do wartości głębi
    depth = DisparityToDepth(disparity, BASELINE, FOCAL_LENGTH)
    depth[depth == 255] = 0
    depth = cv.normalize(depth, depth, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
    depth = np.uint8(depth)

   
   

    # Wartości dla punktu o współrzędnych X=588, Y=755
    x, y = 300, 300


    print(f"depth_gt = {depth_gt[y, x]}m")
    print(f"depth = {disparity[y, x]}m")
    print(f"disparity_gt = {depth_gt[y, x]} ")
    print(f"disparity = {depth[y, x]} ")
    

    plt.subplot(2, 2, 1)
    plt.title('Disparity Map')
    plt.axis('off') 
    plt.imshow(disparity, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Deepth Map')
    plt.axis('off') 
    plt.imshow(depth, cmap='plasma')

    plt.subplot(2, 2, 3)
    plt.title('Groundtruth Disparity Map')
    plt.axis('off') 
    plt.imshow(disparity_gt, cmap='gray')
  

    plt.subplot(2, 2, 4)
    plt.title('Groundtruthv Deepth Map')
    plt.axis('off') 
    plt.imshow(depth_gt, cmap='plasma')

    plt.show()


    cv.imwrite('t0z3a/disparity.png', disparity)
    cv.imwrite('t0z3a/depth.png', depth)



def task4():
    SEARCH_WINDOW = 10

    DOFFS = 170.681 #Depth of Field Scale - skalę głębi ostrości.
    BASELINE = 178.232 # odległość między dwiema kamerami
    F = 2945.377 

    #mode = cv.IMREAD_GRAYSCALE
    leftImage = cv.imread('t0z1a/left.png', cv.IMREAD_GRAYSCALE)
    #leftImage = cv.cvtColor(leftImage, cv.COLOR_BGR2RGB)
    leftImage, _ = Scale(leftImage, 250)

    rightImage =  cv.imread('t0z1a/right.png', cv.IMREAD_GRAYSCALE)
    #rightImage = cv.cvtColor(rightImage, cv.COLOR_BGR2RGB)
    rightImage, _ = Scale(rightImage, 250)



    ##----




    max_disparity = FindEffectiveDisparity(no_samples = 1000, 
                                        leftImg = leftImage, 
                                        rightImg= rightImage,
                                        searchWindow = SEARCH_WINDOW)


    disparity_map = CustomDisparityMap(leftImage, rightImage, SEARCH_WINDOW, max_disparity)

    parameters = (BASELINE, F, DOFFS) 
    custom_deepth_map = ConvertDisparityToDepth(disparity_map.astype(np.float32) / 16, parameters)



    plt.subplot(1, 2, 1)
    plt.title('Custom Disparity Map')
    plt.axis('off') 
    plt.imshow(disparity_map, cmap='turbo')

    plt.subplot(1, 2, 2)
    plt.title('Custom Deepth Map')
    plt.axis('off') 
    plt.imshow(custom_deepth_map, cmap='plasma')

    plt.show()


    
