import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import random
import copy
import open3d as o3d

def DisparityMap(left_image, right_image, search_window, max_disparity, parameters):
    assert left_image.shape == right_image.shape

    height, width  = left_image.shape
    n = (width - 2*search_window) * (height - 2*search_window)
    disparity_map = np.zeros((height, width))
    #deepth_map = np.zeros((height, width))
    i = 0
    start_timestamp = time.time()
    for lx in range(search_window, width - search_window):
        for ly in range(search_window, height - search_window):
            reference_block = ExtractBlock(left_image, search_window, lx, ly)
            (_, _, disparity_map[ly, lx]) = FindMatchBlock(right_image, reference_block, lx, ly, search_window, max_disparity)
            #deepth_map[ly, lx] = ConvertDisparityToDepth(parameters, disparity_map[ly, lx]) - it causes a problem with the frame
            i +=  ProgressBar(i, n, f'{ElapsedTime(start_timestamp):.2f}s')
    
    #deepth_map = ConvertDisparityToDepth(parameters, disparity_map)[search_window:height - search_window, search_window:width - search_window]
    disparity_map = disparity_map[search_window:height - search_window, search_window:width - search_window]
    return (disparity_map, ElapsedTime(start_timestamp))

def ExtractBlock(image, searchWindow, x, y):
    (yStart, yStop) = (y - searchWindow, y + searchWindow)
    (xStart, xStop) = (x - searchWindow, x + searchWindow)
    return image[yStart : yStop,  xStart : xStop]

def FindMatchBlock(image, referenceBlock, lx, ly, searchWindow, max_disparity):
    (startX, endX) = (max(np.int32(lx) - max_disparity, searchWindow), lx)
    (rx, ry) = (startX, ly)
    matchBlock = ExtractBlock(image, searchWindow, startX, ly)
    distance = np.sum((referenceBlock - matchBlock) ** 2)

    for x in range(startX, endX):
        block = ExtractBlock(image, searchWindow, x, ly)
        d = np.sum((referenceBlock - block) ** 2)
        if d < distance:
            rx = x
            distance = d
            matchBlock = block
    disparity = abs(np.int32(rx) - np.int32(lx))
    return (rx, ry, disparity)

def ConvertDisparityToDepth(disparity_map, parameters):
    (baseline, focal_length, doffs) = parameters
    return (baseline * focal_length) / (disparity_map + doffs)

def ConvertDeepthToCloud(deepth_map, image, parameters):
    (baseline, f, doffs) = parameters
    h, w = deepth_map.shape[:2]
    Q = np.float32([[1, 0, 0, -0.5*w],
    [0,-1, 0, 0.5*h], 
    [0, 0, 0, -f], 
    [0, 0, 1 / baseline, 0]])
    points = cv.reprojectImageTo3D(deepth_map, Q)
    colors = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mask = deepth_map > deepth_map.min()
    return points[mask], colors[mask]

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        for v, c in zip(verts, colors):
            f.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))


#------------------------------------------------------------------
# Tools
#------------------------------------------------------------------
def ProgressBar(iteration, total, info = None, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    if info is not None:
        sys.stdout.write(f'\r[{arrow}{spaces}] - {iteration} / {total} ({(progress * 100.0):.2f}%) [ {info} ]')
    else:
        sys.stdout.write(f'\r[{arrow}{spaces}] - {iteration} / {total} ({(progress * 100.0):.2f}%) ')
    sys.stdout.flush()
    return 1

def ElapsedTime(from_timestamp):
    return time.time() - from_timestamp

def Scale(image, scale):
    height, width = image.shape[:2]
    width = int(width * scale)
    height = int(height * scale)
    return cv.resize(image, (width, height), interpolation = cv.INTER_AREA)

def TestRandomPoints(leftImage, rightImage, searchWindow, max_disparity):
    leftImage2 = copy.copy(leftImage)
    rightImage2 = copy.copy(rightImage)
    height, width = leftImage.shape

    left_coordinates = GenerateRandomCoordinates(5, width, height, searchWindow)
    for point in left_coordinates:
        (x, y) = point
        referenceBlock = ExtractBlock(leftImage, searchWindow, x, y)
        (x2, y2, _) = FindMatchBlock(rightImage, referenceBlock, x, y, searchWindow, max_disparity)
        cv.circle(leftImage2, (x, y), searchWindow, color=(0, 255, 255), thickness=2)
        cv.circle(rightImage2, (x2, y2), searchWindow, color=(0, 255, 255), thickness=2)

    combined_image = np.concatenate((leftImage, rightImage), axis=1)
    combined_image2 = np.concatenate((leftImage2, rightImage2), axis=1)
    combined_image3 = np.concatenate((combined_image, combined_image2), axis=0)
    cv.imshow('', combined_image3)
    cv.waitKey(0)
    cv.destroyAllWindows()

def GenerateRandomCoordinates(num_points, width, height, searchWindow):
    random_coordinates = []
    for _ in range(num_points):
        x = random.randint(searchWindow, width - searchWindow)
        y = random.randint(searchWindow, height - searchWindow)
        random_coordinates.append((x, y))
    return random_coordinates

def visualize_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pcd])
