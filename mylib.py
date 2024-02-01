import copy
import cv2 as cv
import numpy as np
import time
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter
import math

def Scale(image, nHeight):
    height, width = image.shape[:2]
    scale = nHeight / height
    nWidth = int(scale * width)

    return (cv.resize(image, (nWidth, nHeight), interpolation = cv.INTER_AREA), scale)


def ExtractBlock(image, searchWindow, x, y):
    height, width = image.shape[:2]
    (yStart, yStop) = (max(y - searchWindow, 0), min(y + searchWindow, height))
    (xStart, xStop) = (max(x - searchWindow, 0), min(x + searchWindow, width))
    return copy.copy(image[yStart : yStop,  xStart : xStop])


def FindMatchBlock(image, referenceBlock, left_x, left_y, searchWindow, max_disparity):
    start_x = max(np.int32(left_x) - max_disparity, searchWindow)
    end_x = left_x

    (right_x, right_y) = (start_x, left_y)
    
    matchBlock = ExtractBlock(image, searchWindow, start_x, left_y)
    distance = np.sum((referenceBlock - matchBlock) ** 2)

    for x in range(start_x, end_x):
        block = ExtractBlock(image, searchWindow, x, left_y)
        d = np.sum((referenceBlock - block) ** 2)
        if d < distance:
            right_x = x
            distance = d
            matchBlock = block
    disparity = abs(np.int32(right_x) - np.int32(left_x))
    return (matchBlock, right_x, right_y, disparity)


def GenerateRandomCoordinates(num_points, width, height, searchWindow):
    random_coordinates = []
    for _ in range(num_points):
        x = random.randint(searchWindow, width - searchWindow)
        y = random.randint(searchWindow, height - searchWindow)
        random_coordinates.append((x, y))
    return random_coordinates

def ElapsedTime(from_timestamp):
    return time.time() - from_timestamp


def FindEffectiveDisparity(no_samples, leftImg, rightImg, searchWindow):
    height, width = leftImg.shape[:2]
    max_disparity = width # TU MOGĄ BYĆ PROBLEMY

    start_timestamp = time.time()
    disp = []
    left_coordinates = GenerateRandomCoordinates(no_samples, width, height, searchWindow)
    for point in tqdm(left_coordinates):
        (left_x, left_y) = point
        leftImageExtracted = ExtractBlock(leftImg, searchWindow, left_x, left_y)
        (rightImageExtracted, right_x, right_y, disparity) = FindMatchBlock(rightImg, leftImageExtracted, left_x, left_y, searchWindow, max_disparity)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        disp.append(disparity)
    
    histogram = Counter(disp)
    mostCommon = histogram.most_common(3)
    max_disparity = max(disparity for disparity, count in mostCommon)

    print(f'Znaleziono efektywne disparity = {max_disparity} w czasie {ElapsedTime(start_timestamp):.2f}s')
    return max_disparity

def CustomDisparityMap(leftImage, rightImage, searchWindow, max_disparity):
    height, width = leftImage.shape[:2]
    disparity_map = np.zeros((height, width))

    start_timestamp = time.time()
    for lx in tqdm(range(searchWindow, width - searchWindow)):
        for ly in range(searchWindow, height - searchWindow):
            reference_block = ExtractBlock(leftImage, searchWindow, lx, ly)
            (_, _, _, disparity_map[ly, lx]) = FindMatchBlock(rightImage, reference_block, lx, ly, searchWindow, max_disparity)

    disparity_map = disparity_map[searchWindow : height - searchWindow, searchWindow : width - searchWindow]
    elapsed = ElapsedTime(start_timestamp)
    print(f'Wyznaczona mapa w czasie {ElapsedTime(start_timestamp):.2f}s')

    return disparity_map


def ConvertDisparityToDepth(disparity_map, parameters):
    (baseline, focal_length, doffs) = parameters
    return (baseline * focal_length) / (disparity_map + doffs)


def GetDepthFromUint24DepthMap(DepthMap, X, Y, max_depth):
    RGB = DepthMap[Y, X]
    (R, G, B) = RGB[0], RGB[1], RGB[2]
    depth = ((R + G*256 + B*256*256) / (256*256*256 - 1)) * max_depth
    print(f'Odległość w metrach w punkcie o współrzędnych X={X}, Y={Y}, depth={depth:.2f}m')
    return depth



def FovToFocalLength(image_width, fov_degrees):
    fov_radians = math.radians(fov_degrees)
    focal_length = image_width / (2 * math.tan(fov_radians / 2))
    return focal_length
    
def DepthToDisparity(depth_map, baseline, focal_length):
    disparity_map = (focal_length * baseline) / (depth_map + 1e-6)
    disparity_map = np.clip(disparity_map, 0, 255).astype(np.uint8)
    return disparity_map

def DisparityToDepth(disparity_map, baseline, focal_length):
    depth_map = (focal_length * baseline) / (disparity_map + 1e-6)
    depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
    return depth_map