import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import random
import copy


def Scale(image, nHeight):
    height, width = image.shape[:2]
    scale = nHeight / height
    nWidth = int(scale * width)
    return cv.resize(image, (nWidth, nHeight), interpolation = cv.INTER_AREA)


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

DOFFS = 170.681
BASELINE = 178.232
F = 2945.377

#mode = cv.IMREAD_GRAYSCALE
leftImage = cv.imread('left.png', cv.IMREAD_GRAYSCALE)
leftImage = cv.cvtColor(leftImage, cv.COLOR_BGR2RGB)
leftImage = Scale(leftImage, 100)

rightImage =  cv.imread('right.png', cv.IMREAD_GRAYSCALE)
rightImage = cv.cvtColor(rightImage, cv.COLOR_BGR2RGB)
rightImage = Scale(rightImage, 100)



##----


leftImage2 = copy.copy(leftImage)
rightImage2 = copy.copy(rightImage)
height, width = leftImage.shape[:2]


max_disparity = int(width * 0.25)
searchWindow = 10

no_points = 5
left_coordinates = GenerateRandomCoordinates(no_points, width, height, searchWindow)

for point in left_coordinates:
    (left_x, left_y) = point
    leftImageExtracted = ExtractBlock(leftImage, searchWindow, left_x, left_y)
    (rightImageExtracted, right_x, right_y, disparity) = FindMatchBlock(rightImage, leftImageExtracted, left_x, left_y, searchWindow, max_disparity)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv.rectangle(leftImage2, (left_x - searchWindow, left_y - searchWindow), (left_x + searchWindow, left_y + searchWindow), color, 1) 
    cv.rectangle(rightImage2, (right_x - searchWindow, right_y - searchWindow), (right_x + searchWindow, right_y + searchWindow), color, 1) 


plt.subplot(1, 2, 1)
plt.imshow(leftImage2)
plt.title('Left Image')

plt.subplot(1, 2, 2)
plt.imshow(rightImage2)
plt.title('Right Image')


plt.tight_layout()
plt.show()


##----
from tqdm import tqdm


def ElapsedTime(from_timestamp):
    return time.time() - from_timestamp

assert leftImage.shape == rightImage.shape

height, width = leftImage.shape[:2]
n = (width - 2*searchWindow) * (height - 2*searchWindow)
disparity_map = np.zeros((height, width))


start_timestamp = time.time()
for lx in tqdm(range(searchWindow, width - searchWindow)):
    for ly in range(searchWindow, height - searchWindow):
        reference_block = ExtractBlock(leftImage, searchWindow, lx, ly)
        (_, _, _, disparity_map[ly, lx]) = FindMatchBlock(rightImage, reference_block, lx, ly, searchWindow, max_disparity)

disparity_map = disparity_map[searchWindow : height - searchWindow, searchWindow : width - searchWindow]
elapsed = ElapsedTime(start_timestamp)
print(elapsed)
plt.subplot(1, 1, 1)
plt.title('Custom Disparity Map')
plt.axis('off') 
plt.imshow(disparity_map, cmap='turbo')
plt.show()

