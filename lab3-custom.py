import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import random

def extractBlock(image, size, x, y):
    (yStart, yStop) = (y - size, y + size)
    (xStart, xStop) = (x - size, x + size)
    return image[yStart : yStop,  xStart : xStop]

def findMatchBlock(image, originBlock, ox, oy, windowSize):
    _, width  = image.shape

    searchRange = 64
    startX = max(ox - searchRange, windowSize)
    endX = min(ox + searchRange, width - windowSize)

    matchBlock = extractBlock(image, windowSize, startX, oy)
    disparity = np.sum((originBlock - matchBlock) ** 2)
    bestX = windowSize

    for x in range(startX, endX):
        block = extractBlock(image, windowSize, x, oy)
        d = np.sum((originBlock - block) ** 2)
        if d < disparity:
            bestX = x
            disparity = d
            matchBlock = block
    return (matchBlock, bestX, oy, disparity)

def print_loading_bar(iteration, total, info = None, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    if info is not None:
        sys.stdout.write(f'\r[{arrow}{spaces}] - {iteration} / {total} ({(progress * 100.0):.2f}%) [ {info} ]')
    else:
        sys.stdout.write(f'\r[{arrow}{spaces}] - {iteration} / {total} ({(progress * 100.0):.2f}%) ')
    sys.stdout.flush()

def generate_random_coordinates(num_points, width, height, windowSize):
    random_coordinates = []
    for _ in range(num_points):
        x = random.randint(windowSize, width - windowSize)
        y = random.randint(windowSize, height - windowSize)
        random_coordinates.append((x, y))
    return random_coordinates

def convertDisparityToDepth(baseline, focal_length, disparity):
    return (baseline * focal_length) / (disparity)




dataset = ['datasets/Lab3/im2.png', 'datasets/Lab3/im6.png', 
           'datasets/Lab3/im0.png', 'datasets/Lab3/im1.png']
 
# leftImage = cv.imread(dataset[2])
# rightImage = cv.imread(dataset[3])

leftImage = cv.imread(dataset[2], cv.IMREAD_GRAYSCALE)
rightImage =  cv.imread(dataset[3], cv.IMREAD_GRAYSCALE)
height, width = leftImage.shape
leftImage = cv.resize(leftImage, (int(height/4), int(width/4)))
rightImage = cv.resize(rightImage, (int(height/4), int(width/4)))
height, width = leftImage.shape
windowSize = 5


num_points_to_generate = 5
random_left_coordinates = generate_random_coordinates(num_points_to_generate, width, height, windowSize)
right_coordinates = []


for point in random_left_coordinates:
    (x, y) = point
    originBlock = extractBlock(leftImage, windowSize, x, y)
    (matchBlock, xr, yr, disparity) = findMatchBlock(rightImage, originBlock, x, y, windowSize)
    cv.circle(leftImage, (x, y), windowSize, (0, 255, 255), 5)
    cv.circle(rightImage, (xr, yr), windowSize, (0, 255, 255), 5)

combined_image = np.concatenate((leftImage, rightImage), axis=1)
cv.imshow('Obrazy z ramkami okręgów', cv.resize(combined_image, (740, 250)))
cv.waitKey(0)
cv.destroyAllWindows()
   
baseline = 193.001
focal_length = 3979.911 


n = (width - 2*windowSize) * (height - 2*windowSize)
i = 0
disparity_map = np.zeros((height, width))
deepth_map = np.zeros((height, width))
timestamp = time.time()
for x in range(windowSize, width - windowSize):
    for y in range(windowSize, height - windowSize):
        originBlock = extractBlock(leftImage, windowSize, x, y)
        (matchBlock, xr, yr, disparity) = findMatchBlock(rightImage, originBlock, x, y, windowSize)
        disparity_map[y, x] = disparity
        deepth_map[y, x] = convertDisparityToDepth(baseline, focal_length, disparity)
        i += 1
        print_loading_bar(i, n, f'{(time.time() - timestamp):.2f}s')
print(f"\nZakończono proces w {(time.time() - timestamp):.2f}s")
color_map = plt.cm.get_cmap('turbo', 8)
plt.imshow(disparity_map, cmap=color_map)
plt.show()             
cv.waitKey(0)
cv.destroyAllWindows()            

plt.imshow(deepth_map, cmap='gray')
plt.show()             
cv.waitKey(0)
cv.destroyAllWindows()    