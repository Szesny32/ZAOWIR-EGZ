import L3LIB
import time
import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt


DOFFS = 170.681
BASELINE = 178.232
F = 2945.377

scale = 0.25
max_disparity = 48
searchWindow = 7
blockSize = 2*searchWindow + 1

leftImage = cv.imread('datasets/Lab3/im0.png', cv.IMREAD_GRAYSCALE)
rightImage =  cv.imread('datasets/Lab3/im1.png', cv.IMREAD_GRAYSCALE)
reference_disparity = cv.imread('datasets/Lab3/disp0.pfm',  -1)
assert leftImage.shape == rightImage.shape


leftImage = L3LIB.Scale(leftImage, scale)
rightImage = L3LIB.Scale(rightImage, scale)

reference_disparity[np.isinf(reference_disparity)] = 0
reference_disparity = np.nan_to_num(reference_disparity, nan=0)
reference_disparity = L3LIB.Scale(reference_disparity, scale)


# ---------------------------------------------------------------------------
# Custom
# ---------------------------------------------------------------------------
# Test - find random matching points
L3LIB.TestRandomPoints(leftImage, rightImage, searchWindow, max_disparity)

# Find Deepth and Disparity map
(custom_disparity_map, custom_elapsed_time) = L3LIB.DisparityMap(leftImage, rightImage, searchWindow, max_disparity, parameters = (BASELINE, F, DOFFS))
print(f"\nCustom: {custom_elapsed_time:.2f}s")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# StereoSGBM
# ---------------------------------------------------------------------------
start_timestamp = time.time()
sgbm = cv.StereoSGBM_create(minDisparity = 0, numDisparities = max_disparity, blockSize = blockSize)
sgbm_disparity_map = sgbm.compute(leftImage, rightImage)
sgbm_elapsed_time = L3LIB.ElapsedTime(start_timestamp)
print(f"SGBM: - {sgbm_elapsed_time:.5f}s")


# ---------------------------------------------------------------------------
# StereoBM
# ---------------------------------------------------------------------------
start_timestamp = time.time()
bm = cv.StereoBM_create(max_disparity, blockSize)
bm_disparity_map  = bm.compute(leftImage, rightImage)
bm_elapsed_time = L3LIB.ElapsedTime(start_timestamp)
print(f"BM: {bm_elapsed_time:.5f}s")


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].set_title('Custom Disparity Map')
axs[0, 0].axis('off') 
cbar1 = fig.colorbar(axs[0, 0].imshow(custom_disparity_map, cmap='turbo'), ax=axs[0, 0], fraction=0.046, pad=0.04)

axs[0, 1].set_title('StereoSGBM Disparity Map')
axs[0, 1].axis('off')
cbar2 = fig.colorbar(axs[0, 1].imshow(sgbm_disparity_map, cmap='turbo'), ax=axs[0, 1], fraction=0.046, pad=0.04)

axs[1, 0].set_title('StereoBM Disparity Map')
axs[1, 0].axis('off')
cbar3 = fig.colorbar(axs[1, 0].imshow(bm_disparity_map, cmap='turbo'), ax=axs[1, 0], fraction=0.046, pad=0.04)

axs[1, 1].set_title('Reference Disparity Map')
axs[1, 1].axis('off')
cbar4 = fig.colorbar(axs[1, 1].imshow(reference_disparity, cmap='turbo'), ax=axs[1, 1], fraction=0.046, pad=0.04)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()



parameters = (BASELINE, F, DOFFS) 
custom_deepth_map = L3LIB.ConvertDisparityToDepth(custom_disparity_map.astype(np.float32) / 16, parameters)
sgbm_deepth_map = L3LIB.ConvertDisparityToDepth(sgbm_disparity_map.astype(np.float32) / 16, parameters)
bm_deepth_map = L3LIB.ConvertDisparityToDepth(bm_disparity_map.astype(np.float32) / 16, parameters)

min_val = np.min(reference_disparity)
max_val = np.max(reference_disparity)
normalized_reference_disparity = (reference_disparity - min_val) / (max_val - min_val)
reference_deepth_map= L3LIB.ConvertDisparityToDepth(reference_disparity, parameters)
#reference_deepth_map= L3LIB.ConvertDisparityToDepth(normalized_reference_disparity, parameters)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].set_title('Custom Deepth Map')
axs[0, 0].axis('off') 
cbar1 = fig.colorbar(axs[0, 0].imshow(custom_deepth_map, cmap='plasma'), ax=axs[0, 0], fraction=0.046, pad=0.04)

axs[0, 1].set_title('StereoSGBM Deepth Map')
axs[0, 1].axis('off')
cbar2 = fig.colorbar(axs[0, 1].imshow(sgbm_deepth_map, cmap='plasma'), ax=axs[0, 1], fraction=0.046, pad=0.04)

axs[1, 0].set_title('StereoBM Deepth Map')
axs[1, 0].axis('off')
cbar3 = fig.colorbar(axs[1, 0].imshow(bm_deepth_map, cmap='plasma'), ax=axs[1, 0], fraction=0.046, pad=0.04)

axs[1, 1].set_title('Reference Deepth Map')
axs[1, 1].axis('off')
cbar4 = fig.colorbar(axs[1, 1].imshow(reference_deepth_map, cmap='plasma'), ax=axs[1, 1], fraction=0.046, pad=0.04)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()


#---

leftImage = cv.imread('datasets/Lab3/im0.png')
leftImage = L3LIB.Scale(leftImage, scale)
h, w = leftImage.shape[:2]
points, colors = L3LIB.ConvertDeepthToCloud(60000*custom_deepth_map, leftImage[searchWindow: h - searchWindow, searchWindow: w - searchWindow], parameters)
L3LIB.write_ply('Custom.ply', points, colors)

points, colors = L3LIB.ConvertDeepthToCloud(sgbm_deepth_map, leftImage, parameters)
L3LIB.write_ply('StereoSGBM.ply', points, colors)

points, colors = L3LIB.ConvertDeepthToCloud(bm_deepth_map, leftImage, parameters)
L3LIB.write_ply('StereoBM.ply', points, colors)

points, colors = L3LIB.ConvertDeepthToCloud(reference_deepth_map, leftImage, parameters)
L3LIB.write_ply('Reference.ply', points, colors)

L3LIB.visualize_ply('Custom.ply')
L3LIB.visualize_ply('StereoSGBM.ply')
L3LIB.visualize_ply('StereoBM.ply')
L3LIB.visualize_ply('Reference.ply')

