import cv2
import numpy as np
from msvcrt import getch
from numba import jit

@jit(nopython=True)
def ssd(left_window, right_window):
    return np.sum((left_window - right_window)**2)

def create_disparity_map_dummy(img1, img2, window_size: int, disparity_range: int) -> np.ndarray:
    img1_width = img1.shape[1]
    img1_height = img1.shape[0]

    disparity_map = np.zeros_like(img2, dtype=np.uint8)

    for y in range(window_size, img1_height - window_size):
        print(y)
        for x in range(window_size, img1_width - window_size):
            best_match = None
            best_match_x = None
            left_window = img1[y - window_size : y + window_size + 1, x - window_size : x + window_size + 1]
            for d in range(disparity_range):
                if x - window_size - d < 0:
                    continue
              
                right_window = img2[y - window_size : y + window_size + 1, x - window_size - d : x + window_size + 1 - d]
                if left_window.shape == right_window.shape:
                    diff = ssd(left_window, right_window)
                    if best_match is None or diff < best_match:
                      best_match = diff
                      best_match_x = x - d

            disparity_map[y, x] = abs(best_match_x - x)

    disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    return disparity_map


def downsize_image(img, n):
    height, width = img.shape
    return cv2.resize(img, (width // n, height // n))


def zad1():
    img = cv2.imread("dane_egz/Z1iZ2/depth.png")
    out_img = np.copy(img)

    height, width, _ = img.shape
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            distance = (r + g*256 + b*256*256) / (256*256*256 - 1) * 20
            if distance > 1.0:
                distance = 1.0
            out_img[y, x] = [distance * 255, distance * 255, distance * 255]
    
    cv2.imwrite("z1depth.png", out_img)

def zad2():
    depth_map = cv2.imread("dane_egz/Z1iZ2/depth.png")
    height, width, _ = depth_map.shape

    disp_map = np.copy(depth_map)

    baseline = 0.1
    focal_length = 2 * np.tan(np.radians(60))
    for y in range(height):
        for x in range(width):
            b, g, r = depth_map[y, x]
            distance = (r + g*256 + b*256*256) / (256*256*256 - 1) * 1000
            disp = baseline * focal_length / distance
            disp_map[y, x] = [disp * 255, disp * 255, disp * 255]

    cv2.imwrite("z2disparity.png", disp_map)

    
def zad3():
    img_left = cv2.imread("dane_egz/Z3iZ4/im0.png", 0)
    img_right = cv2.imread("dane_egz/Z3iZ4/im1.png", 0)

    # img_left = cv2.blur(img_left, (5, 5))
    # img_right = cv2.blur(img_right, (5, 5))

    stereo = cv2.StereoSGBM_create(
        numDisparities=270,
        blockSize=7,
    )

    disparity_map = stereo.compute(img_left, img_right)
    disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map = cv2.convertScaleAbs(disparity_map)

    focal_length = 3979.911
    baseline = 193.001
    doffs = 124.343
    depth_map = (focal_length * baseline) / (disparity_map + doffs)
    # depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite("z3disparity.png", disparity_map)
    cv2.imwrite("z3depth.png", depth_map)

    groundtruth = cv2.imread('dane_egz/Z3iZ4/disp0.pfm', cv2.IMREAD_UNCHANGED)
    groundtruth = np.asarray(groundtruth)
    groundtruth = groundtruth / 256

    gt_depth = focal_length * baseline / (groundtruth + doffs)

    print(groundtruth[256, 1626], disparity_map[256, 1626])
    print(gt_depth[256, 1626], depth_map[256, 1626])


def zad4():
    img_left = cv2.imread("dane_egz/Z3iZ4/im0.png", 0)
    img_left = downsize_image(img_left, 4)

    img_right = cv2.imread("dane_egz/Z3iZ4/im1.png", 0)
    img_right = downsize_image(img_right, 4)
    disparity_map = create_disparity_map_dummy(img_left, img_right, window_size=7, disparity_range=64)

    focal_length = 3979.911
    baseline = 193.001
    doffs = 124.343

    depth_map = (focal_length * baseline) / (disparity_map + doffs)
    
    print(disparity_map[256 // 4, 1626 // 4])
    print(depth_map[256 // 4, 1626 // 4])

