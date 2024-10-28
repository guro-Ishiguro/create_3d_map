import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

DRONE_IMAGE_DIR = "/home/geolab/Projects/create_3d_map/data/images/drone"
DISPARITY_IMAGE_DIR = "/home/geolab/Projects/create_3d_map/data/images/disparity"

def create_disparity_image(image_L, image_R, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity

def to_orthographic_projection(disparity, principal_point_distance):
    """中心投影から正射投影への変換を適用する"""
    rows, cols = disparity.shape
    mid_idx = cols // 2
    col_indices = np.arange(cols)  
    row_indices = np.arange(rows)[:, None]  
    shift = (disparity * (mid_idx - col_indices) / (principal_point_distance + disparity)).astype(int)
    new_col_indices = col_indices + shift
    new_col_indices = np.clip(new_col_indices, 0, cols - 1)
    shifted_disparity = np.zeros_like(disparity)
    shifted_disparity[row_indices, new_col_indices] = disparity
    return shifted_disparity

def save_disparity_colormap(disparity, output_path):
    """視差画像をカラーマップで保存する"""
    # 視差画像を正規化
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = disparity_normalized.astype(np.uint8)
    
    # カラーマップを適用
    disparity_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    # カラーマップを保存
    plt.imsave(output_path, disparity_colormap)
    print(f"Disparity colormap saved as {output_path}")

# 左右の画像を読み込み
img_id = 2000
left_image = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"left_{img_id}.png"), cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"right_{img_id}.png"), cv2.IMREAD_GRAYSCALE)

B, fov_v, height = 0.3, 60, 1440
focal_length = height / (2 * np.tan(fov_v * np.pi / 180 / 2))
principal_point_distance = principal_point_distance = B * focal_length / 20

# 視差画像を生成
disparity = create_disparity_image(left_image, right_image, window_size=5, min_disp=0, num_disp=80)

disparity = to_orthographic_projection(disparity, principal_point_distance)

# 視差画像をカラーマップとして保存
output_path = os.path.join(DISPARITY_IMAGE_DIR, f"disparity_{img_id}.png")
save_disparity_colormap(disparity, output_path)
