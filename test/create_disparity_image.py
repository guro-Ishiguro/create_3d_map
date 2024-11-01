import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import config

DRONE_IMAGE_DIR = os.path.join(config.IMAGE_DIR, "drone")
DISPARITY_IMAGE_DIR = os.path.join(config.IMAGE_DIR, "disparity")


def create_disparity_image(image_L, image_R, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity


def to_orthographic_projection(depth, camera_height):
    """中心投影から正射投影への変換を適用する"""
    rows, cols = depth.shape
    mid_idx = cols // 2
    mid_idy = rows // 2
    col_indices = np.vstack([np.arange(cols)] * rows)
    row_indices = np.vstack([np.arange(rows)] * cols).transpose()
    shift_x = (
        (camera_height - depth) * (mid_idx - col_indices) / camera_height
    ).astype(int)
    shift_y = (
        (camera_height - depth) * (mid_idy - row_indices) / camera_height
    ).astype(int)
    new_x = np.clip(col_indices + shift_x, 0, cols - 1)
    new_y = np.clip(row_indices + shift_y, 0, rows - 1)
    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y, new_x), depth)
    ortho_depth[ortho_depth == np.inf] = 0
    return ortho_depth


def save_disparity_colormap(disparity, output_path):
    """視差画像をカラーマップで保存する"""
    # 視差画像を正規化
    disparity_normalized = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disparity_normalized = disparity_normalized.astype(np.uint8)

    # カラーマップを適用
    disparity_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    # カラーマップを保存
    plt.imsave(output_path, disparity_colormap)
    print(f"Disparity colormap saved as {output_path}")


# 左右の画像を読み込み
img_id = 520
left_image = cv2.imread(
    os.path.join(DRONE_IMAGE_DIR, f"left_{img_id}.png"), cv2.IMREAD_GRAYSCALE
)
left_image = cv2.equalizeHist(left_image)
right_image = cv2.imread(
    os.path.join(DRONE_IMAGE_DIR, f"right_{img_id}.png"), cv2.IMREAD_GRAYSCALE
)
right_image = cv2.equalizeHist(right_image)

B, fov_v, height = 0.3, 60, 1440
focal_length = height / (2 * np.tan(fov_v * np.pi / 180 / 2))
camera_height = 20

# 視差画像を生成
disparity = create_disparity_image(
    left_image, right_image, window_size=11, min_disp=0, num_disp=80
)

# disparity = to_orthographic_projection(disparity, camera_height)

# 視差画像をカラーマップとして保存
output_path = os.path.join(DISPARITY_IMAGE_DIR, f"disparity_{img_id}.png")
save_disparity_colormap(disparity, output_path)
