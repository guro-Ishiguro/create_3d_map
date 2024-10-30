import os
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import config

DRONE_IMAGE_DIR = os.path.join(config.IMAGE_DIR, "drone")

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

def to_orthographic_projection(depth, camera_height):
    """中心投影から正射投影への変換を適用する"""
    rows, cols = depth.shape
    mid_idx = cols // 2
    mid_idy = rows // 2
    col_indices = np.vstack([np.arange(cols)] * rows)
    row_indices = np.vstack([np.arange(rows)] * cols).transpose()
    shift_x = np.where(
        (depth > 23) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idx - col_indices) / camera_height).astype(int)
    )
    shift_y = np.where(
        (depth > 23) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idy - row_indices) / camera_height).astype(int)
    )
    new_x = np.clip(col_indices + shift_x, 0, cols - 1)
    new_y = np.clip(row_indices + shift_y, 0, rows - 1)
    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y, new_x), depth)
    ortho_depth[ortho_depth == np.inf] = 0
    return ortho_depth

def depth_to_world(depth_map, K, R, T, pixel_size):
    """深度マップをワールド座標に変換する"""
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    x_coords = (j - width // 2) * pixel_size
    y_coords = (i - height // 2) * pixel_size
    z_coords = camera_height - depth_map
    local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
    print(local_coords)
    world_coords = (R @ local_coords.T).T + T
    return world_coords

def convert_right_to_left_hand_coordinates(data):
    """右手座標系から左手座標系へ変換する"""
    data[:, 2] = -data[:, 2]
    return data

def save_depth_colormap(depth, output_path):
    """視差画像をカラーマップで保存する"""
    # 視差画像を正規化
    depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # カラーマップを適用
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # カラーマップを保存
    plt.imsave(output_path, depth_colormap)
    print(f"Disparity colormap saved as {output_path}")

# 左右の画像を読み込み
img_id_1 = 1001
left_image_1 = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"left_{img_id_1}.png"), cv2.IMREAD_GRAYSCALE)
right_image_1 = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"right_{img_id_1}.png"), cv2.IMREAD_GRAYSCALE)

img_id_2 = 1015
left_image_2 = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"left_{img_id_2}.png"), cv2.IMREAD_GRAYSCALE)
right_image_2 = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"right_{img_id_2}.png"), cv2.IMREAD_GRAYSCALE)

img_id_3 = 1024
left_image_3 = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"left_{img_id_3}.png"), cv2.IMREAD_GRAYSCALE)
right_image_3 = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"right_{img_id_3}.png"), cv2.IMREAD_GRAYSCALE)

B, fov_v, height = 0.3, 60, 1440
focal_length = height / (2 * np.tan(fov_v * np.pi / 180 / 2))
camera_height = 20
cx, cy = 960, 720
scene_height = 2 * camera_height * np.tan(np.radians(fov_v) / 2)
pixel_size = scene_height / height

K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
R = np.eye(3, dtype=np.float32)
T_1 = np.array([24.30421257, -0.350184917, -0.080562748])
T_2 = np.array([26.124996185, -0.364336908, -0.088445932])
T_3 = np.array([27.342170715, -0.356232464, -0.080338739])

# 視差画像を生成
disparity_1 = create_disparity_image(left_image_1, right_image_1, window_size=5, min_disp=0, num_disp=80)
disparity_2 = create_disparity_image(left_image_2, right_image_2, window_size=5, min_disp=0, num_disp=80)
disparity_3 = create_disparity_image(left_image_3, right_image_3, window_size=5, min_disp=0, num_disp=80)

# 深度画像を生成
depth_1 = B * focal_length / (disparity_1 + 1e-6)
depth_1[(depth_1 < 0) | (depth_1 > 23)] = 0
depth_1 = to_orthographic_projection(depth_1, camera_height)

depth_2 = B * focal_length / (disparity_2 + 1e-6)
depth_2[(depth_2 < 0) | (depth_2 > 23)] = 0
depth_2 = to_orthographic_projection(depth_2, camera_height)

depth_3 = B * focal_length / (disparity_3 + 1e-6)
depth_3[(depth_3 < 0) | (depth_3 > 23)] = 0
depth_3 = to_orthographic_projection(depth_3, camera_height)

world_coords_1 = convert_right_to_left_hand_coordinates(depth_to_world(depth_1, K, R, T_1, pixel_size))
world_coords_1 = world_coords_1[depth_1.reshape(-1) > 0]

world_coords_2 = convert_right_to_left_hand_coordinates(depth_to_world(depth_2, K, R, T_2, pixel_size))
world_coords_2 = world_coords_2[depth_2.reshape(-1) > 0]

world_coords_3 = convert_right_to_left_hand_coordinates(depth_to_world(depth_3, K, R, T_3, pixel_size))
world_coords_3 = world_coords_3[depth_3.reshape(-1) > 0]

# 三次元点群の統合
cumulative_world_coords = np.vstack((world_coords_1, world_coords_2))
cumulative_world_coords = np.vstack((cumulative_world_coords, world_coords_3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cumulative_world_coords)
o3d.visualization.draw_geometries([pcd])