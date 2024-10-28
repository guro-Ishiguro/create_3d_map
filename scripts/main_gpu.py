import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import logging
from sklearn.neighbors import NearestNeighbors
import matching
import time
import torch  # PyTorchをインポートして、GPUでの行列演算に利用

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ディレクトリパスの設定
DRONE_IMAGE_DIR = "/home/geolab/Projects/create_3d_map/data/images/drone"
DRONE_IMAGE_LOG = "/home/geolab/Projects/create_3d_map/data/txt/drone_image_log.txt"
ORB_SLAM_LOG = "/home/geolab/Projects/create_3d_map/data/txt/KeyFrameTrajectory.txt"

def create_disparity_image(image_L, image_R, img_id, window_size, min_disp, num_disp):
    """CPUを使用した視差画像の生成 (StereoSGBM)"""
    if image_L is None or image_R is None:
        logging.error(f"Error: One or both images for ID {img_id} are None.")
        return None
    
    # CPU用のStereoSGBMオブジェクトを作成
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
    
    # 視差計算
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity

def depth_to_world_cuda(depth_map, K, R, T):
    """CUDAを使用して深度マップをワールド座標に変換"""
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    pixels_homogeneous = np.stack((i, j, np.ones_like(i)), axis=-1).reshape(-1, 3)
    
    K_inv = np.linalg.inv(K)
    camera_coords = (K_inv @ pixels_homogeneous.T).T * depth_map.reshape(-1, 1)
    camera_coords = np.hstack((camera_coords, np.ones((camera_coords.shape[0], 1))))
    RT = np.hstack((R, T.reshape(-1, 1)))
    
    # PyTorchでの変換 (GPUを使用)
    camera_coords = torch.from_numpy(camera_coords).float().cuda()
    RT = torch.from_numpy(RT).float().cuda()
    world_coords = torch.matmul(RT, camera_coords.T).T[:, :3].cpu().numpy()  # 結果をCPUに戻す
    return world_coords

def write_ply(filename, vertices):
    """頂点データをPLYファイルに書き込む"""
    header = f'''ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
end_header
'''
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, vertices, fmt='%f %f %f')

def grid_sampling(point_cloud, grid_size=0.1):
    """三次元点群のグリッドサンプリングを行う"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices]

if __name__ == "__main__":
    start_time = time.time()

    # カメラ行列の設定
    B, fov_v, height = 0.3, 60, 1440
    focal_length = height / (2 * np.tan(fov_v * np.pi / 180 / 2))
    cx, cy = 960, 720
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    principal_point_distance = B * focal_length / 20

    drone_image_list = matching.read_file_list(DRONE_IMAGE_LOG)
    orb_slam_pose_list = matching.read_file_list(ORB_SLAM_LOG)
    
    # マッチング開始
    logging.info("Starting timestamp matching...")
    match_start = time.time()
    matches = matching.associate(drone_image_list, orb_slam_pose_list, 0.0, 0.02)
    logging.info(f"Timestamp matching completed in {time.time() - match_start:.2f} seconds")

    if len(matches) < 2:
        logging.error("Couldn't find matching timestamp pairs.")
        sys.exit(1)

    logging.info(f"{len(matches)} points matched.")
    
    # 3Dマップ生成開始
    logging.info("Starting 3D map creation...")
    map_start = time.time()
    cumulative_world_coords = None
    for i in range(len(matches)):
        img_id = matches[i][1][0]
        dx = float(matches[i][2][0])
        dy = float(matches[i][2][1])
        dz = float(matches[i][2][2])
        T = np.array([dx, dy, dz], dtype=np.float32)
        
        left_image = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"left_{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(os.path.join(DRONE_IMAGE_DIR, f"right_{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        
        if left_image is None or right_image is None:
            logging.error(f"Failed to load images for ID {img_id}")
            continue

        # CPUを使用した視差画像の生成
        disparity = create_disparity_image(left_image, right_image, i, window_size=5, min_disp=0, num_disp=80)
        if disparity is None:
            continue

        depth = B * focal_length / (disparity + 1e-6)
        depth[(depth < 0) | (depth > 23)] = 0

        # CUDAでワールド座標への変換
        world_coords = grid_sampling(depth_to_world_cuda(depth, K, R, T))
        
        if cumulative_world_coords is None:
            cumulative_world_coords = world_coords
        else:
            cumulative_world_coords = np.vstack((cumulative_world_coords, world_coords))

    logging.info(f"3D map creation completed in {time.time() - map_start:.2f} seconds")

    # PointCloudフィルタリング
    logging.info("Starting pointcloud filtering...")
    filter_start = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cumulative_world_coords)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    write_ply('output.ply', np.asarray(pcd.points))
    logging.info(f"Pointcloud filtering completed in {time.time() - filter_start:.2f} seconds")

    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")
