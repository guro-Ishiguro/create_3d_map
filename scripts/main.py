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

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ディレクトリパスの設定
DRONE_IMAGE_DIR = "/home/geolab/Projects/create_3d_map/data/images/drone"
DRONE_IMAGE_LOG = "/home/geolab/Projects/create_3d_map/data/txt/drone_image_log.txt"
ORB_SLAM_LOG = "/home/geolab/Projects/create_3d_map/data/txt/KeyFrameTrajectory.txt"

def clear_folder(dir_path):
    """指定フォルダの中身を削除する"""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                logging.info(f"{dir_path} is cleared.")
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")
    else:
        logging.info(f"The folder {dir_path} does not exist.")

def depth_to_world(depth_map, K, R, T):
    """深度マップをワールド座標に変換する"""
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    pixels_homogeneous = np.stack((i, j, np.ones_like(i)), axis=-1).reshape(-1, 3)
    K_inv = np.linalg.inv(K)
    camera_coords = (K_inv @ pixels_homogeneous.T).T * depth_map.reshape(-1, 1)
    camera_coords = np.hstack((camera_coords, np.ones((camera_coords.shape[0], 1))))
    RT = np.hstack((R, T.reshape(-1, 1)))
    world_coords = (RT @ camera_coords.T).T[:, :3]
    return world_coords

def disparity_image(disparity, img_id):
    """視差画像を表示する"""
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity, cmap='jet')
    plt.colorbar(label='Image')
    plt.title(f'Image {img_id}')
    plt.axis('off')
    plt.show()

def create_disparity_image(image_L, image_R, img_id, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    if image_L is None or image_R is None:
        logging.error(f"Error: One or both images for ID {img_id} are None.")
        return None
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

def convert_right_to_left_hand_coordinates(data):
    """右手座標系から左手座標系へ変換する"""
    data[:, 2] = -data[:, 2]
    return data

def display_neighbors(points, k=4):
    """近傍点のヒストグラムを表示し、外れ値の閾値を示す"""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_distances) + 2 * np.std(mean_distances)
    plt.hist(mean_distances, bins=1000, alpha=0.9)
    plt.axvline(np.mean(mean_distances), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(threshold, color='g', linestyle='dashed', linewidth=2)
    plt.xlabel('Mean Distance to k Nearest Neighbors')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mean Distances')
    plt.xlim(0, 0.1)
    plt.legend()
    plt.savefig("figure.png")

def grid_sampling(point_cloud, grid_size=0.1):
    """三次元点群のグリッドサンプリングを行う"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices]

if __name__ == "__main__":
    start_time = time.time()

    B, fov_v, height = 0.3, 60, 1440
    focal_length = height / (2 * np.tan(fov_v * np.pi / 180 / 2))
    cx, cy = 960, 720
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)

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

        disparity = create_disparity_image(left_image, right_image, i, window_size=5, min_disp=0, num_disp=80)
        if disparity is None:
            continue

        depth = B * focal_length / (disparity + 1e-6)
        depth[(depth < 0) | (depth > 23)] = 0
        world_coords = grid_sampling(convert_right_to_left_hand_coordinates(depth_to_world(depth, K, R, T)))
        
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

