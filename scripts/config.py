from dotenv import load_dotenv

load_dotenv()

import os

DATA_TYPE = os.getenv("DATA_TYPE")
HOME_DIR = os.getenv("HOME_DIR")

DATA = ["1_1920_1440", "1_3840_2880"]

# パスの設定
DATA_DIR = os.path.join(HOME_DIR, "data")
DATA_TYPE_DIR = os.path.join(DATA_DIR, DATA[int(DATA_TYPE)])
IMAGE_DIR = os.path.join(DATA_TYPE_DIR, "images")
TXT_DIR = os.path.join(DATA_TYPE_DIR, "txt")
DRONE_IMAGE_DIR = os.path.join(IMAGE_DIR, "drone")
DRONE_IMAGE_LOG = os.path.join(TXT_DIR, "drone_image_log.txt")
ORB_SLAM_LOG = os.path.join(TXT_DIR, "KeyFrameTrajectory.txt")

OUTPUT_DIR = os.path.join(HOME_DIR, "output")
OUTPUT_TYPE_DIR = os.path.join(OUTPUT_DIR, DATA[int(DATA_TYPE)])
POINT_CLOUD_DIR = os.path.join(OUTPUT_TYPE_DIR, "point_cloud")
POINT_CLOUD_FILE_PATH = os.path.join(POINT_CLOUD_DIR, "output.ply")
VIDEO_DIR = os.path.join(OUTPUT_TYPE_DIR, "video")

# スケールの設定
BASIS_WIDTH = int(DATA[0].split("_")[1])
CHOICED_WIDTH = int(DATA[int(DATA_TYPE)].split("_")[1])
SCALE = CHOICED_WIDTH / BASIS_WIDTH
