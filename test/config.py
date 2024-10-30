from dotenv import load_dotenv
load_dotenv()

import os

DATA_TYPE = os.getenv('DATA_TYPE') 
HOME_DIR = os.getenv('HOME_DIR') 

DATA = ["1_1920_1440", "1_3840_2880"]

# パスの設定
WORKING_DIR = os.path.join(HOME_DIR, DATA[int(DATA_TYPE)])
IMAGE_DIR = os.path.join(WORKING_DIR, "images")
TXT_DIR = os.path.join(WORKING_DIR, "txt")

# スケールの設定
BASIS_WIDTH = int(DATA[0].split("_")[1])
CHOICED_WIDTH = int(DATA[int(DATA_TYPE)].split("_")[1])
SCALE = CHOICED_WIDTH / BASIS_WIDTH