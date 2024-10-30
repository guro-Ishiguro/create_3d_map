from dotenv import load_dotenv
load_dotenv()

import os

DATA_TYPE = os.getenv('DATA_TYPE') 
HOME_DIR = os.getenv('HOME_DIR') 

DATA = ["1_1920_1440", "1_3840_2880"]

WORKING_DIR = os.path.join(HOME_DIR, DATA[int(DATA_TYPE)])
IMAGE_DIR = os.path.join(WORKING_DIR, "images")
TXT_DIR = os.path.join(WORKING_DIR), "txt"