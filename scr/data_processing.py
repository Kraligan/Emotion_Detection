# Imports
import kagglehub
import os
import numpy as np
import cv2
from utils import get_face_landmarks

# Dowload dataset fer2013: (Uncomment to download)
# path = kagglehub.dataset_download("msambare/fer2013")
# print("Path to dataset files:", path)

# Dowload dataset affectnet: (Uncomment to download)
# path = kagglehub.dataset_download("mstjebashazida/affectnet")
# print("Path to dataset files:", path)

# Path to train/ and test/ folder with .jpg images
# data_path = os.path.expanduser("~/.cache/kagglehub/datasets/msambare/fer2013/versions/1") 
data_path = os.path.expanduser("~/.cache/kagglehub/datasets/mstjebashazida/affectnet/versions/1") # /!\ move ...versions/1/archive(3)/Train/ to .../versions/1/train/

# extract landmarks for each image
landmarks = []
for emotion_idx, emotion in enumerate(os.listdir(os.path.join(data_path, "train/"))):
    for img_name in os.listdir(os.path.join(data_path, f"train/{emotion}")):
        img_path = os.path.join(data_path, f"train/{emotion}/{img_name}")

        img_landmarks = get_face_landmarks(img_path)
        img_with_label = np.append(img_landmarks, int(emotion_idx)) # We add label at the end of the landmarks table 
        landmarks.append(img_with_label)

# print(landmarks[:3])

# save dataset with landmarks and label
np.savetxt('data/dataset_Affectnet.txt', np.array(landmarks))