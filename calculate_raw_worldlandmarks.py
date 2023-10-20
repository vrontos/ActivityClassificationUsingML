import cv2, mediapipe as mp, os
import numpy as np
from tqdm import tqdm
from tkinter import filedialog
from config import DESIRED_LANDMARKS

pose = mp.solutions.pose.Pose()

# Use a file dialog to allow the user to select one or more MP4 files
file_paths = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")], title="Select Videos to calculate their world-landmarks!")

if not file_paths:
    print("No files selected. Exiting.")
    exit()

# Convert the result from tuple to list
selected_videos = list(file_paths)

# Ensure the raw_worldlandmarks directory exists
raw_landmarks_base_dir = os.path.join(os.getcwd(), "raw_worldlandmarks")
if not os.path.exists(raw_landmarks_base_dir):
    os.makedirs(raw_landmarks_base_dir)

# Process each selected MP4 file
for mp4_file in tqdm(selected_videos, desc="Processing videos", unit="video"):
    cap = cv2.VideoCapture(mp4_file)
    world_landmark_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        world_landmarks = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_world_landmarks
        frame_data = []
        if world_landmarks:
            for i in DESIRED_LANDMARKS:
                frame_data.extend([world_landmarks.landmark[i].x, world_landmarks.landmark[i].y, world_landmarks.landmark[i].z, world_landmarks.landmark[i].visibility])
        else:
            frame_data = [np.nan] * len(DESIRED_LANDMARKS) * 3 + [0] * len(DESIRED_LANDMARKS)
            
        world_landmark_data.append(frame_data)
  
    # Modify save_path to reflect the directory structure
    root_dir = os.path.join(os.getcwd(), "vids")
    relative_path = os.path.relpath(mp4_file, root_dir)  # Get the relative path of the mp4_file to the root directory
    video_name = os.path.basename(relative_path).replace(".mp4", "")
    landmark_sub_directory = os.path.dirname(relative_path)
    landmark_directory = os.path.join(raw_landmarks_base_dir, landmark_sub_directory)
    
    if not os.path.exists(landmark_directory):
        os.makedirs(landmark_directory)
    
    save_path = os.path.join(landmark_directory, f"{video_name}_raw_worldlandmarks.npy")
    
    # TODO: add warning if there are cols/rows complete of NaN
    np.save(save_path, world_landmark_data)
