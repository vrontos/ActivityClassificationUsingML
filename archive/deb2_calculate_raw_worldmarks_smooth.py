import cv2, mediapipe as mp, os
import numpy as np
from tqdm import tqdm
from tkinter import filedialog
from collections import deque
from utils import DESIRED_LANDMARKS, check_for_rows_or_cols_of_nans

pose = mp.solutions.pose.Pose()

def smooth_landmarks(landmarks, history, alpha=0.5):
    smoothed_landmarks = []
    if landmarks:
        for i in DESIRED_LANDMARKS:  # Use DESIRED_LANDMARKS to select specific landmarks
            landmark = landmarks.landmark[i]
            if len(history[i]) >= 5:  # Number of frames to average over
                history[i].popleft()  # Remove oldest
            history[i].append((landmark.x, landmark.y, landmark.z, landmark.visibility))
            avg = np.mean(history[i], axis=0)
            smoothed_landmarks.extend(alpha * np.array((landmark.x, landmark.y, landmark.z, landmark.visibility)) + (1 - alpha) * avg)
    else:
        smoothed_landmarks = [np.nan] * len(DESIRED_LANDMARKS) * 4  # 4 values per landmark (x, y, z, visibility)
    return smoothed_landmarks

# Initialize history for smoothing
history = [deque(maxlen=5) for _ in range(mp.solutions.pose.PoseLandmark.__len__())]  # Adjust the size to the number of landmarks

# Use a file dialog to allow the user to select one or more MP4 files
file_paths = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")], title="Select One Or More Videos!")

if not file_paths:
    print("No files selected. Exiting.")
    exit()

# Convert the result from tuple to list
selected_videos = list(file_paths)

# Ensure the raw_worldlandmarks directory exists
raw_landmarks_base_dir = os.path.join(os.getcwd(), "raw_worldlandmarks_smooth")
if not os.path.exists(raw_landmarks_base_dir):
    os.makedirs(raw_landmarks_base_dir)

# Process each selected MP4 file
for mp4_file in tqdm(selected_videos, desc="Processing videos", unit="video"):
    
    # Modify save_path to reflect the directory structure
    root_dir = os.path.join(os.getcwd(), "vids")
    relative_path = os.path.relpath(mp4_file, root_dir)  # Get the relative path of the mp4_file to the root directory
    video_name = os.path.basename(relative_path).replace(".mp4", "")
    landmark_sub_directory = os.path.dirname(relative_path)
    landmark_directory = os.path.join(raw_landmarks_base_dir, landmark_sub_directory)
    
    if not os.path.exists(landmark_directory):
        os.makedirs(landmark_directory)
    
    save_path = os.path.join(landmark_directory, f"{video_name}_raw_worldlandmarks_smooth.npy")

    print(f"\n\nThe selected video is: {mp4_file}")
    print(f"\nThe world landmarks will be saved here: {save_path}\n")

    cap = cv2.VideoCapture(mp4_file)
    world_landmark_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        world_landmarks = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_world_landmarks
        smoothed_frame_data = smooth_landmarks(world_landmarks, history)
        world_landmark_data.append(smoothed_frame_data)

    # Check for rows or columns full of NaNs
    check_for_rows_or_cols_of_nans(world_landmark_data)
    
    np.save(save_path, world_landmark_data)
