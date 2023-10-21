import cv2, mediapipe as mp, os
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from config import DESIRED_LANDMARKS

pose = mp.solutions.pose.Pose()

# Find all MP4 files in the /vids directory
root_dir = os.path.join(os.getcwd(), "vids")
mp4_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in [f for f in filenames if f.endswith(".mp4")]:
        mp4_files.append(os.path.join(dirpath, filename))

selected_videos = []

def on_select():
    selected_videos.extend([mp4_files[i] for i in range(len(mp4_files)) if var_values[i].get() == 1])
    root.destroy()  # Destroy the window
    
root = tk.Tk()
root.title("Select Videos")

var_values = []
for mp4_file in mp4_files:
    video_name = os.path.basename(mp4_file).replace(".mp4", "")
    landmark_path = os.path.join(os.getcwd(), "raw_worldlandmarks", f"{video_name}_raw_worldlandmarks.npy")
    
    display_name = "* " + mp4_file if not os.path.exists(landmark_path) else mp4_file

    var = tk.IntVar()
    ttk.Checkbutton(root, text=display_name, variable=var).pack(anchor='w', padx=10, pady=2)
    var_values.append(var)

ttk.Button(root, text="Start Processing", command=on_select).pack(pady=20)
root.mainloop()

# Ensure the raw_worldlandmarks directory exists
if not os.path.exists(os.path.join(os.getcwd(), "raw_worldlandmarks")):
    os.makedirs(os.path.join(os.getcwd(), "raw_worldlandmarks"))

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
    
    video_name = os.path.basename(mp4_file).replace(".mp4", "")
    save_path = os.path.join(os.getcwd(), "raw_worldlandmarks", f"{video_name}_raw_worldlandmarks.npy")
    np.save(save_path, world_landmark_data)
