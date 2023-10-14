import cv2, mediapipe as mp, os
import numpy as np

pose = mp.solutions.pose.Pose()

# cap = cv2.VideoCapture(os.path.join(os.getcwd(), "vids", "vid3.mp4"))
# cap = cv2.VideoCapture(os.path.join(os.getcwd(), "vids/train", "combined_train_vids.mp4"))
cap = cv2.VideoCapture(os.path.join(os.getcwd(), "vids/test", "test_vid.mp4"))

# List of landmark indices to keep
desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Using a list to store landmark data for each frame
landmark_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    landmarks = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
    frame_data = []
    if landmarks:
        # Storing desired landmarks' x, y, z coordinates and visibility for this frame
        for i in desired_landmarks:
            frame_data.extend([landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z, landmarks.landmark[i].visibility])
    else:
        # If no landmarks are detected for this frame, use NaN values for x, y, z and 0 for visibility
        frame_data = [np.nan] * len(desired_landmarks) * 3 + [0] * len(desired_landmarks)
        
    landmark_data.append(frame_data)
    
np.save('raw_landmarks.npy', landmark_data)
