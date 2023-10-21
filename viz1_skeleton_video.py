import cv2, mediapipe as mp, os
import tkinter as tk
from tkinter import filedialog
from utils import DESIRED_LANDMARKS

pose = mp.solutions.pose.Pose()

# Open a file dialog to select the video from the 'vids' directory
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
vid_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), "vids"), title="Select video file", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))

# Return if no file is selected
if not vid_path:
    exit()

cap = cv2.VideoCapture(vid_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    landmarks = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks

    if landmarks:
        # Draw landmarks
        for i, landmark in enumerate(landmarks.landmark):
            if i in DESIRED_LANDMARKS:
                color = (0, 255, 0)  # Green for desired landmarks
            else:
                color = (0, 0, 255)  # Red for the rest
    
            landmark_pos = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            cv2.circle(frame, landmark_pos, 4, color, -1)

        # Draw connections
        for connection in mp.solutions.pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            start_pos = (int(landmarks.landmark[start_idx].x * frame.shape[1]), int(landmarks.landmark[start_idx].y * frame.shape[0]))
            end_pos = (int(landmarks.landmark[end_idx].x * frame.shape[1]), int(landmarks.landmark[end_idx].y * frame.shape[0]))
            cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)  # White for all connections


    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


