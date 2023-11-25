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

# Ask user for frame range
start_frame_input = input("From which frame? (click Enter to play the whole video): ")
if start_frame_input.isdigit():
    start_frame = int(start_frame_input)
    end_frame_input = input("Until which frame? (click Enter to not specify): ")
    end_frame = int(end_frame_input) if end_frame_input.isdigit() else float('inf')
else:
    start_frame, end_frame = 0, float('inf')

cap = cv2.VideoCapture(vid_path)
current_frame = 0

while cap.isOpened() and current_frame <= end_frame:
    ret, frame = cap.read()import cv2, mediapipe as mp, os
    import tkinter as tk
    from tkinter import filedialog
    from collections import deque
    import numpy as np
    from utils import DESIRED_LANDMARKS

    pose = mp.solutions.pose.Pose()

    alpha = 0.5  # Smoothing parameter


    def smooth_landmarks(landmarks, history, alpha):
        if landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                if len(history[i]) >= 5:  # Number of frames to average over
                    history[i].popleft()  # Remove oldest
                history[i].append((landmark.x, landmark.y, landmark.z))

                avg = np.mean(history[i], axis=0)
                landmark.x, landmark.y, landmark.z = alpha * np.array((landmark.x, landmark.y, landmark.z)) + (1 - alpha) * avg
        return landmarks

    # Open a file dialog to select the video from the 'vids' directory
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    vid_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), "vids"), title="Select video file", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))

    # Return if no file is selected
    if not vid_path:
        exit()
        
    # Initialize history for smoothing
    history = [deque(maxlen=5) for _ in range(mp.solutions.pose.PoseLandmark.__len__())]  # Adjust the size to the number of landmarks

    # Ask user for frame range
    start_frame_input = input("From which frame? (click Enter to play the whole video): ")
    if start_frame_input.isdigit():
        start_frame = int(start_frame_input)
        end_frame_input = input("Until which frame? (click Enter to not specify): ")
        end_frame = int(end_frame_input) if end_frame_input.isdigit() else float('inf')
    else:
        start_frame, end_frame = 0, float('inf')

    cap = cv2.VideoCapture(vid_path)
    current_frame = 0

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret or current_frame < start_frame:
            current_frame += 1
            continue

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        landmarks = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
        landmarks = smooth_landmarks(landmarks, history, alpha)

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

        # Display current frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Frame: {current_frame}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display smoothing parameters
        cv2.putText(frame, f'Smoothing: alpha={alpha}', (10, frame.shape[0] - 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    if not ret or current_frame < start_frame:
        current_frame += 1
        continue

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

    # Display current frame number
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Frame: {current_frame}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    current_frame += 1

cap.release()
cv2.destroyAllWindows()
