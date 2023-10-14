import cv2, mediapipe as mp, os
import matplotlib.pyplot as plt

pose = mp.solutions.pose.Pose()

cap = cv2.VideoCapture(os.path.join(os.getcwd(), "vids", "vid3.mp4"))

# Lists to store the coordinates and frame numbers
xs, ys, zs, frames = [], [], [], []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    landmarks = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
    if landmarks:
        # Extract nose landmark (landmark zero)
        nose = landmarks.landmark[0]
        
        # Store the nose positions and the frame number
        xs.append(nose.x)
        ys.append(nose.y)
        zs.append(nose.z)
        frames.append(frame_count)

    frame_count += 1
cap.release()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frames, xs, label='x-coordinate')
plt.plot(frames, ys, label='y-coordinate')
plt.plot(frames, zs, label='z-coordinate')
plt.title('Nose Position Over Frames')
plt.xlabel('Frame Number')
plt.ylabel('Normalized Coordinate')
plt.legend()
plt.grid(True)
plt.show()
