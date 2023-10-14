import cv2
import numpy as np
from calculate_feature_matrix import calculate_feature_matrix
from utils import find_video_for_landmark, resize_frame, select_raw_landmark_files, select_descriptors_to_visualize

# 1. Select the raw landmarks file
file_path = select_raw_landmark_files()

# 2. Load the selected file and calculate feature matrix and descriptors
loaded_landmark_data = np.load(file_path[0])
final_feature_matrix, _, descriptors = calculate_feature_matrix(loaded_landmark_data)

# 3. Select which descriptors to visualize
selected_descriptors = select_descriptors_to_visualize(descriptors)
print(f"After function call, selected descriptors: {selected_descriptors}")

# 4. Find the video corresponding to the selected landmarks
video_path = find_video_for_landmark(file_path[0])
if not video_path:
    raise ValueError(f"No video found corresponding to the selected landmark file: {file_path[0]}")

landmark_id = 0
DISPLAY_WIDTH = 800

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate new width and new canvas height
scale_factor = DISPLAY_WIDTH / frame_width
new_width = int(frame_width * scale_factor)
new_canvas_height = int((frame_height // 4) * scale_factor)

canvas_height = frame_height // 4

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Ensure the frame_number is within bounds
    if frame_number >= final_feature_matrix.shape[1]:
        continue  

    # Resize the video frame
    frame = resize_frame(frame, new_width)

    # Prepare a visualization canvas
    canvas = np.zeros((new_canvas_height, new_width, 3), dtype=np.uint8)

    for idx, desc in enumerate(selected_descriptors):
        feature_idx = descriptors.index(desc)
        value = final_feature_matrix[feature_idx, frame_number]
        cv2.putText(canvas, f"{desc}: {value:.2f}", (10, canvas.shape[0] - (90 - idx*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    combined_display = np.vstack((frame, canvas))
    
    cv2.imshow("Video and Feature Visualization", combined_display)
    key = cv2.waitKey(frame_rate)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()