DESIRED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

LANDMARK_NAMES = [
    'nose', 'left eye (inner)', 'left eye', 'left eye (outer)', 'right eye (inner)', 'right eye',
    'right eye (outer)', 'left ear', 'right ear', 'mouth (left)', 'mouth (right)', 'left shoulder',
    'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left pinky',
    'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel', 'right heel', 'left foot index',
    'right foot index'
]

DESIRED_LANDMARK_NAMES = [LANDMARK_NAMES[i] for i in DESIRED_LANDMARKS]

ANGLE_DEFINITIONS = {
    'angle_right_elbow': ['right shoulder', 'right elbow', 'right wrist'],
    'angle_right_shoulder': ['right elbow', 'right shoulder', 'right hip'],
    'angle_right_hip': ['right shoulder', 'right hip', 'right knee'],
    'angle_right_knee': ['right hip', 'right knee', 'right ankle'],
    'angle_left_elbow': ['left shoulder', 'left elbow', 'left wrist'],
    'angle_left_shoulder': ['left elbow', 'left shoulder', 'left hip'],
    'angle_left_hip': ['left shoulder', 'left hip', 'left knee'],
    'angle_left_knee': ['left hip', 'left knee', 'left ankle']
}

# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index