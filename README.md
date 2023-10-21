# ActivityClassificationUsingML

This project uses mediapipe for 3d skeleton and trains a random forest model to classify between daily activities in a video.

This readme demonstrates:

- **How to use the code** from the beggining till the end.
- **What each script does**.

## Table of Contents

- [How to use the code](#how-to-use-the-code)
- [What each script does](#what-each-script-does)


## How to use the code
### 1. Shoot training videos
In order to train the model to classify between N activities, you should shoot N different videos. In each video you should perform only and continuously one singular activity (e.g. talking on the phone while walking around, move around in the kitchen while cooking etc.). The video format tested so far is .mp4.

### 2. Visualize the landmarks (Optional)

- **Visualize the skeleton over a video**: Use the script `skeleton_video.py` to visualize the pose landmarks. Note that the `DESIRED_LANDMARKS`, that are considered for training and testing the model, will appear in green.

### 3. Combine training videos and create .csv with labels
In order to combine the training videos into one, while also creating a `.csv` that contains the label of each frame, run the script `combine_training_vids.py`. The script allows you to choose the folder where your separated training videos are found (e.g. `.\vids\train3`). Normally, the videos are combined into an `.mp4` named `train_all_vids`, and by default, the labels in `labels.csv` are named after the names of the training videos.


### 4. Use MediaPipe to retrieve and store the pose landmarks
The script `calculate_raw_worldlandmarks.py`  (or `calculate_raw_landmarks.py`) allows you to retrieve the  **x, y, z coordinates & the visibility** of worldlandmarks (or landmarks) from videos stored under `.\vids`. It stores the results under `.\raw_worldlandmarks` (or `.\raw_landmarks`). The numpy variables are named as `[video-name]_raw_worldlandmarks.npy` (or `[video-name]_raw_landmarks.npy`). New videos that don't have already been processed appear with \*. 

In order to save time and memory we don't reterieve all the landmarks, but only the `DESIRED_LANDMARKS` as defined in `config.py`.
> **Note:** If you want to change the `DESIRED_LANDMARKS`, it is advised to empty the `.\raw_worldlandmarks` (or `.\raw_landmarks`) folders and calculate all the raw landmarks from the beginning to avoid mess.

For the difference between worldlandmarks and landmarks check the [mediapipe guide for python](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python). In this project we prefer the <u>worldlandmarks</u> because they are in meters! :)


### 5. Calculate the *feature matrix* for training

Use the script `feature_matrix_train` to calculate the feature matrix from the raw landmarks. 

The *feature matrix* contains one column for every frame and the number of rows corresponds to the number of features that we choose to consider for our algorithm. 

We consider the following features:
- the position x, y, z of each landmark (3\*`N_lm`)
- the velocities dx, dy, dz of each landmark (3\*`N_lm`)
- Pair-distances between `DESIRED_LANDMARKS` (13\*`N_lm`)
- Visibility value of each landmark (`N_lm`)
- TODO: add angles

In total, we have 20\* `N_lm` features.

> Note: For non-existing values (e.g. hand was not in the video frame or it was hidden), we interpolate linearly between the previous measured value and the next measured value.
.


.


.


.


## What each script does

