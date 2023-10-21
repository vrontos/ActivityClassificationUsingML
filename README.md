# ActivityClassificationUsingML

This project uses mediapipe to generate 3D skeleton data and trains a random forest model to classify between daily activities in a video.

## 1. Shoot training videos
In order to train the model to classify between N activities, you should shoot N different videos. In each video you should perform only and continuously one singular activity (e.g. talking on the phone while walking around, move around in the kitchen while cooking etc.). The video format tested so far is .mp4.

## 2. Visualize the landmarks (Optional)

**Visualize the skeleton over a video**: Use the script `skeleton_video.py` to visualize the pose landmarks. Note that the DESIRED_LANDMARKS, that are considered for training and testing the model, will appear in green.

## 3. Combine training videos and create .csv with labels
In order to combine the training videos into one, while also creating a .csv that contains the label of each timestamp, run the script `combine_training_vids.py`. The script allows you to choose the folder where your separated training videos are found (e.g. .\vids\train3). Normally, the videos are combined into train_all_vids.mp4, and by default, the labels in labels.csv are named after the names of the training videos.


## 4. Use MediaPipe to retrieve and store the pose worldlandmarks
The script `calculate_raw_worldlandmarks.py` allows you to retrieve the  **x, y, z** coordinates & the **visibility** of worldlandmarks from videos stored under .\vids\xxx\yyy. It stores the results under .\raw_worldlandmarks\xxx\yyy. The numpy variables are named as \[video-name]_raw_worldlandmarks.npy.

In order to save time and memory we don't reterieve all the landmarks, but only the DESIRED_LANDMARKS as defined in `config.py`.
> **Note:** If you want to change the DESIRED_LANDMARKS, it is advised to empty the .\raw_worldlandmarks folders and calculate all the raw landmarks from the beginning to avoid mess.

## 5. Train the model
In `train_model.py`, the *final_feature_matrix* contains one column for every frame and the number of rows corresponds to the number of features that we choose to consider for our algorithm.

We consider the following features:
- the position x, y, z of each landmark (3\*N_lm),
- the velocities dx, dy, dz of each landmark (3\*N_lm),
- pair-distances between DESIRED_LANDMARKS (N_lm\*N_lm),
- visibility value of each landmark (N_lm),
- angle elbows, shoulders, hips, knees (8).

In total, we have <u>255</u> features. The calculation of the features and assembling the feature matrix is done in **`calculate_feature_matrix.py`**.

> Note: For non-existing values (e.g. hand was not in the video frame or it was hidden), we interpolate linearly between the previous measured value and the next measured value.

## 5. Calculate the feature matrix for training

Use the script `feature_matrix_train.py` to calculate the feature matrix from the raw landmarks. 

## 6. Visualize the feature matrix (Optional)

In order to make sure that the feature matrix has valid values we visualize different aspects of it:

- `display_feature_matrix.py`: This script plays back the video that corresponds to the selected raw landmarks, while it displays the chosen features (e.g. position of landmark, velocity of landmark).
- `animate_feature_matrix.py`: *maybe change it to animate_landmarks.py*
- `TODO: plot_feature_matrix.py` 

## 7. Train the model
Use the script `train_3d_model.py` to train the Random Forest model.

As meta-features, we use:

- mean
- std
- min
- max
- TODO: skewness