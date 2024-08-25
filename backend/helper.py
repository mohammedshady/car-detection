import sys
import time
import h5py
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.ndimage.measurements import label

# Extract Features (histogram,hog,binning) from a single image
def extract_feature(image,hog_color, hog_channel, hist_color, spatial_size,
    spatial_color, orient, cell_per_block,pix_per_cell, hist_bins):
    file_features = []
    spatial_features = bin_spatial(image, size=spatial_size)
    file_features.append(spatial_features)
    hist_features = color_hist(image, nbins=hist_bins)
    file_features.append(hist_features)
    hog_features = extract_hog_features(image, orient, pix_per_cell, cell_per_block)
    file_features.append(hog_features)
    return np.concatenate(file_features)

# Extract histogram features
def color_hist(image, nbins=32, bins_range=(0, 256)):
    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    channel1_hist = np.histogram(feature_image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Extract binning features
def bin_spatial(image, size=(32, 32)):
    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return cv2.resize(feature_image, size).ravel()

# Extract hog features
def extract_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False):
    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hog_features = []
    # 3 Channels
    for channel in range(feature_image.shape[2]):
        hog_features.extend(cal_hog_feature(feature_image[:,:,channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=True))
    return hog_features

# call hog function on single channel
def cal_hog_feature(img, orient, pix_per_cell, cell_per_block, vis, feature_vec):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, feature_vector=feature_vec)
    return features


def slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # Set default window boundaries if not specified
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = image.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = image.shape[0]

    # List to collect all the window coordinates
    window_list = []
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    rate = 1/2  # Scaling factor for the vertical position of the windows

    # Predefined window presets and positions based on image height for far/close cars
    xy_window_list = [
        [200, 170, int(yspan*rate), yspan, 0, xspan],
        [150, 120, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate*rate), 0, xspan],
        [120, 110, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate), 0, xspan],
        [120, 96, int(yspan*rate*rate*rate), yspan-int(yspan*rate), 0, xspan],
        [84, 64, 0, int(yspan*rate*rate), 0, xspan]
    ]

    # Calculate step sizes for sliding windows based on overlap
    for xy_window in xy_window_list:
        nx_pix_per_step = int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = int(xy_window[1] * (1 - xy_overlap[1]))

        # Calculate the number of windows possible along x and y dimensions
        nx_windows = int((xy_window[5] - xy_window[4]) / nx_pix_per_step) - 1
        ny_windows = int((xy_window[3] - xy_window[2]) / ny_pix_per_step) - 1

        # Generate windows by iterating over each position increment
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs * nx_pix_per_step + x_start_stop[0] + xy_window[4]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0] + xy_window[2]
                endy = starty + xy_window[1]
                window_list.append(((startx, starty), (endx, endy)))

    return window_list

# heatmaps that shows the important windows
def add_heat(heatmap, bbox_list, value=1):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += value
    return heatmap

# removes redundant windows
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

# search in window for class car
def search_windows(image, windows, svc, scaler, hog_color, hog_channel,
                   spatial_size, spatial_color, hist_color,
                   spatial_feat, orient, cell_per_block):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_feature(test_img, hog_color="YCrCb", hog_channel="ALL", hist_color="YCrCb", spatial_size=(16, 16),
     spatial_color="YCrCb", orient=18, cell_per_block=3,pix_per_cell=8, hist_bins=32)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = svc.predict(test_features)
        if prediction == 1:
            if svc.decision_function(test_features)[0] > 0.3:
                on_windows.append([window, svc.decision_function(test_features)[0]])
    return on_windows

# draw borders on image
def draw_labeled_bboxes(img, labels):
    bbox_list = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img, bbox_list