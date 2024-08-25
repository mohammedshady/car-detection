import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from helper import slide_window, search_windows, add_heat, apply_threshold, draw_labeled_bboxes

svc_file = 'svm.pkl'
scaler_file = 'scaler.npz'


def process_image(test_image,output_path,emit_progress):

    def emit_progress_wrapper(message, progress):
        emit_progress(message + f" ({progress:.2f}%)")

    emit_progress_wrapper('Loading model', 0)
    with open(svc_file, "rb") as f:
        svc = pickle.load(f)
    
    scaler = np.load(scaler_file)
    X_scaler = StandardScaler()
    X_scaler.mean_, X_scaler.scale_ = scaler['mean'], scaler['scale']

    image = cv2.imread(test_image)
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    emit_progress_wrapper('Extracting windows', 20)
    # Generate windows
    windows = slide_window(converted_image, x_start_stop=[None, None], y_start_stop=[390, image.shape[0] - 50],
                        xy_window=(240, 160), xy_overlap=(0.9, 0.9))
    emit_progress_wrapper('windows detection', 40)
    # Detect vehicles
    bboxes = search_windows(converted_image, windows, svc, X_scaler, hog_color="YCrCb", hog_channel="ALL",
                            spatial_size=(16, 16), spatial_color="YCrCb", hist_color="YCrCb",
                            spatial_feat=True, orient=18, cell_per_block=2)
    emit_progress_wrapper('heatmap and thresholding', 70)

    heatmap = np.zeros_like(image[:, :, 0])
    if bboxes:
        heatmap = add_heat(heatmap, [bbox[0] for bbox in bboxes], value=2)
    heatmap = apply_threshold(heatmap, threshold=12)
    labels = label(heatmap)
    emit_progress_wrapper('drawing boxes', 90)
    result_img, bbox_list = draw_labeled_bboxes(np.copy(image), labels)
    cv2.imwrite(output_path,result_img )
    emit_progress_wrapper('finished', 100)

    return result_img
