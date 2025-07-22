import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold, cross_validate

from skimage import feature
from skimage.feature import graycomatrix, graycoprops
# from skimage.feature import greycomatrix, greycoprops
from glob import glob # Used to easily find file paths
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import recall_score, roc_auc_score, roc_curve, auc


# Loud the data folder and the images
col_names = ['REFNUM', 'BG', 'CLASS', 'SEVERITY', 'X', 'Y', 'RADIUS']
df = pd.read_csv('data2.txt', sep="\s+", names=col_names, header=None)
df['CANCER'] = df['SEVERITY'].apply(lambda x: 1 if x in ['B', 'M'] else 0)

images_path = "all-mias"

all_images = []
all_labels = []
all_groups = []

for filename in sorted(os.listdir(images_path)):
    if filename.lower().endswith('.pgm'):
        ref_num = os.path.splitext(filename)[0]
        record = df[df['REFNUM'] == ref_num]

        if not record.empty:
            full_path = os.path.join(images_path, filename)
            img_array = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_eq = clahe.apply(img_array)

            labels = record['CANCER'].iloc[0]
            x, y, radius = record[['X', 'Y', 'RADIUS']].iloc[0]
            
            # Handle ROI extraction
            if labels == 1 and pd.notna(x) and pd.notna(y) and pd.notna(radius):
                # Adjust for bottom-left origin: flip Y coordinate
                x, y, radius = int(x), int(1024 - float(y)), int(radius)  # Y = 1024 - y for top-left origin
                # Crop a square ROI around (x, y) with size 2*radius
                roi = img_eq[max(0, y-radius):min(1024, y+radius), max(0, x-radius):min(1024, x+radius)]
                # Ensure ROI is not empty; resize to a fixed size (e.g., 128x128) for consistency
                if roi.size > 0:
                    roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)
                else:
                    roi = img_eq  # Fallback to full image if ROI is invalid
            else:
                # For normal images or missing coordinates, use the entire image resized
                roi = cv2.resize(img_eq, (256, 256), interpolation=cv2.INTER_AREA)
            
            all_images.append(roi)
            all_labels.append(labels)
            all_groups.append(ref_num[:-1])

print("Data has been louded successfully")
# plt.imshow(img_eq, cmap='gray')
## Print out the table from dataset
print(f"Image list {len(all_images)}")
print(f"Labels list {len(all_labels)}")
df.head(5)

# print(all_images)
# for i in range(4):
#     print("#####")

# print(all_labels)

# for i in range(4):
#     print("#####")
    
# print(all_groups)

def extract_glcm_features(patch, 
                          distances=[1, 3, 5], 
                          angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                          levels=256):
    """
    Compute GLCM and summary stats for a single gray‑scale patch (2D ndarray).
    Returns a dict: e.g. {'contrast_mean':…, 'contrast_var':…, 'homogeneity_mean':…, …}
    """
    patch = (patch * (levels - 1)).astype(np.uint8) if patch.max() <= 1 else patch.astype(np.uint8)
    
    glcm = graycomatrix(patch,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    feats = {}
    for prop in props:
        mat = graycoprops(glcm, prop)  # shape = (len(distances), len(angles))
        feats[f'{prop}_mean'] = mat.mean()
        feats[f'{prop}_var']  = mat.var()
    return feats

# Loop over your ROIs, build a DataFrame of features + labels + group IDs base on GLCM function
records = [] # records the data
for roi, label, grp in zip(all_images, all_labels, all_groups):
    glcm_feats = extract_glcm_features(roi)
    glcm_feats['label'] = label
    glcm_feats['group'] = grp       
    records.append(glcm_feats)

    
df = pd.DataFrame.from_records(records) # sort the recorded data into the dataframe
print("GLCM feature matrix:", df.shape)
print(df.head())

# 3) Prepare for data training 
X = df.drop(columns=['label','group']).values
# y = df['label'].values  

X_glcm = X

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Quick stratified 5‑fold CV on a Random Forest
# clf = RandomForestClassifier(n_estimators=200, random_state=42)
# cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
# print(f"GLCM‑only mean AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")


normalaized_img = [] # holding the LBP maps
print("Computing the image and histogram for pooling and classification...")

for img in all_images:
    # Normalazing the image 
    if img.max() > img.min():
        n_img = ((img - img.min()) / (img.max() - img.min()) * 255) 
    else:
        n_img = img
    # normalaized_img.append(n_img)
    P = 8
    R = 3
    method = 'uniform'
    LBP = feature.local_binary_pattern(n_img, P, R, method)
#     bins_num = int(n_img.max()) + 1
    normalaized_img.append(LBP)
# print(LBP.min())
print("Normalazation process has been Done!.")
print("Start of the histogram processing...")

def compute_histograms_for_each_region(normalaized_img, G=6, n_bins=59):
    #compute th histogram from 255 in image size to 59 =>  reduce the number of bins
    histogram = []
    """
    # Pooling the image in to 6 x 6
    H = 1020 # height
    W = 1020 # width
    #G = 6 # Number of Grid 6 * 6 
    hor = int(H / G) # height of each region
    wor = int(W / G) # width of each region
    """
    H, W = normalaized_img.shape
    hor, wor = H // G, W // G
# Pooling the data into 6 * 6 regions
    for i in range(G):
        for j in range(G):
            row_start = i * hor
            row_end = (i + 1) * hor
            col_start = j * wor
            col_end = (i + 1) * wor
            region = normalaized_img[row_start:row_end, col_start:col_end]

            hist, _ = np.histogram(region.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float)
            if hist.sum() > 0:    
                hist /= hist.sum()
        
            histogram.append(hist)

    return histogram
x_list = []
y_list = []
for lbp_img, label in zip(normalaized_img, all_labels):
    histogram = compute_histograms_for_each_region(lbp_img)

    f_vector = np.concatenate(histogram)
    f_vector /= np.linalg.norm(f_vector) + 1e-10
#     print(f_vector.shape)

    x_list.append(f_vector)
    # lbp_feature_dict = {f'lbp_{i}': val for i, val in enumerate(f_vector)}
    # lbp_feature_dict['label'] = label  # optional here if you're not combining yet
    # x_list.append(lbp_feature_dict)
    y_list.append(label)

X = np.array(x_list)
y = np.array(y_list)

X_lbp = X

# print(hor)

print("Function has been done successfully")

assert X_glcm.shape[0] == X_lbp.shape[0] == y.shape[0], "Mismatch in sample counts!"
X_combined = np.hstack([X_glcm, X_lbp])
print("Fused feature shape : ", X_combined.shape)

scaler = StandardScaler()
X_ready = scaler.fit_transform(X_combined)
clf = RandomForestClassifier(n_estimators=200, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_ready, y, cv=cv, scoring='roc_auc')
print("ROC‑AUC:", scores.mean(), "±", scores.std())


scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    clf, X_scaled, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize():<8} : {scores.mean():.3f} ± {scores.std():.3f}")
unique_codes = np.unique(LBP)
print(" ")
print("Unique LBP codes in this image:", unique_codes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,    # for reproducibility
    stratify=y          # preserves class ratios
)

rf_model = RandomForestClassifier(
    n_estimators=250,
    class_weight = 'balanced',
    max_depth=None,
    min_samples_split=10,
    random_state=65,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 5. Predict
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)  # if you need probabilities


print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))