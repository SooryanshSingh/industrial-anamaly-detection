import cv2
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(path, size=(128, 128)):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img

def load_dataset(image_dir, size=(128, 128)):
    images = []
    
    for img_path in image_dir.glob("*.png"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype("float32") / 255.0
        images.append(img)
        
    return np.array(images)

def load_and_preprocess(img_path, size=(128, 128)):
    img = cv2.imread(str(img_path))          # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)              
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)        # (1,128,128,3)
    return img


def reconstruction_heatmap(x, x_recon, percentile=90):
    error = np.mean((x - x_recon) ** 2, axis=-1)[0]

    thresh = np.percentile(error, percentile)
    mask = error >= thresh

    heatmap = np.zeros_like(error)
    heatmap[mask] = error[mask]

    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def anomaly_score(x, x_recon):
    return np.mean((x - x_recon) ** 2)


def max_anomaly_score(x, x_recon):
    # maximum pixel-wise reconstruction error
    return np.max((x - x_recon) ** 2)



def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    return intersection / (union + 1e-8)


def binarize_heatmap(heatmap, percentile=75):
    thresh = np.percentile(heatmap, percentile)
    return (heatmap >= thresh).astype(np.uint8)

def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0
    return intersection / union

