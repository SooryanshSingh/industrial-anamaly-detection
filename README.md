Industrial Anomaly Detection using Convolutional Autoencoders

This project explores unsupervised visual anomaly detection on industrial images using a convolutional autoencoder.
The goal is to detect and localize defects without using defect labels during training.

The work is based on the MVTec Anomaly Detection dataset (Bottle category) and focuses on both image-level detection and pixel-level localization.


Dataset

MVTec Anomaly Detection Dataset – Bottle Category

Dataset link:
https://www.mvtec.com/company/research/datasets/mvtec-ad

License: Provided by MVTec Software GmbH (not redistributed in this repository)

Dataset structure (Bottle)

Train: only normal (good) images

Test: normal + defect images

broken_large

broken_small

contamination

Ground truth masks: pixel-level defect annotations for test defects

Paths are configurable via src/data_locations.py

## How to Run

1. Download the MVTec Bottle dataset from the official source.

2. Update dataset paths in `src/data_locations.py`.

3. Run notebooks in order:

01_data_visualization.ipynb
02_model_training.ipynb
03_evaluation_and_visualization.ipynb

Notebook overview

01_data_visualization.ipynb

Dataset inspection

Sample images from each category

Understanding defect characteristics

02_model_training.ipynb

Autoencoder training using only normal images

Architecture imported from src/model.py

Reconstruction-based learning

03_evaluation_and_visualization.ipynb

Image-level anomaly scoring

Heatmap-based localization

ROC analysis

Pixel-level IoU evaluation using ground-truth masks


Model Overview

Architecture: Convolutional Autoencoder

Training: Unsupervised (normal images only)

Loss: Mean Squared Error (MSE)

Inference idea:

Normal images → low reconstruction error

Defective regions → high reconstruction error



Anomaly Scoring Methods

Two image-level anomaly scoring strategies were evaluated:

1. Mean Reconstruction Error (MSE)

Average pixel-wise reconstruction error

Common baseline in autoencoder-based anomaly detection

2. Max Reconstruction Error

Uses the maximum pixel error per image

More sensitive to localized defects

Demonstrated stronger separation for small anomalies


Evaluation Metrics
Image-level Metrics

ROC Curve

ROC–AUC

Max-based scoring achieved higher ROC–AUC (~0.82) compared to mean MSE

Thresholding

Percentile-based thresholds (e.g. 75th percentile)

Used to classify images as NORMAL / DEFECT

Higher thresholds reduce false alarms but increase the risk of missed defects, highlighting the trade-off between false positives and false negatives.

Pixel-level Metrics (Advanced)

Pixel-level IoU

Heatmaps are binarized

Compared against ground-truth defect masks

Mean IoU reported per defect category and overall

This pixel-level evaluation goes beyond simple visualization and provides a quantitative measure of localization quality.



Metric                 Value
----------------------------
ROC–AUC (Mean MSE)     ~0.63
ROC–AUC (Max Error)    ~0.82
Mean Pixel IoU         ~0.08


Limitations

Reconstruction-based autoencoders provide coarse localization and may miss subtle or very small defects. Pixel-level IoU scores reflect this limitation, motivating more advanced feature-based methods for precise segmentation.

## Author

Sooryansh Singh  
