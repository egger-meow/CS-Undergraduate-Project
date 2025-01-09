import argparse
import os
import sys
import numpy as np
import pandas as pd
import imageio
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc
import matplotlib.pyplot as plt
from math import sqrt
import torch
from torchvision.utils import save_image
from settings import (
    channels, autoEncoder, norm_trainDataDir, abnorm_trainDataDir,
    autoencoderNormPath, autoencoderAbnormPath, epochs,
    norm_testDataDir, abnorm_testDataDir,
    phaseII_trainSetPath, phaseII_testSetPath
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import logging
import seaborn as sns

def evaluate_model(name, y_true, y_pred, y_pred_proba=None, csv_path='result.csv'):
    """
    Evaluates the model's performance and saves the metrics to a CSV file.

    Parameters:
    - name (str): Name of the model.
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - y_pred_proba (array-like, optional): Predicted probabilities for ROC-AUC.
    - csv_path (str): Path to the CSV file where results will be saved.
    """
    print(f"--- {name} ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.2f}")
    else:
        roc_auc = None

    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\n")

    # Prepare the data to be saved
    data = {
        'Model': [name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    }

    # Optionally include ROC-AUC if available
    if roc_auc is not None:
        data['ROC-AUC'] = [roc_auc]

    df = pd.DataFrame(data)

    # Check if the CSV file exists
    if not os.path.isfile(csv_path):
        # If not, create it and write the header
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        # If it exists, append without writing the header
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Results appended to {csv_path}")

# Load the results
amp, y_test_amp = joblib.load('amp_result.joblib')
vib, y_test_vib = joblib.load('vib_result.joblib')

# Assuming y_test_amp and y_test_vib are the same
# If not, ensure they are aligned or handle accordingly
y_test = y_test_amp  # or use y_test_vib if appropriate

# Combine the results (assuming bitwise OR operation is intended)
result = amp | vib

# Evaluate and save the results
# Uncomment and use the following lines if you want to evaluate individual models as well
# evaluate_model("amp", y_test, amp)
# evaluate_model("vib", y_test, vib)

evaluate_model("mix", y_test, result)
evaluate_model("vib", y_test, vib)
evaluate_model("amp", y_test, amp)
