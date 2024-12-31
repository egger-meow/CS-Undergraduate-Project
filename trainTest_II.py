import argparse, os, sys
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
    norm_testDataDir, abnorm_testDataDir
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from models.VAE import VAE
from models.AE import AE
from utils import get_interpolations
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import logging
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    filename='model_pipeline.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

arch = {
    'AE': AE,
    'VAE': VAE,
}

model = arch[autoEncoder]
torch.manual_seed(42)

def scatterTestReuslt2D(blue_x, blue_y, red_x, red_y):
    plt.figure(figsize=(8, 6))
    plt.scatter(blue_x, blue_y, color='blue', label='normal')
    plt.scatter(red_x, red_y, color='red', label='abnormal')
    plt.xlabel("loss-normalAE")
    plt.ylabel("loss-abnormalAE")
    plt.title("Combined Loss Plots")
    plt.legend()
    plt.grid(True)
    plt.show()

def normalization(data):
    scaler = MinMaxScaler()
    data_reshaped = [[x] for x in data]
    normalized = scaler.fit_transform(data_reshaped)
    return [x[0] for x in normalized]

def dataPrepare(normDir, abnormDir, test_size=0.2):
    """Prepare features/labels by obtaining reconstruction losses from normal & abnormal autoencoders."""
    aeNormal = model(test=True, normalVersion=True, modelPath=autoencoderNormPath)
    aeAbnormal = model(test=True, normalVersion=False, modelPath=autoencoderAbnormPath)
    
    # Obtain loss values
    loss_aeNormal_dataNormal = aeNormal.test(normDir)
    loss_aeAbnormal_dataNormal = aeAbnormal.test(normDir)
    loss_aeNormal_dataAbnormal = aeNormal.test(abnormDir)
    loss_aeAbnormal_dataAbnormal = aeAbnormal.test(abnormDir)
    
    splitPoint = len(loss_aeNormal_dataNormal)
    
    # Normalize loss values
    loss_aeNormal_normalized = normalization(
        loss_aeNormal_dataNormal + loss_aeNormal_dataAbnormal
    )
    loss_aeAbnormal_normalized = normalization(
        loss_aeAbnormal_dataNormal + loss_aeAbnormal_dataAbnormal
    )
    
    # Split back into normal and abnormal
    loss_aeNormal_dataNormal = loss_aeNormal_normalized[:splitPoint]
    loss_aeAbnormal_dataNormal = loss_aeAbnormal_normalized[:splitPoint]
    
    loss_aeNormal_dataAbnormal = loss_aeNormal_normalized[splitPoint:]
    loss_aeAbnormal_dataAbnormal = loss_aeAbnormal_normalized[splitPoint:]
    
    # Prepare Features and Labels
    feature_normal = loss_aeNormal_dataNormal
    feature_abnormal = loss_aeAbnormal_dataNormal
    labels_normal = [0] * len(feature_normal)

    feature_normal_abn = loss_aeNormal_dataAbnormal
    feature_abnormal_abn = loss_aeAbnormal_dataAbnormal
    labels_abnormal = [1] * len(feature_normal_abn)

    X_normal = np.column_stack((feature_normal, feature_abnormal))
    y_normal = np.array(labels_normal)

    X_abnormal = np.column_stack((feature_normal_abn, feature_abnormal_abn))
    y_abnormal = np.array(labels_abnormal)

    X = np.vstack((X_normal, X_abnormal))
    y = np.concatenate((y_normal, y_abnormal))

    # Create DataFrame
    df = pd.DataFrame({
        'loss_aeNormal': X[:, 0],
        'loss_aeAbnormal': X[:, 1],
        'label': y
    })

    # Split Data
    X = df[['loss_aeNormal', 'loss_aeAbnormal']].values
    y = df['label'].values

    print("Feature Matrix X:\n", X)
    print("Labels y:\n", y)

    # Perform stratified train-test split
    if 0 < test_size < 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    elif test_size == 1:
        # All data as test set
        return None, X, None, y
    elif test_size == 0:
        # All data as training set
        return X, None, y, None
    else:
        raise ValueError(
            "test_size should be between 0 and 1, or exactly 0 or 1."
        )

def evaluate_model(name, y_true, y_pred, y_pred_proba=None):
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, zero_division=0):.2f}")
    if y_pred_proba is not None:
        print(f"ROC-AUC Score: {roc_auc_score(y_true, y_pred_proba):.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\n")

def train_phase():
    """Train models if not already existing (or retrain if you prefer)."""
    logging.info("Starting training process...")

    # Check if we have saved training data; if not, prepare it
    if os.path.exists('train_set.joblib'):
        X_train, y_train = joblib.load('train_set.joblib')
        logging.info("Loaded existing training set from train_set.joblib.")
    else:
        logging.info("train_set.joblib not found. Preparing training data.")
        X_train, _, y_train, _  = dataPrepare(norm_trainDataDir, abnorm_trainDataDir, test_size=0)
        joblib.dump((X_train, y_train), 'train_set.joblib')
        logging.info("Training set prepared and saved to train_set.joblib.")

    # Create pipelines using imblearn's Pipeline (which supports SMOTE)
    pipeline_svm = Pipeline([
        # ('scaler', StandardScaler()),
        # ('smote', SMOTE(random_state=42)),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=43))
    ])

    pipeline_knn = Pipeline([
        # ('scaler', StandardScaler()),
        # ('smote', SMOTE(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance'))
    ])

    pipeline_logreg = Pipeline([
        # ('scaler', StandardScaler()),
        # ('smote', SMOTE(random_state=42)),
        ('logreg', LogisticRegression(class_weight='balanced', random_state=43))
    ])

    # Fit each pipeline
    pipeline_svm.fit(X_train, y_train)
    logging.info("SVM pipeline trained.")
    
    pipeline_knn.fit(X_train, y_train)
    logging.info("KNN pipeline trained.")
    
    pipeline_logreg.fit(X_train, y_train)
    logging.info("Logistic Regression pipeline trained.")

    # Save pipelines
    joblib.dump(pipeline_svm, 'checkpoints/svm_pipeline.joblib')
    joblib.dump(pipeline_knn, 'checkpoints/knn_pipeline.joblib')
    joblib.dump(pipeline_logreg, 'checkpoints/logreg_pipeline.joblib')
    logging.info("Trained models saved: svm_pipeline.joblib, knn_pipeline.joblib, logreg_pipeline.joblib")

def test_phase():
    """Evaluate existing models on test set."""
    logging.info("Starting testing phase...")

    # Load trained models
    if not all(os.path.exists(x) for x in [
        'checkpoints/svm_pipeline.joblib',
        'checkpoints/knn_pipeline.joblib',
        'checkpoints/logreg_pipeline.joblib'
    ]):
        raise FileNotFoundError(
            "No trained model found. Please train first using the --train flag."
        )

    pipeline_svm = joblib.load('checkpoints/svm_pipeline.joblib')
    pipeline_knn = joblib.load('checkpoints/knn_pipeline.joblib')
    pipeline_logreg = joblib.load('checkpoints/logreg_pipeline.joblib')

    # Load or prepare test data
    if os.path.exists('test_set.joblib'):
        X_test, y_test = joblib.load('test_set.joblib')
        logging.info("Loaded existing test set from test_set.joblib.")
    else:
        logging.info("test_set.joblib not found. Preparing test data.")
        _, X_test, _, y_test = dataPrepare(norm_testDataDir, abnorm_testDataDir, test_size=1)
        joblib.dump((X_test, y_test), 'test_set.joblib')
        logging.info("Test set prepared and saved to test_set.joblib.")

    print("Evaluating models...")
    svm_preds = pipeline_svm.predict(X_test)
    svm_proba = pipeline_svm.predict_proba(X_test)[:, 1]

    knn_preds = pipeline_knn.predict(X_test)
    knn_proba = pipeline_knn.predict_proba(X_test)[:, 1]

    logreg_preds = pipeline_logreg.predict(X_test)
    logreg_proba = pipeline_logreg.predict_proba(X_test)[:, 1]

    evaluate_model("Support Vector Machine", y_test, svm_preds, svm_proba)
    evaluate_model("K-Nearest Neighbors", y_test, knn_preds, knn_proba)
    evaluate_model("Logistic Regression", y_test, logreg_preds, logreg_proba)

def main():
    """Command-line entry point: use -train, -test, or both."""
    gc.collect()

    parser = argparse.ArgumentParser(description="Train and/or Test anomaly detection models.")
    parser.add_argument('-train', action='store_true', help='Train the model(s).')
    parser.add_argument('-test',  action='store_true', help='Test the model(s).')
    args = parser.parse_args()

    # If no flags passed, do nothing (or you could default to train+test)
    if not args.train and not args.test:
        print("No flags specified. Use -train to train, -test to test, or both.")
        return

    # If train flag set
    if args.train:
        print("Running TRAIN phase...")
        train_phase()
        print("TRAIN phase completed.\n")

    # If test flag set
    if args.test:
        print("Running TEST phase...")
        test_phase()
        print("TEST phase completed.\n")

if __name__ == "__main__":
    main()
