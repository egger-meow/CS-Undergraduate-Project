import argparse, os, sys
import numpy as np

import pandas as pd

import imageio
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import gc
import matplotlib.pyplot as plt

from math import sqrt
import torch
from torchvision.utils import save_image
from settings import channels, autoEncoder, norm_trainDataDir, abnorm_trainDataDir, autoencoderNormPath, autoencoderAbnormPath, epochs
from settings import norm_testDataDir, abnorm_testDataDir

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from models.VAE import VAE
from models.AE import AE

from utils import get_interpolations

import joblib



arch = {
  'AE':   AE,
  'VAE': VAE,
}

model = arch[autoEncoder]

torch.manual_seed(42)
def scatterTestReuslt2D(blue_x, blue_y, red_x, red_y):

    # Plotting the points
    plt.figure(figsize=(8, 6))
    plt.scatter(blue_x, blue_y, color='blue', label='normal')
    plt.scatter(red_x, red_y, color='red', label='abnormal')

    # Adding labels and title
    plt.xlabel("loss-normalAE")
    plt.ylabel("loss-abnormalAE")
    plt.title("conbine the two plots above")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def normalization(data):
    scaler = MinMaxScaler()
    # Reshape to (-1, 1) to make it a 2D array for MinMaxScaler
    data_reshaped = [[x] for x in data]  
    normalized = scaler.fit_transform(data_reshaped)
    # Flatten the result back to a 1D list
    return [x[0] for x in normalized]

    
def dataPrepare(normDir, abnormDir, test_size = 0):
    aeNormal = model(test = True, normalVersion=True, modelPath = autoencoderNormPath)
    aeAbnormal = model(test = True, normalVersion=False, modelPath = autoencoderAbnormPath)
    
    loss_aeNormal_dataNormal        = aeNormal.test(normDir)
    loss_aeAbnormal_dataNormal      = aeAbnormal.test(normDir)
    loss_aeNormal_dataAbnormal      = aeNormal.test(abnormDir)
    loss_aeAbnormal_dataAbnormal    = aeAbnormal.test(abnormDir)
    
    splitPoint = len(loss_aeNormal_dataNormal)
    

    
    loss_aeNormal_normalized    = normalization(loss_aeNormal_dataNormal + loss_aeNormal_dataAbnormal)
    loss_aeAbnormal_normalized  = normalization(loss_aeAbnormal_dataNormal + loss_aeAbnormal_dataAbnormal)
    
    loss_aeNormal_dataNormal        = loss_aeNormal_normalized[:splitPoint]
    loss_aeAbnormal_dataNormal      = loss_aeAbnormal_normalized[:splitPoint]
    
    loss_aeNormal_dataAbnormal        = loss_aeNormal_normalized[splitPoint:]
    loss_aeAbnormal_dataAbnormal      = loss_aeAbnormal_normalized[splitPoint:]
    
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
    print(X,y)
    # print(X)
    # print(y)
    if test_size == 1:
        return None, X, None, y
    else:
        return X, None, y, None


def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n")
    
def train(X_train, X_test, y_train, y_test):
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # Initialize and Train Models
    svm_clf = SVC(kernel='linear', random_state=43, class_weight='balanced')
    svm_clf.fit(X_train, y_train)

    knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn_clf.fit(X_train, y_train)

    logreg_clf = LogisticRegression(random_state=43, class_weight='balanced')
    logreg_clf.fit(X_train, y_train)

    # Save the SVM model
    joblib.dump(svm_clf, 'svm_model.joblib')

    # Save the KNN model
    joblib.dump(knn_clf, 'knn_model.joblib')

    # Save the Logistic Regression model
    joblib.dump(logreg_clf, 'logreg_model.joblib')
    
    return svm_clf, knn_clf, logreg_clf


def test(X_test, y_test, svm_clf, knn_clf, logreg_clf):
    # scaler = StandardScaler()
    # X_test_scaled = scaler.fit_transform(X_test)

    # Make Predictions
    svm_preds = svm_clf.predict(X_test)
    knn_preds = knn_clf.predict(X_test)
    logreg_preds = logreg_clf.predict(X_test)
    
    evaluate_model("Support Vector Machine", y_test, svm_preds)
    evaluate_model("K-Nearest Neighbors", y_test, knn_preds)
    evaluate_model("Logistic Regression", y_test, logreg_preds)

def main():
    gc.collect()
    try:
        # X_train, X_test, y_train, y_test =  dataPrepare(norm_trainDataDir, abnorm_trainDataDir)

        # svm_clf, knn_clf, logreg_clf = train(X_train, X_test, y_train, y_test)
        svm_clf = joblib.load('checkpoints/svm_model1227.joblib')
        knn_clf = joblib.load('checkpoints/knn_model1227.joblib')
        logreg_clf = joblib.load('checkpoints/logreg_model1227.joblib')
        _, X_test, _, y_test =  dataPrepare(norm_testDataDir, abnorm_testDataDir, 1)
        
        test(X_test, y_test, svm_clf, knn_clf, logreg_clf)
        
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")        
    
if __name__ == "__main__":
    main()