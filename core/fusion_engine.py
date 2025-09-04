# core/fusion_engine.py - Fusion strategy implementation
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

from config import config

logger = logging.getLogger(__name__)

class FusionEngine:
    """
    Implements the fusion strategy from background.md:
    - OR fusion: pred_final = pred_amp OR pred_vib
    - Emphasis on high recall (better to over-report than miss anomalies)
    - Combines amp model (high precision) with vib model (high recall)
    """
    
    def __init__(self):
        self.config = config
        self.classifiers = self._initialize_classifiers()
        
    def _initialize_classifiers(self) -> Dict[str, object]:
        """Initialize classifiers mentioned in background.md"""
        return {
            'SVM': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=self.config.system.random_seed
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                random_state=self.config.system.random_seed,
                max_iter=1000
            ),
            'kNN': KNeighborsClassifier(
                n_neighbors=3,
                weights='distance'
            )
        }
    
    def train_classifier(self, features: np.ndarray, labels: np.ndarray, 
                        classifier_name: str) -> object:
        """Train a single classifier on the 2D features"""
        if classifier_name not in self.classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}")
            
        classifier = self.classifiers[classifier_name]
        
        # Apply scaling for SVM and LogReg (kNN works better with original scale)
        if classifier_name in ['SVM', 'LogisticRegression']:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            classifier.fit(features_scaled, labels)
            
            # Store scaler with classifier for later use
            classifier._scaler = scaler
        else:
            classifier.fit(features, labels)
            
        logger.info(f"Trained {classifier_name} classifier")
        return classifier
    
    def evaluate_classifier(self, classifier: object, features: np.ndarray, 
                          labels: np.ndarray, classifier_name: str) -> Dict[str, float]:
        """Evaluate classifier performance"""
        
        # Apply scaling if classifier has it
        if hasattr(classifier, '_scaler'):
            features_scaled = classifier._scaler.transform(features)
            predictions = classifier.predict(features_scaled)
            probabilities = classifier.predict_proba(features_scaled)[:, 1]
        else:
            predictions = classifier.predict(features)
            probabilities = classifier.predict_proba(features)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(labels, probabilities)
        }
        
        logger.info(f"{classifier_name} Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return metrics, predictions, probabilities
    
    def perform_or_fusion(self, predictions_amp: np.ndarray, 
                         predictions_vib: np.ndarray) -> np.ndarray:
        """
        Perform OR fusion as described in background.md:
        - If either model predicts abnormal (1), final prediction is abnormal
        - Sacrifices precision for higher recall (practical for maintenance)
        """
        fused_predictions = np.logical_or(predictions_amp, predictions_vib).astype(int)
        
        logger.info("OR Fusion Statistics:")
        logger.info(f"  Amp predictions: {np.sum(predictions_amp)} anomalies")
        logger.info(f"  Vib predictions: {np.sum(predictions_vib)} anomalies")
        logger.info(f"  Fused predictions: {np.sum(fused_predictions)} anomalies")
        
        return fused_predictions
    
    def cross_validate_model(self, features: np.ndarray, labels: np.ndarray,
                           classifier_name: str, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on a single model"""
        classifier = self.classifiers[classifier_name]
        
        # Prepare features (with scaling if needed)
        if classifier_name in ['SVM', 'LogisticRegression']:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = features
            
        cv_scores = {}
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                               random_state=self.config.system.random_seed)
        
        # Cross-validate different metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(
                classifier, features_scaled, labels,
                cv=kfold, scoring=metric, n_jobs=-1
            )
            cv_scores[f'{metric}_mean'] = np.mean(scores)
            cv_scores[f'{metric}_std'] = np.std(scores)
            
        return cv_scores
    
    def train_and_evaluate_pipeline(self, amp_features: np.ndarray, amp_labels: np.ndarray,
                                  vib_features: np.ndarray, vib_labels: np.ndarray,
                                  test_amp_features: np.ndarray, test_amp_labels: np.ndarray,
                                  test_vib_features: np.ndarray, test_vib_labels: np.ndarray) -> Dict:
        """
        Complete training and evaluation pipeline following background.md methodology:
        1. Train classifiers on amp and vib features separately
        2. Evaluate individual performance
        3. Perform OR fusion
        4. Compare results
        """
        results = {}
        
        # Train and evaluate amp classifiers
        logger.info("Training amplitude classifiers...")
        amp_classifiers = {}
        amp_metrics = {}
        amp_predictions = {}
        
        for clf_name in self.config.evaluation.classifiers:
            clf = self.train_classifier(amp_features, amp_labels, clf_name)
            metrics, preds, probs = self.evaluate_classifier(
                clf, test_amp_features, test_amp_labels, f"Amp-{clf_name}"
            )
            
            amp_classifiers[clf_name] = clf
            amp_metrics[clf_name] = metrics
            amp_predictions[clf_name] = preds
        
        # Train and evaluate vib classifiers
        logger.info("Training vibration classifiers...")
        vib_classifiers = {}
        vib_metrics = {}
        vib_predictions = {}
        
        for clf_name in self.config.evaluation.classifiers:
            clf = self.train_classifier(vib_features, vib_labels, clf_name)
            metrics, preds, probs = self.evaluate_classifier(
                clf, test_vib_features, test_vib_labels, f"Vib-{clf_name}"
            )
            
            vib_classifiers[clf_name] = clf
            vib_metrics[clf_name] = metrics
            vib_predictions[clf_name] = preds
        
        # Perform OR fusion for each classifier pair
        logger.info("Performing OR fusion...")
        fusion_metrics = {}
        
        for clf_name in self.config.evaluation.classifiers:
            fused_preds = self.perform_or_fusion(
                amp_predictions[clf_name], 
                vib_predictions[clf_name]
            )
            
            # Evaluate fused predictions (using amp test labels as reference)
            fusion_metrics[clf_name] = {
                'accuracy': accuracy_score(test_amp_labels, fused_preds),
                'precision': precision_score(test_amp_labels, fused_preds, zero_division=0),
                'recall': recall_score(test_amp_labels, fused_preds, zero_division=0),
                'f1_score': f1_score(test_amp_labels, fused_preds, zero_division=0)
            }
            
            logger.info(f"Fused {clf_name} Performance:")
            logger.info(f"  Accuracy: {fusion_metrics[clf_name]['accuracy']:.3f}")
            logger.info(f"  Precision: {fusion_metrics[clf_name]['precision']:.3f}")
            logger.info(f"  Recall: {fusion_metrics[clf_name]['recall']:.3f}")
            logger.info(f"  F1-Score: {fusion_metrics[clf_name]['f1_score']:.3f}")
        
        # Compile results
        results = {
            'amp_classifiers': amp_classifiers,
            'vib_classifiers': vib_classifiers,
            'amp_metrics': amp_metrics,
            'vib_metrics': vib_metrics,
            'fusion_metrics': fusion_metrics,
            'amp_predictions': amp_predictions,
            'vib_predictions': vib_predictions
        }
        
        return results
    
    def save_pipelines(self, amp_classifiers: Dict, vib_classifiers: Dict) -> None:
        """Save trained classifiers"""
        amp_paths = self.config.get_pipeline_paths('amp')
        vib_paths = self.config.get_pipeline_paths('vib')
        
        # Save amp classifiers
        for clf_name, clf in amp_classifiers.items():
            path = amp_paths.get(clf_name.lower())
            if path:
                joblib.dump(clf, path)
                logger.info(f"Saved {clf_name} amp classifier to {path}")
        
        # Save vib classifiers  
        for clf_name, clf in vib_classifiers.items():
            path = vib_paths.get(clf_name.lower())
            if path:
                joblib.dump(clf, path)
                logger.info(f"Saved {clf_name} vib classifier to {path}")
    
    def create_results_summary(self, results: Dict) -> pd.DataFrame:
        """Create a summary DataFrame of all results matching background.md table format"""
        
        summary_data = []
        
        # Extract amp results (focusing on best performing classifier)
        best_amp_clf = max(results['amp_metrics'].keys(), 
                          key=lambda x: results['amp_metrics'][x]['f1_score'])
        amp_metrics = results['amp_metrics'][best_amp_clf]
        
        summary_data.append({
            'Model': 'Amp (電流0)',
            'Accuracy': amp_metrics['accuracy'],
            'Precision': amp_metrics['precision'], 
            'Recall': amp_metrics['recall'],
            'F1-Score': amp_metrics['f1_score']
        })
        
        # Extract vib results
        best_vib_clf = max(results['vib_metrics'].keys(),
                          key=lambda x: results['vib_metrics'][x]['f1_score'])
        vib_metrics = results['vib_metrics'][best_vib_clf]
        
        summary_data.append({
            'Model': 'Vib (1,2,3)',
            'Accuracy': vib_metrics['accuracy'],
            'Precision': vib_metrics['precision'],
            'Recall': vib_metrics['recall'],
            'F1-Score': vib_metrics['f1_score']
        })
        
        # Extract fusion results
        best_fusion_clf = max(results['fusion_metrics'].keys(),
                             key=lambda x: results['fusion_metrics'][x]['f1_score'])
        fusion_metrics = results['fusion_metrics'][best_fusion_clf]
        
        summary_data.append({
            'Model': 'Mix (OR)',
            'Accuracy': fusion_metrics['accuracy'],
            'Precision': fusion_metrics['precision'],
            'Recall': fusion_metrics['recall'],
            'F1-Score': fusion_metrics['f1_score']
        })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format to match background.md table (2 decimal places)
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            summary_df[col] = summary_df[col].round(2)
            
        return summary_df
