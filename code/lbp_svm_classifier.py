"""
LBP + SVM Classifier for Facial Emotion Recognition
Paper ID: 27738 - TELKOMNIKA Journal

This script implements a conventional machine learning approach using:
- Local Binary Pattern (LBP) for feature extraction
- Support Vector Machine (SVM) for classification

Author: Research Team
Date: 2026-03-25
"""

import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from skimage.feature import local_binary_pattern
import pickle
import json

warnings.filterwarnings('ignore')


class LBPFeatureExtractor:
    """Extract Local Binary Pattern features from images."""
    
    def __init__(self, radius=1, n_points=8, n_bins=256, method='uniform'):
        """
        Initialize LBP feature extractor.
        
        Args:
            radius (int): Radius of the LBP neighborhood
            n_points (int): Number of points in the neighborhood
            n_bins (int): Number of histogram bins
            method (str): LBP method ('uniform', 'nri_uniform', 'var')
        """
        self.radius = radius
        self.n_points = n_points
        self.n_bins = n_bins
        self.method = method
    
    def extract(self, images):
        """
        Extract LBP features from a batch of images.
        
        Args:
            images (np.ndarray): Array of grayscale images (N, H, W)
            
        Returns:
            np.ndarray: LBP histogram features (N, n_bins)
        """
        features = []
        for img in images:
            # Compute LBP
            lbp = local_binary_pattern(
                img, 
                self.n_points, 
                self.radius, 
                method=self.method
            )
            
            # Create histogram
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=self.n_bins,
                range=(0, self.n_bins),
                density=True
            )
            features.append(hist)
        
        return np.array(features)


class LBPSVMClassifier:
    """LBP + SVM classifier for facial emotion recognition."""
    
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, lbp_params=None, svm_params=None):
        """
        Initialize LBP+SVM classifier.
        
        Args:
            lbp_params (dict): LBP feature extraction parameters
            svm_params (dict): SVM hyperparameters
        """
        # Default LBP parameters
        if lbp_params is None:
            lbp_params = {
                'radius': 1,
                'n_points': 8,
                'n_bins': 256,
                'method': 'uniform'
            }
        
        # Default SVM parameters
        if svm_params is None:
            svm_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42,
                'verbose': False
            }
        
        self.lbp_extractor = LBPFeatureExtractor(**lbp_params)
        self.svm_params = svm_params
        self.svm = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def fit(self, X_train, y_train):
        """
        Train the LBP+SVM classifier.
        
        Args:
            X_train (np.ndarray): Training images (N, H, W)
            y_train (np.ndarray): Training labels (N,)
            
        Returns:
            dict: Training statistics
        """
        print("LBP+SVM Training Pipeline")
        print("=" * 60)
        
        # Extract LBP features
        print("Step 1: Extracting LBP features from training data...")
        t0 = time.time()
        X_train_lbp = self.lbp_extractor.extract(X_train)
        t_feature = time.time() - t0
        print(f"  ✓ Feature extraction: {t_feature:.2f}s")
        print(f"  ✓ Feature shape: {X_train_lbp.shape}")
        
        # Normalize features
        print("\nStep 2: Normalizing features...")
        X_train_lbp_scaled = self.scaler.fit_transform(X_train_lbp)
        print(f"  ✓ Normalization complete")
        
        # Train SVM
        print("\nStep 3: Training SVM classifier...")
        print(f"  Parameters: {self.svm_params}")
        t0 = time.time()
        self.svm = SVC(**self.svm_params)
        self.svm.fit(X_train_lbp_scaled, y_train)
        t_train = time.time() - t0
        print(f"  ✓ Training time: {t_train:.2f}s")
        print(f"  ✓ Support vectors: {len(self.svm.support_vectors_)}")
        
        self.is_trained = True
        
        return {
            'feature_extraction_time': t_feature,
            'training_time': t_train,
            'total_time': t_feature + t_train,
            'n_support_vectors': len(self.svm.support_vectors_)
        }
    
    def predict(self, X_test):
        """
        Predict emotion labels for test images.
        
        Args:
            X_test (np.ndarray): Test images (N, H, W)
            
        Returns:
            np.ndarray: Predicted labels (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Extract features
        X_test_lbp = self.lbp_extractor.extract(X_test)
        
        # Normalize
        X_test_lbp_scaled = self.scaler.transform(X_test_lbp)
        
        # Predict
        predictions = self.svm.predict(X_test_lbp_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (np.ndarray): Test images (N, H, W)
            y_test (np.ndarray): Test labels (N,)
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nLBP+SVM Evaluation")
        print("=" * 60)
        
        # Predict
        print("Generating predictions...")
        t0 = time.time()
        y_pred = self.predict(X_test)
        t_infer = time.time() - t0
        
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  Inference time: {t_infer:.3f}s ({len(y_test)} samples)")
        
        # Per-class metrics
        print("\nPer-class metrics:")
        report = classification_report(
            y_test, y_pred,
            target_names=self.EMOTION_LABELS,
            output_dict=True
        )
        
        for emotion in self.EMOTION_LABELS:
            if emotion in report:
                metrics = report[emotion]
                print(f"  {emotion:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return {
            'accuracy': float(acc),
            'f1_weighted': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': cm.tolist(),
            'per_class': {k: v for k, v in report.items() if k in self.EMOTION_LABELS},
            'inference_time': float(t_infer),
            'n_samples': len(y_test)
        }
    
    def save(self, filepath):
        """Save trained model to disk."""
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'lbp_params': {
                'radius': self.lbp_extractor.radius,
                'n_points': self.lbp_extractor.n_points,
                'n_bins': self.lbp_extractor.n_bins,
                'method': self.lbp_extractor.method
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("LBP+SVM Classifier Module")
    print("This module provides LBP feature extraction and SVM classification")
    print("for facial emotion recognition tasks.")
