"""
Deep Learning CNN for Facial Emotion Recognition
Paper ID: 27738 - TELKOMNIKA Journal

This script implements a deep learning approach using:
- Convolutional Neural Networks (CNN)
- TensorFlow/Keras framework
- Multiple architectural variants

Author: Research Team
Date: 2026-03-25
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import time
import json
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class BasicCNNModel:
    """Basic CNN model for facial emotion recognition."""
    
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, input_shape=(48, 48, 1), n_classes=7, model_name='BasicCNN'):
        """
        Initialize CNN model.
        
        Args:
            input_shape (tuple): Input image shape (H, W, C)
            n_classes (int): Number of emotion classes
            model_name (str): Model name
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.is_trained = False
    
    def build_model(self):
        """Build the CNN architecture."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1: Conv + BatchNorm + MaxPool + Dropout
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2: Conv + BatchNorm + MaxPool + Dropout
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3: Conv + BatchNorm + MaxPool + Dropout
            layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Flatten
            layers.Flatten(name='flatten'),
            
            # Fully connected layers
            layers.Dense(512, activation='relu', name='fc1'),
            layers.BatchNormalization(name='bn_fc1'),
            layers.Dropout(0.5, name='dropout_fc1'),
            
            layers.Dense(128, activation='relu', name='fc2'),
            layers.BatchNormalization(name='bn_fc2'),
            layers.Dropout(0.5, name='dropout_fc2'),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax', name='output')
        ], name=self.model_name)
        
        self.model = model
        return model
    
    def compile_model(self, optimizer='adam', learning_rate=1e-3):
        """
        Compile the model.
        
        Args:
            optimizer (str): Optimizer type ('adam', 'sgd', 'rmsprop')
            learning_rate (float): Learning rate
        """
        if self.model is None:
            self.build_model()
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_callbacks(self):
        """Get training callbacks."""
        return [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
    
    def train(self, X_train, y_train, validation_split=0.1, epochs=30, batch_size=64):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training images (N, H, W, C)
            y_train (np.ndarray): Training labels (N,)
            validation_split (float): Validation split ratio
            epochs (int): Number of epochs
            batch_size (int): Batch size
            
        Returns:
            dict: Training statistics
        """
        print("Deep Learning CNN Training Pipeline")
        print("=" * 60)
        
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        print(f"Model: {self.model_name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Training samples: {len(X_train)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Validation split: {validation_split}")
        
        # Train
        print("\nTraining...")
        t0 = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        t_train = time.time() - t0
        
        self.is_trained = True
        
        print(f"\nTraining completed in {t_train:.2f}s")
        
        return {
            'training_time': t_train,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_acc': float(self.history.history['accuracy'][-1]),
            'final_val_acc': float(self.history.history['val_accuracy'][-1]),
            'best_val_acc': float(max(self.history.history['val_accuracy']))
        }
    
    def predict(self, X_test, batch_size=128):
        """
        Predict emotion labels for test images.
        
        Args:
            X_test (np.ndarray): Test images (N, H, W, C)
            batch_size (int): Batch size for prediction
            
        Returns:
            np.ndarray: Predicted labels (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get predictions
        t0 = time.time()
        y_pred_prob = self.model.predict(X_test, batch_size=batch_size, verbose=0)
        t_infer = time.time() - t0
        
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        return y_pred, t_infer
    
    def evaluate(self, X_test, y_test, batch_size=128):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (np.ndarray): Test images (N, H, W, C)
            y_test (np.ndarray): Test labels (N,)
            batch_size (int): Batch size for prediction
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nDeep Learning CNN Evaluation")
        print("=" * 60)
        
        # Predict
        print("Generating predictions...")
        y_pred, t_infer = self.predict(X_test, batch_size=batch_size)
        
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
            'n_samples': len(y_test),
            'training_history': {
                'train_acc': [float(x) for x in self.history.history['accuracy']],
                'val_acc': [float(x) for x in self.history.history['val_accuracy']],
                'train_loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']]
            }
        }
    
    def save(self, filepath):
        """Save trained model to disk."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model from disk."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()


if __name__ == "__main__":
    print("Deep Learning CNN Module")
    print("This module provides CNN architectures for facial emotion recognition")
    print("using TensorFlow/Keras framework.")
