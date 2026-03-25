"""
Complete Training and Evaluation Pipeline for FER
Paper ID: 27738 - TELKOMNIKA Journal

This script orchestrates the complete pipeline:
1. Load FER2013 dataset
2. Train LBP+SVM classifier
3. Train Deep Learning CNN
4. Compare results
5. Generate evaluation reports

Author: Research Team
Date: 2026-03-25
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
from lbp_svm_classifier import LBPSVMClassifier
from deep_learning_cnn import BasicCNNModel


class FER2013DataLoader:
    """Load and preprocess FER2013 dataset."""
    
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    IMG_SIZE = 48
    
    @staticmethod
    def parse_pixels(pixel_str):
        """Parse pixel string to image array."""
        return np.array(pixel_str.split(), dtype=np.uint8).reshape(FER2013DataLoader.IMG_SIZE, FER2013DataLoader.IMG_SIZE)
    
    @staticmethod
    def load_dataset(csv_path):
        """
        Load FER2013 dataset from CSV.
        
        Args:
            csv_path (str): Path to FER2013 CSV file
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print("Loading FER2013 Dataset")
        print("=" * 60)
        
        # Load CSV
        print(f"Reading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Total samples: {len(df)}")
        
        # Split into train and test
        train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
        test_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])].reset_index(drop=True)
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        # Parse images
        print("\nParsing images...")
        X_train = np.stack(train_df['pixels'].apply(FER2013DataLoader.parse_pixels).values)
        y_train = train_df['emotion'].values
        X_test = np.stack(test_df['pixels'].apply(FER2013DataLoader.parse_pixels).values)
        y_test = test_df['emotion'].values
        
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        
        # Print class distribution
        print("\nClass distribution (train):")
        for i, emotion in enumerate(FER2013DataLoader.EMOTION_LABELS):
            count = np.sum(y_train == i)
            pct = 100 * count / len(y_train)
            print(f"  {emotion:12s}: {count:5d} ({pct:5.1f}%)")
        
        return X_train, y_train, X_test, y_test


class ComparisonExperiment:
    """Run comparative experiment between conventional and deep learning methods."""
    
    def __init__(self, output_dir='./results'):
        """Initialize experiment."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_lbp_svm(self, X_train, y_train, X_test, y_test):
        """Train and evaluate LBP+SVM classifier."""
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: LBP + SVM (Conventional Method)")
        print("=" * 60)
        
        # Initialize classifier
        lbp_svm = LBPSVMClassifier(
            lbp_params={
                'radius': 1,
                'n_points': 8,
                'n_bins': 256,
                'method': 'uniform'
            },
            svm_params={
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42,
                'verbose': False
            }
        )
        
        # Train
        train_stats = lbp_svm.fit(X_train, y_train)
        
        # Evaluate
        eval_metrics = lbp_svm.evaluate(X_test, y_test)
        
        # Store results
        self.results['lbp_svm'] = {
            'training': train_stats,
            'evaluation': eval_metrics
        }
        
        return lbp_svm
    
    def run_deep_learning_cnn(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Deep Learning CNN."""
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Deep Learning CNN")
        print("=" * 60)
        
        # Normalize images
        X_train_norm = X_train.astype('float32') / 255.0
        X_test_norm = X_test.astype('float32') / 255.0
        X_train_norm = X_train_norm[..., np.newaxis]
        X_test_norm = X_test_norm[..., np.newaxis]
        
        # Initialize model
        cnn = BasicCNNModel(
            input_shape=(48, 48, 1),
            n_classes=7,
            model_name='BasicCNN_FER'
        )
        
        # Build and compile
        cnn.build_model()
        cnn.compile_model(optimizer='adam', learning_rate=1e-3)
        
        # Print model summary
        print("\nModel Architecture:")
        cnn.summary()
        
        # Train
        train_stats = cnn.train(
            X_train_norm, y_train,
            validation_split=0.1,
            epochs=30,
            batch_size=64
        )
        
        # Evaluate
        eval_metrics = cnn.evaluate(X_test_norm, y_test, batch_size=128)
        
        # Store results
        self.results['deep_learning_cnn'] = {
            'training': train_stats,
            'evaluation': eval_metrics
        }
        
        return cnn
    
    def compare_results(self):
        """Compare results between methods."""
        print("\n" + "=" * 60)
        print("COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        lbp_acc = self.results['lbp_svm']['evaluation']['accuracy']
        cnn_acc = self.results['deep_learning_cnn']['evaluation']['accuracy']
        
        lbp_f1 = self.results['lbp_svm']['evaluation']['f1_weighted']
        cnn_f1 = self.results['deep_learning_cnn']['evaluation']['f1_weighted']
        
        lbp_train_time = self.results['lbp_svm']['training']['training_time']
        cnn_train_time = self.results['deep_learning_cnn']['training']['training_time']
        
        lbp_infer_time = self.results['lbp_svm']['evaluation']['inference_time']
        cnn_infer_time = self.results['deep_learning_cnn']['evaluation']['inference_time']
        
        print("\nPerformance Comparison:")
        print(f"{'Metric':<25} {'LBP+SVM':>15} {'CNN':>15} {'Improvement':>15}")
        print("-" * 70)
        print(f"{'Accuracy (%)':<25} {lbp_acc*100:>14.2f}% {cnn_acc*100:>14.2f}% {(cnn_acc-lbp_acc)*100:>14.2f} pp")
        print(f"{'F1-Score (%)':<25} {lbp_f1*100:>14.2f}% {cnn_f1*100:>14.2f}% {(cnn_f1-lbp_f1)*100:>14.2f} pp")
        print(f"{'Training Time (s)':<25} {lbp_train_time:>15.2f} {cnn_train_time:>15.2f} {cnn_train_time/lbp_train_time:>14.1f}x longer")
        print(f"{'Inference Time (s)':<25} {lbp_infer_time:>15.3f} {cnn_infer_time:>15.3f} {lbp_infer_time/cnn_infer_time:>14.1f}x faster (CNN)")
        
        print("\nKey Findings:")
        print(f"  • CNN accuracy improvement: +{(cnn_acc-lbp_acc)*100:.2f} percentage points")
        print(f"  • CNN F1-score improvement: +{(cnn_f1-lbp_f1)*100:.2f} percentage points")
        print(f"  • CNN training is {cnn_train_time/lbp_train_time:.1f}x slower")
        print(f"  • CNN inference is {lbp_infer_time/cnn_infer_time:.1f}x faster")
    
    def save_results(self, filename='experiment_results.json'):
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def generate_report(self, filename='experiment_report.txt'):
        """Generate text report."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FER COMPARATIVE STUDY - EXPERIMENT REPORT\n")
            f.write("Paper ID: 27738 - TELKOMNIKA Journal\n")
            f.write("=" * 70 + "\n\n")
            
            # LBP+SVM Results
            f.write("EXPERIMENT 1: LBP + SVM (Conventional Method)\n")
            f.write("-" * 70 + "\n")
            lbp_eval = self.results['lbp_svm']['evaluation']
            f.write(f"Accuracy:  {lbp_eval['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score:  {lbp_eval['f1_weighted']*100:.2f}%\n")
            f.write(f"Precision: {lbp_eval['precision']*100:.2f}%\n")
            f.write(f"Recall:    {lbp_eval['recall']*100:.2f}%\n")
            f.write(f"Inference Time: {lbp_eval['inference_time']:.3f}s\n\n")
            
            # CNN Results
            f.write("EXPERIMENT 2: Deep Learning CNN\n")
            f.write("-" * 70 + "\n")
            cnn_eval = self.results['deep_learning_cnn']['evaluation']
            f.write(f"Accuracy:  {cnn_eval['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score:  {cnn_eval['f1_weighted']*100:.2f}%\n")
            f.write(f"Precision: {cnn_eval['precision']*100:.2f}%\n")
            f.write(f"Recall:    {cnn_eval['recall']*100:.2f}%\n")
            f.write(f"Inference Time: {cnn_eval['inference_time']:.3f}s\n\n")
            
            # Comparison
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("-" * 70 + "\n")
            acc_imp = (cnn_eval['accuracy'] - lbp_eval['accuracy']) * 100
            f1_imp = (cnn_eval['f1_weighted'] - lbp_eval['f1_weighted']) * 100
            f.write(f"CNN Accuracy Improvement: +{acc_imp:.2f} percentage points\n")
            f.write(f"CNN F1-Score Improvement: +{f1_imp:.2f} percentage points\n")
        
        print(f"Report saved to {filepath}")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("FER COMPARATIVE STUDY - COMPLETE PIPELINE")
    print("Paper ID: 27738 - TELKOMNIKA Journal")
    print("=" * 60 + "\n")
    
    # Configuration
    FER2013_PATH = '/home/ubuntu/fer2013_data/fer2013/fer2013.csv'
    OUTPUT_DIR = './results'
    
    # Load dataset
    X_train, y_train, X_test, y_test = FER2013DataLoader.load_dataset(FER2013_PATH)
    
    # Run experiments
    experiment = ComparisonExperiment(output_dir=OUTPUT_DIR)
    
    # Experiment 1: LBP+SVM
    lbp_svm = experiment.run_lbp_svm(X_train, y_train, X_test, y_test)
    
    # Experiment 2: Deep Learning CNN
    cnn = experiment.run_deep_learning_cnn(X_train, y_train, X_test, y_test)
    
    # Compare results
    experiment.compare_results()
    
    # Save results
    experiment.save_results()
    experiment.generate_report()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
