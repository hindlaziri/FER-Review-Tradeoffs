import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path

# Import existing modules
from train_and_evaluate import FER2013DataLoader
from sota_model_pytorch import SOTAEfficientNet, FERDatasetPyTorch, train_sota_model, evaluate_sota_model

def main():
    # Configuration
    FER2013_PATH = '/home/ubuntu/fer2013_data/fer2013/fer2013.csv'
    OUTPUT_DIR = Path('./results')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data using existing loader
    X_train, y_train, X_test, y_test = FER2013DataLoader.load_dataset(FER2013_PATH)
    
    # Split train into train/val (10% val)
    val_size = int(0.1 * len(X_train))
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train_final, y_train_final = X_train[train_idx], y_train[train_idx]
    
    # 2. Initialize SOTA Model
    sota_wrapper = SOTAEfficientNet(model_name='efficientnet_b0', n_classes=7, pretrained=True)
    transform = sota_wrapper.get_transforms()
    
    # 3. Create PyTorch DataLoaders
    train_dataset = FERDatasetPyTorch(X_train_final, y_train_final, transform=transform)
    val_dataset = FERDatasetPyTorch(X_val, y_val, transform=transform)
    test_dataset = FERDatasetPyTorch(X_test, y_test, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Train (Fine-tuning)
    # Note: In a real scenario, we would train for more epochs. 
    # For this implementation, we provide the full logic.
    t_train = train_sota_model(sota_wrapper, train_loader, val_loader, epochs=5, lr=1e-4)
    
    # 5. Evaluate
    metrics = evaluate_sota_model(sota_wrapper, test_loader)
    metrics['training_time'] = t_train
    
    # 6. Save Results
    results_path = OUTPUT_DIR / 'sota_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSOTA results saved to {results_path}")
    
    # 7. Final Comparison Summary
    print("\n" + "=" * 60)
    print("FINAL COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 45)
    print(f"{'LBP + SVM':<20} | {27.96:<10}% | {21.48:<10}%")
    print(f"{'Basic CNN':<20} | {62.16:<10}% | {61.60:<10}%")
    print(f"{'EfficientNet-B0':<20} | {metrics['accuracy']*100:<10.2f}% | {metrics['f1_weighted']*100:<10.2f}%")

if __name__ == "__main__":
    main()
