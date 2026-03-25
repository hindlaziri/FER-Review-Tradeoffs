"""
Generate experimental figures for the FER paper:
1. Confusion matrices (LBP+SVM and CNN side by side)
2. CNN training curves (accuracy and loss)
3. Per-class F1-score comparison bar chart
4. Summary comparison table figure
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Load results
with open('/home/ubuntu/experiment_results.json') as f:
    results = json.load(f)

EMOTION_LABELS = results['emotion_labels']
svm = results['lbp_svm']
cnn = results['basic_cnn']

# ── FIGURE 1: Confusion Matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrices on FER2013 Test Set', fontsize=15, fontweight='bold', y=1.02)

cmap = LinearSegmentedColormap.from_list('fer_blue', ['#ffffff', '#1565c0'])

for ax, cm_data, title, acc in [
    (axes[0], svm['confusion_matrix'], 'LBP + SVM', svm['accuracy']),
    (axes[1], cnn['confusion_matrix'], 'Basic CNN', cnn['accuracy'])
]:
    cm = np.array(cm_data)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=ax, linewidths=0.5, linecolor='#e0e0e0',
                cbar_kws={'shrink': 0.8},
                vmin=0, vmax=1)
    ax.set_title(f'{title}\n(Accuracy: {acc*100:.2f}%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('/home/ubuntu/fig_confusion_matrices.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: fig_confusion_matrices.png")

# ── FIGURE 2: CNN Training Curves ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('Basic CNN Training History on FER2013', fontsize=14, fontweight='bold')

epochs = range(1, len(cnn['history']['train_acc']) + 1)

# Accuracy
axes[0].plot(epochs, [x*100 for x in cnn['history']['train_acc']],
             'b-o', markersize=4, linewidth=2, label='Training Accuracy')
axes[0].plot(epochs, [x*100 for x in cnn['history']['val_acc']],
             'r-s', markersize=4, linewidth=2, label='Validation Accuracy')
best_val = max(cnn['history']['val_acc'])
best_ep = cnn['history']['val_acc'].index(best_val) + 1
axes[0].axvline(x=best_ep, color='green', linestyle='--', alpha=0.7, label=f'Best val ({best_val*100:.2f}%)')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 100)

# Loss
axes[1].plot(epochs, cnn['history']['train_loss'],
             'b-o', markersize=4, linewidth=2, label='Training Loss')
axes[1].plot(epochs, cnn['history']['val_loss'],
             'r-s', markersize=4, linewidth=2, label='Validation Loss')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/fig_cnn_training_curves.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: fig_cnn_training_curves.png")

# ── FIGURE 3: Per-class F1-score comparison ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(EMOTION_LABELS))
width = 0.35

svm_f1 = [svm['per_class'].get(e, {}).get('f1-score', 0) * 100 for e in EMOTION_LABELS]
cnn_f1 = [cnn['per_class'].get(e, {}).get('f1-score', 0) * 100 for e in EMOTION_LABELS]

bars1 = ax.bar(x - width/2, svm_f1, width, label='LBP + SVM',
               color='#5c85d6', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, cnn_f1, width, label='Basic CNN',
               color='#e05c5c', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Emotion Class', fontsize=12)
ax.set_ylabel('F1-Score (%)', fontsize=12)
ax.set_title('Per-Class F1-Score: LBP+SVM vs. Basic CNN on FER2013', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(EMOTION_LABELS, fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 100)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

plt.tight_layout()
plt.savefig('/home/ubuntu/fig_per_class_f1.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: fig_per_class_f1.png")

# ── FIGURE 4: Summary comparison table (visual) ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')

table_data = [
    ['Metric', 'LBP + SVM', 'Basic CNN', 'Improvement'],
    ['Accuracy (%)', f'{svm["accuracy"]*100:.2f}', f'{cnn["accuracy"]*100:.2f}',
     f'+{(cnn["accuracy"]-svm["accuracy"])*100:.2f} pp'],
    ['Weighted F1-Score (%)', f'{svm["f1_weighted"]*100:.2f}', f'{cnn["f1_weighted"]*100:.2f}',
     f'+{(cnn["f1_weighted"]-svm["f1_weighted"])*100:.2f} pp'],
    ['Training Time (s)', f'{svm["train_time_s"]:.1f}', f'{cnn["train_time_s"]:.1f}',
     f'×{cnn["train_time_s"]/svm["train_time_s"]:.1f} longer'],
    ['Inference Time (s)', f'{svm["infer_time_s"]:.3f}', f'{cnn["infer_time_s"]:.3f}',
     f'×{svm["infer_time_s"]/cnn["infer_time_s"]:.1f} faster (CNN)'],
    ['Feature Engineering', 'Manual (LBP)', 'Automatic', '—'],
    ['Scalability', 'Limited', 'High', '—'],
]

col_colors = [['#1565c0']*4]
row_colors = [['#e3f2fd', '#e3f2fd', '#e3f2fd', '#e3f2fd']] * (len(table_data)-1)

table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center',
    loc='center',
    colColours=['#1565c0']*4,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Style header
for j in range(4):
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_facecolor('#1565c0')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[i, j].set_facecolor('#e8f4f8')
        else:
            table[i, j].set_facecolor('#f5f5f5')
        if j == 3:  # Improvement column
            table[i, j].set_text_props(color='#1b5e20', fontweight='bold')

ax.set_title('Table: Experimental Comparison — LBP+SVM vs. Basic CNN on FER2013',
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/ubuntu/fig_comparison_table.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: fig_comparison_table.png")

print("\nAll experimental figures generated successfully!")
print(f"\nKey Results Summary:")
print(f"  LBP+SVM  → Accuracy: {svm['accuracy']*100:.2f}%, F1: {svm['f1_weighted']*100:.2f}%")
print(f"  Basic CNN → Accuracy: {cnn['accuracy']*100:.2f}%, F1: {cnn['f1_weighted']*100:.2f}%")
print(f"  CNN improvement: +{(cnn['accuracy']-svm['accuracy'])*100:.2f} percentage points")
