import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ============================================================
# FIGURE 1: FER Pipeline Flowchart
# ============================================================

fig1, ax1 = plt.subplots(figsize=(7, 12))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 22)
ax1.axis('off')

def draw_rect(ax, x, y, w, h, label, color='#D6E4F0', fontsize=10, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          linewidth=1.5, edgecolor='#2C3E50',
                          facecolor=color, zorder=2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=3,
            wrap=True, color='#1A1A2E')

def draw_diamond(ax, x, y, w, h, label, color='#D6E4F0', fontsize=9):
    diamond = plt.Polygon([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]],
                           closed=True, linewidth=1.5, edgecolor='#2C3E50',
                           facecolor=color, zorder=2)
    ax.add_patch(diamond)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='normal', zorder=3, color='#1A1A2E')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5),
                zorder=1)

def draw_group_box(ax, x, y, w, h, label, color='#FDFCE5', border='#A9C934'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.2",
                          linewidth=1.5, edgecolor=border,
                          facecolor=color, zorder=0, alpha=0.6)
    ax.add_patch(rect)
    ax.text(x - w/2 + 0.2, y + h/2 - 0.3, label,
            ha='left', va='top', fontsize=8, color='#5D6D7E', style='italic')

# --- Top: Input Image ---
draw_rect(ax1, 5, 21, 3.5, 0.9, 'Input Image', color='#D6E4F0', bold=True)
draw_arrow(ax1, 5, 20.55, 5, 19.85)
draw_rect(ax1, 5, 19.4, 3.5, 0.9, 'Preprocessing', color='#D6E4F0', bold=True)
draw_arrow(ax1, 5, 18.95, 5, 18.2)

# --- Preprocessing group box ---
draw_group_box(ax1, 5, 16.5, 7.5, 3.5, 'Preprocessing Steps')
draw_diamond(ax1, 5, 17.2, 4.5, 1.8, 'Face Detection\n& Alignment', color='#D6E4F0')
draw_arrow(ax1, 5, 16.3, 5, 15.55)
draw_rect(ax1, 5, 15.1, 4, 0.9, 'Normalization & Resizing', color='#D6E4F0')

draw_arrow(ax1, 5, 14.65, 5, 13.9)

# --- Feature Extraction group box ---
draw_group_box(ax1, 5, 11.5, 7.5, 4.8, 'Feature Extraction & Classification')
draw_diamond(ax1, 5, 13.2, 4.5, 1.5, 'Feature Extraction', color='#D6E4F0')

# Two branches from Feature Extraction
ax1.annotate('', xy=(2.8, 11.8), xytext=(3.8, 12.45),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
ax1.annotate('', xy=(7.2, 11.8), xytext=(6.2, 12.45),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

ax1.text(2.5, 12.1, 'Conventional\nMethods\n(LBP, HOG)', ha='center', va='center',
         fontsize=7.5, color='#5D6D7E', style='italic')
ax1.text(7.5, 12.1, 'Deep Learning\n(CNN, ViT)', ha='center', va='center',
         fontsize=7.5, color='#5D6D7E', style='italic')

draw_rect(ax1, 2.8, 11.1, 2.8, 0.9, 'SVM, K-NN', color='#D6E4F0')
draw_rect(ax1, 7.2, 11.1, 2.8, 0.9, 'Softmax, FC Layers', color='#D6E4F0')

# Classification diamond on the left
draw_arrow(ax1, 2.8, 10.65, 2.8, 9.9)
draw_diamond(ax1, 2.8, 9.3, 3.0, 1.2, 'Classification', color='#D6E4F0')

# Arrows to output
ax1.annotate('', xy=(4.0, 8.0), xytext=(2.8, 8.7),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
ax1.annotate('', xy=(6.0, 8.0), xytext=(7.2, 10.65),
             arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

draw_rect(ax1, 5, 7.6, 4.5, 0.9, 'Emotion Label Output', color='#A8D8EA', bold=True)

plt.tight_layout()
plt.savefig('/home/ubuntu/fer_pipeline.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 1 (pipeline) saved.")


# ============================================================
# FIGURE 2: CNN Architecture for FER
# ============================================================

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 8)
ax2.axis('off')

fig2.patch.set_facecolor('white')

def cnn_box(ax, x, y, w, h, label, sublabel='', color='#D6EAF8', edgecolor='#2E86C1', fontsize=9):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          linewidth=1.5, edgecolor=edgecolor,
                          facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y + (0.25 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#1A1A2E', zorder=3)
    if sublabel:
        ax.text(x, y - 0.35, sublabel, ha='center', va='center',
                fontsize=7.5, color='#555555', zorder=3)

def cnn_arrow(ax, x1, x2, y=4.0):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.0))

# Title
ax2.text(8, 7.5, 'CNN for Facial Emotion Recognition',
         ha='center', va='center', fontsize=14, fontweight='bold', color='#1A1A2E')

# Input Image
cnn_box(ax2, 1.2, 4.0, 1.8, 3.5, 'Input\nImage', '(Face)', color='#EBF5FB', edgecolor='#85929E')

cnn_arrow(ax2, 2.1, 2.8)

# Convolutional Layers group
group_rect = FancyBboxPatch((2.8, 1.5), 4.2, 5.0,
                             boxstyle="round,pad=0.1",
                             linewidth=1.5, edgecolor='#2E86C1',
                             facecolor='#EBF5FB', zorder=1, alpha=0.5)
ax2.add_patch(group_rect)
ax2.text(4.9, 6.7, 'Convolutional Layers', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#1A1A2E')
ax2.text(4.9, 6.3, '(Conv + ReLU + Pooling)', ha='center', va='center',
         fontsize=8, color='#555555')

for i, y_pos in enumerate([5.5, 4.0, 2.5]):
    cnn_box(ax2, 3.7, y_pos, 1.5, 0.9, 'Conv\n3x3, ReLU', color='#AED6F1', edgecolor='#2E86C1', fontsize=8)
    ax2.annotate('', xy=(4.6, y_pos), xytext=(4.45, y_pos),
                 arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
    cnn_box(ax2, 5.5, y_pos, 1.5, 0.9, 'Max\nPooling 2x2', color='#D5D8DC', edgecolor='#85929E', fontsize=8)

cnn_arrow(ax2, 7.0, 8.0)

# Flatten Layer
cnn_box(ax2, 8.7, 4.0, 1.4, 3.5, 'Flatten\nLayer', color='#EBF5FB', edgecolor='#85929E')

cnn_arrow(ax2, 9.4, 10.2)

# Fully Connected Layers group
fc_rect = FancyBboxPatch((10.2, 2.0), 2.8, 4.0,
                          boxstyle="round,pad=0.1",
                          linewidth=1.5, edgecolor='#2E86C1',
                          facecolor='#EBF5FB', zorder=1, alpha=0.5)
ax2.add_patch(fc_rect)
ax2.text(11.6, 6.2, 'Fully Connected\nLayers (FC)', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#1A1A2E')

cnn_box(ax2, 11.6, 5.0, 2.2, 0.9, 'FC 512, ReLU', color='#AED6F1', edgecolor='#2E86C1', fontsize=8.5)
cnn_box(ax2, 11.6, 3.0, 2.2, 0.9, 'FC 128, ReLU', color='#AED6F1', edgecolor='#2E86C1', fontsize=8.5)

cnn_arrow(ax2, 13.0, 13.8)

# Output Layer
cnn_box(ax2, 14.5, 4.0, 1.4, 3.5, 'Output\nLayer\n(Softmax)', color='#D5D8DC', edgecolor='#85929E')

# Emotion labels
emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Contempt']
y_positions = [6.2, 5.5, 4.8, 4.1, 3.4, 2.7, 2.0]
for emotion, yp in zip(emotions, y_positions):
    ax2.annotate('', xy=(15.5, yp), xytext=(15.2, yp),
                 arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2))
    ax2.text(15.6, yp, emotion, ha='left', va='center', fontsize=8.5, color='#1A1A2E')

plt.tight_layout()
plt.savefig('/home/ubuntu/cnn_architecture.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 2 (CNN architecture) saved.")


# ============================================================
# FIGURE 3: Trade-off Accuracy vs Computational Complexity
# ============================================================

fig3, ax3 = plt.subplots(figsize=(9, 6))
fig3.patch.set_facecolor('white')

methods = ['LBP + SVM', 'HOG + K-NN', 'Shallow CNN', 'Deep CNN\n(ResNet)', 'Transformer\n(ViT)']
accuracy = [75, 78, 85, 92, 94]
complexity = [0.5, 1.5, 5.0, 20.0, 50.0]
params_M = [0.01, 0.05, 0.5, 25.0, 86.0]  # millions of parameters (bubble size)

colors = ['#4B2D8A', '#2E7D8A', '#1A8A7A', '#7A9A1A', '#C8D41A']

scatter = ax3.scatter(complexity, accuracy,
                      s=[p * 6 + 50 for p in params_M],
                      c=colors,
                      alpha=0.85, edgecolors='#2C3E50', linewidths=1.2, zorder=3)

for i, (method, acc, comp) in enumerate(zip(methods, accuracy, complexity)):
    offset_x = 0.5
    offset_y = 0.4
    if i == 4:
        offset_x = -8
        offset_y = 0.4
    ax3.annotate(method,
                 xy=(comp, acc),
                 xytext=(comp + offset_x, acc + offset_y),
                 fontsize=8.5, color='#1A1A2E',
                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.8) if i == 4 else None)

ax3.set_xscale('log')
ax3.set_xlabel('Computational Complexity (Log Scale - Lower is Better)',
               fontsize=10, color='#2C3E50')
ax3.set_ylabel('Recognition Accuracy (%)', fontsize=10, color='#2C3E50')
ax3.set_title('Trade-off between Accuracy and Computational Complexity in FER',
              fontsize=11, fontweight='bold', color='#1A1A2E', pad=12)

ax3.set_xlim(0.2, 120)
ax3.set_ylim(72, 97)
ax3.grid(True, linestyle='--', alpha=0.4, color='gray')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Vertical dashed line separating conventional from deep learning
ax3.axvline(x=3.5, color='#85929E', linestyle='--', linewidth=1.0, alpha=0.7)
ax3.text(1.2, 73.5, 'Conventional\nMethods', ha='center', fontsize=8, color='#85929E', style='italic')
ax3.text(25, 73.5, 'Deep Learning\nMethods', ha='center', fontsize=8, color='#85929E', style='italic')

# Note on bubble size
ax3.text(0.98, 0.02, 'Bubble size ∝ Number of parameters',
         transform=ax3.transAxes, ha='right', va='bottom',
         fontsize=7.5, color='#85929E', style='italic')

plt.tight_layout()
plt.savefig('/home/ubuntu/fer_comparison_chart.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Figure 3 (comparison chart) saved.")

print("\nAll figures generated successfully!")
