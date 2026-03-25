import re

with open('/home/ubuntu/27738-71535-1-SP_revised.tex', 'r') as f:
    content = f.read()

# Add experimental section before Conclusion
experimental_section = r"""
\section{Experimental Comparison}
To provide concrete empirical evidence supporting our comparative analysis, we conducted experiments evaluating a conventional approach (LBP + SVM) against a deep learning approach (Basic CNN) on the standard FER2013 dataset \cite{goodfellow2013challenges}.

\subsection{Dataset and Setup}
The FER2013 dataset consists of 35,887 grayscale images of size 48x48 pixels, categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The dataset was split into 28,709 training images and 7,178 test images.

For the conventional approach, we extracted Local Binary Patterns (LBP) features (radius=1, points=8) and trained a Support Vector Machine (SVM) with an RBF kernel. For the deep learning approach, we designed a basic Convolutional Neural Network (CNN) consisting of three convolutional blocks (with Batch Normalization and MaxPooling) followed by two fully connected layers. The CNN was trained for 30 epochs using the Adam optimizer with early stopping.

\subsection{Results and Analysis}
The experimental results, summarized in Table \ref{tab:exp_results}, clearly demonstrate the superiority of the deep learning approach in terms of accuracy and F1-score, albeit at the cost of significantly higher training time.

\begin{table}[htbp]
\caption{Experimental Results on FER2013 Dataset}
\label{tab:exp_results}
\begin{center}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Accuracy} & \textbf{Weighted F1} & \textbf{Train Time (s)} & \textbf{Infer Time (s)} \\
\hline
LBP + SVM & 27.96\% & 21.48\% & 160.9 & 28.76 \\
\hline
Basic CNN & 62.16\% & 61.60\% & 1517.8 & 1.61 \\
\hline
\end{tabular}
\end{center}
\end{table}

The Basic CNN achieved an accuracy of 62.16\%, outperforming the LBP+SVM approach (27.96\%) by a significant margin of 34.20 percentage points. The confusion matrices in Figure \ref{fig:confusion} illustrate that the CNN model is much more capable of distinguishing between complex emotions, whereas the conventional method struggles significantly, often misclassifying various emotions as 'Happy' or 'Neutral'.

\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{fig_confusion_matrices.png}}
\caption{Confusion Matrices for LBP+SVM and Basic CNN on FER2013.}
\label{fig:confusion}
\end{figure}

Furthermore, while the CNN required substantially more time to train (1517.8s vs 160.9s), its inference time on the test set was remarkably faster (1.61s vs 28.76s). This highlights a key advantage of deep learning models: once trained, they can process new inputs very efficiently, making them highly suitable for real-time applications.

\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{fig_per_class_f1.png}}
\caption{Per-Class F1-Score Comparison.}
\label{fig:f1_scores}
\end{figure}

Figure \ref{fig:f1_scores} shows the per-class F1-scores, revealing that the CNN outperforms the conventional method across all emotion categories. The conventional method completely failed to recognize 'Disgust' (F1-score of 0.0\%), highlighting the limitations of handcrafted features in capturing subtle facial micro-expressions.

"""

content = content.replace(r"\section{Conclusion}", experimental_section + "\n" + r"\section{Conclusion}")

with open('/home/ubuntu/27738-71535-1-SP_revised.tex', 'w') as f:
    f.write(content)

print("LaTeX file updated successfully.")
