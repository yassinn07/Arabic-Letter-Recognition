# Arabic Handwritten Character Recognition

This project implements and compares different machine learning and deep learning models to recognize Arabic handwritten characters using a labeled dataset of 32x32 grayscale images.

## 📁 Dataset

The dataset used is from `Arabic-Characters-Recognition`, provided as CSV files:
- `csvTrainImages 13440x1024.csv`: Flattened training images (13440 samples)
- `csvTrainLabel 13440x1.csv`: Training labels
- `csvTestImages 3360x1024.csv`: Flattened test images (3360 samples)
- `csvTestLabel 3360x1.csv`: Test labels

Each image is of shape 32x32, and labels correspond to 29 unique Arabic letters.

## 🧪 Models Compared

### 1. Support Vector Machine (SVM)
- Kernel: RBF
- Preprocessing: Normalized image pixel values
- Metric: Weighted F1-score

### 2. K-Nearest Neighbors (KNN)
- Hyperparameter tuning for `k` values: 1, 3, 5, 7, 9
- Evaluated using weighted F1-score
- Best `k` selected based on validation results

### 3. Neural Networks (Feedforward DNN)
Two architectures were tested:
