# Logistic Regression: Breast Cancer Wisconsin Dataset

## 🧠 Objective
This project builds a binary classification model using **Logistic Regression** to predict whether a tumor is **malignant** or **benign** based on various features extracted from breast mass images.

---

## 📊 Dataset Overview
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features**: 30 numeric features such as radius, texture, perimeter, area, smoothness, etc.
- **Target**: 
  - `M` (Malignant)
  - `B` (Benign)

---

## 🔧 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🔁 Workflow

1. **Load and explore the dataset**
2. **Preprocess** the data:
   - Drop irrelevant columns
   - Encode target labels (`M` = 1, `B` = 0)
3. **Split** into training and testing sets
4. **Standardize** features
5. **Train** a Logistic Regression model
6. **Evaluate** using:
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - ROC Curve and AUC Score
7. **Visualize** predicted probability distributions and threshold behavior

---

## 🚀 Getting Started

### 📁 Prerequisites

Make sure the following Python libraries are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

