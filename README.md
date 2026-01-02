# Image Classification Comparison (Naive Bayes vs Decision Tree vs MLP)

**ENCS3340 – Artificial Intelligence | Project #2**

This project presents a comparative study of image classification using three different machine learning models:
- Gaussian Naive Bayes
- Decision Tree
- Feedforward Neural Network (MLPClassifier)

The comparison is performed on a subset of the **Fashion-MNIST** dataset using standard evaluation metrics.

---

## Dataset
This project uses the **Fashion-MNIST** dataset.

- Source: https://www.openml.org/d/40996
- Description: 28×28 grayscale images of fashion items
- Subset used: 5 classes
- Samples: 800 images per class (4000 images total)

The dataset is loaded programmatically using:
`sklearn.datasets.fetch_openml`, so no manual download is required.

---

## Models Implemented
- **Gaussian Naive Bayes**
- **Decision Tree Classifier**
- **Feedforward Neural Network (MLPClassifier)**

---

## Methodology
1. Load and preprocess the dataset
2. Flatten and normalize image data
3. Split data into training and testing sets
4. Train each classifier
5. Evaluate performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
6. Generate confusion matrices and comparison plots

---

## Results Summary
Based on the experimental results:

- **Naive Bayes** achieved the lowest accuracy
- **Decision Tree** showed significant improvement
- **Neural Network (MLP)** achieved the highest accuracy and best overall performance

(Exact values and analysis are provided in the project report.)

---

## How to Run
```bash
python main.py
