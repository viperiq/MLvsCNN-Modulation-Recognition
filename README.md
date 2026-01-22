# MLvsCNN-Modulation-Recognition
Design and train a lightweight Convolutional Neural Network to recognize modulation schemes (BPSK, QPSK, 16-QAM, etc.) directly from I/Q samples. Compare CNN accuracy with classical ML models. 
# Dataset: 
RadioML Dataset (GitHub mirror)
https://github.com/radioML/dataset
# Reference:
O’Shea, T. J., & Corgan, J. (2016). Convolutional Radio Modulation
Recognition Networks. arXiv:1602.04105 https://arxiv.org/abs/1602.04105
# CNN-Based Automatic Modulation Recognition (AMR)

# CNN-Based Automatic Modulation Recognition (AMR)

> Lightweight CNN for Automatic Modulation Recognition using raw I/Q samples, with comparison against classical machine learning models.

---

## Overview

This repository implements a lightweight Convolutional Neural Network (CNN) for **Automatic Modulation Recognition (AMR)** using raw In-phase and Quadrature (I/Q) samples. The CNN learns discriminative temporal features directly from the signal without handcrafted feature extraction and is benchmarked against classical machine learning models.

The project is developed as part of the **M.Sc. course in Wireless Communications and Machine Learning** (Fall 2025).
## Goal

* Design and train a lightweight CNN to recognize modulation schemes directly from raw I/Q samples
* Support common digital and analog modulations (e.g., BPSK, QPSK, QAM16, etc.)
* Compare CNN performance with classical ML classifiers (KNN, SVM, Random Forest)

## Dataset

* **Name:** RadioML 2016.10a
* **Type:** Synthetic complex baseband I/Q samples
* **Conditions:** SNR range from −20 dB to +20 dB
* **Input Shape:** `(N_samples, 2, 128)`

### Modulation Classes (11)

* AM-DSB, AM-SSB
* BPSK, QPSK, 8PSK
* PAM4
* CPFSK, GFSK
* QAM16, QAM64
* WBFM

## Methodology

### CNN Model

* Based on the **VT-CNN2** architecture
* Operates directly on raw I/Q samples
* Convolutional layers extract shift-invariant temporal features
* ReLU activation and Dropout for regularization
* Softmax output layer for multi-class classification

### Classical ML Baselines

For fair comparison, I/Q samples are flattened into 1D vectors:

* **K-Nearest Neighbors (KNN)** (k = 7)
* **Support Vector Machine (SVM)** with RBF kernel
* **Random Forest (RF)** (100 trees)

## Training Details

* Train/Test split: 50% / 50%
* Optimizer: Adam
* Loss Function: Categorical Cross-Entropy
* Batch size: 1024
* Epochs: 100

## Evaluation Metrics

* Overall classification accuracy
* Accuracy versus SNR
* Confusion matrix

## Results Summary

* CNN outperforms classical ML models across all SNR levels
* Significant performance gain at high SNR
* Robust feature learning without manual feature extraction

## Tools and Frameworks

* Python
* TensorFlow or PyTorch
* NumPy, SciPy
* Scikit-learn

## How to Run

1. Clone the repository
2. Install dependencies
3. Download the RadioML 2016.10a dataset
4. Run the training script or open the provided notebook

## Reference

O’Shea, T. J., & Corgan, J. (2016). *Convolutional Radio Modulation Recognition Networks*. arXiv:1602.04105

## Code

Full implementation and experiments are available at:

[https://colab.research.google.com/drive/1-S9z0C1--yxZino740fPgzwKmstE6p2D](https://colab.research.google.com/drive/1-S9z0C1--yxZino740fPgzwKmstE6p2D)
