# MNIST Handwritten Digit Classification with CNN

This repository contains the implementation of a CNN for classifying handwritten digits from the MNIST dataset.

## Features

- **Dataset**: MNIST (70,000 grayscale images of handwritten digits)
- **Architecture**:
  - 2 Convolutional Layers with ReLU activation
  - Max Pooling for spatial dimension reduction
  - Fully Connected Layers with Dropout for regularization
- **Training**:
  - Adam optimizer with a learning rate of 0.001
  - Cross-entropy loss function
  - Batch size: 256
  - 10 epochs
- **Evaluation**:
  - Achieved test accuracy: **98.24%**
  - Detailed metrics: Precision, Recall, F1-Score
  - Visualized feature maps for CNN layers
