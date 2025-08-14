# Neural Networks Training

A comprehensive PyTorch tutorial demonstrating neural network training fundamentals using the CIFAR-10 dataset. This project covers the complete deep learning pipeline from data loading to model evaluation.

## Overview

This project provides hands-on experience with training neural networks from scratch using PyTorch. It walks through the entire process of building, training, and evaluating a convolutional neural network for image classification on the CIFAR-10 dataset.

## Features

- **Complete Training Pipeline**: Data loading, preprocessing, training, and evaluation
- **CIFAR-10 Classification**: 10-class image classification with 60,000 images
- **Interactive Learning**: Jupyter notebook with detailed explanations and visualizations
- **Best Practices**: Demonstrates modern deep learning techniques and PyTorch patterns
- **Performance Tracking**: Training metrics visualization and model evaluation

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- 50,000 training images and 10,000 test images
- Automatically downloaded when running the notebook

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/neural-networks-training.git
cd neural-networks-training
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Launch the notebook

```bash
jupyter notebook Training_Neural_Networks.ipynb
```

## Usage

Open the Jupyter notebook and follow along with the interactive tutorial:

```bash
jupyter notebook Training_Neural_Networks.ipynb
```

The notebook covers:

- **Data Loading**: Using PyTorch DataLoader and transforms
- **Model Architecture**: Building a CNN from scratch
- **Training Loop**: Forward pass, loss calculation, backpropagation
- **Evaluation**: Testing model performance and accuracy metrics
- **Visualization**: Loss curves and sample predictions

## Model Architecture

The neural network includes:

- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- ReLU activation functions
- Dropout for regularization

## Results

- **Training Accuracy**: ~85-90% after 10 epochs
- **Test Accuracy**: ~80-85% on unseen data
- **Training Time**: ~5-10 minutes on GPU

## Key Learning Objectives

- Understanding PyTorch DataLoader and Dataset classes
- Implementing custom neural network architectures
- Training loop implementation with loss and optimizer
- Model evaluation and performance metrics
- Data augmentation and preprocessing techniques

## Technical Stack

- **Framework**: PyTorch
- **Dataset**: CIFAR-10
- **Environment**: Jupyter Notebook
- **Visualization**: Matplotlib
- **Computing**: CPU/GPU support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CIFAR-10 dataset from the Canadian Institute for Advanced Research
- Part of Udacity's Machine Learning Nanodegree program
