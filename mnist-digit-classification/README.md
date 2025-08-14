# MNIST Handwritten Digit Classification

A neural network implementation for recognizing handwritten digits using the classic MNIST dataset. This project demonstrates fundamental deep learning concepts including data preprocessing, model architecture design, and training optimization.

## Overview

This project implements a neural network classifier for the MNIST dataset, one of the most well-known datasets in machine learning. The model learns to recognize handwritten digits (0-9) from 28x28 pixel grayscale images.

## Features

- **Interactive Jupyter Notebook**: Step-by-step implementation with explanations
- **Multiple Architectures**: Comparison of different neural network designs
- **Performance Visualization**: Training curves and prediction examples
- **High Accuracy**: Achieves >95% accuracy on test set
- **Beginner Friendly**: Clear documentation and educational focus

## Dataset

The MNIST dataset contains:

- **Training Set**: 60,000 handwritten digit images
- **Test Set**: 10,000 handwritten digit images
- **Image Size**: 28x28 pixels, grayscale
- **Classes**: 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
- **Source**: Automatically downloaded via PyTorch

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/mnist-digit-classification.git
cd mnist-digit-classification
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Launch the notebook

```bash
jupyter notebook MNIST_Handwritten_Digits.ipynb
```

## Usage

Open the Jupyter notebook to explore the complete implementation:

```bash
jupyter notebook MNIST_Handwritten_Digits.ipynb
```

The notebook includes:

- **Data Exploration**: Visualizing the MNIST dataset
- **Data Preprocessing**: Normalization and transformation
- **Model Architecture**: Neural network design and implementation
- **Training Process**: Loss optimization and gradient descent
- **Evaluation**: Accuracy metrics and confusion matrix
- **Predictions**: Testing on sample images

## Model Architecture

The neural network features:

- **Input Layer**: 784 neurons (28x28 flattened pixels)
- **Hidden Layers**: Multiple fully connected layers with ReLU activation
- **Output Layer**: 10 neurons (one for each digit class)
- **Optimization**: Adam optimizer with cross-entropy loss
- **Regularization**: Dropout layers to prevent overfitting

## Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: >95%
- **Training Time**: 5-10 minutes on CPU
- **Model Size**: <5MB

## Key Concepts Demonstrated

- **Neural Network Fundamentals**: Forward and backward propagation
- **Loss Functions**: Cross-entropy for multi-class classification
- **Optimization**: Gradient descent and Adam optimizer
- **Regularization**: Dropout and batch normalization
- **Evaluation Metrics**: Accuracy, precision, recall, confusion matrix

## Example Results

```text
Epoch 10/10: Train Loss: 0.045, Train Acc: 98.5%, Test Acc: 96.2%

Sample Predictions:
Image 1: Predicted: 7, Actual: 7 ✓
Image 2: Predicted: 2, Actual: 2 ✓
Image 3: Predicted: 1, Actual: 1 ✓
Image 4: Predicted: 0, Actual: 0 ✓
Image 5: Predicted: 4, Actual: 4 ✓
```

## Technical Stack

- **Framework**: PyTorch
- **Data Handling**: torchvision.datasets
- **Visualization**: Matplotlib
- **Computing**: CPU/GPU support
- **Environment**: Jupyter Notebook

## Educational Value

This project is ideal for:

- Learning neural network fundamentals
- Understanding PyTorch basics
- Exploring computer vision concepts
- Practicing model evaluation techniques
- Building intuition for deep learning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST dataset from Yann LeCun's website
- Part of Udacity's Machine Learning Nanodegree
- PyTorch framework and community
