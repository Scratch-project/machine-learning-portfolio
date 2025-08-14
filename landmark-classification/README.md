# Landmark Classification

A CNN-powered application that automatically predicts the location of images based on landmarks depicted in them. This project classifies images into 50 different landmark categories from around the world using both custom CNN architectures and transfer learning approaches.

## Overview

This landmark classification system helps identify famous landmarks in user-supplied images, enabling automatic location tagging for photo sharing services. The project demonstrates advanced deep learning techniques including CNN architecture design, transfer learning, and web application development.

## Features

- **Multiple CNN Approaches**: Custom CNN from scratch and transfer learning implementations
- **50 Landmark Classes**: Recognizes famous landmarks from around the world
- **Interactive Web App**: User-friendly interface for landmark prediction
- **High Accuracy**: Optimized models for reliable landmark detection
- **GPU Support**: Efficient training and inference on both CPU and GPU

## Project Structure

```text
├── cnn_from_scratch.ipynb           # Custom CNN implementation
├── transfer_learning.ipynb          # Transfer learning approach
├── app.ipynb                        # Interactive web application
├── src/                             # Source code modules
│   ├── data.py                      # Data loading and preprocessing
│   ├── model.py                     # Model architectures
│   ├── train.py                     # Training utilities
│   ├── predictor.py                 # Prediction functions
│   └── helpers.py                   # Helper functions
├── static_images/                   # Sample images and assets
└── requirements.txt                 # Python dependencies
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/landmark-classification.git
cd landmark-classification
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download the landmark dataset

```bash
# Dataset will be automatically handled by the notebooks
# Or download from Google Landmarks Dataset v2
```

## Usage

### CNN from Scratch

Open and run the custom CNN notebook:

```bash
jupyter notebook cnn_from_scratch.ipynb
```

This notebook provides:

- Custom CNN architecture design
- Training from scratch on landmark data
- Model evaluation and visualization
- Performance analysis

### Transfer Learning

Launch the transfer learning notebook:

```bash
jupyter notebook transfer_learning.ipynb
```

Features include:

- Pre-trained model fine-tuning
- Comparison of different architectures
- Optimized training strategies
- Advanced evaluation metrics

### Web Application

Run the interactive app:

```bash
jupyter notebook app.ipynb
```

The app offers:

- User-friendly image upload interface
- Real-time landmark prediction
- Top-k prediction results
- Confidence score visualization

## Model Performance

- **Architecture**: Custom CNN and Transfer Learning (ResNet, VGG)
- **Dataset**: 50 landmark categories from Google Landmarks Dataset v2
- **Accuracy**: >80% on test set with transfer learning
- **Classes**: Famous landmarks including Eiffel Tower, Great Wall, Machu Picchu

## Example Output

```bash
Top 3 Predictions for uploaded image:
1. Eiffel Tower (94.2%)
2. Arc de Triomphe (3.1%)
3. Notre Dame (1.8%)
```

## Technical Details

- **Framework**: PyTorch
- **Data Processing**: Custom data loaders with augmentation
- **Optimization**: SGD and Adam optimizers with learning rate scheduling
- **Regularization**: Dropout and batch normalization
- **Evaluation**: Comprehensive metrics and confusion matrices

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset from Google Landmarks Dataset v2
- Part of Udacity's Computer Vision Nanodegree
