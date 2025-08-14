# Flower Image Classifier

A deep learning application that identifies flower species from images using PyTorch and transfer learning. This project includes both an interactive Jupyter notebook for experimentation and command-line tools for production use.

## Overview

This image classifier can recognize 102 different flower species with high accuracy using convolutional neural networks. The project demonstrates modern deep learning techniques including transfer learning, data augmentation, and model checkpointing.

## Features

- **Interactive Development**: Jupyter notebook for model experimentation and visualization
- **Command-Line Interface**: Production-ready scripts for training and inference
- **Transfer Learning**: Uses pre-trained models (VGG, ResNet, DenseNet) for improved accuracy
- **GPU Support**: Optimized for both CPU and GPU training
- **Model Persistence**: Save and load trained models for reuse

## Project Structure

```text
├── notebook/
│   └── Image_Classifier_Project.ipynb    # Interactive development notebook
├── cli/
│   ├── train.py                          # Command-line training script
│   ├── predict.py                        # Command-line prediction script
│   └── functions.py                      # Utility functions
├── data/
│   └── cat_to_name.json                 # Category name mappings
├── assets/                              # Sample images and visualizations
└── requirements.txt                     # Python dependencies
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/flower-image-classifier.git
cd flower-image-classifier
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download the flower dataset

```bash
# The dataset will be automatically downloaded when running the notebook
# Or download manually from: https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
```

## Usage

### Interactive Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook notebook/Image_Classifier_Project.ipynb
```

The notebook provides:

- Data exploration and visualization
- Model architecture experimentation
- Training process with real-time metrics
- Prediction examples with probability visualization

### Command-Line Interface

#### Training a Model

```bash
python cli/train.py flowers/ --arch vgg19 --learning_rate 0.001 --epochs 10 --gpu
```

**Options:**

- `--save_dir`: Directory to save checkpoints (default: checkpoints/)
- `--arch`: Model architecture (vgg16, vgg19, densenet121)
- `--learning_rate`: Learning rate (default: 0.001)
- `--hidden_units`: Hidden layer size (default: 2048)
- `--epochs`: Training epochs (default: 2)
- `--gpu`: Use GPU for training

#### Making Predictions

```bash
python cli/predict.py input_image.jpg checkpoint.pth --top_k 5 --gpu
```

**Options:**

- `--top_k`: Number of top predictions to return (default: 5)
- `--category_names`: JSON file with category names (default: data/cat_to_name.json)
- `--gpu`: Use GPU for inference

## Model Performance

- **Architecture**: Transfer learning with pre-trained CNN backbone
- **Dataset**: 102 flower categories with ~8,000 images
- **Accuracy**: >85% on test set
- **Training Time**: ~10-15 minutes on GPU for 10 epochs

## Example Output

```bash
$ python cli/predict.py test_image.jpg checkpoint.pth --top_k 3

Predictions for test_image.jpg:
1. Pink Primrose (92.3%)
2. Cyclamen (4.1%)
3. Wild Pansy (2.8%)
```

## Technical Details

- **Framework**: PyTorch
- **Data Augmentation**: Random rotation, flipping, and cropping
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers to prevent overfitting
- **Image Processing**: PIL and torchvision transforms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset from the Visual Geometry Group at Oxford
- Part of Udacity's AI Programming with Python Nanodegree
