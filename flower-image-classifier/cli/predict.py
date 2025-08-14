import argparse
import torch
import json
from PIL import Image
from functions import load_checkpoint, predict

def main():
    """
Usage:
    python predict.py input checkpoint [options]

Arguments:
    input (str): Path to the input image for which you want to make predictions.
    checkpoint (str): Path to the model checkpoint saved by the 'train.py' script.

Options:
    --top_k (int): Return the top K most likely classes. Default is 5.
    --category_names (str): File containing category names in JSON format. Default is 'cat_to_name.json'.
    --gpu: Use GPU for inference (if available). If not specified, inference will be done on CPU.

Examples:
    1. Make predictions using a pre-trained model with default settings:
       python predict.py input.jpg checkpoint.pth

    2. Make predictions and return the top 3 most likely classes:
       python predict.py input.jpg checkpoint.pth --top_k 3

    3. Make predictions and use a custom JSON file for category names:
       python predict.py input.jpg checkpoint.pth --category_names custom_categories.json

    4. Make predictions using GPU for inference:
       python predict.py input.jpg checkpoint.pth --gpu
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Make predictions using a pre-trained neural network")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", default="cat_to_name.json", help="File with category names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    
    args = parser.parse_args()
    
    # Load the pre-trained model checkpoint
    # TODO: use the rest of the returned variables to train the model again
    model, class_to_idx_loaded, _, _, _, _, _, _ = load_checkpoint(args.checkpoint, args.gpu)
    
    # Make predictions on the input image
    probs, classes = predict(args.input, model, class_to_idx_loaded=class_to_idx_loaded, use_gpu=args.gpu, topk=args.top_k)
    
    # Load category names from a JSON file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Map class indices to category names
    class_names = [cat_to_name[cls] for cls in classes]
    
    # Display the results
    for i in range(args.top_k):
        print(f"Top {i + 1}: {class_names[i]} (Class {classes[i]}), Probability: {probs[i]:.4f}")

if __name__ == "__main__":
    main()