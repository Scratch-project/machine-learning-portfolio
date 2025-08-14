import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from functions import load_data, build_model, train_model, save_checkpoint, load_checkpoint



def parse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
    parser.add_argument("data_directory", help="Path to the data directory")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--arch", default="vgg19", help="Architecture (vgg16 or densenet121)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=2048, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    args = parser.parse_args()
    return args

def main():
    """
        Usage:
            python train.py data_directory [options]

        Arguments:
            data_directory (str): The path to the data directory containing training, validation, and testing data.

        Options:
            --save_dir (str): Directory to save model checkpoints. Default is 'checkpoints'.
            --arch (str): Choose the architecture for the base model. Options: 'vgg16' or 'densenet121'. Default is 'vgg19'.
            --learning_rate (float): Learning rate for model training. Default is 0.001.
            --hidden_units (int): Number of hidden units in the classifier. Default is 2048.
            --epochs (int): Number of training epochs. Default is 2.
            --gpu: Use GPU for training (if available). If not specified, training will be done on CPU.

        Examples:
            1. Train a model using default settings:
               python train.py data_directory

            2. Train a model and save checkpoints to a custom directory:
               python train.py data_directory --save_dir my_checkpoints

            3. Train a model with specific hyperparameters:
               python train.py data_directory --arch densenet121 --learning_rate 0.01 --hidden_units 512 --epochs 10

            4. Train a model using GPU for training:
           python train.py data_directory --gpu 
    """
    args = parse()
    # Load and preprocess data
    train_loader, valid_loader, test_loader, class_to_idx = load_data(args.data_directory)
    
    # Build the model
    model, classifier = build_model(architecture=args.arch, hidden_units=args.hidden_units, lr=args.learning_rate, use_gpu=args.gpu)
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train the model
    trained_model, epochs, trained_optimizer = train_model(model, train_loader, valid_loader, criterion, optimizer, use_gpu=args.gpu)
    
    # Save the model checkpoint
    path = save_checkpoint(trained_model, args.save_dir, args.arch, class_to_idx, epochs, args.learning_rate, args.hidden_units, trained_optimizer, classifier)
    
    
    continue_training = input("Do you want to continue training the model ? (Enter 'yes' or 'no'): ")
    if continue_training.strip().lower() == "yes":
        iterations = input("How many epochs do you want to train ? (Enter a number): ")
        improve(train_loader, valid_loader, criterion, args.gpu, args.save_dir, int(iterations.strip()))
        print("Training ended!!")
    else:
        print("Training ended!!")
        
def improve(train_loader, valid_loader, criterion, use_gpu, path, itr):
    
    # load the last version of the model
    model_loaded, class_to_idx_loaded, optimizer_loaded, epochs_loaded, classifier_loaded, arch_loaded, lr_loaded, hidden_arg_loaded = load_checkpoint(path, use_gpu)
    
    # continue training it
    trained_model_loaded, new_epochs, trained_optimizer = train_model(model_loaded, train_loader, valid_loader, criterion, optimizer_loaded, use_gpu=use_gpu, iterations=itr, epochs_loaded=epochs_loaded)
    
    save = input("Do you want to save the result by overwriting the last checkpoint ? ('yes' or 'no'): ").strip().lower()
    if save == "yes":
        # Overwrite the last saved checkpoint to the new model checkpoint
        save_checkpoint(trained_model_loaded, path, arch_loaded, class_to_idx_loaded, new_epochs, lr_loaded, hidden_arg_loaded, optimizer_loaded, classifier_loaded)
    else:
        print(f"Finished training without saving the last {itr} training epochs!!")

if __name__ == "__main__":
    main()