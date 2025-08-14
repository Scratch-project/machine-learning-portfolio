# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import numpy as np
import json

# check if gpu is avaulabe, otherwise use cpu
def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def load_data(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(35),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    testVal_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=testVal_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=testVal_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    
    return train_loader, valid_loader, test_loader, train_data.class_to_idx
    
    
    
    
    
def build_model(architecture, hidden_units, lr, use_gpu):
    
    device = check_device()
    # use cpu if gpu isn't provided in the command line argumnets
    if not use_gpu:
        device = "cpu"
        
    model_classes = {
        "vgg16": models.vgg16,
        "resnet18": models.resnet18,
        "densenet121": models.densenet121,
        "vgg19": models.vgg19
    }

    # Check if the specified architecture is valid
    if architecture in model_classes:
        model = model_classes[architecture](pretrained=True)
    else:
        raise ValueError("Invalid architecture.")
    
    
    # specifiy the input to the model based on the architecture used
    if architecture.strip().lower() == "densenet121":   
        input_units = 1024
    elif architecture.strip().lower() == "resnet18": 
        input_units = 512
    else:
        input_units = 25088

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # define the architecture 
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units, lr),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    
    # define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    
    # transfer the model to the machine working on it
    model.to(device)
    
    return model, classifier
def train_model(model, train_loader, valid_loader, criterion, optimizer, use_gpu, iterations=1, epochs_loaded=0):
    
    device = check_device()
    # use cpu if gpu isn't provided in the command line argumnets
    if not use_gpu:
        device = "cpu"
        
    print(f"Training on {device}")
    # Set the end number of epochs
    epochs = epochs_loaded + iterations

    train_losses, val_losses = [], []
    for e in range(epochs_loaded, epochs):
        running_loss = 0
        start = time.time()
        print(len(train_loader))
        print(device)
        for images, labels in train_loader:
            # Move the images and labels to the specified device (e.g., GPU)
            images, labels = images.to(device), labels.to(device)
            
            # Feed forward and loss calculation
            logp = model.forward(images)
            loss = criterion(logp, labels)

            # Resetting the grad and backpropagating
            optimizer.zero_grad()
            loss.backward()

            # Optimizing the weights
            optimizer.step()

            # Keeping track of the loss per epoch
            running_loss += loss.item()

        else:
            # Define variables for loss and accuracy
            val_loss, accuracy = 0, 0

            with torch.no_grad():
                # Enter the evaluation mode
                model.eval()

                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)

                    # Feed forward and loss calculating
                    val_logp = model.forward(images)
                    val_loss += criterion(val_logp, labels)

                    # Getting the probabilities
                    ps = torch.exp(val_logp)

                    # Getting the top class out of the probabilities
                    top_p, top_class = ps.topk(1, dim=1)

                    # Compare the top classes for each probability of each image in the batch and store it as true or false
                    equality = top_class == labels.view(*top_class.shape)

                    # Take the mean of all predictions and add them to get the total accuracy of all the batches
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

            # Enter the train mode
            model.train()

            # Append the losses
            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss/len(valid_loader))

            # Print information about each epoch
            print(f"Epoch: {e+1}/{epochs}.."
                  f"Training Loss: {train_losses[-1]:.3f}..."
                  f"Validation Loss: {val_losses[-1]:.3f}..."
                  f"Validation Accuracy: {(accuracy/len(valid_loader))*100:.2f}%")

        print(f"Elapsed time: {(int(time.time()-start)/60):.2f} minutes")
        
    return model, epochs, optimizer

    
def save_checkpoint(model, save_dir, architecture, class_to_idx, epochs, lr, hidden_units, optimizer, classifier):
    checkpoint_path = save_dir + '/' + architecture + '_checkpoint.pth'

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'learning_rate': lr,
        'hidden_units': hidden_units,
        'arch': architecture,
        'classifier': classifier
    }

    # Save the checkpoint to the specified file
    torch.save(checkpoint, checkpoint_path)
    print("saved!")
    return checkpoint_path
    
    
    
def load_checkpoint(filepath, use_gpu):
    device = check_device()
    
    # use cpu if gpu isn't provided in the command line argumnets
    if not use_gpu:
        device = "cpu"
        
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    architecture = checkpoint['arch']
    
    # Create a new model with the same architecture as before
    model_classes = {
        "vgg16": models.vgg16,
        "resnet18": models.resnet18,
        "densenet121": models.densenet121,
        "vgg19": models.vgg19
    }

    # Check if the specified architecture is valid
    if architecture in model_classes:
        model = model_classes[architecture](pretrained=True)
    else:
        raise ValueError("Invalid architecture.")
    # define the classifier
    classifier = checkpoint['classifier']
    
    model.classifier = classifier
    
    # Load the model's state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the class-to-index mapping
    class_to_idx = checkpoint['class_to_idx']
    
    # ONLY for training  (Don't know if in part 2 the application should enable the user to continue training or not)
    # Note that I'm just loading the needed things to continue training but the application still doesn't have that feature
    lr = checkpoint['learning_rate']
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = nn.NLLLoss()
    epochs = checkpoint['epochs']
    
    # Move to the device
    model.to(device)
    
    return model, class_to_idx, optimizer, epochs, classifier, architecture, lr, checkpoint['hidden_units']
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # open the image
    image = Image.open(image)
    
    # resize the image
    width, height = image.size
    if width < height:
        ratio = height/width
        width = 256
        height = int(width*ratio)
    else:
        ratio = width/height
        height = 256
        width = int(height*ratio)
    resized_img = image.resize((width, height))
    
    # crop the image
    left = (width-224)/2
    right = left+224
    upper = (height-224)/2
    buttom = upper+224 
    
    final_img = resized_img.crop((left, upper, right, buttom))
    
    # adjusting color channels
    np_image = np.array(final_img) /255.0
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    # normalize using mean and standard deviation
    np_image = (np_image - means) / stds
    
    # rearranging the dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(np_image)
def predict(image_path, model, class_to_idx_loaded, use_gpu, topk=5):
    device = check_device()
    # use cpu if gpu isn't provided in the command line argumnets
    if not use_gpu:
        device = "cpu"
        
    # Load and process the image
    image = process_image(image_path)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move the model to the appropriate device (GPU)
    model.to(device)
    
    # Convert image to float (don't know why, but got an error that it must be float)
    image = image.float()
    
    # Move the input image to the same device (GPU)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    # Calculate class probabilities
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert indices to class labels (just a compacter way using list comprehension)
    idx_to_class = {v: k for k, v in class_to_idx_loaded.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    return top_probabilities[0].tolist(), top_classes
    
    
    
    
    
    
    
# not required for the project part 2 (BUT I would like to have it with all other functions, just a good summary of what I did)
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax   
def plot_prediction(image_path, model, class_to_idx, cat_to_name):
    # Make predictions
    probs, classes = predict(image_path, model)
    
    # Get class names from class indices using cat_to_name
    class_names = [cat_to_name[cls] for cls in classes]
    
    # Get the top predicted class
    top_class = class_names[0]
    
    # Load and display the input image
    image = Image.open(image_path)
    
    # Create a grid layout with two rows and one column
    fig, (ax1, ax2) = plt.subplots(figsize=(3, 6), nrows=2, ncols=1)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(top_class)
    
    # Plot the bar graph
    ax2.barh(class_names, probs, color='blue')
    ax2.set_aspect(0.2)
    ax2.set_yticks(class_names)
    ax2.set_yticklabels(class_names, size='small')
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()  # Invert the labels to have the highest probability at the top

    plt.show()