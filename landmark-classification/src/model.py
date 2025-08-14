import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.2) -> None:
        super().__init__()
        out_channels = 8
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1),  # 224x224x3 -> 224x224x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout),

            nn.Conv2d(out_channels, out_channels*2, 3, padding=1),  #112x112x8 -> 112x112x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout),

            nn.Conv2d(out_channels*2, out_channels*4, 3, padding=1),  # 56x56x16 -> 56x56x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Dropout(p=dropout),

            nn.Conv2d(out_channels*4, out_channels*8, 3, padding=1),  # 28x28x32 -> 28x28x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Dropout(p=dropout),
            
            nn.Conv2d(out_channels*8, out_channels*16, 3, padding=1),  # 28x28x64 -> 28x28x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Dropout(p=dropout),
            
            nn.Conv2d(out_channels*16, out_channels*32, 3, padding=1),  # 28x28x128 -> 28x28x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Dropout(p=dropout),


            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1) 
        )

    def forward(self, x):
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
