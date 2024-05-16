import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    """
    A Convolutional Neural Network model class that extends the PyTorch nn.Module.
    This model has two convolutional layers, followed by three fully connected layers.
    """

    def __init__(self):
        """
        Initialize the ConvolutionalNetwork with two convolutional layers and three fully connected layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3,
                               1)  # First convolutional layer with 1 input channel, 6 output channels, kernel size of 3 and stride of 1
        self.conv2 = nn.Conv2d(6, 16, 3,
                               1)  # Second convolutional layer with 6 input channels, 16 output channels, kernel size of 3 and stride of 1
        self.fc1 = nn.Linear(5 * 5 * 16,
                             120)  # First fully connected layer with input size of 5*5*16 and output size of 120
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer with input size of 120 and output size of 84
        self.fc3 = nn.Linear(84, 10)  # Third fully connected layer with input size of 84 and output size of 10

    def forward(self, X):
        """
        Defines the forward pass of the ConvolutionalNetwork.
        """
        X = F.relu(self.conv1(X))  # Apply ReLU activation function after first convolutional layer
        X = F.max_pool2d(X, 2, 2)  # Apply max pooling with kernel size of 2 and stride of 2
        X = F.relu(self.conv2(X))  # Apply ReLU activation function after second convolutional layer
        X = F.max_pool2d(X, 2, 2)  # Apply max pooling with kernel size of 2 and stride of 2
        X = X.view(-1, 5 * 5 * 16)  # Reshape the tensor for the fully connected layers
        X = F.relu(self.fc1(X))  # Apply ReLU activation function after first fully connected layer
        X = F.relu(self.fc2(X))  # Apply ReLU activation function after second fully connected layer
        X = self.fc3(X)  # Apply the third fully connected layer
        return F.log_softmax(X, dim=1)  # Apply log softmax function to the output of the network