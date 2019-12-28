import torch
from torch import nn
import matplotlib.pyplot as plt

# Custom imports
from Color_Mapping import *


def dicretize_plane(box, num_points_sqrt):
    """ returns tuple of indexable points given the box (dictionary) and the square_root of the number of points"""

    X = np.linspace(box["bottom left"][0],
                    box["bottom right"][0], num_points_sqrt)
    Y = np.linspace(box["bottom left"][1], box["top left"][1], num_points_sqrt)

    indexable_points = tuple([(x, y) for x in X for y in Y])

    return indexable_points

#Setting up discritization
box = {"bottom left": (-2, -2),
       "bottom right": (2, -2),
       "top right": (2, 2),
       "top left": (-2, 2)}

num_points_sqrt = 200
num_points = num_points_sqrt ** 2


#Input could be where the points intersect. So total inputs would be discretized 
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x