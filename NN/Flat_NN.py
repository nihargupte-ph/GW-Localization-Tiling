import torch
from torch import nn
from discretizing import *

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim([box["bottom left"][0], box["bottom right"][0]])
ax.set_ylim([box["bottom left"][1], box["top left"][1]])

# for pixel in pixeled_list:
#     x,y = pixel.exterior.xy
#     ax.plot(x,y,c='k', alpha=.8)

# plt.show()

#NN part
class Network(nn.Module):
    def __init__(self, num_pixels, num_points):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        num_hidden_neurons = (int((2/3) * num_pixels)) + num_points
        self.hidden = nn.Linear(num_pixels, num_hidden_neurons)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(num_hidden_neurons, num_points)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    @staticmethod
    def vectorize(output):
        """ Vectorize our ouput into a single node """


    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

model = Network(num_pixels, num_points)