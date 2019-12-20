import torch
import matplotlib.pyplot as plt

# Custom imports
from Color_Mapping import *


def dicretize_plane(box, num_points_sqrt):
    """ returns tuple of indexable points given the box (dictionary) and the square_root of the number of points"""

    X = np.linspace(box["bottom left"][0],
                    box["bottom right"][0], num_points_sqrt)
    Y = np.linspace(box["bottom left"][1], box["top left"][1], num_points_sqrt)

    indexable_points = ((x, y) for x in X for y in Y)

    return indexable_points


box = {"bottom left": (-2, -2),
       "bottom right": (2, -2),
       "top right": (2, 2),
       "top left": (-2, 2)}

num_points_sqrt = 20
num_points = num_points_sqrt ** 2


plt.plot(*dicretize_plane(box, num_points_sqrt))