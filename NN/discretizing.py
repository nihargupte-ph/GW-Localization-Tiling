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

def pixelize_plane(box, num_points_sqrt):
    """ returns list of shapely boxes which tile the box """

    X = np.linspace(box["bottom left"][0],
                    box["bottom right"][0], num_points_sqrt)
    Y = np.linspace(box["bottom left"][1], box["top left"][1], num_points_sqrt)

    pixel_list = []
    for i in range(0, num_pixels_sqrt):
        for j in range(0, num_pixels_sqrt):
            try:
                poly = geometry.Polygon([(X[i], Y[j]), (X[i+1], Y[j]), (X[i+1], Y[j+1]), (X[i], Y[j+1])])
                pixel_list.append(poly)
            except IndexError:
                break

    return pixel_list

#Setting up discritization
box = {"bottom left": (-2, -2),
       "bottom right": (2, -2),
       "top right": (2, 2),
       "top left": (-2, 2)}

num_points_sqrt = 200
num_points = num_points_sqrt ** 2
num_pixels_sqrt = 50
num_pixels = num_pixels_sqrt ** 2

pixeled_list = pixelize_plane(box, num_pixels_sqrt)