import math
import shapely.geometry as geometry
import matplotlib.pyplot as plt
import numpy as np

def get_circle(radius, center, step=100):
    """ Returns shapely polygon given radius and center """

    point_list = [geometry.Point(radius * np.cos(theta) + center[0], radius * np.sin(
        theta) + center[1]) for theta in np.linspace(0, 2 * np.pi, step)]
    polygon = geometry.Polygon([[p.x, p.y] for p in point_list])

    return polygon


def calc_polygons_new(startx, starty, endx, endy, radius):
    """ https://stackoverflow.com/questions/26691097/faster-way-to-calculate-hexagon-grid-coordinates """
    sl = (2 * radius) * math.tan(math.pi / 6)

    # calculate coordinates of the hexagon points
    p = sl * 0.5
    b = sl * math.cos(math.radians(30))
    w = b * 2
    h = 2 * sl

    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    row = 1

    shifted_xs = []
    straight_xs = []
    shifted_ys = []
    straight_ys = []

    while startx < endx:
        xs = [startx, startx, startx + b, startx + w, startx + w, startx + b, startx]
        straight_xs.append(xs)
        shifted_xs.append([xoffset + x for x in xs])
        startx += w

    while starty < endy:
        ys = [starty + p, starty + (3 * p), starty + h, starty + (3 * p), starty + p, starty, starty + p]
        (straight_ys if row % 2 else shifted_ys).append(ys)
        starty += yoffset
        row += 1

    polygons = [geometry.Polygon(zip(xs, ys)) for xs in shifted_xs for ys in shifted_ys] + [geometry.Polygon(zip(xs, ys)) for xs in straight_xs for ys in straight_ys]
    return polygons

bounding_box = {"bottom left": (-2, -2),
                "bottom right": (2, -2),
                "top right": (2, 2),
                "top left": (-2, 2)}

test_polygon = geometry.Polygon([(-.6, -.6), (.6, -.6), (.6, .6), (-.6, .6)])
hex_list = calc_polygons_new(-.83, -.67, .5, .5, .18)

plt.figure(figsize=(6,6))
plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
plt.plot(*test_polygon.exterior.xy)
for polygon in hex_list:
    circle = get_circle(.2, (polygon.centroid.x, polygon.centroid.y))
    plt.plot(*circle.exterior.xy, c='b')
plt.legend(["Number of Circles: {}".format(len(hex_list))])
plt.show()