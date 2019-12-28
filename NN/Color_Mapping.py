import numpy as np
from shapely import geometry
from shapely import ops
from shapely.ops import cascaded_union
from rtree import index

def double_intersection(polygon_list):

    """ Returns intersection between polygons in polygon_list and the area of their intersection """
   
    intersections = []
    idx = index.Index()       
    # Populate R-tree index with bounds of grid cells
    for pos, cell in enumerate(polygon_list):
        # assuming cell is a shapely object
        idx.insert(pos, cell.bounds)

    for poly in polygon_list:
        merged_circles = cascaded_union([polygon_list[pos] for pos in idx.intersection(poly.bounds) if polygon_list[pos] != poly])
        intersections.append(poly.intersection(merged_circles))

    intersection = cascaded_union(intersections)
    intersection_area = intersection.area
    return intersection, intersection_area

def intersection_region(region, polygon_list):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    polygon_union = cascaded_union(polygon_list)
    intersection = region.intersection(polygon_union)
    fraction = intersection.area / region.area
    nonoverlap = polygon_union.difference(region)

    return intersection, nonoverlap, fraction

class Circle:

    """ Circle class, each circle in the agent is represented by one of these circles """

    def __init__(self, _radius, _center, step = 100):

        """ Creates cirlce given radius and center.  """
        
        #Var inits
        self.center = geometry.Point(_center)
        self.radius = _radius
        self.point_list = [geometry.Point(self.radius * np.cos(theta) + self.center.x, self.radius * np.sin(theta) + self.center.y) for theta in np.linspace(0, 2 * np.pi, step)]
        self.polygon = geometry.Polygon([[p.x, p.y] for p in self.point_list])

    def print_properties(self):
        print("center: {} \n \n radius: {} \n \n points: {} \n \n polygon: {}".format(self.center, self.radius, self.point_list, self.polygon))

    def plot_circle(self, color=None, ax=None, fill=False, zorder=1):

        """ Plot circles """
        
        x,y = self.polygon.exterior.xy
        if fill==False:
            if ax==None:
                plt.plot(x,y, c=color, zorder=zorder)
            else:
                ax.plot(x,y, c=color, zorder=zorder)
        if fill==True:
            if ax==None:
                plt.fill(x,y, c=color, zorder=zorder)
            else:
                ax.fill(x,y, c=color, zorder=zorder)

class CircleCollection:

    def __init__(self, _radius, _centers):

        """ This is what the Neural Net will return at the end """

        self.radius = _radius
        self.center_list = _centers
        self.circle_list = [Circle(_radius, i) for i in _centers]
        self.polygon_list = [circle.polygon for circle in self.circle_list]

    def get_intersections(self, region):

        """ Returns all types of intersections. self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction """

        self_intersection, self_intersection_fraction =  double_intersection(self.polygon_list)
        region_intersection, region_nonintersection, region_intersection_fraction = intersection_region(region, self.polygon_list)

        return self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction


    def plot_intersections(self, region, color1, color2, color3, ax=None, zorder=1):   

        """ Plots circle intersection and non interesection with region as well as self intersection"""

        self_intersection, _, region_intersection, region_nonintersection, _ = self.get_intersections(region)

        for p1 in self_intersection:
            x1,y1 = p1.exterior.xy

            if ax==None:
                plt.fill(x1,y1, c=color1, zorder=zorder+.1)
            else:
                ax.fill(x1,y1, c=color1, zorder=zorder+.1)

        for p2 in region_intersection:
            x2,y2 = p2.exterior.xy

            if ax==None:
                plt.fill(x2,y2, c=color2, zorder=zorder)
            else:
                ax.fill(x2,y2, c=color2, zorder=zorder)

        for p3 in region_nonintersection:
            x3,y3 = p3.exterior.xy

            if ax==None:
                plt.fill(x3,y3, c=color3, zorder=zorder)
            else:
                ax.fill(x3,y3, c=color3, zorder=zorder)

# #Testing and plotting functions section
# import matplotlib.pyplot as plt
# from matplotlib.path import Path
# import matplotlib.patches as patches
# import random

# colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

# n = 8 # Number of possibly sharp edges
# r = .7 # magnitude of the perturbation from the unit circle, 
# # should be between 0 and 1
# N = n*3+1 # number of points in the Path
# # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

# angles = np.linspace(0,2*np.pi,N)

# verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]

# fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
# ax = fig.add_subplot(111)

# #Generating localization region, called region here
# region = geometry.Polygon(verts)

# ax.fill(*region.exterior.xy, color=colors[0], zorder=1)


# #Adding in a few random circles
# random_centers = [(random.uniform(-1.8, 1.8), random.uniform(-1.8, 1.8)) for i in range(0, 100)]
# circ_collection = CircleCollection(.2, random_centers)
# circ_collection.plot_intersections(region, colors[1], colors[2], colors[3], ax=ax, zorder=2)

# temp_circle_list = [Circle(.2,i) for i in random_centers]
# for circle in temp_circle_list:
#     circle.plot_circle(color='k', ax=ax)

# ax.set_xlim([-2, 2])
# ax.set_ylim([-2, 2])
# plt.show()