import numpy as np
import os
import shutil
from shapely import geometry
from shapely import affinity
from shapely import ops
from shapely.ops import cascaded_union
from rtree import index
import random
import matplotlib.pyplot as plt
from scipy.stats import expon
from misc_functions import *
import fiona
import time
import scipy.optimize as optimize
from itertools import zip_longest


def get_circle(radius, center, step=100):
    """ Returns shapely polygon given radius and center """

    point_list = [geometry.Point(radius * np.cos(theta) + center[0], radius * np.sin(
        theta) + center[1]) for theta in np.linspace(0, 2 * np.pi, step)]
    polygon = geometry.Polygon([[p.x, p.y] for p in point_list])

    return polygon


class Agent:

    def __init__(self, radius=None, bounding_box=None, length=None, adam=True):
        """ Agent object"""

        self.fitness = -1000  # Dummy value
        self.radius = radius
        self.bounding_box = bounding_box
        self.length = length

        if adam == True:
            self.circle_list = [get_circle(radius, (random.uniform(bounding_box["bottom left"][0], bounding_box["bottom right"][0]), random.uniform(
                bounding_box["bottom left"][1], bounding_box["top left"][1]))) for _ in range(length)]

        self.center_list = [(circle.centroid.x, circle.centroid.y) for circle in self.circle_list]

    def update_agent(self):
        self.length = len(self.circle_list)
        self.center_list = [(circle.centroid.x, circle.centroid.y) for circle in self.circle_list]

    def update_agent_new_centers(self):
        """ Updates the agent if we changed the centers instead of the circles like we usually do """

        self.center_list = [get_circle(self.radius, center) for center in self.center_list]
        self.length = len(self.circle_list)

    def remove_irrelavent_circles(self, region, threshold):
        """ Removes all circles in circle_list that intrsect the region less than threshold """

        tmp_lst = []
        for circle in self.circle_list:
            if circle.intersection(region).area > threshold:
                tmp_lst.append(circle)
        self.circle_list = tmp_lst

    def get_intersections(self, region, bounding_box):
        """ Returns all types of intersections. self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction """

        self_intersection, self_intersection_fraction = double_intersection(
            self.circle_list)

        region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction = intersection_region(
            region, self.circle_list, bounding_box)

        return self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction

    def plot_agent(self, region, color1, color2, color3, bounding_box, ax=None, zorder=1):

        """ Plots circle intersection and non interesection with region as well as self intersection"""

        #makes sure everything is nice and updated
        self.update_agent()

        self_intersection, _, region_intersection, region_nonintersection, _, _ = self.get_intersections(
            region, bounding_box)

        try: 
            for p1 in self_intersection:
                x1, y1 = p1.exterior.xy

                if ax == None:
                    plt.fill(x1, y1, c=color1, zorder=zorder+.1)
                else:
                    ax.fill(x1, y1, c=color1, zorder=zorder+.1)

        except TypeError:
            p1 = self_intersection
            x1, y1 = p1.exterior.xy

            if ax == None:
                plt.fill(x1, y1, c=color1, zorder=zorder+.1)
            else:
                ax.fill(x1, y1, c=color1, zorder=zorder+.1)

        try: 
            for p3 in region_nonintersection:
                x3, y3 = p3.exterior.xy

                if ax == None:
                    plt.fill(x3, y3, c=color3, zorder=zorder)
                else:
                    ax.fill(x3, y3, c=color3, zorder=zorder)
        except TypeError:
            p3 = region_nonintersection
            x3, y3 = p3.exterior.xy

            if ax == None:
                plt.fill(x3, y3, c=color3, zorder=zorder)
            else:
                ax.fill(x3, y3, c=color3, zorder=zorder)

        try: 
            for p2 in region_intersection:
                x2, y2 = p2.exterior.xy

                if ax == None:
                    plt.fill(x2, y2, c=color2, zorder=zorder)
                else:
                    ax.fill(x2, y2, c=color2, zorder=zorder)
        except TypeError:
            p2 = region_intersection
            x2, y2 = p2.exterior.xy

            if ax == None:
                plt.fill(x2, y2, c=color2, zorder=zorder)
            else:
                ax.fill(x2, y2, c=color2, zorder=zorder)

        plt.legend(["Fitness: {}".format(self.fitness), "Number circles: {}".format(self.length)], loc='upper left')

    def move_circle(self, old_circle, center): 
        """ Moves circle from circle_list to new center """
 
        try:
            self.circle_list.remove(old_circle)
        except:
            raise IndexError("The circle entered was not found in this agent")

        new_circle = get_circle(self.radius, center)

        self.circle_list.append(new_circle)


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
        intersec = poly.intersection(merged_circles)

        if isinstance(intersec, geometry.GeometryCollection): #For some reason linestrings are getting appended so i'm removing them
            new_intersec = geometry.GeometryCollection([shape for shape in intersec if not isinstance(shape, geometry.LineString)])
            intersections.append(new_intersec)
        else:
            intersections.append(intersec)

    intersection = cascaded_union(intersections)
    intersection_area = intersection.area
    return intersection, intersection_area


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def intersection_region(region, polygon_list, bounding_box):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    bounding_box = geometry.Polygon([bounding_box["bottom left"], bounding_box["bottom right"], bounding_box["top right"], bounding_box["top right"]])

    polygon_union = cascaded_union(polygon_list)
    intersection = region.intersection(polygon_union)
    fraction_overlap = intersection.area / region.area
    nonoverlap = polygon_union.difference(region)
    fraction_nonoverlap = nonoverlap.area / bounding_box.area

    return intersection, nonoverlap, fraction_overlap, fraction_nonoverlap


def intersection_region_objective(flat_center_list, radius, region):

    """ Returns amount of area not covered by the polygons. This is especially useful for the BFGS repair function."""

    center_list = grouper(2, flat_center_list)

    polygon_list = [get_circle(radius, center) for center in center_list]
    polygon_union = cascaded_union(polygon_list)
    intersection = region.intersection(polygon_union)
    not_covered_area = region.area - intersection.area 

    return not_covered_area



global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

bounding_box = {"bottom left": (-2, -2),
                "bottom right": (2, -2),
                "top right": (2, 2),
                "top left": (-2, 2)}


test_polygon = geometry.Polygon([(-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2)])


#Testing code region
def repair_agent_BFGS(agent, region, plot=False):
    """ repairs agent to possibly cover the region, if the region is not covered it will return false if it is it will return true """

    #Plotting
    if plot:
        plt.figure(figsize=(6,6))
        plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
        plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
        agent.plot_agent(test_polygon, colors[1], colors[2], colors[3], bounding_box)
        plt.plot(*region.exterior.xy)
        plt.savefig("repair_frames/before")
        plt.close()

    flat_center_list = [item for sublist in agent.center_list for item in sublist]
    
    print(flat_center_list)
    result = optimize.minimize(intersection_region_objective, flat_center_list, args=(agent.radius, region), method='BFGS')
    print(result.x)
    new_centers = list(grouper(2, result.x))
    agent.center_list = new_centers
    agent.update_agent_new_centers()

    #Plotting
    if plot:
        plt.figure(figsize=(6,6))
        plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
        plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
        agent.plot_agent(test_polygon, colors[1], colors[2], colors[3], bounding_box)
        plt.plot(*region.exterior.xy)
        plt.savefig("repair_frames/after")
        plt.close()

        

agent = Agent(radius=.2, bounding_box=bounding_box, length=10)
tmp = repair_agent_BFGS(agent, test_polygon, plot=True)
agent.plot_agent(test_polygon, colors[1], colors[2], colors[3], bounding_box)