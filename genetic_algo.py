import numpy as np
import os
import shutil
from shapely import geometry
from shapely import ops
from shapely.ops import cascaded_union
from rtree import index
import random
import matplotlib.pyplot as plt
from scipy.stats import expon


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

    def update_agent(self):
        self.length = len(self.circle_list)

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

        for p1 in self_intersection:
            x1, y1 = p1.exterior.xy

            if ax == None:
                plt.fill(x1, y1, c=color1, zorder=zorder+.1)
            else:
                ax.fill(x1, y1, c=color1, zorder=zorder+.1)

        for p2 in region_intersection:
            x2, y2 = p2.exterior.xy

            if ax == None:
                plt.fill(x2, y2, c=color2, zorder=zorder)
            else:
                ax.fill(x2, y2, c=color2, zorder=zorder)

        for p3 in region_nonintersection:
            x3, y3 = p3.exterior.xy

            if ax == None:
                plt.fill(x3, y3, c=color3, zorder=zorder)
            else:
                ax.fill(x3, y3, c=color3, zorder=zorder)

        plt.legend(["Fitness: {}".format(self.fitness), "Number circles: {}".format(self.length)], loc='upper left')


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

def intersection_region(region, polygon_list, bounding_box):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    bounding_box = geometry.Polygon([bounding_box["bottom left"], bounding_box["bottom right"], bounding_box["top right"], bounding_box["top right"]])

    polygon_union = cascaded_union(polygon_list)
    intersection = region.intersection(polygon_union)
    fraction_overlap = intersection.area / region.area
    nonoverlap = polygon_union.difference(region)
    fraction_nonoverlap = nonoverlap.area / bounding_box.area

    return intersection, nonoverlap, fraction_overlap, fraction_nonoverlap



def get_highest_area(region, circle_list):
    """ From a circle collection returns an ordered list of circles ordered by how much they intersect with the region """

    temp_list = [(i, circle) for i, circle in enumerate(circle_list)]
    _, ret_list = zip(*sorted(temp_list,key=lambda x: region.intersection(x[1]).area))

    return ret_list
# GA part


def init_agents(radius, bounding_box, length=50):

    return [Agent(radius=radius, bounding_box=bounding_box, length=length) for _ in range(population)]


def fitness(agent_list, region, bounding_box):

    alpha = 100
    beta = 20
    chi = 5

    for agent in agent_list:

        _, _, frac_overlap, frac_nonoverlap = intersection_region(region, agent.circle_list, bounding_box)
        _, frac_self_intersection = double_intersection(agent.circle_list)

        agent.fitness = (alpha * frac_overlap) - (beta * frac_nonoverlap) - (chi * frac_self_intersection)

    return agent_list


def selection(agent_list):

    agent_list = sorted(
        agent_list, key=lambda agent: agent.fitness, reverse=True)
    # DARWINISM HAHHAA
    agent_list = agent_list[:int(.5 * len(agent_list))]

    return agent_list



def crossover(agent_list, region):
    """ Crossover is determined by randomly splitting polygon in half and then taking circles from each side. It'll probably mess up on the
    boundary tbh but I couldn't think of another crossover function """

    # def getExtrapoledLine(p1,p2):
    #     'Creates a line extrapoled in p1->p2 direction'
    #     EXTRAPOL_RATIO = 10
    #     a = (p2[0]+EXTRAPOL_RATIO*(p1[0]-p2[0]), p2[1]+EXTRAPOL_RATIO*(p1[1]-p2[1]))
    #     b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]))
    #     return geometry.LineString([a,b])

    # line = getExtrapoledLine((region.centroid.x, region.centroid.y), (region.centroid.x + random.uniform(-1, 1), region.centroid.y\
    #     + random.uniform(-1, 1)))

    # merged = ops.linemerge([region.boundary, line])
    # borders = ops.unary_union(merged)
    # polygons = list(ops.polygonize(borders))

    offspring = []
    for _ in range(round(len(agent_list) / 2)):

        parent1 = random.choice(agent_list)
        parent2 = random.choice(agent_list)

        #Generates an index distirubtion, we are getting a sorted list later but we want more elements from the beginning of the sorted list rather than the end
        # total_circles = parent1.circle_list + parent2.circle_list
        # sorted_circles = get_highest_area(region, total_circles)

        # data_expon = expon.rvs(scale=1,loc=0,size=len(sorted_circles))
        # prob_dist = sorted(data_expon, reverse=True)
        # circle_pool = np.random.choice(sorted_circles, p=prob_dist)

        # print(circle_pool)

        len_children = (len(parent1.circle_list) + len(parent2.circle_list)) // 2

        parent1_sorted_circles = get_highest_area(region, parent1.circle_list)
        parent2_sorted_circles = get_highest_area(region, parent2.circle_list)

        parent1_sorted_circles[:len_children]
        parent2_sorted_circles[:len_children]

        parent1_circle_half_1 = list(parent1_sorted_circles[0::2])
        parent1_circle_half_2 = list(parent1_sorted_circles[1::2])
        parent2_circle_half_1 = list(parent2_sorted_circles[0::2])
        parent2_circle_half_2 = list(parent2_sorted_circles[1::2])


        # parent1_circle_half_1 = [
        #     circle for circle in parent1.circle_list[:len(parent1.circle_list)//2]]
        # parent1_circle_half_2 = [
        #     circle for circle in parent1.circle_list[len(parent1.circle_list)//2:]]
        # parent2_circle_half_1 = [
        #     circle for circle in parent2.circle_list[:len(parent2.circle_list)//2]]
        # parent2_circle_half_2 = [
        #     circle for circle in parent2.circle_list[len(parent2.circle_list)//2:]]

        child1 = Agent(radius=parent1.radius, bounding_box=parent1.bounding_box, adam=False)
        child2 = Agent(radius=parent1.radius, bounding_box=parent1.bounding_box, adam=False)
        child1.circle_list = parent1_circle_half_1 + parent2_circle_half_2
        child2.circle_list = parent2_circle_half_1 + parent1_circle_half_2
        child1.update_agent()
        child2.update_agent()

        offspring.append(child1)
        offspring.append(child2)

    agent_list.extend(offspring)

    return agent_list


def mutation(agent_list):

    for agent in agent_list:

        for i, param in enumerate(agent.circle_list):

            if random.uniform(0, 1) <= .3:

                add_or_sub = random.choice((0, 1))
                if add_or_sub == 0:
                    agent.circle_list.pop(random.randint(0, agent.length-1))
                    agent.update_agent()
                elif add_or_sub == 1:
                    agent.circle_list.append(get_circle(agent.radius, (random.uniform(bounding_box["bottom left"][0], bounding_box["bottom right"][0]), random.uniform(bounding_box["bottom left"][1], bounding_box["top left"][1]))))
                    agent.update_agent()

            #Maybe add a move circle feature here

    return agent_list


def ga(region, bounding_box):

    print("Initializing Agents...")
    agent_list = init_agents(.1, bounding_box)
    print("Agents initialized.")

    for generation in range(generations):

        print("\ngeneration number: {}".format(generation))

        print("Determining how much they lift..")
        agent_list = fitness(agent_list, region, bounding_box)
        print("Sucessful")
        print()
        print("Executing stragllers")
        agent_list = selection(agent_list)
        print("Darwin has spoken, {} candidates remain".format(len(agent_list)))
        agent_list.sort(key=lambda x: x.fitness, reverse=True)
        plt.figure(figsize=(6,6))
        agent_list[0].plot_agent(
            region, colors[1], colors[2], colors[3], bounding_box, zorder=2)
        plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
        plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
        plt.plot(*region.exterior.xy)
        plt.savefig("frames/generation_{0:03d}".format(generation))
        print("frame saved in frames folder")
        plt.close()
        print()
        print("Beginning candlelight dinner")
        agent_list = crossover(agent_list, region)
        print("Young love phase is over")
        print()
        print("Blasting my guys with Radiation")
        agent_list = mutation(agent_list)
        print("Completed.")


global population
population = 200

global generations
generations = 100

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

bounding_box = {"bottom left": (-2, -2),
                "bottom right": (2, -2),
                "top right": (2, 2),
                "top left": (-2, 2)}


test_polygon = geometry.Polygon([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)])

# Clearing folder before we add new frames
folder = '/home/n/Documents/Research/GW-Localization-Tiling/frames'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

ga(test_polygon, bounding_box)
