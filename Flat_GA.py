import numpy as np
import os
import shutil
from shapely import geometry
from shapely import affinity
import scipy.optimize as optimize
from scipy.spatial import Voronoi
from shapely import ops
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from rtree import index
import random
import pickle
import matplotlib.pyplot as plt
from scipy.stats import expon
from misc_functions import *
import fiona
import time
from scipy.spatial import ConvexHull

random.seed(0)

def get_circle(radius, center, step=100):
    """ Returns shapely polygon given radius and center """

    point_list = [geometry.Point(radius * np.cos(theta) + center[0], radius * np.sin(
        theta) + center[1]) for theta in np.linspace(0, 2 * np.pi, step)]
    polygon = geometry.Polygon([[p.x, p.y] for p in point_list])

    return polygon


class Agent:

    def __init__(self, radius=None, bounding_box=None, length=None, region=None):
        """ Agent object"""

        self.fitness = -1000  # Dummy value
        self.radius = radius
        self.bounding_box = bounding_box
        self.length = length

        if region != None:
            """ Generates random circles inside the region for an inital guess """
            minx, miny, maxx, maxy = region.bounds
            tupled =  generate_random_in_polygon(self.length, region)
            self.circle_list = [get_circle(self.radius, (c[0], c[1])) for c in tupled]
            self.remove_irrelavent_circles(region, .05, .05)

    def update_agent(self):
        self.length = len(self.circle_list)
        self.update_centers()
        self.update_voronoi()

    def remove_irrelavent_circles(self, region, threshold_region, threshold_self):
        """ Removes all circles in circle_list that intrsect the region less than threshold returns circles that were removed """

        original_circle_list = self.circle_list

        kept_circles, removed_circles = [], []
        for circle in self.circle_list:
            frac = circle.intersection(region).area / circle.area
            if frac < threshold_region:
                removed_circles.append(circle)
            else:
                kept_circles.append(circle)
       
        self.circle_list = kept_circles

        kept_circles, removed_circles = [], []
        for circle in self.circle_list:
            rem_circle_list = self.circle_list[:] #Removes circle in list copy so we can check how much it intersects other circles
            rem_circle_list = [circle for circle in rem_circle_list if circle not in removed_circles]
            rem_circle_list.remove(circle)

            frac = np.abs(unary_union(rem_circle_list).intersection(circle).area - circle.area) / circle.area

            if frac < threshold_self:
                removed_circles.append(circle)
            else:
                kept_circles.append(circle)


        self.circle_list = kept_circles

        return [circle for circle in original_circle_list if circle not in self.circle_list]

    def get_intersections(self, region, bounding_box):
        """ Returns all types of intersections. self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction """


        self_intersection, self_intersection_fraction = double_intersection(
            self.circle_list)

        region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction = intersection_region(
            region, self.circle_list, bounding_box)

        if isinstance(self_intersection, geometry.GeometryCollection):
            self_intersection = geometry.MultiPolygon([shape for shape in self_intersection if not isinstance(shape, geometry.LineString)])
        
        if isinstance(region_intersection, geometry.GeometryCollection):
            region_intersection = geometry.MultiPolygon([shape for shape in region_intersection if not isinstance(shape, geometry.LineString)])

        if isinstance(region_nonintersection, geometry.GeometryCollection):
            region_nonintersection = geometry.MultiPolygon([shape for shape in region_nonintersection if not isinstance(shape, geometry.LineString)])

        return self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction

    def plot_agent(self, region, bounding_box, ax=None, zorder=1, fill=True):

        """ Plots circle intersection and non interesection with region as well as self intersection"""

        color1, color2, color3 = colors[1], colors[2], colors[3]

        #makes sure everything is nice and updated
        self.update_agent()

        self_intersection, _, region_intersection, region_nonintersection, _, _ = self.get_intersections(
            region, bounding_box)

        empty_region = region.difference(unary_union(self.circle_list))

        if fill:
            if isinstance(self_intersection, geometry.MultiPolygon):
                for p1 in self_intersection:
                    x1, y1 = p1.exterior.xy

                    if ax == None:
                        plt.fill(x1, y1, c=color1, zorder=zorder+.1)
                    else:
                        ax.fill(x1, y1, c=color1, zorder=zorder+.1)
            elif isinstance(self_intersection, geometry.Polygon):
                p1 = self_intersection
                x1, y1 = p1.exterior.xy

                if ax == None:
                    plt.fill(x1, y1, c=color1, zorder=zorder+.1)
                else:
                    ax.fill(x1, y1, c=color1, zorder=zorder+.1)
            else:
                raise Exception("Not polygon or mutlipolygon")

            if isinstance(region_nonintersection, geometry.MultiPolygon):
                for p3 in region_nonintersection:
                    x3, y3 = p3.exterior.xy

                    if ax == None:
                        plt.fill(x3, y3, c=color3, zorder=zorder)
                    else:
                        ax.fill(x3, y3, c=color3, zorder=zorder)
            elif isinstance(region_nonintersection, geometry.Polygon):
                p3 = region_nonintersection
                x3, y3 = p3.exterior.xy

                if ax == None:
                    plt.fill(x3, y3, c=color3, zorder=zorder)
                else:
                    ax.fill(x3, y3, c=color3, zorder=zorder)
            else:
                raise Exception("Not polygon or mutlipolygon")

            if isinstance(region_intersection, geometry.MultiPolygon):
                for p2 in region_intersection:
                    x2, y2 = p2.exterior.xy

                    if ax == None:
                        plt.fill(x2, y2, c=color2, zorder=zorder)
                    else:
                        ax.fill(x2, y2, c=color2, zorder=zorder)

            elif isinstance(region_intersection, geometry.Polygon):
                p2 = region_intersection
                x2, y2 = p2.exterior.xy

                if ax == None:
                    plt.fill(x2, y2, c=color2, zorder=zorder)
                else:
                    ax.fill(x2, y2, c=color2, zorder=zorder)
            else:
                raise Exception("Not poygon or multipolygon")

            if isinstance(empty_region, geometry.MultiPolygon):
                for p4 in empty_region:
                    x2, y2 = p4.exterior.xy

                    if ax == None:
                        plt.fill(x2, y2, c='k', zorder=zorder-.3)
                    else:
                        ax.fill(x2, y2, c='k', zorder=zorder-.3)

            elif isinstance(empty_region, geometry.Polygon):
                p4 = empty_region
                x2, y2 = p4.exterior.xy

                if ax == None:
                    plt.fill(x2, y2, c='k', zorder=zorder-.3)
                else:
                    ax.fill(x2, y2, c='k', zorder=zorder-.3)

        else:
            if ax == None:
                for circle in self.circle_list:
                    plt.plot(*circle.exterior.xy, c='k')
            else:
                for circle in self.circle_list:
                    ax.plot(*circle.exterior.xy, c='k')

        labels = ["Fitness: {}".format(self.fitness), "Number of Circles: {}".format(self.length)]
        if ax == None:
            plt.legend(labels, loc='upper left')
        else:
            ax.legend(labels, loc='upper left')

    def move_circle(self, old_circle, delta_x, delta_y): 
        """ Moves circle from circle_list to new center """

        old_center = old_circle.centroid
 
        try:
            self.circle_list.remove(old_circle)
        except:
            raise IndexError("The circle entered was not found in this agent")

        new_circle = get_circle(self.radius, (old_center.x + delta_x, old_center.y + delta_y))

        self.circle_list.append(new_circle)

    def update_centers(self):
        """ sets a list of centers given polyogn centers """

        self.center_list = [shape.centroid for shape in self.circle_list]

    def update_voronoi(self):
        """ Get vornoi diagrams given circles """

        self.update_centers() #makes sure there centers are accurate
        center_points = [(item.x, item.y) for item in self.center_list]

        boundary = np.array(
        [bounding_box["bottom left"], bounding_box["bottom right"], bounding_box["top right"], bounding_box["top left"]])

        x, y = boundary.T
        diameter = np.linalg.norm(boundary.ptp(axis=0))

        self.voronoi_list = []
        boundary_polygon = geometry.Polygon(boundary)
        for p in voronoi_polygons(Voronoi(center_points), diameter):
            self.voronoi_list.append(p.intersection(boundary_polygon))

    def plot_centers(self, zorder):
        """ Plots the centers of the circles in black """
        self.update_centers()

        for center in self.center_list:
            plt.scatter(center.x, center.y, c='k', zorder=zorder, s=2)

    def plot_voronoi(self, zorder, alpha):
        """ Plots voronoi diagrams """

        self.update_voronoi()

        for poly in self.voronoi_list:
            plt.fill(*poly.exterior.xy, zorder=zorder, alpha=alpha)

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

        if intersec.is_empty:
            continue

        if isinstance(intersec, geometry.GeometryCollection): #For some reason linestrings are getting appended so i'm removing them
            new_intersec = geometry.GeometryCollection([layer_precision(shape).buffer(0) for shape in intersec if not isinstance(shape, geometry.LineString)])
            intersections.append(new_intersec)
        elif isinstance(intersec, geometry.MultiPolygon):
            new_intersec = unary_union([layer_precision(poly).buffer(0) for poly in list(intersec)])
            intersections.append(new_intersec)
        else:
            intersections.append(layer_precision(intersec).buffer(0))

    intersection = unary_union(intersections)
    intersection_area = intersection.area
    total_area = unary_union(polygon_list).area
    frac_intersection = intersection_area / total_area
    return intersection, frac_intersection

def intersection_region(region, polygon_list, bounding_box):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    bounding_box = geometry.Polygon([bounding_box["bottom left"], bounding_box["bottom right"], bounding_box["top right"], bounding_box["top left"]])

    polygon_union = cascaded_union(polygon_list)
    intersection = region.intersection(polygon_union)
    fraction_overlap = intersection.area / region.area
    nonoverlap = polygon_union.difference(region)
    fraction_nonoverlap = nonoverlap.area / bounding_box.area

    return intersection, nonoverlap, fraction_overlap, fraction_nonoverlap

def intersection_area_inv(center_array, region, radius):
    """ Returns inverse of area intersection with region """

    real_centers = grouper(2, center_array)
    polygon_list = [get_circle(radius, center) for center in real_centers]

    #We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse
    r= region.intersection(unary_union(polygon_list)).area
    s=3
    soft_inv = 1 / ((1 + (r**s)) ** (1/s))
    return soft_inv

def repair_agent_BFGS(agent, region, plot=False, debug=False, generation=0, agent_number=0):
    """ Given agent uses quasi newton secant update to rearrange circles in agent to cover the region """

    if region.difference(unary_union(agent.circle_list)).area < .01: #Check if we even need to update
        return True

    agent.update_centers()

    #Guess is just the current circle list
    agent.update_centers()
    tupled = [(c.x, c.y) for c in agent.center_list]
    guess = [item for sublist in tupled for item in sublist]

   
    optimized = optimize.minimize(intersection_area_inv, guess, args=(region, agent.radius), method="BFGS")


    tupled_guess = grouper(2, guess)
    tupled_optimized = grouper(2, optimized.x)

    #Reassigns circle list
    agent.circle_list = [get_circle(agent.radius, center) for center in tupled_optimized]

    agent.remove_irrelavent_circles(region, .05, .05)

    if debug:
        print("Optimization was {}".format(optimized.success))

    if plot:
        os.mkdir("repair_frames/generation_{}/agent_{}".format(generation, agent_number))
        #Plotting guess
        agent.circle_list = [get_circle(agent.radius, center) for center in tupled_guess]
        plt.figure(figsize=(6,6))
        plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
        plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
        agent.plot_agent(region, bounding_box)
        plt.plot(*region.exterior.xy)
        agent.plot_centers(2)
        plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, "guess"))
        plt.close()

        #Plotting actual
        agent.circle_list = [get_circle(agent.radius, center) for center in tupled_optimized]
        plt.figure(figsize=(6,6))
        plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
        plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
        agent.plot_agent(region, bounding_box)
        plt.plot(*region.exterior.xy)
        agent.plot_centers(2)
        plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, "BFGS optimized"))
        plt.close()

    if region.difference(unary_union(agent.circle_list)).area < .01: #Precision errors
        return True
    else:
        return False


# GA part
def repair_agents(agent_list, region, plot=False, generation=0, guess=False): 
    """ Given a list of agents returns a list of repaired agents """
    if plot == True:
        if generation == 0:
            # Clearing folder before we add new frames
            folder = "/home/n/Documents/Research/GW-Localization-Tiling/repair_frames/"
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.mkdir("/home/n/Documents/Research/GW-Localization-Tiling/repair_frames/generation_{}".format(generation))
    repaired_agent_list = []
    for i, agent in enumerate(agent_list): 
        printProgressBar(i, len(agent_list))
        if repair_agent_BFGS(agent, region, plot=plot, generation=generation, agent_number=i, debug=False):
            repaired_agent_list.append(agent)

    return repaired_agent_list

def init_agents(radius, bounding_box, region, length=20):

    return [Agent(radius=radius, bounding_box=bounding_box, length=length, region=region) for _ in range(population)]
    

def fitness(agent_list, region, bounding_box, initial_length):

    alpha = 2
    beta = 1
    chi = 1

    for agent in agent_list:

        _, _, frac_overlap, frac_nonoverlap = intersection_region(region, agent.circle_list, bounding_box)
        _, frac_self_intersection = double_intersection(agent.circle_list)

        agent.fitness = 10 - (alpha * (agent.length/initial_length)) - (beta * frac_nonoverlap) - (chi * frac_self_intersection)

    return agent_list


def selection(agent_list):

    agent_list = sorted(
        agent_list, key=lambda agent: agent.fitness, reverse=True)
    # DARWINISM HAHHAA
    agent_list = agent_list[:int(.8 * len(agent_list))]

    return agent_list


def crossover(agent_list, region, plot=False, generation=0):
    """ Crossover is determined by mixing voronoi diagrams """

    def breed_agents(parent1, parent2):
        """ Breeds agents based on their voronoi diagrams. Takes two parents as input and spits out 2 children as output """
        parent1.update_centers() #Even though voronoi updates the centers for readability
        parent2.update_centers()

        parent1.update_voronoi()
        parent2.update_voronoi()

        child1_center_list = []
        for i, vor_poly in enumerate(parent1.voronoi_list):
            #Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

            parent1_pt = parent1.center_list[i]

            parent2_pts = [pt for pt in parent2.center_list if vor_poly.contains(pt)] #Generating list of points from parent 2 which are in vor_poly

            choice = random.choice([1,2])

            if choice == 1 or parent2_pts == []:
                child1_center_list.append(parent1_pt)
            elif choice == 2:
                child1_center_list.extend(parent2_pts)

        child2_center_list = []
        for i, vor_poly in enumerate(parent2.voronoi_list):
            #Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

            parent1_pts = [pt for pt in parent1.center_list if vor_poly.contains(pt)] #Generating list of points from parent 2 which are in vor_poly

            parent2_pt = parent2.center_list[i]

            choice = random.choice([1,2])

            if choice == 1 or parent1_pts == []:
                child2_center_list.extend(parent1_pts)
            elif choice == 2:
                child2_center_list.append(parent2_pt)

        child1 = Agent(radius=parent1.radius, bounding_box=parent1.bounding_box)
        child2 = Agent(radius=parent1.radius, bounding_box=parent1.bounding_box)
        child1.circle_list = [get_circle(child1.radius, (c.x, c.y)) for c in child1_center_list]
        child2.circle_list = [get_circle(child2.radius, (c.x, c.y)) for c in child2_center_list]

        return child1, child2

    offspring = []
    for i in range(round(len(agent_list) / 2)):
        
        #Randomly select parents from agent list, the same parent can (and probably will) be selected more than once at least once
        parent1 = random.choice(agent_list)
        parent2 = random.choice(agent_list)

        child1, child2 = breed_agents(parent1, parent2)
        offspring.append(child1)
        offspring.append(child2)

        if plot:
            os.mkdir("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}".format(generation, i))

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            parent1.plot_voronoi(2, .3)
            parent1.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/parent1_voronoi.png".format(generation, i))
            plt.close()

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            parent2.plot_voronoi(2, .3)
            parent2.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/parent2_voronoi.png".format(generation, i))
            plt.close()

            child1, child2 = breed_agents(parent1, parent2)

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            child1.plot_voronoi(2, .3)
            child1.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/child1_voronoi.png".format(generation, i))
            plt.close()

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            child2.plot_voronoi(2, .3)
            child2.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/child2_voronoi.png".format(generation, i))
            plt.close()


            #Plotting actual agents
            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            parent1.plot_agent(test_polygon, bounding_box)
            parent1.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/parent1.png".format(generation, i))
            plt.close()

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            parent2.plot_agent(test_polygon, bounding_box)
            parent2.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/parent2.png".format(generation, i))
            plt.close()

            child1, child2 = breed_agents(parent1, parent2)

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            child1.plot_agent(test_polygon, bounding_box)
            child1.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/child1.png".format(generation, i))
            plt.close()

            plt.figure(figsize=(6,6))
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*test_polygon.exterior.xy)
            child2.plot_agent(test_polygon, bounding_box)
            child2.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/child2.png".format(generation, i))
            plt.close()

    agent_list.extend(offspring)

    return agent_list


def mutation(agent_list, region):

    for agent in agent_list:

        if random.uniform(0, 1) <= .2:
            worst_circle_self = sorted(agent.circle_list, key=lambda x: unary_union(agent.circle_list).intersection(x).area)[0] #Finds circle which intersects with itself the most

            agent.circle_list.remove(worst_circle_self)

        if random.uniform(0, 1) <= .3:
            worst_circle_region = sorted(agent.circle_list, key=lambda x: region.intersection(x).area)[-1] #Finds circle which intersects region the least

            agent.circle_list.remove(worst_circle_region)

        if random.uniform(0, 1) <= .3:

            circle_to_move = random.choice(agent.circle_list) #Chooses a random circle to move
            delta_x = random.uniform(-.1, .1) #How much to move it
            delta_y = random.uniform(-.1, .1)
            agent.move_circle(circle_to_move, delta_x, delta_y)

    return agent_list

def ga(region, radius, bounding_box, initial_length=100, plot_regions=False, save_agents=False, plot_crossover=False):


    start = time.process_time() #Timing entire program

    before = time.process_time()
    print("Initializing Agents...")
    agent_list = init_agents(radius, bounding_box, region, length=initial_length)
    print("Agents initialized. Run time {}".format(time.process_time() - before))

    for generation in range(generations):

        generation_start = time.process_time()

        print("\ngeneration number: {}".format(generation))

        if generation == 0:
            before = time.process_time()
            print("Repairing Agents")
            agent_list = repair_agents(agent_list, region, plot=plot_regions, generation=generation, guess=True)
            if save_agents:
                os.mkdir("/home/n/Documents/Research/GW-Localization-Tiling/saved_agents/generation_{}".format(generation))
                for i, agent in enumerate(agent_list):
                    with open('/home/n/Documents/Research/GW-Localization-Tiling/saved_agents/generation_{}/agent_{}.obj'.format(generation, i), 'wb') as output:
                        pickle.dump(agent, output, pickle.HIGHEST_PROTOCOL)  
            print("Sucessful. {} Agents remain. Run time {}".format(len(agent_list), time.process_time() - before))
            print()

        before = time.process_time()
        print("Determining Fitness")
        agent_list = fitness(agent_list, region, bounding_box, initial_length)
        print("Sucessful. Run time {}".format(time.process_time() - before))
        print()

        before = time.process_time()
        print("Executing stragllers")
        agent_list = selection(agent_list)
        print("Sucessful. Run time {}".format(time.process_time() - before))
        print()

        before = time.process_time()
        agent_list.sort(key=lambda x: x.fitness, reverse=True)

        #Creating folder
        os.mkdir("/home/n/Documents/Research/GW-Localization-Tiling/frames/generation_{}".format(generation))

        for i, agent in enumerate(agent_list):
            plt.figure(figsize=(6,6))
            agent.plot_agent(
                region, bounding_box, zorder=2, fill=True)
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*region.exterior.xy)
            plt.savefig("frames/generation_{}/agent_{}".format(generation, i))
            plt.close()

        print("frame saved in frames/generation_{}. Run time {}".format(generation, time.process_time() - before))
        print()

        before = time.process_time()
        print("Beginning crossover")
        os.mkdir("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}".format(generation))
        agent_list = crossover(agent_list, region, plot=plot_crossover, generation=generation)
        print("Sucessful. Run time {}".format(time.process_time() - before))
        print()

        before = time.process_time()
        print("Mutating random agents")
        agent_list = mutation(agent_list, region)
        print("Sucessful. Run time {}".format(time.process_time() - before))
        print()

        if generation > 0:
            before = time.process_time()
            print("Repairing Agents")
            agent_list = repair_agents(agent_list, region, plot=plot_regions, generation=generation)
            if save_agents:
                os.mkdir("/home/n/Documents/Research/GW-Localization-Tiling/saved_agents/generation_{}".format(generation))
                for i, agent in enumerate(agent_list):
                    with open('/home/n/Documents/Research/GW-Localization-Tiling/saved_agents/generation_{}/agent_{}.obj'.format(generation, i), 'wb') as output:
                        pickle.dump(agent, output, pickle.HIGHEST_PROTOCOL)  
            print("Sucessful. {} Agents remain. Run time {}".format(len(agent_list), time.process_time() - before))
            if len(agent_list) == 0:
                break
            print()

        print()
        print("Completed. Generational run time {}".format(time.process_time() - generation_start))
        print()
        print()

    print("Finished. Total execution time {}".format(time.process_time() - start))

global population
population = 10

global generations
generations = 30

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

bounding_box = {"bottom left": (-2, -2),
                "bottom right": (2, -2),
                "top right": (2, 2),
                "top left": (-2, 2)}

test_polygon = geometry.Polygon([(-.6, -.6), (.6, -.6), (.6, .6), (-.6, .6)])

random_polygon_pts = generatePolygon(ctrX=0, ctrY=0, aveRadius=150, irregularity=0.35, spikeyness=0.2, numVerts=16 )
random_polygon = geometry.Polygon(random_polygon_pts)

# Clearing folder before we add new frames
folders_to_clear = ['/home/n/Documents/Research/GW-Localization-Tiling/frames', '/home/n/Documents/Research/GW-Localization-Tiling/repair_frames', '/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames']
for folder in folders_to_clear:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

ga(test_polygon, .2, bounding_box, initial_length=5, plot_regions=True, save_agents=False, plot_crossover=False)

#Testing code region
# filehandler1 = open("/home/n/Documents/Research/GW-Localization-Tiling/saved_agents/generation_0/agent_0.obj", 'rb') 
# filehandler2 = open("/home/n/Documents/Research/GW-Localization-Tiling/saved_agents/generation_0/agent_1.obj", 'rb') 
# parent1 = pickle.load(filehandler1)
# parent2 = pickle.load(filehandler2)

#agent = Agent(radius=.2, bounding_box=bounding_box, length=10)
#tmp = repair_agent_skewer(agent, test_polygon, plot=True)
# tmp = repair_agent_BFGS(agent, test_polygon, debug=False, plot=True)
# print(tmp)
# plt.figure(figsize=(6,6))
# plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
# plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
# agent.plot_agent(test_polygon, bounding_box)
# plt.plot(*test_polygon.exterior.xy)
# plt.show()