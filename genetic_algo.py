import numpy as np
from shapely import geometry
from shapely import ops
from shapely.ops import cascaded_union
from rtree import index
import random
import matplotlib.pyplot as plt


class Circle:

    def __init__(self, _radius, _center, step = 100):
        
        #Var inits
        self.center = geometry.Point(_center)
        self.radius = _radius
        self.point_list = [geometry.Point(self.radius * np.cos(theta) + self.center.x, self.radius * np.sin(theta) + self.center.y) for theta in np.linspace(0, 2 * np.pi, step)]
        self.polygon = geometry.Polygon([[p.x, p.y] for p in self.point_list])

    def move_circle(self, amt_left, amt_right, step = 100):

        self.center = geometry.Point(self.center.x + amt_left, self.center.y + amt_right)
        self.point_list = [geometry.Point(self.radius * np.cos(theta) + self.center.x, self.radius * np.sin(theta) + self.center.y) for theta in np.linspace(0, 2 * np.pi, step)]
        self.polygon = geometry.Polygon([[p.x, p.y] for p in self.point_list])

    def print_properties(self):
        print("center: {} \n \n radius: {} \n \n points: {} \n \n polygon: {}".format(self.center, self.radius, self.point_list, self.polygon))

class Agent:

    def __init__(self, _radius, bounding_box, length):

        """ bounding_box is a list w/ bottom left corner and top right corner """

        self.fitness = -1000 #Dummy value 
        self.length = length
        self.radius = _radius
        self.bounding_box = bounding_box
        self.circle_list = [Circle(self.radius, (random.uniform(bounding_box[0][0], bounding_box[1][0]),\
            random.uniform(bounding_box[0][1], bounding_box[1][1])))  for _ in range(length)]

    def plot_agent(self):

        for i in range(len(self.circle_list)):
            x,y = self.circle_list[i].polygon.exterior.xy
            plt.plot(x,y)
        
def double_intersection(polygon_list):
   
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
    return intersection_area

# Can come back to this later if the genetic code is getting stuck
# def triple_intersection(polygon_list):
#     listpoly = [a.intersection(b) for a, b in combinations(polygon_list, 2)]
#     rings = [geometry.LineString(list(pol.exterior.coords)) for pol in listpoly]
#     union = ops.unary_union(rings)
#     result = [geom for geom in ops.polygonize(union)]
#     return [intersect.area for intersect in result]

def intersection_region_fraction(region, polygon_list):
    return region.intersection(cascaded_union(polygon_list)).area / region.area

def total_area_fraction(bounding_box, polygon_list):

    bounding_box = geometry.box(bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1])

    return bounding_box.intersection(cascaded_union(polygon_list)).area / bounding_box.area

#GA part
def init_agents(radius, soft_bounding_box, length = 100):

    return [Agent(radius, soft_bounding_box, length) for _ in range(population)]

def fitness(agent_list, region, bounding_box):

    alpha = 1
    beta = 1
    chi = 1

    for agent in agent_list:

        agent_polygon_list = [circle.polygon for circle  in agent.circle_list]
        agent.fitness = (alpha * intersection_region_fraction(region, agent_polygon_list)) - (beta * double_intersection(agent_polygon_list))\
            - (chi * total_area_fraction(bounding_box, agent_polygon_list))

    return agent_list

def selection(agent_list):

    agent_list = sorted(agent_list, key=lambda agent: agent.fitness, reverse=True)
    #DARWINISM HAHHAA
    agent_list = agent_list[:int(.2 * len(agent_list))]

    return agent_list

def crossover(agent_list, region):

    """ Crossover is determined by randomly splitting polygon in half and then taking circles from each side. It'll probably mess up on the
    boundary tbh but I couldn't think of another crossover function """

    def getExtrapoledLine(p1,p2):
        'Creates a line extrapoled in p1->p2 direction'
        EXTRAPOL_RATIO = 10
        a = (p2[0]+EXTRAPOL_RATIO*(p1[0]-p2[0]), p2[1]+EXTRAPOL_RATIO*(p1[1]-p2[1]))
        b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]))
        return geometry.LineString([a,b])

    line = getExtrapoledLine((region.centroid.x, region.centroid.y), (region.centroid.x + random.uniform(-1, 1), region.centroid.y\
        + random.uniform(-1, 1)))

    merged = ops.linemerge([region.boundary, line])
    borders = ops.unary_union(merged)
    polygons = list(ops.polygonize(borders))

    offspring = []
    for _ in range(round(len(agent_list) / 2)):

        parent1 = random.choice(agent_list)
        parent2 = random.choice(agent_list)
        parent1_circle_half_1 = [circle for circle in parent1.circle_list if circle.center.within(polygons[0])]
        parent1_circle_half_2 = [circle for circle in parent1.circle_list if circle.center.within(polygons[1])]
        parent2_circle_half_1 = [circle for circle in parent2.circle_list if circle.center.within(polygons[0])]
        parent2_circle_half_2 = [circle for circle in parent2.circle_list if circle.center.within(polygons[1])]
        child1 = Agent(parent1.radius, parent1.bounding_box, parent1.length)
        child2 = Agent(parent1.radius, parent1.bounding_box, parent1.length)
        child1.circle_list = parent1_circle_half_1 + parent2_circle_half_2
        child2.circle_list = parent2_circle_half_1 + parent1_circle_half_2

        offspring.append(child1)
        offspring.append(child2)

    agent_list.extend(offspring)

    return agent_list

def mutation(agent_list):

    for agent in agent_list:

        for i, param in enumerate(agent.circle_list):
                
            if random.uniform(0, 1) <= .05:

                agent.circle_list.pop(i)
                agent.length -= 1

            if random.uniform(0, 1) <= .2:

                agent.circle_list[i].move_circle(random.uniform(-1, 1), random.uniform(-1, 1))

    return agent_list

def ga(region, soft_bounding_box, hard_bounding_box):

    print("Initializing Agents...")
    agent_list = init_agents(.1, soft_bounding_box)
    print("Agents initialized.")

    for generation in range(generations):

        print("\ngeneration number: {}".format(generation))

        print("Determining how much they lift..")
        agent_list = fitness(agent_list, region, hard_bounding_box)
        print("Sucessful")
        print()
        print("Executing stragllers")
        agent_list = selection(agent_list)
        print("Darwin has spoken")
        print()
        print(len(agent_list))
        agent_list[0].plot_agent()
        plt.plot(*region.exterior.xy)
        plt.savefig("frames/generation_{0:03d}".format(generation))
        print("frame saved in frames folder")
        print()
        print("Beginning candlelight dinner")
        agent_list = crossover(agent_list, region)
        print("Young love phase is over")
        print()
        print("Blasting my guys with Radiation")
        agent_list = mutation(agent_list)
        print("Completed.")

        
        



global population
population = 100

global generations 
generations = 20

soft_bounding_box = [(-1,-1), (1,1)]
hard_bounding_box = [(-5, -5), (5, 5)]

test_polygon = geometry.Polygon([(0,0), (0,1), (1,1), (1,0)])

ga(test_polygon, soft_bounding_box, hard_bounding_box)