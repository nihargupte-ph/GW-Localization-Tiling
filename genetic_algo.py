import numpy as np
import os
import shutil
from shapely import geometry
from shapely import affinity
from scipy.spatial import Voronoi
from shapely import ops
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from rtree import index
import random
import matplotlib.pyplot as plt
from scipy.stats import expon
from misc_functions import *
import fiona
import time

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

        if isinstance(self_intersection, geometry.GeometryCollection):
            self_intersection = geometry.MultiPolygon([shape for shape in self_intersection if not isinstance(shape, geometry.LineString)])
        
        if isinstance(region_intersection, geometry.GeometryCollection):
            region_intersection = geometry.MultiPolygon([shape for shape in region_intersection if not isinstance(shape, geometry.LineString)])

        if isinstance(region_nonintersection, geometry.GeometryCollection):
            region_nonintersection = geometry.MultiPolygon([shape for shape in region_nonintersection if not isinstance(shape, geometry.LineString)])

        return self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction

    def plot_agent(self, region, color1, color2, color3, bounding_box, ax=None, zorder=1):

        """ Plots circle intersection and non interesection with region as well as self intersection"""

        #makes sure everything is nice and updated
        self.update_agent()

        self_intersection, _, region_intersection, region_nonintersection, _, _ = self.get_intersections(
            region, bounding_box)

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


        plt.legend(["Fitness: {}".format(self.fitness), "Number circles: {}".format(self.length)], loc='upper left')

    def move_circle(self, old_circle, center): 
        """ Moves circle from circle_list to new center """
 
        try:
            self.circle_list.remove(old_circle)
        except:
            raise IndexError("The circle entered was not found in this agent")

        new_circle = get_circle(self.radius, center)

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

    def plot_voronoi(self):
        """ Plots voronoi diagrams """

        self.update_voronoi()

        for poly in self.voronoi_list:
            plt.fill(*poly.exterior.xy, zorder=.1, alpha=.8)

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
    return intersection, intersection_area

def intersection_region(region, polygon_list, bounding_box):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    bounding_box = geometry.Polygon([bounding_box["bottom left"], bounding_box["bottom right"], bounding_box["top right"], bounding_box["top left"]])

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

def get_extrapolated_line(p1,p2):
    'Creates a line extrapoled in p1->p2 direction https://stackoverflow.com/questions/33159833/shapely-extending-line-feature'
    EXTRAPOL_RATIO = 1000
    a = p1
    b = (p1.x+EXTRAPOL_RATIO*(p2.x-p1.x), p1.y+EXTRAPOL_RATIO*(p2.y-p1.y) )
    return geometry.LineString([a,b])

def get_ordered_list(region, linrig, point):
    """ Given LinearRing and point returns list of points closest to said point from polygon """
    intersected_multilinrig = linrig.intersection(region)
    if isinstance(intersected_multilinrig, geometry.MultiLineString):
        final_point_list = []
        for intersected_linrig in list(intersected_multilinrig): #iterates through multilinestring and will flatten at the end
            x,y = intersected_linrig.xy
            x, y = list(x), list(y)
            point_list = list(zip(x,y))
            point_list.sort(key = lambda p: np.sqrt((p[0] - point.x)**2 + (p[1] - point.y)**2))
            point_list = [geometry.Point(p[0], p[1]) for p in point_list]
            final_point_list.append(point_list)

        ret  = [item for sublist in final_point_list for item in sublist] #Flattening
    elif isinstance(intersected_multilinrig, geometry.LineString):
        x,y = intersected_multilinrig.xy
        x, y = list(x), list(y)
        point_list = list(zip(x,y))
        point_list.sort(key = lambda p: np.sqrt((p[0] - point.x)**2 + (p[1] - point.y)**2))
        ret = [geometry.Point(p[0], p[1]) for p in point_list]
    elif isinstance(intersected_multilinrig, geometry.GeometryCollection):
        final_point_list = []
        for item in list(intersected_multilinrig):
            if isinstance(item, geometry.Point):
                point_list = [item]
            elif isinstance(item, geometry.LineString):
                x,y = item.xy
                x, y = list(x), list(y)
                point_list = list(zip(x,y))
                point_list.sort(key = lambda p: np.sqrt((p[0] - point.x)**2 + (p[1] - point.y)**2))
                point_list = [geometry.Point(p[0], p[1]) for p in point_list]
            final_point_list.append(point_list)
        ret  = [item for sublist in final_point_list for item in sublist] #Flattening

    else:
        raise Exception("intersected_multilinrig was not a multilinrig, linrig, or geometry collection")

    return ret

def repair_agent_skewer(agent, region, plot=False, save=False, agent_number=0, generation=0, debug=False):
    """ repairs agent to possibly cover the region, if the region is not covered it will return false if it is it will return true """

    timeout = time.time() + 60*1#adding timeout feature, times out after 3 minutes

    if plot == True:
        # Clearing folder before we add new frames
        os.mkdir("repair_frames/generation_{}/agent_{}".format(generation, agent_number))

    center_pt = region.centroid

    bounding_box_poly = geometry.Polygon([bounding_box["bottom left"], bounding_box["bottom right"], bounding_box["top right"], bounding_box["top left"]])

    #Assigns dotted region, if a circle does not already contain the centroid then it will create a small circle around the centroid
    dot_region = get_circle(.01, center_pt.xy)
    translated_circles = [] 
    for circle in agent.circle_list:
        if circle.contains(center_pt):
            dot_region = circle
            translated_circles.append(circle) #Even though it's not technically a translated circled for this purpose its important to add it since later we don't want to double translate it

    count = 0
    new_dot_region = geometry.Polygon([(-.1, -.2), (.2, -.2), (.2, .2), (-.2, .2)]) #NOTE THIS SHOULDN"T BE HERE ITS A TEMPORARY FIX
    while not new_dot_region.contains(region):
            
        if save == True:
            # Define a polygon feature geometry with one attribute
            schema = {
                'geometry': 'Polygon',
                'properties': {'id': 'int'},
            }

            # Write a new Shapefile
            with fiona.open('shape_files/{}.shp'.format(count), 'w', 'ESRI Shapefile', schema) as c:
                ## If there are multiple geometries, put the "for" loop here
                c.write({
                    'geometry': geometry.mapping(dot_region),
                    'properties': {'id': 123},
                })

        ordered_pts = get_ordered_list(region, dot_region.exterior, center_pt)

        for i, closest_pt in enumerate(ordered_pts): #Iterates through all the closest points till it intersects a circle

            if time.time() > timeout:
                if debug:
                    print(sector_list)
                    print(agent_number, count, "fail by timeout in for loop")
                return False

            line_segment = get_extrapolated_line(center_pt, closest_pt) #Creates extrapolated line between closest poitns 

            non_translated_circles = diff(agent.circle_list, translated_circles)
            if cascaded_union(non_translated_circles).intersects(line_segment): #First checks if the extrapolated line even intersects our circles

                intersected_circles = [circle for circle in non_translated_circles if circle.intersects(line_segment)]
            
                skewered_circle = min(intersected_circles, key=center_pt.distance) #Finds the correct skewered circle the closer one

                #Calculate where the line intersects the circle
                lring = geometry.polygon.LinearRing(list(skewered_circle.exterior.coords)) #Translate to line ring
                points_of_intersection = lring.intersection(line_segment)

                #In case the points of intersection is just a point and not a multipoint
                try: 
                    closest_intersection =  min(points_of_intersection, key=center_pt.distance)
                except TypeError:
                    closest_intersection = points_of_intersection

                #Calculate how much to translate the circle by
                delta_x = closest_pt.x - closest_intersection.x
                delta_y = closest_pt.y - closest_intersection.y
                translated_circle = affinity.translate(skewered_circle, xoff=delta_x, yoff=delta_y)


                #Moves circle
                agent.circle_list.remove(skewered_circle)
                agent.circle_list.append(translated_circle)
                
                translated_circles.append(translated_circle)

                #Increases size of dotted region
                try: 
                    dot_region = cascaded_union([dot_region, translated_circle])
                except ValueError:
                    break
  

                try: 
                    x,y = dot_region.exterior.xy #NOTE THIS SHOULDN"T BE HERE ITS A TEMPORARY FIX
                except AttributeError: #If you end up getting a multipolygon
                    if debug:
                        print(agent_number, count, "Exited because of multipolygon")
                    return False

                x,y = list(x), list(y)
                lst = list(zip(x,y))
                new_dot_region = geometry.Polygon(lst)



                #Check if we should end the algorithim via sector scan
                plot_sectors = False
                if region.exterior.intersects(dot_region.exterior):
                    plot_sectors = True
                    remaining_regions = region.difference(dot_region.intersection(region))

                    try: 
                        remaining_regions = list(remaining_regions)
                    except TypeError:
                        remaining_regions = [remaining_regions]
                    
                    try:
                        intersected_sectors = list(region.exterior.intersection(dot_region.exterior))

                        #Creating of list we are going to insert points into
                        grouped_intersected_sectors = [[] for i,_ in enumerate(remaining_regions)]
                        for i, remaining_region in enumerate(remaining_regions):
                            for pt in intersected_sectors:
                                if remaining_region.exterior.contains(pt):
                                    grouped_intersected_sectors[i].append(pt)
                        
                        tmp = []
                        for i, intersected_sector in enumerate(grouped_intersected_sectors):
                            if intersected_sector == []:
                                tmp.append(i)
                            if len(intersected_sector) == 1:
                                tmp.append(i)
                            if isinstance(intersected_sector, geometry.LineString):
                                if debug:
                                    print(agent_number, count, "Exited because of linestring error")
                                return False
                                
                        for index in sorted(tmp, reverse=True):
                            del grouped_intersected_sectors[index]
                            del remaining_regions[index]


                        sector_list = []
                        for intersected_sector, remaining_region in zip(grouped_intersected_sectors, remaining_regions):
                            line1 = get_extrapolated_line(center_pt, intersected_sector[0])
                            line2 = get_extrapolated_line(center_pt, intersected_sector[1])

                            bounding_box_pt1 = bounding_box_poly.exterior.intersection(line1)
                            bounding_box_pt2 = bounding_box_poly.exterior.intersection(line2)

                            #Finds sectors of lines
                            line_split_collection = [line1, line2]
                            line_split_collection.append(bounding_box_poly.boundary) # collection of individual linestrings for splitting in a list and add the polygon lines to it.
                            merged_lines = ops.linemerge(line_split_collection)
                            border_lines = ops.unary_union(merged_lines)
                            sector = list(ops.polygonize(border_lines))

                            try:
                                sector_0_area = sector[0].intersection(remaining_region).area
                                sector_1_area = sector[1].intersection(remaining_region).area
                            except:
                                continue


                            if sector_0_area > sector_1_area:
                                sector_list.append(sector[0])
                            else:
                                sector_list.append(sector[1])
                    except TypeError: #Tangent point intersection
                        plot_sectors = False


                #Plotting
                if plot == True:
                    plt.figure(figsize=(6,6))
                    plt.xlim([bounding_box["bottom left"][0] - .3, bounding_box["bottom right"][0] + .3])
                    plt.ylim([bounding_box["bottom left"][1] - .3, bounding_box["top left"][1] + .3])
                    plt.plot(*bounding_box_poly.exterior.xy, c='k')
                    agent.plot_agent(test_polygon, colors[1], colors[2], colors[3], bounding_box)                     
                    for i, _ in enumerate(dot_region.interiors):
                        plt.plot(*dot_region.interiors[i].coords.xy, c='w')

                    try:
                        for i, _ in enumerate(remaining_regions):
                            plt.fill(*remaining_regions[i].exterior.xy, alpha=.5)
                        plt.plot(*skewered_circle.exterior.xy)

                        if plot_sectors:
                            for sector in sector_list:
                                plt.fill(*sector.exterior.xy, alpha=.3, c='g')

                    except UnboundLocalError:
                        pass

                    plt.plot(*region.exterior.xy)
                    plt.plot(*line_segment.xy, c='y')
                    plt.plot(*dot_region.exterior.xy, c='k', linestyle='--')
                    plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, count))
                    plt.close()

                if region.exterior.intersects(dot_region.exterior):
                    #Checking if sectors contain appropriate area for the program to converge
                    for sector, remaining_region in zip(sector_list, remaining_regions):

                        sector_circle_area = sector.intersection(cascaded_union(non_translated_circles)).area #Area of circles in sector
                        remaining_area = remaining_region.area

                        if (sector_circle_area * 1) < remaining_area: #The .9 is there because there will definatlye be some overlap between the skewered circles
                            if debug:
                                print(agent_number, count, "Exited because of sector scan")
                            return False

                if len(translated_circles) == 1:
                    center_pt = translated_circles[0].centroid            

                count += 1
                break    

            else:
                if i == len(ordered_pts):
                    if debug:
                        print("Iterated through all points")
                        

    agent.remove_irrelavent_circles(region, .01)
    agent.update_agent()

    if debug:
        print(agent_number, " Sucessfully repaired")
    return True


# GA part
def repair_agents(agent_list, region, plot=False, generation=0): 
    """ Given a list of agents returns a list of repaired agents """
    if plot == True:
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
        if repair_agent_skewer(agent, region, plot=plot, generation=generation, agent_number=i, debug=True):
            repaired_agent_list.append(agent)

    return repaired_agent_list

def init_agents(radius, bounding_box, length=20):

    return [Agent(radius=radius, bounding_box=bounding_box, length=length) for _ in range(population)]
    

def fitness(agent_list, region, bounding_box):

    alpha = 10
    beta = 1
    chi = 1

    for agent in agent_list:

        _, _, frac_overlap, frac_nonoverlap = intersection_region(region, agent.circle_list, bounding_box)
        _, frac_self_intersection = double_intersection(agent.circle_list)

        agent.fitness = (alpha * frac_overlap) - (beta * frac_nonoverlap) - (chi * frac_self_intersection)

    return agent_list


def selection(agent_list):

    agent_list = sorted(
        agent_list, key=lambda agent: agent.fitness, reverse=True)
    # DARWINISM HAHHAA
    agent_list = agent_list[:int(.8 * len(agent_list))]

    return agent_list


def crossover(agent_list, region):
    """ Crossover is determined by randomly splitting polygon in half and then taking circles from each side. It'll probably mess up on the
    boundary tbh but I couldn't think of another crossover function """

    offspring = []
    for _ in range(round(len(agent_list) / 2)):

        parent1 = random.choice(agent_list)
        parent2 = random.choice(agent_list)

        len_children = (len(parent1.circle_list) + len(parent2.circle_list)) // 2

        parent1_sorted_circles = get_highest_area(region, parent1.circle_list)
        parent2_sorted_circles = get_highest_area(region, parent2.circle_list)

        parent1_sorted_circles[:len_children]
        parent2_sorted_circles[:len_children]

        parent1_circle_half_1 = list(parent1_sorted_circles[0::2])
        parent1_circle_half_2 = list(parent1_sorted_circles[1::2])
        parent2_circle_half_1 = list(parent2_sorted_circles[0::2])
        parent2_circle_half_2 = list(parent2_sorted_circles[1::2])


        child1 = Agent(radius=parent1.radius, bounding_box=parent1.bounding_box, adam=False)
        child2 = Agent(radius=parent1.radius, bounding_box=parent1.bounding_box, adam=False)
        child1.circle_list = parent1_circle_half_1 + parent2_circle_half_2
        child2.circle_list = parent2_circle_half_1 + parent1_circle_half_2

        child1.remove_irrelavent_circles(region, 1e-5)
        child2.remove_irrelavent_circles(region, 1e-5)

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

def ga(region, radius, bounding_box, initial_length=100, plot_regions=False):


    start = time.process_time() #Timing entire program

    before = time.process_time()
    print("Initializing Agents...")
    agent_list = init_agents(radius, bounding_box, length=initial_length)
    print("Agents initialized. Run time {}".format(time.process_time() - before))

    for generation in range(generations):

        generation_start = time.process_time()

        print("\ngeneration number: {}".format(generation))

        before = time.process_time()
        print("Repairing Agents")
        agent_list = repair_agents(agent_list, region, plot=plot_regions, generation=generation)
        print("Sucessful. {} Agents remain. Run time {}".format(len(agent_list), time.process_time() - before))
        print()

        before = time.process_time()
        print("Determining Fitness")
        agent_list = fitness(agent_list, region, bounding_box)
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
                region, colors[1], colors[2], colors[3], bounding_box, zorder=2)
            plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
            plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
            plt.plot(*region.exterior.xy)
            plt.savefig("frames/generation_{}/agent_{}".format(generation, i))
            plt.close()

        print("frame saved in frames/generation_{}. Run time {}".format(generation, time.process_time() - before))
        print()

        before = time.process_time()
        print("Beginning crossover")
        agent_list = crossover(agent_list, region)
        print("Sucessful. Run time {}".format(time.process_time() - before))
        print()

        before = time.process_time()
        print("Mutating random agents")
        agent_list = mutation(agent_list)
        print("Sucessful. Run time {}".format(time.process_time() - before))
        print()

        print()
        print("Completed. Generational run time {}".format(time.process_time() - generation_start))
        print()
        print()

        if generation == 1:
            exit()

    print("Finished. Total execution time {}".format(time.process_time() - start))
global population
population = 20

global generations
generations = 10

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

bounding_box = {"bottom left": (-2, -2),
                "bottom right": (2, -2),
                "top right": (2, 2),
                "top left": (-2, 2)}


test_polygon = geometry.Polygon([(-.4, -.4), (.4, -.4), (.4, .4), (-.4, .4)])

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

#ga(test_polygon, .2, bounding_box, initial_length=100, plot_regions=True)

#Testing code region
agent = Agent(radius=.2, bounding_box=bounding_box, length=20)
#tmp = repair_agent_skewer(agent, test_polygon, plot=True)
plt.figure(figsize=(6,6))
plt.xlim([bounding_box["bottom left"][0], bounding_box["bottom right"][0]])
plt.ylim([bounding_box["bottom left"][1], bounding_box["top left"][1]])
agent.plot_agent(test_polygon, colors[1], colors[2], colors[3], bounding_box)
plt.plot(*test_polygon.exterior.xy)
agent.plot_voronoi()
plt.show()