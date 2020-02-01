#Import
import healpy as hp
import numpy as np
import shutil
from scipy.spatial import SphericalVoronoi
import math
from shapely import geometry
import scipy.optimize as optimize
from shapely.ops import unary_union
from ligo.skymap.io import fits
from spherical_geometry.polygon import SphericalPolygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from misc_functions import *
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import proj3d                                         
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import time


def get_m(**plot_args):
    m = Basemap(**plot_args)
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
    return m

def get_flat_circle(radius, center, step=100):
    """ Returns shapely polygon given radius and center """

    point_list = [geometry.Point(radius * np.cos(theta) + center[0], radius * np.sin(
        theta) + center[1]) for theta in np.linspace(0, 2 * np.pi, step)]
    polygon = geometry.Polygon([[p.x, p.y] for p in point_list])

    return polygon

def draw_screen_poly(poly, m, **plot_args):
    radec = list(poly.to_radec())
    if radec == []:
        return 0
    lons, lats = radec[0][0], radec[0][1]
    x, y = m( lons, lats )
    xy = list(zip(x,y))
    poly = Polygon(xy, **plot_args)
    plt.gca().add_patch(poly)

def convert_fits_xyz(dataset, number, nested=True, nside = None):

        m, metadata = fits.read_sky_map('data/' + dataset + '/' + str(number) + '.fits', nest=None)

        if nside is None: 
            nside = hp.npix2nside(len(m))
        else:
            nside = nside

        #Obtain pixels covering the 90% region
        #Sorts pixels based on probability, 
        #then takes all pixel until cumulative sum is >= 90%
        mflat = m.flatten()
        i = np.flipud(np.argsort(mflat))
        msort = mflat[i]
        mcum = np.cumsum(msort)            
        ind_to_90 = len(mcum[mcum <= 0.9*mcum[-1]])

        area_pix = i[:ind_to_90]
        max_pix  = i[0]

        x, y, z = hp.pix2vec(nside,area_pix,nest=nested)

        return x, y, z

def get_circle(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """
    radius = fov/2
    lons = [phi + radius * math.cos(angle) for angle in np.linspace(0, 2 * math.pi, step)]
    lats = [theta + radius * math.sin(angle) for angle in np.linspace(0, 2 * math.pi, step)]
    ret = SphericalPolygon.from_radec(lons, lats)
    return ret

def powerful_union_area(polygon_list):
    """ I know spherical geometry has this but it doesn't work... Given a polygonlist returns the area of the union """
    big_poly = polygon_list[0]
    extra_area = 0
    for i, poly in enumerate(polygon_list):
        if i == 1:
            continue

        try:
            if math.isnan(big_poly.union(poly).area()):
                extra_area += poly.area()
                continue
        except AssertionError: #NOTE Shouldn't be here i think? but it might be fine 
            continue

        big_poly = big_poly.union(poly)

    return big_poly.area() + extra_area

def get_center(circle):
    
    """ Given circle returns ra dec of center of circle by projection """ #NOTE needs fixing 360 --> 0

    radec = list(circle.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    poly = geometry.Polygon(list(zip(lons, lats)))
    center = poly.centroid
    return (center.x , center.y)

def spherical_poly_to_poly(poly):
    radec = list(poly.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    poly = geometry.Polygon(list(zip(lons, lats)))
    return poly

def double_intersection(polygon_list):

    """ Returns intersection between polygons in polygon_list and the area of their intersection. Perhaps upgrade to cascaded_union in future if program is taking too long this would be a major speed up i think """
   
    intersections, already_checked = [], []
    for polygon in polygon_list:
        already_checked.append(polygon)
        try:
            union_of_polys = SphericalPolygon.multi_union(diff(polygon_list, already_checked))
        except AssertionError: #No intersection
            continue

        single_intersection = polygon.intersection(union_of_polys)
        intersections.append(single_intersection)

    intersection = SphericalPolygon.multi_union(intersections)
    intersection_area = intersection.area()
    total_area = SphericalPolygon.multi_union(polygon_list).area()
    frac_intersection = intersection_area / total_area


    return intersections, frac_intersection

def intersection_region(region, polygon_list):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    outside = region.invert_polygon()


    interior_intersections = [region.intersection(polygon) for polygon in polygon_list]
    interior_area = powerful_union_area(interior_intersections)
    interior_fraction = interior_area / region.area()

    exterior_intersections = [polygon.intersection(outside) for polygon in polygon_list]
    exterior_area = powerful_union_area(exterior_intersections)
    exterior_fraction = exterior_area / (4 * math.pi)

    
    return interior_intersections, exterior_intersections, interior_fraction, exterior_fraction

class Agent:

    def __init__(self, fov=None, length=None, region=None):
        """ Agent object"""

        self.fitness = -1000  # Dummy value
        self.fov = fov
        self.length = length

        if region != None:
            """ Generates random circles inside the region for an inital guess """
            tupled =  generate_random_in_polygon(self.length, spherical_poly_to_poly(region))
            self.circle_list = [get_circle(i[0], i[1], self.fov) for i in tupled]
            self.update_centers()
            self.remove_irrelavent_circles(region, .05, .05)

    def update_agent(self):
        self.remove_irrelavent_circles(region, .05, .05)

    def get_intersections(self, region):
        """ Returns all types of intersections. self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction """

        self_intersection, self_intersection_fraction = double_intersection(
            self.circle_list)

        region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction = intersection_region(
            region, self.circle_list)

        return self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction

    def remove_irrelavent_circles(self, region, threshold_region, threshold_self):
        """ Removes all circles in circle_list that intrsect the region less than threshold returns circles that were removed """

        original_circle_list = self.circle_list

        kept_circles, removed_circles = [], []
        for circle in self.circle_list:
            frac = circle.intersection(region).area() / circle.area()
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

            double_intersection_lst, _ = double_intersection(rem_circle_list)
            double_intersection_union = SphericalPolygon.multi_union(double_intersection_lst)
            frac = np.abs(double_intersection_union.intersection(circle).area() - circle.area()) / circle.area()

            if frac < threshold_self:
                removed_circles.append(circle)
            else:
                kept_circles.append(circle)


        self.circle_list = kept_circles

        return [circle for circle in original_circle_list if circle not in self.circle_list]

    def plot_agent(self, region, m, zorder=1, fill=True):
        
        color1, color2, color3 = colors[1], colors[2], colors[3]

        #makes sure everything is nice and updated
        self.update_agent()

        if fill:
            self_intersection, _, region_intersection, region_nonintersection, _, _ = self.get_intersections(region)
            for poly in self_intersection: #Filling in the actual circles
                draw_screen_poly(poly, m, color=color1, zorder=zorder+.5)

            for poly in region_intersection: #Filling in the actual circles
                draw_screen_poly(poly, m, color=color2, zorder=zorder)

            for poly in region_nonintersection: #Filling in the actual circles
                draw_screen_poly(poly, m, color=color3, zorder=zorder)
        else:
            for poly in self.circle_list:
                poly.draw(m)

    def move_circle(self, old_circle, delta_lon, delta_lat): 
        """ Moves circle from circle_list to new center """

        old_center = get_center(old_circle)
 
        try:
            self.circle_list.remove(old_circle)
        except:
            raise IndexError("The circle entered was not found in this agent")

        new_circle = get_circle(self.radius, (old_center[0] + delta_lon, old_center[1] + delta_lat))

        self.circle_list.append(new_circle)

    def update_centers(self):
        """ sets a list of centers given polyogn centers """

        self.center_list = [get_center(shape) for shape in self.circle_list]

    def update_voronoi(self):
        """ Get vornoi diagrams given circles """

        self.update_centers() #makes sure there centers are accurate
        lon, lat = zip(*self.center_list)
        X, Y, Z = lon_lat_to_xyz(lon, lat, 1)
        points = np.array(list(zip(X,Y,Z)))
        self.spher_vor = SphericalVoronoi(points)
        self.spher_vor.sort_vertices_of_regions()

        self.voronoi_list = []
        for region in self.spher_vor.regions:
            vert = self.spher_vor.vertices[region] #Gets the verticies for a particular voronoi region
            self.voronoi_list.append(SphericalPolygon(vert))

    def plot_voronoi(self, ax,  **plot_args):
        """ Plots voronoi diagrams """

        self.update_voronoi()

        lon, lat = zip(*self.center_list)
        X, Y, Z = lon_lat_to_xyz(lon, lat, 1)
        points = np.array(list(zip(X,Y,Z)))
        ax.scatter(points[...,0], points[...,1], points[...,2])   

        for region in self.spher_vor.regions:
            polygon = Poly3DCollection([self.spher_vor.vertices[region]], **plot_args)                
            polygon.set_color(np.random.rand(3,))                                             
            ax.add_collection3d(polygon)

    def plot_centers(self, m, zorder):
        """ Plots the centers of the circles in black """
        self.update_centers()

        for center in self.center_list:
            x, y = m(center[0], center[1])
            plt.scatter(x, y, c='k', zorder=zorder, s=2)

def proj_intersection_area_inv(center_array, region, fov):
    """ Returns inverse of area itnersection with region but projectiosn to lon lat space """

    real_centers = grouper(2, center_array)
    polygon_list = [get_flat_circle(fov/2, center) for center in real_centers]
    proj_region = spherical_poly_to_poly(region)

    r = proj_region.intersection(unary_union(polygon_list)).area
    #We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse   
    s=3
    soft_inv = 1 / ((1 + (r**s)) ** (1/s))

    return soft_inv

def repair_agent_BFGS(agent, region, plot=False, debug=False, generation=0, agent_number=0):
    """ Given agent uses quasi newton secant update to rearrange circles in agent to cover the region """

    if region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > .98: #Check if we even need to update
        return True

    agent.update_centers()

    #Guess is just the current circle list
    tupled = [(c[0], c[1]) for c in agent.center_list]
    guess = [item for sublist in tupled for item in sublist]

    optimized = optimize.minimize(proj_intersection_area_inv, guess, args=(region, agent.fov), method="BFGS")

    tupled_guess = grouper(2, guess)
    tupled_optimized = grouper(2, optimized.x)

    #Reassigns circle list
    agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_optimized]
    agent.remove_irrelavent_circles(region, .05, .05)


    if debug:
        print("Optimization was {}".format(optimized.success))

    if plot:
        os.mkdir("repair_frames/generation_{}/agent_{}".format(generation, agent_number))
        #Plotting guess
        m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
        agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_guess]
        agent.plot_agent(region, m)
        region.draw(m)
        agent.plot_centers(m ,2)
        plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, "guess"))
        plt.close()

        #Plotting actual
        m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
        agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_optimized]
        agent.plot_agent(region, m)
        region.draw(m)
        agent.plot_centers(m, 2)
        plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, "BFGS optimized"))
        plt.close()

    if region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > .98: #Precision errors #NOTE
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

def init_agents(fov, region, population, length=20):

    return [Agent(fov=fov, length=length, region=region) for _ in range(population)]
    
def fitness(agent_list, region, initial_length):

    alpha = 2
    beta = 1
    chi = 1

    for agent in agent_list:

        _, _, frac_overlap, frac_nonoverlap = intersection_region(region, agent.circle_list)
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
            parent1.update_voronoi()
            parent2.update_voronoi()

            child1_center_list = []
            for i, vor_poly in enumerate(parent1.voronoi_list):
                #Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

                parent1_pt = parent1.center_list[i]

                lons, lats = zip(*parent2.center_list)
                x,y,z = lon_lat_to_xyz(lons, lats, 1)
                parent2_center_list_xyz = list(zip(x,y,z))

                parent2_pts = [pt for pt in parent2_center_list_xyz if vor_poly.contains_point(pt)] #Generating list of points from parent 2 which are in vor_poly

                choice = random.choice([1,2])

                if choice == 1 or parent2_pts == []:
                    child1_center_list.append(parent1_pt)
                elif choice == 2:
                    x,y,z = zip(*parent2_pts)
                    lon, lat, _ = xyz_to_lon_lat(x, y, z)
                    pt = list(zip(lon, lat))
                    child1_center_list.extend(pt)


            child2_center_list = []
            for i, vor_poly in enumerate(parent2.voronoi_list):
                #Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

                parent2_pt = parent2.center_list[i]

                lons, lats = zip(*parent1.center_list)
                x,y,z = lon_lat_to_xyz(lons, lats, 1)
                parent1_center_list_xyz = list(zip(x,y,z))

                parent1_pts = [pt for pt in parent1_center_list_xyz if vor_poly.contains_point(pt)] #Generating list of points from parent 2 which are in vor_poly

                choice = random.choice([1,2])

                if choice == 1 or parent1_pts == []:
                    child2_center_list.append(parent2_pt)
                elif choice == 2:
                    x,y,z = zip(*parent1_pts)
                    lon, lat, _ = xyz_to_lon_lat(x, y, z)
                    pt = list(zip(lon, lat))
                    child2_center_list.extend(pt)

            child1 = Agent(fov=parent1.fov)
            child2 = Agent(fov=parent1.fov)
            child1.circle_list = [get_circle(c[0], c[1], child1.fov) for c in child1_center_list]
            child2.circle_list = [get_circle(c[0], c[1], child2.fov) for c in child2_center_list]

            child1.update_agent()
            child2.update_agent()

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
            m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
            region.draw(m)
            parent1.plot_voronoi(2, .3)
            parent1.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/parent1_voronoi.png".format(generation, i))
            plt.close()

            m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
            region.draw(m)
            parent2.plot_voronoi(2, .3)
            parent2.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/parent2_voronoi.png".format(generation, i))
            plt.close()

            child1, child2 = breed_agents(parent1, parent2)

            plt.figure(figsize=(6,6))
            m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
            region.draw(m)
            child1.plot_voronoi(2, .3)
            child1.plot_centers(3)
            plt.savefig("/home/n/Documents/Research/GW-Localization-Tiling/crossover_frames/generation_{}/{}/child1_voronoi.png".format(generation, i))
            plt.close()

            plt.figure(figsize=(6,6))
            m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
            region.draw(m)
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


def ga(region, fov, population, generations, initial_length=100, plot_regions=False, plot_crossover=False):

    start = time.process_time() #Timing entire program

    before = time.process_time()
    print("Initializing Agents...")
    agent_list = init_agents(fov, region, population, length=initial_length)
    print("Agents initialized. Run time {}".format(time.process_time() - before))

    for generation in range(generations):

        generation_start = time.process_time()

        print("\ngeneration number: {}".format(generation))

        if generation == 0:
            before = time.process_time()
            print("Repairing Agents")
            agent_list = repair_agents(agent_list, region, plot=plot_regions, generation=generation, guess=True)
            print("Sucessful. {} Agents remain. Run time {}".format(len(agent_list), time.process_time() - before))
            print()

        before = time.process_time()
        print("Determining Fitness")
        agent_list = fitness(agent_list, region, initial_length)
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
            m = get_m(projection=projection, lon_0=lon_0, resolution=resolution)
            agent.plot_agent(region, zorder=2, fill=True)
            region.draw(m)
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
            print("Sucessful. {} Agents remain. Run time {}".format(len(agent_list), time.process_time() - before))
            if len(agent_list) == 0:
                break
            print()

        print()
        print("Completed. Generational run time {}".format(time.process_time() - generation_start))
        print()
        print()

    print("Finished. Total execution time {}".format(time.process_time() - start))


dataset = 'design_bns_astro' # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8 # FOV diameter in degrees

#Open sample file, tested on 100
i = 232

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]



global projection; global lon_0; global resolution;
projection='moll'; lon_0=-70; resolution='c';

X, Y, Z = convert_fits_xyz(dataset, i)
inside_point = X[1], Y[1], Z[1] #It's probably inside ?? NOTE should be cahnged though
#We need to cluster the points before we convex hull
region = SphericalPolygon.convex_hull(list(zip(X,Y,Z)))


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

population = 1
generations = 1
#ga(region, 8, population, generations, initial_length=9, plot_regions=True, plot_crossover=False)


#Testing code region
