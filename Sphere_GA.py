# Import
import healpy as hp
import numpy as np
import pickle
import shutil
from scipy.spatial import SphericalVoronoi
from scipy.spatial import Voronoi
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
import warnings

warnings.filterwarnings("ignore")  # If you want to debug remove this
cwd = os.getcwd()

def get_m(**plot_args):
    """ Given plot args returns a basemap "axis" with the proper plot args. Edit this function if you want different maps """

    #m = Basemap(projection="ortho", resolution="c", lon_0=-20, lat_0=0, **plot_args)
    m = Basemap(projection="moll", resolution="c", lon_0=0)
    m.drawcoastlines()
    return m

def get_flat_circle(radius, center, step=100):
    """ Returns shapely polygon given radius and center. This is the same as get_circle in Flat_GA. Step is the number of verticies in the polygon, 
    the higher the steps the closer it is to a circle """

    point_list = [
        geometry.Point(
            radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1]
        )
        for theta in np.linspace(0, 2 * np.pi, step)
    ]
    polygon = geometry.Polygon([[p.x, p.y] for p in point_list])

    return polygon


def draw_screen_poly(poly, m, **plot_args):
    """ Given a polygon and basemap axis, fills in the polygon in on the basemap  """

    radec = list(poly.to_radec())
    if radec == []:
        return 0
    lons, lats = radec[0][0], radec[0][1]
    x, y = m(lons, lats)
    xy = list(zip(x, y))
    poly = Polygon(xy, **plot_args)
    plt.gca().add_patch(poly)


def get_circle(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """

    radius = fov / 2
    lons = [
        phi + radius * np.cos(angle) for angle in np.linspace(0, 2 * np.pi, step)
    ]
    lats = [
        theta + radius * np.sin(angle) for angle in np.linspace(0, 2 * np.pi, step)
    ]
    ret = SphericalPolygon.from_radec(lons, lats)
    return ret


def powerful_union_area(polygon_list):
    """ I know spherical geometry has this but it doesn't work... Given a polygonlist returns the area of the union """
    big_poly = polygon_list[0]
    extra_area = 0
    for i, poly in enumerate(polygon_list):
        if i == 1:  # Skips first big_poly because we already added it
            continue

        try:  # This is where the problem arises sometimes union results in nan which messes up the rest of the union
            if np.isnan(big_poly.union(poly).area()):
                extra_area += poly.area()
                continue
        except AssertionError:
            continue

        big_poly = big_poly.union(poly)

    return big_poly.area() + extra_area


def get_center(circle):
    """ Given circle returns ra dec of center of circle by projection """  # NOTE needs fixing 360 --> 0

    radec = list(circle.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    poly = geometry.Polygon(list(zip(lons, lats)))
    center = poly.centroid
    return (center.x, center.y)

def spherical_poly_to_poly(poly):
    """ Converts a spherical polygon to a lon lat polygon """

    radec = list(poly.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    poly = geometry.Polygon(list(zip(lons, lats)))
    return poly


def poly_to_spherical_poly(poly):
    """ Given shapely polygon (lon lat) returns spherical polygon """

    lons, lats = poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]
    ret = SphericalPolygon.from_radec(lons, lats)

    return ret


def double_intersection(polygon_list):
    """ Returns intersection between polygons in polygon_list and the area of their intersection. Perhaps upgrade to cascaded_union in future if program is taking too long this would be a major speed up """

    intersections, already_checked = [], []
    for polygon in polygon_list:
        already_checked.append(polygon)  # We don't need to check the polygons we already intersected with
        try:
            union_of_polys = SphericalPolygon.multi_union(diff(polygon_list, already_checked))
        except AssertionError:  # No intersection
            continue

        try:
            single_intersection = polygon.intersection(union_of_polys)
        except ValueError:  # No intersection between polygon and union_of_polys NOTE
            continue
        except AssertionError:  # No intersection between polygon and union_of_polys
            continue

        intersections.append(single_intersection)

    try:
        intersection = SphericalPolygon.multi_union(intersections)
    except:
        return False, False
    intersection_area = intersection.area()
    total_area = SphericalPolygon.multi_union(polygon_list).area()
    frac_intersection = intersection_area / total_area

    return intersections, frac_intersection


def intersection_region(region, polygon_list):
    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    outside = region.invert_polygon()

    try:
        interior_intersections = [
            region.intersection(polygon) for polygon in polygon_list
        ]
    except AssertionError:
        interior_intersections = [
            proj_intersection(region, polygon) for polygon in polygon_list
        ]
    interior_area = powerful_union_area(interior_intersections)
    interior_fraction = interior_area / region.area()

    try:
        exterior_intersections = [
            polygon.intersection(outside) for polygon in polygon_list
        ]
    except AssertionError:
        exterior_intersections = [
            proj_intersection(outside, polygon) for polygon in polygon_list
        ]
    exterior_area = powerful_union_area(exterior_intersections)
    exterior_fraction = exterior_area / (4 * np.pi)

    return (
        interior_intersections,
        exterior_intersections,
        interior_fraction,
        exterior_fraction,
    )


def breed_agents_ll(parent1, parent2):
    """ Breeds agents based on their voronoi diagrams. Takes two parents as input and spits out 2 children as output """
    parent1.update_agent()
    parent2.update_agent()

    child1_center_list = []
    for i, vor_poly in enumerate(parent1.voronoi_list_ll):
        # Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

        parent1_pt = parent1.center_list[i]

        # Generating list of points from parent 2 which are in vor_poly
        parent2_pts = [pt for pt in parent2.center_list if vor_poly.contains(geometry.Point(pt))]

        choice = random.choice([1, 2])

        if choice == 1 or parent2_pts == []:
            child1_center_list.append(parent1_pt)
        elif choice == 2:
            child1_center_list.extend(parent2_pts)

    child2_center_list = []
    for i, vor_poly in enumerate(parent2.voronoi_list_ll):
        # Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

        # Generating list of points from parent 2 which are in vor_poly
        parent1_pts = [pt for pt in parent1.center_list if vor_poly.contains(geometry.Point(pt))]

        parent2_pt = parent2.center_list[i]

        choice = random.choice([1, 2])

        if choice == 1 or parent1_pts == []:
            child2_center_list.extend(parent1_pts)
        elif choice == 2:
            child2_center_list.append(parent2_pt)

    child1 = Agent(fov=parent1.fov)
    child2 = Agent(fov=parent1.fov)
    child1.circle_list = [get_circle(c[0], c[1], child1.fov) for c in child1_center_list]
    child2.circle_list = [get_circle(c[0], c[1], child2.fov) for c in child2_center_list]

    child1.update_agent()
    child2.update_agent()

    return child1, child2


def breed_agents_spher(parent1, parent2):
    """ Breeds agents based on their voronoi diagrams. Takes two parents as input and spits out 2 children as output """
    parent1.update_voronoi()
    parent2.update_voronoi()

    child1_center_list = []
    for i, vor_poly in enumerate(parent1.voronoi_list):
        # Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

        parent1_pt = parent1.center_list[i]

        lons, lats = zip(*parent2.center_list)
        x, y, z = lon_lat_to_xyz(lons, lats, 1)
        parent2_center_list_xyz = list(zip(x, y, z))

        # Generating list of points from parent 2 which are in vor_poly
        parent2_pts = [pt for pt in parent2_center_list_xyz if vor_poly.contains_point(pt)]

        choice = random.choice([1, 2])


        if choice == 1 or parent2_pts == []:
            child1_center_list.append(parent1_pt)
        elif choice == 2:
            x, y, z = zip(*parent2_pts)
            lon, lat, _ = xyz_to_lon_lat(x, y, z)
            pt = list(zip(lon, lat))
            child1_center_list.extend(pt)

    child2_center_list = []
    for i, vor_poly in enumerate(parent2.voronoi_list):
        # Iterating through voronoi list and randomly selecting point_list from either parent 1 or parent 2

        parent2_pt = parent2.center_list[i]

        lons, lats = zip(*parent1.center_list)
        x, y, z = lon_lat_to_xyz(lons, lats, 1)
        parent1_center_list_xyz = list(zip(x, y, z))

        parent1_pts = [
            pt for pt in parent1_center_list_xyz if vor_poly.contains_point(pt)
        ]  # Generating list of points from parent 2 which are in vor_poly

        choice = random.choice([1, 2])

        if choice == 1 or parent1_pts == []:
            child2_center_list.append(parent2_pt)
        elif choice == 2:
            x, y, z = zip(*parent1_pts)
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


class Agent:
    def __init__(self, fov=None, length=None, region=None):
        """ Agent object"""

        self.fitness = -1000  # Dummy value
        self.fov = fov
        self.length = length

        if region != None:
            """ Generates random circles inside the region for an inital guess """
            poly = proj_poly(region)
            pts = generate_random_in_polygon(self.length, poly)

            inv_proj = inv_proj_poly(poly)
            inv_proj_pts = inv_proj_points(pts)
            self.circle_list = [get_circle(i[0], i[1], self.fov) for i in inv_proj_pts]
            self.update_centers()

    def update_agent(self):
        #self.remove_irrelavent_circles(region, 0.05, 0.05)
        self.update_centers()
        self.update_voronoi()
        self.update_voronoi_ll()
        self.length = len(self.circle_list)

    def get_intersections(self, region):
        """ Returns all types of intersections. self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction """

        self_intersection, self_intersection_fraction = double_intersection(
            self.circle_list
        )

        (
            region_intersection,
            region_nonintersection,
            region_intersection_fraction,
            region_nonintersection_fraction,
        ) = intersection_region(region, self.circle_list)

        return (
            self_intersection,
            self_intersection_fraction,
            region_intersection,
            region_nonintersection,
            region_intersection_fraction,
            region_nonintersection_fraction,
        )

    def remove_irrelavent_circles(self, region, threshold_region, threshold_self):
        """ Removes all circles in circle_list that intrsect the region less than threshold returns circles that were removed """
        original_circle_list = self.circle_list

        kept_circles, removed_circles = [], []
        for circle in self.circle_list:
            m = get_m()

            try:
                frac = circle.intersection(region).area() / circle.area()
            except AssertionError:  # Wrapper in case for spherical geometry failure
                intersect = proj_intersection(region, circle)
                frac = intersect.area() / circle.area()

            if frac < threshold_region:
                removed_circles.append(circle)
            else:
                kept_circles.append(circle)

        self.circle_list = kept_circles

        kept_circles, removed_circles = [], []
        for circle in self.circle_list:
            # Removes circle in list copy so we can check how much it intersects other circles
            rem_circle_list = self.circle_list[:]
            rem_circle_list = [circle for circle in rem_circle_list if circle not in removed_circles]
            rem_circle_list.remove(circle)

            double_intersection_lst, _ = double_intersection(rem_circle_list)
            if _ == False:
                continue
            double_intersection_union = SphericalPolygon.multi_union(double_intersection_lst)
            frac = (np.abs(double_intersection_union.intersection(circle).area() - circle.area()) / circle.area())

            if frac < threshold_self:
                removed_circles.append(circle)
            else:
                kept_circles.append(circle)

        self.circle_list = kept_circles

        return [circle for circle in original_circle_list if circle not in self.circle_list]

    def plot_agent(self, region, m, zorder=1, fill=True):

        color1, color2, color3 = colors[1], colors[2], colors[3]

        # makes sure everything is nice and updated
        # self.update_agent()

        if fill:
            (
                self_intersection,
                _,
                region_intersection,
                region_nonintersection,
                _,
                _,
            ) = self.get_intersections(region)
            for poly in self_intersection:  # Filling in the actual circles
                draw_screen_poly(poly, m, color=color1, zorder=zorder + 0.5)

            for poly in region_intersection:  # Filling in the actual circles
                draw_screen_poly(poly, m, color=color2, zorder=zorder)

            for poly in region_nonintersection:  # Filling in the actual circles
                draw_screen_poly(poly, m, color=color3, zorder=zorder)
        else:
            for poly in self.circle_list:
                poly.draw(m, c="r")

    def move_circle(self, old_circle, delta_lon, delta_lat):
        """ Moves circle from circle_list to new center """

        old_center = get_center(old_circle)

        try:
            self.circle_list.remove(old_circle)
        except:
            raise IndexError("The circle entered was not found in this agent")

        new_circle = get_circle(
            old_center[0] + delta_lon, old_center[1] + delta_lat, self.fov
        )

        self.circle_list.append(new_circle)

    def update_centers(self):
        """ sets a list of centers given polyogn centers """

        self.center_list = [get_center(shape) for shape in self.circle_list]

    def update_voronoi(self):
        """ Get spherical voronoi diagrams given circles """

        self.update_centers()  # makes sure there centers are accurate
        lon, lat = zip(*self.center_list)
        X, Y, Z = lon_lat_to_xyz(lon, lat, 1)
        points = np.array(list(zip(X, Y, Z)))
        self.spher_vor = SphericalVoronoi(points)
        self.spher_vor.sort_vertices_of_regions()

        self.voronoi_list = []
        for region in self.spher_vor.regions:
            vert = self.spher_vor.vertices[
                region
            ]  # Gets the verticies for a particular voronoi region
            self.voronoi_list.append(SphericalPolygon(vert))

    def update_voronoi_ll(self):
        """ Get projected voronoi diagrams """

        self.update_centers()

        boundary = np.array(
            [
                (0, 0),
                (360, 0),
                (360, 180),
                (0, 180),
            ]
        )

        x, y = boundary.T
        diameter = np.linalg.norm(boundary.ptp(axis=0))

        self.voronoi_list_ll = []
        boundary_polygon = geometry.Polygon(boundary)
        for p in voronoi_polygons(Voronoi(self.center_list), diameter):
            self.voronoi_list_ll.append(p.intersection(boundary_polygon))

    def plot_voronoi_ll(self, **plot_args):
        """ Ploted lon lat coord voronoi diagrams """

        self.update_voronoi_ll()
        print(self.voronoi_list_ll)

        for poly in self.voronoi_list_ll:
            print(poly)
            plt.fill(*poly.exterior.xy, **plot_args)

    def plot_voronoi(self, ax, basemap=False, **plot_args):
        """ Plots projection of lonlat coordinates"""

        self.update_voronoi()

        if basemap:
            for poly in self.voronoi_list:
                poly.draw(m)

        else:
            lon, lat = zip(*self.center_list)
            X, Y, Z = lon_lat_to_xyz(lon, lat, 1)
            points = np.array(list(zip(X, Y, Z)))
            ax.scatter(points[..., 0], points[..., 1], points[..., 2])

            for region in self.spher_vor.regions:
                polygon = Poly3DCollection([self.spher_vor.vertices[region]], **plot_args)
                polygon.set_color(np.random.rand(3,))
                ax.add_collection3d(polygon)

    def plot_centers(self, m, zorder=2):
        """ Plots the centers of the circles in black """
        self.update_centers()

        for center in self.center_list:
            x, y = m(center[0], center[1])
            plt.scatter(x, y, c="k", zorder=zorder, s=2)


def proj_intersection_area_inv(center_array, region, fov):
    """ Returns inverse of area itnersection with region but projectiosn to lon lat space """

    real_centers = grouper(2, center_array)
    polygon_list = [get_flat_circle(fov / 2, center) for center in real_centers]
    proj_region = spherical_poly_to_poly(region)
    proj_region = proj_region.buffer(0)

    r = proj_region.intersection(unary_union(polygon_list)).area
    # We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse
    s = 3
    soft_inv = 1 / ((1 + (r ** s)) ** (1 / s))

    return soft_inv


def repair_agent_BFGS(
    agent, region, plot=False, debug=False, generation=0, agent_number=0
):
    """ Given agent uses quasi newton secant update to rearrange circles in agent to cover the region """

    try:
        if (region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > 0.98):  # Check if we even need to update
            return True

    except AssertionError:
        spher_union = SphericalPolygon.multi_union(agent.circle_list)
        intersect = proj_intersection(region, spher_union)

        if intersect.area() / region.area() > 0.98:
            return True

    except ValueError:
        spher_union = SphericalPolygon.multi_union(agent.circle_list)
        intersect = proj_intersection(region, spher_union)

        if intersect.area() / region.area() > 0.98:
            return True

    agent.update_centers()

    # Guess is just the current circle list
    tupled = [(c[0], c[1]) for c in agent.center_list]
    guess = [item for sublist in tupled for item in sublist]

    optimized = optimize.minimize(proj_intersection_area_inv, guess, args=(region, agent.fov), method="L-BFGS-B")

    tupled_guess = grouper(2, guess)
    tupled_optimized = grouper(2, optimized.x)

    # Reassigns circle list
    agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_optimized]
    agent.update_centers()
    #agent.remove_irrelavent_circles(region, 0.05, 0.05)

    if debug:
        print("Optimization was {}".format(optimized.success))

    if plot:
        os.mkdir("repair_frames/generation_{}/agent_{}".format(generation, agent_number))
        # Plotting guess
        m = get_m()
        agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_guess]
        agent.plot_agent(region, m, fill=False)
        region.draw(m)
        agent.plot_centers(m, 2)
        plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, "guess"))
        plt.close()

        # Plotting actual
        m = get_m()
        agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_optimized]
        agent.plot_agent(region, m, fill=False)
        region.draw(m)
        agent.plot_centers(m, 2)
        plt.savefig("repair_frames/generation_{}/agent_{}/frame_{}".format(generation, agent_number, "BFGS optimized"))
        plt.close()

    try:
        if (region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > 0.98):  # Check if we even need to update
            return True
        else:
            return False
    except AssertionError:
        spher_union = SphericalPolygon.multi_union(agent.circle_list)
        intersect = proj_intersection(region, spher_union)

        if intersect.area() / region.area() > 0.98:
            return True
        else:
            return False

# GA part
def repair_agents(agent_list, region, plot=False, generation=0, guess=False):
    """ Given a list of agents returns a list of repaired agents """
    if plot == True:
        if generation == 0:
            # Clearing folder before we add new frames
            folder = "{}/repair_frames/".format(cwd)
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))
        os.mkdir("{}/repair_frames/generation_{}".format(cwd, generation))
    repaired_agent_list = []
    for i, agent in enumerate(agent_list):
        printProgressBar(i, len(agent_list))
        if repair_agent_BFGS(agent, region, plot=plot, generation=generation, agent_number=i, debug=False):
            repaired_agent_list.append(agent)

    return repaired_agent_list


def init_agents(fov, region, population, length=20):
    return [Agent(fov=fov, length=length, region=region) for _ in range(population)]


def fitness(agent_list, region, initial_length):

    alpha = 2.1
    beta = 1
    chi = 1
    sm = alpha + beta + chi

    for agent in agent_list:

        _, _, frac_overlap, frac_nonoverlap = intersection_region(
            region, agent.circle_list
        )
        _, frac_self_intersection = double_intersection(agent.circle_list)

        agent.fitness = (
            sm
            - (alpha * (agent.length / initial_length))
            - (beta * frac_nonoverlap)
            - (chi * frac_self_intersection)
        )

    return agent_list


def selection(agent_list):

    agent_list = sorted(agent_list, key=lambda agent: agent.fitness, reverse=True)
    # DARWINISM HAHHAA
    agent_list = agent_list[: int(0.8 * len(agent_list))]

    return agent_list


def crossover(agent_list, region, plot=False, generation=0, scheme='ll'):
    """ Crossover is determined by mixing voronoi diagrams """

    offspring = []
    for i in range(round(len(agent_list) / 2)):

        # Randomly select parents from agent list, the same parent can (and probably will) be selected more than once at least once
        parent1 = random.choice(agent_list)
        parent2 = random.choice(agent_list)

        if scheme == 'll':
            child1, child2 = breed_agents_ll(parent1, parent2)
        else:
            child1, child2 = breed_agents_spher(parent1, parent2)

        offspring.append(child1)
        offspring.append(child2)

        if plot:  # Currently not implemented
            raise NotImplementedError("Not implemented just yet")
            os.mkdir("{}/crossover_frames/generation_{}/{}".format(cwd, generation, i))

    agent_list.extend(offspring)

    return agent_list


def mutation(agent_list, region):

    for agent in agent_list:

        if random.uniform(0, 1) <= 0.2:
            try:
                worst_circle_self = sorted(
                    agent.circle_list,
                    key=lambda x: SphericalPolygon.multi_union(
                        removal_copy(agent.circle_list, x)
                    )
                    .intersection(x)
                    .area(),
                )[
                    0
                ]  # Finds circle which intersects with itself the most
            except AssertionError:
                worst_circle_self = sorted(
                    agent.circle_list,
                    key=lambda x: proj_intersection(
                        SphericalPolygon.multi_union(
                            removal_copy(agent.circle_list, x)
                        ),
                        x,
                    ).area(),
                )[
                    0
                ]  # Finds circle which intersects with itself the most

            agent.circle_list.remove(worst_circle_self)

        if random.uniform(0, 1) <= 0.3:
            # Finds circle which intersects region the least
            #Can't just use worst_circle_region = sorted(agent.circle_list, key=lambda x: region.intersection(x).area())[-1] since spherical_geometry package doesn't like it... I think they are working on this
            area_list = []
            for circ in agent.circle_list:
                #We need to try catch because spherical_geometry bug
                try:
                    area_list.append(region.intersection(circ))
                except AssertionError:
                    area_list.append(1000)

            worst_circle_region = [x for _, x in sorted(zip(area_list, agent.circle_list), key=lambda pair: pair[0])][-1]
            
            agent.circle_list.remove(worst_circle_region)

        if random.uniform(0, 1) <= 0.3:

            circle_to_move = random.choice(
                agent.circle_list
            )  # Chooses a random circle to move
            delta_x = random.uniform(-0.1, 0.1)  # How much to move it
            delta_y = random.uniform(-0.1, 0.1)
            agent.move_circle(circle_to_move, delta_x, delta_y)

    return agent_list


def ga(
    region,
    fov,
    population,
    generations,
    initial_length=100,
    plot_regions=False,
    crossover_scheme=None,
    plot_crossover=False,
    save_agents=False,
    plotting_scheme='basemap',
    dataset_id=None,
):

    start = time.process_time()  # Timing entire program

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
            agent_list = repair_agents(
                agent_list, region, plot=plot_regions, generation=generation, guess=True
            )
            print(
                "Sucessful. {} Agents remain. Run time {}".format(
                    len(agent_list), time.process_time() - before
                )
            )
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

        # Creating folder
        os.mkdir("{}/frames/generation_{}".format(cwd, generation))

        for i, agent in enumerate(agent_list):
            if plotting_scheme == 'basemap':
                m = get_m()
                agent.plot_agent(region, m, zorder=2, fill=False)
                region.draw(m)
                plt.savefig("frames/generation_{}/agent_{}".format(generation, i))
                plt.close()
            elif plotting_scheme == 'ligo':
                if dataset_id == None:
                    raise Exception("If you want to use the ligo plotting you must specify the dataset and id")
                
                dataset, id = zip(*dataset_id)
                path_to_fits = f"{cwd}/data/{dataset}/{id}.fits"
                lons, lats = zip(*agent.center_list)
                lats = [i + 90 for i in lats]
                plot_ligo_style(path_to_fits, f"{cwd}/frames/generation_{generation}_agent_{i}", lons, lats)


        print("frame saved in frames/generation_{}. Run time {}".format(generation, time.process_time() - before))
        print()

        if save_agents:
            for i, agent in enumerate(agent_list):
                file_pi = open('{}/saved_agents/agent_{}.obj'.format(cwd, i), 'wb')
                pickle.dump(agent, file_pi)

        before = time.process_time()
        print("Beginning crossover")
        os.mkdir("{}/crossover_frames/generation_{}".format(cwd, generation))
        agent_list = crossover(agent_list, region, plot=plot_crossover, generation=generation, scheme=crossover_scheme)
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
            if len(agent_list) < 3:
                break
            print()

        print()
        print("Completed. Generational run time {}".format(time.process_time() - generation_start))
        print()
        print()

    print("Finished. Total execution time {}".format(time.process_time() - start))

    return agent_list


dataset = "design_bns_astro"  # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8  # FOV diameter in degrees

# Open sample file, tested on 100
id = 18

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

coords = get_concave_hull(dataset, id)
lon, lat = zip(*coords[0])

lonlat_poly = geometry.Polygon(list(zip(lon, lat)))
inside_pt = generate_random_in_polygon(1, lonlat_poly)[0]

region = SphericalPolygon.from_lonlat(lon, lat, center=inside_pt)


# Clearing folder before we add new frames
if not os.path.exists("{}/frames".format(cwd)):
    os.mkdir("frames")
if not os.path.exists("{}/repair_frames".format(cwd)):
    os.mkdir("repair_frames")
if not os.path.exists("{}/saved_agents".format(cwd)):
    os.mkdir("saved_agents")
if not os.path.exists("{}/crossover_frames".format(cwd)):
    os.mkdir("crossover_frames")

folders_to_clear = [
    "{}/frames".format(cwd),
    "{}/repair_frames".format(cwd),
    "{}/crossover_frames".format(cwd),
]
for folder in folders_to_clear:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

population = 5
generations = 5
final_agent_list = ga(
    region,
    fov_diameter,
    population,
    generations,
    crossover_scheme='ll',
    initial_length=8,
    plot_regions=True,
    plot_crossover=False,
    save_agents=True,
    plotting_scheme='ligo',
    dataset_id=(dataset, id)
)