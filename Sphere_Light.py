import healpy as hp
import numpy as np
import pickle
import os, shutil
from shapely import geometry
from functools import partial
import pyproj
import scipy.optimize as optimize
import cartopy.crs as ccrs
from scipy.spatial import ConvexHull
from shapely.ops import unary_union, transform
from spherical_geometry.polygon import SphericalPolygon
import cartopy.crs as ccrs
from misc_functions import *
from rtree import index
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")  # If you want to debug remove thi
global cwd
cwd = os.getcwd()



def get_m(**plot_args):
    """ Given plot args returns a basemap "axis" with the proper plot args. Edit this function if you want different maps """

    #m = Basemap(projection="ortho", resolution="c", lon_0=-20, lat_0=0, **plot_args)
    m = Basemap(projection="moll", resolution="c", lon_0=0)
    m.drawcoastlines()
    return m


def get_circle(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """

    radius = fov / 2
    lons = [phi + radius * np.cos(angle) for angle in np.linspace(0, 2 * np.pi, step)]
    lats = [theta + radius * np.sin(angle) for angle in np.linspace(0, 2 * np.pi, step)]
    ret = SphericalPolygon.from_radec(lons, lats)
    return ret

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

def get_center(circle):
    """ Given circle returns ra dec of center of circle by projection """  # NOTE needs fixing 360 --> 0

    radec = list(circle.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    poly = geometry.Polygon(list(zip(lons, lats)))
    center = poly.centroid
    return (center.x, center.y)


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


def proj_double_intersection(polygon_list):
    """ Returns intersection between polygons in polygon_list and the area of their intersection. Perhaps upgrade to cascaded_union in future if program is taking too long this would be a major speed up """

    proj_polygon_list = [proj_poly(poly, proj=ccrs.AzimuthalEquidistant().proj4_init) for poly in polygon_list]

    intersections = []
    idx = index.Index()
    # Populate R-tree index with bounds of grid cells
    for pos, cell in enumerate(proj_polygon_list):
        # assuming cell is a shapely object
        idx.insert(pos, cell.bounds)

    for poly in proj_polygon_list:
        merged_circles = unary_union(
            [
                proj_polygon_list[pos]
                for pos in idx.intersection(poly.bounds)
                if proj_polygon_list[pos] != poly
            ]
        )
        intersec = poly.intersection(merged_circles)

        if intersec.is_empty:
            continue

        if isinstance(
            intersec, geometry.GeometryCollection
        ):  # For some reason linestrings are getting appended so i'm removing them
            new_intersec = geometry.GeometryCollection(
                [
                    layer_precision(shape).buffer(0)
                    for shape in intersec
                    if not isinstance(shape, geometry.LineString)
                ]
            )
            intersections.append(new_intersec)
        elif isinstance(intersec, geometry.MultiPolygon):
            new_intersec = unary_union(
                [layer_precision(poly).buffer(0) for poly in list(intersec)]
            )
            intersections.append(new_intersec)
        else:
            intersections.append(layer_precision(intersec).buffer(0))

    intersection = unary_union(intersections)
    intersection_area = intersection.area
    total_area = unary_union(proj_polygon_list).area
    frac_intersection = intersection_area / total_area

    intersection = [inv_proj_poly(poly, init_crs=ccrs.AzimuthalEquidistant().proj4_init) for poly in intersection]

    return intersection, frac_intersection

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
        self.length = len(self.circle_list)

    def update_centers(self):
        """ sets a list of centers given polyogn centers """

        self.center_list = [get_center(shape) for shape in self.circle_list]

    def remove_irrelavent_circles(self, region, threshold_region, threshold_self):
        """ Removes all circles in circle_list that intrsect the region less than threshold returns circles that were removed """
        original_circle_list = self.circle_list

        proj_region = proj_poly(region, proj=ccrs.AzimuthalEquidistant().proj4_init)
        proj_region = proj_region.buffer(0)
        kept_circles, removed_circles = [], []
        for circle in self.circle_list:

            proj_circle = proj_poly(circle, proj=ccrs.AzimuthalEquidistant().proj4_init)
            proj_circle = proj_circle.buffer(0)
            proj_intersect = proj_region.intersection(proj_circle)
            
            frac = proj_intersect.area / proj_circle.area

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

            #The circle we are interested in
            rem_circle_list.remove(circle)

            #Getting projected union of all polygons except the one we're interested in
            proj_circle_list = [proj_poly(poly, proj=ccrs.AzimuthalEquidistant().proj4_init) for poly in rem_circle_list]
            proj_circle_union = unary_union(proj_circle_list)

            #Projected circle we are interested in
            proj_circle = proj_poly(circle, proj=ccrs.AzimuthalEquidistant().proj4_init)

            #Intersection between proj union poly and proj circle
            proj_intersect = proj_circle_union.intersection(proj_circle)

            frac = (proj_circle.area - proj_intersect.area) / proj_circle.area

            if frac < threshold_self:
                removed_circles.append(circle)
            else:
                kept_circles.append(circle)

        self.circle_list = kept_circles

        return [circle for circle in original_circle_list if circle not in self.circle_list]

    def repair_agent(self, region, initial=False, max_iter=3, scheme='standard'):
        """ Repairs agent with BFGS optimization. If the agent is unable to be repaired circles the algorithm with recurse with unrepaired regions """

        if scheme == 'recursive': 
            new_region_list = [region]
            centers_guess_list = [self.center_list]
            tmp = self.circle_list
            self.circle_list = []
            for i in range(0, max_iter):
                m = get_m()

                for new_region, centers_guess in zip(new_region_list, centers_guess_list):
                    circles = repair_agent_BFGS(centers_guess, new_region, self.fov)
                    self.circle_list.extend(circles)
                for circle in self.circle_list:
                    circle.draw(m, c='y', alpha=.5)

                #Removing any additonal irrelavent circles
                #self.remove_irrelavent_circles(region, .05, .03)

                union = SphericalPolygon.multi_union(self.circle_list)
                intersec = region.intersection(union)
                if region.area() - intersec.area() < .0001:
                    return True
                else: 
                    circle_union = proj_poly(union)
                    proj_region = proj_poly(region)
                    proj_new_region = circle_union.difference(proj_region).intersection(proj_region)
                    centers_guess_list, new_region_list = [], []
                    circle_area = proj_poly(self.circle_list[0]).area
                    if isinstance(proj_new_region, geometry.Polygon):
                        proj_new_region = geometry.MultiPolygon([proj_new_region])
                    for poly in list(proj_new_region): 
                        if isinstance(poly, geometry.LineString):
                            continue
                        num_new_circles = np.ceil(poly.area / circle_area) #1 leeway circles
                        pts = generate_random_in_polygon(num_new_circles, poly)
                        if not isinstance(pts, list):
                            pts = [pts]
                        pts = inv_proj_points(pts)
                        centers_guess_list.append(pts)
                        new_region_list.append(inv_proj_poly(poly))
                        inv_proj_poly(poly).draw(m)
                plt.show()
                plt.close()
            return False
        elif scheme == 'standard':
            self.circle_list = repair_agent_BFGS(self.center_list, region, self.fov)
            projected_circles = []
            for circle in self.circle_list:
                projected_circles.append(proj_poly(circle))
            proj_union = unary_union(projected_circles)
            proj_region = proj_poly(region)
            proj_intersec = proj_region.intersection(proj_union)
            plt.close()
            plt.plot(*proj_region.exterior.xy, c='r')
            try:
                plt.plot(*proj_intersec.exterior.xy, c='g')
            except AttributeError:
                for poly in list(proj_intersec):
                    plt.plot(*poly.exterior.xy, c='g')
            try:
                plt.plot(*proj_union.exterior.xy, c='b')
            except AttributeError:

                for poly in list(proj_union):
                    plt.plot(*poly.exterior.xy, c='b')
            for poly in self.circle_list:
                plt.plot(*proj_poly(poly).exterior.xy, c='b')
            print(proj_region.area, proj_intersec.area, len(self.circle_list))
            plt.show()
            plt.close()

            if proj_region.area - proj_intersec.area < .0001:
                return True
            else:
                return False

        agent.remove_irrelavent_circles(region, .05, .03)

def repair_agent_BFGS(center_list, region, fov):
    """ Given center list uses quasi newton secant update to rearrange circles in agent to cover the region. Returns tuple of spherical polygons.  """

    # Guess is just the current circle list
    tupled = [(c[0], c[1]) for c in center_list]
    guess = [item for sublist in tupled for item in sublist]

    optimized = optimize.minimize(proj_intersection_area_inv, guess, args=(region, fov), method="BFGS")
    tupled_optimized = grouper(2, optimized.x)

    # Reassigns circle list
    ret = [get_circle(center[0], center[1], agent.fov) for center in tupled_optimized]
    return ret


def proj_intersection_area_inv(center_array, region, fov):
    """ Returns inverse of area itnersection with region but projectiosn to lon lat space """

    #proj_region = proj_poly(region)
    real_centers = grouper(2, center_array)
    polygon_list = [get_flat_circle(fov / 2, center) for center in real_centers]
    spherical_polygon_list = [SphericalPolygon.from_lonlat(*poly.exterior.coords.xy) for poly in polygon_list]
    #proj_polygon_list = [proj_poly(spher_poly) for spher_poly in spherical_polygon_list]

    #proj_region = spherical_poly_to_poly(region)
    #proj_region = proj_region.buffer(0)

    r = region.intersection(SphericalPolygon.multi_union(spherical_polygon_list)).area()
    #r = proj_region.intersection(unary_union(proj_polygon_list)).area
    # We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse
    s = 3
    soft_inv = 1 / ((1 + (r ** s)) ** (1 / s))

    return soft_inv



dataset = "design_bns_astro"  # name of dataset ('design_bns_astro' or 'design_bbh_astro')
id = 69
fov = 8  # FOV diameter in degrees

# Open sample file, tested on 100
coords_list = get_concave_hull(dataset, id, new=False)

region_list = []
for coords in coords_list:
    lon, lat = zip(*coords)

    lonlat_poly = geometry.Polygon(list(zip(lon, lat)))
    inside_pt = generate_random_in_polygon(1, lonlat_poly)[0]

    region = SphericalPolygon.from_lonlat(lon, lat, center=inside_pt)
    region_list.append(region)


folder = f"{cwd}/repair_frames/"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

area_list = [region.area() for region in region_list]
area_list_norm = [area/sum(area_list) for area in area_list]
#Pick an initial guess for the total amount of circles lower and upper bound then based on the area it will tell you the guess for each region
initial_lower, initial_upper = 30, 200
guess_range = []
for frac in area_list_norm:
    lower = int(np.floor(frac * initial_lower))
    upper = int(np.ceil(frac * initial_upper))
    if lower == 0:
        lower = 1
        upper += 3
    guess_range.append((lower, upper))

#guess_range = [(40, 45), (5, 6), (5, 6), (5, 6), (4, 5), (7, 9), (6, 9), (12, 14), (1, 2), (4, 5), (1, 2), (2, 3), (1, 2), (2, 3), (3, 4), (2, 3), (1, 2), (2, 3), (1, 2), (1, 2), (1, 2), (1, 2)]

region_circle_list_before, region_circle_list_repaired = [], []
for i, region in enumerate(region_list):
    print(f"Started region {i+1}")
    total_circle_list_before, total_circle_list_repaired = [], []
    for length in range(guess_range[i][0], guess_range[i][1]):
        agent = Agent(fov=fov, length=length, region=region)
        total_circle_list_before.append(agent.circle_list)

        success = agent.repair_agent(region, scheme='standard')
        total_circle_list_repaired.append(agent.circle_list)
        if success:
            print(f"Completed region {i+1}")
            break
    region_circle_list_before.append(total_circle_list_before)
    region_circle_list_repaired.append(total_circle_list_repaired)

for i, _ in enumerate(region_list):
    os.mkdir(f"{cwd}/repair_frames/{i}")

for i, region in enumerate(region_list):
    for total_circle_before, total_circle_after in zip(region_circle_list_before[i], region_circle_list_repaired[i]):
        m = get_m()
        plt.figure(figsize=(8,6), dpi=300)
        for circle in total_circle_before:
            circle.draw(m, c='b', linewidth=1)
        region.draw(m)
        plt.savefig(f"{cwd}/repair_frames/{i}/before.png")
        plt.close()


        m = get_m()
        plt.figure(figsize=(8,6), dpi=300)
        for circle in total_circle_after:
            circle.draw(m, c='b', linewidth=1)
        region.draw(m)
        plt.savefig(f"{cwd}/repair_frames/{i}/after.png")
        plt.close()

m = get_m()
plt.figure(figsize=(8,6), dpi=300)
for i, region in enumerate(region_list):
    for circle in region_circle_list_before[i][-1]:
        circle.draw(m, c='b', linewidth=1)
    region.draw(m, c='r')
plt.savefig(f"{cwd}/repair_frames/before.png")
plt.close()

m = get_m()
plt.figure(figsize=(8, 6), dpi=300)
for i, region in enumerate(region_list):
    for circle in region_circle_list_repaired[i][-1]:
        circle.draw(m, c='b', linewidth=1)
    region.draw(m, c='r')
plt.savefig(f"{cwd}/repair_frames/after.png")
plt.close()




path_to_fits = f"{cwd}/data/{dataset}/{id}.fits"
lons, lats = zip(*successful_agent.center_list)
lats = [i + 90 for i in lats]
plot_ligo_style(path_to_fits, f"{cwd}/frames/{id}", lons, lats)