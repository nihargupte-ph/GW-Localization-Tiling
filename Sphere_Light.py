import healpy as hp
import numpy as np
import os
from shapely import geometry
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from shapely.ops import unary_union
from spherical_geometry.polygon import SphericalPolygon
from misc_functions import *
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


def proj_intersection(spher_poly1, spher_poly2):
    """ The spherical geometry module currently has a bug where it will not correctly find the intersection between polygons sometimes. See https://github.com/spacetelescope/spherical_geometry/issues/168. This is a function which projects to 2D (not ideal I know) and returns a new polygon which is the intersection """

    poly1 = spherical_poly_to_poly(spher_poly1)
    poly2 = spherical_poly_to_poly(spher_poly2)
    poly1 = poly1.buffer(0)

    intersec = poly1.intersection(poly2)

    ret = poly_to_spherical_poly(intersec)

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


# def proj_intersection_area_inv(center_array, region, fov):
#     """ Returns inverse of area itnersection with region but projectiosn to lon lat space """

#     real_centers = grouper(2, center_array)
#     polygon_list = [get_circle(center[0], center[1], fov) for center in real_centers]

#     projection = ccrs.AlbersEqualArea().proj4_init

#     proj_circles = [proj_poly(poly, proj=projection) for poly in polygon_list]
#     proj_region = proj_poly(region, proj=projection)



#     r = proj_region.intersection(unary_union(proj_region)).area
#     # We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse
#     s = 3
#     soft_inv = 1 / ((1 + (r ** s)) ** (1 / s))

#     return soft_inv



def repair_agent_BFGS(agent, region, debug=False):
    """ Given agent uses quasi newton secant update to rearrange circles in agent to cover the region """

    #NOTE Insert check if we need to update here

    agent.update_centers()

    # Guess is just the current circle list
    tupled = [(c[0], c[1]) for c in agent.center_list]
    guess = [item for sublist in tupled for item in sublist]

    optimized = optimize.minimize(proj_intersection_area_inv, guess, args=(region, agent.fov), method="BFGS")
    tupled_guess = grouper(2, guess)
    tupled_optimized = grouper(2, optimized.x)

    # Reassigns circle list
    agent.circle_list = [get_circle(center[0], center[1], agent.fov) for center in tupled_optimized]
    union = spherical_unary_union(agent.circle_list)
    intersec = region.intersection(union)

    if debug:
        print("Optimization was {}".format(optimized.success))


    print(region.area(), intersec.area())

    if region.area() - intersec.area() < .001:
        return True
    else:
        return False

dataset = "design_bns_astro"  # name of dataset ('design_bns_astro' or 'design_bbh_astro')
id = 18
fov = 8  # FOV diameter in degrees

# Open sample file, tested on 100
coords = get_concave_hull(dataset, id)

lon, lat = zip(*coords[0])

lonlat_poly = geometry.Polygon(list(zip(lon, lat)))
inside_pt = generate_random_in_polygon(1, lonlat_poly)[0]

region = SphericalPolygon.from_lonlat(lon, lat, center=inside_pt)


intial_guess = 15
max_circles = 100
tmp = []
for length in range(intial_guess, max_circles):
    agent = Agent(fov=fov, length=length, region=region)

    success = repair_agent_BFGS(agent, region)

    m = get_m()
    for circle in agent.circle_list:
        circle.draw(m, c='b', linewidth=1)
    region.draw(m, c='r', linewidth=1)
    plt.savefig(f"{cwd}/repair_frames/{length}.png")
    plt.close()

    if success:
        successful_agent = agent
        break


path_to_fits = f"{cwd}/data/{dataset}/{id}.fits"
lons, lats = zip(*successful_agent.center_list)
lats = [i + 90 for i in lats]
plot_ligo_style(path_to_fits, f"{cwd}/frames/{id}", lons, lats)
