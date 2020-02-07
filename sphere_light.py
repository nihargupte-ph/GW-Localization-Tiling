import healpy as hp
import numpy as np
import math
from shapely import geometry
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from shapely.ops import unary_union
from spherical_geometry.polygon import SphericalPolygon
from misc_functions import *
from LIGO_Plotting import *

def get_circle(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """

    radius = fov/2
    lons = [phi + radius * math.cos(angle) for angle in np.linspace(0, 2 * math.pi, step)]
    lats = [theta + radius * math.sin(angle) for angle in np.linspace(0, 2 * math.pi, step)]
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

def get_center(circle):
    
    """ Given circle returns ra dec of center of circle by projection """ #NOTE needs fixing 360 --> 0

    radec = list(circle.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    poly = geometry.Polygon(list(zip(lons, lats)))
    center = poly.centroid
    return (center.x , center.y)

def double_intersection(polygon_list):

    """ Returns intersection between polygons in polygon_list and the area of their intersection. Perhaps upgrade to cascaded_union in future if program is taking too long this would be a major speed up """
   
    intersections, already_checked = [], []
    for polygon in polygon_list:
        already_checked.append(polygon) #We don't need to check the polygons we already intersected with
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

    try:
        interior_intersections = [region.intersection(polygon) for polygon in polygon_list]
    except AssertionError:
        interior_intersections = [proj_intersection(region, polygon) for polygon in polygon_list]
    interior_area = powerful_union_area(interior_intersections)
    interior_fraction = interior_area / region.area()

    try: 
        exterior_intersections = [polygon.intersection(outside) for polygon in polygon_list]
    except AssertionError:
        exterior_intersections = [proj_intersection(outside, polygon) for polygon in polygon_list]
    exterior_area = powerful_union_area(exterior_intersections)
    exterior_fraction = exterior_area / (4 * math.pi)

    
    return interior_intersections, exterior_intersections, interior_fraction, exterior_fraction

def get_flat_circle(radius, center, step=100):
    """ Returns shapely polygon given radius and center. This is the same as get_circle in Flat_GA. Step is the number of verticies in the polygon, 
    the higher the steps the closer it is to a circle """

    
    point_list = [geometry.Point(radius * np.cos(theta) + center[0], radius * np.sin(
        theta) + center[1]) for theta in np.linspace(0, 2 * np.pi, step)]
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
            tupled =  generate_random_in_polygon(self.length, spherical_poly_to_poly(region))
            self.circle_list = [get_circle(i[0], i[1], self.fov) for i in tupled]
            self.update_centers()
            self.remove_irrelavent_circles(region, .05, .05)

    def update_agent(self):
        self.remove_irrelavent_circles(region, .05, .05)
        self.length = len(self.circle_list)

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

            try:
                frac = circle.intersection(region).area() / circle.area()
            except AssertionError: #Wrapper in case for spherical geometry failure
                intersect = proj_intersection(region, circle)
                frac = intersect.area() / circle.area()

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

    def update_centers(self):
        """ sets a list of centers given polyogn centers """

        self.center_list = [get_center(shape) for shape in self.circle_list]

def proj_intersection_area_inv(center_array, region, fov):
    """ Returns inverse of area itnersection with region but projectiosn to lon lat space """

    real_centers = grouper(2, center_array)
    polygon_list = [get_flat_circle(fov/2, center) for center in real_centers]
    proj_region = spherical_poly_to_poly(region)
    proj_region = proj_region.buffer(0)

    r = proj_region.intersection(unary_union(polygon_list)).area
    #We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse   
    s=3
    soft_inv = 1 / ((1 + (r**s)) ** (1/s))

    return soft_inv

def repair_agent_BFGS(agent, region, debug=False):
    """ Given agent uses quasi newton secant update to rearrange circles in agent to cover the region """

    try: 
        if region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > .98: #Check if we even need to update
            return True
    except AssertionError:
        spher_union = SphericalPolygon.multi_union(agent.circle_list)
        intersect = proj_intersection(region, spher_union)

        if intersect.area() / region.area() > .98:
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
    agent.remove_irrelavent_circles(region, .03, .03)


    if debug:
        print("Optimization was {}".format(optimized.success))

    try: 
        if region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > .98: #Check if we even need to update
            return True

    except AssertionError:
        spher_union = SphericalPolygon.multi_union(agent.circle_list)
        intersect = proj_intersection(region, spher_union)

        if intersect.area() / region.area() > .98:
            return True

dataset = 'design_bns_astro' # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8 # FOV diameter in degrees

#Open sample file, tested on 100
i = 130
fov = 8

#We need to cluster the points before we convex hull
# X, Y, Z = convert_fits_xyz(dataset, i)
# lons, lats, r = xyz_to_lon_lat(X, Y, Z)
# lons = [lon - 180 for lon in lons]
# lats = [lat - 90 for lat in lats]
ras, decs = convert_fits_xyz(dataset, i)
points = np.asarray([(ra, dec) for ra, dec in zip(ras, decs)])

hull = ConvexHull(points)
hull_pts = points[hull.vertices, :]
hull_ra, hull_dec = zip(*hull_pts)
hull_lon, hull_lat = np.asarray(hull_ra) - 180, 90 - np.asarray(hull_dec)
region = SphericalPolygon.from_radec(hull_lon, hull_lat)

intial_guess = 6
for length in range(intial_guess, 0, -1):
    agent = Agent(fov=fov, length=length, region=region)

    success = repair_agent_BFGS(agent, region)

    #Temporarry NOTE
    m=get_m()
    radec = list(region.to_radec())
    lons, lats = radec[0][0], radec[0][1]
    x, y = m(lons, lats)
    m.plot(x,y, c='b')

    for circle in agent.circle_list:
        radec = list(circle.to_radec())
        lons, lats = radec[0][0], radec[0][1]
        x, y = m(lons, lats)
        m.plot(x,y, c='r')
    plt.savefig('LIGO_Plots/fits_number_{}_length_{}_basemap'.format(i, length))
    plt.close()

    ligo_centers = [(a, 90 - b) for a, b in agent.center_list]
    plot_ligo('data/{}/{}.fits'.format(dataset, i), np.asarray(ligo_centers), "LIGO_Plots/fits_number_{}_length_{}".format(i, length))
    plt.close()