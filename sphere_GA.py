#Import
import healpy as hp
import numpy as np
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

random.seed(1)
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

def xyz_to_lon_lat(X, Y, Z):
    """ Takes list of X, Y, and Z coordinates and spits out list of lon lat and rho """

    theta = [math.degrees(math.atan(x/y)) for x, y in zip(X,Y)]
    phi = [math.degrees(math.acos(z / math.sqrt((x**2)+(y**2)+(z**2)))) for x, y, z in zip(X,Y,Z)]
    rho = [x**2 + y**2 + z**2 for x, y, z in zip(X,Y,Z)]

    return theta, phi, rho

# def get_circle(phi, theta, fov, step=16):
#     """ Returns SphericalPolygon given FOV and center of the polygon """
#     radius = math.tan(fov/2)
#     ret = SphericalPolygon.from_cone(phi, theta, radius, steps=step, degrees=True)
#     return ret

def get_circle_redraw(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """
    radius = math.tan(fov/2)
    lons = [phi + radius * math.sin(math.radians(angle)) for angle in np.linspace(0, 2 * math.pi, step)]
    lats = [theta + radius * math.cos(math.radians(angle)) for angle in np.linspace(0, 2 * math.pi, step)]
    ret = SphericalPolygon.from_radec(lons, lats)
    return ret

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

def powerful_union_area(polygon_list):
    """ I know spherical geometry has this but it doesn't work... Given a polygonlist returns the area of the union """
    big_poly = polygon_list[0]
    extra_area = 0
    for i, poly in enumerate(polygon_list):
        if i == 1:
            continue

        if math.isnan(big_poly.union(poly).area()):
            extra_area += poly.area()
            continue

        big_poly = big_poly.union(poly)

    return big_poly.area() + extra_area

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
            self.circle_list = [get_circle_redraw(i[0], i[1], self.fov) for i in tupled]
            self.update_centers()
            self.remove_irrelavent_circles(region, .05, .05)

    def update_agent(self):
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

    def plot_centers(self, m, zorder):
        """ Plots the centers of the circles in black """
        self.update_centers()

        for center in self.center_list:
            x, y = m(center[0], center[1])
            plt.scatter(x, y, c='k', zorder=zorder, s=2)

def intersection_area_inv(center_array, region, fov):
    """ Returns inverse of area intersection with region """

    real_centers = grouper(2, center_array)
    polygon_list = [get_circle_redraw(center[0], center[1], fov) for center in real_centers]

    interior_intersections = [region.intersection(polygon) for polygon in polygon_list]
    interior_area = powerful_union_area(interior_intersections)
    
    r = interior_area

    #We don't want hard inverse because dividing by 0 will error out, so we use a soft inverse   
    s=3
    soft_inv = 1 / ((1 + (r**s)) ** (1/s))

    return soft_inv

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
    agent.circle_list = [get_circle_redraw(center[0], center[1], agent.fov) for center in tupled_optimized]


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

    if region.intersection(SphericalPolygon.multi_union(agent.circle_list)).area() / region.area() > .98: #Precision errors
        return True
    else:
        return False

dataset = 'design_bns_astro' # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8 # FOV diameter in degrees

#Open sample file, tested on 100
i = 232

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

global m
#m = Basemap(projection='ortho',lon_0=10,lat_0=40,resolution='c')
m = Basemap(projection='moll', lon_0=-70, resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')

X, Y, Z = convert_fits_xyz(dataset, i)
inside_point = X[1], Y[1], Z[1] #It's probably inside ?? NOTE should be cahnged though
#We need to cluster the points before we convex hull
region = SphericalPolygon.convex_hull(list(zip(X,Y,Z)))
region.draw(m)

agent = Agent(fov=fov_diameter, length=9, region=region)
success = repair_agent_BFGS(agent, region)

agent.plot_agent(region, m, fill=True, zorder=4)
agent.plot_centers(m, 2)


# tmp_x = [300, 360, 360, 300]
# tmp_y = [-10, -10, 10, 10]
# poly1 = SphericalPolygon.from_radec(tmp_x, tmp_y)
# intersection = region.intersection(poly1)
# exterior = region.invert_polygon().intersection(poly1)
# exterior.draw(m, color='k')
# #intersection.draw(m, color='r')
# draw_screen_poly(poly1, m)
plt.show()

#agent.plot_agent(region, m)


# m, metadata = fits.read_sky_map('data/' + dataset + '/' + str(i) + '.fits', nest=None)
# region90 = area(m,fov_radius,dil=0.99,deg_fact=8)

# n = np.zeros(len(m))
# n[region90.area_pix] = 0.00005

# print(max(n))