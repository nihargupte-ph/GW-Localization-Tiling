#Import
import healpy as hp
import numpy as np
import math
from shapely import geometry
from ligo.skymap.io import fits
from spherical_geometry.polygon import SphericalPolygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from misc_functions import *
from matplotlib.patches import Polygon

random.seed(0)

def draw_screen_poly(poly, m, **plot_args):
    lons, lats = list(poly.to_radec())[0]
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

def get_circle(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """
    radius = math.tan(fov/2)
    ret = SphericalPolygon.from_cone(phi, theta, radius, steps=step, degrees=False)
    ret = ret.invert_polygon()
    return ret

def spherical_poly_to_poly(poly):
    _X, _Y, _Z = zip(*list(poly.points)[0])
    lon, lat, rho = xyz_to_lon_lat(_X, _Y, _Z)
    lon_lat = zip(lon, lat)
    poly = geometry.Polygon(lon_lat)
    return poly

def double_intersection(polygon_list):

    """ Returns intersection between polygons in polygon_list and the area of their intersection """

    interesections = []
    union_of_polygons = SphericalPolygon.multi_intersection(polygon_list)
    for poly in polygon_list:
        try:
            intersect = union_of_polygons.intersection(poly)
            interesections.append(intersect)
        except AssertionError: #NOTE don't really know why this is here
            pass

    
    intersection = SphericalPolygon.multi_intersection(interesections)
    intersection_area = intersection.area

    return intersection, intersection_area

def intersection_region(region, polygon_list):

    """ Returns regions of intersection between the polygon_list and the region. Also returns the non intersection between polygon_list and the region. It will also return the fraction which the polygon list has covered """

    polygon_union = SphericalPolygon.multi_intersection(polygon_list)
    print(polygon_union)
    intersection = region.intersection(polygon_union)
    fraction_overlap = intersection.area() / region.area()
    outside = region.invert_polygon()
    nonoverlap = polygon_union.intersection(outside)
    fraction_nonoverlap = nonoverlap.area() / (4 * math.pi)
    return intersection, nonoverlap, fraction_overlap, fraction_nonoverlap



class Agent:

    def __init__(self, fov=None, length=None, region=None):
        """ Agent object"""

        self.fitness = -1000  # Dummy value
        self.fov = fov
        self.length = length

        if region != None:
            """ Generates random circles inside the region for an inital guess """
            phi_theta = list(zip(list(region.to_radec())[0][0], list(region.to_radec())[0][1]))
            phi_theta_polygon = geometry.Polygon(phi_theta)
            tupled =  generate_random_in_polygon(self.length, phi_theta_polygon)
            self.circle_list = [get_circle(np.radians(phi), np.radians(theta), self.fov) for phi, theta in tupled]

    def update_agent(self):
        self.length = len(self.circle_list)

    def get_intersections(self, region):
        """ Returns all types of intersections. self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction """

        self_intersection, self_intersection_fraction = double_intersection(
            self.circle_list)

        region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction = intersection_region(
            region, self.circle_list)

        return self_intersection, self_intersection_fraction, region_intersection, region_nonintersection, region_intersection_fraction, region_nonintersection_fraction

    def plot_agent(self, region, m, zorder=1, fill=True):
        
        color1, color2, color3 = colors[1], colors[2], colors[3]

        #makes sure everything is nice and updated
        self.update_agent()

        self_intersection, _, region_intersection, region_nonintersection, _, _ = self.get_intersections(region)

        if fill:
            # for circle in agent.circle_list: #Filling in the actual circles
            #     lons, lats = list(circle.to_radec())[0]
            #     draw_screen_poly(lons, lats, m, color=color2, zorder=zorder)

            for poly in self_intersection: #Filling in the actual circles

                draw_screen_poly(poly, m, color=color1, zorder=zorder)

            try:
                for poly in region_intersection: #Filling in the actual circles
                    draw_screen_poly(poly, m, color=color2, zorder=zorder)
            except:
                print("no region intersection")

            try:
                for poly in region_nonintersection: #Filling in the actual circles
                    draw_screen_poly(poly, m, color=color3, zorder=zorder)
            except:
                print("no region non intersection")

dataset = 'design_bns_astro' # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8 # FOV diameter in degrees

# Convert FOV to radius and radians
fov_diameter = np.deg2rad(fov_diameter)

#Open sample file, tested on 100
i = 232

global colors
colors = ["#ade6e6", "#ade6ad", "#e6ade6", "#e6adad"]

X, Y, Z = convert_fits_xyz(dataset, i)
inside_point = X[1], Y[1], Z[1] #It's probably inside ?? NOTE should be cahnged though
#We need to cluster the points before we convex hull
region = SphericalPolygon.convex_hull(list(zip(X,Y,Z)))
region = region.invert_polygon()

agent = Agent(fov=fov_diameter, length=8, region=region)

m = Basemap(projection='moll',lon_0=30,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')

# tmp_x = [-70, 0, 0, -70]
# tmp_y = [-20, -20, 20, 20]
# draw_screen_poly(tmp_x, tmp_y, m)

region.draw(m)
agent.plot_agent(region, m)

plt.show()

# m, metadata = fits.read_sky_map('data/' + dataset + '/' + str(i) + '.fits', nest=None)
# region90 = area(m,fov_radius,dil=0.99,deg_fact=8)

# n = np.zeros(len(m))
# n[region90.area_pix] = 0.00005

# print(max(n))