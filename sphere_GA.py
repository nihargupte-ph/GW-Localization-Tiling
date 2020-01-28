#Import
import healpy as hp
import numpy as np
import math
from shapely import geometry
from ligo.skymap.io import fits
from spherical_geometry.polygon import SphericalPolygon
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import geos
print(geos.__file__)


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
    _X, _Y, _Z = zip(*poly.points)
    lon, lat, rho = xyz_to_lon_lat(_X, _Y, _Z)
    lon_lat = zip(lon, lat)
    poly = geometry.Polygon(lon_lat)
    return poly

dataset = 'design_bns_astro' # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8 # FOV diameter in degrees

# Convert FOV to radius and radians
fov_diameter = np.deg2rad(fov_diameter)

#Open sample file, tested on 100
i = 204

X, Y, Z = convert_fits_xyz(dataset, i)
inside_point = X[1], Y[1], Z[1] #It's probably inside ?? NOTE should be cahnged though
#We need to cluster the points before we convex hull
region = SphericalPolygon.convex_hull(list(zip(X,Y,Z)))
region = region.invert_polygon()

circle = get_circle(0, 0, fov_diameter)

m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')

circle.draw(m)
region.draw(m)

plt.show()

# m, metadata = fits.read_sky_map('data/' + dataset + '/' + str(i) + '.fits', nest=None)
# region90 = area(m,fov_radius,dil=0.99,deg_fact=8)

# n = np.zeros(len(m))
# n[region90.area_pix] = 0.00005

# print(max(n))