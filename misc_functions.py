import shapely.geometry as geometry
from shapely.geometry import Polygon
from itertools import zip_longest
import numpy as np
import random
from ligo.skymap.io import fits
from ligo.skymap.postprocess import find_greedy_credible_levels
import healpy as hp
import math
from mpl_toolkits.basemap import Basemap
from collections import defaultdict

def get_m(**plot_args):
    """ Given plot args returns a basemap "axis" with the proper plot args. Edit this function if you want different maps """
    
    
    m = Basemap(projection='ortho', resolution='c', lon_0 = -70, lat_0 = 50, **plot_args)
    #m.bluemarble()
    return m

def diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return list(zip_longest(fillvalue=fillvalue, *args))

def around(coords, precision=5):
    result = []
    try:
        return round(coords, precision)
    except TypeError:
        for coord in coords:
            result.append(around(coord, precision))
    return result

def layer_precision(geom, precision=5):
    """ https://stackoverflow.com/questions/49099049/geopandas-shapely-spatial-difference-topologyexception """
    geojson = geometry.mapping(geom)
    geojson['coordinates'] = around(geojson['coordinates'],precision)
    return geometry.shape(geojson)

def voronoi_polygons(voronoi, diameter):
    """https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region/38206656#38206656
    Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    """
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ):
    ''' https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (.01 * int(x), .01 * int(y)) )

        angle = angle + angleSteps[i]

    return points

def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x

def generate_random_in_polygon(number, polygon):
    list_of_points = []
    minx, miny, maxx, maxy = polygon.bounds
    counter = 0
    while counter < number:
        pnt = geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            list_of_points.append((pnt.x, pnt.y))
            counter += 1
    return list_of_points

def xyz_to_lon_lat(X, Y, Z):
    """ Takes list of X, Y, and Z coordinates and spits out list of lon lat and rho """

    phi = [math.degrees(np.arctan(y/x))+180 for x, y in zip(X,Y)]
    theta = [math.degrees(np.arccos(z / math.sqrt((x**2)+(y**2)+(z**2))))+90 for x, y, z in zip(X,Y,Z)]
    rho = [x**2 + y**2 + z**2 for x, y, z in zip(X,Y,Z)]

    return phi, theta, rho

def lon_lat_to_xyz(lons, lats, radius):
    """ Converts set of points in longitude and lattitude to XYZ assuming r=1 """

    X = radius*np.cos(np.radians(lons))*np.sin(np.radians(lats))
    Y = radius*np.sin(np.radians(lons))*np.sin(np.radians(lats))
    Z = radius*np.cos(np.radians(lats))

    return X, Y, Z

def removal_copy(lst, x):
    """ Returns a list with the element x removed """
    ret = lst[:]
    try:
        ret.remove(x)
    except:
        raise Exception("The list does not contain this element")

    return ret


import matplotlib.pyplot as plt

def convert_fits_xyz(dataset, number, nested=True, nside = None):
    """ Given a fits file converts into xyz point """

    #Extracts data from fits file
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
    print(ind_to_90)

    area_pix = i[:ind_to_90]
    print(area_pix)

    x, y, z = hp.pix2vec(nside,area_pix,nest=nested)

    lon, lat, r = xyz_to_lon_lat(x, y, z)

    theta, phi = hp.pix2ang(nside, area_pix)
    ra = np.rad2deg(phi - (math.pi))
    dec = np.rad2deg(theta - (math.pi/2))

    l = np.zeros(len(m))
    l[area_pix] = 0.00005

    hp.orthview(l,nest=True,title='Optimized Coverage', rot=(-70, 0, 0), half_sky=True)	
    plt.show()
    
    m = get_m()
    print(lon, lat)
    x,y = m(lon, lat)
    m.scatter(x,y)
    plt.show()

    exit()

    return ra, dec