import shapely.geometry as geometry
from shapely.geometry import Polygon
from itertools import zip_longest
import numpy as np
import random
from scipy.spatial import ConvexHull
from ligo.skymap.io import fits
from ligo.skymap.postprocess import find_greedy_credible_levels
import healpy as hp
from spherical_geometry.polygon import SphericalPolygon
from scipy.spatial import Delaunay
import geopandas as gpd
import alphashape
import cartopy.crs as ccrs
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from mpl_toolkits.basemap import Basemap
import os
from collections import defaultdict
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from matplotlib.path import Path
from matplotlib import pyplot as plt
import numpy as np


def diff(li1, li2):
    """ Helper function which returns the difference between two lists """

    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
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
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return list(zip_longest(fillvalue=fillvalue, *args))


def around(coords, precision=5):
    """ Precision errors in shapely objections because of curled up points """
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
    geojson["coordinates"] = around(geojson["coordinates"], precision)
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
            t = voronoi.points[q] - voronoi.points[p]  # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)  # normal
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
        inf = region.index(-1)  # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)]  # Index of previous vertex.
        k = region[(inf + 1) % len(region)]  # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            (dir_j,) = ridge_direction[i, j]
            (dir_k,) = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [
            voronoi.vertices[j] + dir_j * length,
            voronoi.vertices[k] + dir_k * length,
        ]
        yield Polygon(np.concatenate((finite_part, extra_edge)))


def generatePolygon(ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts):
    """ https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
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
    """

    irregularity = clip(irregularity, 0, 1) * 2 * np.pi / numVerts
    spikeyness = clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2 * np.pi / numVerts) - irregularity
    upper = (2 * np.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * np.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * np.pi)
    for i in range(numVerts):
        r_i = clip(random.gauss(aveRadius, spikeyness), 0, 2 * aveRadius)
        x = ctrX + r_i * np.cos(angle)
        y = ctrY + r_i * np.sin(angle)
        points.append((0.01 * int(x), 0.01 * int(y)))

        angle = angle + angleSteps[i]

    return points


def clip(x, min, max):
    if min > max:
        return x
    elif x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def generate_random_in_polygon(number, polygon):
    """ Given a number of points generate that many random points inside a polygon """

    list_of_points = []
    minx, miny, maxx, maxy = polygon.bounds
    counter = 0
    while counter < number:
        pnt = geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            list_of_points.append((pnt.x, pnt.y))
            counter += 1
    return list_of_points

# Spherical Functions


def xyz_to_lon_lat(X, Y, Z):
    """ Takes list of X, Y, and Z coordinates and spits out list of lon lat and rho """

    phi = [np.degrees(np.arctan(y / x)) for x, y in zip(X, Y)]
    theta = [np.degrees(np.arccos(z / np.sqrt((x ** 2) + (y ** 2) + (z ** 2)))) - 90 for x, y, z in zip(X, Y, Z)]
    rho = [x ** 2 + y ** 2 + z ** 2 for x, y, z in zip(X, Y, Z)]

    return phi, theta, rho


def lon_lat_to_xyz(lons, lats, radius):
    """ Converts set of points in longitude and lattitude to XYZ assuming r=1 """

    X = radius * np.cos(np.radians(lons)) * np.sin(np.radians(lats))
    Y = radius * np.sin(np.radians(lons)) * np.sin(np.radians(lats))
    Z = radius * np.cos(np.radians(lats))

    return X, Y, Z


def removal_copy(lst, x):
    """ Returns a list with the element x removed """
    ret = lst[:]
    try:
        ret.remove(x)
    except:
        raise Exception("The list does not contain this element")

    return ret


def convert_fits_xyz(dataset, number, nested=True, nside=None):
    """ Given a fits file converts into xyz point """

    # Extracts data from fits file
    m, metadata = fits.read_sky_map(
        "data/" + dataset + "/" + str(number) + ".fits", nest=None
    )

    if nside is None:
        nside = hp.npix2nside(len(m))
    else:
        nside = nside

    # Obtain pixels covering the 90% region
    # Sorts pixels based on probability,
    # then takes all pixel until cumulative sum is >= 90%
    mflat = m.flatten()
    i = np.flipud(np.argsort(mflat))
    msort = mflat[i]
    mcum = np.cumsum(msort)
    ind_to_90 = len(mcum[mcum <= 0.9 * mcum[-1]])

    area_pix = i[:ind_to_90]

    x, y, z = hp.pix2vec(nside, area_pix, nest=nested)
    lon, lat, r = xyz_to_lon_lat(x, y, z)

    ra = np.asarray(lon) + 180
    dec = 90 - np.asarray(lat)

    return x, y, z


def make_fits_lonlat(path, nested=True, nside=None):
    """ Returns lon lat of given fits file """

    m, metadata = fits.read_sky_map(path, nest=None)

    """	
    m is array with pixel probabilities	
    rad input in degrees	
    dilation optional argument, reduces distance between pointings to 	
    prevent uncovered area at triple point in tessellation	
    nest takes argument from metadata, True or False	
    """

    if nside is None:
        nside = hp.npix2nside(len(m))
    else:
        nside = nside

    # Obtain pixels covering the 90% region
    # Sorts pixels based on probability,
    # then takes all pixel until cumulative sum is >= 90%
    mflat = m.flatten()
    i = np.flipud(np.argsort(mflat))
    msort = mflat[i]
    mcum = np.cumsum(msort)
    ind_to_90 = len(mcum[mcum <= 0.9*mcum[-1]])

    area_pix = i[:ind_to_90]

    x, y, z = hp.pix2vec(nside, area_pix, nest=nested)
    lon, lat, _ = xyz_to_lon_lat(x, y, z)

    return lon, lat


def get_convex_hull(pt_lst1, pt_lst2):
    """ Given 2 lists of points return convex hull of the points """

    points = np.array(list(zip(pt_lst1, pt_lst2)))
    hull = ConvexHull(points)
    hull_pts = list(zip(points[hull.vertices, 0], points[hull.vertices, 1]))

    return hull_pts


def plot_ligo_style(fits_path, pointing_lons, pointing_lats):
    """ Given path to a fits file and centers of points (in lon lat) will plot out the localization map with the 90% greedy credible interval along with the pointings. Note that lons and lats go from 0 to 360 and 0 to 180 respectivly. Note that this only works with a FOV of 8 """

    m, metadata = fits.read_sky_map(fits_path, nest=None)
    nside = hp.npix2nside(len(m))
    ipix = np.argmax(m)
    lon, lat = hp.pix2ang(nside, ipix, nest=True, lonlat=True)*u.deg
    # Optional: recenter the map to center skymap in inset:
    # lat -= 1*u.deg
    # lon += 3.5*u.deg
    center = SkyCoord(lon, lat)

    phis, thetas = np.radians(pointing_lons), np.radians(pointing_lats)
    ras = np.rad2deg(phis)
    decs = np.rad2deg(0.5 * np.pi - thetas)

    point_coords = zip(ras, decs)

    cls = 100 * find_greedy_credible_levels(m)

    fig = plt.figure(figsize=(4, 4), dpi=300)

    ax = plt.axes(
        [0.05, 0.05, 0.9, 0.9],
        projection='astro globe',
        center=center)


    for key in ['ra', 'dec']:
        ax.coords[key].set_ticklabel_visible(False)
        ax.coords[key].set_ticks_visible(False)
    ax.coords['ra'].set_ticks(spacing=60*u.deg)
    ax.grid()

    ax.imshow_hpx(m, cmap='cylon', nested=True)
    ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='k', linewidths=0.5, levels=[90])


    for coord in point_coords:
        ax.plot(
            coord[0], coord[1],
            transform=ax.get_transform('world'),
            marker=Path.circle(),
            markersize=33,
            alpha=0.25,
            color='blue')



def lon_lat_projhull(lon, lat, init_proj=ccrs.AzimuthalEquidistant().proj4_init):
    """ Given a set of lon and lat points will returns a geodataframe of the concave hull. It will either be a multipolygon or a polygon depending on how many different clusters there are """

    #Computing DBSCAN
    X = np.c_[lon, lat]
    db = DBSCAN(eps=1, min_samples=100).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    clustered_pts = []
    for k in unique_labels:

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        tmp = np.c_[xy[:, 0], xy[:,1]]
        clustered_pts.append(tmp)

    print(f"Number of clustered maps {len(clustered_pts)}")
    
    if len(clustered_pts) == 1:
        clustered_pts = [np.c_[lon, lat]]


    # #For testing we got the right clusters
    # def get_m(**plot_args):
    #     """ Given plot args returns a basemap "axis" with the proper plot args. Edit this function if you want different maps """

    #     #m = Basemap(projection="ortho", resolution="c", lon_0=-20, lat_0=0, **plot_args)
    #     m = Basemap(projection="moll", resolution="c", lon_0=0)
    #     m.drawcoastlines()
    #     return m
    # m = get_m()
    # for i, pts in enumerate(clustered_pts):
    #     if len(pts) == 0:
    #         break
    #     x, y = zip(*pts)
    #     x, y = m(x, y)
    #     m.scatter(x, y, c=np.random.rand(3,))
    # plt.show()
        
    alpha_shape_lst = []
    for i, pts in enumerate(clustered_pts):

        if pts == []:
            break

        if len(pts) == 0:
            break

        clustered_lons, clustered_lats = zip(*pts)

        geom = [geometry.Point(xy) for xy in zip(clustered_lons, clustered_lats)]
        crs = {'init': 'epsg:4326'}
        gdf = gpd.GeoDataFrame(crs=crs, geometry=geom)
        gdf_proj = gdf.to_crs(init_proj)

        before = time.process_time()
        print("Constructing alphashape {}".format(i+1))
        alpha_shape = alphashape.alphashape(gdf_proj)
        try:
            if alpha_shape == None:
                print("Not enough coordinates to construct alphashape. Run time {}".format(time.process_time() - before))
                continue
        except ValueError:
            pass

        alpha_shape_lst.append(alpha_shape)
        print("Alphashape constructed. Run time {}".format(time.process_time() - before))
        print()

    return alpha_shape_lst

def save_concave_hull(dataset, id):
    """ Given path to fits data saves the outline as a concave hull """
    cwd = os.getcwd()
    lon, lat = make_fits_lonlat("{}/data/{}/{}.fits".format(cwd, dataset, id))

    df_lst = lon_lat_projhull(lon, lat)
    os.mkdir(f"{cwd}/data/{dataset}_shp/{id}")
    for i, df in enumerate(df_lst):
        df.to_file("{}/data/{}_shp/{}/{}.shp".format(cwd, dataset, id, i))


def get_concave_hull(dataset, id, new=False):
    """ Given a data set and id attemps to return set of lon lat points corresponding to concave hull of object """
    cwd = os.getcwd()
    # Checks if we already saved the agent
    if not os.path.exists("{}/data/{}_shp/{}".format(cwd, dataset, id)) or new:
        save_concave_hull(dataset, id)

    #It will never actually get to 10000 will trigger error and then we know to leave
    coords_list = []
    for i in range(0, 10000):
        try:
            e = gpd.read_file("{}/data/{}_shp/{}/{}.shp".format(cwd, dataset, id, i))
        except:
            break
        lon_lat_geom = e.to_crs("epsg:4326")
        shp = lon_lat_geom.geometry
        tf = shp.geom_type.values != 'Polygon'
        tf = tf[0]
        if tf:
            continue
        coords = [list(shp.geometry.exterior[row_id].coords) for row_id in range(shp.shape[0])][0]
        coords_list.append(coords)
    return coords_list


def proj_poly(spher_poly, proj=ccrs.AzimuthalEquidistant().proj4_init):
    """ Given a spherical polygon and a projection, creates a shapely geometry on the plane """

    radec = list(spher_poly.to_radec())
    if radec == []:
        return 0
    lons, lats = radec[0][0], radec[0][1]

    pts = zip(lons, lats)

    coords = proj_points(pts, proj=proj)

    poly = geometry.Polygon(coords)
    poly = poly.buffer(0)
    return poly


def inv_proj_poly(poly, init_crs=ccrs.AzimuthalEquidistant().proj4_init, center=None):
    """ Given shapely polygon will return spherical polygon from inverse projection """

    X, Y = poly.exterior.xy
    XY = zip(X, Y)
    coords = inv_proj_points(XY)

    lons, lats = zip(*coords)

    spher_poly = SphericalPolygon.from_lonlat(lons, lats, center=center)

    return spher_poly


def proj_points(points, proj=ccrs.AzimuthalEquidistant().proj4_init):
    """ Given set of spherical points (lons and lats) will project in projection specified """

    lons, lats = zip(*points)

    df = pd.DataFrame({
        'Lon': lons,
        'Lat': lats
    })

    geom = [geometry.Point(xy) for xy in zip(df.Lon, df.Lat)]
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geom)
    gdf_proj = gdf.to_crs(proj)

    shp = gdf_proj.geometry
    coords = [(i.x, i.y) for i in shp]

    return coords


def inv_proj_points(points, init_crs=ccrs.AzimuthalEquidistant().proj4_init):
    """ Given a set of points in projected spaced will return points in spherical space coressponding to those points """

    X, Y = zip(*points)
    df = pd.DataFrame({
        'X': X,
        'Y': Y
    })

    geom = [geometry.Point(xy) for xy in zip(df.X, df.Y)]
    gdf = gpd.GeoDataFrame(df, crs=init_crs, geometry=geom)

    lon_lat_geom = gdf.to_crs("epsg:4326")
    shp = lon_lat_geom.geometry
    coords = [(i.x, i.y) for i in shp]

    return coords


def spherical_unary_union(polygon_list):
    """ Given a list of spherical polygon returns the unary union of them """

    big_poly = polygon_list[0]
    for poly in polygon_list[1:]:
        big_poly = big_poly.union(poly)

    return big_poly


def proj_intersection(spher_poly1, spher_poly2, proj=ccrs.AzimuthalEquidistant().proj4_init):
    """ The spherical geometry module currently has a bug where it will not correctly find the intersection between polygons sometimes. See https://github.com/spacetelescope/spherical_geometry/issues/168. This is a function which projects to 2D (not ideal I know) and returns a new polygon which is the intersection """

    poly1 = proj_poly(spher_poly1, proj=proj)
    poly2 = proj_poly(spher_poly2, proj=proj)

    intersec = poly1.intersection(poly2)
    inside_pt = generate_random_in_polygon(1, intersec)[0]

    ret = inv_proj_poly(intersec, center=inside_pt)

    return ret
