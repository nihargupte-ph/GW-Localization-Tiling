import shapely.geometry as geometry
from shapely.geometry import Polygon
from itertools import zip_longest
import numpy as np
from collections import defaultdict

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

def get_extrapolated_line(p1,p2):
    'Creates a line extrapoled in p1->p2 direction https://stackoverflow.com/questions/33159833/shapely-extending-line-feature'
    EXTRAPOL_RATIO = 1000
    a = p1
    b = (p1.x+EXTRAPOL_RATIO*(p2.x-p1.x), p1.y+EXTRAPOL_RATIO*(p2.y-p1.y) )
    return geometry.LineString([a,b])

def get_ordered_list(region, linrig, point):
    """ Given LinearRing and point returns list of points closest to said point from polygon """
    intersected_multilinrig = linrig.intersection(region)
    if isinstance(intersected_multilinrig, geometry.MultiLineString):
        final_point_list = []
        for intersected_linrig in list(intersected_multilinrig): #iterates through multilinestring and will flatten at the end
            x,y = intersected_linrig.xy
            x, y = list(x), list(y)
            point_list = list(zip(x,y))
            point_list.sort(key = lambda p: np.sqrt((p[0] - point.x)**2 + (p[1] - point.y)**2))
            point_list = [geometry.Point(p[0], p[1]) for p in point_list]
            final_point_list.append(point_list)

        ret  = [item for sublist in final_point_list for item in sublist] #Flattening
    elif isinstance(intersected_multilinrig, geometry.LineString):
        x,y = intersected_multilinrig.xy
        x, y = list(x), list(y)
        point_list = list(zip(x,y))
        point_list.sort(key = lambda p: np.sqrt((p[0] - point.x)**2 + (p[1] - point.y)**2))
        ret = [geometry.Point(p[0], p[1]) for p in point_list]
    elif isinstance(intersected_multilinrig, geometry.GeometryCollection):
        final_point_list = []
        for item in list(intersected_multilinrig):
            if isinstance(item, geometry.Point):
                point_list = [item]
            elif isinstance(item, geometry.LineString):
                x,y = item.xy
                x, y = list(x), list(y)
                point_list = list(zip(x,y))
                point_list.sort(key = lambda p: np.sqrt((p[0] - point.x)**2 + (p[1] - point.y)**2))
                point_list = [geometry.Point(p[0], p[1]) for p in point_list]
            final_point_list.append(point_list)
        ret  = [item for sublist in final_point_list for item in sublist] #Flattening

    else:
        raise Exception("intersected_multilinrig was not a multilinrig, linrig, or geometry collection")

    return ret