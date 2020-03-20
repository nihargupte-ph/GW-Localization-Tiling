from Flat_GA import *
from misc_functions import *
import geopandas as gpd
import alphashape
import cartopy.crs as ccrs
import pandas as pd

def get_m(**plot_args):
    """ Given plot args returns a basemap "axis" with the proper plot args. Edit this function if you want different maps """

    #m = Basemap(projection="ortho", resolution="c", lon_0=-20, lat_0=0, **plot_args)
    m = Basemap(projection="moll", resolution="c", lon_0=20)
    m.drawcoastlines()
    return m

def lon_lat_projhull(lon, lat):
    """ Given a set of lon and lat points will returns a geodataframe of the concave hull"""
    
    df = pd.DataFrame({
        'Lon' : lon,
        'Lat' : lat
    })
    geom = [geometry.Point(xy) for xy in zip(df.Lon, df.Lat)]
    crs = {'init' : 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geom)
    gdf_proj = gdf.to_crs(ccrs.AzimuthalEquidistant().proj4_init)
    gdf_proj.plot(c='r')
    alpha_shape = alphashape.alphashape(gdf_proj)
    alpha_shape.plot()
    plt.show()

    return alpha_shape

def save_concave_hull(dataset, id):
    """ Given path to fits data saves the outline as a concave hull """

    lon, lat = make_fits_lonlat("{}/data/{}/{}.fits".format(cwd, dataset, id))

    df = lon_lat_projhull(lon, lat)

    df.to_file("{}/data/{}_shp/{}.shp".format(cwd, dataset, id))
    
def get_concave_hull(dataset, id):
    """ Given a data set and id attemps to return set of lon lat points corresponding to concave hull of object """
    
    #Checks if we already saved the agent
    if not os.path.exists("{}/data/{}_shp/{}.shp".format(cwd, dataset, id)):
        save_concave_hull(dataset, id)

    e = gpd.read_file("{}/data/{}_shp/{}.shp".format(cwd, dataset, id))
    lon_lat_geom = e.to_crs("epsg:4326")
    shp = lon_lat_geom.geometry
    coords = [list(shp.geometry.exterior[row_id].coords) for row_id in range(shp.shape[0])]
    return coords
