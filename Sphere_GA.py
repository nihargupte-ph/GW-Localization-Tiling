from shapely import geometry
from misc_functions import *
from Flat_GA import ga
import os

cwd = os.getcwd()

def get_m(lon_0=0, lat_0=0, **plot_args):
    """ Given plot args returns a basemap "axis" with the proper plot args. Edit this function if you want different maps """

    #m = Basemap(projection="ortho", resolution="c", lon_0=-20, lat_0=0, **plot_args)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    m = Basemap(projection="moll", resolution="c", lon_0=lon_0, lat_0=lat_0, ax=ax)
    return m

def get_circle(phi, theta, fov, step=16):
    """ Returns SphericalPolygon given FOV and center of the polygon """

    radius = fov / 2
    lons = [
        phi + radius * np.cos(angle) for angle in np.linspace(0, 2 * np.pi, step)
    ]
    lats = [
        theta + radius * np.sin(angle) for angle in np.linspace(0, 2 * np.pi, step)
    ]
    ret = SphericalPolygon.from_radec(lons, lats)
    return ret

dataset = "design_bns_astro"  # name of dataset ('design_bns_astro' or 'design_bbh_astro')
for id in [306]:

    fov = 8  # FOV diameter in degrees

    # Open sample file, tested on 100
    coords_list = get_concave_hull(dataset, id, new=False)

    region_list, spher_region_list = [], []
    for coords in coords_list:
        lon, lat = zip(*coords)

        lonlat_poly = geometry.Polygon(list(zip(lon, lat)))

        inside_pt = generate_random_in_polygon(1, lonlat_poly)[0]
        spher_region_list.append(SphericalPolygon.from_lonlat(lon, lat, center=inside_pt))

        region_list.append(lonlat_poly)
    
    if len(region_list) == 0:
        continue

    total_area = sum([region.area() for region in spher_region_list])
    circ_area = get_circle(0, 0, fov)
    circ_area = circ_area.area()
    area_list = [region.area for region in region_list]
    area_list_norm = [area/sum(area_list) for area in area_list]
    total_initial_guess = 200
    initial_guess_list = []
    for frac in area_list_norm:
        guess = int(np.ceil(frac * total_initial_guess))
        initial_guess_list.append(guess)

    circle_list, best_agent_list = [], []
    for i, region in enumerate(region_list):
        minx, miny, maxx, maxy = region.bounds
        bounding_box = {
            "bottom left": (minx, miny),
            "bottom right": (maxx, miny),
            "top right": (maxx, maxy),
            "top left": (minx, maxy),
        }
        best_agent = ga(
            region,
            4,
            bounding_box,
            initial_length=initial_guess_list[i],
            plot_regions=True,
            save_agents=False,
            plot_crossover=True,
        )
        print([(i.x, i.y) for i in best_agent.center_list])
        best_agent_list.append(best_agent)

        for center in best_agent.center_list:
            circle_list.append(get_circle(center.x, center.y, fov))


    print(id, total_area, len(circle_list))
    #Getting lon lat centroid of region for the map
    av_x = sum([region.centroid.x for region in region_list]) / len(region_list)
    av_y = sum([region.centroid.y for region in region_list]) / len(region_list)
    m = get_m(lon_0=av_x, lat_0=av_y)
    for spher_region in spher_region_list:
        spher_region.draw(m, c='r')
    for circle in circle_list:
        circle.draw(m, c='b', alpha=0.5, linewidth=1)
    plt.savefig(f"{id}.pdf")
    plt.close()