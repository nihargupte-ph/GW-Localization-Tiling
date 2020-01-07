import shapely.geometry as geometry
import fiona
import matplotlib.pyplot as plt

file_opened = fiona.open("shape_files/7.shp", "r")
first = file_opened.next()
dot_region = geometry.shape(first['geometry'])
test_polygon = geometry.Polygon([(-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2)])
test_larger_polygon = geometry.Polygon([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)])


x,y = dot_region.exterior.xy
x,y = list(x), list(y)
lst = list(zip(x,y))
new_dot_region = geometry.Polygon(lst)

print(dot_region.contains(test_polygon))
print(test_larger_polygon.contains(test_polygon))
print(new_dot_region.contains(test_polygon))

for i, _ in enumerate(dot_region.interiors):
    plt.plot(*dot_region.interiors[i].coords.xy)

plt.plot(*new_dot_region.exterior.xy, linestyle='--', alpha=.3, linewidth=8)
plt.plot(*test_larger_polygon.exterior.xy)
plt.plot(*test_polygon.exterior.xy)
plt.plot(*dot_region.exterior.xy)
plt.show()