from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
import healpy as hp
from ligo.skymap.io import fits
import ligo.skymap.plot
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib.path import Path
from matplotlib import pyplot as plt
import numpy as np

def plot_ligo(fname, centers, outname):

    """ Given filename of .fits file, array of centers (phi, theta), and output file name plots skymap and pointing in the same way as the paper  """

    m, metadata = fits.read_sky_map(fname, nest=None)
    nside = hp.npix2nside(len(m))
    ipix = np.argmax(m)
    lon, lat = hp.pix2ang(nside, ipix, nest=True, lonlat=True)*u.deg
    # Optional: recenter the map to center skymap in inset:
    # lat -= 1*u.deg
    # lon += 3.5*u.deg
    center = SkyCoord(lon, lat)

    #phis, thetas = centers
    #thetas, phis = hp.vec2ang(centers)

    # ras = np.rad2deg(phis)
    # decs = np.rad2deg(0.5 * np.pi - thetas)

    #point_coords = zip(ras, decs)
    point_coords = centers

    cls = 100 * find_greedy_credible_levels(m)

    fig = plt.figure(figsize=(4, 4), dpi=300)

    ax = plt.axes(
        [0.05, 0.05, 0.9, 0.9],
        projection='astro globe',
        center=center)

    ax_inset = plt.axes(
        [0.59, 0.3, 0.4, 0.4],
        projection='astro zoom',
        center=center,
        radius=15*u.deg)

    for key in ['ra', 'dec']:
        ax_inset.coords[key].set_ticklabel_visible(False)
        ax_inset.coords[key].set_ticks_visible(False)
    ax.coords['ra'].set_ticks(spacing=60*u.deg)
    ax.grid()
    ax.mark_inset_axes(ax_inset)
    ax.connect_inset_axes(ax_inset, 'upper left')
    ax.connect_inset_axes(ax_inset, 'lower left')
    ax_inset.scalebar((0.1, 0.1), 5 * u.deg).label()
    ax_inset.compass(0.9, 0.1, 0.2)

    ax.imshow_hpx(m, cmap='cylon', nested=True)
    ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='k', linewidths=0.5, levels=[90])
    ax_inset.imshow_hpx(m, cmap='cylon', nested=True)
    ax_inset.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='k', linewidths=0.5, levels=[90])

    for coord in point_coords:
        ax_inset.plot(
            coord[0], coord[1],
            transform=ax_inset.get_transform('world'),
            marker=Path.circle(),
            markersize=33,
            alpha=0.25,
            color='blue')

    plt.savefig(outname, dpi=300, bbox_inches='tight')