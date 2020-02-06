#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:24:26 2019

@author: neilash
"""

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
import healpy as hp
from ligo.skymap.io import fits
import ligo.skymap.plot
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib.path import Path
import matplotlib.patches as mpatch
from matplotlib import pyplot as plt
import numpy as np
from wcsaxes import SphericalCircle

i = str(305)
m, metadata = fits.read_sky_map('/Users/neilash/Documents/cta_pointing/data/design_bns_astro/'+i+'.fits', nest=None)
nside = hp.npix2nside(len(m))
ipix = np.argmax(m)
lon, lat = hp.pix2ang(nside, ipix, nest=True, lonlat=True)*u.deg
# Optional: recenter the map to center skymap in inset:
# lat -= 1*u.deg
# lon += 3.5*u.deg
center = SkyCoord(lon, lat) #lon, lat-30*u.deg


array_handle = open(r"/Users/neilash/Documents/cta_pointing/opt_pointings/design_bns_astro_"+i+"_Comb_Pointings.txt",'r')

whole_list = []
for line in array_handle: 
    temp_list = []
    if line.startswith('[') or line.startswith(' ['):
       line = line.replace('[','').replace(']','').strip()
       for n in range(3):
           if n != 2:
               index = line.index(' ')
               temp_list.append(float(line[:index]))
               line = line[index:].strip()
           else: 
               temp_list.append(float(line))   
       whole_list.append(temp_list)
array_handle.close()

pointing_arr = np.array(whole_list)           
thetas, phis = hp.vec2ang(pointing_arr)

n = np.zeros(m.shape)
for pointing in pointing_arr:
    pixel_ind = hp.query_disc(nside,pointing,np.deg2rad(4),nest=True,inclusive=True)
    n[pixel_ind] = 1 + 0.15*n[pixel_ind]

ras = np.rad2deg(phis)
decs = np.rad2deg(0.5 * np.pi - thetas)

point_coords = zip(ras, decs)

cls = 100 * find_greedy_credible_levels(m)

fig = plt.figure(figsize=(4, 4), dpi=300)

ax = plt.axes(
    [0.05, 0.05, 0.9, 0.9],
    projection='astro globe',
    center=center)

"""
ax_inset = plt.axes(
    [0.05, 0.05, 0.9, 0.9],
    projection='astro globe',
    center=center)


for key in ['ra', 'dec']:
    ax_inset.coords[key].set_ticklabel_visible(False)
    ax_inset.coords[key].set_ticks_visible(False)  
"""

ax.coords['ra'].set_ticks(spacing=60*u.deg)
ax.grid()
"""
ax.mark_inset_axes(ax_inset)
ax.connect_inset_axes(ax_inset, 'upper left')
ax.connect_inset_axes(ax_inset, 'lower left')
ax_inset.scalebar((0.1, 0.1), 5 * u.deg).label()
ax_inset.compass(0.9, 0.1, 0.2)
"""
l = m + n
ax.imshow_hpx(m, cmap='cylon', nested=True, alpha=1)
#ax.imshow_hpx(l, cmap='Blues', nested=True, alpha=0.65) #cmap='Blues'


ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='k', linewidths=0.5, levels=[90])

"""
ax_inset.imshow_hpx(m, cmap='cylon', nested=True)
ax_inset.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='k', linewidths=0.5, levels=[90])
"""
from astropy.coordinates.angles import rotation_matrix
for coord in point_coords:
    circle = SphericalCircle((coord[0]*u.deg, coord[1]*u.deg), 4*u.deg,
        transform=ax.get_transform('world'),
        #markersize=33/1.75,
        alpha=0.25,
        color='blue')
    ax.add_patch(circle)

plt.savefig('big_coverage_example_'+i+'.pdf', dpi=300, bbox_inches='tight')