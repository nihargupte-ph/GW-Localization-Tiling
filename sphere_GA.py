#Import
import healpy as hp
import numpy as np
from ligo.skymap.io import fits
from spherical_geometry.polygon import SphericalPolygon

dataset = 'design_bns_astro' # name of dataset ('design_bns_astro' or 'design_bbh_astro')
fov_diameter = 8 # FOV diameter in degrees

# Convert FOV to radius and radians
fov_diameter = np.deg2rad(fov_diameter)
fov_radius = fov_diameter / 2

#Open sample file, tested on 100
i = 204
m, metadata = fits.read_sky_map('data/' + dataset + '/' + str(i) + '.fits', nest=None)  

# SphericalPolygon.from_wcs()

def convert_fits_xyz(dataset, number, nested=True, nside = None):

        m, metadata = fits.read_sky_map('data/{}/{}.fits'.format(dataset, number), nest=None)  

        dil=0.99
        deg_fact=8
        
        if nside is None: 
            self.nside = hp.npix2nside(len(m))
        else:
            self.nside = nside
        
        #Obtain pixels covering the 90% region
        #Sorts pixels based on probability, 
        #then takes all pixel until cumulative sum is >= 90%
        mflat = m.flatten()
        i = np.flipud(np.argsort(mflat))
        msort = mflat[i]
        mcum = np.cumsum(msort)            
        ind_to_90 = len(mcum[mcum <= 0.9*mcum[-1]])

        self.area_pix = i[:ind_to_90]
        self.max_pix  = i[0]

        x,y,z = hp.pix2vec(self.nside,self.area_pix,nest=nested)

        print(x,y,z)