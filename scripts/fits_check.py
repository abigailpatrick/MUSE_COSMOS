# open this fits and show info 

path = "/home/apatrick/P1/col.fits"
from astropy.io import fits
with fits.open(path) as hdul:
    hdul.info()
    print("Data shape:", [hdu.data.shape if hdu.data is not None else None for hdu in hdul])


from astropy.io import fits
from astropy.wcs import WCS
h = fits.getheader("/home/apatrick/P1/col.fits")
print(WCS(h))

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import numpy as np

path = "/home/apatrick/P1/col.fits"
h = fits.getheader(path)
w = WCS(h).celestial
data = fits.getdata(path)
print("data shape:", data.shape)

# pick a test coord in the middle of the mosaic
pos = SkyCoord(150.125, 2.325, unit="deg")

if data.shape[-1] == 3:
    img = data[...,0]  # first channel
else:
    img = data[0]

cut = Cutout2D(img, position=pos, size=(200,200), wcs=w)
print("cutout shape:", cut.data.shape)