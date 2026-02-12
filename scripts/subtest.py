import numpy as np
from astropy.io import fits
cube = "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/source_1_lya_contsub_cube.fits"
with fits.open(cube) as hdul:
    hdul.info()



"""
cube = "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/source_1_lya_contsub_cube.fits"
cube = "/cephfs/apatrick/musecosmos/dataproducts/extractions/source_17_subcube_20.0_3681.fits"
"""