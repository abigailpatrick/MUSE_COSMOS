# open this fits and print the column names

fits_path = "/home/apatrick/P1/JELSDP/F466N_with_LHa_eta1.fits"
from astropy.io import fits
with fits.open(fits_path) as hdul:
    print(hdul.info())
    print(hdul[1].columns)

