from pathlib import Path
from astropy.table import Table
from astropy.io import fits

# Path to the FITS file (handles the space in the filename)
#fits_path = Path("/Users/s2537809/Downloads/primer_f356w_sed_fitted_catalogue 1.fits")
#fits_path = Path("/Users/s2537809/Downloads/JELS_F466N_Halpha_sfh_continuity_salim_v2_bpass_posteriors.fits")
fits_path = Path("/home/cpirie/JELS/SED_fitting/BAGPIPES/halpha_runs/JELS_Halpha_sfh_continuity_salim_v2_bpass_posteriors.fits")


# Print HDU info to see where the table is located
fits.info(fits_path)

# Read the first table found in the FITS file as an astropy Table
table = Table.read(fits_path, format='fits')

# Quick summary
print(f"\nLoaded table with {len(table)} rows and {len(table.colnames)} columns")
print("Columns:", table.colnames)

# Show the first 10 rows 
table[:10]