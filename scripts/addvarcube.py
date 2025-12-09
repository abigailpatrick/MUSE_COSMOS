from astropy.io import fits

# Paths
data_path = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_withFSF.fits'
var_path  = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_VAR.fits'
output_path = '/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_withFSF_VAR.fits'

def print_structure(label, hdul):
    print(f"\n{'='*70}")
    print(f"Inspecting {label}:")
    print(f"{'-'*70}")
    for i, hdu in enumerate(hdul):
        name = hdu.name if hdu.name else f'HDU{i}'
        dtype = str(hdu.data.dtype) if hdu.data is not None else 'N/A'
        shape = hdu.data.shape if hdu.data is not None else None
        print(f"[{i:2d}] {name:10s} | dtype={dtype:>10s} | shape={shape}")
    print(f"{'='*70}\n")

# Load FITS files
data_hdul = fits.open(data_path)
var_hdul  = fits.open(var_path)

print_structure("FSF Cube (withFSF)", data_hdul)
print_structure("Variance Cube (VAR)", var_hdul)

# Use full header from FSF cube for primary
primary_hdr = data_hdul[0].header.copy()
primary_hdr['EXTEND'] = True  # Required for extensions
primary_hdu = fits.PrimaryHDU(header=primary_hdr)
primary_hdu.header['EXTNAME'] = 'PRIMARY'

# Flux, variance, DQ
flux_hdu    = fits.ImageHDU(data=data_hdul[1].data, header=data_hdul[1].header, name='DATA')
var_hdu_new = fits.ImageHDU(data=var_hdul[0].data, header=data_hdul[1].header, name='STAT')
dq_hdu      = fits.ImageHDU(data=data_hdul[2].data, header=data_hdul[2].header, name='DQ')

# Combine into HDUList
new_hdul = fits.HDUList([primary_hdu, flux_hdu, var_hdu_new, dq_hdu])
new_hdul.writeto(output_path, overwrite=True)

print(f"\nNew cube written to: {output_path}")

# Print final structure
final_hdul = fits.open(output_path)
print_structure("Final Combined Cube (withFSF_VAR)", final_hdul)

# Cleanup
data_hdul.close()
var_hdul.close()
final_hdul.close()
