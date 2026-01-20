from astropy.io import fits
import sys
import os

def inspect_muse_cube(cube_path):
    """Inspect the structure and FSF information of a MUSE FITS cube."""
    if not os.path.isfile(cube_path):
        print(f" File not found: {cube_path}")
        return

    print(f"\nInspecting MUSE cube: {cube_path}\n")

    # Open FITS file
    hdul = fits.open(cube_path, ignore_missing_simple=True)
    print(f"Number of extensions: {len(hdul)}\n")

    # Print extension summary
    print("FITS structure:")
    print("-" * 70)
    for i, hdu in enumerate(hdul):
        name = hdu.name if hdu.name else "PRIMARY"
        shape = getattr(hdu.data, "shape", None)
        print(f"[{i:>2}] {name:<15} | dtype={getattr(hdu.data, 'dtype', 'N/A')} | shape={shape}")
    print("-" * 70)

    # Check for FSF keywords
    primary_header = hdul[0].header
    fsf_keywords = [k for k in primary_header.keys() if "FSF" in k or "PSF" in k or "SEE" in k]
    print("\n FSF-related keywords in PRIMARY header:")
    if fsf_keywords:
        for key in fsf_keywords:
            print(f"  {key:<15} = {primary_header[key]}")
    else:
        print("  (No FSF-related keywords found)")

    # Print key WCS and cube metadata
    print("\n Header from DATA:")
    data_header = hdul['DATA'].header
    print(f"Full data header:  {data_header}")
    print("\n Header from Primary HDU:")
    print(f"Full PRIMARY header:  {primary_header}")

    # Print the size of the data cube
    data_shape = hdul['DATA'].data.shape
    print(f"\n DATA cube shape: {data_shape} (Z, Y, X)")

    # Optionally show first lines of full header (for context)
    print("\nFull PRIMARY header preview:")
    for i, card in enumerate(primary_header.cards[:20]):
        print(f"  {card.keyword:<10} {card.value}")

    hdul.close()
    print("\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_MUSE_CUBE.py /home/apatrick/MUSE/DATACUBE_UDF-MOSAIC.fits")
    else:
        inspect_muse_cube(sys.argv[1])
        

# /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE.fits
# /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_withFSF_VAR.fits
