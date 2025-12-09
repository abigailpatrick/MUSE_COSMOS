#!/usr/bin/env python3

import sys
import os
import re
from astropy.io import fits

def main():
    if len(sys.argv) != 4:
        print("Usage: edit_fsf.py INPUT_SUBCUBE.fits FWHM BETA")
        sys.exit(1)

    in_path = sys.argv[1]
    try:
        fwhm = float(sys.argv[2])
        beta = float(sys.argv[3])
    except ValueError:
        print("FWHM and BETA must be numbers.")
        sys.exit(1)

    root, ext = os.path.splitext(in_path)
    if ext == "":
        ext = ".fits"
    out_path = f"{root}_{fwhm}_{beta}{ext}"

    # Patterns: FSFxxFnn (FWHM), FSFxxBnn (BETA); do not touch counts or other FSF keys
    patt_f = re.compile(r"^FSF\d{2}F\d{2}$")
    patt_b = re.compile(r"^FSF\d{2}B\d{2}$")

    with fits.open(in_path, mode="readonly", memmap=False) as hdul:
        hdr = hdul[0].header

        # Set all FWHM entries
        for k in list(hdr.keys()):
            if patt_f.match(k):
                hdr[k] = float(fwhm)

        # Set all BETA entries
        for k in list(hdr.keys()):
            if patt_b.match(k):
                hdr[k] = float(beta)

        hdul.writeto(out_path, overwrite=True)
        print(f"Written edited FSF header to: {out_path}")


if __name__ == "__main__":
    main()
