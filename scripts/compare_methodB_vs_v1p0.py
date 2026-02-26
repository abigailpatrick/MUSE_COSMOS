#!/usr/bin/env python3
import argparse
import numpy as np
from astropy.table import Table

def as_float(tab, name):
    if name not in tab.colnames:
        return None
    data = tab[name]
    try:
        arr = np.array(data, dtype=float)
    except Exception:
        return None
    if hasattr(data, "mask"):
        arr[data.mask] = np.nan
    return arr

def select_v1p0_flux(tab, band):
    if band == "F466N":
        f1 = as_float(tab, "F_line_F444W_F466N")
        f2 = as_float(tab, "F_line_F470N_F466N")
    else:
        f1 = as_float(tab, "F_line_F444W_F470N")
        f2 = as_float(tab, "F_line_F466N_F470N")
    if f1 is None or f2 is None:
        return None
    return np.where(f1 > 0, f1, f2)

def compare_arrays(a, b, idsA, idsB, idxA, idxB):
    common = idsA & idsB
    diffs = []
    for obj_id in common:
        x = a[idxA[obj_id]]
        y = b[idxB[obj_id]]
        if np.isfinite(x) and np.isfinite(y) and y != 0:
            diffs.append(abs((x - y) / y))
    if not diffs:
        return None
    diffs = np.asarray(diffs, dtype=float)
    return float(np.nanmedian(diffs)), float(np.nanmax(diffs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methodb", required=True, help="Your Method B FITS (e.g., JELS_F466N_with_LHa_match_v1p0.fits)")
    parser.add_argument("--v1p0", required=True, help="v1p0 catalogue FITS (e.g., JELS_F466N_Halpha_cat_v1p0.fits)")
    parser.add_argument("--band", required=True, choices=["F466N", "F470N"])
    args = parser.parse_args()

    A = Table.read(args.methodb)
    B = Table.read(args.v1p0)

    idsA = set(np.array(A["ID"], dtype=int))
    idsB = set(np.array(B["ID"], dtype=int))
    idxA = {int(i): k for k, i in enumerate(np.array(A["ID"], dtype=int))}
    idxB = {int(i): k for k, i in enumerate(np.array(B["ID"], dtype=int))}

    print(f"[INFO] MethodB rows={len(A)} | v1p0 rows={len(B)} | common IDs={len(idsA & idsB)}")

    # Flux comparison
    fluxA = as_float(A, "Ha_flux")
    fluxB = select_v1p0_flux(B, args.band)
    if fluxA is not None and fluxB is not None:
        res = compare_arrays(fluxA, fluxB, idsA, idsB, idxA, idxB)
        print("Ha_flux vs v1p0 flux:", res if res else "no comparable data")
    else:
        print("Ha_flux vs v1p0 flux: missing columns")

    # Luminosity comparisons
    pairs = [
        ("L_Ha_uncorr", "L_halpha_uncorr"),
        ("L_Ha_apcorr", "L_halpha_uncorr"),
        ("L_Ha_ap_dustcorr_cont", "L_halpha_corr_v1"),
        ("L_Ha_ap_dustcorr_line", "L_halpha_corr_v2"),
    ]

    for a_name, b_name in pairs:
        a = as_float(A, a_name)
        b = as_float(B, b_name)
        if a is None or b is None:
            print(f"{a_name} vs {b_name}: missing columns")
            continue
        res = compare_arrays(a, b, idsA, idsB, idxA, idxB)
        print(f"{a_name} vs {b_name}: {res if res else 'no comparable data'}")

if __name__ == "__main__":
    main()
