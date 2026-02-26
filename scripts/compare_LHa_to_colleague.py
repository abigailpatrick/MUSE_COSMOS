#!/usr/bin/env python3

import numpy as np
from astropy.table import Table


# ----------------------------
# Files
# ----------------------------
my_f466 = "/home/apatrick/P1/JELSDP/MY_F466N_colleague_exact.fits"
my_f470 = "/home/apatrick/P1/JELSDP/MY_F470N_colleague_exact.fits"

col_f466 = "/home/apatrick/P1/JELSDP/JELS_F466N_with_LHa_match_v1p0.fits"
col_f470 = "/home/apatrick/P1/JELSDP/JELS_F470N_with_LHa_match_v1p0.fits"


# ----------------------------
# Columns to compare
# ----------------------------
cols = [
    "L_Ha_uncorr",
    "L_Ha_corr_v1",
    "L_Ha_corr_v2",
    "A_Ha_cont",
    "A_Ha_line"
]


# ----------------------------
# Comparison function
# ----------------------------
def compare(myfile, colfile, label):

    print("\n=======================================")
    print(f"Comparing {label}")
    print("=======================================")

    mytab  = Table.read(myfile)
    #print(f"columns in my file: {mytab.colnames}")
    coltab = Table.read(colfile)
    #print(f"columns in colleague's file: {coltab.colnames}")

    my_ids  = np.array(mytab["ID"], dtype=int)
    col_ids = np.array(coltab["ID"], dtype=int)

    idx_col = {int(i): j for j, i in enumerate(col_ids)}

    for c in cols:
        if c not in mytab.colnames or c not in coltab.colnames:
            print(f"[SKIP] Missing column {c}")
            continue

        ratios = []
        diffs  = []

        for i, obj_id in enumerate(my_ids):

            if obj_id not in idx_col:
                continue

            j = idx_col[obj_id]

            v_my  = mytab[c][i]
            v_col = coltab[c][j]

            if np.isfinite(v_my) and np.isfinite(v_col) and v_col != 0:
                ratios.append(v_my / v_col)
                diffs.append(v_my - v_col)

        ratios = np.array(ratios)
        diffs  = np.array(diffs)

        print(f"\nColumn: {c}")
        print(f"   N match = {len(ratios)}")
        print(f"   Median ratio = {np.nanmedian(ratios):.6f}")
        print(f"   Max |diff|    = {np.nanmax(np.abs(diffs)):.3e}")


# ----------------------------
# Run comparisons
# ----------------------------
if __name__ == "__main__":

    compare(my_f466, col_f466, "F466N")
    compare(my_f470, col_f470, "F470N")

