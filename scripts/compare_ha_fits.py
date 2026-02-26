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

def compare_columns(A, B, colA, colB, idsA, idsB, idxA, idxB):
    if colA not in A.colnames or colB not in B.colnames:
        return None
    arrA = as_float(A, colA)
    arrB = as_float(B, colB)
    if arrA is None or arrB is None:
        return None

    common = idsA & idsB
    diffs = []
    for obj_id in common:
        a = arrA[idxA[obj_id]]
        b = arrB[idxB[obj_id]]
        if np.isfinite(a) and np.isfinite(b) and b != 0:
            diffs.append(abs((a - b) / b))
    if not diffs:
        return None
    diffs = np.asarray(diffs, dtype=float)
    return float(np.nanmedian(diffs)), float(np.nanmax(diffs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="First Hα FITS (A)")
    parser.add_argument("--b", required=True, help="Second Hα FITS (B)")
    args = parser.parse_args()

    A = Table.read(args.a)
    B = Table.read(args.b)

    print(f"[INFO] A rows = {len(A)}, B rows = {len(B)}")
    print(f"[INFO] A columns = {A.colnames}")
    print(f"[INFO] B columns = {B.colnames}")

    if "ID" not in A.colnames or "ID" not in B.colnames:
        raise ValueError("Both tables must have ID column.")

    idsA = set(np.array(A["ID"], dtype=int))
    idsB = set(np.array(B["ID"], dtype=int))
    idxA = {int(i): k for k, i in enumerate(np.array(A["ID"], dtype=int))}
    idxB = {int(i): k for k, i in enumerate(np.array(B["ID"], dtype=int))}

    print(f"[INFO] Common IDs: {len(idsA & idsB)}")

    # ---- Column mapping: A -> B ----
    # Add/modify mappings here
    mappings = [
        ("Ha_flux", "Ha_flux"),
        ("Ha_flux_err", "Ha_flux_err"),
        ("L_Ha_uncorr", "L_Ha_uncorr"),
        ("L_Ha_uncorr_err", "L_Ha_uncorr_err"),
        ("L_Ha_apcorr", "L_Ha_apcorr"),
        ("L_Ha_apcorr_err", "L_Ha_apcorr_err"),

        # A's dust-corrected columns vs B's dust-corrected columns
        ("L_Ha_ap_dustcorr_line", "L_Ha_ap_dustcorr_line"),
        ("L_Ha_ap_dustcorr_line_err", "L_Ha_ap_dustcorr_line_err"),

        # If you want to compare A dust vs B v1/v2 explicitly:
        ("L_Ha_ap_dustcorr_cont", "L_Ha_corr_v1"),
        ("L_Ha_ap_dustcorr_line", "L_Ha_corr_v2"),
    ]

    print("\n[INFO] Mapped column comparisons (median |frac diff| / max)")
    for colA, colB in mappings:
        res = compare_columns(A, B, colA, colB, idsA, idsB, idxA, idxB)
        if res is None:
            print(f"{colA:28s} vs {colB:28s} : (no comparable data)")
        else:
            med, mx = res
            print(f"{colA:28s} vs {colB:28s} : median={med:.3e} max={mx:.3e}")

if __name__ == "__main__":
    main()
