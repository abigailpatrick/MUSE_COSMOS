#!/usr/bin/env python3
import argparse
import os
import numpy as np
from astropy.table import Table, join

# Columns we want if present
CANDIDATE_COLS = [
    "ID",
    "Ha_flux", "Ha_flux_err", "Ha_flux_source",
    "F_line_F444W_F466N", "F_line_err_F444W_F466N",
    "F_line_F470N_F466N", "F_line_err_F470N_F466N",
    "L_Ha_uncorr", "L_Ha_uncorr_err",
    "L_Ha_apcorr", "L_Ha_apcorr_err",
    "L_Ha_ap_dustcorr_cont", "L_Ha_ap_dustcorr_cont_err",
    "L_Ha_ap_dustcorr_line", "L_Ha_ap_dustcorr_line_err",
    "L_halpha_uncorr", "L_halpha_err_uncorr",
    "L_halpha_corr_v1", "L_halpha_err_corr_v1",
    "L_halpha_corr_v2", "L_halpha_err_corr_v2",
    "A_Ha_cont", "A_Ha_line", "A_Ha_cont_mag", "A_Ha_line_mag",
    "SOURCE_CAT", "apcorr_factor", "apcorr_source",
    "z1_median", "z_used"
]

def pick_cols(tab):
    return [c for c in CANDIDATE_COLS if c in tab.colnames]

def prefixed_table(path, label):
    t = Table.read(path)
    cols = pick_cols(t)
    t = t[cols]

    # Rename columns (except ID)
    for c in cols:
        if c == "ID":
            continue
        t.rename_column(c, f"{label}__{c}")
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="List of FITS files (F466N tables).")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels for each file.")
    parser.add_argument("--out", required=True, help="Output FITS or CSV.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("If provided, --labels must match number of --inputs.")

    labels = args.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.inputs]

    # Build and join
    t_out = None
    for path, label in zip(args.inputs, labels):
        t = prefixed_table(path, label)
        if t_out is None:
            t_out = t
        else:
            t_out = join(t_out, t, keys="ID", join_type="outer")

    # Write output
    if args.out.endswith(".csv"):
        t_out.write(args.out, format="ascii.csv", overwrite=True)
    else:
        t_out.write(args.out, overwrite=True)

    print(f"[OK] Wrote {args.out} with {len(t_out)} rows and {len(t_out.colnames)} columns.")
    # save as csv too for easy inspection
    csv_out = args.out.rsplit(".", 1)[0] + ".csv"
    t_out.write(csv_out, format="ascii.csv", overwrite=True)
    print(f"[OK] Also saved CSV version to {csv_out}")




if __name__ == "__main__":
    main()
