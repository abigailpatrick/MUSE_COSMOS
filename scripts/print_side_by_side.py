#!/usr/bin/env python3
import argparse
import numpy as np
from astropy.table import Table, join

def as_table(path, cols):
    t = Table.read(path)
    keep = ["ID"] + [c for c in cols if c in t.colnames]
    return t[keep]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="File A")
    parser.add_argument("--b", required=True, help="File B")
    parser.add_argument("--cols-a", nargs="+", required=True, help="Columns to print from A")
    parser.add_argument("--cols-b", nargs="+", required=True, help="Columns to print from B")
    parser.add_argument("--out", default=None, help="Optional CSV output")
    args = parser.parse_args()

    A = as_table(args.a, args.cols_a)
    B = as_table(args.b, args.cols_b)

    # Rename to keep them distinct
    for c in args.cols_a:
        if c in A.colnames:
            A.rename_column(c, f"A__{c}")
    for c in args.cols_b:
        if c in B.colnames:
            B.rename_column(c, f"B__{c}")

    T = join(A, B, keys="ID", join_type="inner")

    # Print to screen
    print(T)

    # Optional save
    if args.out:
        if args.out.endswith(".csv"):
            T.write(args.out, format="ascii.csv", overwrite=True)
        else:
            T.write(args.out, overwrite=True)
        print(f"[OK] Wrote {args.out}")

if __name__ == "__main__":
    main()
