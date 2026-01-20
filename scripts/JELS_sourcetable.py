"""
Open both the photomtery files and the sed files
combine the df's into one 
limit to just the sources I want -remove the agn ? (my brightest source noooooo)
calculate and add in fluxes and luminosities uncorrected
Aperture correct 
Dust correct


"""
#!/usr/bin/env python3
from astropy.table import Table, join, vstack
import numpy as np

# --- File paths ---
cat466_path = "/home/apatrick/P1/JELSDP/jels_F466N_detected_high_z_candidates_v1.01.fits"
cat470_path = "/home/apatrick/P1/JELSDP/jels_F470N_detected_high_z_candidates_v1.01.fits"

sed466_path = "/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_sfh_continuity_salim_v2_bpass_posteriors.fits"
sed470_path = "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_sfh_continuity_salim_v2_bpass_posteriors.fits"

output_fits = "/home/apatrick/P1/JELSDP/combined_selected_sources.fits"

# --- IDs to keep ---
ids_to_keep = [1988, 2962, 3221, 3228, 4126, 4469, 4970, 5364, 5887, 
               3535, 4820, 6680]

def ensure_int_id(t):
    if 'ID' not in t.colnames:
        raise ValueError("Table is missing required 'ID' column.")
    try:
        t['ID'] = t['ID'].astype(int)
    except Exception:
        t['ID'] = [int(x) for x in t['ID']]
    return t

def drop_overlap_columns(left, right, key='ID'):
    """Drop from 'right' any columns that also appear in 'left' (except key)."""
    overlap = sorted((set(left.colnames) & set(right.colnames)) - {key})
    if overlap:
        right = right.copy()
        for c in overlap:
            right.remove_column(c)
    return right, overlap

def main():
    # Load tables
    cat466 = ensure_int_id(Table.read(cat466_path))
    sed466 = ensure_int_id(Table.read(sed466_path))
    cat470 = ensure_int_id(Table.read(cat470_path))
    sed470 = ensure_int_id(Table.read(sed470_path))
    print(f"cat466 columns: {cat466.colnames}")
    print(f"sed470 columns: {sed470.colnames}")

    # To keep original column names, drop from sed tables any non-ID columns
    # that also exist in the corresponding cat tables
    sed466_clean, dup466 = drop_overlap_columns(cat466, sed466, key='ID')
    sed470_clean, dup470 = drop_overlap_columns(cat470, sed470, key='ID')

    if dup466:
        print("F466N: Dropping overlapping columns from SED (kept CAT versions):")
        print("  " + ", ".join(dup466))
    else:
        print("F466N: No overlapping non-ID columns between CAT and SED.")

    if dup470:
        print("F470N: Dropping overlapping columns from SED (kept CAT versions):")
        print("  " + ", ".join(dup470))
    else:
        print("F470N: No overlapping non-ID columns between CAT and SED.")

    # Join on ID without renaming (since no overlapping non-ID columns remain)
    t466 = join(cat466, sed466_clean, keys='ID', join_type='inner')
    t470 = join(cat470, sed470_clean, keys='ID', join_type='inner')

    # --- Add source catalogue column ---
    t466['SOURCE_CAT'] = 'F466N'
    t470['SOURCE_CAT'] = 'F470N'

    print(f"\nJoined F466N: {len(t466)} rows, {len(t466.colnames)} columns.")
    print(f"Joined F470N: {len(t470)} rows, {len(t470.colnames)} columns.")

    # Show column differences between the two joined tables
    cols466 = set(t466.colnames)
    cols470 = set(t470.colnames)
    only_in_466 = sorted(cols466 - cols470)
    only_in_470 = sorted(cols470 - cols466)

    print("\nColumns only in joined F466N (will be blank for F470N rows):")
    print("  " + (", ".join(only_in_466) if only_in_466 else "None"))

    print("\nColumns only in joined F470N (will be blank for F466N rows):")
    print("  " + (", ".join(only_in_470) if only_in_470 else "None"))

    # Combine (outer vstack keeps all columns, blanks missing)
    combined = vstack([t466, t470], join_type="outer", metadata_conflicts="silent")
    print(f"\nCombined (pre-filter): {len(combined)} rows, {len(combined.colnames)} columns.")

    # Filter to IDs of interest
    # Ensure ID is int for comparison
    try:
        id_int = combined['ID'].astype(int)
    except Exception:
        id_int = np.array([int(x) for x in combined['ID']])
    mask_keep = np.isin(id_int, ids_to_keep)
    combined_sel = combined[mask_keep]
    print(f"Combined (post-filter): {len(combined_sel)} rows kept out of {len(combined)} total.")
    # print full table columns names
    print(combined_sel.colnames)

    # Save
    combined_sel.write(output_fits, overwrite=True)
    print(f"\nWrote filtered, combined table to: {output_fits}")

if __name__ == "__main__":
    main()
