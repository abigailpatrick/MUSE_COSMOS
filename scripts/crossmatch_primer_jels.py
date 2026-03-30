import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


# ----------------------------
# Load PRIMER catalog (FITS)
# ----------------------------
def load_primer_catalog(fits_file):
    """Load RA, Dec, and full data table from a FITS catalog."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        ra = np.array(data["ALPHA_J2000"], dtype=float)
        dec = np.array(data["DELTA_J2000"], dtype=float)
    return ra, dec, data


# ----------------------------
# Load JELS MUSE catalog (CSV)
# ----------------------------
def load_jels_catalog(csv_file):
    """Load JELS-MUSE source catalog from CSV."""
    df = pd.read_csv(csv_file).reset_index(drop=True)
    return df


# ----------------------------
# Create SkyCoord objects
# ----------------------------
def make_coords(ra, dec):
    """Construct an astropy SkyCoord array from RA/Dec in degrees."""
    return SkyCoord(ra=np.asarray(ra, dtype=float) * u.deg,
                    dec=np.asarray(dec, dtype=float) * u.deg)


# ----------------------------
# Crossmatch within radius
# ----------------------------
def crossmatch(primer_coords, jels_coords, radius_arcsec=1.0):
    """
    primer_coords.search_around_sky(jels_coords) returns:
        [0] indices into jels_coords   (the argument / searchcoord)
        [1] indices into primer_coords (the caller / catalogcoord)
    """
    idx_jels, idx_primer, d2d, _ = primer_coords.search_around_sky(
        jels_coords, radius_arcsec * u.arcsec
    )
    return idx_primer, idx_jels, d2d


# ----------------------------
# Build output table
# ----------------------------
def build_output(primer_data, idx_primer, idx_jels, d2d, jels_df):
    """
    Build a DataFrame of matched PRIMER sources with selected columns only,
    plus matched JELS IDs and separations.
    """
    keep_cols = ["ALPHA_J2000", "DELTA_J2000", "z1_median"]

    results = {}
    for p_idx, j_idx, dist in zip(idx_primer, idx_jels, d2d):
        p_idx = int(p_idx)
        j_idx = int(j_idx)
        if p_idx not in results:
            results[p_idx] = {"jels_ids": [], "distances": []}
        results[p_idx]["jels_ids"].append(str(jels_df.iloc[j_idx]["ID"]))
        results[p_idx]["distances"].append(round(float(dist.arcsec), 4))

    rows = []
    for p_idx, info in results.items():
        row = {col: primer_data[p_idx][col] for col in keep_cols}
        row["matched_jels_ids"] = ";".join(info["jels_ids"])
        row["match_dist_arcsec"] = ";".join(str(d) for d in info["distances"])
        row["n_jels_matches"] = len(info["jels_ids"])
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Save output
# ----------------------------
def save_output(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} matched PRIMER sources to {output_file}")


# ----------------------------
# Main function
# ----------------------------
def main():
    primer_file = "/home/apatrick/P1/JELSDP/primer_f356w_sed_fitted_catalogue.fits"
    jels_file   = "/cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv"
    output_file = "primer_jels_crossmatch.csv"
    radius_arcsec = 1.0

    # Load data
    primer_ra, primer_dec, primer_data = load_primer_catalog(primer_file)
    jels_df = load_jels_catalog(jels_file)

    print(f"Loaded {len(primer_ra)} PRIMER sources")
    print(f"Loaded {len(jels_df)} JELS-MUSE sources")

    # Coordinates
    primer_coords = make_coords(primer_ra, primer_dec)
    jels_coords   = make_coords(jels_df["ra_jels"], jels_df["dec_jels"])

    # Crossmatch
    idx_primer, idx_jels, d2d = crossmatch(primer_coords, jels_coords,
                                            radius_arcsec=radius_arcsec)
    

    print(f"Found {len(idx_primer)} total pairwise matches within {radius_arcsec}\"")

    if len(idx_primer) == 0:
        print("No matches found. Check coordinate column names and search radius.")
        return

    # Build and save output table
    output_df = build_output(primer_data, idx_primer, idx_jels, d2d, jels_df)
    save_output(output_df, output_file)


if __name__ == "__main__":
    main()