import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


def load_primer_catalog(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        ra  = np.array(data["ALPHA_J2000"], dtype=float)
        dec = np.array(data["DELTA_J2000"], dtype=float)
        print(f" There are {len(ra)} sources in the PRIMER catalog.")
    return ra, dec, data


def load_jels_catalog(csv_file):
    df = pd.read_csv(csv_file).reset_index(drop=True)
    # Sanity check column names
    print("JELS columns:", df.columns.tolist())
    return df


def make_coords(ra, dec):
    return SkyCoord(ra=np.asarray(ra, dtype=float) * u.deg,
                    dec=np.asarray(dec, dtype=float) * u.deg)


def crossmatch(primer_coords, jels_coords, radius_arcsec=3.0):
    """
    search_around_sky returns ALL pairs within the radius.
    idx_jels[i], idx_primer[i] are matched pair i.
    """
    idx_jels, idx_primer, d2d, _ = primer_coords.search_around_sky(
        jels_coords, radius_arcsec * u.arcsec
    )
    return idx_primer, idx_jels, d2d


def build_output(primer_data, jels_df, idx_primer, idx_jels, d2d):
    """
    Build a flat table with one row per match pair.
    This way nothing is lost — you can always group later.
    """
    keep_primer = ["ALPHA_J2000", "DELTA_J2000", "z1_median"]

    rows = []
    for p_idx, j_idx, dist in zip(idx_primer, idx_jels, d2d):
        p_idx = int(p_idx)
        j_idx = int(j_idx)
        row = {}
        # JELS source info
        row["jels_id"]       = jels_df.iloc[j_idx]["ID"]
        row["jels_ra"]       = jels_df.iloc[j_idx]["ra_jels"]
        row["jels_dec"]      = jels_df.iloc[j_idx]["dec_jels"]
        # PRIMER source info
        for col in keep_primer:
            row[f"primer_{col}"] = primer_data[p_idx][col]
        row["primer_idx"]        = p_idx
        row["sep_arcsec"]        = round(float(dist.arcsec), 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["jels_id", "sep_arcsec"]).reset_index(drop=True)
    return df


def summarise_matches(df):
    """Print a quick summary of matches per JELS source."""
    print("\n--- Matches per JELS source ---")
    for jels_id, grp in df.groupby("jels_id"):
        print(f"  JELS {jels_id}: {len(grp)} PRIMER match(es)")
        for _, row in grp.iterrows():
            print(f"    PRIMER idx {int(row['primer_idx'])}  "
                  f"sep={row['sep_arcsec']}\"  "
                  f"z={row['primer_z1_median']:.3f}")
    print()


def main():
    primer_file   = "/home/apatrick/P1/JELSDP/primer_f356w_sed_fitted_catalogue.fits"
    jels_file     = "/cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv"
    output_file   = "primer_jels_crossmatch.csv"
    radius_arcsec = 3.0

    primer_ra, primer_dec, primer_data = load_primer_catalog(primer_file)
    jels_df = load_jels_catalog(jels_file)

    print(f"Loaded {len(primer_ra)} PRIMER sources")
    print(f"Loaded {len(jels_df)} JELS-MUSE sources")

    primer_coords = make_coords(primer_ra, primer_dec)
    jels_coords   = make_coords(jels_df["ra_jels"], jels_df["dec_jels"])

    idx_primer, idx_jels, d2d = crossmatch(primer_coords, jels_coords,
                                            radius_arcsec=radius_arcsec)

    print(f"Found {len(idx_primer)} total pairwise matches within {radius_arcsec}\"")

    if len(idx_primer) == 0:
        print("No matches found. Check coordinate columns and search radius.")
        return

    output_df = build_output(primer_data, jels_df, idx_primer, idx_jels, d2d)
    summarise_matches(output_df)

    output_df.to_csv(output_file, index=False)
    print(f"Saved {len(output_df)} match pairs to {output_file}")


if __name__ == "__main__":
    main()