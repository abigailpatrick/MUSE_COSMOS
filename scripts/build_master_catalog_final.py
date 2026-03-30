import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import astropy.units as u

def _read_table(path):
    """Read a CSV or FITS file into a pandas DataFrame."""
    if path.endswith((".fits", ".fit")):
        df = Table.read(path).to_pandas()
        # FITS stores strings as byte strings — decode any that are
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        return df
    return pd.read_csv(path)


def add_columns(old_table_path, old_col_names, new_col_names, new_table_path):
    """
    Add columns from an old CSV or FITS file into the master catalog CSV,
    matched by ID.

    If a column already exists in the master catalog, values from the old
    table are used to fill any NaNs in that column rather than creating a
    duplicate.

    Parameters
    ----------
    old_table_path : str
        Path to the CSV or FITS file containing the columns to add.
    old_col_names : list of str
        Column names as they appear in the old table.
    new_col_names : list of str
        What to call those columns in the master catalog (same order).
    new_table_path : str
        Path to the master catalog CSV (read in, updated, and saved back).
    """
    old = _read_table(old_table_path)
    new = pd.read_csv(new_table_path)

    # Pull only ID + the columns we want from the old table
    old_subset = old[["ID"] + old_col_names].copy()

    # Rename to the new column headings
    rename_map = dict(zip(old_col_names, new_col_names))
    old_subset = old_subset.rename(columns=rename_map)

    # Split into columns that already exist in the catalog and those that don't
    existing_cols = [c for c in new_col_names if c in new.columns]
    new_cols     = [c for c in new_col_names if c not in new.columns]

    # For genuinely new columns: standard merge
    if new_cols:
        new = new.merge(old_subset[["ID"] + new_cols], on="ID", how="left")

    # For already-existing columns: merge with suffixes then fill gaps
    if existing_cols:
        new = new.merge(old_subset[["ID"] + existing_cols], on="ID", how="left", suffixes=("", "_incoming"))
        for col in existing_cols:
            incoming = col + "_incoming"
            new[col] = new[col].fillna(new[incoming])
            new.drop(columns=incoming, inplace=True)

    new.to_csv(new_table_path, index=False)
    print(f"Added/filled columns {new_col_names} to {new_table_path}")
    return new
def flux_to_luminosity(flux, redshift):
    """
    Convert flux to luminosity given redshift.

    Inputs
        flux     : Integrated line flux (1e-20 erg/s/cm^2).
        redshift : Redshift.

    Output
        Luminosity in erg/s.
    """
    flux = np.asarray(flux, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if flux.size == 0 or redshift.size == 0:
        return np.array([], dtype=float)

    flux_cgs = flux * 1e-20
    d_l = cosmo.luminosity_distance(redshift).to(u.cm).value
    return flux_cgs * 4 * np.pi * d_l**2


def flux_err_to_luminosity_err(flux_err, redshift):
    """
    Convert flux error to luminosity error given redshift.

    Inputs
        flux_err : Flux error (1e-20 erg/s/cm^2).
        redshift : Redshift.

    Output
        Luminosity error in erg/s.
    """
    flux_err = np.asarray(flux_err, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if flux_err.size == 0 or redshift.size == 0:
        return np.array([], dtype=float)

    out = np.full_like(flux_err, np.nan, dtype=float)
    m = np.isfinite(flux_err) & np.isfinite(redshift)
    if not np.any(m):
        return out if out.shape != () else np.nan

    flux_err_cgs = flux_err[m] * 1e-20
    d_l = cosmo.luminosity_distance(redshift[m]).to(u.cm).value
    out[m] = flux_err_cgs * 4 * np.pi * d_l**2
    return out if out.shape != () else float(out)

def remove_columns(col_names, table_path):
    """
    Remove columns from the master catalog CSV.

    Parameters
    ----------
    col_names : list of str
        Column names to remove from the catalog.
    table_path : str
        Path to the master catalog CSV (read in, updated, and saved back).
    """
    catalog = pd.read_csv(table_path)

    missing = [c for c in col_names if c not in catalog.columns]
    if missing:
        print(f"Warning: columns not found in catalog and will be skipped: {missing}")

    to_drop = [c for c in col_names if c in catalog.columns]
    if to_drop:
        catalog.drop(columns=to_drop, inplace=True)
        catalog.to_csv(table_path, index=False)
        print(f"Removed columns {to_drop} from {table_path}")
    
    return catalog


def main():
    output_path = "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_sample.csv"

    # -------------------------------------------------------------------------
    # Source IDs
    # -------------------------------------------------------------------------
    ids = [1988,2144,2962,3221,3228,3297,4126,4469,4873,4970,5049,5120,5364,5446,5849,5887,6456,6906,6938,7479,7650,7654,7902,3535,4419,4820,6680]


    # -------------------------------------------------------------------------
    # Build catalog
    # -------------------------------------------------------------------------
    catalog = pd.DataFrame({"ID": ids})
    catalog.to_csv(output_path, index=False)

    # -------------------------------------------------------------------------
    # Add columns from other tables
    # -------------------------------------------------------------------------
    jels_cols_old = ["ra", "dec", "z1_median_1", "F_Ha_used", "F_Ha_used_err", "L_halpha_uncorr", "L_halpha_err_uncorr", "L_halpha_apcorr", "L_halpha_err_apcorr", "L_halpha_corr_v1", "L_halpha_err_corr_v1"]
    jels_cols_new = ["ra", "dec", "z_jels", "ha_flux", "ha_flux_err", "ha_lumin_uncorr", "ha_lumin_err_uncorr", "ha_lumin_apcorr", "ha_lumin_err_apcorr", "ha_lumin_fullcorr", "ha_lumin_err_fullcorr"]

    catalog = add_columns(
        old_table_path="/home/apatrick/P1/JELSDP/F466N_with_LHa_Corey_method.fits",
        old_col_names=jels_cols_old,
        new_col_names=jels_cols_new,
        new_table_path=output_path,
    )
    catalog = add_columns(
        old_table_path="/home/apatrick/P1/JELSDP/F470N_with_LHa_Corey_method.fits",
        old_col_names=jels_cols_old,
        new_col_names=jels_cols_new,
        new_table_path=output_path,
    )

    muse_cols_old = ["band", "z_used", "flux_fit", "flux_fit_err", "flux_upper_limit"]
    muse_cols_new = ["band", "z_muse", "lya_flux", "lya_flux_err", "lya_flux_upper_limit"]
    catalog = add_columns(
        old_table_path="/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv",
        old_col_names=muse_cols_old,
        new_col_names=muse_cols_new,
        new_table_path=output_path,
    )
    
    muse2_cols_old = ["peak_snr", "lya_detect_flag", "pair"]
    muse2_cols_new = ["lya_peak_snr", "lya_detect_flag", "merge_pair"]
    catalog = add_columns(
        old_table_path="/ceph/cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv",
        old_col_names=muse2_cols_old,
        new_col_names=muse2_cols_new,
        new_table_path=output_path,
    )

    jels_props_old = ["Av_50", "Av_16", "Av_84", "beta", "beta_err", "M_UV_AB_uncorr", "M_UV_AB_err_uncorr", "sSFR_10Myr_50", "sSFR_10Myr_16", "sSFR_10Myr_84", "sSFR_100Myr_50", "sSFR_100Myr_16", "sSFR_100Myr_84", "M_star_50", "M_star_16", "M_star_84"]
    jels_props_new = ["av_50", "av_16", "av_84", "beta", "beta_err", "m_uv_ab_uncorr", "m_uv_ab_err_uncorr", "ssfr_10myr_50", "ssfr_10myr_16", "ssfr_10myr_84", "ssfr_100myr_50", "ssfr_100myr_16", "ssfr_100myr_84", "m_star_50", "m_star_16", "m_star_84"]

    catalog = add_columns(
        old_table_path="/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits",
        old_col_names=jels_props_old,
        new_col_names=jels_props_new,
        new_table_path=output_path,
    )
    catalog = add_columns(
        old_table_path="/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits",
        old_col_names=jels_props_old,
        new_col_names=jels_props_new,
        new_table_path=output_path,
    )
    ew_cats_old = ["ew_muv", "ew_muv_err", "ew_f115_err", "ew_f150", "ew_f150_err"]
    ew_cats_new = ["ew_muv", "ew_muv_err", "ew_f115_err", "ew_f150", "ew_f150_err"]

    catalog = add_columns(
        old_table_path="/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv",
        old_col_names=ew_cats_old,
        new_col_names=ew_cats_new,
        new_table_path=output_path,
    )

    # -------------------------------------------------------------------------
    # Add other properties 
    #-------------------------------------------------------------------------
    
    # AGN Flag
    AGN_IDS = [1988, 5887]
    catalog["agn_flag"] = catalog["ID"].apply(lambda x: 1 if x in AGN_IDS else 0)
    print("Added AGN flag based on known IDs.")
  

    # Calculate E(b-v) from Av
    R_V = 4.05  # Calzetti 
    catalog["ebv"] = catalog["av_50"] / R_V
    print("Calculated E(B-V) from Av using R_V =", R_V)

    # Calculate Lyα luminosity from flux and redshift
    catalog["lya_lumin"] = flux_to_luminosity(catalog["lya_flux"], catalog["z_muse"])
    catalog["lya_lumin_err"] = flux_err_to_luminosity_err(catalog["lya_flux_err"], catalog["z_muse"])
    print("Calculated Lyα luminosity and its error from flux and redshift.")

    # Calculate Lyα luminosity upper limit from upper limit flux and redshift
    catalog["lya_lumin_upper_limit"] = flux_to_luminosity(catalog["lya_flux_upper_limit"], catalog["z_muse"])
    print("Calculated Lyα luminosity upper limit from upper limit flux and redshift.")

    #catalog = remove_columns(
    #col_names=["ew_f115_lower"],
    #table_path=output_path,
    #)


    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    catalog.to_csv(output_path, index=False)
    print(f"Saved catalog with {len(catalog)} sources to {output_path}")

    # -------------------------------------------------------------------------
    # Make secondary catalog which merges the 'merger'sources into on row each
    # -------------------------------------------------------------------------
    merger_1 = catalog[catalog["merge_pair"] == 1]
    merger_2 = catalog[catalog["merge_pair"] == 2]
    merger_3 = catalog[catalog["merge_pair"] == 3]

    merger_1_IDs = merger_1["ID"].values
    merger_2_IDs = merger_2["ID"].values
    merger_3_IDs = merger_3["ID"].values
    print (f"Merger pairs IDS: {merger_1_IDs} & {merger_2_IDs}, and {merger_3_IDs}")

if __name__ == "__main__":
    main()