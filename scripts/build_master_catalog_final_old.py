import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import astropy.units as u
from astropy.io import fits as astrofits

# ---- constants copied from ew script so the merged EW calc is self-contained ----
c_ang = 2.99792458e18        # speed of light in Å/s
lya_rest = 1215.67           # Lyα rest wavelength (Å)
lambda_f150 = 15000.0        # F150W central wavelength (Å)
beta_set = -2.0              # fixed beta used in ew script for band-based continuum

def _read_table(path):
    if path.endswith((".fits", ".fit")):
        df = Table.read(path).to_pandas()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        return df
    return pd.read_csv(path)


def add_columns(old_table_path, old_col_names, new_col_names, new_table_path):
    old = _read_table(old_table_path)
    new = pd.read_csv(new_table_path)
    old_subset = old[["ID"] + old_col_names].copy()
    rename_map = dict(zip(old_col_names, new_col_names))
    old_subset = old_subset.rename(columns=rename_map)
    existing_cols = [c for c in new_col_names if c in new.columns]
    new_cols     = [c for c in new_col_names if c not in new.columns]
    if new_cols:
        new = new.merge(old_subset[["ID"] + new_cols], on="ID", how="left")
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
    flux = np.asarray(flux, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if flux.size == 0 or redshift.size == 0:
        return np.array([], dtype=float)
    flux_cgs = flux * 1e-20
    d_l = cosmo.luminosity_distance(redshift).to(u.cm).value
    return flux_cgs * 4 * np.pi * d_l**2


def flux_err_to_luminosity_err(flux_err, redshift):
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


# =============================================================================
# NEW: EW helper — recompute ew_f150 from scratch given fluxes and redshift.
# This mirrors the exact logic in lya_ew.py so the merged rows are consistent.
# =============================================================================
def compute_ew_f150(lya_flux_cgs, lya_flux_err_cgs, fnu_150_Jy, fnu_150_err_Jy, z):
    """
    Recompute rest-frame Lyα EW using F150W continuum.

    Parameters
    ----------
    lya_flux_cgs     : Lyα line flux in erg/s/cm² (already in cgs, i.e. *1e-20 applied)
    lya_flux_err_cgs : error on above
    fnu_150_Jy       : F150W flux density in units of 1e-29 erg/s/cm²/Hz  (nJy effectively)
    fnu_150_err_Jy   : error on above
    z                : redshift

    Returns
    -------
    ew_f150, ew_f150_err  (rest-frame Å), or (nan, nan) if inputs invalid
    """
    if not np.isfinite([lya_flux_cgs, fnu_150_Jy, z]).all():
        return np.nan, np.nan

    # observed Lyα wavelength
    lambda_lya_obs = lya_rest * (1 + z)

    # convert F150W fnu → flambda at observed F150W wavelength, then
    # scale with beta to Lyα observed wavelength (matches ew script exactly)
    fnu_cgs   = fnu_150_Jy  * 1e-29          # → erg/s/cm²/Hz
    fnu_e_cgs = fnu_150_err_Jy * 1e-29

    flambda_150     = fnu_cgs   * c_ang / lambda_f150**2
    flambda_150_err = fnu_e_cgs * c_ang / lambda_f150**2

    fcont     = flambda_150     * (lambda_lya_obs / lambda_f150) ** beta_set
    fcont_err = flambda_150_err * (lambda_lya_obs / lambda_f150) ** beta_set

    if fcont <= 0 or not np.isfinite(fcont):
        return np.nan, np.nan

    # observed EW then rest-frame
    ew_obs  = lya_flux_cgs / fcont
    ew_err  = ew_obs * np.sqrt((lya_flux_err_cgs / lya_flux_cgs)**2
                               + (fcont_err / fcont)**2)
    return ew_obs / (1 + z), ew_err / (1 + z)


# =============================================================================
# NEW: Build one merged row for a pair of sources.
#
# What this does:
#   1. Takes the row of the KEPT source as the base — all SED properties
#      (beta, Mstar, sSFR, Av, MUV, ra, dec, z, lya_*, agn_flag, ebv, etc.)
#      come directly from this row unchanged.
#   2. Ha luminosity: sums ha_lumin_fullcorr from both sources (total flux in
#      the MUSE aperture).  Error propagated in quadrature.
#      All other ha_* columns set to NaN (uncorr / apcorr would be
#      internally inconsistent for a merged system).
#   3. EW: recomputed using summed F150W flux (both sources) as continuum
#      and the kept source's Lyα flux.  ew_muv, ew_f115* set to NaN.
# =============================================================================
def build_merged_row(row_keep, row_drop):
    """
    Parameters
    ----------
    row_keep : pd.Series  — row of the source whose single-source values we keep
    row_drop : pd.Series  — row of the other source in the pair

    Returns
    -------
    pd.Series with the merged values
    """
    merged = row_keep.copy()   # start from the kept source

    # ------------------------------------------------------------------
    # 1. Ha luminosity: sum fullcorr only; NaN everything else
    # ------------------------------------------------------------------
    ha_nan_cols = [
        "ha_flux", "ha_flux_err",
        "ha_lumin_uncorr", "ha_lumin_err_uncorr",
        "ha_lumin_apcorr", "ha_lumin_err_apcorr",
    ]
    for col in ha_nan_cols:
        merged[col] = np.nan

    # Sum the fully-corrected Ha luminosities
    # If either is NaN we still want a sensible result, so use nansum
    # but flag if BOTH are NaN
    L1 = row_keep["ha_lumin_fullcorr"]
    L2 = row_drop["ha_lumin_fullcorr"]
    e1 = row_keep["ha_lumin_err_fullcorr"]
    e2 = row_drop["ha_lumin_err_fullcorr"]

    if np.isfinite(L1) or np.isfinite(L2):
        merged["ha_lumin_fullcorr"] = np.nansum([L1, L2])
        # quadrature error propagation: σ_tot = sqrt(σ1² + σ2²)
        errs = [e for e in [e1, e2] if np.isfinite(e)]
        merged["ha_lumin_err_fullcorr"] = np.sqrt(np.sum(np.array(errs)**2)) if errs else np.nan
    else:
        merged["ha_lumin_fullcorr"]     = np.nan
        merged["ha_lumin_err_fullcorr"] = np.nan

    print(f"  Ha fullcorr: {L1:.3e} + {L2:.3e} = {merged['ha_lumin_fullcorr']:.3e}")

    # ------------------------------------------------------------------
    # 2. EW: recompute with summed F150W continuum, kept Lyα flux
    #    NaN out muv and f115 EWs — they can't be cleanly combined
    # ------------------------------------------------------------------
    for col in ["ew_muv", "ew_muv_err", "ew_f115", "ew_f115_err", "ew_f115_lower"]:
        merged[col] = np.nan

    # Sum F150W fluxes (units: 1e-29 erg/s/cm²/Hz as in ew script)
    f150_sum     = np.nansum([row_keep["F150W_auto_flux"],  row_drop["F150W_auto_flux"]])
    f150_err_sum = np.sqrt(np.nansum([row_keep["F150W_auto_fluxerr"]**2,
                                      row_drop["F150W_auto_fluxerr"]**2]))

    # Lyα flux from kept source (convert from 1e-20 cgs units used in catalog)
    lya_f   = row_keep["lya_flux"]   * 1e-20   # → erg/s/cm²
    lya_err = row_keep["lya_flux_err"] * 1e-20

    z = row_keep["z_muse"]

    ew, ew_err = compute_ew_f150(lya_f, lya_err, f150_sum, f150_err_sum, z)
    merged["ew_f150"]     = ew
    merged["ew_f150_err"] = ew_err

    print(f"  EW f150 recomputed: {ew:.1f} ± {ew_err:.1f} Å (rest)")

    return merged


def main():
    output_path = "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_sample.csv"

    ids = [1988,2144,2962,3221,3228,3297,4126,4469,4873,4970,5049,5120,5364,5446,5849,5887,6456,6906,6938,7479,7650,7654,7902,3535,4419,4820,6680]

    catalog = pd.DataFrame({"ID": ids})
    catalog.to_csv(output_path, index=False)

    jels_cols_old = ["ra", "dec", "z1_median_1", "F_Ha_used", "F_Ha_used_err", "L_halpha_uncorr", "L_halpha_err_uncorr", "L_halpha_apcorr", "L_halpha_err_apcorr", "L_halpha_corr_v1", "L_halpha_err_corr_v1"]
    jels_cols_new = ["ra", "dec", "z_jels", "ha_flux", "ha_flux_err", "ha_lumin_uncorr", "ha_lumin_err_uncorr", "ha_lumin_apcorr", "ha_lumin_err_apcorr", "ha_lumin_fullcorr", "ha_lumin_err_fullcorr"]
    catalog = add_columns("/home/apatrick/P1/JELSDP/F466N_with_LHa_Corey_method.fits", jels_cols_old, jels_cols_new, output_path)
    catalog = add_columns("/home/apatrick/P1/JELSDP/F470N_with_LHa_Corey_method.fits", jels_cols_old, jels_cols_new, output_path)

    muse_cols_old = ["band", "z_used", "flux_fit", "flux_fit_err", "flux_upper_limit"]
    muse_cols_new = ["band", "z_muse", "lya_flux", "lya_flux_err", "lya_flux_upper_limit"]
    catalog = add_columns("/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv", muse_cols_old, muse_cols_new, output_path)

    muse2_cols_old = ["peak_snr", "lya_detect_flag", "pair"]
    muse2_cols_new = ["lya_peak_snr", "lya_detect_flag", "merge_pair"]
    catalog = add_columns("/ceph/cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv", muse2_cols_old, muse2_cols_new, output_path)

    jels_props_old = ["Av_50", "Av_16", "Av_84", "beta", "beta_err", "M_UV_AB_uncorr", "M_UV_AB_err_uncorr", "sSFR_10Myr_50", "sSFR_10Myr_16", "sSFR_10Myr_84", "sSFR_100Myr_50", "sSFR_100Myr_16", "sSFR_100Myr_84", "M_star_50", "M_star_16", "M_star_84"]
    jels_props_new = ["av_50", "av_16", "av_84", "beta", "beta_err", "m_uv_ab_uncorr", "m_uv_ab_err_uncorr", "ssfr_10myr_50", "ssfr_10myr_16", "ssfr_10myr_84", "ssfr_100myr_50", "ssfr_100myr_16", "ssfr_100myr_84", "m_star_50", "m_star_16", "m_star_84"]
    catalog = add_columns("/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits", jels_props_old, jels_props_new, output_path)
    catalog = add_columns("/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits", jels_props_old, jels_props_new, output_path)

    ew_cats_old = ["ew_muv", "ew_muv_err", "ew_f115_err", "ew_f150", "ew_f150_err"]
    ew_cats_new = ["ew_muv", "ew_muv_err", "ew_f115_err", "ew_f150", "ew_f150_err"]
    catalog = add_columns("/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv", ew_cats_old, ew_cats_new, output_path)

    # AGN Flag
    AGN_IDS = [1988, 5887]
    catalog["agn_flag"] = catalog["ID"].apply(lambda x: 1 if x in AGN_IDS else 0)

    # E(B-V) from Av
    R_V = 4.05
    catalog["ebv"] = catalog["av_50"] / R_V

    # Lyα luminosity
    catalog["lya_lumin"] = flux_to_luminosity(catalog["lya_flux"], catalog["z_muse"])
    catalog["lya_lumin_err"] = flux_err_to_luminosity_err(catalog["lya_flux_err"], catalog["z_muse"])
    catalog["lya_lumin_upper_limit"] = flux_to_luminosity(catalog["lya_flux_upper_limit"], catalog["z_muse"])

    catalog.to_csv(output_path, index=False)
    print(f"Saved catalog with {len(catalog)} sources to {output_path}")

    # =========================================================================
    # NEW BLOCK: Build final_merged catalog
    #
    # Three unresolved MUSE pairs that are resolved in JWST.
    # We need one row per pair in the merged catalog.
    #
    # IDS_KEEP: pick one ID from each pair — the one whose SED/Lya values
    # you want to represent the merged system.  Fill these in yourself.
    # IDS_DROP: the other member of each pair (derived automatically below).
    # =========================================================================

    # --- The IDs out of the pairs to keep the core lya values for ---
    IDS_KEEP = [3221, 6938, 7654]

    # All six merger IDs across the three pairs
    MERGER_PAIRS = [
        (3221, 3228),
        (6906, 6938),
        (7650, 7654),
    ]

    # Derive the dropped ID for each pair from IDS_KEEP
    # (whichever member of the pair is not in IDS_KEEP)
    pairs_keep_drop = []
    for pair in MERGER_PAIRS:
        keep = [i for i in pair if i in IDS_KEEP]
        drop = [i for i in pair if i not in IDS_KEEP]
        if len(keep) != 1 or len(drop) != 1:
            raise ValueError(
                f"Pair {pair}: expected exactly one ID in IDS_KEEP, "
                f"got keep={keep}, drop={drop}. Check IDS_KEEP."
            )
        pairs_keep_drop.append((keep[0], drop[0]))



    def load_f150w(fits_path):
        """Return a small DataFrame with ID, F150W_auto_flux, F150W_auto_fluxerr."""
        t = Table.read(fits_path).to_pandas()
        for col in t.select_dtypes(include="object").columns:
            t[col] = t[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        return t[["ID", "F150W_auto_flux", "F150W_auto_fluxerr"]]

    f150_466 = load_f150w("/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits")
    f150_470 = load_f150w("/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits")
    # Combine: same fill-NaN logic as add_columns (466 first, 470 fills gaps)
    f150_all = pd.concat([f150_466, f150_470], ignore_index=True).drop_duplicates(subset="ID", keep="first")

    # Merge F150W into catalog (temporary columns, not saved to final_sample.csv)
    catalog_with_f150 = catalog.merge(f150_all, on="ID", how="left")

    # -------------------------------------------------------------------------
    # Build the merged rows
    # -------------------------------------------------------------------------
    all_merger_ids = [i for pair in MERGER_PAIRS for i in pair]
    # Start merged catalog from all NON-merger sources
    non_merger = catalog_with_f150[~catalog_with_f150["ID"].isin(all_merger_ids)].copy()

    merged_rows = []
    for id_keep, id_drop in pairs_keep_drop:
        row_keep = catalog_with_f150[catalog_with_f150["ID"] == id_keep].iloc[0]
        row_drop = catalog_with_f150[catalog_with_f150["ID"] == id_drop].iloc[0]
        print(f"\nMerging pair: KEEP={id_keep}, DROP={id_drop}")
        merged_row = build_merged_row(row_keep, row_drop)
        merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)

    # Drop the temporary F150W columns before saving
    # (they weren't in the original catalog schema)
    for col in ["F150W_auto_flux", "F150W_auto_fluxerr"]:
        if col in non_merger.columns:
            non_merger = non_merger.drop(columns=[col])
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])

    final_merged = pd.concat([non_merger, merged_df], ignore_index=True)

    merged_path = output_path.replace("final_sample.csv", "final_merged.csv")
    final_merged.to_csv(merged_path, index=False)
    print(f"\nSaved merged catalog with {len(final_merged)} sources to {merged_path}")
    print(f"  ({len(catalog)} original → removed {len(all_merger_ids)} merger IDs → added {len(merged_rows)} merged rows)")

    # Merger pair summary
    merger_summary = catalog[catalog["ID"].isin(all_merger_ids)][
        ["ID", "merge_pair", "ha_lumin_fullcorr", "lya_lumin", "ew_f150"]
    ]
    print("\nMerger pair input values:")
    print(merger_summary.to_string(index=False))


if __name__ == "__main__":
    main()