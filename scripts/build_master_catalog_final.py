"""
build_catalog.py
================
Builds two output catalogs from JELS + MUSE source data:

  final_sample_full.csv  —  one row per JWST source (all 27 IDs), no merging.
                            Use this for individual-source science.

  final_merged.csv       —  unresolved MUSE pairs collapsed to one row each.
                            Three pairs (6 IDs → 3 rows), giving 24 rows total.
                            Use this for Lyα / Hα / EW / fesc science where the
                            MUSE aperture cannot distinguish the two galaxies.
  
  final_merged_noagn.csv  — Same as final_merged but with the two AGN (IDs 1988 and 5887) removed.
                            Use this for analyses where AGN contamination is a concern.
  

Column schema (both catalogs):
  Identifiers  : ID, ra, dec
  Redshifts    : z_jels, z_muse
  Hα           : ha_flux, ha_flux_err,
                 ha_lumin_uncorr,  ha_lumin_err_uncorr,
                 ha_lumin_apcorr,  ha_lumin_err_apcorr,
                 ha_lumin_fullcorr, ha_lumin_err_fullcorr
  MUSE Lyα     : band, lya_flux, lya_flux_err, lya_flux_upper_limit,
                 lya_peak_snr, lya_detect_flag, merge_pair
  Lyα lumin    : lya_lumin, lya_lumin_err, lya_lumin_upper_limit
  SED (BAGPIPES): av_50, av_16, av_84, ebv,
                  beta, beta_err,
                  m_uv_ab_uncorr, m_uv_ab_err_uncorr,
                  ssfr_10myr_50, ssfr_10myr_16, ssfr_10myr_84,
                  ssfr_100myr_50, ssfr_100myr_16, ssfr_100myr_84,
                  m_star_50, m_star_16, m_star_84
  EW           : ew_f150, ew_f150_err, ew_muv, ew_muv_err
  fesc         : fesc_lya_dustcorr, fesc_lya_dustcorr_err, fesc_lya_dustcorr_ul
  Flags        : agn_flag
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_ANG        = 2.99792458e18   # speed of light in Å/s
LYA_REST     = 1215.67         # Lyα rest wavelength (Å)
LAMBDA_F150  = 15000.0         # F150W central wavelength (Å)
BETA_CONT    = -2.14            # UV slope assumed when computing continuum from F150W
R_V          = 4.05            # Calzetti R_V for E(B-V) from Av
CASEB        = 8.7             # Case B Lyα/Hα ratio (intrinsic)

# ---------------------------------------------------------------------------
# Source IDs
# ---------------------------------------------------------------------------
ALL_IDS = [
    1988, 2144, 2962, 3221, 3228, 3297, 4126, 4469, 4873, 4970,
    5049, 5120, 5364, 5446, 5849, 5887, 6456, 6906, 6938, 7479,
    7650, 7654, 7902, 3535, 4419, 4820, 6680,
]
AGN_IDS = [1988, 5887]

# Unresolved MUSE pairs: (id_keep, id_drop)
# id_keep  → source whose Lyα flux / redshift / SED we adopt for the merged row
# id_drop  → source whose Hα luminosity is added to id_keep's
MERGER_PAIRS = [
    (3221, 3228),
    (6906, 6938),   # NOTE: keep=6906, drop=6938
    (7650, 7654),
]

# DJA matched sources and their zspecs full infor store din jels_dja_specz_matches.csv
zspec_map = {
        4873: 6.0864,
        4970: 6.0673,
        5049: 6.0863,
}

# ---------------------------------------------------------------------------
# File paths  — edit these to match your environment
# ---------------------------------------------------------------------------
eta = "eta_1"  # "eta_1" or "eta_2p27"

PATHS = {
    "jels_466_ha"    : f"/home/apatrick/P1/JELSDP/F466N_with_LHa_{eta}.fits",
    "jels_470_ha"    : f"/home/apatrick/P1/JELSDP/F470N_with_LHa_{eta}.fits",
    "jels_466_cat"   : "/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits",
    "jels_470_cat"   : "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits",
    "muse_lya"       : "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv",
    "muse_flags"     : "/ceph/cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv",
    "ew_results"     : "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv",
    "out_full"       : f"/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_sample_full_{eta}.csv",
    "out_merged"     : f"/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged_{eta}.csv",
    "out_merged_noagn"  : f"/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged_noagn_{eta}.csv"

}

# Final column order for both output catalogs
FINAL_COLS = [
    "ID", "ra", "dec",
    "z_jels", "z_muse", "z_spec",
    "ha_flux", "ha_flux_err",
    "ha_lumin_uncorr", "ha_lumin_err_uncorr",
    "ha_lumin_apcorr", "ha_lumin_err_apcorr",
    "ha_lumin_fullcorr", "ha_lumin_err_fullcorr",
    "band", "lya_flux", "lya_flux_err", "lya_flux_upper_limit", "lya_skew",
    "lya_peak_snr", "lya_detect_flag", "merge_pair",
    "lya_lumin", "lya_lumin_err", "lya_lumin_upper_limit",
    "av_50", "av_16", "av_84", "ebv",
    "beta", "beta_err",
    "m_uv_ab_uncorr", "m_uv_ab_err_uncorr",
    "ssfr_10myr_50", "ssfr_10myr_16", "ssfr_10myr_84",
    "ssfr_100myr_50", "ssfr_100myr_16", "ssfr_100myr_84",
    "m_star_50", "m_star_16", "m_star_84",
    "ew_f150", "ew_f150_err",
    "ew_muv", "ew_muv_err",
    "fesc_lya_dustcorr", "fesc_lya_dustcorr_err", "fesc_lya_dustcorr_ul",
    "agn_flag",
]


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_fits(path, cols):
    """Read selected columns from a FITS catalog into a DataFrame."""
    t = Table.read(path).to_pandas()
    for c in t.select_dtypes(include="object").columns:
        t[c] = t[c].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    return t[["ID"] + [c for c in cols if c in t.columns]]


def read_csv(path, cols):
    t = pd.read_csv(path)
    return t[["ID"] + [c for c in cols if c in t.columns]]


def left_merge(base, other, on="ID"):
    """Left-merge other into base, filling NaNs in existing columns."""
    shared = [c for c in other.columns if c != on and c in base.columns]
    new    = [c for c in other.columns if c != on and c not in base.columns]

    if new:
        base = base.merge(other[["ID"] + new], on=on, how="left")
    if shared:
        base = base.merge(other[["ID"] + shared], on=on, how="left", suffixes=("", "_in"))
        for c in shared:
            base[c] = base[c].fillna(base[c + "_in"])
            base.drop(columns=c + "_in", inplace=True)
    return base


# ===========================================================================
# Physics helpers
# ===========================================================================

def lya_luminosity(flux_1e20, z):
    """Convert Lyα flux (units of 1e-20 erg/s/cm²) → luminosity (erg/s)."""
    flux = np.asarray(flux_1e20, dtype=float) * 1e-20
    d_l  = cosmo.luminosity_distance(np.asarray(z, dtype=float)).to(u.cm).value
    return flux * 4 * np.pi * d_l**2


def lya_luminosity_err(flux_err_1e20, z):
    out = np.full(len(flux_err_1e20), np.nan)
    m   = np.isfinite(flux_err_1e20) & np.isfinite(z)
    d_l = cosmo.luminosity_distance(z[m]).to(u.cm).value
    out[m] = flux_err_1e20[m] * 1e-20 * 4 * np.pi * d_l**2
    return out


def compute_ew_f150(lya_flux_cgs, lya_flux_err_cgs, fnu_150_nJy, fnu_150_err_nJy, z):
    """
    Rest-frame Lyα EW from F150W continuum.

    lya_flux_cgs   : Lyα flux in erg/s/cm²
    fnu_150_nJy    : F150W flux density in nJy (= 1e-29 erg/s/cm²/Hz internally)
    Returns (ew_rest_Å, ew_err_rest_Å) or (nan, nan) if inputs invalid.
    """
    if not np.isfinite([lya_flux_cgs, fnu_150_nJy, z]).all():
        return np.nan, np.nan

    lya_obs     = LYA_REST * (1 + z)
    fnu_cgs     = fnu_150_nJy     * 1e-29
    fnu_err_cgs = fnu_150_err_nJy * 1e-29

    # f_lambda at F150W pivot, then power-law extrapolate to Lyα observed wavelength
    fl_150     = fnu_cgs     * C_ANG / LAMBDA_F150**2
    fl_150_err = fnu_err_cgs * C_ANG / LAMBDA_F150**2
    fcont      = fl_150     * (lya_obs / LAMBDA_F150) ** BETA_CONT
    fcont_err  = fl_150_err * (lya_obs / LAMBDA_F150) ** BETA_CONT

    if fcont <= 0 or not np.isfinite(fcont):
        return np.nan, np.nan

    ew_obs = lya_flux_cgs / fcont
    ew_err = ew_obs * np.sqrt((lya_flux_err_cgs / lya_flux_cgs)**2
                               + (fcont_err / fcont)**2)
    return ew_obs / (1 + z), ew_err / (1 + z)


def compute_fesc(df):
    """
    Add Lyα escape fraction columns using dust-corrected Hα luminosity.
    fesc = L_lya / (caseb * L_ha_fullcorr)
    """
    lya_L     = df["lya_lumin"].values
    lya_L_err = df["lya_lumin_err"].values
    lya_L_ul  = df["lya_lumin_upper_limit"].values
    ha_L      = df["ha_lumin_fullcorr"].values
    ha_L_err  = df["ha_lumin_err_fullcorr"].values

    denom = CASEB * ha_L
    with np.errstate(divide="ignore", invalid="ignore"):
        fesc    = np.where(denom > 0, lya_L    / denom, np.nan)
        fesc_ul = np.where(denom > 0, lya_L_ul / denom, np.nan)
        fesc_err = fesc * np.sqrt(
            np.where(lya_L > 0, (lya_L_err / lya_L)**2, 0) +
            np.where(ha_L  > 0, (ha_L_err  / ha_L )**2, 0)
        )

    df = df.copy()
    df["fesc_lya_dustcorr"]     = fesc
    df["fesc_lya_dustcorr_err"] = fesc_err
    df["fesc_lya_dustcorr_ul"]  = fesc_ul
    return df


# ===========================================================================
# SED merger helpers
# ===========================================================================

def sum_linear(v1, v2):
    """Sum two linear quantities with NaN handling."""
    vals = [v for v in [v1, v2] if np.isfinite(v)]
    return np.sum(vals) if vals else np.nan


def quad_err(e1, e2):
    """Quadrature combination of uncertainties."""
    errs = [e for e in [e1, e2] if np.isfinite(e)]
    return np.sqrt(np.sum(np.array(errs) ** 2)) if errs else np.nan


def weighted_avg(v1, v2, w1, w2):
    """
    Hα-luminosity weighted average with NaN handling.
    """

    vals = np.array([v1, v2], dtype=float)
    wgts = np.array([w1, w2], dtype=float)

    mask = np.isfinite(vals) & np.isfinite(wgts) & (wgts > 0)

    if not np.any(mask):
        return np.nan

    return np.average(vals[mask], weights=wgts[mask])


def combine_log_mass(logm1, logm2):
    """
    Combine stellar masses stored as log10(M/Msun).

    Converts to linear space, sums, then converts back to log10.
    """

    vals = [v for v in [logm1, logm2] if np.isfinite(v)]

    if len(vals) == 0:
        return np.nan

    linear_mass = np.sum([10**v for v in vals])

    return np.log10(linear_mass)


def combine_ab_mags_with_err(m1, e1, m2, e2):
    """
    Combine AB magnitudes by summing fluxes.

    Returns
    -------
    mag_total, mag_err_total
    """

    vals = []

    for m, e in [(m1, e1), (m2, e2)]:

        if np.isfinite(m):

            # relative flux normalization
            f = 10**(-0.4 * m)

            if np.isfinite(e):
                ferr = f * (np.log(10) / 2.5) * e
            else:
                ferr = np.nan

            vals.append((f, ferr))

    if len(vals) == 0:
        return np.nan, np.nan

    fluxes = np.array([v[0] for v in vals])
    ferrs  = np.array([v[1] for v in vals])

    f_tot = np.sum(fluxes)

    ferr_tot = np.sqrt(np.nansum(ferrs**2))

    mag_tot = -2.5 * np.log10(f_tot)

    mag_err = (2.5 / np.log(10)) * ferr_tot / f_tot

    return mag_tot, mag_err


def build_merged_row(row_keep, row_drop, f150_lookup):
    """
    Combine a MUSE-unresolved pair into one merged row.

    Strategy
    --------
    Lyα quantities
        → taken from row_keep unchanged.

    Hα luminosities
        → summed.

    SED extensive quantities
        → summed appropriately.

    SED relative quantities
        → Hα-luminosity weighted averages.

    EW
        → recomputed using summed F150W continuum.
    """

    merged = row_keep.copy()

    # ------------------------------------------------------------------
    # Hα luminosity weights for SED averaging
    # ------------------------------------------------------------------

    w1 = row_keep["ha_lumin_fullcorr"]
    w2 = row_drop["ha_lumin_fullcorr"]

    # ------------------------------------------------------------------
    # 1. Hα: sum luminosities
    # ------------------------------------------------------------------

    for col in [
        "ha_flux",
        "ha_flux_err",
        "ha_lumin_uncorr",
        "ha_lumin_err_uncorr",
        "ha_lumin_apcorr",
        "ha_lumin_err_apcorr",
    ]:
        merged[col] = np.nan

    L1 = row_keep["ha_lumin_fullcorr"]
    L2 = row_drop["ha_lumin_fullcorr"]

    e1 = row_keep["ha_lumin_err_fullcorr"]
    e2 = row_drop["ha_lumin_err_fullcorr"]

    merged["ha_lumin_fullcorr"] = sum_linear(L1, L2)
    merged["ha_lumin_err_fullcorr"] = quad_err(e1, e2)

    print(
        f"  Hα fullcorr: "
        f"{L1:.3e} + {L2:.3e} = "
        f"{merged['ha_lumin_fullcorr']:.3e}"
    )

    # ------------------------------------------------------------------
    # 2. Weighted-average SED quantities
    # ------------------------------------------------------------------

    weighted_cols = [

        # Dust attenuation
        "av_50",
        "av_16",
        "av_84",
        "ebv",

        # UV slope
        "beta",
        "beta_err",

        # sSFRs
        "ssfr_10myr_50",
        "ssfr_10myr_16",
        "ssfr_10myr_84",

        "ssfr_100myr_50",
        "ssfr_100myr_16",
        "ssfr_100myr_84",
    ]

    for col in weighted_cols:

        merged[col] = weighted_avg(
            row_keep[col],
            row_drop[col],
            w1,
            w2
        )

    # ------------------------------------------------------------------
    # 3. Stellar masses (stored as log10(M/Msun))
    # ------------------------------------------------------------------

    for col in ["m_star_50", "m_star_16", "m_star_84"]:

        merged[col] = combine_log_mass(
            row_keep[col],
            row_drop[col]
        )

    # ------------------------------------------------------------------
    # 4. UV magnitudes
    # ------------------------------------------------------------------

    (
        merged["m_uv_ab_uncorr"],
        merged["m_uv_ab_err_uncorr"]
    ) = combine_ab_mags_with_err(
        row_keep["m_uv_ab_uncorr"],
        row_keep["m_uv_ab_err_uncorr"],
        row_drop["m_uv_ab_uncorr"],
        row_drop["m_uv_ab_err_uncorr"]
    )

    # ------------------------------------------------------------------
    # 5. EW: recompute using summed F150W continuum
    # ------------------------------------------------------------------

    merged["ew_muv"] = np.nan
    merged["ew_muv_err"] = np.nan

    f150_k = f150_lookup.get(
        row_keep["ID"],
        (np.nan, np.nan)
    )

    f150_d = f150_lookup.get(
        row_drop["ID"],
        (np.nan, np.nan)
    )

    f150_sum = np.nansum([f150_k[0], f150_d[0]])

    f150_err_sum = np.sqrt(
        np.nansum([
            f150_k[1]**2,
            f150_d[1]**2
        ])
    )

    lya_f = row_keep["lya_flux"] * 1e-20
    lya_err = row_keep["lya_flux_err"] * 1e-20

    ew, ew_err = compute_ew_f150(
        lya_f,
        lya_err,
        f150_sum,
        f150_err_sum,
        row_keep["z_muse"]
    )

    merged["ew_f150"] = ew
    merged["ew_f150_err"] = ew_err

    print(
        f"  EW f150 (recomputed): "
        f"{ew:.1f} ± {ew_err:.1f} Å (rest)"
    )

    return merged


# ===========================================================================
# Main
# ===========================================================================

def main():

    # -----------------------------------------------------------------------
    # 1. Start with the list of all IDs
    # -----------------------------------------------------------------------
    cat = pd.DataFrame({"ID": ALL_IDS})

    # -----------------------------------------------------------------------
    # 2. Add Hα fluxes and luminosities from JELS (F466N then F470N)
    # -----------------------------------------------------------------------
    ha_cols_fits = ["ra", "dec", "z1_median_1",
                    "F_Ha_used", "F_Ha_used_err",
                    "L_halpha_uncorr", "L_halpha_err_uncorr",
                    "L_halpha_apcorr", "L_halpha_err_apcorr",
                    "L_halpha_corr", "L_halpha_err_corr"]
    ha_cols_new  = ["ra", "dec", "z_jels",
                    "ha_flux", "ha_flux_err",
                    "ha_lumin_uncorr", "ha_lumin_err_uncorr",
                    "ha_lumin_apcorr", "ha_lumin_err_apcorr",
                    "ha_lumin_fullcorr", "ha_lumin_err_fullcorr"]

    for fits_path in [PATHS["jels_466_ha"], PATHS["jels_470_ha"]]:
        data = read_fits(fits_path, ha_cols_fits).rename(
            columns=dict(zip(ha_cols_fits, ha_cols_new))
        )
        cat = left_merge(cat, data)

    # -----------------------------------------------------------------------
    # 3. Add MUSE Lyα fluxes
    # -----------------------------------------------------------------------
    lya_cols_fits = ["band", "z_used", "flux_fit", "flux_fit_err", "flux_upper_limit", "alpha_skew"]
    lya_cols_new  = ["band", "z_muse", "lya_flux", "lya_flux_err", "lya_flux_upper_limit", "lya_skew"]
    data = read_csv(PATHS["muse_lya"], lya_cols_fits).rename(
        columns=dict(zip(lya_cols_fits, lya_cols_new))
    )
    cat = left_merge(cat, data)

    # -----------------------------------------------------------------------
    # 4. Add MUSE detection flags and pair labels
    # -----------------------------------------------------------------------
    flag_cols = ["peak_snr", "lya_detect_flag", "pair"]
    flag_new  = ["lya_peak_snr", "lya_detect_flag", "merge_pair"]
    data = read_csv(PATHS["muse_flags"], flag_cols).rename(
        columns=dict(zip(flag_cols, flag_new))
    )
    cat = left_merge(cat, data)

# -----------------------------------------------------------------------
    # 5a. Add SED properties from LHa catalogs (Av, beta, sSFR, M_star)
    # -----------------------------------------------------------------------
    sed_cols_fits = ["Av_50", "Av_16", "Av_84",
                     "sSFR_10Myr_50", "sSFR_10Myr_16", "sSFR_10Myr_84",
                     "sSFR_100Myr_50", "sSFR_100Myr_16", "sSFR_100Myr_84",
                     "M_star_50", "M_star_16", "M_star_84"]
    sed_cols_new  = ["av_50", "av_16", "av_84",
                     "ssfr_10myr_50", "ssfr_10myr_16", "ssfr_10myr_84",
                     "ssfr_100myr_50", "ssfr_100myr_16", "ssfr_100myr_84",
                     "m_star_50", "m_star_16", "m_star_84"]

    for fits_path in [PATHS["jels_466_ha"], PATHS["jels_470_ha"]]:
        data = read_fits(fits_path, sed_cols_fits).rename(
            columns=dict(zip(sed_cols_fits, sed_cols_new))
        )
        cat = left_merge(cat, data)

    # -----------------------------------------------------------------------
    # 5b. Add UV photometry from photometric catalogs
    # -----------------------------------------------------------------------
    uv_cols_fits = ["M_UV_AB_uncorr", "M_UV_AB_err_uncorr", "beta", "beta_err"]
    uv_cols_new  = ["m_uv_ab_uncorr", "m_uv_ab_err_uncorr", "beta", "beta_err"]

    for fits_path in [PATHS["jels_466_cat"], PATHS["jels_470_cat"]]:
        data = read_fits(fits_path, uv_cols_fits).rename(
            columns=dict(zip(uv_cols_fits, uv_cols_new))
        )
        cat = left_merge(cat, data)

    # -----------------------------------------------------------------------
    # 6. Add EW results (f150 and muv only — we drop f115 and ew_lya)
    # -----------------------------------------------------------------------
    ew_cols = ["ew_muv", "ew_muv_err", "ew_f150", "ew_f150_err"]
    data = read_csv(PATHS["ew_results"], ew_cols)
    cat = left_merge(cat, data)

    # -----------------------------------------------------------------------
    # 7. Derived columns
    # -----------------------------------------------------------------------

    cat["z_spec"] = cat["ID"].map(zspec_map)
    
    cat["agn_flag"] = cat["ID"].isin(AGN_IDS).astype(int)
    cat["ebv"]      = cat["av_50"] / R_V

    cat["lya_lumin"]             = lya_luminosity(cat["lya_flux"], cat["z_muse"])
    cat["lya_lumin_err"]         = lya_luminosity_err(
        cat["lya_flux_err"].values, cat["z_muse"].values
    )
    cat["lya_lumin_upper_limit"] = lya_luminosity(
        cat["lya_flux_upper_limit"], cat["z_muse"]
    )

    cat = compute_fesc(cat)

    # -----------------------------------------------------------------------
    # 8. final_sample_full: one row per source, enforce column order
    # -----------------------------------------------------------------------
    cols_present = [c for c in FINAL_COLS if c in cat.columns]
    full = cat[cols_present].copy()
    full.to_csv(PATHS["out_full"], index=False)
    print(f"\nSaved final_sample_full: {len(full)} sources → {PATHS['out_full']}")

    # -----------------------------------------------------------------------
    # 9. Build final_merged: collapse MUSE-unresolved pairs
    # -----------------------------------------------------------------------

    # Load F150W fluxes for EW recomputation (not stored in main catalog)
    f150_all = pd.concat([
        read_fits(PATHS["jels_466_cat"], ["F150W_auto_flux", "F150W_auto_fluxerr"]),
        read_fits(PATHS["jels_470_cat"], ["F150W_auto_flux", "F150W_auto_fluxerr"]),
    ]).drop_duplicates(subset="ID", keep="first")

    # dict: ID → (flux_nJy, flux_err_nJy)
    f150_lookup = {
        row["ID"]: (row["F150W_auto_flux"], row["F150W_auto_fluxerr"])
        for _, row in f150_all.iterrows()
    }

    all_merger_ids = [i for pair in MERGER_PAIRS for i in pair]
    non_merger = full[~full["ID"].isin(all_merger_ids)].copy()

    merged_rows = []
    for id_keep, id_drop in MERGER_PAIRS:
        row_keep = full[full["ID"] == id_keep].iloc[0]
        row_drop = full[full["ID"] == id_drop].iloc[0]
        print(f"\nMerging: KEEP={id_keep}, DROP={id_drop}")
        merged_row = build_merged_row(row_keep, row_drop, f150_lookup)
        merged_rows.append(merged_row)

    merged = pd.concat([non_merger, pd.DataFrame(merged_rows)], ignore_index=True)
    merged = compute_fesc(merged)           # recompute fesc with summed Hα
    merged = merged[[c for c in FINAL_COLS if c in merged.columns]]
    merged.to_csv(PATHS["out_merged"], index=False)
    print(f"\nSaved final_merged: {len(merged)} sources → {PATHS['out_merged']}")
    print(f"  ({len(full)} sources → removed {len(all_merger_ids)} merger IDs → added {len(merged_rows)} merged rows)")

    # Summary of merger pairs
    summary = full[full["ID"].isin(all_merger_ids)][
        ["ID", "merge_pair", "ha_lumin_fullcorr", "lya_lumin", "ew_f150"]
    ]
    print("\nMerger pair input values:")
    print(summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # 10. final_merged_noagn: remove AGN
    # -----------------------------------------------------------------------
    merged_noagn = merged[merged["agn_flag"] == 0].copy()
    merged_noagn.to_csv(PATHS["out_merged_noagn"], index=False)
    print(f"\nSaved final_merged_noagn: {len(merged_noagn)} sources → {PATHS['out_merged_noagn']}")


if __name__ == "__main__":
    main()