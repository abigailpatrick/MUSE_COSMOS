#!/usr/bin/env python3

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from astropy.io import fits

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
C_A_S = 2.99792458e18
LYA_REST = 1215.67


# ------------------------------------------------------------
# AB mag → f_lambda
# ------------------------------------------------------------
def mag_to_flambda(m_ab, lambda_eff):
    f_nu = 10**(-0.4 * (m_ab + 48.6))
    return f_nu * C_A_S / (lambda_eff**2)


# ------------------------------------------------------------
# METHOD 1
# ------------------------------------------------------------
def ew_method1(F, F_err, m, m_err, beta, beta_err, lambda_eff, z):

    lambda_lya_obs = LYA_REST * (1 + z)

    f_eff = mag_to_flambda(m, lambda_eff)
    f_lya = f_eff * (lambda_lya_obs / lambda_eff)**beta

    # continuum fractional error from magnitude
    frac_mag = 0.4 * np.log(10) * m_err

    # fractional error from beta
    ln_term = np.log(lambda_lya_obs / lambda_eff)
    frac_beta = ln_term * beta_err

    frac_f = np.sqrt(frac_mag**2 + frac_beta**2)

    W_obs = F / f_lya
    W_rest = W_obs / (1 + z)

    frac_W = np.sqrt((F_err / F)**2 + frac_f**2)
    W_err = W_rest * frac_W

    return W_rest, W_err


# ------------------------------------------------------------
# METHOD 2
# ------------------------------------------------------------
def ew_method2(F, F_err, Muv, Muv_err, beta, beta_err, z, lambda0=1500.):

    d_L = Planck18.luminosity_distance(z).to("cm").value

    L_nu = 10**(-0.4 * (Muv + 48.6)) * 4 * np.pi * (10 * 3.086e18)**2
    L_lambda_0 = L_nu * C_A_S / (lambda0**2)

    L_lambda_1216 = L_lambda_0 * (1216. / lambda0)**beta

    f_lya = L_lambda_1216 / (4 * np.pi * d_L**2 * (1 + z))

    # fractional errors
    frac_mag = 0.4 * np.log(10) * Muv_err
    ln_term = np.log(1216. / lambda0)
    frac_beta = ln_term * beta_err

    frac_f = np.sqrt(frac_mag**2 + frac_beta**2)

    W_obs = F / f_lya
    W_rest = W_obs / (1 + z)

    frac_W = np.sqrt((F_err / F)**2 + frac_f**2)
    W_err = W_rest * frac_W

    return W_rest, W_err


# ------------------------------------------------------------
# LOAD + MERGE
# ------------------------------------------------------------
def load_sed_catalog(path):
    with fits.open(path) as hdul:
        return pd.DataFrame(hdul[1].data)


def main():

    # ---- Files ----
    flux_csv = "lya_flux_ap0p6.csv"
    f466_path = "/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits"
    f470_path = "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits"

    # ---- Load ----
    flux = pd.read_csv(flux_csv)

    sed466 = load_sed_catalog(f466_path)
    sed470 = load_sed_catalog(f470_path)

    sed_all = pd.concat([sed466, sed470], ignore_index=True)

    # Keep only sources in flux table
    sed = sed_all[sed_all["ID"].isin(flux["ID"])]

    # Merge
    df = flux.merge(sed, on="ID")

    # ---- Convert flux units ----
    df["F"] = df["flux_fit"] * 1e-20
    df["F_err"] = df["flux_fit_err"] * 1e-20

    df["z"] = df["z_used"]

    # ---- PRINT CHECK ARRAYS ----
    print("\n--- CHECK INPUT ARRAYS ---")
    print("IDs:", df["ID"].values)
    print("beta:", df["beta"].values)
    print("beta_err:", df["beta_err"].values)
    print("F125W mag:", df["WFC3IR_F125W"].values)
    print("M_UV_AB_uncorr:", df["M_UV_AB_uncorr"].values)
    print("M_UV_AB_err_uncorr:", df["M_UV_AB_err_uncorr"].values)
    print("--------------------------\n")

    lambda_eff = 12500.0

    # ---- Compute EWs ----
    results = []

    for _, r in df.iterrows():

        W1, W1_err = ew_method1(
            r["F"], r["F_err"],
            r["WFC3IR_F125W"], r["WFC3IR_F125W_err"],
            r["beta"], r["beta_err"],
            lambda_eff, r["z"]
        )

        W2, W2_err = ew_method2(
            r["F"], r["F_err"],
            r["M_UV_AB_uncorr"], r["M_UV_AB_err_uncorr"],
            r["beta"], r["beta_err"],
            r["z"]
        )

        results.append([
            r["ID"],
            r["F"],
            W1, W1_err,
            W2, W2_err
        ])

    out = pd.DataFrame(results, columns=[
        "ID",
        "F_lya_cgs",
        "EW_rest_method1",
        "EW_rest_method1_err",
        "EW_rest_method2",
        "EW_rest_method2_err"
    ])

    out.to_csv("lya_ew_results_with_errors.csv", index=False)
    print("Saved: lya_ew_results_with_errors.csv")


if __name__ == "__main__":
    main()