import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# ================================
# Constants
# ================================

c = 2.99792458e18  # speed of light in Angstrom/s
lya_rest = 1215.67  # Angstrom
uv_ref = 1500.0     # Angstrom
f115w_wave = 11500.0  # Angstrom


# =====================================================
# Magnitude -> flux density
# =====================================================

def mag_to_fnu(mag):
    """
    Convert AB magnitude to f_nu
    """

    fnu = 10 ** (-0.4 * (mag + 48.6))
    return fnu


def magerr_to_fnuerr(mag, mag_err):

    fnu = mag_to_fnu(mag)
    err = fnu * (np.log(10) / 2.5) * mag_err

    return err


# =====================================================
# fnu -> flambda
# =====================================================

def fnu_to_flambda(fnu, wavelength):

    """
    Convert f_nu to f_lambda
    """

    return fnu * c / wavelength**2


def fnuerr_to_flambdaerr(fnu_err, wavelength):

    return fnu_err * c / wavelength**2


# =====================================================
# MUV -> luminosity density
# =====================================================

def muv_to_lnu(Muv):

    """
    Convert absolute UV magnitude to luminosity density L_nu
    """

    Lnu = 10 ** (-0.4 * (Muv + 48.6)) * 4 * np.pi * (10 * u.pc.to(u.cm))**2
    return Lnu


# =====================================================
# luminosity -> observed flux
# =====================================================

def lnu_to_fnu(Lnu, z):

    """
    Convert luminosity density to observed flux density
    """

    DL = cosmo.luminosity_distance(z).to(u.cm).value

    fnu = Lnu * (1 + z) / (4 * np.pi * DL**2)

    return fnu


# =====================================================
# beta scaling
# =====================================================

def beta_scale_flux(flambda, beta, lambda_from, lambda_to):

    """
    Scale continuum using UV slope beta
    """

    return flambda * (lambda_to / lambda_from) ** beta


# =====================================================
# EW calculations
# =====================================================

def ew_obs(F_line, f_cont):

    """
    Observed equivalent width
    """

    return F_line / f_cont


def ew_rest(EW_obs, z):

    """
    Convert observed EW to rest EW
    """

    return EW_obs / (1 + z)


# =====================================================
# Error propagation
# =====================================================

def ew_error(F, Ferr, f, ferr):

    """
    Propagate errors for EW_obs
    """

    EW = F / f

    err = EW * np.sqrt((Ferr / F) ** 2 + (ferr / f) ** 2)

    return err


# =====================================================
# F115W continuum
# =====================================================

def f115w_continuum(mag, mag_err, beta, z):

    fnu = mag_to_fnu(mag)
    fnu_err = magerr_to_fnuerr(mag, mag_err)

    flambda = fnu_to_flambda(fnu, f115w_wave)
    flambda_err = fnuerr_to_flambdaerr(fnu_err, f115w_wave)

    lya_obs = lya_rest * (1 + z)

    flambda_lya = beta_scale_flux(flambda, beta, f115w_wave, lya_obs)
    flambda_lya_err = beta_scale_flux(flambda_err, beta, f115w_wave, lya_obs)

    return flambda_lya, flambda_lya_err


# =====================================================
# MUV continuum
# =====================================================

def muv_continuum(Muv, beta, z):

    Lnu = muv_to_lnu(Muv)

    fnu = lnu_to_fnu(Lnu, z)

    lam_uv_obs = uv_ref * (1 + z)

    flambda = fnu_to_flambda(fnu, lam_uv_obs)

    lya_obs = lya_rest * (1 + z)

    flambda_lya = beta_scale_flux(flambda, beta, lam_uv_obs, lya_obs)

    return flambda_lya


# =====================================================
# 2 sigma flux limit
# =====================================================

def flux_limit_2sigma(mag_err):

    """
    Convert magnitude error into flux limit
    """

    snr = 1.0857 / mag_err

    return snr < 2


# =====================================================
# Main
# =====================================================

def main():

    flux = pd.read_csv("/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv")

    sed466 = pd.DataFrame(fits.open(
        "/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits")[1].data)

    sed470 = pd.DataFrame(fits.open(
        "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits")[1].data)

    sed = pd.concat([sed466, sed470], ignore_index=True)

    sed = sed[sed["ID"].isin(flux["ID"])]

    df = flux.merge(sed, on="ID")

    df = df[df["lya_detect_flag"].isin([1, 2])]

    df["F"] = df["flux_fit"] * 1e-20
    df["F_err"] = df["flux_fit_err"] * 1e-20


    # ======================
    # continuum calculations
    # ======================

    cont_muv = []
    cont_f115 = []
    cont_f115_err = []

    for i, row in df.iterrows():

        z = row["z_used"]
        beta = row["beta"]
        beta_set = -1.6

        # MUV continuum
        cont_muv.append(
            muv_continuum(row["M_UV_AB_uncorr"], beta, z)
        )

        # F115W continuum
        f, ferr = f115w_continuum(
            row["F115W_auto_mag"],
            row["F115W_auto_magerr"],
            beta_set,
            z
        )

        cont_f115.append(f)
        cont_f115_err.append(ferr)


    df["cont_muv"] = cont_muv
    df["cont_f115"] = cont_f115
    df["cont_f115_err"] = cont_f115_err


    # ======================
    # EW calculations
    # ======================

    ew_muv = []
    ew_f115 = []
    ew_f115_lim = []

    for i, row in df.iterrows():

        z = row["z_used"]

        # MUV EW
        ew_o = ew_obs(row["F"], row["cont_muv"])
        ew_muv.append(ew_rest(ew_o, z))

        # F115W EW
        ew_o = ew_obs(row["F"], row["cont_f115"])
        ew_f115.append(ew_rest(ew_o, z))

        # detection check
        if flux_limit_2sigma(row["F115W_auto_magerr"]):

            ew_lim = ew_rest(
                ew_obs(row["F"], row["cont_f115"]),
                z
            )

        else:

            ew_lim = np.nan

        ew_f115_lim.append(ew_lim)


    df["ew_muv"] = ew_muv
    df["ew_f115"] = ew_f115
    df["ew_f115_lim"] = ew_f115_lim


    # ======================
    # diagnostics table
    # ======================

    diagnostics_cols = [
        "ID",
        "flux_fit",
        "flux_fit_err",
        "z_used",
        "beta",
        "beta_err",
        "M_UV_AB_uncorr",
        "M_UV_AB_err_uncorr",
        "F115W_auto_mag",
        "F115W_auto_magerr",
        "cont_muv",
        "cont_f115",
        "cont_f115_err"
    ]

    df[diagnostics_cols].to_csv("/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_diagnostics.csv", index=False)
    print ("Diagnostic table saved as /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_diagnostics.csv")


    # ======================
    # final output
    # ======================

    df[["ID", "ew_muv", "ew_f115", "ew_f115_lim"]].to_csv(
        "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv",
        index=False
    )
    print ("EW results saved as /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv")


if __name__ == "__main__":
    main()