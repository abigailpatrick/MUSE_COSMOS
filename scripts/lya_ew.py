import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# -----------------------------
# Constants
# -----------------------------
c = 2.99792458e18   # speed of light in Angstrom/s
lya_rest = 1215.67  # Lyα rest wavelength (Å)
lambda_uv = 1500.0      # wavelength where MUV defined
lambda_f115 = 11500.0   # F115W central wavelength (Å)
lambda_f150 = 15000.0   # F150W central wavelength (Å)

# -----------------------------
# EW functions
# -----------------------------
def ew_obs(F_line, f_lambda):
    """Observed equivalent width"""
    return F_line / f_lambda

def ew_rest(EW_obs, z):
    """Convert observed EW to rest-frame"""
    return EW_obs / (1 + z)

def ew_lower_limit(F_line, f_limit):
    """EW lower limit when continuum undetected"""
    return F_line / f_limit

def ew_error(F, F_err, f, f_err):
    """Propagate errors for EW"""
    return (F/f) * np.sqrt((F_err/F)**2 + (f_err/f)**2)

# -----------------------------
# Continuum helper functions
# -----------------------------
def muv_to_lnu(Muv):
    """Convert absolute UV magnitude to luminosity density"""
    return 10**(-0.4*(Muv + 48.6)) * 4*np.pi*(10*u.pc.to(u.cm))**2

def lnu_to_fnu(Lnu, z):
    """Convert luminosity density to observed flux density"""
    z = np.asarray(z)
    DL = cosmo.luminosity_distance(z).to(u.cm).value
    return Lnu * (1+z) / (4*np.pi*DL**2)

def fnu_to_flambda(fnu, wavelength):
    """Convert f_nu to f_lambda"""
    return fnu * c / wavelength**2

def scale_beta(f_lambda, beta, lambda_from, lambda_to):
    """Scale continuum using UV slope beta"""
    return f_lambda * (lambda_to/lambda_from)**beta

# -----------------------------
# Main
# -----------------------------
def main():
    # -----------------------------
    # Read input catalogues
    # -----------------------------
    flux = pd.read_csv("/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv")
    sed466 = pd.DataFrame(fits.open("/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits")[1].data)
    sed470 = pd.DataFrame(fits.open("/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits")[1].data)
    sed = pd.concat([sed466, sed470], ignore_index=True)
    sed = sed[sed["ID"].isin(flux["ID"])]
    df = flux.merge(sed, on="ID")

    df = df[df["lya_detect_flag"].isin([1,2])]

    # for source ID 4419, print every value in the table 
    source_4419 = df[df["ID"] == 4419]
    #print("Values for source ID 4419:")
    #for col in df.columns:
        #print(f"{col}: {source_4419[col].values[0]}")

    # -----------------------------
    # Line flux (convert to cgs)
    # -----------------------------
    df["F"] = df["flux_fit"] * 1e-20
    df["F_err"] = df["flux_fit_err"] * 1e-20
    z = df["z_used"].values
    lambda_lya_obs = lya_rest * (1+z)

    # =====================================================
    # MUV CONTINUUM
    # =====================================================
    Lnu = muv_to_lnu(df["M_UV_AB_uncorr"])
    fnu_uv = lnu_to_fnu(Lnu, z)
    flambda_uv = fnu_to_flambda(fnu_uv, lambda_uv*(1+z))
    df["fcont_muv"] = scale_beta(flambda_uv, df["beta"], lambda_uv*(1+z), lambda_lya_obs)

    # Assume small error in MUV to propagate (from M_UV_AB_err_uncorr)
    f_lambda_err_muv = fcont_err_muv = flambda_uv * 0.4 * np.log(10) * df["M_UV_AB_err_uncorr"].fillna(0.1)
    
    # =====================================================
    # F115W CONTINUUM (USING FLUX COLUMN)
    # =====================================================
    beta_set = -2.0
    fnu_115 = df["F115W_auto_flux"].values * 1e-29
    fnu_115_err = df["F115W_auto_fluxerr"].values * 1e-29
    flambda_115 = fnu_to_flambda(fnu_115, lambda_f115)
    df["fcont_f115"] = scale_beta(flambda_115, beta_set, lambda_f115, lambda_lya_obs)
    fcont_err_f115 = scale_beta(fnu_to_flambda(fnu_115_err, lambda_f115), beta_set, lambda_f115, lambda_lya_obs)

    # =====================================================
    # F150W CONTINUUM (USING FLUX COLUMN)
    # =====================================================
    beta_set = -2.0
    fnu_150 = df["F150W_auto_flux"].values * 1e-29
    fnu_150_err = df["F150W_auto_fluxerr"].values * 1e-29
    flambda_150 = fnu_to_flambda(fnu_150, lambda_f150)  
    df["fcont_f150"] = scale_beta(flambda_150, beta_set, lambda_f150, lambda_lya_obs)
    fcont_err_f150 = scale_beta(fnu_to_flambda(fnu_150_err, lambda_f150), beta_set, lambda_f150, lambda_lya_obs)

    # =====================================================
    # SNR AND CONTINUUM LIMITS
    # =====================================================
    df["snr115"] = fnu_115 / fnu_115_err
    df["snr150"] = fnu_150 / fnu_150_err
    detected_115 = (df["snr115"] >= 2) & (fnu_115 > 0)
    detected_150 = (df["snr150"] >= 2) & (fnu_150 > 0)


    # 2σ continuum limit for non-detections
    limit_fnu = 3 * fnu_115_err
    limit_flambda = fnu_to_flambda(limit_fnu, lambda_f115)
    df["fcont_f115_lim"] = scale_beta(limit_flambda, beta_set, lambda_f115, lambda_lya_obs)

    # =====================================================
    # EW CALCULATIONS
    # =====================================================
    ewobs_muv = ew_obs(df["F"], df["fcont_muv"])
    ewerr_muv = ew_error(df["F"], df["F_err"], df["fcont_muv"], fcont_err_muv)
    df["ew_muv"] = ew_rest(ewobs_muv, z)
    df["ew_muv_err"] = ewerr_muv / (1+z)

    ewobs_f115 = ew_obs(df["F"], df["fcont_f115"])
    ewerr_f115 = ew_error(df["F"], df["F_err"], df["fcont_f115"], fcont_err_f115)
    ewobs_lim = ew_lower_limit(df["F"], df["fcont_f115_lim"])
    df["ew_f115"] = np.where(detected_115, ew_rest(ewobs_f115, z), np.nan)  
    df["ew_f115_err"] = np.where(detected_115, ewerr_f115/(1+z), np.nan)
    df["ew_f115_lower"] = np.where(~detected_115, ew_rest(ewobs_lim, z), np.nan)

    ewobs_f150 = ew_obs(df["F"], df["fcont_f150"])
    ewerr_f150 = ew_error(df["F"], df["F_err"], df["fcont_f150"], fcont_err_f150)
    df["ew_f150"] = np.where(detected_150, ew_rest(ewobs_f150, z), np.nan)
    df["ew_f150_err"] = np.where(detected_150, ewerr_f150/(1+z), np.nan)

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================
    diagnostics_cols = [
        "ID","flux_fit","flux_fit_err","z_used","beta","beta_err",
        "M_UV_AB_uncorr","M_UV_AB_err_uncorr",
        "F115W_auto_flux","F115W_auto_fluxerr",
        "F150W_auto_flux","F150W_auto_fluxerr",
        "fcont_muv","fcont_f115","fcont_f115_lim","snr115", 
    ]
    df[diagnostics_cols].to_csv(
        "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_diagnostics.csv",
        index=False
    )

    df[["ID","ew_muv","ew_muv_err","ew_f115","ew_f115_err","ew_f115_lower","snr115","ew_f150","ew_f150_err"]].to_csv(
        "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv",
        index=False
    )

    print("EW results saved to /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results.csv")

if __name__ == "__main__":
    main()