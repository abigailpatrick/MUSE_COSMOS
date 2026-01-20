
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from scipy.ndimage import gaussian_filter1d
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import line_flux
from specutils.analysis import equivalent_width
from astropy.cosmology import Planck18 as cosmo # for luminosity distance note if I want to change this 
import astropy.units as u
import os, csv, shutil


# ------------------------------------------------------------
# 1 — Load subcube
# ------------------------------------------------------------
def load_cube(path):
    return Cube(path)


# ------------------------------------------------------------
# 2 — Convert RA/Dec → pixel
# ------------------------------------------------------------
def coords_to_pixel(cube, ra, dec):
    coords = np.array([[dec, ra]])
    pix = cube.wcs.sky2pix(coords, nearest=True, unit='deg')
    y, x = pix[0]
    return int(x), int(y)


# ------------------------------------------------------------
# 3 — Build circular aperture mask
# ------------------------------------------------------------
def build_aperture_mask(nx, ny, x0, y0, radius_pix):
    yy, xx = np.indices((ny, nx))
    rr = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    return rr <= radius_pix


# ------------------------------------------------------------
# 4 — Extract 1D aperture spectrum
# ------------------------------------------------------------
def extract_1d_spectrum_with_var(
    cube, xpix, ypix,
    aperture_arcsec, pixel_scale,
    wave_min, wave_max
):
    # aperture
    radius_pix = aperture_arcsec / pixel_scale
    ny, nx = cube.shape[1], cube.shape[2]
    apmask = build_aperture_mask(nx, ny, xpix, ypix, radius_pix)

    # wavelength slice
    sub = cube.select_lambda(wave_min, wave_max)
    wave = sub.wave.coord()

    data = sub.data
    var  = sub.var   # ← EXT=2 (variance cube)

    # Flux spectrum
    flux_1d = (data * apmask).sum(axis=(1, 2))

    # Variance spectrum (sum of variances)
    var_1d = (var * apmask).sum(axis=(1, 2))

    return wave, flux_1d, var_1d



# ------------------------------------------------------------
# 5 — Optional smoothing
# ------------------------------------------------------------
def smooth_spectrum(wave, flux, fwhm):
    if fwhm is None:
        return None
    dw = np.median(np.diff(wave))
    sigma_A = fwhm / 2.355
    sigma_pix = sigma_A / dw
    return gaussian_filter1d(flux, sigma_pix)


# ------------------------------------------------------------
# 6 — Measure flux in region
# ------------------------------------------------------------
def integrate_flux(wave, flux, region):
    λ1, λ2 = region
    mask = (wave >= λ1) & (wave <= λ2)
    return np.trapz(flux[mask], wave[mask])

def integrate_flux_error(wave, var, region):
    λ1, λ2 = region
    mask = (wave >= λ1) & (wave <= λ2)

    if np.sum(mask) < 2:
        return np.nan

    dw = np.median(np.diff(wave))

    # variance of integrated flux
    flux_var = np.sum(var[mask] * dw**2)

    return np.sqrt(flux_var)

# Using specutils
def integrate_flux_lineflux(wave, flux, region):
    λ1, λ2 = region
    spectrum = Spectrum1D(
        spectral_axis = wave * u.AA,
        flux          = flux * u.Unit("")
    )
    reg = SpectralRegion(λ1 * u.AA, λ2 * u.AA)
    result = line_flux(spectrum, regions=reg)
    return result.value

def line_snr(line_flux, line_flux_err):
    if not np.isfinite(line_flux_err) or line_flux_err <= 0:
        return np.nan
    return line_flux / line_flux_err


# ------------------------------------------------------------
# 7 - Measure Equivalent Width 
# ------------------------------------------------------------


def equivalent_width_numpy(wave, flux, region, continuum=1.0):
    λ1, λ2 = region
    mask = (wave >= λ1) & (wave <= λ2)
    if np.sum(mask) < 2:
        return np.nan
    norm = flux[mask] / continuum
    integrand = (1 - norm)
    return np.trapz(integrand, wave[mask])

def equivalent_width_specutils(wave, flux, region, continuum=1.0):
    λ1, λ2 = region
    spec = Spectrum1D(
        spectral_axis = wave * u.AA,
        flux          = flux * u.Unit("")
    )
    reg = SpectralRegion(λ1 * u.AA, λ2 * u.AA)
    # specutils returns a Quantity in Å
    ew = equivalent_width(spec, regions=reg, continuum=continuum)

    return ew.to(u.AA).value


# ------------------------------------------------------------
# 8 — Plot spectrum
# ------------------------------------------------------------
def plot_spectrum(
        wave, flux, flux_smooth,
        aperture_arcsec,
        flux_region,
        cont_level=None,
        sideband_regions=None,
        output_path="spectrum.png"
    ):

    plt.figure(figsize=(14, 4))
    plt.step(wave, flux, where='mid', color='gray', lw=1, alpha=0.5, label="Extracted")

    if flux_smooth is not None:
        plt.plot(wave, flux_smooth, color="purple", lw=1.8, label="Smoothed")

    # Mark flux measurement region
    if flux_region is not None:
        λ1, λ2 = flux_region
        plt.axvspan(λ1, λ2, color='cyan', alpha=0.25,
                    label=f"Flux region {λ1:.1f}–{λ2:.1f} Å")

    # Plot sideband continuum estimate
    if cont_level is not None and sideband_regions is not None:
        for region in sideband_regions:
            λ1, λ2 = region
            plt.hlines(cont_level, λ1, λ2, color='red', lw=2, label="Sideband continuum")
    
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux (integrated over aperture)")
    plt.title(f"Spectrum (aperture = {aperture_arcsec*2}\" )")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved spectrum to /cephfs/apatrick/musecosmos/scripts/{output_path}")


# ------------------------------------------------------------
# 8 - Continuum estimation
# ------------------------------------------------------------

def median_sideband_continuum(wave, flux, line_region, blue_region, red_region):
    """
    Estimate the local continuum using the median of sidebands around a line.

    Parameters
    ----------
    wave : array-like
        Wavelength array (Å).
    flux : array-like
        Flux array (same length as wave).
    line_region : tuple
        (λ_line_min, λ_line_max) wavelength range of the line.
    blue_region : tuple
        (λ_blue_min, λ_blue_max) wavelength range of blue sideband.
    red_region : tuple
        (λ_red_min, λ_red_max) wavelength range of red sideband.

    Returns
    -------
    float
        Estimated continuum (median flux of sidebands).
    """
    λ1, λ2 = line_region
    λb1, λb2 = blue_region
    λr1, λr2 = red_region

    # Mask blue and red sidebands
    blue_mask = (wave >= λb1) & (wave <= λb2)
    red_mask  = (wave >= λr1) & (wave <= λr2)

    # Combine sidebands
    sideband_flux = np.concatenate([flux[blue_mask], flux[red_mask]])

    if len(sideband_flux) == 0:
        print("[WARN] No flux in sideband regions! Returning 1.0 as continuum.")
        return 1.0  # default fallback

    return np.median(sideband_flux)

# ------------------------------------------------------------
# 9 - Integrrated line flux to luminosity 
# ------------------------------------------------------------
def flux_to_luminosity(flux, redshift):
    """
    Convert flux to luminosity given a redshift.

    Parameters
    ----------
    flux : float
        Integrated line flux (erg/s/cm²).
    redshift : float
        Redshift of the source.

    Returns
    -------
    float
        Luminosity (erg/s).
    """

    # Convert flux to erg/s/cm^2
    flux = flux * 1e-20
 
    d_L = cosmo.luminosity_distance(redshift).to(u.cm).value  # in cm
    luminosity = flux * 4 * np.pi * d_L**2
    return luminosity


def flux_error_to_luminosity_error(flux_err, redshift):
    """
    Convert flux error to luminosity error given a redshift.

    Parameters
    ----------
    flux_err : float
        Error on integrated line flux (same units as flux, i.e. 1e-20 erg/s/cm^2).
    redshift : float
        Redshift of the source.

    Returns
    -------
    float
        Luminosity error (erg/s).
    """

    if not np.isfinite(flux_err):
        return np.nan

    # Apply same flux scaling as luminosity conversion
    flux_err = flux_err * 1e-20

    d_L = cosmo.luminosity_distance(redshift).to(u.cm).value
    lum_err = flux_err * 4 * np.pi * d_L**2

    return lum_err


# ------------------------------------------------------------
# 10 - Update the CSV
# ------------------------------------------------------------

def update_csv_spec(csv_path, out_filename, spec_ew, spec_flux, trap_flux, lya_flux_err, lya_l, lya_l_err, line_snr):
    """
    Update CSV row (matching row_index) with spectral measurements.
    """

    row_id = os.path.basename(out_filename).split("_", 1)[0].strip()
    if not row_id:
        print(f"[WARN] Could not infer row_index from out filename: {out_filename}")
        return

    tmp_path = csv_path + ".tmp"
    updated = False

    with open(csv_path, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            print(f"[WARN] CSV has no header: {csv_path}")
            return
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Ensure required columns exist
    required_cols = [
        "spec_ew",
        "spec_flux",
        "trap_flux",
        "lya_flux_err",
        "lya_l",
        "lya_l_err",
        "line_snr",
    ]

    for col in required_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    # Format values
    def fmt(x):
        return f"{x:.6e}" if np.isfinite(x) else ""

    for row in rows:
        if row.get("row_index", "").strip() == row_id:
            row["spec_ew"]      = fmt(spec_ew)
            row["spec_flux"]    = fmt(spec_flux)
            row["trap_flux"]    = fmt(trap_flux)
            row["lya_flux_err"] = fmt(lya_flux_err)
            row["lya_l"]        = fmt(lya_l)
            row["lya_l_err"]    = fmt(lya_l_err)
            row["line_snr"] = f"{line_snr:.3f}" if np.isfinite(line_snr) else ""
            updated = True
            break

    if not updated:
        print(f"[WARN] row_index={row_id} not found in {csv_path}; no CSV update.")
        return

    with open(tmp_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for c in fieldnames:
                row.setdefault(c, "")
            writer.writerow(row)

    os.replace(tmp_path, csv_path)
    print(f"[UPDATED CSV] {csv_path}: row_index={row_id}")



# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a 1D spectrum from a MUSE subcube and measure Lyα flux."
    )

    parser.add_argument("--cube", required=True, help="Path to FITS subcube")
    parser.add_argument("--ra", type=float, required=True, help="RA of source (deg)")
    parser.add_argument("--dec", type=float, required=True, help="Dec of source (deg)")
    parser.add_argument("--z", type=float, required=False, help="Redshift of source (for luminosity calc)")
    parser.add_argument("--aperture", type=float, required=True, help="Aperture radius (arcsec)")
    parser.add_argument("--pixscale", type=float, default=0.2, help="Arcsec per pixel")
    parser.add_argument("--smooth_fwhm", type=float, default=None, help="FWHM for Gaussian smoothing (Å)")

    parser.add_argument("--wmin", type=float, required=True, help="Min wavelength (Å)")
    parser.add_argument("--wmax", type=float, required=True, help="Max wavelength (Å)")

    parser.add_argument("--fluxmin", type=float, required=True, help="Flux region min λ (Å)")
    parser.add_argument("--fluxmax", type=float, required=True, help="Flux region max λ (Å)")

    parser.add_argument("--out", default="spectrum.png", help="Output spectrum PNG")

    return parser.parse_args()


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
def main():
    args = parse_args()

    print("Loading subcube...")
    # Load cube
    cube = load_cube(args.cube)
    print("Cube loaded.")

    # Coordinates → pixels
    xpix, ypix = coords_to_pixel(cube, args.ra, args.dec)
    print(f"[INFO] Pixel position: x={xpix:.2f}, y={ypix:.2f}")

    print("Extracting 1D spectrum...")
    # Extract 1D spectrum
    wave, flux, var = extract_1d_spectrum_with_var(
        cube=cube,
        xpix=xpix,
        ypix=ypix,
        aperture_arcsec=args.aperture,
        pixel_scale=args.pixscale,
        wave_min=args.wmin,
        wave_max=args.wmax
    )

    print("1D spectrum extracted.")

    # Smooth
    flux_smooth = smooth_spectrum(wave, flux, args.smooth_fwhm)

    # Measure flux
    print("Measuring Lyα flux...")
    lya_flux_numpy = integrate_flux(wave, flux, (args.fluxmin, args.fluxmax))
    lya_flux_specutils = integrate_flux_lineflux(wave, flux, (args.fluxmin, args.fluxmax))
    lya_flux_err = integrate_flux_error(wave, var, (args.fluxmin, args.fluxmax))
    lya_snr = line_snr(lya_flux_numpy, lya_flux_err)
    
    print(f"[RESULT] Line S/N = {lya_snr:.2f}")
    print("[RESULT] NumPy trapezoid =", lya_flux_numpy)
    print("[RESULT] specutils line_flux =", lya_flux_specutils)
    print("[RESULT] Flux error =", lya_flux_err)

    if args.z is not None:
        lya_luminosity = flux_to_luminosity(lya_flux_numpy, args.z)
        lya_luminosity_err = flux_error_to_luminosity_error(lya_flux_err, args.z)

        print(f"[RESULT] Lyα Luminosity at z={args.z} = {lya_luminosity:.3e} erg/s")
        print(f"[RESULT] Lyα Luminosity error       = {lya_luminosity_err:.3e} erg/s")
    else:
        lya_luminosity = np.nan
        lya_luminosity_err = np.nan

    # Measure equivalent width
    line_region = (args.fluxmin, args.fluxmax)      # line region in Å
    blue_region = (args.fluxmin - 110, args.fluxmin - 10)      # blue continuum sideband
    red_region  = (args.fluxmax + 10, args.fluxmax + 110)      # red continuum sideband
    sidebands = [blue_region,red_region]

    # Estimate continuum
    cont_est = median_sideband_continuum(wave, flux, line_region, blue_region, red_region)
    print(f"[INFO] Estimated continuum from sidebands = {cont_est:.3f}")
    cont_est = 1.0 # I think it's just noise as continuum here so leave out
    
    # Compute EW
    print("Measuring equivalent width...")
    ew_np = equivalent_width_numpy(wave, flux, line_region, continuum=cont_est)
    ew_su = equivalent_width_specutils(wave, flux, line_region, continuum=cont_est)
    print(f"[RESULT] EW (NumPy, sidebands cont)      = {ew_np:.3f} Å")
    print(f"[RESULT] EW (specutils, sidebands cont) = {ew_su:.3f} Å")


    # Plot
    print("Plotting spectrum...")
    plot_spectrum(
        wave=wave,
        flux=flux,
        flux_smooth=flux_smooth,
        aperture_arcsec=args.aperture,
        flux_region=(args.fluxmin, args.fluxmax),
        cont_level=cont_est, sideband_regions=sidebands,
        output_path=args.out
        )
    
    update_csv_spec(
        "/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv",
        args.out,
        ew_su,
        lya_flux_specutils,
        lya_flux_numpy,
        lya_flux_err,
        lya_luminosity,
        lya_luminosity_err,
        lya_snr,

    )

    


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
