import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from mpdaf.obj import Cube
import argparse
import os

from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Circle
from astropy.visualization import quantity_support
import matplotlib.image as mpimg
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm

import warnings
from astropy.wcs import FITSFixedWarning


from glob import glob
import re


# Suppress specific warnings
warnings.simplefilter('ignore', FITSFixedWarning)
warnings.filterwarnings('ignore', message=".*'partition' will ignore the 'mask'.*")


"""
This version replaces separate spectrum/variance plots with:
 - A combined plot showing Flux and Noise on the same axes
 - A second plot showing S/N.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Extract pseudo-narrowband images and 1D spectra from a MUSE cube for a list of objects.")

    parser.add_argument("--cube", type=str, 
                        default="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_withFSF_VAR.fits", 
                        help="Path to the MUSE cube FITS file.")
    
    parser.add_argument("--objects", type=str,
                        default="/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all.csv",
                        help="Path to a CSV file containing RA, Dec, and redshift (z).")
    
    parser.add_argument("--spatial_width", type=float, 
                        default=3.0, 
                        help="Spatial width for extraction in arcsec.")

    parser.add_argument("--spectral_width", type=float, 
                        default=800,
                        help="Spectral width for subcube extraction in Angstroms.")

    parser.add_argument("--rest_wavelength", type=float,
                        default=1215.7, 
                        help="Rest-frame wavelength of the emission line in Angstroms (default is Lyman-alpha).")
    
    parser.add_argument("--wavelengths", type=float, nargs='+', default=[1215.7, 1240.0, 1260.0, 1303, 1336, 1395, 1548.2],
                        help="List of rest-frame wavelengths of interest in Angstroms (default includes Lyα, NV, CIV).")

    parser.add_argument("--spectrum_radius", type=float,
                        default=0.5,
                        help="Radius for 1D spectrum aperture in arcsec (default is 0.5 arcsec).")

    parser.add_argument("--spectrum_width", type=float,
                        default=800.0,
                        help="Width around central wavelength for 1D spectrum in Angstroms (default is 800 Å).")

    parser.add_argument("--pixel_scale", type=float,
                        default=0.2, 
                        help="Pixel scale of the cube in arcsec/pixel (default is 0.2 for MUSE).")
    
    parser.add_argument("--nb_image_width", type=float,
                        default=60.0,
                        help="Width of the pseudo-narrowband image in Angstroms (default is 60 Å).")

    parser.add_argument("--b_region", type=float,
                        default=30.0,
                        help="Width of the bandpass highlight region in Angstroms (default is 30 Å).")

    parser.add_argument("--output", type=str,
                        default="/cephfs/apatrick/musecosmos/dataproducts/extractions/", 
                        help="Path to the output directory.")
    
    parser.add_argument("--use_saved_subcubes", action="store_true",
                        help="If set, use already saved subcubes instead of extracting them again.")

    parser.add_argument("--smooth_fwhm", type=float, default=5.0,
                        help="FWHM in Å for Gaussian smoothing used in the overlays.")
    
    args = parser.parse_args()
    return args


def ra_dec_to_xy(ra, dec, cube):
    coords = np.array([[dec, ra]])  # (dec, ra)
    pix = cube.wcs.sky2pix(coords, nearest=True, unit='deg')  # returns (y, x)
    y, x = pix[0]
    return int(x), int(y)


def z_to_wavelength(z, rest_wavelength):
    obs_wavelength = rest_wavelength * (1 + z) * u.AA
    return obs_wavelength


def extract_subcube(cube, x, y, obs_wavelength, spatial_width, spectral_width, pixel_scale):
    """
    Extract a subcube for a single source if within bounds.
    Returns: subcube or None if outside bounds.
    """
    # Convert spatial width from arcsec to pixels
    spatial_width = int(spatial_width / pixel_scale)
    half_size = spatial_width // 2
    x_min, x_max = int(max(x - half_size, 0)), int(min(x + half_size, cube.shape[2]))
    y_min, y_max = int(max(y - half_size, 0)), int(min(y + half_size, cube.shape[1]))

    wave_min = obs_wavelength.value - spectral_width / 2
    wave_max = obs_wavelength.value + spectral_width / 2

    # Check wavelength coverage
    if wave_max < cube.wave.coord()[0] or wave_min > cube.wave.coord()[-1]:
        print(f"Source at {obs_wavelength:.1f} Å is outside wavelength coverage.")
        return None

    # Extract subcube
    subcube = cube.select_lambda(wave_min, wave_max)[:, y_min:y_max, x_min:x_max]

    if subcube.shape[1] == 0 or subcube.shape[2] == 0:
        print(f"Source outside spatial bounds: x={x}, y={y}")
        return None
    
    return subcube


def ensure_angstrom(value):
    """Ensure value is an astropy Quantity in Å."""
    return value if isinstance(value, u.Quantity) else value * u.AA


def build_aperture_mask(ny, nx, radius_pix, cx=None, cy=None):
    if cx is None: cx = nx // 2
    if cy is None: cy = ny // 2
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (r <= radius_pix).astype(float)


def extract_aperture_spectra(flux_subcube, stat_subcube, spectrum_radius, central_wavelength,
                             spectrum_width, pixel_scale, fwhm):
    """
    Extract 1D flux, noise and S/N spectra from a circular aperture.

    Returns:
        wave               [N]
        flux_1d            [N]  (sum over aperture)
        noise_1d           [N]  (sqrt(sum variances))
        flux_smooth        [N]
        noise_smooth       [N]
        snr_1d             [N]  (flux/noise)
        snr_smooth         [N]
    """
    # Aperture
    radius_pix = int(spectrum_radius / pixel_scale)
    ny, nx = flux_subcube.data.shape[1], flux_subcube.data.shape[2]
    aper = build_aperture_mask(ny, nx, radius_pix)

    # Wavelength window
    central_wavelength = ensure_angstrom(central_wavelength)
    spectrum_width = ensure_angstrom(spectrum_width)
    wave_min = central_wavelength - spectrum_width / 2
    wave_max = central_wavelength + spectrum_width / 2

    # Slice both subcubes
    flux_spec = flux_subcube.select_lambda(wave_min.value, wave_max.value)
    var_spec = stat_subcube.select_lambda(wave_min.value, wave_max.value)

    # Arrays
    wave = flux_spec.wave.coord()  # Å
    data3d = np.array(flux_spec.data.filled(np.nan), dtype=float)  # (Nλ, Ny, Nx)
    var3d = np.array(var_spec.data.filled(np.nan), dtype=float)    # (Nλ, Ny, Nx)

    # Apply aperture and sum spatially
    # Use NaNs so masked data doesn't contribute
    data_ap = data3d * aper[None, :, :]
    var_ap = var3d * aper[None, :, :]

    flux_1d = np.nansum(np.nansum(data_ap, axis=2), axis=1)    # Σ flux
    var_1d = np.nansum(np.nansum(var_ap, axis=2), axis=1)      # Σ variances
    var_1d[var_1d < 0] = np.nan
    noise_1d = np.sqrt(var_1d)                                 # sqrt(Σ var)

    # Smoothing
    dw = np.nanmedian(np.diff(wave))
    sigma_A = fwhm / 2.355
    sigma_pix = sigma_A / dw if dw and np.isfinite(dw) else 0.0
    if sigma_pix > 0:
        flux_smooth = gaussian_filter1d(flux_1d, sigma_pix)
        noise_smooth = gaussian_filter1d(noise_1d, sigma_pix)
    else:
        flux_smooth = flux_1d.copy()
        noise_smooth = noise_1d.copy()

    # S/N with safety
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_1d = np.divide(flux_1d, noise_1d, out=np.full_like(flux_1d, np.nan), where=np.isfinite(noise_1d) & (noise_1d > 0))
        snr_smooth = np.divide(flux_smooth, noise_smooth, out=np.full_like(flux_smooth, np.nan), where=np.isfinite(noise_smooth) & (noise_smooth > 0))

    return wave, flux_1d, noise_1d, flux_smooth, noise_smooth, snr_1d, snr_smooth, wave_min.value, wave_max.value


def plot_flux_plus_noise(wave, flux_1d, noise_1d, flux_smooth, noise_smooth,
                         wave_min, wave_max, central_wavelength, b_region, wavelengths,
                         spectrum_radius, output_path):
    plt.figure(figsize=(12, 4))

    # Flux and noise
    plt.step(wave, flux_1d, color='grey', where='mid', lw=1.0, alpha=0.7, label='Flux (aperture sum)')
    plt.plot(wave, flux_smooth, color='#B63DE2', lw=1.8, alpha=0.9, label='Flux (smoothed)')

    plt.step(wave, noise_1d, color='royalblue', where='mid', lw=1.0, alpha=0.9, label='Noise = sqrt(Σ var)')
    plt.plot(wave, noise_smooth, color='cyan', lw=1.8, alpha=0.9, label='Noise (smoothed)')

    # Band highlight
    lymin = central_wavelength.value - b_region
    lymax = central_wavelength.value + b_region
    plt.axvspan(lymin, lymax, color='lightblue', alpha=0.15, label=f"±{b_region} Å")

    # Line markers
    line_labels = ["Lyα", "N V", "Si II", "O I", "C II", "Si IV", "C IV"]
    for i, wl in enumerate(wavelengths):
        wl = wl.value if hasattr(wl, 'value') else wl
        if wl < wave_min or wl > wave_max:
            continue
        label = line_labels[i] if i < len(line_labels) else f"{wl:.1f} Å"
        plt.axvline(wl, color='orange', linestyle='--', lw=1.0, alpha=0.8)
        plt.text(wl + 3, 0.95, label, color='orange', rotation=90, va='top', ha='left', transform=plt.gca().get_xaxis_transform())

    plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.6)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux / Noise (same units)")
    plt.title(f"Flux and Noise spectra (aperture {2*spectrum_radius:.1f}\" diameter)")
    plt.xlim(wave_min, wave_max)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, pad_inches=0.2)
    plt.close()
    print(f"Saved flux+noise spectrum to {output_path}")


def plot_snr(wave, snr_1d, snr_smooth, wave_min, wave_max, central_wavelength, b_region, wavelengths,
             spectrum_radius, output_path):
    plt.figure(figsize=(12, 4))
    plt.step(wave, snr_1d, color='black', where='mid', lw=1.0, alpha=0.7, label='S/N')
    plt.plot(wave, snr_smooth, color='crimson', lw=1.8, alpha=0.9, label='S/N (smoothed)')

    # 1σ and 2σ reference lines
    plt.axhline(1.0, color='grey', linestyle='--', lw=1.0, alpha=0.8, label='1σ')
    plt.axhline(2.0, color='grey', linestyle=':',  lw=1.0, alpha=0.8, label='2σ')

    # Band highlight
    lymin = central_wavelength.value - b_region
    lymax = central_wavelength.value + b_region
    plt.axvspan(lymin, lymax, color='lightblue', alpha=0.15, label=f"±{b_region} Å")

    # Line markers
    line_labels = ["Lyα", "N V", "Si II", "O I", "C II", "Si IV", "C IV"]
    for i, wl in enumerate(wavelengths):
        wl = wl.value if hasattr(wl, 'value') else wl
        if wl < wave_min or wl > wave_max:
            continue
        label = line_labels[i] if i < len(line_labels) else f"{wl:.1f} Å"
        plt.axvline(wl, color='orange', linestyle='--', lw=1.0, alpha=0.8)
        plt.text(wl + 3, 0.95, label, color='orange', rotation=90, va='top', ha='left',
                 transform=plt.gca().get_xaxis_transform())

    plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.6)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("S/N")
    plt.title(f"S/N spectrum (aperture {2*spectrum_radius:.1f}\" diameter)")
    plt.xlim(wave_min, wave_max)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, pad_inches=0.2)
    plt.close()
    print(f"Saved S/N spectrum to {output_path}")


def make_flux_noise_and_snr_spectra(flux_subcube, stat_subcube, spectrum_radius, central_wavelength,
                                    spectrum_width, pixel_scale, b_region, fwhm, wavelengths,
                                    out_prefix):
    # Extract
    (wave, flux_1d, noise_1d, flux_smooth, noise_smooth,
     snr_1d, snr_smooth, wmin, wmax) = extract_aperture_spectra(
        flux_subcube, stat_subcube, spectrum_radius, central_wavelength, spectrum_width, pixel_scale, fwhm
    )

    # Plots
    plot_flux_plus_noise(
        wave, flux_1d, noise_1d, flux_smooth, noise_smooth,
        wmin, wmax, central_wavelength, b_region, wavelengths,
        spectrum_radius, output_path=f"{out_prefix}_flux_plus_noise.png"
    )

    plot_snr(
        wave, snr_1d, snr_smooth, wmin, wmax, central_wavelength, b_region, wavelengths,
        spectrum_radius, output_path=f"{out_prefix}_snr.png"
    )

    plot_flux_minus_noise(
        wave, flux_1d, noise_1d, flux_smooth, noise_smooth,
        wmin, wmax, central_wavelength, b_region, wavelengths,
        spectrum_radius, output_path=f"{out_prefix}_flux_minus_noise.png"
    )

    # Return useful arrays too
    return {
        "wave": wave,
        "flux": flux_1d,
        "noise": noise_1d,
        "snr": snr_1d,
        "flux_smooth": flux_smooth,
        "noise_smooth": noise_smooth,
        "snr_smooth": snr_smooth
    }


def plot_flux_minus_noise(wave, flux_1d, noise_1d, flux_smooth, noise_smooth,
                          wave_min, wave_max, central_wavelength, b_region, wavelengths,
                          spectrum_radius, output_path):
    """
    Plot flux minus noise (and smoothed versions) for a given aperture.
    """
    plt.figure(figsize=(12, 4))

    # Compute flux minus noise
    flux_minus_noise = flux_1d - noise_1d
    flux_minus_noise_smooth = flux_smooth - noise_smooth

    # Plot
    plt.step(wave, flux_minus_noise, color='grey', where='mid', lw=1.0, alpha=0.7, label='Flux - Noise')
    plt.plot(wave, flux_minus_noise_smooth, color='#B63DE2', lw=1.8, alpha=0.9, label='Flux - Noise (smoothed)')

    # Highlight band region
    lymin = central_wavelength.value - b_region
    lymax = central_wavelength.value + b_region
    plt.axvspan(lymin, lymax, color='lightblue', alpha=0.15, label=f"±{b_region} Å")

    # Line markers
    line_labels = ["Lyα", "N V", "Si II", "O I", "C II", "Si IV", "C IV"]
    for i, wl in enumerate(wavelengths):
        wl = wl.value if hasattr(wl, 'value') else wl
        if wl < wave_min or wl > wave_max:
            continue
        label = line_labels[i] if i < len(line_labels) else f"{wl:.1f} Å"
        plt.axvline(wl, color='orange', linestyle='--', lw=1.0, alpha=0.8)
        plt.text(wl + 3, 0.95, label, color='orange', rotation=90, va='top', ha='left',
                 transform=plt.gca().get_xaxis_transform())

    # Horizontal line at zero
    plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.6)

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux - Noise")
    plt.title(f"Flux minus Noise (aperture {2*spectrum_radius:.1f}\" diameter)")
    plt.xlim(wave_min, wave_max)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, pad_inches=0.2)
    plt.close()
    print(f"Saved flux-minus-noise spectrum to {output_path}")

def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load MUSE cubes
    cube = Cube(args.cube, ext=1, memmap=True)             # DATA
    stat_cube = Cube(args.cube, ext='STAT', memmap=True)   # STAT (variance)
    print("Full DATA and STAT cubes loaded")

    # Load object catalog
    df = pd.read_csv(args.objects)
    print(f"Columns: {df.columns}")
    print(f"Loaded {len(df)} objects from {args.objects}")

    for i, row in df.iterrows():
        ra, dec, z = row['ra'], row['dec'], row['z1_median']
        subcube_path = f"{args.output}source_{i+1}_subcube_{args.spatial_width}_{args.spectral_width}.fits"

        # Observed wavelength(s) of interest
        obs_wavelength = z_to_wavelength(z, args.rest_wavelength)
        wavelengths = z_to_wavelength(z, np.array(args.wavelengths))

        # Use existing subcube if available
        if args.use_saved_subcubes and os.path.exists(subcube_path):
            print(f"Using existing subcube: {subcube_path}")
            subcube = Cube(subcube_path, ext='DATA')
            stat_subcube = Cube(subcube_path, ext='STAT')

        else:
            # Convert coords and extract region
            x, y = ra_dec_to_xy(ra, dec, cube)
            print(f"Object {i+1}: RA={ra:.6f}, Dec={dec:.6f}, z={z:.3f} → x={x}, y={y}, λ={obs_wavelength.value:.2f} Å")

            # Extract DATA and STAT subcubes
            flux_sub = extract_subcube(
                cube, x, y, obs_wavelength,
                spatial_width=args.spatial_width,
                spectral_width=args.spectral_width,
                pixel_scale=args.pixel_scale
            )
            if flux_sub is None:
                continue

            stat_sub = extract_subcube(
                stat_cube, x, y, obs_wavelength,
                spatial_width=args.spatial_width,
                spectral_width=args.spectral_width,
                pixel_scale=args.pixel_scale
            )
            if stat_sub is None:
                continue

            # Save compact FITS with DATA + STAT
            data_hdr = flux_sub.data_header.copy()
            stat_hdr = stat_sub.data_header.copy()
            for k in ('NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3'):
                data_hdr.pop(k, None)
                stat_hdr.pop(k, None)

            hdul = fits.HDUList([
                fits.PrimaryHDU(header=flux_sub.primary_header),
                fits.ImageHDU(
                    data=flux_sub.data.filled(np.nan).astype(np.float32),
                    header=data_hdr,
                    name='DATA'
                ),
                fits.ImageHDU(
                    data=stat_sub.data.filled(np.nan).astype(np.float32),
                    header=stat_hdr,
                    name='STAT'
                ),
            ])
            hdul.writeto(subcube_path, overwrite=True)

            # Reload as mpdaf Cubes
            subcube = Cube(subcube_path, ext='DATA')
            stat_subcube = Cube(subcube_path, ext='STAT')

        if subcube is None or stat_subcube is None:
            print(f"Skipping object {i+1}: missing subcubes.")
            continue

        # Make all spectra and plots (flux+noise, SNR, flux-noise)
        out_prefix = f"{args.output}source_{i+1}"
        _ = make_flux_noise_and_snr_spectra(
            subcube, stat_subcube,
            spectrum_radius=args.spectrum_radius,
            central_wavelength=obs_wavelength,
            spectrum_width=args.spectrum_width,
            pixel_scale=args.pixel_scale,
            b_region=args.b_region,
            fwhm=args.smooth_fwhm,
            wavelengths=wavelengths,
            out_prefix=out_prefix
        )

    print("All sources processed. Now collecting PNGs into PDF...")

    # Collect ALL PNGs: flux+noise, SNR, flux-noise
    png_files = sorted(
        glob(f"{args.output}*_flux_plus_noise.png") +
        glob(f"{args.output}*_snr.png") +
        glob(f"{args.output}*_flux_minus_noise.png")
    )

    pdf_path = os.path.join(args.output, "all_sources_spectra.pdf")

    with PdfPages(pdf_path) as pdf:
        for png in png_files:
            img = mpimg.imread(png)

            # Extract source number
            match = re.search(r"source_(\d+)_", os.path.basename(png))
            source_num = match.group(1) if match else "Unknown"

            # Create figure
            plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100))
            plt.imshow(img)
            plt.axis('off')

            # Title with source #
            plt.title(f"Source {source_num}", fontsize=16, color='black')

            pdf.savefig(bbox_inches='tight', pad_inches=0.2)
            plt.close()

    print(f"Saved all spectra PNGs into {pdf_path}")


if __name__ == "__main__":
    main()



