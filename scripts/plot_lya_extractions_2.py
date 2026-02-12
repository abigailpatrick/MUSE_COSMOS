#!/usr/bin/env python3
"""
Plot Lyα spectra with cutouts in a single horizontal row (aperture 0p6 by default).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from scipy.stats import skewnorm
from matplotlib.patches import Circle


LYA_REST_A = 1215.67


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def read_csv(path):
    """
    Read CSV with source information.
    """
    return pd.read_csv(path)


def build_wave_axis(wcs, nwave):
    """
    Build wavelength axis from a cube WCS.
    """
    pix = np.arange(nwave)
    pix_world = wcs.wcs_pix2world(
        np.zeros_like(pix) + 1,
        np.zeros_like(pix) + 1,
        pix + 1,
        0,
    )
    return np.array(pix_world[2], dtype=float)


def skew_model(wave, flux_total, mu, sigma, alpha):
    """
    Skew-normal model with integrated flux normalization.
    """
    return flux_total * skewnorm.pdf(wave, alpha, loc=mu, scale=sigma)


def load_extraction(extractions_dir, row_index, ap_key):
    """
    Load NPZ extraction file.
    """
    path = os.path.join(extractions_dir, f"source_{row_index}_ap{ap_key}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    meta = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files if k not in ("wave", "flux", "var")}
    spec = {
        "wave": data["wave"],
        "flux": data["flux"],
        "var": data["var"],
    }
    return spec, meta


def extract_pseudonb(cube_path, xpix, ypix, pixscale, wave, center, half_width, size_arcsec):
    """
    Extract pseudo-NB cutout from cube.
    """
    with fits.open(cube_path) as hdul:
        data = hdul["DATA"].data.astype(float)
        wcs = WCS(hdul["DATA"].header)

    sel = (wave >= center - half_width) & (wave <= center + half_width)
    if not np.any(sel):
        return np.zeros((1, 1))
    img = np.nansum(data[sel, :, :], axis=0)

    half_pix = (size_arcsec / pixscale) / 2.0
    x0, x1 = int(xpix - half_pix), int(xpix + half_pix)
    y0, y1 = int(ypix - half_pix), int(ypix + half_pix)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, img.shape[1])
    y1 = min(y1, img.shape[0])
    return img[y0:y1, x0:x1]


def jels_cutout(jels_template, ra, dec, band, cutout_size, pixel_scale):
    """
    JWST JELS cutout in correct filter.
    """
    if band is None or (isinstance(band, float) and np.isnan(band)):
        return None, None
    path = jels_template.format(band=str(band))
    if not os.path.exists(path):
        return None, path

    with fits.open(path) as hdul:
        data = hdul[1].data
        wcs = WCS(hdul[1].header)

    position = SkyCoord(ra, dec, unit="deg")
    cut = Cutout2D(data, position=position, size=cutout_size, wcs=wcs)
    return cut.data, path


def rgb_cutout(primer_path, ra, dec, cutout_size):
    """
    Create RGB cutout from PRIMER mosaic.

    Inputs
        primer_path : Path to RGB FITS (3, ny, nx) or (ny, nx, 3).
        ra, dec     : Coordinates in degrees.
        cutout_size : (nx, ny) in pixels.

    Output
        rgb_image : (ny, nx, 3) array or None.
    """
    if primer_path is None:
        return None
    if not os.path.exists(primer_path):
        return None

    with fits.open(primer_path) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header).celestial

    if data is None:
        return None

    # Normalize shape to (3, ny, nx)
    if data.ndim == 3 and data.shape[0] == 3:
        rgb = data
    elif data.ndim == 3 and data.shape[-1] == 3:
        rgb = np.moveaxis(data, -1, 0)
    else:
        return None

    position = SkyCoord(ra, dec, unit="deg")
    cutouts = []
    for i in range(3):
        cut = Cutout2D(rgb[i], position=position, size=cutout_size, wcs=wcs)
        cutouts.append(cut.data)

    rgb_cutout = np.stack(cutouts, axis=0)

    # Normalize each channel
    rgb_norm = np.zeros_like(rgb_cutout)
    for i in range(3):
        norm = simple_norm(rgb_cutout[i], "asinh", percent=99.5)
        rgb_norm[i] = (rgb_cutout[i] - norm.vmin) / (norm.vmax - norm.vmin)
    rgb_norm = np.clip(rgb_norm, 0, 1)
    rgb_image = np.moveaxis(rgb_norm, 0, -1)
    rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)

    return rgb_image


def plot_panel(
    outpath,
    spec,
    meta,
    lya_center,
    band,
    jels_data,
    nb_img,
    rgb_img,
    plot_window_ang,
    cutout_title,
    aperture_arcsec,
):
    """
    Build spectrum plot with 3 inset cutouts on the top-right.
    """
    wave = spec["wave"]
    flux = spec["flux"]
    var = spec["var"]
    noise = np.sqrt(np.clip(var, 0.0, np.inf))

    fig = plt.figure(figsize=(19.0, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[4.5, 1, 1, 1], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax.step(wave, flux, where="mid", color="black", lw=1.0, alpha=0.75, label="Flux")
    ax.fill_between(wave, -noise, noise, color="gray", alpha=0.3, step="mid", label=r"$\pm 1\sigma$")

    # Fit (if available)
    if int(meta.get("fit_success", 0)) == 1:
        model = skew_model(wave, meta.get("fit_flux"), meta.get("fit_mu"), meta.get("fit_sigma"), meta.get("fit_alpha"))
        ax.plot(wave, model, color="crimson", lw=1.6, label="Skewed Gaussian")

    # Catalog Lyα center
    ax.axvline(lya_center, color="orange", ls="--", lw=1.4, label="Lyα center (catalog)")

    # Limits
    xlim = (lya_center - plot_window_ang, lya_center + plot_window_ang)
    ax.set_xlim(*xlim)

    # Headroom
    in_window = (wave >= xlim[0]) & (wave <= xlim[1])
    if np.any(in_window):
        y_min = np.nanmin(flux[in_window])
        y_max = np.nanmax(flux[in_window])
    else:
        y_min = np.nanmin(flux)
        y_max = np.nanmax(flux)
    y_range = max(y_max - y_min, 1e-12)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.5 * y_range)

    ax.set_ylabel(r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax.set_xlabel("Wavelength [Å]")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # Cutouts in a line to the right of the spectrum
    ax1 = fig.add_subplot(gs[0, 1])
    if nb_img is not None and nb_img.size > 1:
        norm = simple_norm(nb_img, "asinh", percent=99.5)
        ax1.imshow(nb_img, origin="lower", cmap="magma", norm=norm)
        # Aperture circle
        if np.isfinite(aperture_arcsec) and aperture_arcsec > 0:
            pixscale = float(meta.get("pixscale", np.nan))
            if np.isfinite(pixscale) and pixscale > 0:
                ny, nx = nb_img.shape
                r_pix = aperture_arcsec / pixscale
                circ = Circle(((nx - 1) / 2.0, (ny - 1) / 2.0), r_pix, edgecolor="cyan", facecolor="none", lw=1.5)
                ax1.add_patch(circ)
    else:
        ax1.text(0.5, 0.5, "NB missing", ha="center", va="center")
    ax1.set_title(f"Pseudo-NB ({meta.get('cutout_arcsec', np.nan):.1f}\")", fontsize=8)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    if jels_data is not None and jels_data.size > 1:
        norm = simple_norm(jels_data, "asinh", percent=99.5)
        ax2.imshow(jels_data, origin="lower", cmap="magma", norm=norm)
    else:
        ax2.text(0.5, 0.5, "JELS missing", ha="center", va="center")
    if band is not None:
        ax2.set_title(f"JELS {band} ({meta.get('cutout_arcsec', np.nan):.1f}\")", fontsize=8)
    else:
        ax2.set_title(f"JELS ({meta.get('cutout_arcsec', np.nan):.1f}\")", fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[0, 3])
    if rgb_img is not None:
        ax3.imshow(rgb_img, origin="lower")
    else:
        ax3.text(0.5, 0.5, "RGB placeholder", ha="center", va="center")
    ax3.set_title(f"RGB ({meta.get('cutout_arcsec', np.nan):.1f}\")", fontsize=8)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax.set_title(cutout_title, fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.22, wspace=0.10)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(outpath)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Plot spectra + cutouts from saved extractions.")
    parser.add_argument("--lya-csv", required=True, help="Path to lya_flux_ap0p6.csv")
    parser.add_argument("--extractions-dir", required=True, help="Directory with saved NPZ extractions")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--aperture-key", default="0p6", help="Aperture key (e.g., 0p6)")
    parser.add_argument("--plot-window-ang", type=float, default=300.0, help="Half-width for spectrum plot")
    parser.add_argument("--nb-cutout-arcsec", type=float, default=3.0, help="Pseudo-NB cutout size (arcsec)")
    parser.add_argument("--jels-template", default="/home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits",
                        help="JELS FITS template with {band}")
    parser.add_argument("--jels-pixscale", type=float, default=0.03, help="JELS pixel scale (arcsec/pix)")
    parser.add_argument("--rgb-fits", default=None, help="Path to PRIMER RGB FITS (e.g., /home/apatrick/P1/col.fits)")
    parser.add_argument("--rgb-pixscale", type=float, default=0.03, help="RGB pixel scale (arcsec/pix)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Cutout size: {args.nb_cutout_arcsec:.2f}\" on a side")

    df = read_csv(args.lya_csv)
    df = df[df["aperture_arcsec"] == 0.6] if "aperture_arcsec" in df.columns else df

    for _, row in df.iterrows():
        row_index = row["row_index"]
        spec, meta = load_extraction(args.extractions_dir, row_index, args.aperture_key)
        if spec is None:
            continue

        lya_center = row.get("lya_center", meta.get("lya_center"))
        ra = row.get("ra_used", meta.get("ra_used"))
        dec = row.get("dec_used", meta.get("dec_used"))
        band = row.get("band", None)

        # Pseudo-NB
        nb_img = None
        cube_path = meta.get("cube_path")
        if cube_path and os.path.exists(cube_path):
            nb_img = extract_pseudonb(
                cube_path=cube_path,
                xpix=float(meta.get("xpix")),
                ypix=float(meta.get("ypix")),
                pixscale=float(meta.get("pixscale")),
                wave=spec["wave"],
                center=float(meta.get("lya_center")),
                half_width=float(meta.get("integration_half_width")),
                size_arcsec=args.nb_cutout_arcsec,
            )

        # JELS cutout
        jels_size_pix = int(args.nb_cutout_arcsec / args.jels_pixscale)
        jels_data, jels_path = jels_cutout(
            args.jels_template,
            ra=ra,
            dec=dec,
            band=band,
            cutout_size=(jels_size_pix, jels_size_pix),
            pixel_scale=args.jels_pixscale,
        )

        # RGB cutout (optional)
        rgb_size_pix = int(args.nb_cutout_arcsec / args.rgb_pixscale)
        rgb_img = rgb_cutout(
            primer_path=args.rgb_fits,
            ra=ra,
            dec=dec,
            cutout_size=(rgb_size_pix, rgb_size_pix),
        )

        meta["cutout_arcsec"] = float(args.nb_cutout_arcsec)

        outpath = os.path.join(args.out_dir, f"source_{row_index}_ap{args.aperture_key}_panel.png")
        title = f"ID={row.get('ID', row_index)} | band={band}"
        plot_panel(
            outpath=outpath,
            spec=spec,
            meta=meta,
            lya_center=lya_center,
            band=band,
            jels_data=jels_data,
            nb_img=nb_img,
            rgb_img=rgb_img,
            plot_window_ang=args.plot_window_ang,
            cutout_title=title,
            aperture_arcsec=float(row.get("aperture_arcsec", 0.6)),
        )


if __name__ == "__main__":
    main()

"""
python plot_lya_extractions.py \
  --lya-csv /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv \
  --extractions-dir /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/extractions \
  --out-dir /cephfs/apatrick/musecosmos/scripts/sample_select/plots \
  --plot-window-ang 300 \
  --rgb-fits /home/apatrick/P1/col.fits

 """
