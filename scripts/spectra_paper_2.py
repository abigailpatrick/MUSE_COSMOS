#!/usr/bin/env python3
"""
plot_lya_pubready_2col.py

Publication-ready Lyα spectra with inset cutouts for MUSE + JWST data.
OPTIMIZED FOR TWO-COLUMN MNRAS PAPERS: Two spectra per row.

Produces
--------
  out_dir/singles/source_N_ap{key}.pdf   — one panel per source
  out_dir/detections_figure.pdf          — 2 spectra per row, all flag=1 sources
  out_dir/tentative_figure.pdf           — 2 spectra per row, all flag=2 sources
  out_dir/nondetections_figure.pdf       — 2 spectra per row, all flag=0 sources

Each panel: compact spectrum + three insets at top-right corner
  inset order left to right: Pseudo-NB | JELS | RGB

"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import cmasher as cmr  # pip install cmasher

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.stats import skewnorm


# ──────────────────────────────────────────────────────────────────────────────
# Palette / colour helpers
# ──────────────────────────────────────────────────────────────────────────────

THEME = "torch"


def make_palette(name, n=8):
    hexes = cmr.take_cmap_colors(
        f"cmr.{name}", n, cmap_range=(0.10, 0.90), return_fmt="hex"
    )
    palette = {f"c{i}": h for i, h in enumerate(hexes)}
    palette["near_black"]   = mcolors.to_hex((0.10, 0.10, 0.12, 1.0))
    palette["neutral_grey"] = "#aab4c8"
    palette["white"]        = "#ffffff"
    palette["mid_grey"]     = "#888888"
    return palette


def lya_spectra_colours(palette):
    return {
        "flux":        palette["near_black"],
        "noise_fill":  palette["neutral_grey"],
        "fit":         palette["c6"],
        "lya_line":    palette["c5"],
        "search_span": palette["c3"],
        "aperture":    palette["c1"],
        "cutout_text": palette["white"],
        "zeroline":    palette["mid_grey"],
    }


P           = make_palette(THEME)
COLOURS     = lya_spectra_colours(P)
CUTOUT_CMAP = f"cmr.{THEME}"

LYA_REST_A = 1215.67

FIG_WIDTH_IN  = 8.5  # Standard MNRAS full width; will be split 2-col
ROW_HEIGHT_IN = 1.3  # Keep original height — tall spectra

# Extra whitespace padding (inches) added around individual PDFs for PowerPoint
SINGLE_PAD_IN = 0.25


# ──────────────────────────────────────────────────────────────────────────────
# Global matplotlib style
# ──────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":          "serif",
    "font.size":            6.5,  # Slightly smaller for compact layout
    "axes.labelsize":       6,
    "xtick.labelsize":      5.5,
    "ytick.labelsize":      5.5,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "axes.linewidth":       0.5,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.minor.width":    0.3,
    "ytick.minor.width":    0.3,
    "xtick.major.size":     2.0,
    "ytick.major.size":     2.0,
    "xtick.minor.size":     1.0,
    "ytick.minor.size":     1.0,
    "legend.fontsize":      5.0,
    "legend.frameon":       False,
    "legend.handlelength":  1.0,
})


# ──────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_source_csv(path):
    """Load main source catalogue; add row_index if absent."""
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        df.insert(0, "row_index", range(1, len(df) + 1))
    return df


def load_zminmax_csv(path):
    """
    Load z1_min / z1_max catalogue and convert redshifts to observed
    Lyα wavelengths (Å).  Returns DataFrame indexed by ID.
    """
    df = pd.read_csv(path)
    df["lya_wmin"] = LYA_REST_A * (1.0 + df["z1_min"])
    df["lya_wmax"] = LYA_REST_A * (1.0 + df["z1_max"])
    return df.set_index("ID")[["lya_wmin", "lya_wmax"]]


def load_extraction(extractions_dir, row_index, ap_key):
    """Load extraction.  Returns (spec_dict, meta_dict) or (None, None)."""
    path = os.path.join(extractions_dir, f"source_{row_index}_ap{ap_key}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    meta = {
        k: (data[k].item() if data[k].shape == () else data[k])
        for k in data.files
        if k not in ("wave", "flux", "var")
    }
    spec = {"wave": data["wave"], "flux": data["flux"], "var": data["var"]}
    return spec, meta


# ──────────────────────────────────────────────────────────────────────────────
# JELS name tag
# ──────────────────────────────────────────────────────────────────────────────

def get_jels_name(ra, dec):
    """
    Return the JELS name string for a source given its RA/Dec in degrees.

    Format: JELS Jhhmmss.s+ddmmss.s
    Follows the convention from the JELS team:
      'JELS J' + RA as hhmmss.s (hourangle) + Dec as ddmmss.s (signed, padded)
    Returns None if ra or dec is not finite.
    """
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ra_str  = coord.ra.to_string(sep='', precision=1, unit=u.hourangle)
    dec_str = coord.dec.to_string(sep='', precision=1, pad=True, alwayssign=True)
    return f"JELS J{ra_str}{dec_str}"


# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-NB image
# ──────────────────────────────────────────────────────────────────────────────

def make_pseudonb(cube_path, xpix, ypix,
                  wave, lya_center, half_width,
                  size_arcsec, fallback_pixscale=0.2):
    """
    Collapse MUSE cube over the Lya window and return a square stamp.

    Parameters
    ----------
    cube_path         : path to the per-source FITS cube
    xpix, ypix        : 0-based pixel coordinates of the source centre
    wave              : wavelength array (Å)
    lya_center        : Lya line centre (Å)
    half_width        : integration half-width (Å)
    size_arcsec       : stamp side length on sky (arcsec)
    fallback_pixscale : arcsec/pix used only if WCS read fails

    Returns
    -------
    stamp      : 2-D float array or None
    pixscale   : arcsec/pix actually used (from WCS when possible)
    """
    try:
        with fits.open(cube_path) as hdul:
            cube_data = hdul["DATA"].data.astype(float)
            wcs_cube  = WCS(hdul["DATA"].header)

        pix_scales_deg = proj_plane_pixel_scales(wcs_cube.celestial)
        pixscale = float(np.mean(pix_scales_deg) * 3600.0)

    except Exception as exc:
        print(f"  [warn] could not read WCS from {cube_path}: {exc}")
        print(f"  [warn] falling back to pixscale={fallback_pixscale} arcsec/pix")
        pixscale = fallback_pixscale
        try:
            with fits.open(cube_path) as hdul:
                cube_data = hdul["DATA"].data.astype(float)
        except Exception:
            return None, pixscale

    sel = (wave >= lya_center - half_width) & (wave <= lya_center + half_width)
    if not np.any(sel):
        return None, pixscale

    img = np.nansum(cube_data[sel], axis=0)

    half_pix = (size_arcsec / pixscale) / 2.0
    x0 = max(int(xpix - half_pix), 0)
    x1 = min(int(xpix + half_pix), img.shape[1])
    y0 = max(int(ypix - half_pix), 0)
    y1 = min(int(ypix + half_pix), img.shape[0])

    stamp = img[y0:y1, x0:x1]
    return (stamp if stamp.size > 1 else None), pixscale


# ──────────────────────────────────────────────────────────────────────────────
# JELS / RGB cutouts
# ──────────────────────────────────────────────────────────────────────────────

def make_jels_stamp(jels_template, ra, dec, band, size_arcsec, pixscale):
    """Return JELS cutout centred on (ra, dec), or None."""
    if band is None or (isinstance(band, float) and np.isnan(band)):
        return None
    path = jels_template.format(band=str(band))
    if not os.path.exists(path):
        return None
    with fits.open(path) as hdul:
        data = hdul[1].data
        wcs  = WCS(hdul[1].header)
    size_pix = int(size_arcsec / pixscale)
    cut = Cutout2D(data, SkyCoord(ra, dec, unit="deg"),
                   size=(size_pix, size_pix), wcs=wcs)
    return cut.data


def make_rgb_stamp(rgb_fits, ra, dec, size_arcsec, pixscale):
    """
    Return normalised RGB array (ny, nx, 3) centred on source, or None.
    Expects a 3-channel FITS stored as (3, ny, nx) or (ny, nx, 3).
    """
    if rgb_fits is None or not os.path.exists(rgb_fits):
        return None
    with fits.open(rgb_fits) as hdul:
        data = hdul[0].data
        wcs  = WCS(hdul[0].header).celestial
    if data is None or data.ndim != 3:
        return None
    cube = data if data.shape[0] == 3 else np.moveaxis(data, -1, 0)
    size_pix = int(size_arcsec / pixscale)
    pos = SkyCoord(ra, dec, unit="deg")
    channels = []
    for i in range(3):
        cut = Cutout2D(cube[i], pos, size=(size_pix, size_pix), wcs=wcs)
        channels.append(cut.data)
    rgb = np.stack(channels, axis=0)
    normed = np.zeros_like(rgb)
    for i in range(3):
        n    = simple_norm(rgb[i], "asinh", percent=99.5)
        span = max(n.vmax - n.vmin, 1e-30)
        normed[i] = np.clip((rgb[i] - n.vmin) / span, 0, 1)
    return np.nan_to_num(np.moveaxis(normed, 0, -1))


# ──────────────────────────────────────────────────────────────────────────────
# Skew-normal model
# ──────────────────────────────────────────────────────────────────────────────

def skew_gaussian(wave, flux_total, mu, sigma, alpha):
    return flux_total * skewnorm.pdf(wave, alpha, loc=mu, scale=sigma)


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def render_cutout(ax, img, cmap, label, aperture_pix,
                  show_aperture=False, aperture_colour=None):
    """
    Draw one cutout image into ax.
    """
    if img is None or img.size <= 1:
        ax.set_facecolor("#1a1a2e")
        ax.text(0.5, 0.5, "\u2014",
                color="white", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
    elif img.ndim == 3:
        ax.imshow(img, origin="lower", interpolation="nearest")
    else:
        norm = simple_norm(img, "asinh", percent=99.5)
        ax.imshow(img, origin="lower", cmap=cmap, norm=norm,
                  interpolation="nearest")

    from matplotlib.patheffects import withStroke
    ax.text(0.5, 0.95, label,
            color=COLOURS["cutout_text"],
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=3.5, fontweight="bold",
            fontfamily="sans-serif",
            path_effects=[withStroke(linewidth=0.8, foreground="black")])

    if show_aperture and aperture_pix is not None and img is not None:
        colour = aperture_colour if aperture_colour is not None else COLOURS["aperture"]
        ny, nx = img.shape[:2]
        ax.add_patch(Circle(
            ((nx - 1) / 2.0, (ny - 1) / 2.0),
            aperture_pix,
            edgecolor=colour,
            facecolor="none", lw=0.8,
        ))

    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444");  sp.set_linewidth(0.3)


def render_spectrum(ax, spec, meta, lya_center,
                    lya_wmin, lya_wmax, 
                    plot_window_left, plot_window_right,
                    jels_name=None):
    """
    Draw flux step, ±1σ fill, optional skew-Gaussian fit,
    Lyα dashed line, and search-window shading onto ax.
    
    Parameters
    ----------
    plot_window_left : wavelength range to left of lya_center (Å)
    plot_window_right : wavelength range to right of lya_center (Å)
    """
    wave  = spec["wave"]
    flux  = spec["flux"]
    noise = np.sqrt(np.clip(spec["var"], 0.0, np.inf))

    if np.isfinite(lya_wmin) and np.isfinite(lya_wmax):
        ax.axvspan(lya_wmin, lya_wmax,
                   color=COLOURS["search_span"], alpha=0.13,
                   label="Search window", zorder=1)

    ax.axhline(0.0, color=COLOURS["zeroline"], lw=0.4, ls=":", zorder=1)

    ax.fill_between(wave, -noise, noise,
                    step="mid",
                    color=COLOURS["noise_fill"], alpha=0.35,
                    label=r"$\pm1\sigma$", zorder=2)

    ax.step(wave, flux, where="mid",
            color=COLOURS["flux"], lw=0.5,
            label="Flux", zorder=3)

    ax.axvline(lya_center,
               color=COLOURS["lya_line"], ls="--", lw=0.9,
               label=r"Ly$\alpha$ (JELS)", zorder=4)

    if int(meta.get("fit_success", 0)) == 1:
        model = skew_gaussian(
            wave,
            meta.get("fit_flux"),
            meta.get("fit_mu"),
            meta.get("fit_sigma"),
            meta.get("fit_alpha"),
        )
        ax.plot(wave, model,
                color=COLOURS["fit"], lw=0.8,
                label="Skewed Gaussian", zorder=5)

    xlim = (lya_center - plot_window_left, lya_center + plot_window_right)
    ax.set_xlim(*xlim)
    ax.set_ylim(-60, 150)
    ax.set_xlabel(r"Wavelength [$\rm\AA$]", fontsize=5)
    ax.set_ylabel(
        r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\rm\AA^{-1}$]",
        fontsize=4.5,
    )
    ax.legend(loc="upper left", fontsize=4.5, ncol=2, framealpha=0.9)

    # ── JELS name tag on the right-hand side, outside the axes ───────────────
    if jels_name is not None:
        ax.annotate(
            jels_name,
            xy=(1.002, 0.5),
            xycoords="axes fraction",
            ha="left", va="center",
            rotation=270,
            fontsize=5.0,
            fontfamily="monospace",
            color=COLOURS["flux"],
            fontweight="bold",
            annotation_clip=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Full panel builder
# ──────────────────────────────────────────────────────────────────────────────

def build_panel(fig, outer_cell,
                spec, meta, lya_center, lya_wmin, lya_wmax,
                nb_img, nb_pixscale,
                jels_img, rgb_img,
                band, plot_window_left, plot_window_right, aperture_arcsec,
                jels_pixscale, rgb_pixscale,
                jels_name=None):
    """
    Spectrum with three compact image cutouts at the top-right corner:
      [Pseudo-NB | JELS | RGB]  left to right.

    Optimized for 2 per row in MNRAS paper.
    
    Parameters
    ----------
    plot_window_left : Å left of line centre
    plot_window_right : Å right of line centre
    """
    inner   = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_cell)
    ax_spec = fig.add_subplot(inner[0, 0])
    render_spectrum(ax_spec, spec, meta,
                    lya_center, lya_wmin, lya_wmax,
                    plot_window_left, plot_window_right,
                    jels_name=jels_name)

    # ── Inset geometry — large cutouts on right, more legend space on left ────
    inset_h  = 0.70  # Original height
    inset_w  = 0.130  # Larger width (back to ~original size)
    gap      = 0.008
    top      = 1.09
    right    = 0.992   # Closer to edge for more cutout space
    bottom   = top - inset_h

    left_rgb  = right    - inset_w
    left_jels = left_rgb  - gap - inset_w
    left_nb   = left_jels - gap - inset_w

    try:
        band_nan = isinstance(band, float) and np.isnan(band)
    except (TypeError, ValueError):
        band_nan = False
    band_str = str(band) if (band is not None and not band_nan) else "-"

    # Aperture circle radii in pixels
    ap_nb   = aperture_arcsec / nb_pixscale  if nb_pixscale   > 0 else None
    ap_jels = aperture_arcsec / jels_pixscale if jels_pixscale > 0 else None
    ap_rgb  = aperture_arcsec / rgb_pixscale  if rgb_pixscale  > 0 else None

    cutouts = [
        (left_nb,   nb_img,   CUTOUT_CMAP, "Pseudo-NB",        ap_nb,   True,  COLOURS["aperture"]),
        (left_jels, jels_img, CUTOUT_CMAP, "JELS " + band_str, ap_jels, True,  "white"),
        (left_rgb,  rgb_img,  None,        "PRIMER RGB",        ap_rgb,  True,  "white"),
    ]

    for left, img, cmap, label, ap_pix, show_ap, ap_colour in cutouts:
        ax_in = ax_spec.inset_axes([left, bottom, inset_w, inset_h])
        render_cutout(ax_in, img, cmap, label, ap_pix,
                      show_aperture=show_ap, aperture_colour=ap_colour)

    return ax_spec


# ──────────────────────────────────────────────────────────────────────────────
# Per-source data loader
# ──────────────────────────────────────────────────────────────────────────────

def load_one_source(row, zminmax_df,
                    extractions_dir, ap_key,
                    jels_template, rgb_fits,
                    cutout_arcsec, jels_pixscale,
                    rgb_pixscale, fallback_muse_pixscale):
    """
    Load spec + meta + all three cutout images for one catalogue row.

    Returns a dict ready for build_panel, or None if extraction missing.
    """
    row_index = int(row["row_index"])
    spec, meta = load_extraction(extractions_dir, row_index, ap_key)
    if spec is None:
        return None

    ra   = float(meta.get("ra_used",  np.nan))
    dec  = float(meta.get("dec_used", np.nan))
    band = row.get("band", None)

    lya_center = float(meta.get("lya_center", np.nan))

    src_id   = row.get("ID", None)
    lya_wmin = lya_wmax = np.nan
    if src_id is not None and src_id in zminmax_df.index:
        lya_wmin = float(zminmax_df.loc[src_id, "lya_wmin"])
        lya_wmax = float(zminmax_df.loc[src_id, "lya_wmax"])

    # ── Pseudo-NB ─────────────────────────────────────────────────────────────
    nb_img      = None
    nb_pixscale = fallback_muse_pixscale

    cube_path = meta.get("cube_path")
    if cube_path and os.path.exists(str(cube_path)):
        half_width = float(meta.get("integration_half_width", 10.0))

        nb_img, nb_pixscale = make_pseudonb(
            str(cube_path),
            xpix=float(meta.get("xpix")),
            ypix=float(meta.get("ypix")),
            wave=spec["wave"],
            lya_center=lya_center,
            half_width=half_width,
            size_arcsec=cutout_arcsec,
            fallback_pixscale=fallback_muse_pixscale,
        )

    jels_img = make_jels_stamp(jels_template, ra, dec, band,
                               cutout_arcsec, jels_pixscale)
    rgb_img  = make_rgb_stamp(rgb_fits, ra, dec,
                              cutout_arcsec, rgb_pixscale)

    jels_name = get_jels_name(ra, dec)

    return dict(
        spec=spec, meta=meta,
        lya_center=lya_center,
        lya_wmin=lya_wmin, lya_wmax=lya_wmax,
        nb_img=nb_img, nb_pixscale=nb_pixscale,
        jels_img=jels_img, rgb_img=rgb_img,
        band=band,
        jels_name=jels_name,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Figure savers
# ──────────────────────────────────────────────────────────────────────────────

def save_single(out_path, src, plot_window_left, plot_window_right,
                aperture_arcsec, jels_pixscale, rgb_pixscale):
    """
    Save one source as its own PDF for PowerPoint use.
    The spectrum panel is identical to the original; pad_inches adds a
    uniform white border around the outside without changing the axes.
    """
    single_width = FIG_WIDTH_IN / 2.2  # ~3.9 inches per spectrum

    fig = plt.figure(figsize=(single_width, ROW_HEIGHT_IN))

    # Original margins — spectrum is exactly as before
    gs = gridspec.GridSpec(
        1, 1, figure=fig,
        left=0.15, right=0.98,
        top=0.950, bottom=0.220,
    )

    build_panel(
        fig, gs[0, 0],
        src["spec"], src["meta"],
        src["lya_center"], src["lya_wmin"], src["lya_wmax"],
        src["nb_img"], src["nb_pixscale"],
        src["jels_img"], src["rgb_img"],
        src["band"], plot_window_left, plot_window_right, aperture_arcsec,
        jels_pixscale, rgb_pixscale,
        jels_name=src.get("jels_name"),
    )

    # bbox_inches='tight' captures everything (including inset cutouts that
    # extend above the axes), then pad_inches adds a uniform white border on
    # all four sides — this is the only difference from the original.
    fig.savefig(out_path, dpi=600,
                bbox_inches="tight", pad_inches=SINGLE_PAD_IN)
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_multirow(out_path, sources, plot_window_left, plot_window_right, 
                  aperture_arcsec, jels_pixscale, rgb_pixscale):
    """
    Save a stacked multi-row figure with TWO spectra per row.
    
    If odd number of sources, last row has 1 spectrum.
    """
    n_sources = len(sources)
    if n_sources == 0:
        print(f"  [skip -- 0 sources] {out_path}")
        return

    n_rows = (n_sources + 1) // 2  # Ceiling division for 2 per row
    
    fig = plt.figure(figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * n_rows))
    gs  = gridspec.GridSpec(
        n_rows, 2, figure=fig,
        left=0.08, right=0.98,    # Leave room for legend on left
        top=0.98,  bottom=0.02,
        hspace=0.30, wspace=0.15,  # Slightly more space between columns
    )

    idx = 0
    for row in range(n_rows):
        for col in range(2):
            if idx >= n_sources:
                break
            
            src = sources[idx]
            build_panel(
                fig, gs[row, col],
                src["spec"], src["meta"],
                src["lya_center"], src["lya_wmin"], src["lya_wmax"],
                src["nb_img"], src["nb_pixscale"],
                src["jels_img"], src["rgb_img"],
                src["band"], plot_window_left, plot_window_right, 
                aperture_arcsec,
                jels_pixscale, rgb_pixscale,
                jels_name=src.get("jels_name"),
            )
            idx += 1

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path} ({n_sources} sources, {n_rows} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Publication-ready Lya spectra for 2-column MNRAS papers. "
                    "Two spectra per row."
    )
    p.add_argument("--csv", required=True,
                   help="Main source catalogue (jels_muse_sources.csv). "
                        "Must contain: ID, lya_detect_flag.")
    p.add_argument("--zminmax-csv", required=True,
                   help="Catalogue with z1_min / z1_max columns. "
                        "Joined to main catalogue on ID.")
    p.add_argument("--extractions", required=True,
                   help="Directory containing source_N_apKEY.npz files.")
    p.add_argument("--out-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--jels-template",
                   default="/home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits",
                   help="JELS FITS path template with {band} placeholder.")
    p.add_argument("--rgb-fits", default=None,
                   help="Path to PRIMER RGB FITS (optional).")
    p.add_argument("--aperture-key",     default="0p6",
                   help="Aperture key in NPZ filenames (default: 0p6).")
    p.add_argument("--aperture-arcsec",  type=float, default=0.6,
                   help="Aperture radius in arcsec for circle overlay (default: 0.6).")
    p.add_argument("--plot-window-left",  type=float, default=120.0,
                   help="Wavelength range left of Lyα centre in Å (default: 120).")
    p.add_argument("--plot-window-right", type=float, default=280.0,
                   help="Wavelength range right of Lyα centre in Å (default: 280).")
    p.add_argument("--cutout-arcsec",    type=float, default=3.0,
                   help="Cutout size on sky in arcsec (default: 3.0).")
    p.add_argument("--jels-pixscale",    type=float, default=0.03,
                   help="JELS pixel scale arcsec/pix (default: 0.03).")
    p.add_argument("--rgb-pixscale",     type=float, default=0.03,
                   help="RGB pixel scale arcsec/pix (default: 0.03).")
    p.add_argument("--muse-pixscale",    type=float, default=0.2,
                   help="MUSE pixel scale arcsec/pix — fallback if WCS fails "
                        "(default: 0.2).")
    p.add_argument("--single-pad",       type=float, default=SINGLE_PAD_IN,
                   help=f"Whitespace padding (inches) around individual PDFs "
                        f"(default: {SINGLE_PAD_IN}).  Increase for more "
                        f"breathing room in PowerPoint.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Allow the user to override the module-level constant at runtime
    global SINGLE_PAD_IN
    SINGLE_PAD_IN = args.single_pad

    singles_dir = os.path.join(args.out_dir, "singles")
    os.makedirs(singles_dir, exist_ok=True)

    df      = load_source_csv(args.csv)
    zminmax = load_zminmax_csv(args.zminmax_csv)

    buckets = {0: [], 1: [], 2: []}

    # IDs to exclude from multirow plots (merger candidates, etc.)
    EXCLUDE_IDS = {3228, 6906, 7650}

    print(f"Processing {len(df)} sources (2-col layout)  [cmap: cmr.{THEME}] ...")
    print(f"  Spectrum window: [{-args.plot_window_left:+.0f}, {args.plot_window_right:+.0f}] Å "
          f"around Lyα")
    print(f"  Single-PDF padding: {SINGLE_PAD_IN:.2f} in")

    for _, row in df.iterrows():
        row_index = int(row["row_index"])
        flag      = int(row.get("lya_detect_flag", 0))

        src = load_one_source(
            row, zminmax,
            args.extractions, args.aperture_key,
            args.jels_template, args.rgb_fits,
            args.cutout_arcsec,
            args.jels_pixscale, args.rgb_pixscale,
            fallback_muse_pixscale=args.muse_pixscale,
        )

        if src is None:
            print(f"  [skip] row_index={row_index} -- extraction not found")
            continue

        save_single(
            os.path.join(singles_dir,
                         f"source_{row_index}_ap{args.aperture_key}.png"),
            src, args.plot_window_left, args.plot_window_right, 
            args.aperture_arcsec,
            args.jels_pixscale, args.rgb_pixscale,
        )

        src_id = row.get("ID")
        if flag in buckets:
            if src_id not in EXCLUDE_IDS:
                buckets[flag].append(src)

    for flag, name in [(1, "detections"), (2, "tentative"), (0, "nondetections")]:
        save_multirow(
            os.path.join(args.out_dir, f"{name}_figure.pdf"),
            buckets[flag],
            args.plot_window_left, args.plot_window_right,
            args.aperture_arcsec,
            args.jels_pixscale, args.rgb_pixscale,
        )

    print("Done.")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# Example run
# ──────────────────────────────────────────────────────────────────────────────
"""
python spectra_paper_2.py \
    --csv         /cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv \
    --zminmax-csv /home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_with_zmin_zmax_lya.csv \
    --extractions /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/extractions \
    --out-dir     /cephfs/apatrick/musecosmos/scripts/sample_select/plots/pubready_2col \
    --jels-template /home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits \
    --rgb-fits    /home/apatrick/P1/col.fits \
    --plot-window-left 120 \
    --plot-window-right 280 \
    --single-pad  0.25
"""
