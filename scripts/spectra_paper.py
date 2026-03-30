#!/usr/bin/env python3
"""
plot_lya_pubready.py

Publication-ready Lyα spectra with inset cutouts for MUSE + JWST data.

Produces
--------
  out_dir/singles/source_N_ap{key}.pdf   — one panel per source
  out_dir/detections_figure.pdf          — all flag=1 sources stacked
  out_dir/tentative_figure.pdf           — all flag=2 sources stacked
  out_dir/nondetections_figure.pdf       — all flag=0 sources stacked

Each panel: spectrum + three insets at top-right corner
  inset order left to right: Pseudo-NB | JELS | RGB


Usage
-----
python plot_lya_pubready.py \
    --csv          /path/to/jels_muse_sources.csv \
    --zminmax-csv  /path/to/jels_halpha_candidates_mosaic_all_with_zmin_zmax_lya.csv \
    --extractions  /path/to/extractions_dir \
    --out-dir      /path/to/output \
    --jels-template /path/JELS_v1_{band}_30mas_i2d.fits \
    --rgb-fits     /path/to/col.fits \
    [--aperture-key    0p6] \
    [--aperture-arcsec 0.6] \
    [--plot-window     300] \
    [--cutout-arcsec   3.0] \
    [--jels-pixscale   0.03] \
    [--rgb-pixscale    0.03] \
    [--muse-pixscale   0.2]
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
# Palette / colour helpers  (unchanged from original)
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

FIG_WIDTH_IN  = 8.0
ROW_HEIGHT_IN = 1.75


# ──────────────────────────────────────────────────────────────────────────────
# Global matplotlib style
# ──────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":          "serif",
    "font.size":            7,
    "axes.labelsize":       7,
    "xtick.labelsize":      6,
    "ytick.labelsize":      6,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "legend.fontsize":      5.5,
    "legend.frameon":       False,
    "legend.handlelength":  1.2,
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
# Pseudo-NB image  (KEY FIX: pixscale derived from cube WCS, not CLI arg)
# ──────────────────────────────────────────────────────────────────────────────

def make_pseudonb(cube_path, xpix, ypix,
                  wave, lya_center, half_width,
                  size_arcsec, fallback_pixscale=0.2):
    """
    Collapse MUSE cube over the Lya window and return a square stamp.

    Parameters
    ----------
    cube_path         : path to the per-source FITS cube
    xpix, ypix        : 0-based pixel coordinates of the source centre,
                        stored in the NPZ by lya_flux_pipeline.py via
                        coords_to_pixel(..., origin=0)  — used directly,
                        no recomputation needed.
    wave              : wavelength array (Å)
    lya_center        : Lya line centre (Å)
    half_width        : integration half-width (Å) — same value stored in
                        meta["integration_half_width"] by the pipeline
    size_arcsec       : stamp side length on sky (arcsec)
    fallback_pixscale : arcsec/pix used only if WCS read fails

    Returns
    -------
    stamp      : 2-D float array or None
    pixscale   : arcsec/pix actually used (from WCS when possible)
    """
    # ── Open cube and read WCS to get the true pixel scale ────────────────────
    try:
        with fits.open(cube_path) as hdul:
            cube_data = hdul["DATA"].data.astype(float)
            wcs_cube  = WCS(hdul["DATA"].header)

        # Derive pixscale the same way lya_flux_pipeline.py does it:
        #   pix_scales_deg = proj_plane_pixel_scales(wcs.celestial)
        #   pixscale = float(np.mean(pix_scales_deg) * 3600.0)
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

    # ── Collapse over wavelength window ───────────────────────────────────────
    sel = (wave >= lya_center - half_width) & (wave <= lya_center + half_width)
    if not np.any(sel):
        return None, pixscale

    img = np.nansum(cube_data[sel], axis=0)

    # ── Cut stamp centred on source pixel ─────────────────────────────────────
    # Use the same integer truncation as lya_flux_pipeline.py's
    # extract_pseudonb_image:  int(xpix - half_pix) … int(xpix + half_pix)
    half_pix = (size_arcsec / pixscale) / 2.0
    x0 = max(int(xpix - half_pix), 0)
    x1 = min(int(xpix + half_pix), img.shape[1])
    y0 = max(int(ypix - half_pix), 0)
    y1 = min(int(ypix + half_pix), img.shape[0])

    stamp = img[y0:y1, x0:x1]
    return (stamp if stamp.size > 1 else None), pixscale


# ──────────────────────────────────────────────────────────────────────────────
# JELS / RGB cutouts  (unchanged logic, WCS-based via Cutout2D)
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

    aperture_pix is now always computed from the WCS-derived pixscale so
    the circle radius matches the actual extraction aperture on the sky.
    """
    if img is None or img.size <= 1:
        ax.set_facecolor("#1a1a2e")
        ax.text(0.5, 0.5, "\u2014",
                color="white", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
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
            fontsize=4, fontweight="bold",
            fontfamily="sans-serif",
            path_effects=[withStroke(linewidth=1.0, foreground="black")])

    if show_aperture and aperture_pix is not None and img is not None:
        colour = aperture_colour if aperture_colour is not None else COLOURS["aperture"]
        ny, nx = img.shape[:2]
        ax.add_patch(Circle(
            ((nx - 1) / 2.0, (ny - 1) / 2.0),
            aperture_pix,
            edgecolor=colour,
            facecolor="none", lw=1.1,
        ))

    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444");  sp.set_linewidth(0.4)


def render_spectrum(ax, spec, meta, lya_center,
                    lya_wmin, lya_wmax, plot_window_ang,
                    jels_name=None,
                    row_i=0, n_rows=1):
    """
    Draw flux step, ±1σ fill, optional skew-Gaussian fit,
    Lyα dashed line, and search-window shading onto ax.
    Adds a JELS name tag as a right-hand axis label if provided.
    """
    wave  = spec["wave"]
    flux  = spec["flux"]
    noise = np.sqrt(np.clip(spec["var"], 0.0, np.inf))

    if np.isfinite(lya_wmin) and np.isfinite(lya_wmax):
        ax.axvspan(lya_wmin, lya_wmax,
                   color=COLOURS["search_span"], alpha=0.13,
                   label="Search window", zorder=1)

    ax.axhline(0.0, color=COLOURS["zeroline"], lw=0.5, ls=":", zorder=1)

    ax.fill_between(wave, -noise, noise,
                    step="mid",
                    color=COLOURS["noise_fill"], alpha=0.35,
                    label=r"$\pm1\sigma$", zorder=2)

    ax.step(wave, flux, where="mid",
            color=COLOURS["flux"], lw=0.55,
            label="Flux", zorder=3)

    ax.axvline(lya_center,
               color=COLOURS["lya_line"], ls="--", lw=1.1,
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
                color=COLOURS["fit"], lw=1.0,
                label="Skewed Gaussian", zorder=5)

    xlim = (lya_center - plot_window_ang, lya_center + plot_window_ang)
    ax.set_xlim(*xlim)
    ax.set_ylim(-70, 175)
    ax.set_xlabel(r"Wavelength [$\rm\AA$]")
    ax.set_ylabel(
        r"Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\rm\AA^{-1}$]",
        fontsize=5,
    )
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)

    # ── JELS name tag on the right-hand side, outside the axes ───────────────
    if jels_name is not None:
        ax.annotate(
            jels_name,
            xy=(1.005, 0.5),
            xycoords="axes fraction",
            ha="left", va="center",
            rotation=270,
            fontsize=6.5,
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
                band, plot_window_ang, aperture_arcsec,
                jels_pixscale, rgb_pixscale,
                jels_name=None,
                row_i=0, n_rows=1):
    """
    Full-width spectrum with three image cutouts at the top-right corner:
      [Pseudo-NB | JELS | RGB]  left to right.

    nb_pixscale is the WCS-derived pixel scale for the pseudo-NB panel —
    used to compute the correct aperture-circle radius in pixels.

    Cutouts are enlarged and may overlap the spectrum.
    A JELS name tag is added to the right spine if jels_name is provided.
    """
    inner   = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_cell)
    ax_spec = fig.add_subplot(inner[0, 0])
    render_spectrum(ax_spec, spec, meta,
                    lya_center, lya_wmin, lya_wmax,
                    plot_window_ang,
                    jels_name=jels_name,
                    row_i=row_i, n_rows=n_rows)

    # ── Inset geometry ────────────────────────────────────────────────────────
    inset_h  = 0.5
    inset_w  = 0.110
    gap      = 0.008
    top      = 0.97
    right    = 0.985
    bottom   = top - inset_h

    left_rgb  = right    - inset_w
    left_jels = left_rgb  - gap - inset_w
    left_nb   = left_jels - gap - inset_w

    try:
        band_nan = isinstance(band, float) and np.isnan(band)
    except (TypeError, ValueError):
        band_nan = False
    band_str = str(band) if (band is not None and not band_nan) else "-"

    # Aperture circle radii in pixels for each image type.
    # For pseudo-NB we use nb_pixscale (WCS-derived), matching the pipeline.
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

    The pseudo-NB is built with:
      - xpix/ypix read from the NPZ (0-based, same convention as the
        pipeline's coords_to_pixel call with origin=0)
      - pixscale derived live from the cube WCS via proj_plane_pixel_scales,
        identical to how lya_flux_pipeline.py computes it
      - integration_half_width from the NPZ, so the collapsed wavelength
        window matches what the pipeline actually used

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
    nb_pixscale = fallback_muse_pixscale  # overwritten below when cube opens OK

    cube_path = meta.get("cube_path")
    if cube_path and os.path.exists(str(cube_path)):
        # integration_half_width stored by the pipeline — use it so the
        # collapsed wavelength window is identical to what was measured
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

    # ── JELS name tag ─────────────────────────────────────────────────────────
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

def save_single(out_path, src, plot_window_ang, aperture_arcsec,
                jels_pixscale, rgb_pixscale):
    """Save one source as its own PDF."""
    fig = plt.figure(figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN))
    gs  = gridspec.GridSpec(1, 1, figure=fig,
                            left=0.090, right=0.995,
                            top=0.960,  bottom=0.240)
    build_panel(
        fig, gs[0, 0],
        src["spec"], src["meta"],
        src["lya_center"], src["lya_wmin"], src["lya_wmax"],
        src["nb_img"], src["nb_pixscale"],
        src["jels_img"], src["rgb_img"],
        src["band"], plot_window_ang, aperture_arcsec,
        jels_pixscale, rgb_pixscale,
        jels_name=src.get("jels_name"),
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_multirow(out_path, sources, plot_window_ang, aperture_arcsec,
                  jels_pixscale, rgb_pixscale):
    """Save a stacked multi-row figure (one row per source)."""
    n = len(sources)
    if n == 0:
        print(f"  [skip -- 0 sources] {out_path}")
        return

    bot = max(0.02, 0.10 / n)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * n))
    gs  = gridspec.GridSpec(
        n, 1, figure=fig,
        left=0.090, right=0.995,
        top=0.995,  bottom=bot,
        hspace=0.25,
    )

    for i, src in enumerate(sources):
        build_panel(
            fig, gs[i, 0],
            src["spec"], src["meta"],
            src["lya_center"], src["lya_wmin"], src["lya_wmax"],
            src["nb_img"], src["nb_pixscale"],
            src["jels_img"], src["rgb_img"],
            src["band"], plot_window_ang, aperture_arcsec,
            jels_pixscale, rgb_pixscale,
            jels_name=src.get("jels_name"),
            row_i=i, n_rows=n,
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Publication-ready Lya spectra + cutout panels."
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
    p.add_argument("--plot-window",      type=float, default=300.0,
                   help="Half-width of spectrum x-axis in Ang (default: 300).")
    p.add_argument("--cutout-arcsec",    type=float, default=3.0,
                   help="Cutout size on sky in arcsec (default: 3.0).")
    p.add_argument("--jels-pixscale",    type=float, default=0.03,
                   help="JELS pixel scale arcsec/pix (default: 0.03).")
    p.add_argument("--rgb-pixscale",     type=float, default=0.03,
                   help="RGB pixel scale arcsec/pix (default: 0.03).")
    p.add_argument("--muse-pixscale",    type=float, default=0.2,
                   help="MUSE pixel scale arcsec/pix — used only as fallback "
                        "if cube WCS cannot be read (default: 0.2).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    singles_dir = os.path.join(args.out_dir, "singles")
    os.makedirs(singles_dir, exist_ok=True)

    df      = load_source_csv(args.csv)
    zminmax = load_zminmax_csv(args.zminmax_csv)

    buckets = {0: [], 1: [], 2: []}

    # IDs to exclude from multirow plots (merger candidates)
    EXCLUDE_IDS = {3228, 6938, 7650}
    #EXCLUDE_IDS = {}

    print(f"Processing {len(df)} sources  [cmap: cmr.{THEME}] ...")

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
                         f"source_{row_index}_ap{args.aperture_key}.pdf"),
            src, args.plot_window, args.aperture_arcsec,
            args.jels_pixscale, args.rgb_pixscale,
        )

        src_id = row.get("ID")
        if flag in buckets:
            if src_id not in EXCLUDE_IDS:
                buckets[flag].append(src)

    for flag, name in [(1, "detections"), (2, "tentative"), (0, "nondetections")]:
        save_multirow(
            os.path.join(args.out_dir, f"{name}_figurep4.pdf"),
            buckets[flag],
            args.plot_window, args.aperture_arcsec,
            args.jels_pixscale, args.rgb_pixscale,
        )

    print("Done.")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# Example run
# ──────────────────────────────────────────────────────────────────────────────
"""
python spectra_paper.py \
    --csv         /cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv \
    --zminmax-csv /home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_with_zmin_zmax_lya.csv \
    --extractions /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/extractions \
    --out-dir     /cephfs/apatrick/musecosmos/scripts/sample_select/plots/pubready \
    --jels-template /home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits \
    --rgb-fits    /home/apatrick/P1/col.fits \
    --plot-window 300 \
    --cutout-arcsec 3.0
"""