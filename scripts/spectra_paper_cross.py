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

CROSSMATCH OVERLAY
------------------
If --crossmatch-csv is provided (primer_jels_crossmatch.csv), a cross (+)
is drawn on all three cutouts at the PRIMER source position for any JELS ID
that appears in the matched_jels_ids column.  Multiple matches are supported
(e.g. "6906;6938" → both IDs get a cross at that PRIMER position).

Usage
-----
python plot_lya_pubready.py \
    --csv          /path/to/jels_muse_sources.csv \
    --zminmax-csv  /path/to/jels_halpha_candidates_mosaic_all_with_zmin_zmax_lya.csv \
    --extractions  /path/to/extractions_dir \
    --out-dir      /path/to/output \
    --jels-template /path/JELS_v1_{band}_30mas_i2d.fits \
    --rgb-fits     /path/to/col.fits \
    --crossmatch-csv /path/to/primer_jels_crossmatch.csv \
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
    palette["cross"]        = "#ff4466"   # colour used for crossmatch markers
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
        "cross":       palette["cross"],
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


def load_crossmatch_csv(path):
    """
    Load the PRIMER–JELS crossmatch catalogue.

    The matched_jels_ids column may contain multiple IDs separated by ';'
    (e.g. "6906;6938").  Returns a dict mapping each JELS integer ID to a
    list of (ra_deg, dec_deg) tuples for all PRIMER sources that match it.

    Parameters
    ----------
    path : str or None
        Path to primer_jels_crossmatch.csv, or None / empty string if
        crossmatch overlay is not requested.

    Returns
    -------
    dict[int, list[tuple[float, float]]]
        {jels_id: [(ra1, dec1), (ra2, dec2), ...], ...}
        Returns an empty dict if path is None/empty or file is missing.
    """
    if not path or not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    lookup = {}   # jels_id (int) → [(ra, dec), ...]

    for _, row in df.iterrows():
        ra  = float(row["ALPHA_J2000"])
        dec = float(row["DELTA_J2000"])
        ids_raw = str(row["matched_jels_ids"]).strip()

        for id_str in ids_raw.split(";"):
            id_str = id_str.strip()
            if not id_str:
                continue
            try:
                jels_id = int(float(id_str))
            except ValueError:
                continue
            lookup.setdefault(jels_id, []).append((ra, dec))

    n_ids    = len(lookup)
    n_coords = sum(len(v) for v in lookup.values())
    print(f"  [crossmatch] loaded {n_coords} PRIMER positions "
          f"mapped to {n_ids} unique JELS IDs from {path}")
    return lookup


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
    """
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ra_str  = coord.ra.to_string(sep='', precision=1, unit=u.hourangle)
    dec_str = coord.dec.to_string(sep='', precision=1, pad=True, alwayssign=True)
    return f"JELS J{ra_str}{dec_str}"


# ──────────────────────────────────────────────────────────────────────────────
# Crossmatch pixel-coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def sky_to_pixel_pseudonb(primer_ra, primer_dec,
                           source_ra, source_dec,
                           nb_pixscale,
                           stamp_shape):
    """
    Convert a sky position (primer_ra, primer_dec) to pixel coordinates
    inside a pseudo-NB stamp that was built by make_pseudonb().

    make_pseudonb() crops a rectangular region:
        x: [int(xpix - half_pix), int(xpix + half_pix)]
        y: [int(ypix - half_pix), int(ypix + half_pix)]
    and the source centre is not guaranteed to lie exactly at the centre of
    the stamp (integer truncation).  We therefore compute the offset in
    arcsec between the PRIMER source and the MUSE source, convert to pixels
    using nb_pixscale, and add that offset to the stamp centre.

    Parameters
    ----------
    primer_ra, primer_dec : float  — PRIMER source position (deg)
    source_ra,  source_dec : float  — MUSE/JELS source position (deg)
    nb_pixscale            : float  — arcsec/pix of the pseudo-NB
    stamp_shape            : (ny, nx) tuple

    Returns
    -------
    (x_pix, y_pix) : pixel coordinates in the stamp (0-based, origin=lower)
                     May be outside [0, nx) / [0, ny) if off-stamp.
    """
    ny, nx = stamp_shape[:2]

    # Offset in arcsec (small-angle, cos(dec) correction on RA)
    cos_dec = np.cos(np.deg2rad(source_dec))
    dra_arcsec  = (primer_ra  - source_ra)  * cos_dec * 3600.0
    ddec_arcsec = (primer_dec - source_dec) * 3600.0

    # Convert to pixels  (RA increases left → negative x direction)
    dx_pix =  dra_arcsec  / nb_pixscale   # note: RA east = +x in imshow lower-origin
    dy_pix =  ddec_arcsec / nb_pixscale

    # Stamp centre (accounting for integer truncation in make_pseudonb)
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    return cx + dx_pix, cy + dy_pix


def sky_to_pixel_cutout2d(primer_ra, primer_dec,
                           cutout_center_ra, cutout_center_dec,
                           pixscale,
                           stamp_shape):
    """
    Convert a sky position to pixel coordinates inside a Cutout2D stamp.

    Cutout2D centres the stamp on (cutout_center_ra, cutout_center_dec) with
    the requested pixel size; fractional-pixel offsets are absorbed into the
    centre.  We use the same small-angle approximation as above.

    The x-axis direction convention for JWST/HST images (FITS standard) has
    RA increasing to the left (negative column direction), so:
        dx_pix = -dra_arcsec / pixscale

    Parameters
    ----------
    primer_ra, primer_dec         : float  — PRIMER position (deg)
    cutout_center_ra,
    cutout_center_dec             : float  — centre of the Cutout2D (deg)
                                            (= source RA/Dec for JELS & RGB)
    pixscale                      : float  — arcsec/pix
    stamp_shape                   : (ny, nx)

    Returns
    -------
    (x_pix, y_pix) pixel coords (0-based, origin=lower)
    """
    ny, nx = stamp_shape[:2]

    cos_dec = np.cos(np.deg2rad(cutout_center_dec))
    dra_arcsec  = (primer_ra  - cutout_center_ra)  * cos_dec * 3600.0
    ddec_arcsec = (primer_dec - cutout_center_dec) * 3600.0

    # FITS/WCS: RA increases to the left → dx_pix is *negative* dra
    dx_pix = -dra_arcsec / pixscale
    dy_pix =  ddec_arcsec / pixscale

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    return cx + dx_pix, cy + dy_pix


def draw_cross(ax, x, y, size=4, color=None, lw=1.0, zorder=10):
    """
    Draw a '+' cross marker at pixel position (x, y) in ax.

    Uses ax.plot rather than scatter so the cross size is in data (pixel)
    units and scales correctly with the image.

    Parameters
    ----------
    ax   : matplotlib Axes (the cutout inset axes)
    x, y : float  — pixel position (0-based, origin=lower)
    size : float  — half-length of each arm in pixels
    """
    if color is None:
        color = COLOURS["cross"]
    ax.plot([x - size, x + size], [y, y],
            color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    ax.plot([x, x], [y - size, y + size],
            color=color, lw=lw, zorder=zorder, solid_capstyle="round")


# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-NB image  (pixscale derived from cube WCS, not CLI arg)
# ──────────────────────────────────────────────────────────────────────────────

def make_pseudonb(cube_path, xpix, ypix,
                  wave, lya_center, half_width,
                  size_arcsec, fallback_pixscale=0.2):
    """
    Collapse MUSE cube over the Lya window and return a square stamp.

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
# JELS / RGB cutouts  (WCS-based via Cutout2D)
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
                  show_aperture=False, aperture_colour=None,
                  crossmatch_positions=None):
    """
    Draw one cutout image into ax, then overlay:
      • aperture circle (if show_aperture)
      • crossmatch crosses (if crossmatch_positions is not empty)

    Parameters
    ----------
    crossmatch_positions : list of (x_pix, y_pix) tuples, or None
        Pixel coordinates *within this stamp* at which to draw crosses.
        Positions outside the stamp bounds are silently skipped.
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

    # ── Crossmatch crosses ────────────────────────────────────────────────────
    if crossmatch_positions and img is not None and img.size > 1:
        ny, nx = img.shape[:2]
        cross_size = max(2.5, min(nx, ny) * 0.08)   # ~8 % of stamp, min 2.5 px
        for (xc, yc) in crossmatch_positions:
            # Only draw if the cross centre is within the stamp
            if 0 <= xc < nx and 0 <= yc < ny:
                draw_cross(ax, xc, yc,
                           size=cross_size,
                           color=COLOURS["cross"],
                           lw=1.2, zorder=15)

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
                row_i=0, n_rows=1,
                primer_positions=None,
                source_ra=None, source_dec=None):
    """
    Full-width spectrum with three image cutouts at the top-right corner:
      [Pseudo-NB | JELS | RGB]  left to right.

    Parameters
    ----------
    primer_positions : list of (ra_deg, dec_deg) or None
        PRIMER crossmatch positions to overlay as crosses on all three cutouts.
    source_ra, source_dec : float or None
        Sky position of the JELS/MUSE source (used to compute cutout-frame
        pixel positions for the crosses).

    Returns ax_spec so callers can add per-row titles.
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

    ap_nb   = aperture_arcsec / nb_pixscale  if nb_pixscale   > 0 else None
    ap_jels = aperture_arcsec / jels_pixscale if jels_pixscale > 0 else None
    ap_rgb  = aperture_arcsec / rgb_pixscale  if rgb_pixscale  > 0 else None

    # ── Compute crossmatch pixel positions for each cutout type ───────────────
    # We resolve pixel positions lazily here, just before rendering, so we
    # can pass the correct (img.shape, pixscale, centre) for each panel.

    def resolve_nb_crosses(img):
        """Pixel coords of all PRIMER sources in the pseudo-NB stamp."""
        if not primer_positions or img is None or img.size <= 1:
            return []
        if source_ra is None or source_dec is None:
            return []
        coords = []
        for (p_ra, p_dec) in primer_positions:
            xc, yc = sky_to_pixel_pseudonb(
                p_ra, p_dec,
                source_ra, source_dec,
                nb_pixscale, img.shape,
            )
            coords.append((xc, yc))
        return coords

    def resolve_wcs_crosses(img, pixscale):
        """Pixel coords of all PRIMER sources in a Cutout2D stamp."""
        if not primer_positions or img is None or img.size <= 1:
            return []
        if source_ra is None or source_dec is None:
            return []
        coords = []
        for (p_ra, p_dec) in primer_positions:
            xc, yc = sky_to_pixel_cutout2d(
                p_ra, p_dec,
                source_ra, source_dec,
                pixscale, img.shape,
            )
            coords.append((xc, yc))
        return coords

    cutouts = [
        (left_nb,   nb_img,   CUTOUT_CMAP, "Pseudo-NB",        ap_nb,
         True, COLOURS["aperture"],  resolve_nb_crosses(nb_img)),
        (left_jels, jels_img, CUTOUT_CMAP, "JELS " + band_str, ap_jels,
         True, "white",              resolve_wcs_crosses(jels_img, jels_pixscale)),
        (left_rgb,  rgb_img,  None,        "PRIMER RGB",        ap_rgb,
         True, "white",              resolve_wcs_crosses(rgb_img, rgb_pixscale)),
    ]

    for left, img, cmap, label, ap_pix, show_ap, ap_colour, cross_pos in cutouts:
        ax_in = ax_spec.inset_axes([left, bottom, inset_w, inset_h])
        render_cutout(ax_in, img, cmap, label, ap_pix,
                      show_aperture=show_ap, aperture_colour=ap_colour,
                      crossmatch_positions=cross_pos)

    return ax_spec


# ──────────────────────────────────────────────────────────────────────────────
# Per-source data loader
# ──────────────────────────────────────────────────────────────────────────────

def load_one_source(row, zminmax_df,
                    extractions_dir, ap_key,
                    jels_template, rgb_fits,
                    cutout_arcsec, jels_pixscale,
                    rgb_pixscale, fallback_muse_pixscale,
                    crossmatch_lookup=None):
    """
    Load spec + meta + all three cutout images for one catalogue row.

    Also stores primer_positions: the list of (ra, dec) tuples for any
    PRIMER source that was matched to this JELS ID.

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

    # ── PRIMER crossmatch positions for this JELS ID ──────────────────────────
    primer_positions = []
    if crossmatch_lookup and src_id is not None:
        try:
            key = int(float(src_id))
        except (ValueError, TypeError):
            key = None
        if key is not None and key in crossmatch_lookup:
            primer_positions = crossmatch_lookup[key]
            print(f"  [crossmatch] ID={src_id}: "
                  f"{len(primer_positions)} PRIMER position(s) to overlay")

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
        src_id=src_id,
        primer_positions=primer_positions,   # ← NEW: list of (ra, dec)
        source_ra=ra,                         # ← NEW: JELS source RA
        source_dec=dec,                       # ← NEW: JELS source Dec
    )


# ──────────────────────────────────────────────────────────────────────────────
# Figure savers
# ──────────────────────────────────────────────────────────────────────────────

def save_single(out_path, src, plot_window_ang, aperture_arcsec,
                jels_pixscale, rgb_pixscale):
    """Save one source as its own PDF, with source ID as the figure title."""
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
        primer_positions=src.get("primer_positions"),
        source_ra=src.get("source_ra"),
        source_dec=src.get("source_dec"),
    )

    src_id = src.get("src_id")
    if src_id is not None:
        fig.suptitle(
            f"ID {src_id}",
            x=0.09, ha="left",
            fontsize=7, fontfamily="monospace",
            color=COLOURS["flux"],
        )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  saved: {out_path}")


def save_multirow(out_path, sources, plot_window_ang, aperture_arcsec,
                  jels_pixscale, rgb_pixscale):
    """
    Save a stacked multi-row figure (one row per source).
    Each row is labelled with its source ID as an axes title.
    """
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
        ax = build_panel(
            fig, gs[i, 0],
            src["spec"], src["meta"],
            src["lya_center"], src["lya_wmin"], src["lya_wmax"],
            src["nb_img"], src["nb_pixscale"],
            src["jels_img"], src["rgb_img"],
            src["band"], plot_window_ang, aperture_arcsec,
            jels_pixscale, rgb_pixscale,
            jels_name=src.get("jels_name"),
            row_i=i, n_rows=n,
            primer_positions=src.get("primer_positions"),
            source_ra=src.get("source_ra"),
            source_dec=src.get("source_dec"),
        )

        src_id = src.get("src_id")
        if src_id is not None:
            ax.set_title(
                f"ID {src_id}",
                loc="left",
                fontsize=6,
                fontfamily="monospace",
                color=COLOURS["flux"],
                pad=2,
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
    p.add_argument("--crossmatch-csv", default=None,
                   help="Path to primer_jels_crossmatch.csv. "
                        "If provided, a cross is drawn on each cutout at the "
                        "PRIMER source position for matched JELS IDs.")
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

    df         = load_source_csv(args.csv)
    zminmax    = load_zminmax_csv(args.zminmax_csv)
    crossmatch = load_crossmatch_csv(args.crossmatch_csv)

    buckets = {0: [], 1: [], 2: []}

    EXCLUDE_IDS = {3228, 6938, 7650}

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
            crossmatch_lookup=crossmatch,
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
python spectra_paper_cross.py \
    --csv         /cephfs/apatrick/musecosmos/scripts/jels_muse_sources.csv \
    --zminmax-csv /home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_with_zmin_zmax_lya.csv \
    --extractions /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/extractions \
    --out-dir     /cephfs/apatrick/musecosmos/scripts/sample_select/plots/pubready \
    --jels-template /home/apatrick/P1/JELSDP/JELS_v1_{band}_30mas_i2d.fits \
    --rgb-fits    /home/apatrick/P1/col.fits \
    --crossmatch-csv /cephfs/apatrick/musecosmos/scripts/primer_jels_crossmatch.csv \
    --plot-window 300 \
    --cutout-arcsec 3.0
"""