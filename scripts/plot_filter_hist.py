#!/usr/bin/env python3
"""
plot_filter_hist.py

Plot z(Halpha) histograms with JWST/NIRCam filter transmission curves.
Publication-ready for MNRAS single column (84 mm = 3.307 in).
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import cmasher as cmr

try:
    from astropy.table import Table
except Exception:
    Table = None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION TYPOGRAPHY
# Match MNRAS body text (9 pt Times). Figures scale to column width so
# use 5.5 pt for labels and 4.0 pt for tick labels.
# ══════════════════════════════════════════════════════════════════════════════

mpl.rcParams.update({
    "font.family":            "serif",
    "font.serif":             ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":       "dejavuserif",
    "font.size":              5.5,
    "axes.labelsize":         6.5,
    "axes.titlesize":         5.5,
    "xtick.labelsize":        5.0,
    "ytick.labelsize":        5.0,
    "legend.fontsize":        5.0,
    "legend.title_fontsize":  5.5,
    "lines.linewidth":        0.7,
    "axes.linewidth":         0.4,
    "xtick.major.width":      0.4,
    "ytick.major.width":      0.4,
    "xtick.minor.width":      0.3,
    "ytick.minor.width":      0.3,
    "xtick.major.size":       2.0,
    "ytick.major.size":       2.0,
    "xtick.minor.size":       1.0,
    "ytick.minor.size":       1.0,
    "xtick.direction":        "in",
    "ytick.direction":        "in",
    "xtick.top":              True,
    "ytick.right":            True,
    "xtick.minor.visible":    False,
    "ytick.minor.visible":    False,
    "legend.framealpha":      0.85,
    "legend.edgecolor":       "0.6",
    "legend.handlelength":    1.0,
    "legend.handletextpad":   0.3,
    "legend.borderpad":       0.3,
    "legend.labelspacing":    0.2,
})


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE DIMENSIONS — single column MNRAS
# 84 mm = 3.307 in; height via 3:4 aspect ratio
# ══════════════════════════════════════════════════════════════════════════════

FIG_WIDTH  = 3.307          # inches  (84 mm — MNRAS single column)
FIG_HEIGHT = FIG_WIDTH * 0.75


# ══════════════════════════════════════════════════════════════════════════════
# THEME — change here to retheme the whole script
# ══════════════════════════════════════════════════════════════════════════════

THEME = "torch"


def make_palette(name, n=8):
    """
    Build a colour palette from a cmasher colourmap.

    Inputs
        name : cmasher colourmap name (without 'cmr.' prefix).
        n    : Number of colours to sample.

    Output
        Dict mapping 'c0'–'c{n-1}' and named extras to hex strings.
    """
    hexes = cmr.take_cmap_colors(
        f"cmr.{name}", n, cmap_range=(0.10, 0.90), return_fmt="hex"
    )
    palette = {f"c{i}": h for i, h in enumerate(hexes)}
    palette["near_black"]   = mcolors.to_hex((0.10, 0.10, 0.12, 1.0))
    palette["neutral_grey"] = "#aab4c8"
    palette["white"]        = "#ffffff"
    palette["mid_grey"]     = "#888888"
    return palette


P = make_palette(THEME)


def parse_int_list(value):
    """
    Parse comma-separated integers.

    Inputs
        value : String with comma-separated integers.

    Output
        List of integers.
    """
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def find_column(colnames, keywords):
    """
    Find the first column name containing any keyword.

    Inputs
        colnames : Iterable of column names.
        keywords : Iterable of lowercase keywords.

    Output
        Column name or None.
    """
    for name in colnames:
        lname = str(name).lower()
        for key in keywords:
            if key in lname:
                return name
    return None


def load_transmission(path, wave_col, throughput_col, file_format):
    """
    Load transmission data from ascii or VOTable.

    Inputs
        path           : File path.
        wave_col       : Optional wavelength column name.
        throughput_col : Optional transmission column name.
        file_format    : Optional astropy format string.

    Output
        Tuple (wavelength, throughput) as numpy arrays.
    """
    try:
        df = pd.read_csv(path)
        colnames = list(df.columns)
        use_wave = wave_col or find_column(colnames, ["wave", "lambda", "wavelength"])
        use_thr = throughput_col or find_column(colnames, ["trans", "throughput", "response"])
        if use_wave is None or use_thr is None:
            if len(colnames) < 2:
                raise SystemExit("Transmission file needs at least two columns.")
            use_wave = colnames[0]
            use_thr = colnames[1]
        wave = np.array(df[use_wave], dtype=float)
        thr = np.array(df[use_thr], dtype=float)
        return wave, thr
    except Exception:
        pass

    if Table is None:
        data = np.genfromtxt(path, comments="#")
        if data.ndim == 1:
            data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise SystemExit("Transmission file needs at least two columns.")
        wave = np.array(data[:, 0], dtype=float)
        thr = np.array(data[:, 1], dtype=float)
        return wave, thr

    fmt = file_format
    if fmt is None:
        lower = str(path).lower()
        if lower.endswith((".xml", ".vot", ".votable")):
            fmt = "votable"
        elif lower.endswith((".csv",)):
            fmt = "ascii.csv"
        elif lower.endswith((".dat", ".txt", ".tbl")):
            fmt = "ascii.basic"

    try:
        tab = Table.read(path, format=fmt) if fmt else Table.read(path)
    except Exception:
        try:
            tab = Table.read(path, format="ascii.basic")
        except Exception:
            try:
                tab = Table.read(path, format="votable")
            except Exception:
                data = np.genfromtxt(path, comments="#")
                if data.ndim == 1:
                    data = np.atleast_2d(data)
                if data.shape[1] < 2:
                    raise SystemExit("Transmission file needs at least two columns.")
                wave = np.array(data[:, 0], dtype=float)
                thr = np.array(data[:, 1], dtype=float)
                return wave, thr

    colnames = list(tab.colnames)
    use_wave = wave_col
    use_thr = throughput_col

    if use_wave is None:
        use_wave = find_column(colnames, ["wave", "lambda", "wavelength"])
    if use_thr is None:
        use_thr = find_column(colnames, ["trans", "throughput", "response"])

    if use_wave is None or use_thr is None:
        if len(colnames) < 2:
            raise SystemExit("Transmission file needs at least two columns.")
        use_wave = colnames[0]
        use_thr = colnames[1]

    wave = np.array(tab[use_wave], dtype=float)
    thr = np.array(tab[use_thr], dtype=float)
    return wave, thr


def to_angstrom(wave, unit):
    """
    Convert wavelength to Angstrom.

    Inputs
        wave : Numpy array of wavelength.
        unit : String unit (angstrom, nm, um, micron).

    Output
        Numpy array in Angstrom.
    """
    u = unit.lower()
    if u in ("angstrom", "a", "ang"):
        return wave
    if u in ("nm", "nanometer", "nanometers"):
        return wave * 10.0
    if u in ("um", "micron", "microns"):
        return wave * 1.0e4
    raise SystemExit(f"Unknown wavelength unit: {unit}")


def wavelength_to_z_halpha(wave, unit, ha_wave):
    """
    Convert wavelength to Halpha redshift.

    Inputs
        wave    : Numpy array of wavelength.
        unit    : Wavelength unit (angstrom, nm, um, micron).
        ha_wave : Rest Halpha wavelength in Angstrom.

    Output
        Numpy array of redshift.
    """
    wave_a = to_angstrom(wave, unit)
    return (wave_a / ha_wave) - 1.0


def split_by_filter(df, filter_col):
    """
    Build boolean masks for F466N and F470N selections.

    Inputs
        df         : Pandas DataFrame.
        filter_col : Filter column name (F466N/F470N).

    Output
        Tuple (is_f466, is_f470).
    """
    if not filter_col or filter_col not in df.columns:
        raise SystemExit("Provide --filter-col with F466N/F470N labels.")

    filt = df[filter_col].astype(str).str.lower()
    is_f466 = filt.str.contains("466")
    is_f470 = filt.str.contains("470")
    return is_f466, is_f470


def normalize_to_peak(values, bins, weights=None):
    """
    Compute per-histogram peak-normalised weights so the histogram
    reaches a maximum bar height of 1.

    Inputs
        values  : Numpy array of values to histogram.
        bins    : Bin edges array.
        weights : Optional existing weights array.

    Output
        Numpy array of rescaled weights (or None if values is empty).
    """
    if values.size == 0:
        return None
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    peak = np.nanmax(counts)
    if peak <= 0 or not np.isfinite(peak):
        return None
    base_w = weights if weights is not None else np.ones_like(values, dtype=float)
    return base_w / peak


def plot_hist(ax, values, bins, color, label, weights, linestyle, linewidth, zorder, alpha=1.0):
    """
    Plot a step (outline-only) histogram on an axis.

    Inputs
        ax        : Matplotlib Axes.
        values    : Numpy array of values.
        bins      : Bin edges array.
        color     : Histogram colour.
        label     : Legend label.
        weights   : Optional weights array.
        linestyle : Matplotlib line style string.
        linewidth : Line width float.
        zorder    : Matplotlib zorder.
        alpha     : Line alpha.

    Output
        None.
    """
    if values.size == 0:
        return
    ax.hist(
        values,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        zorder=zorder,
        label=label,
        alpha=alpha,
    )


def plot_hist_filled(ax, values, bins, color, label, weights, zorder, fill_alpha=0.30, edge_alpha=0.70):
    """
    Plot a filled step histogram (stepfilled) for non-detections.

    The fill gives spatial extent at a glance; the semi-transparent edge
    keeps the outline readable without competing with detection outlines.

    Inputs
        ax          : Matplotlib Axes.
        values      : Numpy array of values.
        bins        : Bin edges array.
        color       : Histogram colour.
        label       : Legend label.
        weights     : Optional weights array.
        zorder      : Matplotlib zorder.
        fill_alpha  : Alpha for the filled region.
        edge_alpha  : Alpha for the step outline.

    Output
        None.
    """
    if values.size == 0:
        return
    # Filled region
    ax.hist(
        values,
        bins=bins,
        weights=weights,
        histtype="stepfilled",
        linewidth=0,
        color=color,
        zorder=zorder,
        label=label,
        alpha=fill_alpha,
    )
    # Outline on top of fill — dashed, semi-transparent
    ax.hist(
        values,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=1.0,
        color=color,
        linestyle="-",
        zorder=zorder + 1,
        alpha=edge_alpha,
    )


def normalize_curve(values):
    """
    Normalize a curve to max 1.

    Inputs
        values : Numpy array.

    Output
        Numpy array normalized to max 1.
    """
    vmax = np.nanmax(values)
    if vmax <= 0 or not np.isfinite(vmax):
        return values
    return values / vmax


def main():
    """
    Entry point.

    Inputs
        None.

    Output
        None.
    """
    parser = argparse.ArgumentParser(
        description="Plot z_used histograms with JWST/NIRCam filter curves."
    )
    parser.add_argument(
        "--csv",
        default="/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged.csv",
        help="Path to CSV file.",
    )
    parser.add_argument("--z-col", default="z_jels", help="Redshift column.")
    parser.add_argument(
        "--filter-col",
        default="band",
        help="Filter column with F466N/F470N labels.",
    )
    parser.add_argument(
        "--detect-col",
        default="lya_detect_flag",
        help="Column used to define detections.",
    )
    parser.add_argument(
        "--detect-values",
        default="1,2",
        help="Comma-separated values that count as detections.",
    )
    parser.add_argument("--bins", type=int, default=15, help="Number of bins.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot raw counts instead of normalizing each histogram to its own peak.",
    )
    parser.add_argument(
        "--f466n-file",
        default="/home/apatrick/P1/JELSDP/JWST_NIRCam_F466N.csv",
        help="Transmission file for JWST/NIRCam F466N.",
    )
    parser.add_argument(
        "--f470n-file",
        default="/home/apatrick/P1/JELSDP/JWST_NIRCam_F470N.csv",
        help="Transmission file for JWST/NIRCam F470N.",
    )
    parser.add_argument(
        "--wave-unit",
        default="angstrom",
        help="Wavelength unit for transmission files.",
    )
    parser.add_argument(
        "--ha-wave",
        type=float,
        default=6562.8,
        help="Rest Halpha wavelength in Angstrom.",
    )
    parser.add_argument(
        "--wave-col",
        default="",
        help="Optional wavelength column name in transmission files.",
    )
    parser.add_argument(
        "--throughput-col",
        default="",
        help="Optional throughput column name in transmission files.",
    )
    parser.add_argument(
        "--trans-format",
        default="",
        help="Astropy format for transmission files (e.g., ascii.basic or votable).",
    )
    parser.add_argument(
        "--out",
        default="/cephfs/apatrick/musecosmos/scripts/sample_select/filter_hist.pdf",
        help="Optional output path to save figure (png/pdf).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load catalogue
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    z_col = args.z_col

    if z_col not in df.columns:
        raise SystemExit(f"Missing column: {z_col}")
    if args.detect_col not in df.columns:
        raise SystemExit(f"Missing column: {args.detect_col}")

    detect_values = set(parse_int_list(args.detect_values))
    detections = df[args.detect_col].isin(detect_values)

    filter_col = args.filter_col.strip() or None
    is_f466, is_f470 = split_by_filter(df, filter_col)

    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy()
    z_valid = np.isfinite(z)
    normalize_hist = not args.no_normalize

    wave_col = args.wave_col.strip() or None
    throughput_col = args.throughput_col.strip() or None
    trans_format = args.trans_format.strip() or None

    # ------------------------------------------------------------------
    # Load filter transmission curves
    # ------------------------------------------------------------------
    w466, t466 = load_transmission(
        args.f466n_file, wave_col, throughput_col, trans_format
    )
    w470, t470 = load_transmission(
        args.f470n_file, wave_col, throughput_col, trans_format
    )

    z466 = wavelength_to_z_halpha(w466, args.wave_unit, args.ha_wave)
    z470 = wavelength_to_z_halpha(w470, args.wave_unit, args.ha_wave)

    t466 = normalize_curve(t466)
    t470 = normalize_curve(t470)

    # ------------------------------------------------------------------
    # Build masks and bin edges
    # ------------------------------------------------------------------
    mask_f466_det  = is_f466 & z_valid & detections
    mask_f466_non  = is_f466 & z_valid & ~detections
    mask_f470_det  = is_f470 & z_valid & detections
    mask_f470_non  = is_f470 & z_valid & ~detections

    n_f466_det    = int(np.sum(mask_f466_det))
    n_f470_det    = int(np.sum(mask_f470_det))
    n_f466_nondet = int(np.sum(mask_f466_non))
    n_f470_nondet = int(np.sum(mask_f470_non))

    z_all = z[z_valid]
    bin_edges = np.histogram_bin_edges(z_all, bins=args.bins)

    # ------------------------------------------------------------------
    # Colour palette — drawn from cmr.torch via shared make_palette()
    #   torch c0–c7 runs dark-purple → rust → orange → amber → yellow
    #
    #   F466N → orange family  (c2 det, c1 non-det)
    #   F470N → amber/gold     (c5 det, c4 non-det)
    # ------------------------------------------------------------------
    C466_DET = P["c2"]   # burnt orange  — F466N detections
    C466_NON = P["c1"]   # deep rust     — F466N non-detections
    C470_DET = P["c5"]   # golden yellow — F470N detections
    C470_NON = P["c4"]   # amber         — F470N non-detections

    # ------------------------------------------------------------------
    # Compute per-histogram peak-normalised weights (optional)
    # ------------------------------------------------------------------
    if normalize_hist:
        w466_det = normalize_to_peak(z[mask_f466_det], bin_edges)
        w466_non = normalize_to_peak(z[mask_f466_non], bin_edges)
        w470_det = normalize_to_peak(z[mask_f470_det], bin_edges)
        w470_non = normalize_to_peak(z[mask_f470_non], bin_edges)
    else:
        w466_det = w466_non = w470_det = w470_non = None

    # ------------------------------------------------------------------
    # Figure — MNRAS single column
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Transmission curves scaled to 1.1 — drawn first, sit behind histograms
    t466_scaled = t466 * 1.1
    t470_scaled = t470 * 1.1
    ax.fill_between(z466, t466_scaled, color=C466_DET, alpha=0.08, zorder=1)
    ax.plot(z466, t466_scaled, color=C466_DET, linewidth=0.7, alpha=0.55,
            zorder=1, label="F466N transmission")
    ax.fill_between(z470, t470_scaled, color=C470_DET, alpha=0.08, zorder=1)
    ax.plot(z470, t470_scaled, color=C470_DET, linewidth=0.7, alpha=0.55,
            zorder=1, label="F470N transmission")

    # Non-detections — filled (soft, background)
    plot_hist_filled(ax, z[mask_f466_non], bin_edges,
                     color=C466_NON,
                     label=rf"F466N non-detection",
                     weights=w466_non, zorder=2)

    plot_hist_filled(ax, z[mask_f470_non], bin_edges,
                     color=C470_NON,
                     label=rf"F470N non-detection",
                     weights=w470_non, zorder=2)

    # Detections — thick solid outlines on top
    plot_hist(ax, z[mask_f466_det], bin_edges,
              color=C466_DET,
              label=rf"F466N detection",
              weights=w466_det, linestyle="-", linewidth=1.2, zorder=4)

    plot_hist(ax, z[mask_f470_det], bin_edges,
              color=C470_DET,
              label=rf"F470N detection ",
              weights=w470_det, linestyle="-", linewidth=1.2, zorder=4)

    # ------------------------------------------------------------------
    # Axis labels and ticks
    # ------------------------------------------------------------------
    ax.set_xlabel(r"$z(\mathrm{H\alpha})$", fontsize=6.5)

    if normalize_hist:
        ax.set_ylabel("Normalised counts", fontsize=6.5)
        ax.set_ylim(0.0, 1.25)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    else:
        ax2 = ax.twinx()
        ax2.fill_between(z466, t466, color=C466_DET, alpha=0.10, zorder=1)
        ax2.plot(z466, t466, color=C466_DET, linewidth=0.7, alpha=0.55, zorder=1)
        ax2.fill_between(z470, t470, color=C470_DET, alpha=0.10, zorder=1)
        ax2.plot(z470, t470, color=C470_DET, linewidth=0.7, alpha=0.55, zorder=1)
        ax.set_ylabel("$N$", fontsize=5.5)
        ax2.set_ylabel("Filter transmission", fontsize=5.5)
        ax2.set_ylim(0.0, 1.15)
        ax2.tick_params(labelsize=4.0)

    # ------------------------------------------------------------------
    # Legend — F466N entries upper-left, F470N entries upper-right
    # ------------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()

    f466_h, f466_l, f470_h, f470_l = [], [], [], []
    for h, l in zip(handles, labels):
        if "F466N" in l:
            f466_h.append(h); f466_l.append(l)
        elif "F470N" in l:
            f470_h.append(h); f470_l.append(l)

    leg466 = ax.legend(f466_h, f466_l, frameon=False, fontsize=5.0, loc="upper left")
    ax.add_artist(leg466)
    ax.legend(f470_h, f470_l, frameon=False, fontsize=5.0, loc="upper right")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout(pad=0.3)

    if args.out:
        fig.savefig(args.out, bbox_inches="tight", facecolor="white", dpi=600)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()