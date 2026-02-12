#!/usr/bin/env python3
"""
Plot z_used histograms with JWST/NIRCam filter transmission curves.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from astropy.table import Table
except Exception:
    Table = None


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


def plot_hist(ax, values, bins, color, label, weights, linestyle, zorder):
    """
    Plot an unfilled histogram on an axis.

    Inputs
        ax        : Matplotlib Axes.
        values    : Numpy array of values.
        bins      : Number of bins.
        color     : Histogram color.
        label     : Legend label.
        weights   : Optional weights array.
        linestyle : Matplotlib line style.
        zorder    : Matplotlib zorder.

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
        linewidth=1.8,
        color=color,
        linestyle=linestyle,
        zorder=zorder,
        label=label,
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


def weights_for_total(values, total_count, scale):
    """
    Build global area-normalized weights.

    Inputs
        values      : Numpy array of values.
        total_count : Total number of sources across filters.
        scale       : Scale factor for histogram height.

    Output
        Numpy array of weights or None.
    """
    if total_count <= 0:
        return None
    return scale * np.ones_like(values, dtype=float) / float(total_count)


def max_hist_height(values_list, bins, weights_list):
    """
    Compute max histogram height across multiple samples.

    Inputs
        values_list  : List of numpy arrays.
        bins         : Bin edges or bin count.
        weights_list : List of weight arrays or None.

    Output
        Float max height.
    """
    max_val = 0.0
    for values, weights in zip(values_list, weights_list):
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bins, weights=weights)
        if counts.size:
            max_val = max(max_val, float(np.nanmax(counts)))
    return max_val


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
        default="/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv",
        help="Path to CSV file.",
    )
    parser.add_argument("--z-col", default="z_used", help="Redshift column.")
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
    parser.add_argument("--bins", type=int, default=25, help="Number of bins.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot raw counts instead of normalizing to 1.",
    )
    parser.add_argument(
        "--hist-scale",
        type=float,
        default=6.0,
        help="Scale factor for normalized histograms (e.g., 1.5).",
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
        help="Wavelength unit for transmission files ",
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
        default="/cephfs/apatrick/musecosmos/scripts/sample_select/filter_hist.png",
        help="Optional output path to save figure (png/pdf).",
    )
    args = parser.parse_args()

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

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)

    mask_f466 = is_f466 & z_valid
    mask_f470 = is_f470 & z_valid
    mask_det = detections & z_valid
    mask_nondet = (~detections) & z_valid

    n_f466_det = int(np.sum(mask_f466 & detections))
    n_f470_det = int(np.sum(mask_f470 & detections))
    n_f466_nondet = int(np.sum(mask_f466 & ~detections))
    n_f470_nondet = int(np.sum(mask_f470 & ~detections))
    n_f466_total = n_f466_det + n_f466_nondet
    n_f470_total = n_f470_det + n_f470_nondet
    n_total = n_f466_total + n_f470_total

    z_all = z[z_valid]
    bin_edges = np.histogram_bin_edges(z_all, bins=args.bins)
    weights_466_det = None
    weights_466_nondet = None
    weights_470_det = None
    weights_470_nondet = None
    if normalize_hist:
        scale = args.hist_scale
        weights_466_det = weights_for_total(z[mask_f466 & detections], n_total, scale)
        weights_466_nondet = weights_for_total(z[mask_f466 & ~detections], n_total, scale)
        weights_470_det = weights_for_total(z[mask_f470 & detections], n_total, scale)
        weights_470_nondet = weights_for_total(z[mask_f470 & ~detections], n_total, scale)

    plot_hist(
        ax,
        z[mask_f466 & detections],
        bin_edges,
        color="#8B0000",
        label=f"F466N detection (N={n_f466_det})",
        weights=weights_466_det,
        linestyle="-",
        zorder=3,
    )
    plot_hist(
        ax,
        z[mask_f470 & detections],
        bin_edges,
        color="#003366",
        label=f"F470N detection (N={n_f470_det})",
        weights=weights_470_det,
        linestyle="-",
        zorder=3,
    )
    plot_hist(
        ax,
        z[mask_f466 & ~detections],
        bin_edges,
        color="#f4a3a3",
        label=f"F466N non-detection (N={n_f466_nondet})",
        weights=weights_466_nondet,
        linestyle="--",
        zorder=4,
    )
    plot_hist(
        ax,
        z[mask_f470 & ~detections],
        bin_edges,
        color="#9ecae1",
        label=f"F470N non-detection (N={n_f470_nondet})",
        weights=weights_470_nondet,
        linestyle="--",
        zorder=4,
    )

    if normalize_hist:
        ax.fill_between(z466, t466, color="#f4a3a3", alpha=0.35, label="F466N transmission")
        ax.plot(z466, t466, color="#8B0000", linewidth=1.2)
        ax.fill_between(z470, t470, color="#9ecae1", alpha=0.35, label="F470N transmission")
        ax.plot(z470, t470, color="#003366", linewidth=1.2)
        ax.set_ylabel("Normalized counts (area=1 overall) / transmission")
        hist_max = max_hist_height(
            [
                z[mask_f466 & detections],
                z[mask_f470 & detections],
                z[mask_f466 & ~detections],
                z[mask_f470 & ~detections],
            ],
            bins=bin_edges,
            weights_list=[
                weights_466_det,
                weights_470_det,
                weights_466_nondet,
                weights_470_nondet,
            ],
        )
        ylim_top = max(1.05, 1.05 * hist_max)
        ax.set_ylim(0.0, ylim_top)
    else:
        ax2 = ax.twinx()
        ax2.fill_between(z466, t466, color="#f4a3a3", alpha=0.35, label="F466N transmission")
        ax2.plot(z466, t466, color="#8B0000", linewidth=1.2)
        ax2.fill_between(z470, t470, color="#9ecae1", alpha=0.35, label="F470N transmission")
        ax2.plot(z470, t470, color="#003366", linewidth=1.2)
        ax.set_ylabel("Counts")
        ax2.set_ylabel("Transmission (normalized)")

    ax.set_xlabel(f"Halpha redshift (from wavelength)")
    ax.set_title("Halpha redshift distribution with NIRCam filter transmission")

    handles, labels = ax.get_legend_handles_labels()
    if not normalize_hist:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    f466_handles = []
    f466_labels = []
    f470_handles = []
    f470_labels = []
    for handle, label in zip(handles, labels):
        if "F466N" in label:
            f466_handles.append(handle)
            f466_labels.append(label)
        if "F470N" in label:
            f470_handles.append(handle)
            f470_labels.append(label)

    legend_f466 = ax.legend(
        f466_handles,
        f466_labels,
        frameon=False,
        fontsize=9,
        loc="upper left",
    )
    ax.add_artist(legend_f466)
    ax.legend(
        f470_handles,
        f470_labels,
        frameon=False,
        fontsize=9,
        loc="upper right",
    )

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
