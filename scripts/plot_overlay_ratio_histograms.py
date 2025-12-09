#!/usr/bin/env python
import os
import argparse
import numpy as np

# Use a non-interactive backend (safe for SLURM/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.io import fits


def compute_slice_wavelength(slice_idx, start=4749.9, step=1.25):
    return start + step * slice_idx


def valid_ratio_linear(vap, var):
    """
    Return linear ratios R = vap/var for valid pixels only (finite, >0).
    """
    var = np.array(var, dtype=float)
    vap = np.array(vap, dtype=float)
    good = np.isfinite(var) & np.isfinite(vap) & (var > 0.0) & (vap > 0.0)
    if not np.any(good):
        return np.array([])
    return vap[good] / var[good]


def load_fits_pair(output_dir, slice_idx):
    var_path = os.path.join(output_dir, f"var_mosaic_slice_{slice_idx}.fits")
    vap_path = os.path.join(output_dir, f"var_mosaic_slice_{slice_idx}_vap.fits")
    var = fits.getdata(var_path)
    vap = fits.getdata(vap_path)
    return vap, var


def robust_stats_linear(R):
    """
    Compute robust stats on linear ratio R.
    Returns dict with N, median, MAD (1.4826 * MAD), p16, p50, p84.
    """
    if R.size == 0:
        return dict(N=0, median=np.nan, mad=np.nan, p16=np.nan, p50=np.nan, p84=np.nan)
    median = float(np.nanmedian(R))
    mad = float(1.4826 * np.nanmedian(np.abs(R - median)))
    p16, p50, p84 = [float(q) for q in np.nanpercentile(R, [16, 50, 84])]
    return dict(N=int(R.size), median=median, mad=mad, p16=p16, p50=p50, p84=p84)


def parse_slice_list(args):
    """
    Build the list of slice indices from one of:
      --slices (explicit list),
      --starts + --step + --count (possibly multiple starts),
      or fallback to single --start + --step + --count.
    """
    if args.slices is not None and len(args.slices) > 0:
        slices = list(map(int, args.slices))
    elif args.starts is not None and len(args.starts) > 0:
        slices = []
        for s0 in args.starts:
            slices.extend([s0 + i * args.step for i in range(args.count)])
    else:
        # fallback to single start
        slices = [args.start + i * args.step for i in range(args.count)]
    # unique and stable order
    slices = sorted(set(slices))
    return slices

def plot_median_vs_wavelength(stats_rows, output_dir, base_outfile,
                              use_logy=False, marker='o', color='tab:blue'):
    """
    Create a plot of wavelength (Å) vs median(Back_Var/IVAR), with MAD error bars,
    a least-squares fit line, and Pearson correlation.

    Saves:
      <stem>_median_vs_wavelength.png
      <stem>_median_vs_wavelength.csv

    Parameters
    ----------
    stats_rows : list of dict
        Each dict must contain keys: 'wavelength_A', 'median', 'mad', 'slice', 'N'.
        In this script these rows are already computed in linear ratio space.
    output_dir : str
        Directory to write outputs.
    base_outfile : str
        The base outfile name used for the overlay figure; stem is reused.
    use_logy : bool
        If True, set y-axis to log scale.
    marker : str
        Matplotlib marker for scatter points.
    color : str
        Color for scatter and fit line.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Filter valid rows
    rows = [r for r in stats_rows
            if np.isfinite(r.get('wavelength_A', np.nan))
            and np.isfinite(r.get('median', np.nan))
            and np.isfinite(r.get('mad', np.nan))
            and r.get('N', 0) > 0]
    if len(rows) == 0:
        print("No valid rows for median vs wavelength plot.")
        return None

    # Sort by wavelength
    rows = sorted(rows, key=lambda r: r['wavelength_A'])
    lam = np.array([r['wavelength_A'] for r in rows], dtype=float)
    med = np.array([r['median'] for r in rows], dtype=float)
    mad = np.array([r['mad'] for r in rows], dtype=float)
    N   = np.array([r['N'] for r in rows], dtype=int)
    sli = np.array([r['slice'] for r in rows], dtype=int)

    # Pearson correlation (linear)
    if med.size >= 2:
        r = float(np.corrcoef(lam, med)[0, 1])
    else:
        r = np.nan

    # Linear least-squares fit: med ~ a * lam + b
    if med.size >= 2:
        a, b = np.polyfit(lam, med, deg=1)
        fit = a * lam + b
    else:
        a, b, fit = np.nan, np.nan, np.full_like(med, np.nan)

    # Make figure
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(lam, med, yerr=mad, fmt=marker, color=color, ecolor='lightgray',
                elinewidth=1, capsize=2, alpha=0.9, label='Median ± MAD')
    if np.all(np.isfinite(fit)):
        ax.plot(lam, fit, color=color, lw=2,
                label=f"Fit: median = {a:.3g}·λ + {b:.3g}   (r={r:.3f})")

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Median(Back_Var / IVAR)")
    ax.set_title("Median variance ratio vs wavelength")
    if use_logy:
        ax.set_yscale('log')
    ax.grid(alpha=0.2)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()

    stem = os.path.splitext(base_outfile)[0]
    png_path = os.path.join(output_dir, f"{stem}_median_vs_wavelength.png")
    fig.savefig(png_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved wavelength vs median plot to {png_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"{stem}_median_vs_wavelength.csv")
    with open(csv_path, "w") as f:
        f.write("slice,wavelength_A,N,median,mad,fit_slope,fit_intercept,pearson_r\n")
        for i in range(len(rows)):
            f.write(f"{sli[i]},{lam[i]:.3f},{N[i]},{med[i]:.6g},{mad[i]:.6g},"
                    f"{a:.6g},{b:.6g},{r:.6g}\n")
    print(f"Saved wavelength vs median CSV to {csv_path}")

    return {"png": png_path, "csv": csv_path, "slope": a, "intercept": b, "pearson_r": r}


def main():
    ap = argparse.ArgumentParser(description="Overlay Back_Var/IVAR histograms with per-slice stats")
    ap.add_argument("--output_dir", required=True, help="Directory with var_mosaic_slice_xxx.fits and _vap.fits")

    # New: multiple starts or explicit slices
    ap.add_argument("--slices", type=int, nargs="+", default=None,
                    help="Explicit list of slice indices to include (overrides --starts/--start)")
    ap.add_argument("--starts", type=int, nargs="+", default=None,
                    help="One or more starting slice indices; each produces a sequence of length --count with step --step")

    # Backwards-compatible single start/step/count
    ap.add_argument("--start", type=int, default=100, help="First slice index (used if --slices/--starts not provided)")
    ap.add_argument("--step", type=int, default=350, help="Slice step")
    ap.add_argument("--count", type=int, default=10, help="Number of slices per start")

    ap.add_argument("--bins", type=int, default=140, help="Number of histogram bins")
    ap.add_argument("--logx", action="store_true", help="Plot histograms in log10(Back_Var/IVAR)")
    ap.add_argument("--clip_lo", type=float, default=1.0, help="Lower percentile for global clipping (display x-space)")
    ap.add_argument("--clip_hi", type=float, default=99.0, help="Upper percentile for global clipping (display x-space)")
    ap.add_argument("--wavestart", type=float, default=4749.9, help="Wavelength mapping: start (Å)")
    ap.add_argument("--wavestep", type=float, default=1.25, help="Wavelength mapping: step (Å per slice)")
    ap.add_argument("--outfile", default="overlay_ratio_histograms.png", help="Output PNG filename")
    args = ap.parse_args()

    # Build slice list from args
    slices = parse_slice_list(args)
    if len(slices) == 0:
        raise RuntimeError("No slices selected. Provide --slices or --starts/--start with --count.")

    # Load per-slice ratios (linear) and compute display-x values (log or linear)
    ratios_disp = []     # list of 1D arrays of X to plot (log10(R) or R)
    wavelengths = []
    slice_ids = []
    stats_rows = []      # dicts with stats (computed on linear ratios)

    for sl in slices:
        vap, var = load_fits_pair(args.output_dir, sl)
        R = valid_ratio_linear(vap, var)
        if R.size == 0:
            print(f"WARNING: slice {sl} has no valid pixels; skipping.")
            continue

        # Save display-space version
        X = np.log10(R) if args.logx else R
        ratios_disp.append(X)
        lam_A = compute_slice_wavelength(sl, args.wavestart, args.wavestep)
        wavelengths.append(lam_A)
        slice_ids.append(sl)

        # Stats on linear space
        s = robust_stats_linear(R)
        stats_rows.append({
            "slice": sl,
            "wavelength_A": lam_A,
            **s
        })

    if len(ratios_disp) == 0:
        raise RuntimeError("No valid slices found for overlay.")

    # Global robust range for consistent binning across slices (in display x-space)
    all_vals = np.concatenate(ratios_disp)
    lo = np.nanpercentile(all_vals, args.clip_lo)
    hi = np.nanpercentile(all_vals, args.clip_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise RuntimeError("Failed to determine a valid histogram range.")

    bins = np.linspace(lo, hi, args.bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Colormap by wavelength
    lam = np.array(wavelengths)
    norm = mpl.colors.Normalize(vmin=np.min(lam), vmax=np.max(lam))
    cmap = mpl.cm.viridis

    # Figure with two rows: top = overlay plot, bottom = stats table
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.5, 1.2], hspace=0.25)
    ax = fig.add_subplot(gs[0])

    # Plot overlay histograms (density=True with in-window clipping)
    for X, w in zip(ratios_disp, lam):
        clipped = X[(X >= lo) & (X <= hi)]
        if clipped.size == 0:
            continue
        counts, _ = np.histogram(clipped, bins=bins, density=True)
        color = cmap(norm(w))
        ax.step(centers, counts, where='mid', lw=2, color=color)

    # Axis labels, title
    xlabel = "log10(Back_Var / IVAR)" if args.logx else "Back_Var / IVAR"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(f"Overlay of Back_Var/IVAR histograms for {len(ratios_disp)} slices")

    # Add colorbar for wavelength
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Wavelength (Å)")

    # Range of medians (computed on linear ratios), draw on the plot in display x-space
    medians_lin = np.array([row["median"] for row in stats_rows if np.isfinite(row["median"])])
    if medians_lin.size > 0:
        med_min_lin = float(np.nanmin(medians_lin))
        med_max_lin = float(np.nanmax(medians_lin))
        med_min_disp = np.log10(med_min_lin) if args.logx else med_min_lin
        med_max_disp = np.log10(med_max_lin) if args.logx else med_max_lin
        x0, x1 = sorted([med_min_disp, med_max_disp])
        ax.axvspan(x0, x1, color='k', alpha=0.08, zorder=0)
        ax.axvline(x0, color='k', ls='--', lw=1.5, alpha=0.7)
        ax.axvline(x1, color='k', ls='--', lw=1.5, alpha=0.7)
        ax.text(0.02, 0.98,
                f"Median range (linear): [{med_min_lin:.3g}, {med_max_lin:.3g}]",
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Build stats table (linear-space stats)
    order = np.argsort(lam)
    cols = ["Slice", "Wavelength (Å)", "N", "Median", "MAD", "p16", "p50", "p84"]
    table_data = []
    for idx in order:
        row = stats_rows[idx]
        table_data.append([
            int(row["slice"]),
            f"{row['wavelength_A']:.1f}",
            int(row["N"]),
            f"{row['median']:.4g}",
            f"{row['mad']:.4g}",
            f"{row['p16']:.4g}",
            f"{row['p50']:.4g}",
            f"{row['p84']:.4g}",
        ])

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')
    tbl = ax_tbl.table(cellText=table_data,
                       colLabels=cols,
                       loc='center',
                       cellLoc='center',
                       colLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)

    # Save figure
    fig.tight_layout()
    outpath = os.path.join(args.output_dir, args.outfile)
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved overlay histogram + stats table to {outpath}")

    # Save CSV of the same stats
    csv_path = os.path.join(args.output_dir, os.path.splitext(args.outfile)[0] + "_stats.csv")
    header = "slice,wavelength_A,N,median,mad,p16,p50,p84\n"
    with open(csv_path, "w") as f:
        f.write(header)
        for idx in order:
            row = stats_rows[idx]
            f.write(f"{int(row['slice'])},{row['wavelength_A']:.3f},{int(row['N'])},"
                    f"{row['median']:.6g},{row['mad']:.6g},{row['p16']:.6g},"
                    f"{row['p50']:.6g},{row['p84']:.6g}\n")
    print(f"Saved per-slice stats CSV to {csv_path}")

    # Also make wavelength vs median plot
    plot_median_vs_wavelength(
        stats_rows=stats_rows,
        output_dir=args.output_dir,
        base_outfile=args.outfile,
        use_logy=False  # set True if you want a log y-axis
    )



if __name__ == "__main__":
    main()
