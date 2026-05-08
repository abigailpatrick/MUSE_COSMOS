#!/usr/bin/env python3

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from astropy.table import Table
from lifelines import KaplanMeierFitter
from scipy import stats

# ── optional cmasher ─────────────────────────────────────────
try:
    import cmasher as cmr
    HAS_CMASHER = True
except ImportError:
    HAS_CMASHER = False
    warnings.warn("cmasher not found; using fallback colors")

# ── matplotlib configuration ─────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.size": 4,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "figure.dpi": 150,
})

# ── colour palette ───────────────────────────────────────────
def make_palette(name: str, n: int = 8) -> dict:
    if HAS_CMASHER:
        hexes = cmr.take_cmap_colors(
            f"cmr.{name}", n, cmap_range=(0.15, 0.90), return_fmt="hex"
        )
        palette = {f"c{i}": h for i, h in enumerate(hexes)}
    else:
        palette = {
            "c0": "#440154",
            "c1": "#31688e",
            "c2": "#35b779",
            "c3": "#fde724",
            "c4": "#ff6e3a",
            "c5": "#d62728",
            "c6": "#8b4789",
            "c7": "#cccccc",
        }

    palette["near_black"]   = "#0a0e27"
    palette["neutral_grey"] = "#6b7280"
    return palette

P = make_palette("torch", n=8)

# ── models ───────────────────────────────────────────────────
def konno2016(z):
    return 5e-4 * (1 + z) ** 2.8

def hayes2011(z):
    return 1e-3 * (1 + z) ** 2.6

# ── KM median ────────────────────────────────────────────────
def km_median(values, detected):
    kmf = KaplanMeierFitter()
    kmf.fit(values, event_observed=detected.astype(int))
    return kmf.median_survival_time_

# ── main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--show-individual", action="store_true")
    parser.add_argument("--hayes-csv", default="/cephfs/apatrick/musecosmos/scripts/Hayes.csv",
                        help="Path to Hayes et al. compilation CSV")

    parser.add_argument("--z-col", default="z_jels")
    parser.add_argument("--fesc-col", default="fesc_lya_dustcorr")
    parser.add_argument("--fesc-ul-col", default="fesc_lya_dustcorr_ul")
    parser.add_argument("--det-col", default="lya_detect_flag")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load main dataset ─────────────────────────────────────
    tab = Table.read(args.csv)

    z = np.array(tab[args.z_col], float)
    fesc = np.array(tab[args.fesc_col], float)
    fesc_ul = np.array(tab[args.fesc_ul_col], float)

    if args.det_col in tab.colnames:
        detected = np.array(tab[args.det_col], float) > 0
    else:
        detected = np.isnan(fesc_ul)

    mask = np.isfinite(z)
    z, fesc, fesc_ul, detected = z[mask], fesc[mask], fesc_ul[mask], detected[mask]

    values = np.where(detected, fesc, fesc_ul)

    # ── KM median with bootstrap CI ──────────────────────────
    kmf = KaplanMeierFitter()
    kmf.fit(values, event_observed=detected.astype(int))
    median_fesc = kmf.median_survival_time_
    mean_z = np.mean(z)

    n_bootstrap = 1000
    bootstrap_medians = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(values), len(values), replace=True)
        kmf_boot = KaplanMeierFitter()
        kmf_boot.fit(values[idx], event_observed=detected[idx].astype(int))
        if np.isfinite(kmf_boot.median_survival_time_):
            bootstrap_medians.append(kmf_boot.median_survival_time_)

    bootstrap_medians = np.array(bootstrap_medians)
    median_lower = np.percentile(bootstrap_medians, 16)
    median_upper = np.percentile(bootstrap_medians, 84)

    print(f"KM median f_esc = {median_fesc:.3e} at z = {mean_z:.2f}")
    print(f"Bootstrap 68% CI: [{median_lower:.3e}, {median_upper:.3e}]")

    # ── load JADES data ───────────────────────────────────────
    deep_df   = pd.read_csv("jades_deep.csv")
    medium_df = pd.read_csv("jades_medium.csv")

    # ── combine JADES and fit correlation ────────────────────
    jades_all   = pd.concat([deep_df, medium_df], ignore_index=True)
    jades_clean = jades_all[["zspec", "fesc"]].dropna()
    jades_z     = jades_clean["zspec"].values
    jades_fesc  = jades_clean["fesc"].values

    # Fit in log(fesc) vs z — consistent with log y-axis
    log_fesc_jades = np.log10(jades_fesc)
    slope, intercept, _, _, _ = stats.linregress(jades_z, log_fesc_jades)

    # Spearman p-value (rank-based, matches what the paper quotes)
    rho_jades, p_jades = stats.spearmanr(jades_z, log_fesc_jades)
    print(f"JADES all — Spearman: rho={rho_jades:.3f}, p={p_jades:.3f}")

    # ── load Hayes et al. compilation ────────────────────────
    hayes_df = pd.read_csv(args.hayes_csv)
    hayes_df["lya_escape_fraction"] = hayes_df["lya_escape_fraction"] / 100.0

    # ── plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    z_solid = np.linspace(0, 6, 300)
    z_dashed = np.linspace(6, 9, 200)

    ax.plot(z_solid, konno2016(z_solid),
                "-", color=P["c6"], lw=1.5, alpha=0.7, label="Konno+2016")
    ax.plot(z_dashed, konno2016(z_dashed),
                ":", color=P["c6"], lw=1.5, alpha=0.7)

    ax.plot(z_solid, hayes2011(z_solid),
                "-", color=P["c5"], lw=1.5, alpha=0.8, label="Hayes+2011")
    ax.plot(z_dashed, hayes2011(z_dashed),
                ":", color=P["c5"], lw=1.5, alpha=0.8)

    if args.show_individual:
        ax.scatter(z[detected], fesc[detected],
                   s=35, marker="D",
                   facecolors=P["near_black"], edgecolors=P["near_black"],
                   linewidths=0.4, alpha=0.6,
                   label="Detections")

        ax.scatter(z[~detected], fesc_ul[~detected],
                   s=35, marker="v",
                   facecolors=P["near_black"], edgecolors=P["near_black"],
                   linewidths=0.5, alpha=0.6,
                   label="Upper limits")

    # ── KM median ────────────────────────────────────────────
    ax.scatter(mean_z, median_fesc,
               s=300, marker="*",
               facecolors=P["c3"],
               edgecolors=P["near_black"],
               linewidths=1.2,
               zorder=10,
               label="This work median")

    ax.errorbar(mean_z, median_fesc,
                yerr=[[median_fesc - median_lower], [median_upper - median_fesc]],
                fmt="none",
                ecolor=P["near_black"],
                elinewidth=1.0,
                capsize=4,
                capthick=1.0,
                zorder=9,
                alpha=0.7)

    # ── JADES scatter ─────────────────────────────────────────
    ax.scatter(deep_df["zspec"], deep_df["fesc"],
               marker="o",
               facecolors="none",
               edgecolors=P["c1"],
               linewidths=1.2,
               label="JADES Deep (Saxena+2024)")

    ax.scatter(medium_df["zspec"], medium_df["fesc"],
               marker="s",
               facecolors="none",
               edgecolors=P["c1"],
               linewidths=1.2,
               label="JADES Medium (Saxena+2024)")

    # ── JADES correlation line ────────────────────────────────
    z_fit = np.linspace(jades_z.min(), jades_z.max(), 300)
    ax.plot(z_fit, 10 ** (slope * z_fit + intercept),
            "--", color=P["c1"], lw=1.8, alpha=0.7,
            label=rf"JADES all (Saxena+2024)")

    # ── Hayes et al. compilation ──────────────────────────────
    ax.scatter(hayes_df["redshift"],
               hayes_df["lya_escape_fraction"],
               marker="^",
               facecolors="none",
               edgecolors=P["c5"],
               linewidths=1.2,
               s=50,
               label="Hayes+2011 compilation")

    # ── formatting ───────────────────────────────────────────
    ax.set_yscale("log")
    ax.set_xlim(4.3, 8.05)
    ax.set_ylim(0.008, 1.0)

    ax.yaxis.set_major_locator(LogLocator(base=10))

    def decimal_formatter(y, pos):
        return f"{y:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")

    for spine in ax.spines.values():
        spine.set_color(P["near_black"])
        spine.set_linewidth(0.8)

    ax.legend(loc="lower left", fontsize=6.5, markerscale=0.7, frameon=False, ncol=1)

    fig.tight_layout()

    # ── save ─────────────────────────────────────────────────
    out_pdf = os.path.join(args.out_dir, "fesc_km_median.pdf")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()


"""
python fesc_lya_evolution.py --csv /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged.csv --out-dir ./figs --show-individual --hayes-csv /cephfs/apatrick/musecosmos/scripts/Hayes.csv

"""