#!/usr/bin/env python3
"""
Global fesc,Lyα evolution + distribution plot
USING PRECOMPUTED fesc VALUES
Improved version with flexible mean plotting and fixed violin scatter alignment
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import os


# -------------------------------------------------
# Helper
# -------------------------------------------------

def as_float(tab, name):
    if name not in tab.colnames:
        raise ValueError(f"Column {name} not found in table.")
    return np.array(tab[name], dtype=float)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="/cephfs/apatrick/musecosmos/scripts/sample_select/lya_flux_ap0p6_with_fesc.csv"
    )

    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--fesc-type", default="dustcorr",
                        choices=["uncorr", "apcorr", "dustcorr"])

    parser.add_argument("--plot-mode", default="all",
                        choices=["all", "detections",
                                 "non-detections",
                                 "overall-detection-fraction"])

    parser.add_argument("--show-individual", action="store_true")

    parser.add_argument("--show-mean",
                        choices=["detections", "non-detections", "all"],
                        help="Overlay mean value")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------
    # Load table
    # -------------------------------------------------
    tab = Table.read(args.csv, format="ascii.csv")

    print(f"[INFO] Loaded rows = {len(tab)}")

    z = as_float(tab, "z_used")
    det_flag = as_float(tab, "lya_detect_flag")

    fesc_col = f"fesc_lya_{args.fesc_type}"
    fesc_ul_col = f"{fesc_col}_ul"

    fesc = as_float(tab, fesc_col)
    fesc_ul = as_float(tab, fesc_ul_col)

    detections = det_flag > 0

    good = np.isfinite(z)

    z = z[good]
    fesc = fesc[good]
    fesc_ul = fesc_ul[good]
    detections = detections[good]

    # -------------------------------------------------
    # DEBUG: Print full arrays being used
    # -------------------------------------------------
    print("\n================ DEBUG OUTPUT ================")

    print("\nFull fesc (detections only):")
    print(fesc[detections])

    print("\nFull fesc upper limits (non-detections only):")
    print(fesc_ul[~detections])

    print("\nNumber of detections:", np.sum(detections))
    print("Number of non-detections:", np.sum(~detections))

    print("==============================================\n")

    # -------------------------------------------------
    # Prepare data
    # -------------------------------------------------

    fig = plt.figure(figsize=(7, 10))

    # =================================================
    # PANEL 1 — EVOLUTION
    # =================================================
    ax1 = plt.subplot(3, 1, 1)

    zgrid = np.linspace(0, 8, 500)
    f_konno = 5e-4 * (1 + zgrid) ** 2.8
    f_hayes = 1e-3 * (1 + zgrid) ** 2.6

    ax1.plot(zgrid, f_konno, "--", label="Konno+2016")
    ax1.plot(zgrid, f_hayes, "-", label="Hayes+2011")

    # Smaller individual points
    if args.show_individual:

        if args.plot_mode in ["all", "detections"]:
            ax1.scatter(z[detections],
                        fesc[detections],
                        s=25, marker="D",
                        facecolors="green",
                        edgecolors="green",
                        alpha=0.7,
                        label="Detections")

        if args.plot_mode in ["all", "non-detections"]:
            ax1.scatter(z[~detections],
                        fesc_ul[~detections],
                        s=25, marker="D",
                        facecolors="none",
                        edgecolors="red",
                        alpha=0.7,
                        label="Upper limits")


    # ---------------------------
    # Mean overlay (plotted LAST)
    # ---------------------------
    if args.show_mean is not None:

        if args.show_mean == "detections":
            mean_z = np.mean(z[detections])
            mean_f = np.mean(fesc[detections])
            label = "Mean (detections)"

        elif args.show_mean == "non-detections":
            mean_z = np.mean(z[~detections])
            mean_f = np.mean(fesc_ul[~detections])
            label = "Mean (non-detections)"

        else:  # all
            combined = np.concatenate([fesc[detections],
                                       fesc_ul[~detections]])
            mean_z = np.mean(z)
            mean_f = np.mean(combined)
            label = "Mean (all)"

        ax1.scatter(mean_z, mean_f,
                    s=180,
                    marker="D",
                    facecolors="yellow",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=10,
                    label=label)

    ax1.set_yscale("log")
    ax1.set_xlim(0, 8)
    ax1.set_ylim(1e-4, 1.5)
    ax1.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")
    ax1.set_title("Redshift Evolution")
    ax1.legend(frameon=False)

    # Count points plotted in evolution panel
    if args.plot_mode == "detections":
        n_evo = np.sum(detections)

    elif args.plot_mode == "non-detections":
        n_evo = np.sum(~detections)

    elif args.plot_mode == "all":
        n_evo = len(z)

    else:  # overall-detection-fraction
        n_evo = 1

    print("Number of sources plotted in Evolution panel:", n_evo)

    # =================================================
    # PANEL 2 — BOX PLOT
    # =================================================
    ax2 = plt.subplot(3, 1, 2)

    if args.plot_mode == "detections":
        data = fesc[detections]

    elif args.plot_mode == "non-detections":
        data = fesc_ul[~detections]

    else:
        data = np.concatenate([
            fesc[detections],
            fesc_ul[~detections]
        ])



    data = data[data > 0]

    ax2.boxplot(data, showfliers=True)

    if args.show_individual:
        x = 1 + 0.04 * np.random.randn(len(data))
        ax2.scatter(x, data, alpha=0.5, s=20)

    ax2.set_yscale("log")
    ax2.set_xticks([])
    ax2.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")
    ax2.set_title("Box Plot Distribution")

    # =================================================
    # PANEL 3 — VIOLIN PLOT
    # =================================================
    ax3 = plt.subplot(3, 1, 3)

    ax3.violinplot(data, showmedians=True)

    if args.show_individual:
        x = 1 + 0.05 * np.random.randn(len(data))
        ax3.scatter(x, data, alpha=0.5, s=20)

    ax3.set_yscale("log")
    ax3.set_xticks([])
    ax3.set_ylabel(r"$f_{\rm esc}^{\rm Ly\alpha}$")
    ax3.set_title("Violin Plot Distribution")

    fig.tight_layout()

    outpath = os.path.join(args.out_dir,
                           "fesc_evolution_distribution.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"\n[OK] Saved figure to {outpath}")


if __name__ == "__main__":
    main()