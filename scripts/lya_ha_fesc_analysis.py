#!/usr/bin/env python3
"""
Lyα–Hα luminosity and f_esc analysis.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.table import Table, join


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def as_float(tab, name):
    """
    Load a column as float with NaNs handled.

    Inputs
        tab  : Astropy Table.
        name : Column name.

    Output
        Numpy array (float) with NaNs.
    """
    if name not in tab.colnames:
        return np.full(len(tab), np.nan)

    data = tab[name]
    try:
        arr = np.array(data, dtype=float)
    except Exception:
        arr = np.array(
            [float(x) if str(x).strip() not in ("", "nan", "None") else np.nan for x in data],
            dtype=float,
        )

    if hasattr(data, "mask"):
        arr[data.mask] = np.nan

    return arr


def flux_to_luminosity(flux, redshift):
    """
    Convert flux to luminosity given redshift.

    Inputs
        flux     : Integrated line flux (1e-20 erg/s/cm^2).
        redshift : Redshift.

    Output
        Luminosity in erg/s.
    """
    flux_cgs = flux * 1e-20
    d_l = cosmo.luminosity_distance(redshift).to(u.cm).value
    return flux_cgs * 4 * np.pi * d_l**2


def flux_err_to_luminosity_err(flux_err, redshift):
    """
    Convert flux error to luminosity error given redshift.

    Inputs
        flux_err : Flux error (1e-20 erg/s/cm^2).
        redshift : Redshift.

    Output
        Luminosity error in erg/s.
    """
    flux_err = np.asarray(flux_err, dtype=float)
    redshift = np.asarray(redshift, dtype=float)

    out = np.full_like(flux_err, np.nan, dtype=float)
    m = np.isfinite(flux_err) & np.isfinite(redshift)
    if not np.any(m):
        return out if out.shape != () else np.nan

    flux_err_cgs = flux_err[m] * 1e-20
    d_l = cosmo.luminosity_distance(redshift[m]).to(u.cm).value
    out[m] = flux_err_cgs * 4 * np.pi * d_l**2
    return out if out.shape != () else float(out)


def good_xy(x, y, xerr=None, yerr=None):
    """
    Mask finite, positive values.
    """
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if xerr is not None:
        m &= np.isfinite(xerr)
    if yerr is not None:
        m &= np.isfinite(yerr)
    return m


def log_limits(values, pad=0.2):
    """
    Compute log-space limits with padding.

    Inputs
        values : Array of values.
        pad    : Log10 padding.

    Output
        (vmin, vmax) tuple or (None, None) if invalid.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return None, None
    vmin = np.min(v)
    vmax = np.max(v)
    factor = 10 ** pad
    return vmin / factor, vmax * factor


def draw_ul_arrows(ax, x, y, frac=0.35, color="royalblue", alpha=0.6, lw=1.2):
    """
    Draw downward arrows from each (x, y) point.

    Inputs
        ax    : Matplotlib axis.
        x, y  : Arrays of coordinates.
        frac  : Fractional arrow length in log space (0.35 ~ factor 2.2).
        color : Arrow color.
    """
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi) or xi <= 0 or yi <= 0:
            continue
        y2 = yi / (10 ** frac)
        ax.annotate(
            "",
            xy=(xi, y2),
            xytext=(xi, yi),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha),
            zorder=2,
        )


def plot_lya_ha(
    x,
    y,
    xerr,
    yerr,
    xlabel,
    ylabel,
    title,
    outpath,
    detections_mask,
    upper_mask,
    y_upper,
    bg_x=None,
    bg_y=None,
    add_caseb=True,
    xlim=None,
    ylim=None,
    ul_alpha=0.55,
):
    """
    Scatter plot with upper limits.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    if bg_x is not None and bg_y is not None:
        ax.scatter(bg_x, bg_y, c="0.8", s=25, alpha=0.5, edgecolor="none", zorder=0)

    if np.any(detections_mask):
        ax.errorbar(
            x[detections_mask],
            y[detections_mask],
            xerr=xerr[detections_mask] if xerr is not None else None,
            yerr=yerr[detections_mask] if yerr is not None else None,
            fmt="none",
            ecolor="0.6",
            elinewidth=1.2,
            capsize=2,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            x[detections_mask],
            y[detections_mask],
            c="black",
            s=30,
            alpha=0.9,
            edgecolor="none",
            zorder=2,
            label="detections",
        )

    if np.any(upper_mask):
        ax.scatter(
            x[upper_mask],
            y_upper[upper_mask],
            c="royalblue",
            s=30,
            alpha=ul_alpha,
            edgecolor="none",
            zorder=2,
            label="upper limits",
        )
        draw_ul_arrows(ax, x[upper_mask], y_upper[upper_mask], frac=0.35, color="royalblue", alpha=ul_alpha)

    if np.any(x[detections_mask | upper_mask] > 0):
        ax.set_xscale("log")
    if np.any(np.concatenate([y[detections_mask], y_upper[upper_mask]]) > 0):
        ax.set_yscale("log")

    if add_caseb:
        xx = np.logspace(39, 44, 200)
        ax.plot(xx, xx, color="black", lw=1, alpha=0.4, linestyle="--", label="1:1")
        ax.plot(xx, xx * 8.7, color="black", alpha=0.9, linestyle="-", lw=1, label="L(Lyα)=8.7×L(Hα)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {outpath}")


def plot_fesc_vs_property(
    tab,
    y_col,
    y_err_col,
    y_ul_col,
    x_col,
    xlabel,
    out_path,
    logy=True,
    logx=False,
    xlim=None,
    ylim=None,
    sphinx_tab=None,
    sphinx_x_col=None,
    sphinx_y_col=None,
):
    """
    Plot f_esc versus a property, with optional SPHINX overlay and upper limits.
    """
    x = np.asarray(tab[x_col], dtype=float)
    y = np.asarray(tab[y_col], dtype=float)
    yerr = np.asarray(tab[y_err_col], dtype=float)
    y_ul = np.asarray(tab[y_ul_col], dtype=float)

    mstar = np.asarray(tab["M_star_50"], dtype=float) if "M_star_50" in tab.colnames else np.full(len(tab), np.nan)

    det_mask = np.isfinite(x) & np.isfinite(y)
    ul_mask = np.isfinite(x) & np.isfinite(y_ul)

    if logx:
        det_mask &= x > 0
        ul_mask &= x > 0
    if logy:
        det_mask &= y > 0
        ul_mask &= y_ul > 0

    fig, ax = plt.subplots(figsize=(6.5, 5))

    if sphinx_tab is not None and sphinx_x_col is not None and sphinx_y_col is not None:
        if sphinx_x_col in sphinx_tab.colnames and sphinx_y_col in sphinx_tab.colnames:
            sx = np.asarray(sphinx_tab[sphinx_x_col], dtype=float)
            sy = np.asarray(sphinx_tab[sphinx_y_col], dtype=float)
            sm = np.isfinite(sx) & np.isfinite(sy)
            ax.scatter(sx[sm], sy[sm], c="0.8", s=25, alpha=0.4, zorder=0)

    has_mstar = np.any(np.isfinite(mstar[det_mask]))
    if np.any(det_mask):
        if has_mstar:
            sc = ax.scatter(
                x[det_mask],
                y[det_mask],
                c=mstar[det_mask],
                cmap="viridis",
                s=40,
                edgecolor="none",
                zorder=2,
            )
            fig.colorbar(sc, ax=ax, label=r"$M_\star\,[M_\odot]$")
        else:
            ax.scatter(
                x[det_mask],
                y[det_mask],
                c="black",
                s=40,
                edgecolor="none",
                zorder=2,
            )

        ax.errorbar(
            x[det_mask],
            y[det_mask],
            xerr=None,
            yerr=yerr[det_mask],
            fmt="none",
            ecolor="gray",
            alpha=0.6,
            capsize=2,
            zorder=1,
        )

    if np.any(ul_mask):
        ax.scatter(
            x[ul_mask],
            y_ul[ul_mask],
            c="royalblue",
            s=28,
            alpha=0.55,
            edgecolor="none",
            zorder=2,
        )
        draw_ul_arrows(ax, x[ul_mask], y_ul[ul_mask], frac=0.35, color="royalblue", alpha=0.55)

    if logx and np.any(x[det_mask | ul_mask] > 0):
        ax.set_xscale("log")
    if logy and np.any(np.concatenate([y[det_mask], y_ul[ul_mask]]) > 0):
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$f^{\mathrm{Ly}\alpha}_\mathrm{esc}$")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Lyα–Hα luminosity + f_esc analysis.")
    parser.add_argument("--lya-csv", required=True, help="Path to Lyα CSV (e.g., lya_flux_ap0p6.csv).")
    parser.add_argument("--ha-fits", required=True, help="Path to Hα FITS table.")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots.")
    parser.add_argument("--flux-mode", choices=["flux_fit", "flux_int"], default="flux_fit",
                        help="Which Lyα flux to use for detections.")
    parser.add_argument("--caseb", type=float, default=8.7, help="Case B ratio Lyα/Hα.")
    parser.add_argument("--sphinx-table", default=None, help="Optional SPHINX FITS table.")
    parser.add_argument("--aperture-tag", default="0p6", help="Aperture tag for titles.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lya_tab = Table.read(args.lya_csv, format="ascii.csv")
    ha_tab = Table.read(args.ha_fits)

    lya_tab["ID"] = lya_tab["ID"].astype(int)
    ha_tab["ID"] = ha_tab["ID"].astype(int)

    tab = join(lya_tab, ha_tab, keys="ID", join_type="inner")
    print(f"[INFO] Matched rows = {len(tab)}")

    # Lyα flux choice
    lya_flux_det = as_float(tab, args.flux_mode)
    lya_flux_det_err = as_float(tab, f"{args.flux_mode}_err")
    lya_flux_ul = as_float(tab, "flux_upper_limit")
    det_flag = as_float(tab, "lya_detect_flag")
    is_det = np.isfinite(det_flag) & (det_flag > 0)
    is_nondet = np.isfinite(det_flag) & (det_flag == 0)

    # Redshift for luminosity
    if "z_used" in tab.colnames:
        z = as_float(tab, "z_used")
    else:
        z = as_float(tab, "z1_median")

    # Luminosities
    lya_L = np.full(len(tab), np.nan)
    lya_L_err = np.full(len(tab), np.nan)
    lya_L_ul = np.full(len(tab), np.nan)

    m_det = is_det & np.isfinite(lya_flux_det) & np.isfinite(z)
    lya_L[m_det] = flux_to_luminosity(lya_flux_det[m_det], z[m_det])
    lya_L_err[m_det] = flux_err_to_luminosity_err(lya_flux_det_err[m_det], z[m_det])

    m_ul = is_nondet & np.isfinite(lya_flux_ul) & np.isfinite(z)
    lya_L_ul[m_ul] = flux_to_luminosity(lya_flux_ul[m_ul], z[m_ul])

    tab["lya_flux_used"] = lya_flux_det
    tab["lya_flux_used_err"] = lya_flux_det_err
    tab["lya_L"] = lya_L
    tab["lya_L_err"] = lya_L_err
    tab["lya_L_ul"] = lya_L_ul

    # Hα luminosities
    HaL_unc = as_float(tab, "L_Ha_uncorr")
    HaL_unc_err = as_float(tab, "L_Ha_uncorr_err")
    HaL_ap = as_float(tab, "L_Ha_apcorr")
    HaL_ap_err = as_float(tab, "L_Ha_apcorr_err")
    HaL_dust = as_float(tab, "L_Ha_ap_dustcorr_line")
    HaL_dust_err = as_float(tab, "L_Ha_ap_dustcorr_line_err")

    caseb = args.caseb

    tab["fesc_lya_uncorr"] = lya_L / (caseb * HaL_unc)
    tab["fesc_lya_uncorr_err"] = tab["fesc_lya_uncorr"] * np.sqrt(
        (lya_L_err / lya_L) ** 2 + (HaL_unc_err / HaL_unc) ** 2
    )
    tab["fesc_lya_uncorr_ul"] = lya_L_ul / (caseb * HaL_unc)

    tab["fesc_lya_apcorr"] = lya_L / (caseb * HaL_ap)
    tab["fesc_lya_apcorr_err"] = tab["fesc_lya_apcorr"] * np.sqrt(
        (lya_L_err / lya_L) ** 2 + (HaL_ap_err / HaL_ap) ** 2
    )
    tab["fesc_lya_apcorr_ul"] = lya_L_ul / (caseb * HaL_ap)

    tab["fesc_lya_dustcorr"] = lya_L / (caseb * HaL_dust)
    tab["fesc_lya_dustcorr_err"] = tab["fesc_lya_dustcorr"] * np.sqrt(
        (lya_L_err / lya_L) ** 2 + (HaL_dust_err / HaL_dust) ** 2
    )
    tab["fesc_lya_dustcorr_ul"] = lya_L_ul / (caseb * HaL_dust)

    out_csv = os.path.join(args.out_dir, f"lya_flux_ap{args.aperture_tag}_with_fesc.csv")
    tab.write(out_csv, format="ascii.csv", overwrite=True)
    print(f"[OK] Wrote updated CSV to {out_csv}")

    # SPHINX overlay
    sphinx_tab = None
    sphinx_Lya = None
    sphinx_Ha = None
    if args.sphinx_table:
        sphinx_tab = Table.read(args.sphinx_table)
        if "L_lya" in sphinx_tab.colnames and "L_ha" in sphinx_tab.colnames:
            sphinx_Lya = np.asarray(sphinx_tab["L_lya"], dtype=float)
            sphinx_Ha = np.asarray(sphinx_tab["L_ha"], dtype=float)

    # Plot L(Lyα) vs L(Hα)
    for label, HaL, HaL_err, suffix in [
        ("uncorrected", HaL_unc, HaL_unc_err, "uncorr"),
        ("apcorr", HaL_ap, HaL_ap_err, "apcorr"),
        ("dust corrected", HaL_dust, HaL_dust_err, "dustcorr"),
    ]:
        x = HaL
        y = lya_L
        y_ul = lya_L_ul
        m_det = is_det & good_xy(x, y, xerr=HaL_err, yerr=lya_L_err)
        m_ul = is_nondet & np.isfinite(x) & np.isfinite(y_ul) & (x > 0) & (y_ul > 0)
        title = f"Lyα vs Hα ({label}) | {args.flux_mode} | ap={args.aperture_tag}"
        out = os.path.join(args.out_dir, f"Lya_vs_Ha_{suffix}_{args.aperture_tag}.png")
        xlim = log_limits(np.concatenate([x[m_det], x[m_ul]]), pad=0.4)
        plot_lya_ha(
            x, y, HaL_err, lya_L_err,
            xlabel=f"L(Hα) [{label}, erg s⁻¹]",
            ylabel="L(Lyα) [erg s⁻¹]",
            title=title,
            outpath=out,
            detections_mask=m_det,
            upper_mask=m_ul,
            y_upper=y_ul,
            bg_x=sphinx_Ha,
            bg_y=sphinx_Lya,
            add_caseb=True,
            xlim=xlim,
            ylim=log_limits(np.concatenate([y[m_det], y_ul[m_ul]]), pad=0.8)
        )

    # Plot flux–flux
    ha_flux = as_float(tab, "Ha_flux")
    ha_flux_err = as_float(tab, "Ha_flux_err")
    lya_flux_det_cgs = lya_flux_det * 1e-20
    lya_flux_ul_cgs = lya_flux_ul * 1e-20

    x = ha_flux
    y = lya_flux_det_cgs
    y_ul = lya_flux_ul_cgs
    m_det = is_det & good_xy(x, y, xerr=ha_flux_err, yerr=lya_flux_det_err)
    m_ul = is_nondet & np.isfinite(x) & np.isfinite(y_ul) & (x > 0) & (y_ul > 0)
    title = f"Lyα Flux vs Hα Flux | {args.flux_mode} | ap={args.aperture_tag}"
    out = os.path.join(args.out_dir, f"Flux_Lya_vs_Ha_{args.aperture_tag}.png")
    plot_lya_ha(
        x, y, ha_flux_err, lya_flux_det_err * 1e-20,
        xlabel="F(Hα) [erg s⁻¹ cm⁻²]",
        ylabel="F(Lyα) [erg s⁻¹ cm⁻²]",
        title=title,
        outpath=out,
        detections_mask=m_det,
        upper_mask=m_ul,
        y_upper=y_ul,
        add_caseb=False,
    )

    # f_esc vs properties (with upper limits)
    if "E_BV_50" not in tab.colnames and "Av_50" in tab.colnames:
        Rv = 4.05
        tab["E_BV_50"] = as_float(tab, "Av_50") / Rv

    if "E_BV_50" in tab.colnames:
        plot_fesc_vs_property(
            tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="E_BV_50",
            xlabel="E(B-V)",
            out_path=os.path.join(args.out_dir, f"fesc_vs_EBV_{args.aperture_tag}.png"),
            logy=True,
            sphinx_tab=sphinx_tab,
            sphinx_x_col="E_BV_median" if sphinx_tab is not None and "E_BV_median" in sphinx_tab.colnames else None,
            sphinx_y_col="fesc_lya" if sphinx_tab is not None and "fesc_lya" in sphinx_tab.colnames else None,
        )

    for x_col, label, fname in [
        ("M_star_50", r"log $(M_\star/M_\odot)$", "fesc_vs_Mstar"),
        ("sSFR_100Myr_50", r"sSFR [yr$^{-1}$]", "fesc_vs_sSFR"),
        ("SFR_ratio_50", "SFR ratio", "fesc_vs_SFRratio"),
        ("size_50", "Half-light radius [kpc]", "fesc_vs_size"),
    ]:
        if x_col in tab.colnames:
            plot_fesc_vs_property(
                tab,
                y_col="fesc_lya_dustcorr",
                y_err_col="fesc_lya_dustcorr_err",
                y_ul_col="fesc_lya_dustcorr_ul",
                x_col=x_col,
                xlabel=label,
                out_path=os.path.join(args.out_dir, f"{fname}_{args.aperture_tag}.png"),
                logy=True,
                logx=True if "sSFR" in x_col or "SFR_ratio" in x_col else False,
            )


if __name__ == "__main__":
    main()
