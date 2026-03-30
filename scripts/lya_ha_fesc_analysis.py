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
from astropy.table import Table, join, vstack
from scipy.stats import spearmanr, kendalltau



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
    flux = np.asarray(flux, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if flux.size == 0 or redshift.size == 0:
        return np.array([], dtype=float)

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
    if flux_err.size == 0 or redshift.size == 0:
        return np.array([], dtype=float)

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
    ids=None,
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
        if ids is not None:
            for xi, yi, sid in zip(x[detections_mask], y[detections_mask], ids[detections_mask]):
                if np.isfinite(xi) and np.isfinite(yi) and xi > 0 and yi > 0:
                    ax.annotate(
                        str(int(sid)),
                        xy=(xi, yi),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="black",
                        alpha=0.8,
                        zorder=3,
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
        if ids is not None:
            for xi, yi, sid in zip(x[upper_mask], y_upper[upper_mask], ids[upper_mask]):
                if np.isfinite(xi) and np.isfinite(yi) and xi > 0 and yi > 0:
                    ax.annotate(
                        str(int(sid)),
                        xy=(xi, yi),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="royalblue",
                        alpha=0.8,
                        zorder=3,
                    )

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

def compute_correlations(x, y_det, y_ul, det_mask, ul_mask):
    """
    Compute Spearman (using measured values) and
    Kendall (using 3σ upper limits for non-detections).

    Returns:
        rho, p_rho, tau, p_tau
    """

    # -------------------------------
    # Spearman: use measured values
    # -------------------------------
    x_s = np.concatenate([x[det_mask], x[ul_mask]])
    y_s = np.concatenate([y_det[det_mask], y_det[ul_mask]])

    m_s = np.isfinite(x_s) & np.isfinite(y_s)
    if np.sum(m_s) > 2:
        rho, p_rho = spearmanr(x_s[m_s], y_s[m_s])
    else:
        rho, p_rho = np.nan, np.nan

    # --------------------------------
    # Kendall: use upper limits for UL
    # --------------------------------
    x_k = np.concatenate([x[det_mask], x[ul_mask]])
    y_k = np.concatenate([y_det[det_mask], y_ul[ul_mask]])

    m_k = np.isfinite(x_k) & np.isfinite(y_k)
    if np.sum(m_k) > 2:
        tau, p_tau = kendalltau(x_k[m_k], y_k[m_k])
    else:
        tau, p_tau = np.nan, np.nan

    print(f"[INFO] Spearman ρ = {rho:.2f} (p={p_rho:.2g}), Kendall τ = {tau:.2f} (p={p_tau:.2g})")

    return rho, p_rho, tau, p_tau


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
    ids=None,
    sphinx_tab=None,
    sphinx_x_col=None,
    sphinx_y_col=None,
    color_by=None,     
    color_label=None,       

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

    # -----------------------------------
    # Compute correlations
    # -----------------------------------
    rho, p_rho, tau, p_tau = compute_correlations(
        x,
        y,
        y_ul,
        det_mask,
        ul_mask,
    )

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

    has_mstar = np.any(np.isfinite(mstar[det_mask])) # the dault is coloring by M_star if available
    cvals = color_by if color_by is not None else (mstar if has_mstar else None)
    if np.any(det_mask):
        if cvals is not None and np.any(np.isfinite(cvals[det_mask])):
            sc = ax.scatter(
                x[det_mask],
                y[det_mask],
                c=cvals[det_mask],
                cmap="viridis",
                s=40,
                edgecolor="none",
                zorder=2,
            )
            if color_label:
                fig.colorbar(sc, ax=ax, label=color_label)
        else:
            ax.scatter(
                x[det_mask],
                y[det_mask],
                c="black",
                s=40,
                edgecolor="none",
                zorder=2,
            )
        if ids is not None:
            for xi, yi, sid in zip(x[det_mask], y[det_mask], ids[det_mask]):
                if np.isfinite(xi) and np.isfinite(yi) and (not logx or xi > 0) and (not logy or yi > 0):
                    ax.annotate(
                        str(int(sid)),
                        xy=(xi, yi),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="black",
                        alpha=0.8,
                        zorder=3,
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
        if ids is not None:
            for xi, yi, sid in zip(x[ul_mask], y_ul[ul_mask], ids[ul_mask]):
                if np.isfinite(xi) and np.isfinite(yi) and (not logx or xi > 0) and (not logy or yi > 0):
                    ax.annotate(
                        str(int(sid)),
                        xy=(xi, yi),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="royalblue",
                        alpha=0.8,
                        zorder=3,
                    )

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

    # -----------------------------------
    # Annotate correlation coefficients
    # -----------------------------------
    textstr = (
        r"$\rho$ = {:.2f} (p={:.2g})".format(rho, p_rho) + "\n" +
        r"$\tau$ = {:.2f} (p={:.2g})".format(tau, p_tau)
    )

    ax.text(
        0.05,
        0.12,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

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
    parser.add_argument(
        "--ha-fits",
        required=True,
        nargs="+",
        help="One or more Hα FITS tables (e.g., F466N and F470N).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for plots.")
    parser.add_argument("--flux-mode", choices=["flux_fit", "flux_int"], default="flux_fit",
                        help="Which Lyα flux to use for detections.")
    parser.add_argument("--caseb", type=float, default=8.7, help="Case B ratio Lyα/Hα.")
    parser.add_argument("--sphinx-table", default=None, help="Optional SPHINX FITS table.")
    parser.add_argument("--aperture-tag", default="0p6", help="Aperture tag for titles.")
    parser.add_argument("--remove-agn", action="store_true", help="Remove AGN contaminant IDs.")
    parser.add_argument(
        "--prop-fits",
        nargs="+",
        default=[
            "/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits",
            "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits",
        ],
        help="One or more Halpha catalog FITS files that contain properties (Av_50, beta, M_UV_AB).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lya_tab = Table.read(args.lya_csv, format="ascii.csv")
    ha_tabs = [Table.read(p) for p in args.ha_fits]
    if len(ha_tabs) == 1:
        ha_tab = ha_tabs[0]
    else:
        ha_tab = vstack(ha_tabs, join_type="outer", metadata_conflicts="silent")

    lya_tab["ID"] = lya_tab["ID"].astype(int)
    ha_tab["ID"] = ha_tab["ID"].astype(int)

    tab = join(lya_tab, ha_tab, keys="ID", join_type="inner")
    print(f"[INFO] Matched rows = {len(tab)}")

    if args.remove_agn:
        ids_contam = [1988, 2394, 3962, 5887, 7164, 8303, 9336]
        mask = np.isin(np.array(tab["ID"], dtype=int), ids_contam)
        tab = tab[~mask]
        print(f"[INFO] Removed AGN rows: {np.sum(mask)} | Remaining: {len(tab)}")

    ids = np.asarray(tab["ID"], dtype=int)

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
    HaL_unc = as_float(tab, "L_halpha_uncorr")
    HaL_unc_err = as_float(tab, "L_halpha_err_uncorr")

    HaL_ap = as_float(tab, "L_halpha_apcorr")
    HaL_ap_err = as_float(tab, "L_halpha_err_apcorr")

    HaL_dust = as_float(tab, "L_halpha_corr_v1")       # dust-corrected (line)
    HaL_dust_err = as_float(tab, "L_halpha_err_corr_v1")

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
        xlim = log_limits(np.concatenate([x[m_det], x[m_ul]]), pad=0.25)
        ylim = log_limits(np.concatenate([y[m_det], y_ul[m_ul]]), pad=0.25)
        plot_lya_ha(
            x, y, HaL_err, lya_L_err,
            xlabel=f"L(Hα) [{label}, erg s⁻¹]",
            ylabel="L(Lyα) [erg s⁻¹]",
            title=title,
            outpath=out,
            detections_mask=m_det,
            upper_mask=m_ul,
            y_upper=y_ul,
            ids=ids,
            bg_x=sphinx_Ha,
            bg_y=sphinx_Lya,
            add_caseb=True,
            xlim=xlim,
            ylim=ylim,
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
        ids=ids,
        add_caseb=False,
    )

    # Load properties from the Halpha catalog FITS tables (F466N/F470N)
    prop_tabs = [Table.read(p) for p in args.prop_fits]
    if len(prop_tabs) == 1:
        prop_tab = prop_tabs[0]
    else:
        prop_tab = vstack(prop_tabs, join_type="outer", metadata_conflicts="silent")

    prop_tab["ID"] = prop_tab["ID"].astype(int)
    # Keep first occurrence of each ID (IDs are expected to appear in only one table)
    prop_ids = np.asarray(prop_tab["ID"], dtype=int)
    _, uniq_idx = np.unique(prop_ids, return_index=True)
    prop_tab = prop_tab[np.sort(uniq_idx)]

    # Map properties to arrays aligned with the main table by ID
    # Only keep properties requested for the f_esc plots.
    prop_cols = ["Av_50", "beta", "M_UV_AB_uncorr", "sSFR_10Myr_50", "sSFR_100Myr_50", "Z_50"]
    prop_ids = np.asarray(prop_tab["ID"], dtype=int)
    sort_idx = np.argsort(prop_ids)
    prop_ids_sorted = prop_ids[sort_idx]
    tab_ids = np.asarray(tab["ID"], dtype=int)
    pos = np.searchsorted(prop_ids_sorted, tab_ids)
    match = (pos < len(prop_ids_sorted)) & (prop_ids_sorted[pos] == tab_ids)

    prop_arrays = {}
    for col in prop_cols:
        if col not in prop_tab.colnames:
            continue
        col_vals = as_float(prop_tab, col)
        col_vals_sorted = col_vals[sort_idx]
        out = np.full(len(tab), np.nan, dtype=float)
        out[match] = col_vals_sorted[pos[match]]
        prop_arrays[col] = out

    # Build a minimal table for property plotting only (does not modify main tab)
    prop_plot_tab = Table()
    prop_plot_tab["fesc_lya_dustcorr"] = tab["fesc_lya_dustcorr"]
    prop_plot_tab["fesc_lya_dustcorr_err"] = tab["fesc_lya_dustcorr_err"]
    prop_plot_tab["fesc_lya_dustcorr_ul"] = tab["fesc_lya_dustcorr_ul"]
    if "M_star_50" in tab.colnames:
        prop_plot_tab["M_star_50"] = as_float(tab, "M_star_50")
    for col, arr in prop_arrays.items():
        prop_plot_tab[col] = arr

    print(prop_plot_tab.colnames)
    # f_esc vs properties (with upper limits) from the Halpha catalogs
    if "E_BV_50" not in prop_plot_tab.colnames and "Av_50" in prop_plot_tab.colnames:
        Rv = 4.05  # Calzetti
        prop_plot_tab["E_BV_50"] = as_float(prop_plot_tab, "Av_50") / Rv

    # -------------------------------------------------
    # Load EW_rest from external CSV and match by ID
    # -------------------------------------------------
    ew_csv_path = "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_ew_results_final.csv"
    ew_tab = Table.read(ew_csv_path, format="ascii.csv")
    ew_tab["ID"] = ew_tab["ID"].astype(int)

    ew_ids = np.asarray(ew_tab["ID"], dtype=int)
    sort_idx = np.argsort(ew_ids)
    ew_ids_sorted = ew_ids[sort_idx]

    tab_ids = np.asarray(tab["ID"], dtype=int)
    pos = np.searchsorted(ew_ids_sorted, tab_ids)
    match = (pos < len(ew_ids_sorted)) & (ew_ids_sorted[pos] == tab_ids)

    # Detections
    ew_det = np.full(len(tab), np.nan, dtype=float)
    if "EW_rest_method2" in ew_tab.colnames:
        ew_vals = as_float(ew_tab, "EW_rest_method2")[sort_idx]
        ew_det[match] = ew_vals[pos[match]]

    # Upper limits
    ew_ul = np.full(len(tab), np.nan, dtype=float)
    if "EW_rest_method2_upper_limit" in ew_tab.colnames:
        ew_ul_vals = as_float(ew_tab, "EW_rest_method2_upper_limit")[sort_idx]
        ew_ul[match] = ew_ul_vals[pos[match]]

    prop_plot_tab["EW_rest_method2"] = ew_det
    prop_plot_tab["EW_rest_method2_ul"] = ew_ul
       

    # 1) E(B-V)
    if "E_BV_50" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="E_BV_50",
            xlabel="E(B-V)",
            out_path=os.path.join(args.out_dir, f"fesc_vs_EBV_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            sphinx_tab=sphinx_tab,
            sphinx_x_col="E_BV_median" if sphinx_tab is not None and "E_BV_median" in sphinx_tab.colnames else None,
            sphinx_y_col="fesc_lya" if sphinx_tab is not None and "fesc_lya" in sphinx_tab.colnames else None,
            ylim=None, # (1e-3, 1.05)
            xlim=None, #(0, 0.2)
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )

    # 2) A_V
    if "Av_50" in prop_plot_tab.colnames:
        print("\n Fesc detections:")
        print(np.array(prop_plot_tab["fesc_lya_dustcorr"]))   
        print("\n Fesc upper limits:")
        print(np.array(prop_plot_tab["fesc_lya_dustcorr_ul"]))
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="Av_50",
            xlabel="A_V",
            out_path=os.path.join(args.out_dir, f"fesc_vs_Av_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )

    # 3) UV beta slope
    if "beta" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="beta",
            xlabel=r"$\beta$",
            out_path=os.path.join(args.out_dir, f"fesc_vs_beta_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["M_star_50"],
            color_label=r"$M_\star$",
        )

    # 4) M_UV (uncorrected)
    if "M_UV_AB_uncorr" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="M_UV_AB_uncorr",
            xlabel=r"$M_{\rm UV}$ (AB, uncorr)",
            out_path=os.path.join(args.out_dir, f"fesc_vs_Muv_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )

    # 5) sSFR (specific star formation rate)
    if "sSFR_10Myr_50" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="sSFR_10Myr_50",
            xlabel=r"$sSFR_{10\rm Myr}$",
            out_path=os.path.join(args.out_dir, f"fesc_vs_sSFR_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )
    
    # 6) sSFR (specific star formation rate)
    if "sSFR_100Myr_50" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="sSFR_100Myr_50",
            xlabel=r"$sSFR_{100\rm Myr}$",
            out_path=os.path.join(args.out_dir, f"fesc_vs_sSFR_100Myr_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )


    # 7) Metallicity 
    if "Z_50" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="Z_50",
            xlabel=r"$Z_{50}$",
            out_path=os.path.join(args.out_dir, f"fesc_vs_Z_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )

    # 8) Stellar mass (M_star_50)
    if "M_star_50" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="M_star_50",
            xlabel=r"$M_\star$",
            out_path=os.path.join(args.out_dir, f"fesc_vs_Mstar_{args.aperture_tag}.png"),
            logy=True,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
        )
    
    # 9) Lyα Rest-frame EW 
    if "EW_rest_method2" in prop_plot_tab.colnames:
        plot_fesc_vs_property(
            prop_plot_tab,
            y_col="fesc_lya_dustcorr",
            y_err_col="fesc_lya_dustcorr_err",
            y_ul_col="fesc_lya_dustcorr_ul",
            x_col="EW_rest_method2",
            xlabel=r"$\mathrm{EW}_{\rm rest}$ (Å)",
            out_path=os.path.join(args.out_dir, f"fesc_vs_EWrest_{args.aperture_tag}.png"),
            logy=True,
            logx=False,
            ids=ids,
            color_by=prop_plot_tab["beta"],
            color_label=r"$\beta$",
            xlim=(0, 250),   # hides the extreme outlier
        )


if __name__ == "__main__":
    main()


"""

python lya_ha_fesc_analysis.py   --lya-csv /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv   --ha-fits  /home/apatrick/P1/JELSDP/F466N_with_LHa_Corey_method.fits /home/apatrick/P1/JELSDP/F470N_with_LHa_Corey_method.fits   --out-dir /cephfs/apatrick/musecosmos/scripts/sample_select/   --flux-mode flux_fit   --aperture-tag 0p6   --sphinx-table /home/apatrick/P1/JELSDP/sphinx_lya_ha_fesc_table.fits --prop-fits /home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits /home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits 



"""