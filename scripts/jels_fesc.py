import numpy as np
from astropy.table import Table, join
import matplotlib.pyplot as plt


# -------------------------------------------------
# Paths
# -------------------------------------------------
csv_path  = "/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv"
fits_path = "/home/apatrick/P1/JELSDP/combined_selected_sources_with_LHa.fits"
out_dir   = "/home/apatrick/P1/outputfiles"


# -------------------------------------------------
# Helper: load a column as float with NaNs handled
# -------------------------------------------------
def as_float(tab, name):
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


# -------------------------------------------------
# Load input tables and join
# -------------------------------------------------
ly_tab = Table.read(csv_path, format="ascii.csv")
ha_tab = Table.read(fits_path)

ly_tab["ID"] = ly_tab["ID"].astype(int)
ha_tab["ID"] = ha_tab["ID"].astype(int)

tab = join(ly_tab, ha_tab, keys="ID", join_type="inner")
print(f"[INFO] Matched rows = {len(tab)}")

# Remove AGN
AGN_IDs = [1988, 5887]
tab = tab[~np.isin(tab["ID"], AGN_IDs)]
print(f"[INFO] Removed AGN. Remaining rows = {len(tab)}")

# Remove merged sources
merged_ids = [3228]
tab = tab[~np.isin(tab["ID"], merged_ids)]
print(f"[INFO] Removed merged sources. Remaining rows = {len(tab)}")


# -------------------------------------------------
# Extract Lyα and Hα quantities
# -------------------------------------------------
lya_L = as_float(tab, "lya_l")
lya_L_err = as_float(tab, "lya_l_err")

HaL_unc      = as_float(tab, "L_Ha_uncorr")
HaL_unc_err  = as_float(tab, "L_Ha_uncorr_err")

HaL_dust     = as_float(tab, "L_Ha_ap_dustcorr_line")
HaL_dust_err = as_float(tab, "L_Ha_ap_dustcorr_line_err")


# -------------------------------------------------
# Compute Lyα escape fractions
# -------------------------------------------------
caseB = 8.7

fesc_uncorr = lya_L / (caseB * HaL_unc)
fesc_corr   = lya_L / (caseB * HaL_dust)

fesc_uncorr_err = fesc_uncorr * np.sqrt(
    (lya_L_err / lya_L) ** 2 + (HaL_unc_err / HaL_unc) ** 2
)

fesc_corr_err = fesc_corr * np.sqrt(
    (lya_L_err / lya_L) ** 2 + (HaL_dust_err / HaL_dust) ** 2
)

tab["fesc_lya_uncorr"] = fesc_uncorr
tab["fesc_lya_uncorr_err"] = fesc_uncorr_err
tab["fesc_lya_corr"] = fesc_corr
tab["fesc_lya_corr_err"] = fesc_corr_err


# -------------------------------------------------
# Dust: E(B-V) from Av
# -------------------------------------------------
Rv = 4.05

Av_16 = as_float(tab, "Av_16")
Av_50 = as_float(tab, "Av_50")
Av_84 = as_float(tab, "Av_84")

tab["E_BV_50"] = Av_50 / Rv
tab["E_BV_16"] = Av_16 / Rv
tab["E_BV_84"] = Av_84 / Rv

tab["E_BV_err_low"]  = tab["E_BV_50"] - tab["E_BV_16"]
tab["E_BV_err_high"] = tab["E_BV_84"] - tab["E_BV_50"]


# -------------------------------------------------
# SED properties
# -------------------------------------------------
tab["M_star_50"] = as_float(tab, "M_star_50")
tab["sSFR_50"]   = as_float(tab, "sSFR_100Myr_50")
tab["SFR_ratio_50"] = as_float(tab, "SFR_ratio_50")


# -------------------------------------------------
# Sizes
# -------------------------------------------------
tab["size_50"] = np.full(len(tab), np.nan)

mask_F470N = tab["SOURCE_CAT"] == "F470N"
mask_F466N = tab["SOURCE_CAT"] == "F466N"

tab["size_50"][mask_F470N] = as_float(tab[mask_F470N], "F470N_half_light_radius")
tab["size_50"][mask_F466N] = as_float(tab[mask_F466N], "F466N_half_light_radius")


# -------------------------------------------------
# ORIGINAL plotting function (unchanged)
# -------------------------------------------------
def plot_fesc_vs_property(tab, y_col, y_err_col, x_col, xlim=None, ylim=None,
                          x_err_col=None, xlabel="",
                          ylabel=r"$f^{\mathrm{Ly}\alpha}_\mathrm{esc}$",
                          out_path="fesc_vs_prop.png",
                          logy=False, logx=False):

    x = np.array(tab[x_col])
    y = np.array(tab[y_col])
    yerr = np.array(tab[y_err_col])
    z = np.array(tab["z1_median_1"])

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if logx: mask &= x > 0
    if logy: mask &= y > 0

    plt.figure(figsize=(6.5,5))
    sc = plt.scatter(
        x[mask], y[mask],
        c=z[mask],
        s=40,
        alpha=0.85,
        edgecolor='none',
        cmap="viridis"
    )
    plt.colorbar(sc, label="Redshift")

    plt.errorbar(
        x[mask], y[mask],
        yerr=yerr[mask],
        fmt='none',
        ecolor='gray',
        alpha=0.6,
        capsize=2
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_path}")


# -------------------------------------------------
# FIXED SPHINX plotting function (asymmetric xerr)
# -------------------------------------------------
def plot_fesc_vs_property_with_sphinx(
    tab, y_col, y_err_col, x_col,
    x_err_cols=None,
    sphinx_tab=None, sphinx_x_col=None, sphinx_y_col=None,
    xlabel="", ylabel=r"$f^{\mathrm{Ly}\alpha}_\mathrm{esc}$",
    xlim=None, ylim=(1e-2,1),
    logx=False, logy=False,
    out_path="plot.png",
    bg_color="0.8", bg_alpha=0.4,
):

    x = np.asarray(tab[x_col], dtype=float)
    y = np.asarray(tab[y_col], dtype=float)
    yerr = np.asarray(tab[y_err_col], dtype=float)
    mstar = np.asarray(tab["M_star_50"], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(mstar)
    if logx: mask &= x > 0
    if logy: mask &= y > 0

    if x_err_cols is not None:
        xerr_low  = np.asarray(tab[x_err_cols[0]], dtype=float)[mask]
        xerr_high = np.asarray(tab[x_err_cols[1]], dtype=float)[mask]
        xerr = np.vstack([xerr_low, xerr_high])
    else:
        xerr = None

    plt.figure(figsize=(6.5,5))

    if sphinx_tab is not None:
        sx = np.asarray(sphinx_tab[sphinx_x_col], dtype=float)
        sy = np.asarray(sphinx_tab[sphinx_y_col], dtype=float)
        sm = np.isfinite(sx) & np.isfinite(sy)
        plt.scatter(sx[sm], sy[sm], c=bg_color, s=25, alpha=bg_alpha, zorder=0)

    sc = plt.scatter(
        x[mask], y[mask],
        c=mstar[mask],
        cmap="viridis",
        s=40,
        edgecolor="none",
        zorder=2
    )
    plt.colorbar(sc, label=r"$M_\star\,[M_\odot]$")

    plt.errorbar(
        x[mask], y[mask],
        yerr=yerr[mask],
        xerr=xerr,
        fmt='none',
        ecolor='gray',
        alpha=0.6,
        capsize=2,
        zorder=1
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_path}")


# -------------------------------------------------
# ORIGINAL plots
# -------------------------------------------------
plot_fesc_vs_property(tab, "fesc_lya_corr", "fesc_lya_corr_err", "E_BV_50",
                      xlabel="E(B-V)", logy=True,
                      out_path=f"{out_dir}/fesc_vs_EBV.png")

plot_fesc_vs_property(tab, "fesc_lya_corr", "fesc_lya_corr_err", "M_star_50",
                      xlabel=r"log $(M_\star/M_\odot)$", logy=True,
                      out_path=f"{out_dir}/fesc_vs_Mstar.png")

plot_fesc_vs_property(tab, "fesc_lya_corr", "fesc_lya_corr_err", "sSFR_50",
                      xlabel=r"sSFR [yr$^{-1}$]", logy=True,
                      out_path=f"{out_dir}/fesc_vs_sSFR.png")

plot_fesc_vs_property(tab, "fesc_lya_corr", "fesc_lya_corr_err", "SFR_ratio_50",
                      xlabel="SFR ratio", logy=True, logx=True,
                      out_path=f"{out_dir}/fesc_vs_SFRratio.png")

plot_fesc_vs_property(tab, "fesc_lya_corr", "fesc_lya_corr_err", "size_50",
                      xlabel="Half-light radius [kpc]", logy=True,
                      out_path=f"{out_dir}/fesc_vs_size.png")


# -------------------------------------------------
# SPHINX overlay plot
# -------------------------------------------------
sphinx_tab = Table.read("/home/apatrick/P1/JELSDP/sphinx_lya_ha_fesc_table.fits")
sphinx_tab["E_BV_median"] = np.median(
    [sphinx_tab[f"ebmv_dir_{i}"] for i in range(10)], axis=0
)

plot_fesc_vs_property_with_sphinx(
    tab,
    y_col="fesc_lya_corr",
    y_err_col="fesc_lya_corr_err",
    x_col="E_BV_50",
    x_err_cols=None,
    sphinx_tab=sphinx_tab,
    sphinx_x_col="E_BV_median",
    sphinx_y_col="fesc_lya",
    xlabel="E(B-V)",
    xlim=(0, 0.2),
    logy=True,
    out_path=f"{out_dir}/fesc_vs_EBV_with_sphinx.png"
)
