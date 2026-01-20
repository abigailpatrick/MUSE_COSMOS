import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join

# -------------------------------------------------
# Paths
# -------------------------------------------------
csv_path  = "/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv"
fits_path = "/home/apatrick/P1/JELSDP/combined_selected_sources_with_LHa.fits"
out_dir   = "/home/apatrick/P1/outputfiles"
sphinx_table_path = "/home/apatrick/P1/JELSDP/sphinx_lya_ha_fesc_table.fits"

sphinx_tab = Table.read(sphinx_table_path)

# Use angle-averaged Lya & Ha
sphinx_Lya = sphinx_tab["L_lya"]
sphinx_Ha  = sphinx_tab["L_ha"]
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
        arr = np.array([
            float(x) if str(x).strip() not in ("", "nan", "None") else np.nan
            for x in data
        ], dtype=float)

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



# -------------------------------------------------
# Extract columns
# -------------------------------------------------
lya_flux = as_float(tab, "spec_flux")
lya_L    = as_float(tab, "lya_l")

ha_flux  = as_float(tab, "Ha_flux")
ha_flux_err = as_float(tab, "Ha_flux_err")

HaL_unc  = as_float(tab, "L_Ha_uncorr")
HaL_unc_err = as_float(tab, "L_Ha_uncorr_err")

HaL_ap   = as_float(tab, "L_Ha_apcorr")
HaL_ap_err = as_float(tab, "L_Ha_apcorr_err")

HaL_dust = as_float(tab, "L_Ha_ap_dustcorr_line")
HaL_dust_err = as_float(tab, "L_Ha_ap_dustcorr_line_err")

z = as_float(tab, "z1_median_1")
M_star = as_float(tab, "M_star_50")

# -------------------------------------------------
# Filtering helpers
# -------------------------------------------------
def good(x, y, xerr=None):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if xerr is not None:
        m &= np.isfinite(xerr) & (xerr > 0) & (x > xerr)
    return m


# -------------------------------------------------
# Plotting function
# -------------------------------------------------
def scatter_with_xerr(x, y, xerr, z, M_star, xlabel, ylabel, title, outfile,
                      xlim=None, ylim=None, lines=True,
                      bg_x=None, bg_y=None, bg_color='0.8', bg_alpha=0.5):
    plt.figure(figsize=(6.3, 5.4))

    # Grey background points first
    if bg_x is not None and bg_y is not None:
        plt.scatter(bg_x, bg_y,
                    c=bg_color,
                    s=30,
                    alpha=bg_alpha,
                    edgecolor='none',
                    zorder=0, label="SPHINX z = 6")

    # Error bars (behind points)
    plt.errorbar(
        x, y,
        xerr=xerr,
        fmt='none',
        ecolor='0.6',
        elinewidth=1.3,
        capsize=2,
        alpha=0.8,
        zorder=1
    )

    # Scatter points
    sc = plt.scatter(
        x, y,
        c=M_star,
        s=40,
        alpha=0.95,
        edgecolor='none',
        cmap='viridis',
        zorder=2
    )

    cbar = plt.colorbar(sc)
    cbar.set_label(" M$_*$ [M$_☉$]")

    plt.xscale("log")
    plt.yscale("log")

    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)

    if lines == True:
        xx = np.logspace(39, 44, 200)
        plt.plot(xx, xx, color="black", lw=1, alpha=0.4,linestyle="--", label="1:1")
        plt.plot(xx, xx * 8.7, color="black", alpha=0.9, linestyle="-", lw=1, label="L(Lyα)= 8.7 L(Hα) (Case B)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if lines == True:
        plt.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"[OK] Saved plot to {outfile}")


# -------------------------------------------------
# 1) Uncorrected luminosity
# -------------------------------------------------
mask = good(HaL_unc, lya_L, HaL_unc_err)
scatter_with_xerr(
    HaL_unc[mask],
    lya_L[mask],
    HaL_unc_err[mask],
    z[mask],
    M_star[mask],
    "L(Hα) [uncorrected, erg s⁻¹]",
    "L(Lyα) [erg s⁻¹]",
    "Lyα vs Hα (uncorrected)",
    f"{out_dir}/Lya_vs_Ha_uncorr.png",
    xlim=(5e39, 5e43),
    ylim=(1e41, 5e43),
    bg_x=sphinx_Ha,
    bg_y=sphinx_Lya,
    bg_color='0.8',
    bg_alpha=0.6
)

# -------------------------------------------------
# 2) Aperture-corrected luminosity
# -------------------------------------------------
mask = good(HaL_ap, lya_L, HaL_ap_err)
scatter_with_xerr(
    HaL_ap[mask],
    lya_L[mask],
    HaL_ap_err[mask],
    z[mask],
    M_star[mask],
    "L(Hα) [apcorr, erg s⁻¹]",
    "L(Lyα) [erg s⁻¹]",
    "Lyα vs Hα (apcorr)",
    f"{out_dir}/Lya_vs_Ha_apcorr.png",
    xlim=(5e39, 5e43),
    ylim=(1e41, 5e43)
)

# -------------------------------------------------
# 3) Dust-corrected luminosity
# -------------------------------------------------
mask = good(HaL_dust, lya_L, HaL_dust_err)
scatter_with_xerr(
    HaL_dust[mask],
    lya_L[mask],
    HaL_dust_err[mask],
    z[mask],
    M_star[mask],
    "L(Hα) [dust corrected, erg s⁻¹]",
    "L(Lyα) [erg s⁻¹]",
    "Lyα vs Hα (dust corrected)",
    f"{out_dir}/Lya_vs_Ha_dustcorr.png",
    xlim=(5e39, 5e43),
    ylim=(1e41, 5e43),
    bg_x=sphinx_Ha,
    bg_y=sphinx_Lya,
    bg_color='0.8',
    bg_alpha=0.6
)

# -------------------------------------------------
# 4) Flux–flux
# -------------------------------------------------
mask = good(ha_flux, lya_flux, ha_flux_err)
scatter_with_xerr(
    ha_flux[mask],
    lya_flux[mask] * 1e-20,
    ha_flux_err[mask],
    z[mask],
    M_star[mask],
    "F(Hα) [erg s⁻¹ cm⁻²]",
    "F(Lyα) [erg s⁻¹ cm⁻²]",
    "Lyα Flux vs Hα Flux",
    f"{out_dir}/Flux_Lya_vs_Ha.png",
    lines=False,
)

