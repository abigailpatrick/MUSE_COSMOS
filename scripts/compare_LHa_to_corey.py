
#!/usr/bin/env python3
import numpy as np
from astropy.table import Table, join
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

PHOT = "/home/apatrick/P1/JELSDP/jels_F470N_detected_high_z_candidates_v1.01.fits"
SED  = "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_sfh_continuity_salim_v2_bpass_posteriors.fits"
COREY= "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits"

def safe(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan

def pick_col(tab, base):
    # Prefer your (my) side, then Corey (co), then unsuffixed if present
    for cand in (f"{base}_my", f"{base}_co", base):
        if cand in tab.colnames:
            return cand
    return None

phot = Table.read(PHOT)
sed  = Table.read(SED)
tab  = join(phot, sed, keys="ID", join_type="inner", metadata_conflicts="silent")
corey= Table.read(COREY)
both = join(
    tab, corey, keys="ID", join_type="inner",
    table_names=["my","co"], uniq_col_name="{col_name}_{table_name}",
    metadata_conflicts="silent"
)

cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Tcmb0=2.725*u.K, Om0=0.3)

# Column names (band-specific)
col_auto = pick_col(both, "F470N_auto_flux")
col_600  = pick_col(both, "F470N_600mas_flux")
col_z    = pick_col(both, "z1_median_1") or pick_col(both, "z1_median")
col_f1   = pick_col(both, "F_line_F444W_F470N")   # NB–BB
col_f2   = pick_col(both, "F_line_F466N_F470N")   # NB–NB
col_Lc   = pick_col(both, "L_halpha_uncorr")      # Corey’s L (should come from _co)
if col_Lc and not col_Lc.endswith("_co"):
    # If it found an unsuffixed version, try to prefer Corey's if available
    co_pref = "L_halpha_uncorr_co"
    if co_pref in both.colnames:
        col_Lc = co_pref

missing = [c for c in [col_auto, col_600, col_z, col_f1, col_f2, col_Lc] if c is None]
if missing:
    raise RuntimeError(f"Missing needed columns after join: {missing}")

rows = []
for r in both:
    ID = int(r["ID"])
    z = safe(r[col_z])
    if not np.isfinite(z):
        continue
    dL = cosmo.luminosity_distance(z).to(u.cm).value
    four_pi_d2 = 4*np.pi*dL**2

    ap = safe(r[col_auto]) / safe(r[col_600])
    f_nbbb = safe(r[col_f1])
    f_nbnb = safe(r[col_f2])
    L_corey = safe(r[col_Lc])

    pred_nbbb = four_pi_d2 * f_nbbb * ap if np.isfinite(f_nbbb) and np.isfinite(ap) else np.nan
    pred_nbnb = four_pi_d2 * f_nbnb * ap if np.isfinite(f_nbnb) and np.isfinite(ap) else np.nan
    f_implied = L_corey / (four_pi_d2 * ap) if np.isfinite(L_corey) and np.isfinite(ap) and ap > 0 else np.nan

    # Best match error
    diffs = [d for d in [abs(pred_nbbb - L_corey), abs(pred_nbnb - L_corey)] if np.isfinite(d) and np.isfinite(L_corey)]
    rel_err = (min(diffs) / L_corey) if diffs and L_corey != 0 else np.nan

    rows.append((ID, pred_nbbb, pred_nbnb, L_corey, f_nbbb, f_nbnb, f_implied, rel_err))

# Print summary sorted by relative error (largest first)
rows = [r for r in rows if np.isfinite(r[7])]
rows.sort(key=lambda x: x[7], reverse=True)

print("ID     rel_err   Lpred_nbbb     Lpred_nbnb     L_corey        f_nbbb_my    f_nbnb_my    f_implied")
for (ID, p1, p2, Lc, f1, f2, fim, re) in rows:
    print(f"{ID:5d}  {re:7.3f}  {p1:12.3e}  {p2:12.3e}  {Lc:12.3e}  {f1:10.3e}  {f2:10.3e}  {fim:10.3e}")


# ============================
# File paths
# ============================
MY_F466  = "/home/apatrick/P1/JELSDP/F466N_with_LHa_Corey_method.fits"
MY_F470  = "/home/apatrick/P1/JELSDP/F470N_4820_nbnb.fits"

COREY_F466 = "/home/apatrick/P1/JELSDP/JELS_F466N_Halpha_cat_v1p0.fits"
COREY_F470 = "/home/apatrick/P1/JELSDP/JELS_F470N_Halpha_cat_v1p0.fits"


# ============================
# Which sources to inspect
# ============================
INSPECT_IDS = [4820, 6906, 6456]  # add/remove IDs as needed
INSPECT_SAMPLE_N = 0              # set >0 to also inspect first N matched IDs


# ============================
# Helpers
# ============================
def pick_first(tab, candidates):
    for c in candidates:
        if c in tab.colnames:
            return c
    return None

def pick_z_col(tab):
    # Try common z column names (your table tends to have z1_median_1, Corey has z1_median)
    return pick_first(tab, ["z1_median_1", "z1_median", "z_phot_50", "z_peak", "z", "z_best"])

def pick_lineflux_col(tab, band):
    # Band-aware line flux columns (NB–BB first, then NB–NB; just for diagnostics)
    if band == "F466N":
        return pick_first(tab, ["F_line_F444W_F466N", "F_line_F470N_F466N"])
    if band == "F470N":
        return pick_first(tab, ["F_line_F444W_F470N", "F_line_F466N_F470N"])
    return None

def safe(x):
    return float(x) if np.isfinite(x) else np.nan

def fmt(x):
    return f"{x:.6g}" if np.isfinite(x) else "nan"

def nanmedian_ratio(a, b):
    # a/b with filtering
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b) & (b != 0)
    if not np.any(good):
        return np.nan
    return np.nanmedian(a[good] / b[good])

def print_stat(name, a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    n = int(np.sum(good))
    if n == 0:
        print(f"  {name:24s} N=0")
        return
    diff = a[good] - b[good]
    rat  = a[good] / b[good]
    print(f"  {name:24s} N={n:3d}  med_ratio={np.nanmedian(rat):.6f}  max|diff|={np.nanmax(np.abs(diff)):.3e}")


# ============================
# Core compare per band
# ============================
def compare_band(band, myfile, coreyfile, inspect_ids=None, sample_n=0):
    print("\n" + "="*39)
    print(f"Comparing band: {band}")
    print("="*39)

    # Load
    mytab   = Table.read(myfile)
    cortab  = Table.read(coreyfile)

    # Index by ID
    my_ids  = np.array(mytab["ID"], dtype=int)
    cor_ids = np.array(cortab["ID"], dtype=int)
    idx_my  = {int(i): j for j, i in enumerate(my_ids)}
    idx_cor = {int(i): j for j, i in enumerate(cor_ids)}

    # Columns
    zcol_my  = pick_z_col(mytab)
    zcol_cor = pick_z_col(cortab)

    nb_auto_col = f"{band}_auto_flux"
    nb_600_col  = f"{band}_600mas_flux"

    # Line-flux columns for L/F diagnostics (not used in ratios; just helpful prints)
    lfcol_my  = pick_lineflux_col(mytab, band)
    lfcol_cor = pick_lineflux_col(cortab, band)

    # Gather matched IDs
    common_ids = [i for i in my_ids if int(i) in idx_cor]

    # Build arrays for summary stats
    my_L_ap   = []
    cor_L_pre = []
    my_Lc1    = []
    cor_Lc1   = []
    my_Lc2    = []
    cor_Lc2   = []
    my_Ac     = []
    cor_Ac    = []
    my_Al     = []
    cor_Al    = []

    # Name shortcuts
    name_L_pre      = "L_halpha_uncorr"      # my pre-ap, Corey post-ap (hence compare to my apcorr)
    name_L_ap       = "L_halpha_apcorr"
    name_Lc1        = "L_halpha_corr_v1"
    name_Lc2        = "L_halpha_corr_v2"
    name_Ac         = "A_halpha_cont"
    name_Al         = "A_halpha_line"

    for obj_id in common_ids:
        i = idx_my[int(obj_id)]
        j = idx_cor[int(obj_id)]

        # Append for stats where columns exist in both
        if name_L_ap in mytab.colnames and name_L_pre in cortab.colnames:
            my_L_ap.append(safe(mytab[name_L_ap][i]))
            cor_L_pre.append(safe(cortab[name_L_pre][j]))

        if name_Lc1 in mytab.colnames and name_Lc1 in cortab.colnames:
            my_Lc1.append(safe(mytab[name_Lc1][i]))
            cor_Lc1.append(safe(cortab[name_Lc1][j]))

        if name_Lc2 in mytab.colnames and name_Lc2 in cortab.colnames:
            my_Lc2.append(safe(mytab[name_Lc2][i]))
            cor_Lc2.append(safe(cortab[name_Lc2][j]))

        if name_Ac in mytab.colnames and name_Ac in cortab.colnames:
            my_Ac.append(safe(mytab[name_Ac][i]))
            cor_Ac.append(safe(cortab[name_Ac][j]))

        if name_Al in mytab.colnames and name_Al in cortab.colnames:
            my_Al.append(safe(mytab[name_Al][i]))
            cor_Al.append(safe(cortab[name_Al][j]))

    # Summary stats
    print_stat("my L_apcorr vs Corey L_uncorr", my_L_ap, cor_L_pre)
    print_stat("L_corr_v1", my_Lc1, cor_Lc1)
    print_stat("L_corr_v2", my_Lc2, cor_Lc2)
    print_stat("A_halpha_cont", my_Ac, cor_Ac)
    print_stat("A_halpha_line", my_Al, cor_Al)

    # ----------------------------
    # Detailed per-source output
    # ----------------------------
    if inspect_ids or sample_n > 0:
        ids_to_show = []
        if inspect_ids:
            ids_to_show.extend([int(i) for i in inspect_ids if int(i) in idx_my and int(i) in idx_cor])
        if sample_n > 0:
            for i_ in common_ids:
                ii = int(i_)
                if ii not in ids_to_show:
                    ids_to_show.append(ii)
                if len(ids_to_show) >= (len(inspect_ids or []) + sample_n):
                    break

        print("\nDetailed per-source diagnostics:")
        for obj_id in ids_to_show:
            i = idx_my[int(obj_id)]
            j = idx_cor[int(obj_id)]

            print(f"\nID {obj_id}")

            # z diagnostics
            if zcol_my and zcol_cor:
                z_my  = safe(mytab[zcol_my][i])
                z_cor = safe(cortab[zcol_cor][j])
                print(f"  z (my {zcol_my})={fmt(z_my)}   z (Corey {zcol_cor})={fmt(z_cor)}")
            else:
                print("  z: [SKIP] no common z column found")

            # Aperture factor from NB auto/600 for this band (from my table)
            ap = np.nan
            if nb_auto_col in mytab.colnames and nb_600_col in mytab.colnames:
                nb_auto = safe(mytab[nb_auto_col][i])
                nb_600  = safe(mytab[nb_600_col][i])
                ap = (nb_auto / nb_600) if (np.isfinite(nb_auto) and np.isfinite(nb_600) and nb_600 != 0) else np.nan
                print(f"  {nb_auto_col}={fmt(nb_auto)}   {nb_600_col}={fmt(nb_600)}   ap_factor={fmt(ap)}")
            else:
                print(f"  [SKIP] aperture flux columns missing: {nb_auto_col}, {nb_600_col}")

            # Core check: (my L_uncorr × ap) vs Corey L_uncorr
            Lpre_my   = safe(mytab[name_L_pre][i]) if name_L_pre in mytab.colnames else np.nan
            Lap_my    = safe(mytab[name_L_ap][i])  if name_L_ap in mytab.colnames else np.nan
            Lpre_cor  = safe(cortab[name_L_pre][j]) if name_L_pre in cortab.colnames else np.nan

            if np.isfinite(Lpre_my) and np.isfinite(ap) and np.isfinite(Lpre_cor) and Lpre_cor != 0:
                ratio_aligned = (Lpre_my * ap) / Lpre_cor
                print(f"  (my L_uncorr × ap) / Corey L_uncorr = {fmt(ratio_aligned)}")
            elif np.isfinite(Lap_my) and np.isfinite(Lpre_cor) and Lpre_cor != 0:
                ratio_aligned = Lap_my / Lpre_cor
                print(f"  (my L_apcorr) / Corey L_uncorr     = {fmt(ratio_aligned)}")
            else:
                print("  [SKIP] cannot compute aligned L comparison")

            # Dust-corrected luminosities (ratios)
            if name_Lc1 in mytab.colnames and name_Lc1 in cortab.colnames:
                r1 = (safe(mytab[name_Lc1][i]) / safe(cortab[name_Lc1][j])) if safe(cortab[name_Lc1][j]) != 0 else np.nan
                print(f"  ratio L_corr_v1 (my/Corey) = {fmt(r1)}")
            if name_Lc2 in mytab.colnames and name_Lc2 in cortab.colnames:
                r2 = (safe(mytab[name_Lc2][i]) / safe(cortab[name_Lc2][j])) if safe(cortab[name_Lc2][j]) != 0 else np.nan
                print(f"  ratio L_corr_v2 (my/Corey) = {fmt(r2)}")

            # Optional L/F diagnostics (band-aware)
            if lfcol_my and lfcol_cor and name_L_pre in mytab.colnames:
                f_my  = safe(mytab[lfcol_my][i])
                f_cor = safe(cortab[lfcol_cor][j]) if lfcol_cor in cortab.colnames else np.nan
                Lpre_my = safe(mytab[name_L_pre][i])
                Lpre_cor= safe(cortab[name_L_pre][j]) if name_L_pre in cortab.colnames else np.nan

                lf_my  = (Lpre_my / f_my)  if np.isfinite(Lpre_my)  and np.isfinite(f_my)  and f_my != 0 else np.nan
                lf_cor = (Lpre_cor / f_cor) if np.isfinite(Lpre_cor) and np.isfinite(f_cor) and f_cor != 0 else np.nan
                print(f"  L/F (my {lfcol_my})={fmt(lf_my)}   L/F (Corey {lfcol_cor})={fmt(lf_cor)}")
            else:
                print("  L/F: [SKIP] missing L or flux column")



# ============================
# Run both bands
# ============================
if __name__ == "__main__":
    compare_band("F466N", MY_F466, COREY_F466, inspect_ids=INSPECT_IDS, sample_n=INSPECT_SAMPLE_N)
    compare_band("F470N", MY_F470, COREY_F470, inspect_ids=INSPECT_IDS, sample_n=INSPECT_SAMPLE_N)
