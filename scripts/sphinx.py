import json
import numpy as np
from astropy.table import Table, join

# -------------------------------------------------
# Paths
# -------------------------------------------------
base         = "/home/apatrick/P1/JELSDP"
lya_file     = f"{base}/lya_spec_prof_z6.json"
ha_file      = f"{base}/ha_spec_prof_z6.json"
galaxy_table = f"{base}/all_basic_data.csv"
out_table    = f"{base}/sphinx_lya_ha_fesc_table.fits"

# -------------------------------------------------
# Load JSON — extract intrinsic (top-level) and
# observed (mean over 10 post-RT sightlines)
# -------------------------------------------------
def load_sphinx_json(fname):
    """
    Returns:
        halo_ids  : (N,)    int
        logL_int  : (N,)    intrinsic log10 erg/s  [top-level lum]
        logL_obs  : (N,)    observed  log10 erg/s  [mean of 10 sightlines, linear avg]
        logL_dirs : (N, 10) per-sightline observed log10 erg/s
    """
    with open(fname) as f:
        d = json.load(f)

    is_lya  = "lya" in fname
    lum_key = "lya_lum" if is_lya else "ha_lum"

    halo_ids  = []
    logL_int  = []
    logL_obs  = []
    logL_dirs = []

    for hid, v in d.items():
        halo_ids.append(int(hid))

        # Intrinsic: top-level scalar
        logL_int.append(v[lum_key])

        # Observed: per-sightline post-RT values
        sightlines = np.array([v[f"dir_{i}"][lum_key] for i in range(10)])
        logL_dirs.append(sightlines)

        # Average in linear space → back to log
        logL_obs.append(np.log10(np.mean(10**sightlines)))

    return (
        np.array(halo_ids),
        np.array(logL_int),
        np.array(logL_obs),
        np.array(logL_dirs),
    )

# -------------------------------------------------
# Load Lyα and Hα
# -------------------------------------------------
lya_id, logL_lya_int, logL_lya_obs, logL_lya_dirs = load_sphinx_json(lya_file)
ha_id,  logL_ha_int,  logL_ha_obs,  logL_ha_dirs  = load_sphinx_json(ha_file)

# -------------------------------------------------
# Build table
# -------------------------------------------------
tab = Table()
tab["halo_id"] = lya_id

# Intrinsic luminosities (pre-RT, from top-level JSON)
tab["logL_lya_int"] = logL_lya_int
tab["logL_ha_int"]  = logL_ha_int
tab["int_lya"]      = 10**logL_lya_int
tab["int_ha"]       = 10**logL_ha_int

# Observed luminosities (post-RT, mean over 10 sightlines)
tab["logL_lya_obs"] = logL_lya_obs
tab["logL_ha_obs"]  = logL_ha_obs
tab["obs_lya"]      = 10**logL_lya_obs
tab["obs_ha"]       = 10**logL_ha_obs

# -------------------------------------------------
# Escape fractions
#
# fesc_obs_obs : obs Lya / (8.7 * obs Ha)
#                both post-RT — analogous to uncorrected observations
#
# fesc_obs_int : obs Lya / (8.7 * int Ha)
#                your observational definition — dust-corrected Ha denominator
# -------------------------------------------------
tab["fesc_obs_obs"] = tab["obs_lya"] / (8.7 * tab["obs_ha"])
tab["fesc_obs_int"] = tab["obs_lya"] / (8.7 * tab["int_ha"])

tab["logfesc_obs_obs"] = np.log10(tab["fesc_obs_obs"])
tab["logfesc_obs_int"] = np.log10(tab["fesc_obs_int"])

# -------------------------------------------------
# Join with galaxy properties
# -------------------------------------------------
gal_tab = Table.read(galaxy_table, format="ascii.csv")
gal_tab["halo_id"] = gal_tab["halo_id"].astype(int)

final_tab = join(tab, gal_tab, keys="halo_id", join_type="inner")

# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
print(f"[INFO] Final table size: {len(final_tab)} galaxies")
print(f"[CHECK] Median logL_lya_int vs CSV H__1_1215.67A_int: "
      f"{np.median(final_tab['logL_lya_int'] - np.log10(final_tab['H__1_1215.67A_int'])):.4f}  (expect ~0)")
print(f"[CHECK] Median logL_ha_int  vs CSV H__1_6562.80A_int: "
      f"{np.median(final_tab['logL_ha_int']  - np.log10(final_tab['H__1_6562.80A_int'])):.4f}  (expect ~0)")
print(f"[CHECK] Median fesc_obs_obs = {np.nanmedian(final_tab['fesc_obs_obs']):.3f}")
print(f"[CHECK] Median fesc_obs_int = {np.nanmedian(final_tab['fesc_obs_int']):.3f}")

# -------------------------------------------------
# Save
# -------------------------------------------------
final_tab.write(out_table, overwrite=True)
print(f"[OK] Saved to {out_table}")