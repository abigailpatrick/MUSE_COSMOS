import json
import numpy as np
from astropy.table import Table, join

# -------------------------------------------------
# Paths 
# -------------------------------------------------
base = "/home/apatrick/P1/JELSDP"

lya_file = f"{base}/lya_spec_prof_z6.json"
ha_file  = f"{base}/ha_spec_prof_z6.json"

galaxy_table = f"{base}/all_basic_data.csv"
out_table = f"{base}/sphinx_lya_ha_fesc_table.fits"

# -------------------------------------------------
# Helper: load SPHINX line luminosity JSON
# -------------------------------------------------
def load_sphinx_line_luminosity(fname):
    """
    Returns:
        halo_ids : np.array
        logL_mean : np.array   (angle-averaged log10 erg/s)
        logL_dirs : np.array   (Nhalo, 10) sightlines
    """
    with open(fname) as f:
        d = json.load(f)

    halo_ids = []
    logL_mean = []
    logL_dirs = []

    for hid, v in d.items():
        halo_ids.append(int(hid))

        # angle-averaged
        logL_mean.append(v["lya_lum"] if "lya" in fname else v["ha_lum"])

        # sightlines
        dirs = [v[f"dir_{i}"] for i in range(10)]
        logL_dirs.append(dirs)

    return (
        np.array(halo_ids),
        np.array(logL_mean),
        np.array(logL_dirs)
    )

# -------------------------------------------------
# Load Lyα and Hα luminosities
# -------------------------------------------------
lya_id, logL_lya, logL_lya_dirs = load_sphinx_line_luminosity(lya_file)
ha_id,  logL_ha,  logL_ha_dirs  = load_sphinx_line_luminosity(ha_file)

# -------------------------------------------------
# Build luminosity table
# -------------------------------------------------
tab_lum = Table()
tab_lum["halo_id"] = lya_id

tab_lum["logL_lya"] = logL_lya
tab_lum["logL_ha"]  = logL_ha

# linear luminosities
tab_lum["L_lya"] = 10**logL_lya
tab_lum["L_ha"]  = 10**logL_ha

# -------------------------------------------------
# Lyα escape fraction
# fesc = L(Lyα) / [8.7 L(Hα)]
# -------------------------------------------------
tab_lum["fesc_lya"] = tab_lum["L_lya"] / (8.7 * tab_lum["L_ha"])

# Optional: log fesc
tab_lum["logfesc_lya"] = np.log10(tab_lum["fesc_lya"])

# -------------------------------------------------
# Load galaxy properties
# -------------------------------------------------
gal_tab = Table.read(galaxy_table, format="ascii.csv")

# ensure matching type
gal_tab["halo_id"] = gal_tab["halo_id"].astype(int)

# -------------------------------------------------
# Join everything
# -------------------------------------------------
final_tab = join(tab_lum, gal_tab, keys="halo_id", join_type="inner")

print(f"[INFO] Final SPHINX table size: {len(final_tab)} galaxies")

# -------------------------------------------------
# Save
# -------------------------------------------------
final_tab.write(out_table, overwrite=True)
print(f"[OK] Saved to {out_table}")



