"""
Create a DataFrame to store all the JELS Ha, SED derived and MUSE Lya properties for the JELS sources.
"""

from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

# Paths, Parquet is csv with metadata - think it's useful here
table_path = Path("/cephfs/apatrick/musecosmos/dataproducts/JELS_lyaprop.fits")
df_path = table_path.with_suffix(".parquet")
meta_path = table_path.with_suffix(".meta.json")
df_path.parent.mkdir(parents=True, exist_ok=True)

# Define schema (empty DataFrame with dtypes)
dtypes = {
    # IDs / positioning
    "jels_id": "string",
    "jels_ra": "float64",
    "jels_dec": "float64",
    "jels_z": "float64",
    "jels_predict_lya": "float64",
    "muse_ra": "float64",
    "muse_dec": "float64",
    "muse_lya_wave": "float64",


    # JELSHa properties
    "ha_flux": "float64",
    "ha_flux_err": "float64",
    "ha_half_light_r": "float64",
    "ha_ew_obs": "float64",
    "ha_fwhm_kms": "float64",
    "ha_sn": "float64",

    # SED-derived
    "sed_logmstar": "float64",  # log10(M*/Msun)
    "sed_sfr": "float64",
    "sed_av": "float64",

    # Lya (MUSE)
    "lya_flux": "float64",
    "lya_flux_err": "float64",
    "lya_ew_obs": "float64",
    "lya_vshift_kms": "float64",
    "lya_fesc": "float64",
    "lyc_fesc": "float64", # Use Sophias code to derive

    # Flags / notes
    "quality_flag": "Int16",    # nullable integer
    "notes": "string",
}

df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})

# Optional: attach metadata (units/descriptions). These won’t be kept in CSV sooo hence doing like this
units = {
    "jels_id": "",
    "jels_ra": "deg",
    "jels_dec": "deg",
    "jels_z": "",
    "jels_predict_lya": "Angstrom",
    "muse_ra": "deg",
    "muse_dec": "deg",
    "muse_lya_wave": "Angstrom",
    "ha_flux": "erg s^-1 cm^-2",
    "ha_flux_err": "erg s^-1 cm^-2",
    "ha_half_light_r": "",
    "ha_ew_obs": "Angstrom",
    "ha_fwhm_kms": "km s^-1",
    "ha_sn": "",
    "sed_logmstar": "dex(Msun)",
    "sed_sfr": "Msun/yr",
    "sed_av": "mag",
    "lya_flux": "erg s^-1 cm^-2",
    "lya_flux_err": "erg s^-1 cm^-2",
    "lya_ew_obs": "Angstrom",
    "lya_vshift_kms": "km s^-1",
    "lya_fesc": "",
    "lyc_fesc": "",
    "quality_flag": "",
    "notes": "",
}
descriptions = {
    "jels_id": "JELS source identifier",
    "jels_ra": "JELS Right Ascension (J2000)",
    "jels_dec": "JELS Declination (J2000)",
    "jels_z": "JELS Redshift",
    "jels_predict_lya": "Lya emission line predicted from JELS redshift",
    "muse_ra": "MUSE Right Ascension (J2000)",
    "muse_dec": "MUSE Declination (J2000)",
    "muse_lya_wave": "Measured centroid of MUSE Lya emission line",
    "ha_flux": "Ha integrated flux",
    "ha_flux_err": "Uncertainty on Ha flux",
    "ha_half_light_r": "Ha half light radius ? Sophia said could be good to get",
    "ha_ew_obs": "Observed-frame equivalent width (Ha)",
    "ha_fwhm_kms": "FWHM from Ha line fit",
    "ha_sn": "Signal-to-noise (Ha)",
    "sed_logmstar": "log10(M*/Msun) from SED fitting",
    "sed_sfr": "SFR from SED fitting",
    "sed_av": "V-band attenuation from SED fitting",
    "lya_flux": "Lya integrated flux (MUSE)",
    "lya_flux_err": "Uncertainty on Lya flux",
    "lya_ew_obs": "Observed-frame equivalent width (Lya)",
    "lya_vshift_kms": "Velocity offset of Lya vs systemic",
    "lya_fesc": "Lya escape fraction",
    "lyc_fesc": " Use Sophias LyC code with some of the SED parameters inputted to get a Lyc fesc (careful not to be circular in comparisons)",
    "quality_flag": "Quality flag (0=good)",
    "notes": "Free-form notes",
}
metadata = {
    "title": "JELS Ha, SED, and MUSE Lya properties",
    "created_utc": datetime.utcnow().isoformat() + "Z",
    "units": units,
    "descriptions": descriptions,
    "dtypes": dtypes,
}

# Persist the empty DataFrame (can change to to_csv if want to scrap the metadata and just look at the table )
df.to_parquet(df_path, index=False)
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Created empty DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"Data:   {df_path}")
print(f"Meta:   {meta_path}")


df_path = Path("/cephfs/apatrick/musecosmos/dataproducts/JELS_lyaprop.parquet")

def upsert_row(df_path, row):
    df = pd.read_parquet(df_path)
    jid = row.get("jels_id")
    if jid is None:
        raise ValueError("row must include 'jels_id'")
    mask = df["jels_id"] == jid
    if mask.any():
        for k, v in row.items():
            if k in df.columns:
                df.loc[mask, k] = v
    else:
        # include pd.NA for nullable ints if you have them
        df.loc[len(df)] = row
    df.to_parquet(df_path, index=False)


fo
# Add sources 
upsert_row(df_path, {
    "jels_id": "JELS_0001",
    "jels_ra": 150.160247,
    "jels_dec": 2.286542,
    "jels_z": 6.076,
    "jels_predict_lya": 8602.39,
    "muse_lya_wave": pd.NA,
    "quality_flag": pd.NA,  # keeps Int16 nullable dtype intact
})
 
upsert_row(df_path, {
    "jels_id": "JELS_0002",
    "jels_ra": 150.160247,
    "jels_dec": 2.286542,
    "jels_z": 6.076,
    "jels_predict_lya": 8602.39,
    "muse_lya_wave": pd.NA,
    "quality_flag": pd.NA,  # keeps Int16 nullable dtype intact
})



def print_parquet_as_rich_table(path, max_rows=None, width=220):
    df = pd.read_parquet(path)
    if max_rows is not None:
        df = df.head(max_rows)

    table = Table(box=box.MINIMAL, header_style="bold cyan")
    for col in df.columns:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        table.add_column(col, justify=justify, overflow="fold")

    def fmt(val, dtype):
        import pandas as pd, numpy as np
        if pd.isna(val):
            return ""
        if pd.api.types.is_float_dtype(dtype):
            v = float(val)
            return f"{v:.3g}" if (abs(v) >= 1e4 or (0 < abs(v) < 1e-2)) else f"{v:.4f}"
        return str(val)

    for _, row in df.iterrows():
        table.add_row(*[fmt(row[c], df.dtypes[c]) for c in df.columns])

    console = Console(width=width)
    console.print(table)

print_parquet_as_rich_table("/cephfs/apatrick/musecosmos/dataproducts/JELS_lyaprop.parquet", max_rows=None)
