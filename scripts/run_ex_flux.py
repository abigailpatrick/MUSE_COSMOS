import subprocess
import pandas as pd

CSV = "/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv"
SCRIPT = "/cephfs/apatrick/musecosmos/scripts/ex_flux.py"

ROW_LIST = {1, 3, 4, 5, 7, 8, 10, 13, 16, 24, 26, 27}

df = pd.read_csv(CSV)

for _, row in df.iterrows():
    row_index = int(row["row_index"])

    if row_index not in ROW_LIST:
        continue

    if pd.isna(row["fluxmin"]) or pd.isna(row["fluxmax"]):
        print(f"Skipping row {row_index}: missing flux bounds")
        continue

    ra = row["muse_ra"] if not pd.isna(row["muse_ra"]) else row["ra"]
    dec = row["muse_dec"] if not pd.isna(row["muse_dec"]) else row["dec"]

    wmin = row["fluxmin"] - 150
    wmax = row["fluxmax"] + 150

    cube = f"/cephfs/apatrick/musecosmos/dataproducts/extractions/source_{row_index}_subcube_20.0_3681.fits"
    outfile = f"{row_index}_flux_spectrum.png"

    cmd = [
        "python", SCRIPT,
        "--cube", cube,
        "--ra", str(ra),
        "--dec", str(dec),
        "--aperture", "0.5",
        "--pixscale", "0.2",
        "--smooth_fwhm", "4.0",
        "--wmin", str(wmin),
        "--wmax", str(wmax),
        "--fluxmin", str(row["fluxmin"]),
        "--fluxmax", str(row["fluxmax"]),
        "--z", str(row["z1_median"]),
        "--out", outfile,
    ]

    subprocess.run(cmd, check=True)
