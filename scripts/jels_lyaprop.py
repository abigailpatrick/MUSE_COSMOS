import pandas as pd

# Load CSV
df = pd.read_csv("/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv")

# Add columns 
if "muse_ra" not in df.columns:
    df["muse_ra"] = pd.NA
if "muse_dec" not in df.columns:
    df["muse_dec"] = pd.NA
if "muse_lya" not in df.columns:
    df["muse_lya"] = pd.NA
if "origin_lya_flux" not in df.columns:
    df["origin_lya_flux"] = pd.NA

# Set values 
ra = {3:150.149, 7:150.116, 10:150.157, 14:150.099, 16:150.078}
dec = {3:2.302, 7:2.322, 10:2.337, 14:2.344, 16:2.350}
lya = {3:8606, 7:8608, 10:8593, 14:8572, 16:8578}
flux = {3:902.9, 7:296.5, 10:1218.9, 14:457.9, 16:938.1}

for r, v in ra.items():
    if r <= len(df):
        df.loc[df["row_index"] == r, "muse_ra"] = v

for r, v in dec.items():
    if r <= len(df):
        df.loc[df["row_index"] == r, "muse_dec"] = v

for r, v in lya.items():
    if r <= len(df):
        df.loc[df["row_index"] == r, "muse_lya"] = v

for r, v in flux.items():
    if r <= len(df):
        df.loc[df["row_index"] == r, "origin_lya_flux"] = v


# Save updated CSV
df.to_csv("/home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv", index=False)
print("Saved to /home/apatrick/P1/outputfiles/jels_halpha_candidates_mosaic_all_updated.csv")

"""


# Set values for row_index 4, 8, 12
updates = {4: 1000, 8: 2345, 12: 6745}
for r, v in updates.items():
    if r <= len(df):
        df.loc[df["row_index"] == r, "jels_predict_lya"] = v

if "row_index" not in df.columns:
    df.insert(0, "row_index", range(1, len(df) + 1))
if "jels_predict_lya" not in df.columns:
    df["jels_predict_lya"] = pd.NA


# from the cube extraction script 
jels_predict_lya =[8602.39,8612.05,8601.60,8620.63,8612.82,8654.13,8610.31,8622.67,8609.27,8619.23,8609.61,8608.49,8613.97,8602.89,8604.93,8599.07,8628.45,8613.70,8619.19,8608.31,8619.73,8611.96,8610.05,8718.34,8718.9,8715.94,8708.50]
df["jels_predict_lya"] = jels_predict_lya
"""