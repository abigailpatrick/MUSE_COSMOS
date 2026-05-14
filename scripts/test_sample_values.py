import pandas as pd

df = pd.read_csv(
    "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged_eta_1.csv"
)

col = "lya_skew"

print("Column:", col)
print("Mean =", df[col].mean())
print("Median =", df[col].median())
print("Min =", df[col].min())
print("Max =", df[col].max())