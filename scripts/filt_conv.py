import numpy as np
import pandas as pd

path = "/home/apatrick/P1/JELSDP/JWST_NIRCam.F470N.dat"
out = "/home/apatrick/P1/JELSDP/JWST_NIRCam_F470N.csv"

data = np.genfromtxt(path, comments="#")
data = np.atleast_2d(data)
df = pd.DataFrame({"wave": data[:, 0], "throughput": data[:, 1]})
df.to_csv(out, index=False)
print(f"Saved filtered transmission data to {out}")