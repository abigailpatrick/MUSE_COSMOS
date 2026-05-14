import numpy as np

# Input values
detections = 9
total_sample = 24

# Detection fraction
f_det = detections / total_sample
f_det_p = f_det*100

# Binomial uncertainty
sigma = np.sqrt(f_det * (1 - f_det) / total_sample)
sigma_p = sigma*100

print(f"Detection fraction = {f_det_p:.1f}%")
print(f"Binomial uncertainty = {sigma_p:.1f}%")
print(f"{f_det_p:.1f}% ± {sigma_p:.1f}%")