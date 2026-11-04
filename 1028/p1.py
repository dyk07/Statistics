# prob1.py
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

# KNNL 3.9 Electricity consumption residuals
X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)
e = np.array([3.2, 2.9, -1.7, -2.0, -2.3, -1.2, -0.9, 0.8, 0.7, 0.5], dtype=float)

plt.figure(figsize=(6, 4))
plt.scatter(X, e, color="tab:blue")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel(r"$X_i$")
plt.ylabel(r"$e_i$")
plt.title("Residuals vs. number of rooms")
plt.tight_layout()
plt.savefig("figs/p1_resid_x.png", dpi=300)
plt.close()

print("X =", X)
print("e =", e)
print("Residual plot saved to figs/p1_resid_x.png")