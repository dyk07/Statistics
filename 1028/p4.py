# prob4.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

os.makedirs("figs", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

# Michaelis-Menten data
DATA = pd.DataFrame({
    "C": [0.02, 0.02, 0.06, 0.06, 0.11, 0.11,
          0.22, 0.22, 0.56, 0.56, 1.10, 1.10],
    "V": [76, 47, 97, 107, 123, 139,
          159, 152, 191, 201, 207, 200]
})

C = DATA["C"].to_numpy(dtype=float)
V = DATA["V"].to_numpy(dtype=float)

# (a) Scatterplot V vs C
plt.figure(figsize=(6, 4))
plt.scatter(C, V, color="tab:blue")
plt.xlabel("Substrate concentration C")
plt.ylabel("Velocity V")
plt.title("Velocity vs. concentration")
plt.tight_layout()
plt.savefig("figs/p4_vc.png", dpi=300)
plt.close()

# (b) Transform 1/V and 1/C, scatterplot
invC = 1.0 / C
invV = 1.0 / V

plt.figure(figsize=(6, 4))
plt.scatter(invC, invV, color="tab:blue")
plt.xlabel(r"$1/C$")
plt.ylabel(r"$1/V$")
plt.title(r"$1/V$ vs. $1/C$")
plt.tight_layout()
plt.savefig("figs/p4_inv.png", dpi=300)
plt.close()

# (c) Distribution and influence of 1/C
Xinv = sm.add_constant(invC)
fit_inv = sm.OLS(invV, Xinv).fit()
infl = fit_inv.get_influence()
hat_diag = infl.hat_matrix_diag
cooks_d = infl.cooks_distance[0]

influence_table = pd.DataFrame({
    "C": C,
    "invC": invC,
    "hat_diag": hat_diag,
    "cooks_d": cooks_d
}).sort_values("cooks_d", ascending=False)

print("(c) Influence diagnostics for 1/V ~ 1/C:")
print(influence_table.to_string(index=False))

# (d) Least squares regression for 1/V vs 1/C, residual plot
b0_inv, b1_inv = fit_inv.params
print("\n(d) Estimated linear model:")
print(f"1/V_hat = {b0_inv:.8f} + ({b1_inv:.8f}) * (1/C)")
print(fit_inv.summary())

plt.figure(figsize=(6, 4))
plt.scatter(fit_inv.fittedvalues, fit_inv.resid, color="tab:blue")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel(r"Fitted value $\widehat{1/V}_i$")
plt.ylabel(r"Residual $e_i$")
plt.title(r"Residuals vs. fitted values: $1/V$ model")
plt.tight_layout()
plt.savefig("figs/p4_resid.png", dpi=300)
plt.close()

# (e) Scatterplot V vs C and add predicted V from linear model
Cgrid = np.linspace(C.min(), C.max(), 200)
Vhat_grid = 1.0 / (b0_inv + b1_inv / Cgrid)

plt.figure(figsize=(6, 4))
plt.scatter(C, V, color="tab:blue", label="observed")
plt.plot(Cgrid, Vhat_grid, color="red", label="predicted from 1/V model")
plt.xlabel("Substrate concentration C")
plt.ylabel("Velocity V")
plt.title("Observed V and predicted V from transformed model")
plt.legend()
plt.tight_layout()
plt.savefig("figs/p4_vc_fit.png", dpi=300)
plt.close()

print("\n(e) Predicted V = 1 / (b0 + b1 / C) is plotted in figs/p4_vc_fit.png")