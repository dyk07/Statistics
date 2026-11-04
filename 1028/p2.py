# prob2.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

os.makedirs("figs", exist_ok=True)
plt.rcParams["figure.dpi"] = 120

# CH03PR15.txt: 第一列 Y，第二列 X
DATA = pd.read_csv("CH03PR15.txt", sep=r"\s+", header=None, names=["Y", "X"])

X = DATA["X"].to_numpy(dtype=float)
Y = DATA["Y"].to_numpy(dtype=float)

# (a) Fit linear regression Y = b0 + b1 X
Xmat = sm.add_constant(X)
fit = sm.OLS(Y, Xmat).fit()
print(fit.summary())

b0, b1 = fit.params
print("\n(a) Estimated linear regression:")
print(f"Y_hat = {b0:.6f} + ({b1:.6f}) X")

# Residual plot
plt.figure(figsize=(6, 4))
plt.scatter(fit.fittedvalues, fit.resid, color="tab:blue")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel(r"Fitted value $\hat Y_i$")
plt.ylabel(r"Residual $e_i$")
plt.title("Residuals vs. fitted values")
plt.tight_layout()
plt.savefig("figs/p2_resid_fit.png", dpi=300)
plt.close()

# (b) Lack-of-fit F test
SSE = np.sum(fit.resid ** 2)
n = len(Y)
df_error = n - 2

grouped = DATA.groupby("X")["Y"]
SSPE = 0.0
for _, group in grouped:
    yj = group.to_numpy(dtype=float)
    SSPE += np.sum((yj - yj.mean()) ** 2)

dfPE = n - len(grouped)          # 15 - 5 = 10
SSLF = SSE - SSPE
dfLF = df_error - dfPE           # 13 - 10 = 3

MSLF = SSLF / dfLF
MSPE = SSPE / dfPE
F_stat = MSLF / MSPE
alpha = 0.025
F_crit = stats.f.ppf(1 - alpha, dfLF, dfPE)
p_value = stats.f.sf(F_stat, dfLF, dfPE)

print("\n(b) Lack-of-fit F test:")
print(f"SSE   = {SSE:.6f}")
print(f"SSPE  = {SSPE:.6f}")
print(f"SSLF  = {SSLF:.6f}")
print(f"dfPE  = {dfPE}")
print(f"dfLF  = {dfLF}")
print(f"MSLF  = {MSLF:.6f}")
print(f"MSPE  = {MSPE:.6f}")
print(f"F     = {F_stat:.6f}")
print(f"F_crit(0.975, {dfLF}, {dfPE}) = {F_crit:.6f}")
print(f"p-value = {p_value:.6e}")

if F_stat > F_crit:
    print("Decision: reject H0 at alpha = 0.025; lack of fit is significant.")
else:
    print("Decision: do not reject H0 at alpha = 0.025; no sufficient evidence of lack of fit.")