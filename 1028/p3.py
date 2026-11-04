# prob3.py
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

# (a) Scatter plot of Y vs X
plt.figure(figsize=(6, 4))
plt.scatter(X, Y, color="tab:blue")
plt.xlabel("Time X")
plt.ylabel("Concentration Y")
plt.title("Concentration vs. time")
plt.tight_layout()
plt.savefig("figs/p3_scatter.png", dpi=300)
plt.close()

# (b) Standardized Box-Cox transformation
def boxcox_std(y, lam):
    y = np.asarray(y, dtype=float)
    g = np.exp(np.mean(np.log(y)))
    if abs(lam) < 1e-12:
        return g * np.log(y)
    return (y ** lam - 1.0) / (lam * g ** (lam - 1.0))

lambdas = [-0.2, -0.1, 0.0, 0.1, 0.2]
rows = []
for lam in lambdas:
    yt = boxcox_std(Y, lam)
    fit_t = sm.OLS(yt, sm.add_constant(X)).fit()
    sse_t = np.sum(fit_t.resid ** 2)
    rows.append((lam, sse_t))
    print(f"lambda = {lam:>2}: SSE = {sse_t:.8f}")

print("\nBox-Cox standardized SSE table:")
for lam, sse_t in rows:
    print(f"{lam:>4}  {sse_t:.8f}")

# (c) Fit using Y' = log10(Y)
Ylog = np.log10(Y)
fit_log = sm.OLS(Ylog, sm.add_constant(X)).fit()
b0_log, b1_log = fit_log.params
print("\n(c) Fit with Y' = log10(Y):")
print(f"log10(Y_hat) = {b0_log:.6f} + ({b1_log:.6f}) X")
print(fit_log.summary())

# (d) Plot transformed data and fitted line
Xgrid = np.linspace(X.min(), X.max(), 100)
Ylog_hat_grid = b0_log + b1_log * Xgrid

plt.figure(figsize=(6, 4))
plt.scatter(X, Ylog, color="tab:blue", label="transformed data")
plt.plot(Xgrid, Ylog_hat_grid, color="red", label="fitted line")
plt.xlabel("Time X")
plt.ylabel(r"$\log_{10}(Y)$")
plt.title("Transformed data and fitted regression line")
plt.legend()
plt.tight_layout()
plt.savefig("figs/p3_logfit.png", dpi=300)
plt.close()

# (e) Residual plot and normal probability plot
resid_log = fit_log.resid
fitted_log = fit_log.fittedvalues

plt.figure(figsize=(6, 4))
plt.scatter(fitted_log, resid_log, color="tab:blue")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel(r"Fitted value $\hat Y'_i$")
plt.ylabel(r"Residual $e_i$")
plt.title("Residuals vs. fitted values: log10(Y) model")
plt.tight_layout()
plt.savefig("figs/p3_resid_fit.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
stats.probplot(resid_log, dist="norm", plot=plt)
plt.title("Normal probability plot of residuals")
plt.tight_layout()
plt.savefig("figs/p3_qq.png", dpi=300)
plt.close()

# (f) Original units
print("\n(f) Estimated regression function in original units:")
print(f"Y_hat = 10^({b0_log:.6f} + ({b1_log:.6f}) X)")