# 1. Read data (CH01PR27.txt, two‑column whitespace‑separated)
muscle <- read.table("1014\\CH01PR27.txt", col.names = c("X", "Y"))

# inspect data
head(muscle)
dim(muscle)

# 2. Fit first‑order simple linear regression Y ~ X
fit <- lm(Y ~ X, data = muscle)
summary(fit)

# print estimated regression coefficients
coef(fit)

# 3. Plot data points and overlay fitted regression line (Prob‑2a graph)
plot(Y ~ X, data = muscle,
     main = "Muscle Mass vs Age for Women",
     xlab = "Age (X)",
     ylab = "Muscle Mass (Y)",
     pch = 16, col = "darkblue")

# add regression line from lm‑fit
abline(fit, col = "red", lwd = 2)

# optional: add legend
legend("topright",
       legend = c("Observed data", expression(paste("Fitted regression: ",hat(Y) == beta[0] + beta[1]*X))),
       col = c("darkblue","red"),
       pch = c(16,NA),
       lty = c(NA,1), lwd = c(NA,2))
