# week6.R
# CH01PR27.txt: first column = Y (muscle mass), second column = X (age)

dat <- read.table("CH01PR27.txt", header = FALSE)
y <- dat$V1
x <- dat$V2

model <- lm(y ~ x)
summary(model)
anova(model)

b <- coef(model)
b
sigma2 <- sum(residuals(model)^2) / model$df.residual
sigma2
sqrt(sigma2)

# ---------- Problem 1.27 ----------

# 1.27(a) data and estimated regression line
png("p1_27_fit.png", width = 700, height = 500)
plot(x, y,
     main = "Muscle mass vs age with fitted regression line",
     xlab = "Age X", ylab = "Muscle mass Y",
     pch = 19)
abline(model, lwd = 2, col = "red")
dev.off()

# 1.27(b) four point estimates
# (1) difference in mean muscle mass for one-year age difference
b[2]

# (2) estimated mean muscle mass at X = 60
predict(model, newdata = data.frame(x = 60))

# (3) residual for the 8th case
e8 <- y[8] - predict(model, newdata = data.frame(x = x[8]))
e8

# (4) point estimate of sigma^2
sigma2

# ---------- Problem 3.7 ----------

# 3.7(a) stem-and-leaf plot of age
stem(x, scale = 1)

# 3.7(b) residuals and histogram
res <- residuals(model)
round(res, 4)
summary(res)
png("p1b_hist.png", width = 700, height = 500)
hist(res, breaks = 12,
     main = "Histogram of residuals",
     xlab = "Residual", ylab = "Frequency")
dev.off()

# 3.7(c) residual plots
png("p1c_fitted.png", width = 700, height = 500)
plot(fitted(model), res,
     main = "Residuals vs fitted values",
     xlab = expression(hat(Y)[i]), ylab = expression(e[i]))
abline(h = 0, lty = 2)
dev.off()

png("p1c_x.png", width = 700, height = 500)
plot(x, res,
     main = "Residuals vs age",
     xlab = "Age X", ylab = expression(e[i]))
abline(h = 0, lty = 2)
dev.off()

# 3.7(d) normal probability plot and normal-score correlation
png("p1d_qq.png", width = 700, height = 500)
qqnorm(res, main = "Normal probability plot of residuals")
qqline(res)
dev.off()

n <- length(res)
z <- qnorm(ppoints(n, a = 3/8))
rQ <- cor(sort(res), z)
rQ

# 3.7(e) Breusch-Pagan test
u <- res^2 / sigma2
aux <- lm(u ~ x)
summary(aux)

SSRstar <- anova(aux)["x", "Sum Sq"]
BP <- SSRstar / 2
p.BP <- pchisq(BP, df = 1, lower.tail = FALSE)

SSRstar
BP
p.BP
qchisq(0.99, df = 1)