# 读取附件数据：第一列 GPA，第二列 ACT
dat <- read.table("0923/CH01PR19.txt", header = FALSE,
                  col.names = c("GPA", "ACT"))

# 查看前几行，确认读入正确
head(dat)

# ========== a. 最小二乘估计 ==========
fit <- lm(GPA ~ ACT, data = dat)

# 输出回归结果
summary(fit)
coef(fit)   # β0 和 β1 的估计值

# 写出估计回归函数
b0 <- coef(fit)[1]
b1 <- coef(fit)[2]
cat(sprintf("Estimated regression function: GPA_hat = %.4f + %.4f * ACT\n", b0, b1))

# ========== b. 绘制估计回归函数与数据 ==========
plot(dat$ACT, dat$GPA,
     xlab = "ACT test score",
     ylab = "Freshman GPA",
     main = "Estimated Regression Function and Data",
     pch = 19, col = "blue")

abline(fit, col = "red", lwd = 2)

# 可选：查看 R^2，辅助判断拟合效果
summary(fit)$r.squared