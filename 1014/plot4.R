# CH01PR27 原始数据
X <- c(106,106,97,113,96,119,92,112,92,102,107,107,102,115,101,87,91,97,82,78,95,98,94,96,100,84,70,104,76,93,73,73,76,80,84,71,64,88,79,88,73,74,76,87,70,69,54,62,78,65,64,74,87,63,82,80,52,56,70,74)
Y <- c(43,41,47,46,45,41,47,41,48,48,42,47,43,44,42,55,57,56,59,57,54,53,52,53,54,60,59,51,59,57,68,63,60,63,63,64,66,65,60,65,65,69,61,70,68,78,78,78,72,70,73,76,78,78,71,75,77,76,72,76)

muscle <- data.frame(X,Y)
fit <- lm(Y ~ X, data = muscle)

yhat <- fitted(fit)          # \hat Y_i
e <- residuals(fit)          # Y_i - \hat Y_i 残差
ybar <- mean(Y)
dev_reg <- yhat - ybar       # \hat Y_i - \bar Y 回归偏差

# 设置统一y轴范围，两张图标尺完全一致
y_range <- range(c(e, dev_reg))

# 输出p4.png，2行1列子图
png("p4.png", width = 900, height = 700)
par(mfrow = c(2,1))  # 2行1列布局

# 第一张：残差 Y_i‑Ŷ_i 对 X
plot(x = muscle$X, y = e,
     main = expression(paste("Residuals: ", Y[i] - hat(Y[i]), " vs X")),
     xlab = "Age X",
     ylab = expression(Y[i] - hat(Y[i])),
     ylim = y_range,
     pch = 16, col = "darkred")
abline(h = 0, lty = 2, col = "black")

# 第二张：回归偏差 Ŷ_i‑Ȳ 对 X，用完全一样ylim
plot(x = muscle$X, y = dev_reg,
     main = expression(paste("Regression deviations: ", hat(Y[i]) - bar(Y), " vs X")),
     xlab = "Age X",
     ylab = expression(hat(Y[i]) - bar(Y)),
     ylim = y_range,
     pch = 16, col = "darkblue")
abline(h = 0, lty = 2, col = "black")

par(mfrow=c(1,1))
dev.off()
