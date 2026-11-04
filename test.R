# 1. 生成模拟数据
set.seed(42)                      # 固定随机种子，保证结果可复现
n <- 100
df <- data.frame(
  group = rep(c("A", "B"), each = n / 2),
  value = c(rnorm(n / 2, mean = 5), rnorm(n / 2, mean = 7))
)

# 2. 查看前几行
head(df)

# 3. 分组描述统计（均值、标准差）
aggregate(value ~ group, data = df,
          FUN = function(x) c(mean = mean(x), sd = sd(x)))

# 4. 独立样本 t 检验
t.test(value ~ group, data = df)

# 5. 画箱线图
boxplot(value ~ group, data = df,
        col = c("lightblue", "lightgreen"),
        main = "两组数据分布对比",
        xlab = "组别", ylab = "数值")