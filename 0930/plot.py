import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中文字体（防止图表中的中文显示为方块，如果是Mac用户可将 'SimHei' 改为 'Arial Unicode MS'）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 录入原始数据 (Prob-3)
X = np.array([1, 0, 2, 0, 3, 1, 0, 1, 2, 0])
Y = np.array([16, 9, 17, 12, 22, 13, 8, 15, 19, 11])

# 3. 现推回归方程（基于最小二乘法代数公式）
n = len(X)
X_bar = np.mean(X)
Y_bar = np.mean(Y)

# 计算斜率 b1 和 截距 b0
b1 = np.sum((X - X_bar) * (Y - Y_bar)) / np.sum((X - X_bar) ** 2)
b0 = Y_bar - b1 * X_bar

# 打印现推的结果
print(f"现推回归方程: Y_hat = {b0:.2f} + {b1:.2f}X")

# 4. 生成用于绘制回归线的连续数据点
# 为了不让线刚好贴着边缘，X范围稍微扩大一点 (从 -0.5 到 3.5)
X_line = np.linspace(-0.5, 3.5, 100)
Y_line = b0 + b1 * X_line

# 5. 开始绘图
plt.figure(figsize=(9, 6), dpi=100) # 设置画布大小与分辨率

# 绘制散点图（原始数据）
plt.scatter(X, Y, color='#1f77b4', s=80, alpha=0.8, edgecolor='k', zorder=3, label='原始数据点 (X, Y)')

# 绘制估计的回归直线
plt.plot(X_line, Y_line, color='#d62728', linewidth=2.5, zorder=2, 
         label=f'估计回归线: $\hat{{Y}} = {b0:.1f} + {b1:.1f}X$')

# 标记样本均值点 (X_bar, Y_bar)，回归线必定穿过此点
plt.scatter([X_bar], [Y_bar], color='green', marker='*', s=200, zorder=4, 
            label=f'均值点 ({X_bar:.1f}, {Y_bar:.1f})')

# 6. 图表美化
plt.title('航班转运次数与破损安瓿数量的回归分析', fontsize=15, pad=15)
plt.xlabel('转运次数 (X)', fontsize=13)
plt.ylabel('破损安瓿数量 (Y)', fontsize=13)

# 设置坐标轴刻度
plt.xticks(np.arange(0, 4, 1), fontsize=11)
plt.yticks(fontsize=11)

# 添加网格线（半透明虚线）
plt.grid(True, linestyle='--', alpha=0.6, zorder=1)

# 显示图例
plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)

# 紧凑布局
plt.tight_layout()

# 7. 显示图像
plt.show()