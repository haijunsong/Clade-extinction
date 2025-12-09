import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 1. 参数设置
num_niches = 100                 # 总生态位数K
a_proportions = np.arange(100, 0, -1)  # 类群A占据比例（100%~1%）
q_bg = 0.1                       # 假设单物种背景灭绝概率
q_mass = 0.9                     # 假设单物种极端灭绝概率

# 2. 解析解
p_arr = a_proportions / 100  # 0~1
ana_bg_prob = q_bg ** (p_arr * num_niches)
ana_mass_prob = q_mass ** (p_arr * num_niches)

# 3. 成图
plt.figure(figsize=(8,6))
plt.plot(a_proportions, ana_bg_prob, '-', color='blue', label='Analytical (Background)')
plt.plot(a_proportions, ana_mass_prob, '-', color='red', label='Analytical (Mass)')

plt.xlabel('Proportion of Niches Occupied by Category A (%)', fontsize=12)
plt.ylabel('Probability of Complete Extinction of Category A', fontsize=12)
plt.title('Extinction Probability (Analytical Solution)', fontsize=14)
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('extinction_probability_analytical_only.png', dpi=300)
plt.show()

# 控制台输出
print(f"Background extinction, single species q = {q_bg}")
print(f"Mass extinction, single species q = {q_mass}")