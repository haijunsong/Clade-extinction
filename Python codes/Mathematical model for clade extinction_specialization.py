import matplotlib
matplotlib.use('Qt5Agg')  # 必须在import pyplot之前。也可以用 'QtAgg'，新版推荐
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ====== 可调节参数 ======
num_niches = 100      # 总生态位数

use_variable_q = True # True: q随生态位宽度变化；False: q为常数

a_proportions = np.arange(100, 0, -1) / 100  # 1.0 ~ 0.01

# 生态位宽度函数
def niche_width(p):
    return 0.1 + 0.9 * p

# 灭绝概率q(p)
def q_of_p(p, sigma=1.0):
    ba = niche_width(p)
    return 2 * (1 - norm.cdf(ba, loc=0, scale=sigma))

def get_q(p, mode='bg'):
    if use_variable_q:
        sigma = 1.0 if mode == 'bg' else 8.0
        return q_of_p(p, sigma)
    else:
        return q_bg if mode == 'bg' else q_mass

# 解析解
def analytical_extinction(num_niches, a_proportions, mode='bg'):
    probs = []
    for p in a_proportions:
        n = int(np.round(num_niches * p))
        q = get_q(p, mode)
        probs.append(q ** n if n > 0 else 0)
    return np.array(probs)

if __name__ == '__main__':
    ana_prob_bg = analytical_extinction(num_niches, a_proportions, mode='bg')
    ana_prob_mass = analytical_extinction(num_niches, a_proportions, mode='mass')

    plt.figure(figsize=(8,5))
    plt.plot(a_proportions*100, ana_prob_bg, '-', color='blue', label='Analytical (Background)')
    plt.plot(a_proportions*100, ana_prob_mass, '-', color='red', label='Analytical (Mass extinction)')

    plt.xlabel('Proportion of Niches Occupied by Category A (%)', fontsize=12)
    plt.ylabel('Probability of Complete Extinction of Category A', fontsize=12)
    plt.title('Analytical Extinction Probability\n(Background vs Mass Extinction)', fontsize=14)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('analytical_extinction_probability.png', dpi=300)
    plt.show()  # 现在会弹窗显示图片

    # 控制台输出典型p值下参数
    print(f"num_niches = {num_niches}, q_bg = {q_bg}, q_mass = {q_mass}, use_variable_q = {use_variable_q}")
    for p in [1.0, 0.5, 0.1]:
        if use_variable_q:
            q_bg_val = get_q(p, 'bg')
            q_mass_val = get_q(p, 'mass')
        else:
            q_bg_val = q_bg
            q_mass_val = q_mass
        print(f"p={p:.2f}, ba={niche_width(p):.3f}, q_bg={q_bg_val:.5f}, q_mass={q_mass_val:.5f}")