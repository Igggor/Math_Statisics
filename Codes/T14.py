from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def power(tetha):
    return 1 - norm.cdf(1.645 - tetha / np.sqrt(7/6))


alpha = 0.05
quantile = norm.ppf(1 - alpha)
print(f"Квантиль порядка {1-alpha} для N(0, 1): {quantile:.4f}")

x, y = np.array([-1.11, -6.1, 2.42]), np.array([-2.29, -2.91])
x_mean, y_mean = np.mean(x), np.mean(y)
delta = (x_mean - y_mean) / np.sqrt(7 / 6)
print(x_mean, y_mean, delta)


thetta_vals = np.linspace(0, 10, 500)
p_vals = [power(theta) for theta in thetta_vals]
plt.figure(figsize=(10, 6))
plt.plot(thetta_vals, p_vals, 'b-', linewidth=2)
plt.xlabel('tetta = (a - b)', fontsize=12)
plt.ylabel('Мощность критерия', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='alpha = 0.05')
plt.legend()
plt.tight_layout()
plt.savefig("T14_График.png")
