from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np


def power(tetha):
    return f.cdf(0.77 / tetha, n-1, m-1) + 1 - (f.cdf(1.27 / tetha, n-1, m-1))

n = 139
s1_length, s1_width = 5.722, 4.612

m = 1000
s2_length, s2_width = 6.161, 5.055

x_length, x_width = s1_length ** 2 / s2_length ** 2, s1_width ** 2 / s2_width ** 2

alpha = 0.05

F_left, F_right = f.ppf(alpha / 2, n-1, m-1), f.ppf(1 - alpha / 2, n-1, m-1)

print(x_length, x_width)
print(F_left, F_right)




thetta_values = np.linspace(0.1, 3, 500)
pv = [power(theta) for theta in thetta_values]
plt.figure(figsize=(10, 6))
plt.plot(thetta_values, pv, 'b-', linewidth=2, label=f'F({n - 1}, {m - 1})')
plt.xlabel('Тетта', fontsize=12)
plt.ylabel('Мощность критерия', fontsize=12)
plt.title(f'Функция мощности для F-распределения (n={n}, m={m})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Уровень значимости 0.05')
plt.axvline(x=1.0, color='g', linestyle='--', alpha=0.5, label='Тетта = 1 (нулевая гипотеза)')
plt.legend()
plt.tight_layout()
plt.savefig("T13.png")
