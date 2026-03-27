import numpy as np
from scipy.stats import norm, chi2

values = np.array([0,1,2,3,4,5,6,7,8,9])
counts = np.array([5,8,6,12,14,18,11,6,13,7])
n = np.sum(counts)

data = np.repeat(values, counts)

mu = np.mean(data)
sigma = np.std(data, ddof=0)

print("mu =", mu)
print("sigma^2 =", sigma**2)

bins = np.array([-np.inf, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, np.inf])

probs = []
for i in range(len(bins)-1):
    p = norm.cdf(bins[i+1], mu, sigma) - norm.cdf(bins[i], mu, sigma)
    probs.append(p)

probs = np.array(probs)

expected = n * probs

chi2_stat = np.sum((counts - expected)**2 / expected)

print("chi2 =", chi2_stat)

k = len(counts)
m = 2
df = k - 1 - m

print("df =", df)

alpha = 0.05
chi2_crit = chi2.ppf(1 - alpha, df)

print("chi2_crit =", chi2_crit)

p_value = 1 - chi2.cdf(chi2_stat, df)

print("p-value =", p_value)

if chi2_stat > chi2_crit:
    print("Гипотеза отвергается")
else:
    print("Нет оснований отвергать гипотезу")