import numpy as np
from scipy.stats import norm

values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
counts = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])

data = np.repeat(values, counts)
n = len(data)

# ОММ
mu = np.mean(data)
sigma = np.std(data, ddof=0)

print("mu, sigma =", mu, sigma)


def F(x, mu, sigma):
    return norm.cdf(x, loc=mu, scale=sigma)


def kolmogorov_stat(sample, mu, sigma):
    sample_sorted = np.sort(sample)
    n = len(sample)

    Fn = np.arange(1, n + 1) / n
    F_theor = F(sample_sorted, mu, sigma)

    return np.sqrt(n) * np.max(np.abs(Fn - F_theor))


Delta_hat = kolmogorov_stat(data, mu, sigma)
print("Delta_hat =", Delta_hat)

N = 50000
count = 0

for _ in range(N):
    sample = np.random.normal(mu, sigma, n)

    mu_star = np.mean(sample)
    sigma_star = np.std(sample, ddof=0)

    Delta_star = kolmogorov_stat(sample, mu_star, sigma_star)

    if Delta_star >= Delta_hat:
        count += 1


print(f"l={count}")
p_value = count / N

print("p-value =", p_value)