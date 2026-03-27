import numpy as np

values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
counts = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])

data = np.repeat(values, counts)
n = len(data)

# ОММ
mean = np.mean(data)
var = np.var(data, ddof=0)

theta1 = mean - np.sqrt(3 * var)
theta2 = mean + np.sqrt(3 * var)

print("theta1, theta2 =", theta1, theta2)


def F(x, t1, t2):
    return np.clip((x - t1) / (t2 - t1), 0, 1)


def kolmogorov_stat(sample, t1, t2):
    sample_sorted = np.sort(sample)
    n = len(sample)

    Fn = np.arange(1, n + 1) / n
    F_theor = F(sample_sorted, t1, t2)

    return np.sqrt(n) * np.max(np.abs(Fn - F_theor))


Delta_hat = kolmogorov_stat(data, theta1, theta2)

print("Delta_hat =", Delta_hat)

N = 50000
count = 0

for _ in range(N):
    sample = np.random.uniform(theta1, theta2, n)
    Delta_star = kolmogorov_stat(sample, theta1, theta2)

    if Delta_star >= Delta_hat:
        count += 1
print(f"l={count}")
p_value = count / N

print("p-value =", p_value)