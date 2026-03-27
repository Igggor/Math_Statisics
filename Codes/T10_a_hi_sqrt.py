import numpy as np
from scipy.optimize import minimize


def neg_log_likelihood(params):
    theta1, theta2 = params

    # ограничения (иначе лог не определен)
    if theta1 >= 0.5 or theta2 <= 8.5 or theta2 <= theta1:
        return np.inf

    return -(
            5 * np.log(0.5 - theta1)
            + 7 * np.log(theta2 - 8.5)
            - 100 * np.log(theta2 - theta1)
    )


# начальное приближение
x0 = [0, 9]

tetta = minimize(neg_log_likelihood, x0)
tetta1 = tetta.x[0]
tetta2 = tetta.x[1]
print("theta1 =", tetta.x[0])
print("theta2 =", tetta.x[1])

p_0 = (0.5-tetta1) / (tetta2-tetta1)
p_1 = 1 / (tetta2 - tetta1)
p_9 = (tetta2 - 8.5) / (tetta2 - tetta1)

print(p_0, p_1, p_9)

m = [5, 8, 6, 12, 14, 18, 11, 6, 13, 7]
delta = ((5 - 100*p_0)**2) / (100*p_0) + ((7 - 100*p_9)**2) / (100*p_9)
for i in range(1, 9):
    delta += ((m[i] - 100*p_1)**2) / (100*p_1)
print(delta)