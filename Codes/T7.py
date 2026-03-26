import numpy as np
from math import factorial

indexes = np.arange(0, 6)
m = np.array([109, 65, 22, 3, 1, 0])

def puass(lam, k: int):
    return ((lam**k)/factorial(k)) * (np.exp(-lam))

for i in indexes:
    pu = puass(0.61, i)
    print(f"Вероятность, что произошло {i} смертей за год: {pu}; Произведение n*P_i: {200*pu}")

delta = 0

for i in range(2):
    pu = puass(0.61, i)
    res = (m[i] - 200*pu)**2 / (200*pu)
    delta += res


sum_npi = 0
for i in range(2, 6):
    pu = puass(0.61, i)
    sum_npi += 200*pu

delta += ((26 - sum_npi) ** 2) / sum_npi
print(f"Нормальная дельта: {delta}")
