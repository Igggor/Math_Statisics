import numpy as np

thetta = 10
n = 100
"""А вот тут, кстати, прикольно. Если увеличивать размер выборки n, то видно, что все методы сходятся к 10"""
xn = thetta + thetta * np.random.random(n)


def current_method():
    x_max = np.max(xn)
    t1 = 1 + 0.025 ** (1 / n)
    t2 = 1 + 0.975 ** (1 / n)
    left = x_max / t2
    right = x_max / t1
    print(f'Точный метод: ({left}, {right}), длина интервала: {right - left}')

def assimp_method():
    alpha_1 = np.mean(xn)
    alpha_2 = np.mean(xn ** 2)
    thetta_1 = 2 / 3 * alpha_1
    t1 = -1.96
    t2 = 1.96
    left = thetta_1 - 4 / 9 * t2 * np.sqrt((alpha_2 - alpha_1 ** 2) / n)
    right = thetta_1 - 4 / 9 * t1 * np.sqrt((alpha_2 - alpha_1 ** 2) / n)
    print(f'Ассимптотический метод (ОММ): ({left}, {right}), длина интервала: {right - left}')


def bootstrap_method():
    alpha_1 = np.mean(xn)
    thetta_1 = 2 / 3 * alpha_1
    samples = np.random.choice(xn, (1000, n), replace=True)
    points = np.sort(np.apply_along_axis(lambda x: 2 / 3 * np.mean(x) - thetta_1, 1, samples))
    k1, k2 = 24, 974
    left = thetta_1 - points[k2]
    right = thetta_1 - points[k1]
    print(f'Непараметрический Bootstrap: ({left}, {right}), длина интервала: {right - left}')



if __name__ == '__main__':
    current_method()
    assimp_method()
    bootstrap_method()


"""
А вот это вот вывод работы кода:


Точный метод: (9.998, 10.181), длина интервала: 0.1831003628626977
Ассимптотический метод (ОММ): (10.015, 10.523), длина интервала: 0.5080577594714377
Непараметрический Bootstrap: (9.884, 10.634), длина интервала: 0.7508200191676107
"""
