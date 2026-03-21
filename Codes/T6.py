import numpy as np

theta = 17
n = 100
"""Опять же, увеличивая размер выборки приятно получается, что интервалы сходятся к нужному нам числу"""
xn = np.random.pareto(a=theta-1, size=n) + 1


def med(x: float) -> float:
    return 2 ** (1 / (x - 1))


def sigma(x: float) -> float:
    return np.log(2) * med(x) / (x - 1)


def I(x: float) -> float: 
    return 1 / (x - 1) ** 2


def asimptotic_for_med():
    thetta_wave = 1 + n / np.sum(np.log(xn))
    t1, t2 = -1.96, 1.96
    print(med(thetta_wave))
    left, right = med(thetta_wave) - sigma(thetta_wave) / np.sqrt(n) * t2, med(thetta_wave) - sigma(thetta_wave) / np.sqrt(n) * t1
    print(f'Ассимптотический для медианы: ({(left):.4f}, {(right):.4f})')

def asimptotic_for_thetta():
    thetta_wave = 1 + n / (np.sum(np.log(xn)))
    t1, t2 = -1.96, 1.96
    left, right = thetta_wave - t2 / np.sqrt(n * I(thetta_wave)), thetta_wave - t1 / np.sqrt(n * I(thetta_wave))
    print(
        f'Ассимптотический для тетта: ({left}, {right})', 
        f'Длина интервала: {right - left}'
    )


def unparametric_bootstrap():
    thetta_wave = 1 + n / (np.sum(np.log(xn)))
    boostrap_samples = np.random.choice(xn, (1000, n), replace=True)
    points = np.sort(
        np.apply_along_axis(
            lambda x: 1 + n / np.sum(np.log(x)) - thetta_wave,
            axis=1,
            arr=boostrap_samples
        )
    )
    k1, k2 = 24, 974
    left, right = thetta_wave - points[k2], thetta_wave - points[k1]
    print(
        f'Непараметрический Bootstrap: ({left}, {right})', 
        f'Длина интервала: {right - left}'
    )

def parametric_bootstrap():
    thetta_wave = 1 + n / (np.sum(np.log(xn)))
    boostrap_samples = np.random.pareto(a=thetta_wave-1, size=(50000, n)) + 1
    points = np.sort(
        np.apply_along_axis(
            lambda x: 1 + n / np.sum(np.log(x)) - thetta_wave,
            axis=1,
            arr=boostrap_samples
        )
    )
    k1, k2 = 1250, 48750
    left, right = thetta_wave - points[k2], thetta_wave - points[k1]
    print(f'Параметрический Bootstrap: ({left}, {right})',
          f'Длина интервала: {right - left}')

if __name__ == "__main__":
    asimptotic_for_med()
    asimptotic_for_thetta()
    unparametric_bootstrap()
    parametric_bootstrap()
    
