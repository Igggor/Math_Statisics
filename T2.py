import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import expon, gaussian_kde


N = 25 # Размер выборки
"""
С этими значениями вообще какая-то бяка. С N=25 расхождения сильные. 


Ставлю 100, 1000 или даже 10_000, и все очень красиво и почти сходится
"""
B = 1_000 # Бутстраповская переменная.
"""
Я заметил, что если брать B чуть больше, результат получается чуть точнее. 


Он тоже завязан на случайных величинах, поэтому даже при B=100_000 местами получается нечто некрасивое
"""







def generate_elem() -> float:
    y = np.random.random()
    x = -np.log(1-y)
    return x

def find_moda(arr: np.array) -> np.float64:
    return np.float64(stats.mode(arr)[0])


def find_median(arr: np.array) -> np.float64:
    median = np.median(arr)
    return np.float64(median)


def find_razmah(arr: np.array) -> np.float64:
    return np.float64(np.max(arr) - np.min(arr))


def calculate_skewness(arr: np.array) -> float:
    mean = np.mean(arr)

    m3 = np.mean((arr - mean) ** 3)
    m2 = np.mean((arr - mean) ** 2)
    return float(m3 / (m2**(3/2)))


def task_b(sample: np.array):
    n = N

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]

    x_sorted = np.sort(sample)
    y_ecdf = np.arange(1, n + 1) / n

    ax.step(x_sorted, y_ecdf, where='post', label='Эксперементальная')
    x_theor = np.linspace(0, max(sample), 100)
    y_theor = expon.cdf(x_theor, scale=1)
    ax.plot(x_theor, y_theor, 'r--', label='Теоретическая функция распределения')
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_title('Эмпирическая функция распределения')
    ax.legend()
    ax.grid(True, alpha=0.3)

    """Гистограмма"""
    ax = axes[1]
    counts, bins, patches = ax.hist(sample, bins=(1 + int(np.ceil(np.log2(n)))), density=True, alpha=0.7, edgecolor='black',
                                    label='Гистограмма')
    x_theor = np.linspace(0, max(sample), 100)
    y_theor_pdf = expon.pdf(x_theor, scale=1)
    ax.plot(x_theor, y_theor_pdf, 'r-', linewidth=2, label='Теоретическая плотность')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_title('Гистограмма и теоретическая плотность')
    ax.legend()
    ax.grid(True, alpha=0.3)

    """Boxplot"""
    ax = axes[2]
    ax.boxplot(sample, vert=False, patch_artist=True)
    ax.set_ylabel('Значения')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_title('Boxplot')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('imgs/task_b_plots.png', dpi=150)
    plt.show()


def task_c(sample: np.array):
    bootstrap_means = []

    for _ in range(B):
        bootstrap_sample = np.random.choice(sample, size=N, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)


    mu_theor = 1
    sigma_theor = np.sqrt(1 / N)


    print(f"Теоретическое среднее Exp(1): {mu_theor}")
    print(f"Теоретическая стандартная ошибка среднего: {sigma_theor:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    """Гистограмма бутстраповских средних и нормальное приближение (ЦПТ)"""
    ax = axes[0]
    ax.hist(bootstrap_means, bins=30, density=True, alpha=0.7, edgecolor='black', label='Бутстрап-средние')

    # Нормальное приближение по ЦПТ с теоретическими параметрами
    x_norm = np.linspace(min(bootstrap_means), max(bootstrap_means), 100)
    y_norm_theor = 1 / (np.sqrt(2 * np.pi) * sigma_theor) * np.exp(-(x_norm - mu_theor) ** 2 / (2 * sigma_theor ** 2))
    ax.plot(x_norm, y_norm_theor, 'r-', linewidth=2, label='ЦПТ')

    ax.set_xlabel('Среднее значение')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_ylabel('Плотность')
    ax.set_title('Распределение выборочного среднего')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    kde_bootstrap = gaussian_kde(bootstrap_means)
    x_kde = np.linspace(min(bootstrap_means), max(bootstrap_means), 200)
    ax.plot(x_kde, kde_bootstrap(x_kde), 'b-', linewidth=2, label='бутстрап')

    """Нормальное приближение по ЦПТ"""
    ax.plot(x_norm, y_norm_theor, 'r-', linewidth=2, label='ЦПТ')

    ax.set_xlabel('Среднее значение')
    ax.set_ylabel('Плотность')
    ax.set_title('Бутстрап распределения среднего vs ЦПТ')
    ax.legend()
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('imgs/task_c_central_limit_theorem.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    data = np.array([generate_elem() for i in range(N)])

    print("="*50)
    print("Задача a)")
    print(("="*50) + "\n")
    print(f"Данные (первые 10 значений): {data[:10]}")
    moda = find_moda(arr=data)
    print(f"Мода: {moda}")
    print(f"Медиана: {find_median(data)}")
    print(f"Размах: {find_razmah(data)}")
    print(f"Коэффициент асимметрии: {calculate_skewness(data)}")

    """=========================================================="""
    print("\n\n" + "="*50)
    print("Задача b)")
    print(("="*50) + "\n")
    task_b(data)

    print("\n\n" + "=" * 50)
    print("ЗАДАНИЕ c: Сравнение распределения среднего (ЦПТ vs Бутстрап)")
    print(("="*50) + "\n")
    task_c(data)
