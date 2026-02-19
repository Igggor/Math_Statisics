import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import expon


N = 25 # Размер выборки

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
