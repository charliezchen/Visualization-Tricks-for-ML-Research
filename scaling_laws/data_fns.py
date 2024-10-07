import numpy as np


def get_scaling_law_data():
    data = []
    cases = [0.25, 0.5, 1., 2., 4.]
    start_stop_map = {0.25: (1, 20), 0.5: (3, 30), 1.: (6, 20), 2.: (10, 50), 4.: (10, 80)}
    for case in cases:
        start, stop = start_stop_map[case]
        x = np.linspace(start, stop, num=5)
        y = taylor(x, A=3, C=0, k=0.15, x0=x[2])
        data.append({"compute": x, "loss": y, "params": case})
    return data


def taylor(x, A, C, k, x0):
    f0 = exponential_decay(x0, A, C, k)
    f1 = -k * A * np.exp(-k * x)
    f2 = k ** 2 * A * np.exp(-k * x)
    out = f0 + (x - x0) * f1 + 0.5 * (x - x0) ** 2. * f2
    return out


def exponential_decay(x, A, C, k):
    out = A * np.exp(-k * x) + C
    return out
