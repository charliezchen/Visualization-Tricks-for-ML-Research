import numpy as np


def get_scaling_law_data():
    data = []
    cases = [0.25, 0.5, 1., 2., 4.]
    start_stop_map = {0.25: (1, 20), 0.5: (3, 30), 1.: (6, 20), 2.: (10, 50), 4.: (10, 80)}
    for case in cases:
        start, stop = start_stop_map[case]
        plateau = exponential_decay(case, A=3., k=0.15, C=1.0)
        x = np.linspace(start, stop, num=10)
        y = exponential_decay(x, A=5, C=plateau * 0.5, k=0.25)
        data.append({"compute": x, "loss": y, "params": case})
    return data


def exponential_decay(x, A, C, k):
    out = A * np.exp(-k * x) + C
    return out
