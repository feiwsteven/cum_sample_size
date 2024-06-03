"""Module for the prediction of cumulative sample size"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def cum_equation(alpha: np.ndarray, daily_sample: np.ndarray, cum_sample: np.ndarray):
    t = alpha.shape[0] + 1
    equ = np.zeros(t)
    aug_alpha = np.zeros(t)
    aug_alpha[0] = 1
    aug_alpha[1:] = alpha

    b = np.zeros(t)
    b[0] = daily_sample[0]
    for i in range(1, min(t, daily_sample.shape[0])):
        b[i] = daily_sample[i] - np.sum(b[:i] * aug_alpha[1 : (i + 1)][::-1])
        equ[i] = cum_sample[i] - b[: (i + 1)].sum()

    return equ


def cum_equations(alpha: np.ndarray, data_dir: str):
    t = alpha.shape[0] + 1
    equ = np.zeros(t)
    aug_alpha = np.zeros(t)
    aug_alpha[0] = 1
    aug_alpha[1:] = alpha

    k = np.zeros(t)
    files = os.listdir(data_dir)

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            equ_f = cum_equation(alpha, df.n.values, df.cum.values)
            k += 1 - (equ_f == 0)
            equ += equ_f

    equ[1:] = equ[1:] / k[1:]
    print(alpha)
    return np.sum(equ * equ)


def optim_cum(data_dir, t: int):
    bnds = ((0, 1) for _ in range(t - 1))
    out = minimize(
        cum_equations,
        np.linspace(0.1, 1, t - 1),
        args=(data_dir),
        bounds=bnds,
        method="L-BFGS-B",
        options={"maxiter": 10000},
    )

    return np.insert(out.x, 0, 1)


def predict_cum(alpha: np.ndarray, daily_sample: np.ndarray):
    t = daily_sample.shape[0]

    b = np.zeros(t)
    b[0] = daily_sample[0]
    for i in range(1, t):
        b[i] = daily_sample[i] - np.sum(b[:i] * alpha[1 : (i + 1)][::-1])

    return b.cumsum(), b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line arguments for sample size prediction"
    )
    parser.add_argument("--train_data", type=str, help="Training dataset")
    parser.add_argument("--data_dir", type=str, help="Training dataset folder")

    args = parser.parse_args()

    alpha = np.zeros(10)
    equ = cum_equations(alpha, args.data_dir)
    alpha = optim_cum(args.data_dir, 11)
    print(f"alpha={alpha}")

    fig, (ax1, ax2) = plt.subplots(2)

    files = os.listdir(args.data_dir)

    for file in files:
        file_path = os.path.join(args.data_dir, file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            p_cum_sample, _ = predict_cum(alpha, df.n.values)

            ax1.plot(p_cum_sample, linestyle="--", label="predicted cum")
            # ax1.plot(df.n)
            ax1.plot(df.cum, linestyle="-", label="cumulative sample size")
            ax1.legend()

    ax2.plot(alpha)
    ax2.set_title("alpha")

    plt.tight_layout()
    plt.show()
