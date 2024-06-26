"""Module for the prediction of cumulative sample size"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class CumSample(nn.Module):
    def __init__(self, alpha_size) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(alpha_size))

    def forward(self, data_dir):
        y = torch_cum_equations(self.alpha, data_dir)

        return y


def torch_cum_equation(
    alpha: torch.tensor, daily_sample: torch.tensor, cum_sample: torch.tensor
) -> torch.tensor:
    t = alpha.shape[0]
    equ = torch.zeros(t)

    b = torch.zeros(t + 1)
    b[0] = daily_sample[0]
    for i in range(1, min(t + 1, daily_sample.shape[0])):
        # b[i] = b[i] + daily_sample[i] - torch.sum(b[:i] * alpha[0:(i)])
        new_value = daily_sample[i] - torch.sum(
            b[:i].clone() * torch.flip(alpha[:i], dims=[0])
        )

        b[i] = new_value.clone()

        equ[i - 1] = cum_sample[i] - b[: (i + 1)].sum()

    return equ


def torch_cum_equations(alpha: torch.tensor, data_dir: str):
    t = alpha.shape[0]
    equ = torch.zeros(t)
    k = torch.zeros(t)
    files = os.listdir(data_dir)

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            # Use numpy to read the CSV file
            df = np.genfromtxt(file_path, delimiter=",", skip_header=1)
            # Convert the numpy array to a PyTorch tensor
            df = torch.from_numpy(df)
            n_values = df[:, 0]
            cum_values = df[:, 1]
            equ_f = torch_cum_equation(alpha, n_values, cum_values)
            equ = equ + equ_f

    # equ = equ / k
    new_equ = equ.clone()
    return torch.norm(new_equ)


def cum_equation(
    alpha: np.ndarray, daily_sample: np.ndarray, cum_sample: np.ndarray
) -> np.ndarray:
    t = alpha.shape[0]
    equ = np.zeros(t)

    b = np.zeros(t + 1)
    b[0] = daily_sample[0]
    for i in range(1, min(b.shape[0], daily_sample.shape[0])):
        b[i] = daily_sample[i] - np.sum(b[:i] * alpha[0:i][::-1])
        equ[i - 1] = cum_sample[i] - b[: (i + 1)].sum()

    return equ


def cum_equations(alpha: np.ndarray, data_dir: str):
    t = alpha.shape[0]
    equ = np.zeros(t)
    k = np.zeros(t)
    files = os.listdir(data_dir)

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            equ_f = cum_equation(alpha, df.n.values, df.cum.values)
            k += 1 - (equ_f == 0)
            equ += equ_f

    equ = equ / k
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
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Training dataset folder"
    )
    parser.add_argument(
        "--max_days", type=int, required=True, help="The maximum days of experiments"
    )

    args = parser.parse_args()

    alpha = optim_cum(args.data_dir, args.max_days)
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

    model = CumSample(args.max_days - 1)
    loss = model(args.data_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    def closure():
        optimizer.zero_grad()
        loss = model(args.data_dir)
        loss.backward()
        # Project gradients (replace epsilon with your desired value)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad = project(param.grad, epsilon=1.0)

        return loss

    num_epochs = 1000
    for epoch in range(num_epochs):
        with torch.autograd.set_detect_anomaly(True):
            #optimizer.zero_grad()
            loss = model(args.data_dir)
            #loss.backward()
            optimizer.step(closure)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad}")
    print(f"pytorch alpha={model.alpha}")
    print(f"numpy alpha = {alpha}")
