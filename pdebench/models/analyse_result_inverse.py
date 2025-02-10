

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    filename = "inverse.csv"
    data = pd.read_csv(filename)
    pdes = data["pde"].drop_duplicates()
    num_pdes = len(pdes)
    models = list(data.columns.to_numpy()[-2:])
    num_models = len(models)
    x = np.arange(num_pdes)
    width = 0.5 / (num_models)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_models):
        pos = x - 0.125 + 0.5 / (num_models) * i
        ax.bar(
            pos,
            data[data.iloc[:, 1] == "mean"][models[i]],
            yerr=data[data.iloc[:, 1] == "std"][models[i]],
            width=width,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pdes, rotation=45, fontsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.set_yscale("log")
    ax.set_xlabel("PDEs", fontsize=30)
    ax.set_ylabel("MSE", fontsize=30)
    fig.legend(models, loc=1, ncol=num_models, fontsize=20)
    plt.tight_layout()
    plt.savefig("ResultsInverse.pdf")


if __name__ == "__main__":
    main()
