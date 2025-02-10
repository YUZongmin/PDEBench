

from __future__ import annotations

import _pickle as cPickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # get results
    files = list(Path().glob("*.pickle"))
    files.sort()

    # metric names
    var_names = [
        "MSE",
        "normalized MSE",
        "Conservation MSE",
        "Maximum Error",
        "MSE at boundary",
        "MSE FT low",
        "MSE FT mid",
        "MSE FT high",
    ]

    # define index
    index1, index2, index3 = [], [], []
    for _j, fl in enumerate(files):
        with Path(fl).open("rb") as f:
            title = fl.split("\\")[-1][:-7].split("_")
            if title[0] == "1D":
                if title[1] == "CFD":
                    index1.append(title[0] + title[1])
                    index2.append(title[3] + title[4] + "_" + title[2] + "_" + title[5])
                    index3.append(title[7])
                else:
                    index1.append(title[1])
                    index2.append(title[3])
                    index3.append(title[4])
            elif title[0] == "2D":
                if title[1] == "CFD":
                    index1.append(title[0] + title[1])
                    index2.append(
                        title[3] + title[3] + title[4] + "_" + title[2] + "_" + title[6]
                    )
                    index3.append(title[9])
                else:
                    index1.append(title[1])
                    index2.append(title[2])
                    index3.append(title[4])
            elif title[0] == "3D":
                index1.append(title[0] + title[1])
                index2.append(
                    title[3] + title[4] + title[5] + "_" + title[2] + "_" + title[6]
                )
                index3.append(title[8])
            else:
                index1.append(title[0])
                index2.append(title[1] + title[2])
                index3.append(title[3])
    indexes = [index1, index2, index3]

    # create dataframe
    data = np.zeros([len(files), 8])
    for j, fl in enumerate(files):
        with Path(fl).open("rb") as f:
            test = cPickle.load(f)
            for i, var in enumerate(test):
                if i == 5:
                    data[j, i:] = var
                else:
                    data[j, i] = var

    index = pd.MultiIndex.from_arrays(indexes, names=("PDE", "param", "model"))
    data = pd.DataFrame(data, columns=var_names, index=index)
    data.to_csv("Results.csv")

    pdes = index.get_level_values(0).drop_duplicates()
    num_pdes = len(pdes)
    models = index.get_level_values(2).drop_duplicates()
    num_models = len(models)
    x = np.arange(num_pdes)

    width = 0.5 if num_models == 1 else 0.5 / (num_models - 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_models):
        pos = x - 0.3 + 0.5 / (num_models - 1) * i
        ax.bar(pos, data[data.index.isin([models[i]], level=2)]["MSE"], width)

    ax.set_xticks(x)
    ax.set_xticklabels(pdes, fontsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.set_yscale("log")
    ax.set_xlabel("PDEs", fontsize=30)
    ax.set_ylabel("MSE", fontsize=30)
    fig.legend(models, loc=8, ncol=num_models, fontsize=20)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("Results.pdf")


if __name__ == "__main__":
    main()
