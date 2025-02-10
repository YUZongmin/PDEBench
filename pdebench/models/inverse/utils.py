

from __future__ import annotations

import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import welch

logger = logging.getLogger(__name__)


def plot_ic_solution_mcmc(
    latent,
    x,
    y,
    grid,
    model_inverse,
    model,
    device,
    fname_save="IC_inverse_problem_mcmc.pdf",
):
    """
    Plots the prediction of the initial condition estimated using MCMC from the latent with the model "model"
    y  = model(x)
    y[i] = model(latent[i]), i =0, ...

    June 2022, F.Alesiani
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    u0 = model_inverse.latent2source(latent[0]).to(device)
    pred_u0 = model(u0, grid)
    ax.plot(u0.detach().cpu().flatten(), "r", label="Predicted Initial Condition")
    for _latent in latent:
        u0 = model_inverse.latent2source(_latent).to(device)
        ax.plot(u0.detach().cpu().flatten(), "r", alpha=0.1)
    ax.plot(x.detach().cpu().flatten(), "b--", label="True Initial Condition")
    ax.legend()
    # plt.show()

    ax = axes[1]
    ax.plot(pred_u0.detach().cpu().flatten(), "r", label="Predicted forward value")
    ax.plot(y.detach().cpu().flatten(), "b--", label="True forward value")
    for _latent in latent:
        u0 = model_inverse.latent2source(_latent).to(device)
        pred_u0 = model(u0, grid)
        ax.plot(pred_u0.detach().cpu().flatten(), "r", alpha=0.1)
    ax.legend()
    if fname_save:
        plt.savefig(fname_save, bbox_inches="tight")


def plot_ic_solution_grad(
    model_ic, x, y, grid, model, device, fname_save="IC_inverse_problem_grad.pdf"
):
    """
    Plots the prediction of the initial condition estimated using model_ic with the model "model"
    y  = model(x)
    y' = model(model_ic())

    June 2022, F.Alesiani
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    u0 = model_ic().to(device).unsqueeze(0).unsqueeze(-1)
    pred_u0 = model(u0, grid)
    ax.plot(u0.detach().cpu().flatten(), "r", label="Predicted Initial Condition")
    ax.plot(x.detach().cpu().flatten(), "b--", label="True Initial Condition")
    ax.legend()
    # plt.show()

    ax = axes[1]
    ax.plot(pred_u0.detach().cpu().flatten(), "r", label="Predicted forward value")
    ax.plot(y.detach().cpu().flatten(), "b--", label="True forward value")
    ax.legend()
    if fname_save:
        plt.savefig(fname_save, bbox_inches="tight")


def plot_ic_solution_grad_psd(
    model_ic, x, y, grid, model, device, fname_save="IC_inverse_problem_grad_psd.pdf"
):
    """
    Plots the prediction of the initial condition estimated using model_ic with the model "model"
    y  = model(x)
    y' = model(model_ic())
    It also shows the power density

    June 2022, F.Alesiani
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax = axes[0]
    u0 = model_ic().to(device).unsqueeze(0).unsqueeze(-1)
    pred_u0 = model(u0, grid)
    ax.plot(u0.detach().cpu().flatten(), "r", label="Predicted Initial Condition")
    ax.plot(x.detach().cpu().flatten(), "b--", label="True Initial Condition")
    ax.legend()
    # plt.show()

    ax = axes[1]
    ax.plot(pred_u0.detach().cpu().flatten(), "r", label="Predicted forward value")
    ax.plot(y.detach().cpu().flatten(), "b--", label="True forward value")
    ax.legend()

    _u0 = u0.detach().cpu().flatten()
    _x = x[0].detach().cpu().flatten()

    fz = u0.shape[1]

    fu, puu = welch(_u0, fz)
    fx, pxx = welch(_x, fz)

    ax = axes[2]
    ax.semilogy(fu, puu, "r", label="predicted u0")
    ax.semilogy(fx, pxx, "b--", label="x true")
    ax.set_xlabel("spatial frequency")
    ax.set_ylabel("PSD")
    ax.legend()

    if fname_save:
        plt.savefig(fname_save, bbox_inches="tight")


def get_metric_name(filename, model_name, base_path, inverse_model_type):
    """
    returns the name convention for the result file

    June 2022, F.Alesiani
    """
    return (
        base_path
        + filename[:-5]
        + "_"
        + model_name
        + "_"
        + inverse_model_type
        + ".pickle"
    )


def read_results(
    model_names, inverse_model_type, base_path, filenames, shortfilenames, verbose=False
):
    """
    reads and merges the result files.
    Shortnames are used for the name of the dataset as alternative to the file name.

    June 2022, F.Alesiani
    """
    dfs = []
    for model_name in model_names:
        for filename, shortfilename in zip(filenames, shortfilenames):
            # print(filename)
            inverse_metric_filename = get_metric_name(
                filename, model_name, base_path, inverse_model_type
            )
            if verbose:
                msg = f"reading result file: {inverse_metric_filename}"
                logger.info(msg)

            dframe = pd.read_pickle(inverse_metric_filename)
            dframe["model"] = model_name
            dframe["pde"] = shortfilename
            dfs += [dframe]
    keys = ["pde", "model"]
    dframe = pd.concat(dfs, axis=0)
    return dframe, keys


@hydra.main(config_path="../config", config_name="results")
def process_results(cfg: DictConfig):
    """
    reads and merges the result files and aggregate the results with the selected values. The results are aggregated by datafile.

    June 2022, F.Alesiani
    """
    logger.info(cfg.args)

    df, keys = read_results(
        cfg.args.model_names,
        cfg.args.inverse_model_type,
        cfg.args.base_path,
        cfg.args.filenames,
        cfg.args.shortfilenames,
    )
    df1p3 = df[keys + list(cfg.args.results_values)]
    df2p3 = df1p3.groupby(by=keys).agg([np.mean, np.std]).reset_index()
    msg = "saving results into: {cfg.args.base_path + cfg.args.result_filename}"
    logger.info(msg)
    df2p3.to_csv(cfg.args.base_path + cfg.args.result_filename)


if __name__ == "__main__":
    process_results()
    msg = "Done."
    logger.info(msg)
