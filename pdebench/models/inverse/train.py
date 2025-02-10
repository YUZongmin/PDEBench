

from __future__ import annotations

import logging
from timeit import default_timer

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from pdebench.models.fno.fno import FNO1d, FNO2d, FNO3d
from pdebench.models.fno.utils import FNODatasetSingle
from pdebench.models.inverse.inverse import (
    ElementStandardScaler,
    InitialConditionInterp,
    ProbRasterLatent,
)
from pdebench.models.metrics import inverse_metrics
from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
from pdebench.models.unet.utils import UNetDatasetSingle
from pyro.infer import MCMC, NUTS
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(cfg.args.filename)
    logger.info(cfg.args)

    # we use the test data
    if cfg.args.model_name in ["FNO"]:
        inverse_data = FNODatasetSingle(
            cfg.args.filename,
            saved_folder=cfg.args.base_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            initial_step=cfg.args.initial_step,
            if_test=True,
            num_samples_max=cfg.args.num_samples_max,
        )

        _data, _, _ = next(iter(inverse_data))
        dimensions = len(_data.shape)
        spatial_dim = dimensions - 3

    if cfg.args.model_name in ["UNET", "Unet"]:
        inverse_data = UNetDatasetSingle(
            cfg.args.filename,
            saved_folder=cfg.args.base_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            initial_step=cfg.args.initial_step,
            if_test=True,
            num_samples_max=cfg.args.num_samples_max,
        )

        inverse_loader = torch.utils.data.DataLoader(
            inverse_data, batch_size=1, shuffle=False
        )
        _data, _ = next(iter(inverse_loader))
        dimensions = len(_data.shape)
        spatial_dim = dimensions - 3

    t_train = cfg.args.t_train

    model_name = cfg.args.filename[:-5] + "_" + cfg.args.model_name
    model_path = cfg.args.base_path + model_name + ".pt"

    if cfg.args.model_name in ["FNO"]:
        if dimensions == 4:
            logger.info(cfg.args.num_channels)
            model = FNO1d(
                num_channels=cfg.args.num_channels,
                width=cfg.args.width,
                modes=cfg.args.modes,
                initial_step=cfg.args.initial_step,
            ).to(device)

        if dimensions == 5:
            model = FNO2d(
                num_channels=cfg.args.num_channels,
                width=cfg.args.width,
                modes1=cfg.args.modes,
                modes2=cfg.args.modes,
                initial_step=cfg.args.initial_step,
            ).to(device)

        if dimensions == 6:
            model = FNO3d(
                num_channels=cfg.args.num_channels,
                width=cfg.args.width,
                modes1=cfg.args.modes,
                modes2=cfg.args.modes,
                modes3=cfg.args.modes,
                initial_step=cfg.args.initial_step,
            ).to(device)

    if cfg.args.model_name in ["UNET", "Unet"]:
        if dimensions == 4:
            model = UNet1d(cfg.args.in_channels, cfg.args.out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(cfg.args.in_channels, cfg.args.out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(cfg.args.in_channels, cfg.args.out_channels).to(device)

    model = load_model(model, model_path, device)

    model.eval()
    if cfg.args.inverse_model_type in ["ProbRasterLatent"]:
        assert spatial_dim == 1, "give me time"
        if spatial_dim == 1:
            ns, nx, nt, nc = _data.shape
            model_inverse = ProbRasterLatent(
                model.to(device),
                dims=[nx, 1],
                latent_dims=[1, cfg.args.in_channels_hid, 1],
                prior_scale=0.1,
                obs_scale=0.01,
                prior_std=0.01,
                device=device,
            )

    if cfg.args.inverse_model_type in ["InitialConditionInterp"]:
        loss_fn = nn.MSELoss(reduction="mean")
        input_dims = list(_data.shape[1 : 1 + spatial_dim])
        latent_dims = len(input_dims) * [cfg.args.in_channels_hid]
        if cfg.args.num_channels > 1:
            input_dims = [*input_dims, cfg.args.num_channels]
            latent_dims = [*latent_dims, cfg.args.num_channels]

        model_ic = InitialConditionInterp(input_dims, latent_dims).to(device)
        model.to(device)

    scaler = ElementStandardScaler()
    loss_fn = nn.MSELoss(reduction="mean")

    inverse_u0_l2_full, inverse_y_l2_full = 0, 0
    all_metric = []
    t1 = default_timer()
    for ks, sample in enumerate(inverse_loader):
        if cfg.args.model_name in ["FNO"]:
            (xx, yy, grid) = sample
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)

            def model_(x, grid):
                return model(x, grid)

        if cfg.args.model_name in ["UNET", "Unet"]:
            (xx, yy) = sample
            grid = None
            xx = xx.to(device)
            yy = yy.to(device)

            def model_(x, grid):
                return model(x.permute([0, 2, 1])).permute([0, 2, 1])

        num_samples = ks + 1

        x = xx[..., 0, :]
        y = yy[..., t_train : t_train + 1, :]

        if ks == 0:
            msg = f"{x.shape}, {y.shape}"
            logger.info(msg)

        # scale the input and output
        x = scaler.fit_transform(x)
        y = scaler.transform(y)

        if cfg.args.inverse_model_type in ["ProbRasterLatent"]:
            # Create model
            model_inverse.to(device)
            nuts_kernel = NUTS(
                model_inverse, full_mass=False, max_tree_depth=5, jit_compile=True
            )  # high performacne config

            mcmc = MCMC(
                nuts_kernel,
                num_samples=cfg.args.mcmc_num_samples,
                warmup_steps=cfg.args.mcmc_warmup_steps,
                num_chains=cfg.args.mcmc_num_chains,
                disable_progbar=True,
            )
            mcmc.run(grid, y)
            mc_samples = {
                k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()
            }

            # get the initial solution
            latent = torch.tensor(mc_samples["latent"])
            u0 = model_inverse.latent2source(latent[0]).to(device)
            pred_u0 = model(u0, grid)

        if cfg.args.inverse_model_type in ["InitialConditionInterp"]:
            optimizer = torch.optim.Adam(
                model_ic.parameters(),
                lr=cfg.args.inverse_learning_rate,
                weight_decay=1e-4,
            )
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
            if cfg.args.inverse_verbose_flag:
                _iter = tqdm(range(cfg.args.inverse_epochs))
            else:
                _iter = range(cfg.args.inverse_epochs)
            for _ in _iter:
                if cfg.args.num_channels > 1:
                    u0 = model_ic().unsqueeze(0)
                else:
                    u0 = model_ic().unsqueeze(0).unsqueeze(-1)

                pred_u0 = model_(u0, grid)

                loss_u0 = loss_fn(pred_u0, y)
                optimizer.zero_grad()
                loss_u0.backward()
                optimizer.step()

                t2 = default_timer()
                if cfg.args.inverse_verbose_flag:
                    _iter.set_description(f"loss={loss_u0.item()}, t2-t1= {t2-t1}")

        # compute losses
        loss_u0 = loss_fn(u0.reshape(1, -1), x.reshape(1, -1)).item()
        loss_y = loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
        inverse_u0_l2_full += loss_u0
        inverse_y_l2_full += loss_y

        metric = inverse_metrics(u0, x, pred_u0, y)
        metric["sample"] = ks

        all_metric += [metric]

        t2 = default_timer()
        msg = ", ".join(
            [
                f"samples: {ks + 1}",
                f"loss_u0: {loss_u0:.5f}",
                f"loss_y: {loss_y:.5f}",
                f"t2-t1: {t2 - t1:.5f}",
                f"mse_inverse_u0_L2: {inverse_u0_l2_full / num_samples:.5f}",
                f"mse_inverse_y_L2: {inverse_y_l2_full / num_samples:.5f}",
            ]
        )
        logger.info(msg)

    df_metric = pd.DataFrame(all_metric)
    inverse_metric_filename = (
        cfg.args.base_path
        + cfg.args.filename[:-5]
        + "_"
        + cfg.args.model_name
        + "_"
        + cfg.args.inverse_model_type
        + ".csv"
    )
    msg = f"saving in : {inverse_metric_filename}"
    logger.info(msg)
    df_metric.to_csv(inverse_metric_filename)

    inverse_metric_filename = (
        cfg.args.base_path
        + cfg.args.filename[:-5]
        + "_"
        + cfg.args.model_name
        + "_"
        + cfg.args.inverse_model_type
        + ".pickle"
    )
    msg = f"saving in : {inverse_metric_filename}"
    logger.info(msg)
    df_metric.to_pickle(inverse_metric_filename)

    inverse_metric_filename = (
        cfg.args.base_path
        + cfg.args.filename[:-5]
        + "_"
        + cfg.args.model_name
        + "_"
        + cfg.args.inverse_model_type
        + "_stats.csv"
    )
    msg = f"saving in : {inverse_metric_filename}"
    logger.info(msg)
    df_metric = df_metric.describe()
    df_metric.to_csv(inverse_metric_filename)


if __name__ == "__main__":
    main()
