

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from pdebench.models.fno.train import run_training as run_training_FNO
from pdebench.models.pinn.train import run_training as run_training_PINN
from pdebench.models.unet.train import run_training as run_training_Unet

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info(cfg.args)
    if cfg.args.model_name == "FNO":
        logger.info("FNO")
        run_training_FNO(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            modes=cfg.args.modes,
            width=cfg.args.width,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            num_channels=cfg.args.num_channels,
            batch_size=cfg.args.batch_size,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
            base_path=cfg.args.base_path,
            training_type=cfg.args.training_type,
        )
    elif cfg.args.model_name == "Unet":
        logger.info("Unet")
        run_training_Unet(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            in_channels=cfg.args.in_channels,
            out_channels=cfg.args.out_channels,
            batch_size=cfg.args.batch_size,
            unroll_step=cfg.args.unroll_step,
            ar_mode=cfg.args.ar_mode,
            pushforward=cfg.args.pushforward,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
            base_path=cfg.args.base_path,
            training_type=cfg.args.training_type,
        )
    elif cfg.args.model_name == "PINN":
        logger.info("PINN")
        run_training_PINN(
            scenario=cfg.args.scenario,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            seed=cfg.args.seed,
        )


if __name__ == "__main__":
    main()
